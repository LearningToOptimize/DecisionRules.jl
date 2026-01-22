#!/usr/bin/env julia
# ============================================================================
# Train a path-following neural network policy for quadruped locomotion
# ============================================================================
# Usage: julia --project train_policy.jl <data_dir> [options]
# Example: julia --project train_policy.jl ./data_large --epochs 100 --batch_size 256
#
# The policy learns to map (current_state, future_path) -> action
# where future_path is a sequence of future (x,y,z) positions to follow.
# ============================================================================

using Flux
using Flux: DataLoader, gpu, cpu
using Flux.Optimisers: ClipGrad, OptimiserChain
using CUDA
using cuDNN
using HDF5
using Statistics
using Random
using LinearAlgebra
using Dates
using JLD2: jldsave, load
using ProgressMeter

# check cuda availability
if CUDA.has_cuda()
    println("CUDA is available. GPU computations enabled.")
else
    println("CUDA is not available. Using CPU.")
end

# ============================
# Configuration
# ============================
@kwdef mutable struct TrainConfig
    # Data
    data_dir::String = "./data"
    train_split::Float64 = 0.9
    
    # Path horizon (how many future timesteps to look ahead)
    path_horizon::Int = 50        # ~50ms at 1kHz
    path_subsample::Int = 10      # subsample path every N steps (reduce input size)
    
    # Network architecture
    state_dim::Int = 36           # from quadruped state
    action_dim::Int = 12          # 12 joint torques
    hidden_dims::Vector{Int} = [256, 256, 128]
    
    # Training
    epochs::Int = 100
    batch_size::Int = 256
    learning_rate::Float64 = 1e-3
    weight_decay::Float64 = 1e-4
    lr_decay_epoch::Int = 50
    lr_decay_factor::Float64 = 0.1
    gradient_clip::Float64 = 1.0
    
    # Device
    use_gpu::Bool = true
    
    # Checkpointing
    checkpoint_dir::String = "./checkpoints"
    checkpoint_every::Int = 10
    
    # Logging
    log_every::Int = 100  # log every N batches
end

# ============================
# Parse command line arguments
# ============================
function parse_args()
    config = TrainConfig()
    
    if length(ARGS) < 1
        println("Usage: julia train_policy.jl <data_dir> [options]")
        println("Options:")
        println("  --epochs N         Number of training epochs (default: 100)")
        println("  --batch_size N     Batch size (default: 256)")
        println("  --lr RATE          Learning rate (default: 1e-3)")
        println("  --hidden DIMS      Hidden layer dimensions (default: 256,256,128)")
        println("  --path_horizon N   Future path horizon in timesteps (default: 50)")
        println("  --checkpoint_dir D Directory for checkpoints (default: ./checkpoints)")
        println("  --no_gpu           Disable GPU training")
        exit(1)
    end
    
    config.data_dir = ARGS[1]
    
    i = 2
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--epochs"
            config.epochs = parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--batch_size"
            config.batch_size = parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--lr"
            config.learning_rate = parse(Float64, ARGS[i+1])
            i += 2
        elseif arg == "--hidden"
            config.hidden_dims = parse.(Int, split(ARGS[i+1], ","))
            i += 2
        elseif arg == "--path_horizon"
            config.path_horizon = parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--checkpoint_dir"
            config.checkpoint_dir = ARGS[i+1]
            i += 2
        elseif arg == "--no_gpu"
            config.use_gpu = false
            i += 1
        else
            println("Unknown argument: $arg")
            i += 1
        end
    end
    
    return config
end

# ============================
# Data loading utilities
# ============================
"""
Load all trajectories from the data directory.
Returns (all_states, all_actions, all_paths) where paths are the (x,y,z) positions.
"""
function load_dataset(data_dir::String; verbose::Bool=true)
    traj_files = filter(f -> startswith(f, "trajectory_") && endswith(f, ".h5"), readdir(data_dir))
    
    if isempty(traj_files)
        error("No trajectory files found in $data_dir")
    end
    
    verbose && println("Found $(length(traj_files)) trajectory files")
    
    all_states = Vector{Matrix{Float32}}()
    all_actions = Vector{Matrix{Float32}}()
    all_paths = Vector{Matrix{Float32}}()  # x,y,z positions
    
    for (i, fname) in enumerate(traj_files)
        fpath = joinpath(data_dir, fname)
        
        states = Float32.(h5read(fpath, "states"))
        actions = Float32.(h5read(fpath, "actions"))
        
        # Extract x,y,z path from states (first 3 components)
        paths = states[1:3, :]
        
        push!(all_states, states)
        push!(all_actions, actions)
        push!(all_paths, paths)
        
        if verbose && i % 100 == 0
            println("  Loaded $i / $(length(traj_files)) trajectories")
        end
    end
    
    verbose && println("Dataset loaded: $(length(all_states)) trajectories")
    
    return all_states, all_actions, all_paths
end

"""
Create training samples from trajectories.
Each sample: (state, future_path) -> action

The future_path is the next `path_horizon` positions, subsampled every `subsample` steps.
"""
function create_samples(
    all_states::Vector{Matrix{Float32}},
    all_actions::Vector{Matrix{Float32}},
    all_paths::Vector{Matrix{Float32}};
    path_horizon::Int=50,
    subsample::Int=10
)
    # Calculate path feature dimension
    n_path_points = div(path_horizon, subsample)
    path_dim = 3 * n_path_points  # x,y,z for each point
    
    # Count total samples
    total_samples = sum(size(s, 2) - path_horizon for s in all_states)
    
    state_dim = size(all_states[1], 1)
    action_dim = size(all_actions[1], 1)
    
    println("Creating $(total_samples) samples...")
    println("  State dim: $state_dim")
    println("  Path dim: $path_dim ($(n_path_points) points × 3)")
    println("  Action dim: $action_dim")
    
    # Pre-allocate
    X_state = zeros(Float32, state_dim, total_samples)
    X_path = zeros(Float32, path_dim, total_samples)
    Y_action = zeros(Float32, action_dim, total_samples)
    
    idx = 1
    for (states, actions, paths) in zip(all_states, all_actions, all_paths)
        T = size(states, 2)
        
        for t in 1:(T - path_horizon)
            # Current state
            X_state[:, idx] .= states[:, t]
            
            # Future path (relative to current position)
            current_pos = paths[:, t]
            path_indices = t .+ (subsample:subsample:path_horizon)
            
            for (j, pidx) in enumerate(path_indices)
                # Store relative position (delta from current)
                delta = paths[:, pidx] .- current_pos
                X_path[(j-1)*3+1:j*3, idx] .= delta
            end
            
            # Action at current timestep
            Y_action[:, idx] .= actions[:, t]
            
            idx += 1
        end
    end
    
    @assert idx - 1 == total_samples
    
    return X_state, X_path, Y_action
end

# ============================
# Neural Network Model
# ============================
"""
Path-following policy network.
Takes concatenated (state, relative_future_path) and outputs action.
"""
function create_policy_network(state_dim::Int, path_dim::Int, action_dim::Int, hidden_dims::Vector{Int})
    input_dim = state_dim + path_dim
    
    layers = []
    prev_dim = input_dim
    
    for hdim in hidden_dims
        push!(layers, Dense(prev_dim, hdim, relu))
        push!(layers, LayerNorm(hdim))  # Better for RL-style tasks
        prev_dim = hdim
    end
    
    # Output layer (no activation for torque values)
    push!(layers, Dense(prev_dim, action_dim))
    
    return Chain(layers...)
end

# ============================
# Training Loop
# ============================
function train!(
    model,
    train_loader::DataLoader,
    val_loader::DataLoader,
    config::TrainConfig;
    device=cpu
)
    # Move model to device
    model = model |> device
    
    # Optimizer with weight decay and gradient clipping
    opt = OptimiserChain(ClipGrad(config.gradient_clip), AdamW(config.learning_rate, (0.9, 0.999), config.weight_decay))
    opt_state = Flux.setup(opt, model)
    
    # Loss function
    loss_fn(x, y) = Flux.mse(model(x), y)
    
    # Create checkpoint directory
    mkpath(config.checkpoint_dir)
    
    # Training history
    history = Dict(
        "train_loss" => Float64[],
        "val_loss" => Float64[],
        "epoch_time" => Float64[]
    )
    
    best_val_loss = Inf
    
    println("\n" * "="^60)
    println("Starting training")
    println("="^60)
    println("Device: $(device == gpu ? "GPU" : "CPU")")
    println("Train batches: $(length(train_loader))")
    println("Val batches: $(length(val_loader))")
    println("="^60 * "\n")
    
    for epoch in 1:config.epochs
        epoch_start = time()
        
        # Learning rate decay
        if epoch == config.lr_decay_epoch
            new_lr = config.learning_rate * config.lr_decay_factor
            Flux.adjust!(opt_state, config.learning_rate * config.lr_decay_factor)
            println("  Learning rate decayed to $new_lr")
        end
        
        # Training
        model_training = Flux.trainmode!(model)
        train_losses = Float64[]
        
        for (batch_idx, (x, y)) in enumerate(train_loader)
            x, y = x |> device, y |> device
            
            loss, grads = Flux.withgradient(model_training) do m
                Flux.mse(m(x), y)
            end
            
            Flux.update!(opt_state, model_training, grads[1])
            
            push!(train_losses, loss)
            
            if batch_idx % config.log_every == 0
                println("  Epoch $epoch, Batch $batch_idx/$(length(train_loader)), Loss: $(round(loss, digits=6))")
            end
        end
        
        # Validation
        model_eval = Flux.testmode!(model)
        val_losses = Float64[]
        
        for (x, y) in val_loader
            x, y = x |> device, y |> device
            loss = Flux.mse(model_eval(x), y)
            push!(val_losses, loss)
        end
        
        train_loss = mean(train_losses)
        val_loss = mean(val_losses)
        epoch_time = time() - epoch_start
        
        push!(history["train_loss"], train_loss)
        push!(history["val_loss"], val_loss)
        push!(history["epoch_time"], epoch_time)
        
        println("Epoch $epoch/$( config.epochs): train_loss=$(round(train_loss, digits=6)), val_loss=$(round(val_loss, digits=6)), time=$(round(epoch_time, digits=1))s")
        
        # Save best model
        if val_loss < best_val_loss
            best_val_loss = val_loss
            model_cpu = model |> cpu
            model_state = Flux.state(model_cpu)
            jldsave(joinpath(config.checkpoint_dir, "best_model.jld2"); model_state=model_state)
            println("  ✓ New best model saved (val_loss=$( round(val_loss, digits=6)))")
        end
        
        # Periodic checkpointing
        if epoch % config.checkpoint_every == 0
            model_cpu = model |> cpu
            model_state = Flux.state(model_cpu)
            jldsave(joinpath(config.checkpoint_dir, "model_epoch_$(epoch).jld2"); model_state=model_state)
            jldsave(joinpath(config.checkpoint_dir, "history.jld2"); history=history)
        end
    end
    
    # Final save
    model_cpu = model |> cpu
    model_state = Flux.state(model_cpu)
    jldsave(joinpath(config.checkpoint_dir, "final_model.jld2"); model_state=model_state)
    jldsave(joinpath(config.checkpoint_dir, "history.jld2"); history=history)
    
    # Save config and architecture info (needed to reconstruct model)
    arch_info = Dict(
        "state_dim" => config.state_dim,
        "path_dim" => 3 * div(config.path_horizon, config.path_subsample),
        "action_dim" => config.action_dim,
        "hidden_dims" => config.hidden_dims,
        "path_horizon" => config.path_horizon,
        "path_subsample" => config.path_subsample
    )
    jldsave(joinpath(config.checkpoint_dir, "config.jld2"); config=config, arch_info=arch_info)
    
    println("\n" * "="^60)
    println("Training complete!")
    println("Best validation loss: $(round(best_val_loss, digits=6))")
    println("Models saved to: $(config.checkpoint_dir)")
    println("="^60 * "\n")
    
    return model |> cpu, history
end

# ============================
# Main
# ============================
function main()
    config = parse_args()
    
    println("\n" * "="^60)
    println("Path-Following Policy Training")
    println("="^60)
    println("Data directory: $(config.data_dir)")
    println("Path horizon: $(config.path_horizon) steps")
    println("Path subsample: $(config.path_subsample)")
    println("Hidden dims: $(config.hidden_dims)")
    println("Batch size: $(config.batch_size)")
    println("Learning rate: $(config.learning_rate)")
    println("Epochs: $(config.epochs)")
    println("GPU enabled: $(config.use_gpu)")
    println("="^60 * "\n")
    
    # Check GPU
    device = cpu
    if config.use_gpu
        if CUDA.functional()
            device = gpu
            println("Using GPU: $(CUDA.name(CUDA.device()))")
            println("GPU memory: $(round(CUDA.total_memory() / 1e9, digits=2)) GB")
        else
            println("WARNING: GPU requested but CUDA not functional, using CPU")
            config.use_gpu = false
        end
    end
    
    # Load data
    println("\nLoading dataset...")
    all_states, all_actions, all_paths = load_dataset(config.data_dir)
    
    # Create samples
    println("\nCreating training samples...")
    X_state, X_path, Y_action = create_samples(
        all_states, all_actions, all_paths;
        path_horizon=config.path_horizon,
        subsample=config.path_subsample
    )
    
    # Combine inputs
    X = vcat(X_state, X_path)
    Y = Y_action
    
    println("Input shape: $(size(X))")
    println("Output shape: $(size(Y))")
    
    # Train/val split
    n_samples = size(X, 2)
    n_train = Int(floor(n_samples * config.train_split))
    
    # Shuffle
    Random.seed!(42)
    perm = randperm(n_samples)
    train_idx = perm[1:n_train]
    val_idx = perm[n_train+1:end]
    
    X_train, Y_train = X[:, train_idx], Y[:, train_idx]
    X_val, Y_val = X[:, val_idx], Y[:, val_idx]
    
    println("Train samples: $(size(X_train, 2))")
    println("Val samples: $(size(X_val, 2))")
    
    # Normalize inputs (important for NN training)
    X_mean = Float32.(mean(X_train, dims=2))
    X_std = Float32.(std(X_train, dims=2) .+ 1e-8)
    
    X_train = Float32.((X_train .- X_mean) ./ X_std)
    X_val = Float32.((X_val .- X_mean) ./ X_std)
    
    # Normalize outputs
    Y_mean = Float32.(mean(Y_train, dims=2))
    Y_std = Float32.(std(Y_train, dims=2) .+ 1e-8)
    
    Y_train = Float32.((Y_train .- Y_mean) ./ Y_std)
    Y_val = Float32.((Y_val .- Y_mean) ./ Y_std)
    
    # Save normalization stats (needed for inference)
    mkpath(config.checkpoint_dir)
    jldsave(joinpath(config.checkpoint_dir, "normalization.jld2"); X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std)
    
    # Create data loaders
    train_loader = DataLoader((X_train, Y_train); batchsize=config.batch_size, shuffle=true)
    val_loader = DataLoader((X_val, Y_val); batchsize=config.batch_size, shuffle=false)
    
    # Create model
    state_dim = config.state_dim
    path_dim = 3 * div(config.path_horizon, config.path_subsample)
    action_dim = config.action_dim
    
    println("\nCreating model...")
    println("  Input dim: $(state_dim + path_dim) (state: $state_dim, path: $path_dim)")
    println("  Output dim: $action_dim")
    
    model = create_policy_network(state_dim, path_dim, action_dim, config.hidden_dims)
    
    n_params = sum(length(p) for p in Flux.trainables(model))
    println("  Total parameters: $n_params")
    
    # Train
    trained_model, history = train!(model, train_loader, val_loader, config; device=device)
    
    println("\nDone!")
end

# Run
main()
