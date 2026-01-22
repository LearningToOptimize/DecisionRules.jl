#!/usr/bin/env julia
# ============================================================================
# Evaluate trained path-following policy in simulation
# ============================================================================
# Usage: julia --project -i eval_policy.jl <model_path> [options]
# Example: julia --project -i eval_policy.jl ./checkpoints/best_model.jld2
#
# This script:
# 1. Loads the trained policy
# 2. Generates a target path (or loads one)
# 3. Runs closed-loop simulation with the policy
# 4. Visualizes the result and compares to target path
# ============================================================================

using Dojo
using DojoEnvironments
using LinearAlgebra
using Flux
using JLD2: load, jldsave
using Statistics

# Include shared controller (for environment setup)
include("quadruped_controller.jl")

# ============================
# Configuration
# ============================
@kwdef struct EvalConfig
    model_path::String = "./checkpoints/best_model.jld2"
    norm_path::String = "./checkpoints/normalization.jld2"
    
    # Simulation
    dt::Float64 = 0.001
    duration::Float64 = 10.0  # seconds
    
    # Path parameters
    path_horizon::Int = 50
    path_subsample::Int = 10
    
    # Target path type: :circle, :figure8, :straight, :custom
    path_type::Symbol = :circle
    path_radius::Float64 = 1.0  # for circle/figure8
    path_speed::Float64 = 0.3   # m/s
end

# ============================
# Parse arguments
# ============================
function parse_eval_args()
    config = EvalConfig()
    
    if length(ARGS) >= 1
        config = EvalConfig(model_path=ARGS[1])
    end
    
    # Check for normalization file
    model_dir = dirname(config.model_path)
    norm_path = joinpath(model_dir, "normalization.jld2")
    if isfile(norm_path)
        config = EvalConfig(
            model_path=config.model_path,
            norm_path=norm_path,
            path_type=config.path_type
        )
    end
    
    return config
end

# ============================
# Path generation
# ============================
"""
Generate a target path for the robot to follow.
Returns a function path(t) -> (x, y, z)
"""
function generate_target_path(path_type::Symbol; radius::Float64=1.0, speed::Float64=0.3)
    if path_type == :circle
        # Circular path
        omega = speed / radius
        return t -> (radius * sin(omega * t), radius * (1 - cos(omega * t)), -0.5)
        
    elseif path_type == :figure8
        # Figure-8 path
        omega = speed / radius
        return t -> (radius * sin(omega * t), radius * sin(2 * omega * t) / 2, -0.5)
        
    elseif path_type == :straight
        # Straight line forward
        return t -> (speed * t, 0.0, -0.5)
        
    elseif path_type == :zigzag
        # Zigzag path
        period = 4.0  # seconds per zigzag
        amplitude = 0.5
        return t -> begin
            x = speed * t
            phase = mod(t, period) / period
            y = phase < 0.5 ? amplitude * (2 * phase) : amplitude * (2 - 2 * phase)
            (x, y - amplitude/2, -0.5)
        end
    else
        error("Unknown path type: $path_type")
    end
end

"""
Sample future path points for policy input.
"""
function sample_future_path(
    path_fn,
    current_time::Float64,
    current_pos::Vector{Float64};
    horizon::Int=50,
    subsample::Int=10,
    dt::Float64=0.001
)
    n_points = div(horizon, subsample)
    path_features = zeros(Float32, 3 * n_points)
    
    for (i, k) in enumerate(subsample:subsample:horizon)
        future_t = current_time + k * dt
        future_pos = collect(path_fn(future_t))
        
        # Relative to current position
        delta = future_pos .- current_pos
        path_features[(i-1)*3+1:i*3] .= Float32.(delta)
    end
    
    return path_features
end

# ============================
# Policy wrapper
# ============================
struct PolicyController
    model::Chain
    X_mean::Matrix{Float32}
    X_std::Matrix{Float32}
    Y_mean::Matrix{Float32}
    Y_std::Matrix{Float32}
    path_fn::Function
    path_horizon::Int
    path_subsample::Int
    dt::Float64
end

function (pc::PolicyController)(state::Vector{Float64}, t::Float64)
    # Get current position
    current_pos = state[1:3]
    
    # Sample future path
    path_features = sample_future_path(
        pc.path_fn, t, current_pos;
        horizon=pc.path_horizon,
        subsample=pc.path_subsample,
        dt=pc.dt
    )
    
    # Combine state and path
    x = vcat(Float32.(state), path_features)
    
    # Normalize
    x = (x .- pc.X_mean[:]) ./ pc.X_std[:]
    
    # Forward pass
    y = pc.model(x)
    
    # Denormalize
    action = y .* pc.Y_std[:] .+ pc.Y_mean[:]
    
    return Float64.(action)
end

# ============================
# Simulation
# ============================
function run_policy_simulation(
    policy::PolicyController,
    config::EvalConfig
)
    N_total = Int(round(config.duration / config.dt))
    
    println("Setting up simulation...")
    println("  Duration: $(config.duration)s")
    println("  Timesteps: $N_total")
    
    # Create environment
    env = qc_create_environment(N_total; dt=config.dt)
    
    # Initialize
    qc_reset_state!(env)
    
    # Storage
    state_dim = length(get_state(env))
    states = zeros(state_dim, N_total)
    actions = zeros(12, N_total)
    target_positions = zeros(3, N_total)
    
    # Simulate
    println("Running closed-loop simulation with learned policy...")
    
    function policy_controller!(environment, k)
        t = (k - 1) * config.dt
        x = get_state(environment)
        states[:, k] .= x
        
        # Get target position for logging
        target_positions[:, k] .= collect(policy.path_fn(t))
        
        # Get action from policy
        u12 = try
            policy(x, t)
        catch e
            println("Policy error at t=$t: $e")
            zeros(12)
        end
        
        # Clamp actions
        u12 = clamp.(u12, -QC_U_MAX, QC_U_MAX)
        
        actions[:, k] .= u12
        set_input!(environment, u12)
        return nothing
    end
    
    try
        simulate!(env, policy_controller!; record=true)
    catch e
        println("Simulation error: $e")
    end
    
    return env, states, actions, target_positions
end

# ============================
# Evaluation metrics
# ============================
function compute_metrics(states::Matrix{Float64}, target_positions::Matrix{Float64})
    actual_pos = states[1:3, :]
    
    # Position tracking error
    pos_errors = actual_pos .- target_positions
    tracking_error = sqrt.(sum(pos_errors.^2, dims=1))[:]
    
    mean_error = mean(tracking_error)
    max_error = maximum(tracking_error)
    final_error = tracking_error[end]
    
    # Path length traveled
    path_length = sum(sqrt.(sum(diff(actual_pos, dims=2).^2, dims=1)))
    
    # Stability metrics
    z_positions = states[3, :]
    z_mean = mean(z_positions)
    z_std = std(z_positions)
    
    return Dict(
        "mean_tracking_error" => mean_error,
        "max_tracking_error" => max_error,
        "final_tracking_error" => final_error,
        "path_length" => path_length,
        "z_mean" => z_mean,
        "z_std" => z_std
    )
end

# ============================
# Model loading utility
# ============================
"""
Load a trained policy model from checkpoint.
Reconstructs the architecture and loads saved weights.
"""
function load_policy_model(model_path::String, config_path::String)
    # Load architecture info
    config_data = load(config_path)
    arch_info = config_data["arch_info"]
    
    state_dim = arch_info["state_dim"]
    path_dim = arch_info["path_dim"]
    action_dim = arch_info["action_dim"]
    hidden_dims = arch_info["hidden_dims"]
    
    # Reconstruct model architecture
    model = create_policy_network(state_dim, path_dim, action_dim, hidden_dims)
    
    # Load saved weights
    model_data = load(model_path)
    model_state = model_data["model_state"]
    Flux.loadmodel!(model, model_state)
    
    return model, arch_info
end

"""
Create policy network (same as in train_policy.jl).
"""
function create_policy_network(state_dim::Int, path_dim::Int, action_dim::Int, hidden_dims::Vector{Int})
    input_dim = state_dim + path_dim
    
    layers = []
    prev_dim = input_dim
    
    for hdim in hidden_dims
        push!(layers, Dense(prev_dim, hdim, relu))
        push!(layers, LayerNorm(hdim))
        prev_dim = hdim
    end
    
    push!(layers, Dense(prev_dim, action_dim))
    
    return Chain(layers...)
end

# ============================
# Main
# ============================
function main()
    config = parse_eval_args()
    
    println("\n" * "="^60)
    println("Policy Evaluation")
    println("="^60)
    println("Model: $(config.model_path)")
    println("Path type: $(config.path_type)")
    println("Duration: $(config.duration)s")
    println("="^60 * "\n")
    
    # Load model
    println("Loading model...")
    if !isfile(config.model_path)
        error("Model file not found: $(config.model_path)")
    end
    
    model_dir = dirname(config.model_path)
    config_path = joinpath(model_dir, "config.jld2")
    if !isfile(config_path)
        error("Config file not found: $config_path (needed for model architecture)")
    end
    
    model, arch_info = load_policy_model(config.model_path, config_path)
    println("  Architecture: state=$(arch_info["state_dim"]), path=$(arch_info["path_dim"]), action=$(arch_info["action_dim"])")
    println("  Hidden dims: $(arch_info["hidden_dims"])")
    
    # Update config with loaded architecture info
    config = EvalConfig(
        model_path=config.model_path,
        norm_path=config.norm_path,
        path_horizon=arch_info["path_horizon"],
        path_subsample=arch_info["path_subsample"],
        path_type=config.path_type,
        path_radius=config.path_radius,
        path_speed=config.path_speed,
        dt=config.dt,
        duration=config.duration
    )
    
    # Load normalization
    println("Loading normalization stats...")
    if !isfile(config.norm_path)
        error("Normalization file not found: $(config.norm_path)")
    end
    
    norm_data = load(config.norm_path)
    X_mean = norm_data["X_mean"]
    X_std = norm_data["X_std"]
    Y_mean = norm_data["Y_mean"]
    Y_std = norm_data["Y_std"]
    
    # Generate target path
    println("Generating target path: $(config.path_type)")
    path_fn = generate_target_path(
        config.path_type;
        radius=config.path_radius,
        speed=config.path_speed
    )
    
    # Create policy controller
    policy = PolicyController(
        model,
        X_mean, X_std, Y_mean, Y_std,
        path_fn,
        config.path_horizon,
        config.path_subsample,
        config.dt
    )
    
    # Run simulation
    env, states, actions, target_positions = run_policy_simulation(policy, config)
    
    # Compute metrics
    println("\n" * "="^60)
    println("Evaluation Metrics")
    println("="^60)
    
    metrics = compute_metrics(states, target_positions)
    
    println("Tracking Error:")
    println("  Mean: $(round(metrics["mean_tracking_error"], digits=4)) m")
    println("  Max:  $(round(metrics["max_tracking_error"], digits=4)) m")
    println("  Final: $(round(metrics["final_tracking_error"], digits=4)) m")
    println("\nPath:")
    println("  Length traveled: $(round(metrics["path_length"], digits=4)) m")
    println("\nStability:")
    println("  Z mean: $(round(metrics["z_mean"], digits=4)) m")
    println("  Z std:  $(round(metrics["z_std"], digits=4)) m")
    println("="^60 * "\n")
    
    # Visualize
    println("Rendering visualization...")
    vis = visualize(env)
    render(vis)
    
    println("\n" * "="^60)
    println("Visualization ready!")
    println("="^60 * "\n")
    
    # Return for interactive use
    return env, states, actions, target_positions, metrics
end

# Run if not interactive
if !isinteractive() || length(ARGS) > 0
    env, states, actions, target_positions, metrics = main()
end
