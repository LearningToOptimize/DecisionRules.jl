#!/usr/bin/env julia
# ==========================================================================================
# train_policy_dojo.jl
#
# Train a neural network policy using Flux.jl that maps:
#   - Input:  [state (minimal coordinates); next_waypoint_x; next_waypoint_y; target_z]
#   - Output: action (control input)
#
# The robot follows a curved path specified as a vector of (x, y) waypoints.
#
# Loss function mirrors the trajectory optimization:
#   - Tracking cost for reaching next waypoint on path
#   - Action regularization (L2)
#   - Dynamics constraints via rollout
#
# Dependencies:
#   ] add Flux Zygote Statistics Random
#   ] add Dojo DojoEnvironments
# ==========================================================================================

using Flux
using Zygote
using ChainRulesCore
using Statistics
using Random

using Dojo
using DojoEnvironments

# ------------------------------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------------------------------
const N_TRAJ = 21               # trajectory length (must match optimization)
const dt = 0.02              # time step (must match optimization)
const N_epochs = 10           # number of training epochs; adjust as needed for convergence
const batch_size = 32
const learning_rate = 1e-3

# Loss weights (from optimization)
const w_track_xy = 50.0
const w_track_z  = 100.0
const w_u        = 1e-2
const w_du       = 1e-2

# Indices
const BASE_X_IDX = 1
const BASE_Y_IDX = 2
const BASE_Z_IDX = 3

const u_max_abs = 40.0
const z_nom = 0.8  # nominal height

# ------------------------------------------------------------------------------------------
# Dojo helpers
# ------------------------------------------------------------------------------------------
const DOJO_OPTS = Dojo.SolverOptions{Float64}(;
    rtol = 1e-4,
    btol = 1e-3,
    max_iter = 20,
    max_ls = 8,
    verbose = false
)

function dojo_step(mech, x::AbstractVector, u::AbstractVector; opts=DOJO_OPTS)
    # Non-mutating version for Zygote compatibility
    Dojo.set_minimal_state!(mech, x)
    Dojo.set_input!(mech, u)
    Dojo.step_minimal_coordinates!(mech, x, u; opts=opts)
    return Dojo.get_minimal_state(mech)
end

# Custom rrule for Zygote to differentiate through dojo_step
# This computes linear dynamics: x_{k+1} ≈ x_k + A(x_k, u_k) * Δx + B(x_k, u_k) * Δu
function Zygote.rrule(::typeof(dojo_step), mech::Dojo.Mechanism, x::AbstractVector, u::AbstractVector; opts=DOJO_OPTS)
    # Forward pass: compute next state
    x_next = dojo_step(mech, x, u; opts=opts)
    
    # Compute linearization (A, B matrices) at (x, u)
    mech_lin = deepcopy(mech)
    Dojo.set_minimal_state!(mech_lin, x)
    Dojo.set_input!(mech_lin, u)
    
    # Get Jacobians: x_next ≈ f(x, u) ≈ x + A*Δx + B*Δu
    # where A = I + df/dx, B = df/du
    try
        A, B = Dojo.get_minimal_gradients!(mech_lin, x, u; opts=opts)
        
        # Pullback function: compute gradients w.r.t. u only
        # (state x is not trainable, only action u is)
        function dojo_step_pullback(Δx_next)
            # Gradient w.r.t. u: Δu = B^T * Δx_next
            ∇u = B' * Δx_next
            
            # Return gradients in the same order as inputs
            # NoTangent for mech and x (not trainable), only ∇u for action
            return (NoTangent(), NoTangent(), NoTangent(), ∇u, NoTangent())
        end
        
        return x_next, dojo_step_pullback
    catch
        # Fallback if linearization fails: no gradients
        function dojo_step_pullback_fallback(Δx_next)
            return (NoTangent(), NoTangent(), NoTangent(), zero(u), NoTangent())
        end
        return x_next, dojo_step_pullback_fallback
    end
end

function build_quadruped()
    mech = DojoEnvironments.get_mechanism(:quadruped)
    x0 = Vector(Dojo.get_minimal_state(mech))
    m  = Dojo.input_dimension(mech)
    u0 = zeros(m)
    return mech, x0, u0
end

# ------------------------------------------------------------------------------------------
# Generate training data: trajectories following curved paths
# ------------------------------------------------------------------------------------------
function generate_waypoint_dataset(mech_ref::Dojo.Mechanism, x0::Vector, path_xy::Matrix, n_trajectories::Int)
    """
    Generate training data for following a path defined by waypoints.
    
    Args:
        mech_ref: Dojo mechanism
        x0: initial state
        path_xy: (n_waypoints, 2) matrix where each row is [x, y] coordinate
        n_trajectories: number of training trajectories to generate
    """
    n = length(x0)
    m = Dojo.input_dimension(mech_ref)
    n_waypoints = size(path_xy, 1)
    
    dataset = []
    
    for traj_idx in 1:n_trajectories
        # Random starting waypoint index (not the last one)
        start_wp = rand(1:(n_waypoints - 1))
        
        # Initialize state with small perturbation
        x_current = copy(x0) .+ randn(n) * 0.02
        
        mech = deepcopy(mech_ref)
        
        # Rollout trajectory following the path
        wp_idx = start_wp
        steps_on_path = 0
        max_steps_per_segment = 10
        
        for step in 1:N_TRAJ
            # Get current and next waypoint
            current_wp = path_xy[wp_idx, :]
            next_wp = path_xy[min(wp_idx + 1, n_waypoints), :]
            
            # Target is the next waypoint
            target_x = next_wp[1]
            target_y = next_wp[2]
            
            # Create training example: (state, target) -> action
            input = vcat(x_current, target_x, target_y, z_nom)
            
            # Random action for data collection (exploration)
            u_current = (randn(m) .- 0.5) * u_max_abs * 0.5
            push!(dataset, (input, u_current))
            
            # Step dynamics
            if step < N_TRAJ
                x_current = dojo_step(mech, x_current, u_current; opts=DOJO_OPTS)
                
                # Check if we reached the waypoint (with tolerance)
                dist_to_wp = sqrt((x_current[BASE_X_IDX] - next_wp[1])^2 + (x_current[BASE_Y_IDX] - next_wp[2])^2)
                steps_on_path += 1
                
                if dist_to_wp < 0.1 && steps_on_path > 3 && wp_idx < n_waypoints
                    wp_idx += 1
                    steps_on_path = 0
                end
            end
        end
    end
    
    return dataset
end

# ------------------------------------------------------------------------------------------
# Build Flux policy network
# ------------------------------------------------------------------------------------------
function create_policy_network(n::Int, m::Int; hidden_dims=[128, 128])
    # Input: [state (n); target_x, target_y, target_z]  = (n+3)
    # Output: action (m)
    input_dim = n + 3
    
    return Chain(
        Dense(input_dim, hidden_dims[1], relu),
        Dense(hidden_dims[1], hidden_dims[2], relu),
        Dense(hidden_dims[2], m)
    )
end

# ------------------------------------------------------------------------------------------
# Loss function
# ------------------------------------------------------------------------------------------
function compute_loss(
    policy,
    batch_data::Vector,
    mech_ref::Dojo.Mechanism,
    x0::Vector
)
    n = length(x0)
    m = Dojo.input_dimension(mech_ref)
    
    total_loss = 0.0f0
    n_samples = length(batch_data)
    
    for (input_vec, u_target) in batch_data
        # Extract targets (keep as scalars to avoid dimension issues)
        target_x = input_vec[n+1]
        target_y = input_vec[n+2]
        target_z = input_vec[n+3]
        
        # Extract state as a copy (not a view) to avoid indexing gradient issues
        x_state = Vector(input_vec[1:n])
        
        # Convert to Float32 for Flux network
        input_f32 = Float32.(input_vec)
        
        # Predict action
        u_pred = policy(input_f32)
        
        # Convert back to Float64 for Dojo and clamp action to bounds
        u_pred_f64 = Float64.(u_pred)
        u_pred_clamped = clamp.(u_pred_f64, -u_max_abs, u_max_abs)
        
        # Loss 1: Action regularization
        loss_u = w_u * sum(u_pred_clamped .^ 2)
        
        # Loss 2: Rollout one step and compute tracking error
        mech = deepcopy(mech_ref)
        x_next = dojo_step(mech, x_state, u_pred_clamped; opts=DOJO_OPTS)
        
        loss_track = (
            w_track_xy * ((x_next[BASE_X_IDX] - target_x)^2 + (x_next[BASE_Y_IDX] - target_y)^2) +
            w_track_z  * (x_next[BASE_Z_IDX] - target_z)^2
        )
        
        total_loss += loss_track + loss_u
    end
    
    return total_loss / n_samples
end

# ------------------------------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------------------------------
function train_policy(
    mech_ref::Dojo.Mechanism,
    x0::Vector,
    path_xy::Matrix;
    n_trajectories::Int = 50
)
    n = length(x0)
    m = Dojo.input_dimension(mech_ref)
    
    println("Generating training dataset with $n_trajectories trajectories...")
    dataset = generate_waypoint_dataset(mech_ref, x0, path_xy, n_trajectories)
    println("Generated $(length(dataset)) training samples")
    
    println("Creating policy network...")
    policy = create_policy_network(n, m)
    
    # Setup optimizer
    opt_state = Flux.setup(Adam(learning_rate), policy)
    
    println("\nTraining for $N_epochs epochs...")
    
    for epoch in 1:N_epochs
        # Shuffle and batch data
        shuffled = shuffle(dataset)
        batches = Iterators.partition(shuffled, batch_size)
        
        epoch_loss = 0.0
        n_batches = 0
        
        for (batch_idx, batch) in enumerate(batches)
            # Convert batch to Vector (Iterators.partition returns SubArray)
            batch_vec = collect(batch)
            
            # Compute loss and gradients
            loss, grads = Flux.withgradient(
                policy -> compute_loss(policy, batch_vec, mech_ref, x0),
                policy
            )
            
            # Update parameters
            Flux.update!(opt_state, policy, grads[1])
            
            epoch_loss += loss
            n_batches += 1
            
            # Print progress every 10 batches
            if mod(batch_idx, 10) == 0
                println("  Epoch $epoch, Batch $batch_idx/$n_batches - Batch Loss: $(round(loss, digits=6))")
            end
        end
        
        epoch_loss /= n_batches
        
        if mod(epoch, 10) == 0 || epoch == 1
            println("Epoch $epoch/$N_epochs - Avg Loss: $(round(epoch_loss, digits=6))")
            
            # Visualize policy rollout every 10 epochs
            println("  Testing policy rollout...")
            test_traj_x, test_traj_u, test_wp_indices = evaluate_policy(policy, mech_ref, x0, path_xy; horizon=N_TRAJ)
            x_final = test_traj_x[end]
            final_waypoint = path_xy[end, :]
            dist_to_goal = sqrt((x_final[BASE_X_IDX] - final_waypoint[1])^2 + (x_final[BASE_Y_IDX] - final_waypoint[2])^2)
            println("  Distance to goal: $(round(dist_to_goal, digits=4)) | Waypoints reached: $(unique(test_wp_indices))")
            
            # Visualize every 20 epochs
            if mod(epoch, 20) == 0
                println("  Rendering visualization...")
                viz_mech = deepcopy(mech_ref)
                Dojo.set_minimal_state!(viz_mech, x0)
                
                function temp_controller!(mechanism, k)
                    x_k = Dojo.get_minimal_state(mechanism)
                    min_dist = Inf
                    closest_wp_idx = 1
                    for (i, wp) in enumerate(eachrow(path_xy))
                        dist = sqrt((x_k[BASE_X_IDX] - wp[1])^2 + (x_k[BASE_Y_IDX] - wp[2])^2)
                        if dist < min_dist
                            min_dist = dist
                            closest_wp_idx = i
                        end
                    end
                    next_wp_idx = min(closest_wp_idx + 1, size(path_xy, 1))
                    next_wp = path_xy[next_wp_idx, :]
                    input = vcat(x_k, next_wp..., z_nom)
                    input_f32 = Float32.(input)
                    u_k = policy(input_f32)
                    u_k = Float64.(u_k)
                    u_k = clamp.(u_k, -u_max_abs, u_max_abs)
                    Dojo.set_input!(mechanism, u_k)
                end
                
                Tsim = dt * (N_TRAJ - 1)
                storage = Dojo.simulate!(viz_mech, Tsim, temp_controller!; record=true, opts=DOJO_OPTS)
                vis = Dojo.visualize(viz_mech, storage)
                Dojo.render(vis)
                println("  Visualization rendered (Epoch $epoch)")
            end
        end
    end
    
    println("\nTraining complete!")
    
    return policy
end

# ------------------------------------------------------------------------------------------
# Evaluation and rollout
# ------------------------------------------------------------------------------------------
function evaluate_policy(
    policy,
    mech_ref::Dojo.Mechanism,
    x0::Vector,
    path_xy::Matrix;
    horizon::Int = N_TRAJ
)
    """
    Evaluate policy by rolling out trajectory following the path.
    
    Returns:
        trajectory_x: list of states
        trajectory_u: list of actions
        waypoint_indices: which waypoint was active at each step
    """
    n = length(x0)
    m = Dojo.input_dimension(mech_ref)
    n_waypoints = size(path_xy, 1)
    
    mech = deepcopy(mech_ref)
    x_current = copy(x0)
    
    trajectory_x = [copy(x_current)]
    trajectory_u = []
    waypoint_indices = [1]
    
    wp_idx = 1
    steps_on_path = 0
    
    for step in 1:(horizon - 1)
        # Get next waypoint
        next_wp = path_xy[min(wp_idx + 1, n_waypoints), :]
        
        # Create input: [state; target]
        input = vcat(x_current, next_wp..., z_nom)
        
        # Get action from policy
        # Get action from policy
        input_f32 = Float32.(input)
        u_pred = policy(input_f32)
        u_pred = Float64.(u_pred)
        u_pred = clamp.(u_pred, -u_max_abs, u_max_abs)
        
        push!(trajectory_u, u_pred)
        
        # Step dynamics
        x_current = dojo_step(mech, x_current, u_pred; opts=DOJO_OPTS)
        push!(trajectory_x, copy(x_current))
        
        # Check if we reached the waypoint
        dist_to_wp = sqrt((x_current[BASE_X_IDX] - next_wp[1])^2 + (x_current[BASE_Y_IDX] - next_wp[2])^2)
        steps_on_path += 1
        
        if dist_to_wp < 0.1 && steps_on_path > 3 && wp_idx < n_waypoints
            wp_idx += 1
            steps_on_path = 0
        end
        
        push!(waypoint_indices, wp_idx)
    end
    
    return trajectory_x, trajectory_u, waypoint_indices
end

# ------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------
println("=" ^ 80)
println("Training Quadruped Policy with Flux.jl - Path Following")
println("=" ^ 80)

mech, x0, u0 = build_quadruped()
n, m = length(x0), length(u0)

# Define a curved path for the robot to follow
# You can modify this to any curved path you want
# Format: (n_waypoints, 2) matrix with [x, y] coordinates
path_xy = hcat(
    range(x0[BASE_X_IDX], x0[BASE_X_IDX] + 1.0, length=15),  # x coordinates (curved line)
    range(x0[BASE_Y_IDX], x0[BASE_Y_IDX] + 0.5, length=15)   # y coordinates
)

# You can also create more complex curved paths, for example:
# θ = range(0, 2π, length=20)
# path_xy = hcat(x0[BASE_X_IDX] .+ 0.5*cos.(θ), x0[BASE_Y_IDX] .+ 0.5*sin.(θ))

println("\nQuadruped dims: n=$n, m=$m")
println("Path: $(size(path_xy, 1)) waypoints")
println("Trajectory length: $N_TRAJ steps")
println("Learning rate: $learning_rate")
println("Batch size: $batch_size")
println("\nPath waypoints:")
for (i, wp) in eachrow(path_xy) |> enumerate
    println("  [$i]: x=$(round(wp[1], digits=3)), y=$(round(wp[2], digits=3))")
end

# Train policy
policy = train_policy(mech, x0, path_xy; n_trajectories=50)

# Evaluate on the path
println("\n" * "=" ^ 80)
println("Evaluating Policy on Path")
println("=" ^ 80)

println("\nRolling out with policy following path...")
traj_x, traj_u, wp_indices = evaluate_policy(policy, mech, x0, path_xy; horizon=N_TRAJ)

println("\nFinal base position:")
x_final = traj_x[end]
println("  x = $(round(x_final[BASE_X_IDX], digits=4))")
println("  y = $(round(x_final[BASE_Y_IDX], digits=4))")
println("  z = $(round(x_final[BASE_Z_IDX], digits=4))")

println("\nWaypoint progression: $(unique(wp_indices))")

# Distance to final path waypoint
final_waypoint = path_xy[end, :]
dist_xy = sqrt((x_final[BASE_X_IDX] - final_waypoint[1])^2 + (x_final[BASE_Y_IDX] - final_waypoint[2])^2)
dist_z  = abs(x_final[BASE_Z_IDX] - z_nom)
println("\nDistance to final waypoint:")
println("  xy distance = $(round(dist_xy, digits=4))")
println("  z distance  = $(round(dist_z, digits=4))")

# Simulate with visualization
println("\n" * "=" ^ 80)
println("Simulating and Visualizing")
println("=" ^ 80)

function controller_policy!(mechanism, k)
    # Get current state
    x_k = Dojo.get_minimal_state(mechanism)
    
    # Find closest waypoint
    min_dist = Inf
    closest_wp_idx = 1
    for (i, wp) in eachrow(path_xy) |> enumerate
        dist = sqrt((x_k[BASE_X_IDX] - wp[1])^2 + (x_k[BASE_Y_IDX] - wp[2])^2)
        if dist < min_dist
            min_dist = dist
            closest_wp_idx = i
        end
    end
    
    # Use next waypoint as target
    next_wp_idx = min(closest_wp_idx + 1, size(path_xy, 1))
    next_wp = path_xy[next_wp_idx, :]
    
    # Create policy input
    input = vcat(x_k, next_wp..., z_nom)
    
    # Get action (convert to Float32 for policy, then back to Float64)
    input_f32 = Float32.(input)
    u_k = policy(input_f32)
    u_k = Float64.(u_k)
    u_k = clamp.(u_k, -u_max_abs, u_max_abs)
    
    Dojo.set_input!(mechanism, u_k)
end

Dojo.set_minimal_state!(mech, x0)
Tsim = dt * (N_TRAJ - 1)
storage = Dojo.simulate!(mech, Tsim, controller_policy!; record=true, opts=DOJO_OPTS)

# Visualize
vis = Dojo.visualize(mech, storage)
Dojo.render(vis)

println("\nTrajectory visualization complete!")
