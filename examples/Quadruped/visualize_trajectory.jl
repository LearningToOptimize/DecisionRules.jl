#!/usr/bin/env julia
# usage example:
# julia --project -i visualize_trajectory.jl ./data_test/trajectory_00001.h5
using Dojo
using DojoEnvironments
using LinearAlgebra
using HDF5

# ============================
# Parse command line arguments
# ============================
if length(ARGS) < 1
    println("Usage: julia visualize_trajectory.jl <trajectory_file.h5>")
    println("Example: julia visualize_trajectory.jl ./data/trajectory_00001.h5")
    exit(1)
end

const TRAJECTORY_FILE = ARGS[1]

if !isfile(TRAJECTORY_FILE)
    println("ERROR: File not found: $TRAJECTORY_FILE")
    exit(1)
end

println("Loading trajectory from: $TRAJECTORY_FILE")

# ============================
# Load trajectory data
# ============================
states = h5read(TRAJECTORY_FILE, "states")
actions = h5read(TRAJECTORY_FILE, "actions")
command_sequence = h5read(TRAJECTORY_FILE, "command_sequence")

file_attrs = h5readattr(TRAJECTORY_FILE, "/")
trajectory_id = file_attrs["trajectory_id"]
is_stable = file_attrs["is_stable"]
stability_reason = file_attrs["stability_reason"]
dt = file_attrs["dt"]
cmd_duration = file_attrs["cmd_duration"]

N_total = size(states, 2)
state_dim = size(states, 1)

println("\n" * "="^60)
println("Trajectory Information")
println("="^60)
println("Trajectory ID: $trajectory_id")
println("Total timesteps: $N_total")
println("Duration: $(round(N_total * dt, digits=2)) seconds")
println("State dimension: $state_dim")
println("Action dimension: $(size(actions, 1))")
println("Command sequence: $command_sequence")
println("Stable: $is_stable ($stability_reason)")
println("="^60 * "\n")

# ============================
# Controller parameters (must match generate_dataset.jl)
# ============================
const p_fwd = [0.26525665033578044,
               0.04331492786316298,
               0.8633149317316002,
              -0.3418982967378247,
              -1.3331979248705708]

# ============================
# Replay simulation
# ============================
println("Setting up environment for replay...")

# Create environment with same parameters
env = get_environment(
    :quadruped_sampling;
    horizon=N_total,
    timestep=dt,
    joint_limits=Dict(),
    gravity=-9.81,
    contact_body=false
)

# Initialize to same starting configuration as data generation
function reset_state!(env, p)
    _, _, Cth, _, Ccf = p
    initialize!(env, :quadruped; body_position=[0.0;0.0;-0.43], hip_angle=0.0, thigh_angle=Cth, calf_angle=Ccf)
    calf_state = get_body(env.mechanism, :FR_calf).state
    position = get_sdf(get_contact(env.mechanism, :FR_calf_contact),
                       Dojo.current_position(calf_state),
                       Dojo.current_orientation(calf_state))
    initialize!(env, :quadruped; body_position=[0.0;0.0;-position-0.43], hip_angle=0.0, thigh_angle=Cth, calf_angle=Ccf)
end

reset_state!(env, p_fwd)

# Controller that just plays back recorded actions
function replay_controller!(environment, k)
    u12 = actions[:, k]
    set_input!(environment, u12)
    return nothing
end

println("Simulating with recorded actions...")
simulate!(env, replay_controller!; record=true)

println("Rendering visualization...")
vis = visualize(env)
render(vis)

println("\n✓ Visualization complete!")
println("Trajectory: $(is_stable ? "STABLE" : "UNSTABLE")")
println("="^60 * "\n")
