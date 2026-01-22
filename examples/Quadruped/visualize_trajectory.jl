#!/usr/bin/env julia
# ============================================================================
# Visualize and validate a saved quadruped trajectory
# ============================================================================
# usage example:
#   julia --project -i visualize_trajectory.jl ./data_test/trajectory_00001.h5
#
# This script:
# 1. Loads the saved trajectory (states, actions, command sequence)
# 2. Re-simulates with the same controller and command sequence
# 3. Compares the recorded vs re-simulated trajectories
# 4. Visualizes the re-simulated trajectory
# ============================================================================

using Dojo
using DojoEnvironments
using LinearAlgebra
using HDF5
using Statistics

# Include shared controller
include("quadruped_controller.jl")

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
recorded_states = h5read(TRAJECTORY_FILE, "states")
recorded_actions = h5read(TRAJECTORY_FILE, "actions")
command_sequence = h5read(TRAJECTORY_FILE, "command_sequence")

file_attrs = h5readattr(TRAJECTORY_FILE, "/")
trajectory_id = file_attrs["trajectory_id"]
is_stable = file_attrs["is_stable"]
stability_reason = file_attrs["stability_reason"]
dt = file_attrs["dt"]
cmd_duration = file_attrs["cmd_duration"]

N_total = size(recorded_states, 2)
state_dim = size(recorded_states, 1)

println("\n" * "="^60)
println("Recorded Trajectory Information")
println("="^60)
println("Trajectory ID: $trajectory_id")
println("Total timesteps: $N_total")
println("Duration: $(round(N_total * dt, digits=2)) seconds")
println("State dimension: $state_dim")
println("Action dimension: $(size(recorded_actions, 1))")
println("Command sequence: $command_sequence")
println("Recorded stability: $is_stable ($stability_reason)")
println("="^60 * "\n")

# ============================
# Re-simulate with same controller and command sequence
# ============================
println("Re-simulating with feedback controller...")
println("Using command sequence: $command_sequence")

sim_states, sim_actions, sim_stable, sim_reason, env = qc_simulate_trajectory(
    command_sequence; 
    dt=dt, 
    record=true
)

println("\n" * "="^60)
println("Re-simulation Results")
println("="^60)
println("Stability: $sim_stable ($sim_reason)")
println("States shape: $(size(sim_states))")
println("Actions shape: $(size(sim_actions))")
println("="^60 * "\n")

# ============================
# Compare recorded vs re-simulated
# ============================
println("="^60)
println("Trajectory Comparison: Recorded vs Re-simulated")
println("="^60)

N_compare = min(size(recorded_states, 2), size(sim_states, 2))

# Position comparison
recorded_pos = recorded_states[1:3, 1:N_compare]
sim_pos = sim_states[1:3, 1:N_compare]

pos_errors = abs.(recorded_pos .- sim_pos)
x_error = mean(pos_errors[1, :])
y_error = mean(pos_errors[2, :])
z_error = mean(pos_errors[3, :])
total_pos_error = sqrt(x_error^2 + y_error^2 + z_error^2)

println("\nPosition Error (MAE) over $N_compare timesteps:")
println("  X: $(round(x_error, digits=6)) m")
println("  Y: $(round(y_error, digits=6)) m")
println("  Z: $(round(z_error, digits=6)) m")
println("  Total: $(round(total_pos_error, digits=6)) m")

# Final position comparison
println("\nFinal position comparison:")
println("  Recorded: [$(round(recorded_states[1, end], digits=4)), $(round(recorded_states[2, end], digits=4)), $(round(recorded_states[3, end], digits=4))]")
println("  Re-sim:   [$(round(sim_states[1, end], digits=4)), $(round(sim_states[2, end], digits=4)), $(round(sim_states[3, end], digits=4))]")

final_pos_diff = norm(recorded_states[1:3, end] - sim_states[1:3, end])
println("  Distance: $(round(final_pos_diff, digits=4)) m")

# Action comparison
action_errors = abs.(recorded_actions[:, 1:N_compare] .- sim_actions[:, 1:N_compare])
mean_action_error = mean(action_errors)
max_action_error = maximum(action_errors)

println("\nAction Error:")
println("  Mean: $(round(mean_action_error, digits=4)) Nm")
println("  Max:  $(round(max_action_error, digits=4)) Nm")

# Assessment
println("\n" * "-"^60)
if total_pos_error < 1e-6 && mean_action_error < 1e-6
    println("✓ PERFECT: Trajectories are identical (deterministic replay)")
elseif total_pos_error < 0.01
    println("✓ EXCELLENT: Very close match (< 1cm average error)")
elseif total_pos_error < 0.1
    println("✓ GOOD: Close match (< 10cm average error)")
elseif total_pos_error < 0.5
    println("⚠ FAIR: Some divergence (this may be expected with feedback control)")
else
    println("✗ POOR: Significant divergence")
    println("  This could indicate:")
    println("  - Different initial conditions")
    println("  - Numerical precision differences")
    println("  - Controller parameter mismatch")
end
println("-"^60)

# ============================
# Visualize
# ============================
println("\nRendering visualization of re-simulated trajectory...")
vis = visualize(env)
render(vis)

println("\n" * "="^60)
println("Visualization ready!")
println("The visualization shows the RE-SIMULATED trajectory")
println("(using the same controller and command sequence)")
println("="^60 * "\n")
