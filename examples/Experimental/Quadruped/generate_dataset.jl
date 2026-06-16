#!/usr/bin/env julia
# ============================================================================
# Dataset generation for quadruped imitation learning
# ============================================================================
# Usage: julia --project generate_dataset.jl <num_trajectories> <output_dir> [random_seed]
# Example: julia --project generate_dataset.jl 100 ./data_train 42
# ============================================================================

using Dojo
using DojoEnvironments
using LinearAlgebra
using Random
using HDF5
using Statistics

# Include shared controller
include("quadruped_controller.jl")

# ============================
# Parse command line arguments
# ============================
if length(ARGS) < 2
    println("Usage: julia generate_dataset.jl <num_trajectories> <output_dir> [random_seed]")
    println("Example: julia generate_dataset.jl 5 ./data 42")
    exit(1)
end

const NUM_TRAJECTORIES = parse(Int, ARGS[1])
const OUTPUT_DIR = ARGS[2]
const RANDOM_SEED = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 42

println("Generating $NUM_TRAJECTORIES trajectories")
println("Output directory: $OUTPUT_DIR")
println("Random seed: $RANDOM_SEED")

Random.seed!(RANDOM_SEED)

# Create output directory
mkpath(OUTPUT_DIR)

# ============================
# Command generation parameters
# ============================
const MIN_COMMANDS = 5
const MAX_COMMANDS = 15
const COMMAND_POOL = [1, 2, 3, 4]  # forward, backward, turn right, turn left

# ============================
# Command sequence generation
# ============================
"""
Generate a random command sequence with standing breaks for stability.
Returns vector of command integers.
"""
function generate_command_sequence()
    num_cmds = rand(MIN_COMMANDS:MAX_COMMANDS)
    sequence = Int[]
    
    for i in 1:num_cmds
        # Add a movement command
        push!(sequence, rand(COMMAND_POOL))
        # Add standing break (except after last command)
        if i < num_cmds
            push!(sequence, QC_CMD_STAND)
        end
    end
    
    return sequence
end

# ============================
# Trajectory simulation wrapper
# ============================
"""
Simulate one trajectory and return states, actions, and stability info.
"""
function simulate_trajectory(traj_id::Int, sequence::Vector{Int})
    println("  Trajectory $traj_id: $(length(sequence)) commands")
    println("  Command sequence: $sequence")
    
    # Use shared controller for simulation
    states, actions, is_stable, reason, _ = qc_simulate_trajectory(sequence; dt=QC_dt, record=false)
    
    println("  Stability: $is_stable ($reason)")
    
    return (states, actions, is_stable, reason)
end

# ============================
# Main data generation loop
# ============================
function generate_dataset()
    println("\n" * "="^60)
    println("Starting dataset generation")
    println("="^60 * "\n")
    
    # Check for existing trajectories to resume from
    existing_files = filter(f -> startswith(f, "trajectory_") && endswith(f, ".h5") && !contains(f, "task"), readdir(OUTPUT_DIR))
    start_from = length(existing_files)
    
    if start_from > 0
        println("Found $start_from existing trajectories - resuming from trajectory $(start_from + 1)")
        println()
    end
    
    stable_count = start_from
    rejected_count = 0
    attempts = 0
    
    while stable_count < NUM_TRAJECTORIES
        attempts += 1
        traj_id = stable_count + 1
        
        println("\nGenerating trajectory $traj_id/$NUM_TRAJECTORIES (attempt $attempts)...")
        
        # Generate command sequence
        sequence = generate_command_sequence()
        
        # Simulate
        states, actions, is_stable, reason = simulate_trajectory(traj_id, sequence)
        
        # CHECK STABILITY FIRST - only save if stable
        if !is_stable
            rejected_count += 1
            println("  ✗ Rejected: $reason")
            continue
        end
        
        # Save trajectory (only stable ones)
        output_file = joinpath(OUTPUT_DIR, "trajectory_$(lpad(stable_count + 1, 5, '0')).h5")
        
        try
            h5open(output_file, "w") do file
                # Create datasets with compression
                file["states"] = states
                file["actions"] = actions
                file["command_sequence"] = sequence
                
                # Save initial state for exact replay
                file["initial_state"] = states[:, 1]
                
                # Store metadata as attributes
                attrs(file)["trajectory_id"] = stable_count + 1
                attrs(file)["is_stable"] = is_stable
                attrs(file)["stability_reason"] = reason
                attrs(file)["dt"] = QC_dt
                attrs(file)["cmd_duration"] = QC_cmd_duration
                attrs(file)["num_timesteps"] = size(states, 2)
                attrs(file)["state_dim"] = size(states, 1)
                
                # Enable compression on datasets
                file["states"]["compress"] = 4  # gzip level 4
                file["actions"]["compress"] = 4
            end
            
            stable_count += 1
            file_size_kb = filesize(output_file) / 1024
            println("  ✓ Saved stable trajectory to $output_file ($(round(file_size_kb, digits=1)) KB)")
        catch e
            println("  ERROR: Failed to save trajectory - $e")
        end
    end
    
    println("\n" * "="^60)
    println("Dataset generation complete!")
    println("="^60)
    println("Total stable trajectories saved: $stable_count")
    if start_from > 0
        println("  (resumed from $start_from, generated $(stable_count - start_from) new)")
    end
    println("Total rejected (unstable): $rejected_count")
    println("Total attempts: $attempts")
    if attempts > 0
        println("Acceptance rate: $(round(100*(stable_count - start_from)/attempts, digits=1))%")
    end
    println("Output directory: $OUTPUT_DIR")
    println("="^60 * "\n")
    
    # Save summary
    summary_file = joinpath(OUTPUT_DIR, "summary.h5")
    h5open(summary_file, "w") do file
        attrs(file)["num_trajectories"] = stable_count
        attrs(file)["stable_count"] = stable_count
        attrs(file)["rejected_count"] = rejected_count
        attrs(file)["total_attempts"] = attempts
        attrs(file)["acceptance_rate"] = attempts > 0 ? (stable_count - start_from) / attempts : 1.0
        attrs(file)["random_seed"] = RANDOM_SEED
        attrs(file)["dt"] = QC_dt
        attrs(file)["cmd_duration"] = QC_cmd_duration
        attrs(file)["resumed_from"] = start_from
    end
    println("Summary saved to $summary_file\n")
end

# Run!
generate_dataset()
