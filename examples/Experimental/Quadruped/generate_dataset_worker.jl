#!/usr/bin/env julia
# ============================================================================
# Worker script for parallel dataset generation (called by run_array_job.sbatch)
# ============================================================================
# Usage: julia generate_dataset_worker.jl <num_trajs> <output_dir> <seed> <task_id> <start_idx>
# ============================================================================

# instantiate environment
println("Activating project environment...")
using Pkg
Pkg.activate(@__DIR__)
# Pkg.instantiate()

### Load packages
println("Loading packages...")

using Dojo
using DojoEnvironments
using LinearAlgebra
using Random
using HDF5
using Statistics

# Include shared controller
println("Including shared controller...")
include("quadruped_controller.jl")

# ============================
# Parse command line arguments
# ============================
if length(ARGS) < 5
    println("Usage: julia generate_dataset_worker.jl <num_trajectories> <output_dir> <seed> <task_id> <start_idx>")
    println("This script is meant to be called by run_array_job.sbatch")
    exit(1)
end

const NUM_TRAJECTORIES = parse(Int, ARGS[1])
const OUTPUT_DIR = ARGS[2]
const RANDOM_SEED = parse(Int, ARGS[3])
const TASK_ID = parse(Int, ARGS[4])
const START_IDX = parse(Int, ARGS[5])

println("Worker Task $TASK_ID: Generating $NUM_TRAJECTORIES trajectories")
println("Output directory: $OUTPUT_DIR")
println("Random seed: $RANDOM_SEED")
println("Starting index: $START_IDX")

Random.seed!(RANDOM_SEED)

# Create output directory
mkpath(OUTPUT_DIR)

# ============================
# Command generation parameters
# ============================
const MIN_COMMANDS = 5
const MAX_COMMANDS = 15
const COMMAND_POOL = [1, 2, 3, 4]

# ============================
# Command sequence generation
# ============================
function generate_command_sequence()
    num_cmds = rand(MIN_COMMANDS:MAX_COMMANDS)
    sequence = Int[]
    for i in 1:num_cmds
        push!(sequence, rand(COMMAND_POOL))
        if i < num_cmds
            push!(sequence, QC_CMD_STAND)
        end
    end
    return sequence
end

# ============================
# Trajectory simulation wrapper
# ============================
function simulate_trajectory(traj_id::Int, sequence::Vector{Int})
    println("  Trajectory $traj_id: $(length(sequence)) commands")
    
    # Use shared controller for simulation
    states, actions, is_stable, reason, _ = qc_simulate_trajectory(sequence; dt=QC_dt, record=false)
    
    println("  Stability: $is_stable ($reason)")
    
    return (states, actions, is_stable, reason)
end

# ============================
# Worker main loop
# ============================
function generate_dataset_worker()
    println("\n" * "="^60)
    println("Worker Task $TASK_ID: Starting dataset generation")
    println("="^60 * "\n")
    
    # Check for existing trajectories from this task to resume
    task_prefix = "trajectory_task$(lpad(TASK_ID, 3, '0'))_"
    existing_files = filter(f -> startswith(f, task_prefix) && endswith(f, ".h5"), readdir(OUTPUT_DIR))
    start_from = length(existing_files)
    
    if start_from > 0
        println("Task $TASK_ID: Found $start_from existing trajectories - resuming from $(start_from + 1)")
        # Note: RNG state cannot be perfectly restored because we don't know how many
        # rejected trajectories occurred. The resumed run will generate different 
        # command sequences, but will still produce valid stable trajectories.
        # This is fine for training data where we just need diverse stable trajectories.
        println("Task $TASK_ID: Note - resumed trajectories may have different command sequences")
        println()
    end
    
    stable_count = start_from
    rejected_count = 0
    attempts = 0
    
    while stable_count < NUM_TRAJECTORIES
        attempts += 1
        global_traj_id = START_IDX + stable_count
        
        println("\nTask $TASK_ID: Generating trajectory $(stable_count + 1)/$NUM_TRAJECTORIES (global ID: $global_traj_id, attempt $attempts)...")
        
        sequence = generate_command_sequence()
        states, actions, is_stable, reason = simulate_trajectory(global_traj_id, sequence)
        
        if !is_stable
            rejected_count += 1
            println("  ✗ Rejected: $reason")
            continue
        end
        
        # Save with task ID prefix for later merging
        output_file = joinpath(OUTPUT_DIR, "trajectory_task$(lpad(TASK_ID, 3, '0'))_$(lpad(stable_count + 1, 5, '0')).h5")
        
        try
            h5open(output_file, "w") do file
                file["states"] = states
                file["actions"] = actions
                file["command_sequence"] = sequence
                file["initial_state"] = states[:, 1]
                
                attrs(file)["trajectory_id"] = global_traj_id
                attrs(file)["task_id"] = TASK_ID
                attrs(file)["local_id"] = stable_count + 1
                attrs(file)["is_stable"] = is_stable
                attrs(file)["stability_reason"] = reason
                attrs(file)["dt"] = QC_dt
                attrs(file)["cmd_duration"] = QC_cmd_duration
                attrs(file)["num_timesteps"] = size(states, 2)
                attrs(file)["state_dim"] = size(states, 1)
            end
            
            stable_count += 1
            file_size_kb = filesize(output_file) / 1024
            println("  ✓ Saved to $output_file ($(round(file_size_kb, digits=1)) KB)")
        catch e
            println("  ERROR: Failed to save trajectory - $e")
        end
    end
    
    println("\n" * "="^60)
    println("Worker Task $TASK_ID: Complete!")
    println("="^60)
    println("Stable trajectories saved: $stable_count")
    if start_from > 0
        println("  (resumed from $start_from, generated $(stable_count - start_from) new)")
    end
    println("Rejected (unstable): $rejected_count")
    println("Total attempts: $attempts")
    if attempts > 0
        println("Acceptance rate: $(round(100*(stable_count - start_from)/attempts, digits=1))%")
    end
    println("="^60 * "\n")
    
    # Save task summary
    summary_file = joinpath(OUTPUT_DIR, "summary_task$(lpad(TASK_ID, 3, '0')).h5")
    h5open(summary_file, "w") do file
        attrs(file)["task_id"] = TASK_ID
        attrs(file)["num_trajectories"] = stable_count
        attrs(file)["rejected_count"] = rejected_count
        attrs(file)["total_attempts"] = attempts
        attrs(file)["acceptance_rate"] = attempts > 0 ? (stable_count - start_from) / attempts : 1.0
        attrs(file)["random_seed"] = RANDOM_SEED
        attrs(file)["dt"] = QC_dt
        attrs(file)["cmd_duration"] = QC_cmd_duration
        attrs(file)["resumed_from"] = start_from
    end
    println("Task summary saved to $summary_file\n")
end

# Run!
generate_dataset_worker()
