#!/usr/bin/env julia
using HDF5
using Statistics

# ============================
# Parse command line arguments
# ============================
if length(ARGS) < 1
    println("Usage: julia analyze_dataset.jl <data_directory>")
    println("Example: julia analyze_dataset.jl ./data_test")
    exit(1)
end

const DATA_DIR = ARGS[1]

if !isdir(DATA_DIR)
    println("ERROR: Directory not found: $DATA_DIR")
    exit(1)
end

println("\n" * "="^60)
println("Dataset Analysis")
println("="^60)
println("Directory: $DATA_DIR\n")

# Load summary
summary_file = joinpath(DATA_DIR, "summary.h5")
if !isfile(summary_file)
    println("ERROR: summary.h5 not found in $DATA_DIR")
    exit(1)
end

summary_attrs = h5readattr(summary_file, "/")
num_trajectories = summary_attrs["num_trajectories"]
stable_count = summary_attrs["stable_count"]
rejected_count = summary_attrs["rejected_count"]
total_attempts = summary_attrs["total_attempts"]
acceptance_rate = summary_attrs["acceptance_rate"]

println("Overall Statistics:")
println("  Total stable trajectories saved: $num_trajectories")
println("  Total rejected (unstable): $rejected_count")
println("  Total attempts: $total_attempts")
println("  Acceptance rate: $(round(100*acceptance_rate, digits=1))%")
println()

# Analyze individual trajectories
trajectory_files = filter(f -> startswith(f, "trajectory_") && endswith(f, ".h5"), 
                          readdir(DATA_DIR))
sort!(trajectory_files)

if isempty(trajectory_files)
    println("No trajectory files found!")
    exit(1)
end

println("Analyzing $(length(trajectory_files)) trajectory files...\n")

# Collect statistics
durations = Float64[]
state_means = []
state_stds = []
action_means = []
action_stds = []
command_lengths = Int[]
total_disk_size = 0

for traj_file in trajectory_files
    try
        filepath = joinpath(DATA_DIR, traj_file)
        
        states = h5read(filepath, "states")
        actions = h5read(filepath, "actions")
        cmd_seq = h5read(filepath, "command_sequence")
        attrs = h5readattr(filepath, "/")
        
        dt = attrs["dt"]
        
        # Collect stats
        push!(durations, size(states, 2) * dt)
        push!(command_lengths, length(cmd_seq))
        
        # State/action statistics
        push!(state_means, mean(states, dims=2)[:])
        push!(state_stds, std(states, dims=2)[:])
        push!(action_means, mean(actions, dims=2)[:])
        push!(action_stds, std(actions, dims=2)[:])
        
        # Track disk usage
        total_disk_size += filesize(filepath)
    catch e
        println("WARNING: Could not load $traj_file: $e")
    end
end

println("Trajectory Durations:")
println("  Mean: $(round(mean(durations), digits=2))s")
println("  Std: $(round(std(durations), digits=2))s")
println("  Min: $(round(minimum(durations), digits=2))s")
println("  Max: $(round(maximum(durations), digits=2))s")
println()

println("Command Sequence Lengths:")
println("  Mean: $(round(mean(command_lengths), digits=1))")
println("  Std: $(round(std(command_lengths), digits=1))")
println("  Min: $(minimum(command_lengths))")
println("  Max: $(maximum(command_lengths))")
println()

println("Disk Usage:")
total_mb = total_disk_size / (1024^2)
avg_mb = total_mb / length(trajectory_files)
println("  Total: $(round(total_mb, digits=1)) MB")
println("  Average per trajectory: $(round(avg_mb, digits=2)) MB")
println()

if !isempty(state_means)
    println("Saved Trajectory Statistics (all are stable):")
    
    # Average state statistics
    avg_state_mean = mean(hcat(state_means...), dims=2)[:]
    avg_state_std = mean(hcat(state_stds...), dims=2)[:]
    
    println("  Average state means (first 6 dims):")
    for i in 1:min(6, length(avg_state_mean))
        println("    dim $i: $(round(avg_state_mean[i], digits=3))")
    end
    println()
    
    # Average action statistics
    avg_action_mean = mean(hcat(action_means...), dims=2)[:]
    avg_action_std = mean(hcat(action_stds...), dims=2)[:]
    
    println("  Average action means (first 6 dims):")
    for i in 1:min(6, length(avg_action_mean))
        println("    dim $i: $(round(avg_action_mean[i], digits=3))")
    end
    println()
    
    println("  Average action stds (first 6 dims):")
    for i in 1:min(6, length(avg_action_std))
        println("    dim $i: $(round(avg_action_std[i], digits=3))")
    end
end

println("\n" * "="^60)
println("Analysis complete!")
println("="^60 * "\n")

# Print file list for visualization
if !isempty(trajectory_files)
    println("Sample trajectories (for visualization):")
    for (i, f) in enumerate(trajectory_files[1:min(10, length(trajectory_files))])
        println("  julia --project visualize_trajectory.jl $DATA_DIR/$f")
    end
    if length(trajectory_files) > 10
        println("  ... and $(length(trajectory_files) - 10) more")
    end
    println()
end
