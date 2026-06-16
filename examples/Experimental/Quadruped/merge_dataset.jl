#!/usr/bin/env julia
# ============================================================================
# Merge parallel dataset files into a single clean dataset
# ============================================================================
# Usage: julia merge_dataset.jl <input_dir> [output_dir]
#
# This script:
# 1. Reads all trajectory_task*_*.h5 files from parallel generation
# 2. Renumbers them sequentially as trajectory_00001.h5, trajectory_00002.h5, etc.
# 3. Creates a combined summary.h5 file
# ============================================================================

using HDF5
using Statistics

if length(ARGS) < 1
    println("Usage: julia merge_dataset.jl <input_dir> [output_dir]")
    println("Example: julia merge_dataset.jl ./data_parallel ./data_merged")
    exit(1)
end

const INPUT_DIR = ARGS[1]
const OUTPUT_DIR = length(ARGS) >= 2 ? ARGS[2] : INPUT_DIR

println("\n" * "="^60)
println("Merging Parallel Dataset")
println("="^60)
println("Input directory: $INPUT_DIR")
println("Output directory: $OUTPUT_DIR")
println()

# Create output directory if different from input
if OUTPUT_DIR != INPUT_DIR
    mkpath(OUTPUT_DIR)
end

# Find all trajectory files from parallel tasks
trajectory_files = filter(f -> startswith(f, "trajectory_task") && endswith(f, ".h5"), readdir(INPUT_DIR))
sort!(trajectory_files)

if isempty(trajectory_files)
    println("No trajectory_task*_*.h5 files found in $INPUT_DIR")
    println("Looking for regular trajectory files...")
    trajectory_files = filter(f -> startswith(f, "trajectory_") && endswith(f, ".h5") && !contains(f, "task"), readdir(INPUT_DIR))
    if isempty(trajectory_files)
        println("No trajectory files found!")
        exit(1)
    else
        println("Found $(length(trajectory_files)) regular trajectory files - no merge needed")
        exit(0)
    end
end

println("Found $(length(trajectory_files)) trajectory files from parallel tasks")

# Collect task summaries
summary_files = filter(f -> startswith(f, "summary_task") && endswith(f, ".h5"), readdir(INPUT_DIR))
total_rejected = 0
total_attempts = 0

for sf in summary_files
    try
        attrs = h5readattr(joinpath(INPUT_DIR, sf), "/")
        total_rejected += attrs["rejected_count"]
        total_attempts += attrs["total_attempts"]
    catch
    end
end

# Rename and copy files
println("\nRenumbering trajectories...")
new_id = 0

for old_file in trajectory_files
    new_id += 1
    old_path = joinpath(INPUT_DIR, old_file)
    new_file = "trajectory_$(lpad(new_id, 5, '0')).h5"
    new_path = joinpath(OUTPUT_DIR, new_file)
    
    if OUTPUT_DIR == INPUT_DIR
        # Rename in place
        mv(old_path, new_path; force=true)
    else
        # Copy to new location with new name
        cp(old_path, new_path; force=true)
        
        # Update trajectory_id in the file
        h5open(new_path, "r+") do file
            attrs(file)["trajectory_id"] = new_id
        end
    end
    
    if new_id % 100 == 0 || new_id == length(trajectory_files)
        println("  Processed $new_id / $(length(trajectory_files)) files")
    end
end

# Get dt and cmd_duration from first file
dt = 0.001
cmd_duration = 1.6
try
    first_file = joinpath(OUTPUT_DIR, "trajectory_00001.h5")
    attrs = h5readattr(first_file, "/")
    dt = attrs["dt"]
    cmd_duration = attrs["cmd_duration"]
catch
end

# Create combined summary
num_trajectories = new_id
acceptance_rate = total_attempts > 0 ? num_trajectories / total_attempts : 1.0

summary_file = joinpath(OUTPUT_DIR, "summary.h5")
h5open(summary_file, "w") do file
    attrs(file)["num_trajectories"] = num_trajectories
    attrs(file)["stable_count"] = num_trajectories
    attrs(file)["rejected_count"] = total_rejected
    attrs(file)["total_attempts"] = total_attempts
    attrs(file)["acceptance_rate"] = acceptance_rate
    attrs(file)["random_seed"] = 0  # Mixed seeds from parallel tasks
    attrs(file)["dt"] = dt
    attrs(file)["cmd_duration"] = cmd_duration
    attrs(file)["merged"] = true
    attrs(file)["num_tasks"] = length(summary_files)
end

# Clean up task-specific files if merging in place
if OUTPUT_DIR == INPUT_DIR
    println("\nCleaning up task-specific files...")
    for sf in summary_files
        rm(joinpath(INPUT_DIR, sf); force=true)
    end
    println("  Removed $(length(summary_files)) task summary files")
end

println("\n" * "="^60)
println("Merge Complete!")
println("="^60)
println("Total trajectories: $num_trajectories")
println("Total rejected: $total_rejected")
println("Total attempts: $total_attempts")
println("Acceptance rate: $(round(100*acceptance_rate, digits=1))%")
println("Output directory: $OUTPUT_DIR")
println("Summary: $summary_file")
println("="^60 * "\n")
