# Quadruped Imitation Learning Dataset Generation

This directory contains scripts for generating a dataset of quadruped locomotion trajectories for training imitation learning policies.

## Scripts

### 1. `generate_dataset.jl`
Generates a dataset of quadruped trajectories with random command sequences.

**Features:**
- Random command sequences with standing breaks for stability
- Automatic stability checking for each trajectory
- Saves states, actions, and metadata to h5 files
- Configurable number of trajectories for testing

**Usage:**
```bash
julia --project generate_dataset.jl <num_trajectories> <output_dir> [random_seed]
```

**Examples:**
```bash
# Generate 5 test trajectories (good for testing)
julia --project generate_dataset.jl 5 ./data_test 42

# Generate 100 trajectories for training
julia --project generate_dataset.jl 100 ./data_train 123

# Generate 1000 trajectories for full dataset
julia --project generate_dataset.jl 1000 ./data_full 456
```

### 2. `visualize_trajectory.jl`
Visualizes a saved trajectory by replaying the recorded actions.

**Usage:**
```bash
julia --project -i visualize_trajectory.jl <trajectory_file.h5>
```

**Example:**
```bash
julia --project -i visualize_trajectory.jl ./data_test/trajectory_00001.h5
```

### 3. `run_job.sbatch`
SLURM batch script for generating datasets on the cluster.

**Usage:**
```bash
# Submit with default parameters (100 trajectories, ./data directory)
sbatch run_job.sbatch

# Submit with custom parameters
sbatch run_job.sbatch 500 ./my_data 789
```

**Parameters:**
- `$1` (default: 100): Number of trajectories
- `$2` (default: ./data): Output directory
- `$3` (default: 42): Random seed

## Output Format

Each stable trajectory is saved as an HDF5 file containing:
- `states`: Matrix of states (state_dim × T)
- `actions`: Matrix of actions (12 × T)
- `command_sequence`: Vector of high-level commands
- Attributes (stored in file metadata):
  - `trajectory_id`: Unique identifier
  - `is_stable`: Boolean (always True for saved trajectories)
  - `stability_reason`: String describing stability status
  - `dt`: Timestep duration
  - `cmd_duration`: Duration of each command
  - `num_timesteps`: Total timesteps
  - `state_dim`: Dimension of state vector

**Note:** Only stable trajectories are saved. Unstable trajectories are rejected before saving.

A `summary.h5` file is also created with dataset statistics.

## Command Encoding

High-level commands:
- `1`: Forward
- `2`: Backward
- `3`: Turn right
- `4`: Turn left
- `5`: Stand/stop

## Stability Checking

The `check_stability()` function evaluates trajectories based on:
1. Z-position (height above ground)
2. Numerical stability (no NaN/Inf values)
3. Position bounds (no explosions)
4. Velocity bounds
5. Orientation consistency

**Only stable trajectories are saved to disk.** Unstable trajectories are rejected during generation, resulting in a cleaner dataset and faster training.

## Quick Start

1. **Test with a few trajectories:**
   ```bash
   julia --project generate_dataset.jl 5 ./data_test
   ```

2. **Visualize a trajectory:**
   ```bash
   julia --project visualize_trajectory.jl ./data_test/trajectory_00001.h5
   ```

3. **Generate full dataset on cluster:**
   ```bash
   sbatch run_job.sbatch 1000 ./data_train
   ```

4. **Check results:**
   ```julia
   using HDF5
   attrs = h5readattr("./data_train/summary.h5", "/")
   println("Stable trajectories: ", attrs["stable_count"])
   println("Acceptance rate: ", round(100*attrs["acceptance_rate"], digits=1), "%")
   ```

## Performance Notes

### File Format: HDF5 vs h5

**HDF5 advantages for training:**
- 50-70% smaller files (with compression) → faster I/O
- Efficient partial loading (load specific trajectories without reading entire file)
- Better compression (gzip level 4)
- Industry standard for machine learning datasets
- Faster disk access patterns

### Stability Filtering

By rejecting unstable trajectories **before saving:**
- Cleaner dataset (all saved data is usable)
- Faster training loop (no need to filter during epoch)
- Reduced disk usage (only keep good trajectories)
- Automatic re-sampling: if you request N trajectories, you get N stable ones

**Acceptance rate** metric shows what percentage of simulations produce stable trajectories.

## Next Steps

After generating a sufficient dataset:
1. Filter for stable trajectories only
2. Train a neural network policy to predict actions from states
3. Evaluate the learned policy in simulation
4. Fine-tune with online learning if needed
