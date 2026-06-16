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
SLURM batch script for generating datasets on the cluster (sequential).

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

### 4. `run_array_job.sbatch`
SLURM **array job** for parallel dataset generation across multiple nodes.

**Usage:**
```bash
# Generate 1000 trajectories using 10 parallel tasks (default)
sbatch run_array_job.sbatch 1000 ./data_parallel 42

# Generate 5000 trajectories using 50 parallel tasks
sbatch --array=1-50 run_array_job.sbatch 5000 ./data_large 42

# Test with 3 tasks generating 30 trajectories
sbatch --array=1-3 run_array_job.sbatch 30 ./data_test 42
```

**Parameters:**
- `$1` (default: 100): Total number of trajectories
- `$2` (default: ./data_parallel): Output directory
- `$3` (default: 42): Base random seed (each task gets seed + task_id*1000)
- `--array=1-N`: Number of parallel tasks (default: 10)

**After completion, merge the results:**
```bash
julia --project merge_dataset.jl ./data_parallel
```

### 5. `merge_dataset.jl`
Merges trajectory files from parallel generation into a clean sequential dataset.

**Usage:**
```bash
# Merge in place (renumber files)
julia --project merge_dataset.jl ./data_parallel

# Merge to a new directory
julia --project merge_dataset.jl ./data_parallel ./data_final
```

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

## Checkpointing & Resume

Both sequential and parallel scripts support **automatic resume** if a job fails midway:

- **Sequential (`generate_dataset.jl`)**: Counts existing `trajectory_*.h5` files in the output directory and resumes from where it left off.

- **Parallel (`generate_dataset_worker.jl`)**: Each task counts its own `trajectory_taskXXX_*.h5` files and resumes independently.

**To resume a failed job:**
```bash
# Sequential - just resubmit with the same parameters
sbatch run_job.sbatch 1000 ./data_train 42

# Parallel - resubmit with the same array range
sbatch --array=1-20 run_array_job.sbatch 2000 ./data_large 42
```

The scripts will automatically detect existing files and continue from where they stopped. No data is lost or duplicated.

**Important note on reproducibility:** When resuming, the RNG state cannot be perfectly restored because we don't track how many rejected trajectories occurred before the failure. This means resumed runs may generate **different command sequences** than if the job had run to completion. However, this is fine for training data since we only need diverse stable trajectories—the exact reproducibility of which specific trajectories are generated is not critical.

## Quick Start

1. **Test with a few trajectories (sequential):**
   ```bash
   julia --project generate_dataset.jl 5 ./data_test
   ```

2. **Visualize a trajectory:**
   ```bash
   julia --project visualize_trajectory.jl ./data_test/trajectory_00001.h5
   ```

3. **Generate full dataset on cluster (sequential):**
   ```bash
   sbatch run_job.sbatch 1000 ./data_train
   ```

4. **Generate large dataset in parallel:**
   ```bash
   # Submit array job with 20 parallel tasks
   sbatch --array=1-20 run_array_job.sbatch 2000 ./data_large 42
   
   # After all tasks complete, merge the results
   julia --project merge_dataset.jl ./data_large
   ```

5. **Check results:**
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

---

## Training a Path-Following Policy

### 6. `train_policy.jl`
Trains a neural network policy using Flux.jl on GPU to follow arbitrary paths.

**Architecture:**
- Input: Current state (36D) + Future path waypoints (relative x,y,z positions)
- Hidden: MLP with LayerNorm [256, 256, 128]
- Output: Joint torques (12D)

**Usage:**
```bash
julia --project train_policy.jl <data_dir> [options]
```

**Options:**
- `--epochs N`: Number of training epochs (default: 100)
- `--batch_size N`: Batch size (default: 256)
- `--lr RATE`: Learning rate (default: 1e-3)
- `--hidden DIMS`: Hidden layer dimensions, comma-separated (default: 256,256,128)
- `--path_horizon N`: Future path horizon in timesteps (default: 50)
- `--checkpoint_dir D`: Directory for model checkpoints (default: ./checkpoints)
- `--no_gpu`: Disable GPU training

**Examples:**
```bash
# Basic training
julia --project train_policy.jl ./data_large

# Custom training
julia --project train_policy.jl ./data_large --epochs 200 --batch_size 512 --lr 5e-4
```

### 7. `run_train_policy.sbatch`
SLURM script for GPU training on the cluster.

**Usage:**
```bash
# Basic submission
sbatch run_train_policy.sbatch ./data_large

# With custom options
sbatch run_train_policy.sbatch ./data_large --epochs 200 --batch_size 512
```

**Note:** Update `--account` and `--partition` in the script for your cluster.

### 8. `eval_policy.jl`
Evaluates a trained policy in closed-loop simulation.

**Usage:**
```bash
julia --project -i eval_policy.jl ./checkpoints/best_model.jld2
```

**Path types for evaluation:**
- `:circle` - Circular path (default)
- `:figure8` - Figure-8 pattern
- `:straight` - Straight line forward
- `:zigzag` - Zigzag pattern

## Complete Training Pipeline

```bash
# 1. Generate dataset (parallel on cluster)
sbatch --array=1-20 run_array_job.sbatch 5000 ./data_large 42

# 2. Wait for completion, then merge
julia --project merge_dataset.jl ./data_large

# 3. Train policy on GPU
sbatch run_train_policy.sbatch ./data_large --epochs 100

# 4. Evaluate trained policy
julia --project -i eval_policy.jl ./checkpoints/best_model.jld2
```

## Policy Architecture Details

The path-following policy learns:

```
Input = [state (36D), relative_future_path (15D)]
        ↓
      Dense(51 → 256, ReLU)
      LayerNorm(256)
        ↓
      Dense(256 → 256, ReLU)
      LayerNorm(256)
        ↓
      Dense(256 → 128, ReLU)
      LayerNorm(128)
        ↓
      Dense(128 → 12)
        ↓
Output = joint_torques (12D)
```

**Key features:**
- **Relative path encoding**: Future waypoints are encoded relative to current position
- **Path subsampling**: Looking ahead 50 timesteps, sampled every 10 steps = 5 waypoints
- **LayerNorm**: Better training stability than BatchNorm for RL-style tasks
- **Normalized I/O**: Input/output normalization for stable gradients
