# Examples

This directory contains worked examples for DecisionRules.jl, covering the
applications from the paper ([arXiv:2405.14973](https://arxiv.org/abs/2405.14973))
and additional experiments.

## Directories

| Directory | Application | Paper section |
|-----------|------------|---------------|
| [`HydroPowerModels/`](HydroPowerModels/) | Bolivia Long-Term Hydrothermal Dispatching (10 hydro units, AC/SOC/DC OPF, 96 stages). Trains TS-DDR (LSTM) and TS-LDR (linear) policies. | §4, Extension §1–§4 |
| [`inventory_control/`](inventory_control/) | Stochastic lot-sizing with fixed ordering costs (relaxed LP and integer MIP). Demonstrates score-function (REINFORCE) gradient mixing for integer variables. | §3 |
| [`rocket_control/`](rocket_control/) | Goddard rocket altitude maximization with stochastic wind | §3 |
| [`RL/`](RL/) | Reinforcement learning baselines (REINFORCE, PPO, DDPG, TD3, SAC) on Bolivia LTHD | Beyond paper |
| `Experimental/` | Work-in-progress experiments (not documented) | — |

## Utility scripts

| Script | Description |
|--------|-------------|
| `slurm.jl` | SLURM launcher: starts Distributed workers via `ClusterManagers` and includes a target script |
| `solve_dataset.jl` | Distributed batch solver for L2O dataset generation (uses [L2O.jl](https://github.com/andrewrosemberg/L2O.jl)) |

## Quick start

Each subdirectory has its own `Project.toml`.  Activate and instantiate
before running:

```julia
using Pkg
Pkg.activate("examples/HydroPowerModels")
Pkg.instantiate()
include("examples/HydroPowerModels/train_dr_hydropowermodels_subproblems.jl")
```

For GPU-accelerated training on SLURM, see the `.sbatch` files in each
subdirectory.

## Training methods compared

All three decomposition strategies from the paper can be trained on the
same problem:

1. **Deterministic Equivalent** — single coupled NLP over all stages (Extension §1)
2. **Stage-wise / Single Shooting** — solve one subproblem per stage, backpropagate through the chain (Extension §2)
3. **Windowed / Multiple Shooting** — partition stages into windows, parallelize window solves (Extension §3)

The HydroPowerModels directory contains a training script for each strategy,
a TS-LDR training script (linear policy baseline), and an evaluation script
(`evaluate_hydro_policies.jl`) that runs all trained policies on a common
out-of-sample scenario set.
