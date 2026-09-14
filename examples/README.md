# Examples

This directory contains worked examples for DecisionRules.jl, covering the
applications from the paper ([arXiv:2405.14973](https://arxiv.org/abs/2405.14973))
and additional experiments.

## Directories

| Directory | Application | Paper section |
|-----------|------------|---------------|
| [`inventory_control/`](inventory_control/) | Stochastic lot-sizing with fixed ordering costs (relaxed LP and integer MIP). Demonstrates score-function (REINFORCE) gradient mixing for integer variables. | §3 |
| [`rocket_control/`](rocket_control/) | Goddard rocket altitude maximization with stochastic wind | §3 |
| `Experimental/` | Work-in-progress experiments (not documented) | — |

## Utility scripts

| Script | Description |
|--------|-------------|
| `slurm.jl` | SLURM launcher: starts Distributed workers via `ClusterManagers` and includes a target script |

## Quick start

Each subdirectory has its own `Project.toml`.  Activate and instantiate
before running:

```julia
using Pkg
Pkg.activate("examples/inventory_control")
Pkg.instantiate()
include("examples/inventory_control/train_dr_inventory.jl")
```

For GPU-accelerated training on SLURM, see the `.sbatch` files in each
subdirectory.
