# Rocket Control — Goddard Rocket with Stochastic Wind

This directory demonstrates TS-DDR on a continuous optimal-control problem:
maximizing the altitude of a vertically ascending rocket under stochastic
wind disturbances, subject to mass, velocity, and thrust constraints.

## Problem overview

The Goddard rocket problem is a classic optimal-control benchmark
(parameters from [COPS3](https://www.mcs.anl.gov/~more/cops/cops3.pdf)).
We discretize the dynamics with forward Euler (T=1000 steps) and add
multiplicative wind noise to the drag force.  The decision rule maps the
wind realization sequence to a thrust profile.

## Scripts

| Script | Description |
|--------|-------------|
| `build_rocket_problem.jl` | Build the JuMP deterministic-equivalent and stage-wise subproblems with DiffOpt |
| `train_dr_rocket.jl` | Train a TS-DDR policy (LSTM + Dense) on both DE and subproblem formulations, then plot height trajectories |
| `test_dr_rocket.jl` | Unit-test style script verifying that the DE and subproblem builds produce consistent results |
| `run_mpc_rocket.jl` | Model Predictive Control (MPC) baseline: re-solve a deterministic problem at each step as a rolling-horizon controller |
| `compare_results.jl` | Compare TS-DDR vs MPC trajectories and costs |

## Running

```julia
# From the repo root:
using Pkg; Pkg.activate("examples/rocket_control")
include("examples/rocket_control/train_dr_rocket.jl")
```

## Dependencies

See `Project.toml`.  Only requires DecisionRules, DiffOpt, Ipopt+HSL,
Flux, JuMP, and Plots for visualization.
