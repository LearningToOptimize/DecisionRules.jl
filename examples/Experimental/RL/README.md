# Reinforcement Learning Baselines

This directory benchmarks TS-DDR against standard deep reinforcement
learning algorithms (REINFORCE, PPO, DDPG, TD3, SAC) on the Bolivia
hydrothermal dispatching problem, using [Crux.jl](https://github.com/ancorso/Crux.jl)
for RL training.

## Problem overview

The Bolivia LTHD problem is cast as a POMDP: the agent observes rainfall
(partial observability), selects reservoir-level actions, and receives
negative dispatch cost as reward.  The environment wraps DecisionRules
stage-wise subproblems, so each step solves a full OPF.

## Scripts

| Script | Description |
|--------|-------------|
| `hydropowermodels_rl.jl` | Train RL agents (REINFORCE, PPO, DDPG, TD3, SAC) from scratch on Bolivia LTHD and plot learning curves |
| `hydro_pre_trained.jl` | Fine-tune RL agents from a TS-DDR pre-trained policy (warm-start the actor from a saved JLD2 model) |
| `test.jl` | Quick sanity check of the MDP construction |

## Benchmark results

Pre-generated learning curves are saved as:
- `hydro_benchmark.pdf` — RL from scratch
- `hydro_benchmark_warm.pdf` — RL fine-tuned from TS-DDR

## Running

Requires Distributed workers (uses `@everywhere`).  Launch with:

```bash
julia -p 4 examples/RL/hydropowermodels_rl.jl
```

## Dependencies

See `Project.toml`.  Key additional packages beyond DecisionRules: Crux,
POMDPs, POMDPTools, CommonRLInterface, Distributions.
