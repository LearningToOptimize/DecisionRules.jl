# Quadruped in Dojo

Experiments with a quadruped simulated by the Dojo.jl physics engine, used to
compare a differentiable-simulation policy with reinforcement learning on a
path-following task. These scripts do not use the DecisionRules.jl training
loop; they are kept as baselines and as a reference for wrapping a
non-JuMP simulator.

## Scripts

| Script | Description |
|--------|-------------|
| `train_policy_dojo.jl` | Trains a Flux policy that maps the state and the next waypoint to a control, by differentiating through `dojo_step` (a Zygote `rrule` around the Dojo step, with a finite-difference fallback) |
| `sampling.jl` | Random-search tuning of a parametric gait controller that follows the waypoints, without gradients |
| `quadruped_env.jl` | The quadruped environment: reset, step, observation and reward |
| `quadruped_dqn.jl` | DQN on that environment with Crux.jl through the POMDPs.jl interface |
| `visualize_actions.jl` | Plays back a recorded control sequence |

## Running

```bash
cd examples/Experimental/Dojo
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. train_policy_dojo.jl
```

Dojo simulates on the CPU only; a GPU, when present, is used for network
training only.
