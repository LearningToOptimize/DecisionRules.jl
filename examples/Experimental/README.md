# Experimental examples

Cases that exercise DecisionRules.jl beyond the published case studies. Each
directory has its own `Project.toml` and README; run each from its own directory
with `--project=.`.

| Directory | What it is |
|-----------|------------|
| `Atlas/` | Humanoid balancing under random pushes: nonlinear rigid-body dynamics in every stage, trained stage-wise, as a deterministic equivalent and with multiple shooting |
| `Quadruped/` | Quadruped trajectory optimization with multiple shooting, and dataset generation for imitation learning |
| `Dojo/` | A quadruped in the Dojo.jl simulator: a policy trained through the differentiable simulator, and RL baselines |
| `Go2/` | Unitree Go2 model loading and a differentiable open-loop simulation, as a starting point |
| `RL/` | Deep RL baselines (REINFORCE, PPO, DDPG, TD3, SAC) on the hydrothermal case, from scratch and warm-started from a TS-DDR policy |
