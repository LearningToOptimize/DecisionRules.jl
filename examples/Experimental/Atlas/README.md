# Atlas balancing

An experimental robotics case for DecisionRules.jl: the Boston Dynamics Atlas
humanoid must hold a reference posture while random pushes perturb it at every
stage. The rigid-body dynamics (RigidBodyDynamics.jl, from the URDF in `urdf/`)
enter each stage problem as nonlinear equality constraints discretized with step
`h`, so every stage is a small nonlinear program solved by Ipopt and
differentiated with DiffOpt. The problem is built the same way as the Goddard
rocket example (`examples/rocket_control`): a policy proposes the next state, the
stage problem projects that target onto the dynamics, and the multiplier of the
target constraint is the gradient signal.

The case is experimental: it exercises the nonlinear-dynamics path of the package
and the three training formulations side by side, but it is not part of the
published results.

## Scripts

| Script | Description |
|--------|-------------|
| `atlas_utils.jl` | Atlas model and dynamics from the URDF (`Atlas`, state layout, reference posture) |
| `rigidbodyutils.jl` | Small overloads that let RigidBodyDynamics evaluate on JuMP nonlinear expressions |
| `atlas_balancing.jl` | The starting point: a plain MPC balancing controller in JuMP + Ipopt, no learning |
| `build_atlas_problem.jl` | `build_atlas_subproblems`: the stochastic multi-stage problem (one subproblem per stage, perturbation parameters, target constraints) used by every training script |
| `test_atlas_problem.jl` | Checks the problem before training: open-loop simulation, one subproblem solve, a rollout with a dummy policy |
| `train_dr_atlas.jl` | Trains a policy on the stage-wise decomposition (`train_multistage`) |
| `train_dr_atlas_det_eq.jl` | Trains on the deterministic equivalent: one large problem over the whole horizon |
| `train_dr_atlas_multipleshooting.jl` | Trains on windows of stages (`train_multiple_shooting`) |
| `visualize_atlas_policy.jl`, `visualize_atlas_policy_multipleshooting.jl` | Load a trained policy, roll it out and animate it in MeshCat, with state and perturbation plots |
| `evaluate_transient_atlas_policy.jl` | One perturbation applied only at the first stage, one rollout per SLURM array task; results in `transient_eval_results/<policy>/` |
| `plot_transient_eval_results.jl` | Perturbation size against time to return to equilibrium, and the state deviation over time, from those results |
| `mwe_get_next_state_failure.jl` | Gradient check of one subproblem: the reverse-mode product returned by DiffOpt against finite differences, with respect to the incoming state and to the target |
| `test.jl` | RigidBodyDynamics sanity check on a double pendulum |

## Running

```bash
cd examples/Experimental/Atlas
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test_atlas_problem.jl          # check the problem first
julia --project=. train_dr_atlas.jl              # or _det_eq / _multipleshooting
julia --project=. visualize_atlas_policy.jl      # set the model path at the top
```

The training scripts set the horizon, step, perturbation scale and network size
in a parameter block at the top of the file. Several scripts select an HSL
linear solver for Ipopt (`"linear_solver" => "ma27"`, from `HSL_jll`), which needs
a licence; set it to `"mumps"` to run without one.
