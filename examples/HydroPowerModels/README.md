# HydroPowerModels — Long-Term Hydrothermal Dispatching (Bolivia LTHD)

This directory contains the primary application from the paper: training
Two-Stage Deep Decision Rules (TS-DDR) for the Bolivia Long-Term
Hydrothermal Dispatching problem with 10 hydro units, 96 monthly stages,
and AC/SOC/DC power-flow formulations.

## Problem overview

The Bolivia LTHD problem couples hydro reservoir dynamics (water balance)
with a power network dispatch (OPF) at each stage.  Stochastic inflows
drive reservoir levels; the decision rule maps (inflow history, current
state) to reservoir-level targets, and an NLP optimizer dispatches
generation to meet those targets at minimum cost.

## Training scripts

| Script | Decomposition | Reference |
|--------|--------------|-----------|
| `train_dr_hydropowermodels.jl` | Deterministic equivalent (GPU-enabled) | Extension §1 |
| `train_dr_hydropowermodels_subproblems.jl` | Stage-wise (single shooting) | Extension §2 |
| `train_dr_hydropowermodels_multipleshooting.jl` | Windowed (multiple shooting) | Extension §3 |

All three share the same data loader (`load_hydropowermodels.jl`) and
policy architecture (LSTM encoder + state-conditioned dense layers).
Training logs to Weights & Biases and saves the best model to JLD2.

### GPU training

`train_dr_hydropowermodels.jl` auto-detects CUDA and switches to
MadNLP+CUDSS on GPU when available.  Submit via:

```bash
cd examples/HydroPowerModels
mkdir -p logs
sbatch run_train_deteq_gpu.sbatch
```

### Penalty schedule

All training scripts support `:default_annealed` penalty schedules that
gradually increase target-violation penalties during training, improving
convergence on the nonconvex AC formulation.

### Rollout metrics

For deterministic-equivalent training, `metrics/loss` is computed on the same
target-state history produced by the policy. The matching held-out metric is
`metrics/rollout_objective_no_deficit`, which now uses `RolloutEvaluation(...;
policy_state=:target)` in `train_dr_hydropowermodels.jl`.

The same script also logs
`metrics/rollout_realized_objective_no_deficit` with `policy_state=:realized`.
That is the closed-loop deployment diagnostic: each stage passes the optimizer's
realized reservoir state back to the policy. It can be harder than the target-state
metric, especially while the policy is trained through the deterministic equivalent.

All rollout objective metrics exclude the target-slack/deficit penalty term. Track
the paired target-violation share and `metrics/target_penalty_multiplier` to see
whether a low operational objective is coming from feasible targets or from the
policy relying on slack.

## Evaluation and baselines

| Script | Purpose |
|--------|---------|
| `eval_jump_de.jl` | Solve the DE with a constant policy and save a reference solution (JLD2) for cross-validation with ExaModels |
| `test_dr_hydropowermodels.jl` | Load a trained model and produce volume/generation/cost comparison plots and CSVs |
| `check_consistent_state_paths.jl` | Verify that stage-wise, deterministic equivalent, and multiple-shooting decompositions produce identical state trajectories under the same policy and inflows |

## SDDP baselines

These scripts require [HydroPowerModels.jl](https://github.com/andrewrosemberg/HydroPowerModels.jl),
Gurobi, and Mosek licenses.

| Script | Description |
|--------|-------------|
| `run_sddp.jl` | Train SDDP with a consistent convex (SOCWRConic) formulation |
| `run_sddp_inconsistent.jl` | Train SDDP with SOCWRConic backward pass and ACP forward pass |
| `simulate_sddp_policy.jl` | Simulate a pre-trained SDDP policy under ACP and produce comparison plots |

## Learning-to-Optimize (L2O) pipeline

| Script | Description |
|--------|-------------|
| `gen_inputs_l2O_hydropowermodels.jl` | Generate input datasets for the L2O supervised pipeline (requires [L2O.jl](https://github.com/andrewrosemberg/L2O.jl)) |
| `train_dr_l2O_supervised.jl` | Supervised pre-training of a decision rule from L2O-generated optimal solutions |

## Subproblem export (generating `.mof.json` files)

The training pipeline (`load_hydropowermodels.jl`) reads pre-exported `.mof.json`
subproblem templates rather than depending on HydroPowerModels.jl at training time.
These files already ship with the repository:

```
bolivia/ACPPowerModel.mof.json
bolivia/SOCWRConicPowerModel.mof.json
bolivia/DCPPowerModel.mof.json
case3/ACPPowerModel.mof.json
```

To regenerate them (e.g. after updating HydroPowerModels data or adding a new
formulation), use `export_subproblem_mof.jl`:

```bash
julia export_subproblem_mof.jl bolivia ACPPowerModel
julia export_subproblem_mof.jl bolivia SOCWRConicPowerModel
```

This builds the full SDDP model via HydroPowerModels.jl, extracts one stage's
subproblem from the policy graph, removes the unnamed slack variable that
HydroPowerModels adds, and writes a clean JuMP `.mof.json` to disk.  Requires
HydroPowerModels.jl and a solver (Mosek by default).

## Data

- `bolivia/` — Bolivia case: `hydro.json` (10 hydro units), `inflows.csv` (historical scenarios), `ACPPowerModel.mof.json` / `SOCWRConicPowerModel.mof.json` / `DCPPowerModel.mof.json` (subproblem templates)
- `case3/` — Small 3-bus test case for development

## Dependencies

See `Project.toml` in this directory.  Key packages: DecisionRules, DiffOpt,
Ipopt+HSL, MadNLP+MadNLPGPU+CUDA (GPU), Flux, JuMP, Wandb.
