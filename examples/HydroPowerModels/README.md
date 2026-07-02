# HydroPowerModels — Long-Term Hydrothermal Dispatching (Bolivia LTHD)

This directory contains the primary application from the paper: training
Two-Stage Deep Decision Rules (TS-DDR) for the Bolivia Long-Term
Hydrothermal Dispatching problem with 11 hydro units, 96 monthly stages,
and AC power-flow constraints.

## Problem overview

The Bolivia LTHD problem couples hydro reservoir dynamics (water balance)
with a power network dispatch (OPF) at each stage.  Stochastic inflows
drive reservoir levels; the decision rule maps (inflow history, current
state) to reservoir-level targets, and an NLP optimizer dispatches
generation to meet those targets at minimum cost.

## Quick start: strict mode (recommended)

Strict mode eliminates the target-penalty hyperparameter by enforcing hard
equality constraints between targets and realized states. A
`HydroReachablePolicy` guarantees that every target is physically achievable
from the state used as policy input.

```bash
# Train with stage-wise decomposition, strict mode, 126 stages
julia --project -t auto train_dr_hydropowermodels_subproblems.jl \
    --strict --stages 126

# Evaluate on 100 held-out scenarios, 96 stages
julia --project -t auto eval_strict_rollout.jl \
    bolivia/ACPPowerModel/models/<checkpoint>.jld2
```

The training script logs to Weights & Biases and saves model checkpoints.
The evaluation script writes per-scenario costs, mean reservoir volumes,
and mean thermal generation to CSV files for comparison against SDDP.

## Strict Reachability Logic

Strict mode is normally safe only when the policy can see the current state:

- Stage-wise subproblems are closed-loop because the next policy call receives
  the realized state from the previous solve.
- Embedded deterministic equivalents are closed-loop because the policy is
  evaluated inside the NLP against realized state variables.
- A generic regular deterministic equivalent is open-loop after the initial
  state, because all targets are computed before the multi-stage solve.

The hydro reachable policy gives a special regular-DE exception used in the Exa
companion package. If targets are rolled out as
`target[0] = initial_state` and `target[t] = policy(inflow[t], target[t-1])`,
and the policy maps into the one-stage reachable set from its input state, then
all targets are feasible by induction. Strict equalities then force realized
states to equal the reachable target trajectory.

## Training scripts

### Strict subproblems (no penalty tuning)

| Script | Description |
|--------|-------------|
| `train_dr_hydropowermodels_subproblems.jl` | Stage-wise decomposition with `--strict` flag; uses `HydroReachablePolicy` for feasibility-guaranteed targets |
| `train_dr_hydropowermodels_strict.jl` | Dedicated strict stage-wise training entrypoint with `DR_ENCODER_LAYERS` and `DR_HEAD_LAYERS` environment knobs |

### Non-strict formulations (require penalty tuning)

| Script | Decomposition |
|--------|--------------|
| `train_dr_hydropowermodels.jl` | Deterministic equivalent (full-horizon coupled NLP) |
| `train_dr_hydropowermodels_subproblems.jl` | Stage-wise (single shooting) |
| `train_dr_hydropowermodels_multipleshooting.jl` | Windowed (multiple shooting, `W=12`) |
| `train_ldr_hydropowermodels.jl` | Linear decision rules (identity activation) |

All training scripts share the data loader (`load_hydropowermodels.jl`),
log to Weights & Biases, and save the best model to JLD2.

### Key files

| File | Description |
|------|-------------|
| `load_hydropowermodels.jl` | Builds JuMP stage subproblems from MOF + hydro JSON + inflow CSV; supports `strict=true` for penalty-free targets |
| `hydro_reachable_policy.jl` | `HydroReachablePolicy` — bounds LSTM output to the one-stage reachable set via sigmoid; `load_policy_weights!` for checkpoint loading |

### Policy architecture knobs

`HydroReachablePolicy` and `state_conditioned_policy` separate temporal memory
from state conditioning:

- `layers` / `DR_ENCODER_LAYERS`: recurrent LSTM layers over inflows only.
- `combiner_layers` / `DR_HEAD_LAYERS`: optional nonrecurrent feed-forward
  hidden layers over `[encoded_inflow; reservoir_state]`.

This lets TS-DDR depart from linear decision rules in the state-to-target map
without adding recurrence over the reservoir state.

### GPU training

`train_dr_hydropowermodels.jl` auto-detects CUDA and switches to
MadNLP+CUDSS on GPU when available.  Submit via:

```bash
cd examples/HydroPowerModels
mkdir -p logs
sbatch run_train_deteq_gpu.sbatch
```

For GPU-accelerated training using ExaModels (recommended for large NLPs),
see the companion package
[DecisionRulesExa.jl](https://github.com/LearningToOptimize/DecisionRulesExa.jl).

## Evaluation

| Script | Purpose |
|--------|---------|
| `eval_strict_rollout.jl` | 100-scenario stage-wise rollout of a strict-mode policy; writes costs, mean volumes, and mean thermal generation to CSVs |
| `evaluate_hydro_policies.jl` | Auto-discovers all saved checkpoints and evaluates them on a common scenario set; writes `eval_costs.csv` |
| `eval_jump_de.jl` | Solve the DE with a constant policy and save a reference solution (JLD2) for cross-validation with ExaModels |
| `check_consistent_state_paths.jl` | Verify that stage-wise, DE, and multiple-shooting decompositions produce identical state trajectories |

## File Inventory

| Path | Purpose |
|---|---|
| `Project.toml` | Hydro example environment |
| `LocalPreferences.toml` | Local solver/GPU preferences when present |
| `README.md` | This guide |
| `load_hydropowermodels.jl` | JuMP/DiffOpt stage-problem builder with optional strict targets |
| `hydro_reachable_policy.jl` | Reachability-guaranteed hydro policy and checkpoint loading helpers |
| `train_dr_hydropowermodels.jl` | Regular deterministic-equivalent TS-DDR training |
| `train_dr_hydropowermodels_subproblems.jl` | Stage-wise TS-DDR training; supports strict mode |
| `train_dr_hydropowermodels_strict.jl` | Dedicated strict stage-wise training script |
| `train_dr_hydropowermodels_multipleshooting.jl` | Multiple-shooting TS-DDR training |
| `train_ldr_hydropowermodels.jl` | Linear decision-rule baseline |
| `train_dr_l2O_supervised.jl` | Supervised learning-to-optimize training utility |
| `eval_strict_rollout.jl` | Rollout evaluation for strict reachable policies |
| `evaluate_hydro_policies.jl` | Batch checkpoint evaluator |
| `eval_paired_tsddr.jl` | Paired TS-DDR scenario evaluation |
| `eval_jump_de.jl` | JuMP deterministic-equivalent reference evaluation |
| `compare_hydro_results.jl` | Result aggregation/comparison helper |
| `check_consistent_state_paths.jl` | State-path consistency diagnostic |
| `validate_sddp_vs_jump.jl` | Validation against SDDP/JuMP references |
| `test_strict_mode.jl` | Strict-mode and reachable-policy checks |
| `test_sampling_consistency.jl` | Scenario sampling consistency tests |
| `export_subproblem_mof.jl` | MOF template export utility |
| `gen_inputs_l2O_hydropowermodels.jl` | Dataset/input generation for supervised L2O experiments |
| `sddp/run_sddp.jl` | Consistent SDDP baseline training |
| `sddp/run_sddp_inconsistent.jl` | SOC backward / AC forward SDDP training |
| `sddp/simulate_sddp_policy.jl` | AC simulation of a trained SDDP policy |
| `sddp/eval_paired_sddp.jl` | Paired SDDP scenario evaluation |
| `sddp/extract_sddp_trajectories.jl` | Extract SDDP trajectory data for diagnostics |
| `bolivia/` | Bolivia case data, MOF templates, figures, and saved models/results |
| `case3/` | Small development case |

## SDDP baselines

The SDDP baseline uses an **inconsistent formulation**: SOC-WR relaxation
for the backward pass (cut generation) and ACP for the forward pass
(simulation).  Scripts are in `sddp/` with a dedicated Julia environment.

| Script | Description |
|--------|-------------|
| `sddp/run_sddp.jl` | Train SDDP with a consistent convex (SOCWRConic) formulation |
| `sddp/run_sddp_inconsistent.jl` | Train SDDP with SOCWRConic backward / ACP forward |
| `sddp/simulate_sddp_policy.jl` | Simulate a pre-trained SDDP policy under ACP (100 scenarios, 96 stages); writes costs, mean volumes, mean thermal generation to CSVs |

**SDDP 96-stage simulation cost**: 303 684 (mean, 100 scenarios, std 5 453).
**SDDP 126-stage lower bound**: 378 207 (SOC-WR relaxation — not beatable).

## Data

- `bolivia/` — Bolivia case: `hydro.json` (11 hydro units), `inflows.csv`
  (historical scenarios), `ACPPowerModel.mof.json` / `SOCWRConicPowerModel.mof.json` /
  `DCPPowerModel.mof.json` (subproblem templates), `PowerModels.json` (39 buses,
  55 branches, 34 generators)
- `case3/` — Small 3-bus test case for development

## Subproblem export

The training pipeline reads pre-exported `.mof.json` subproblem templates.
To regenerate (e.g., after updating data or adding a formulation):

```bash
julia export_subproblem_mof.jl bolivia ACPPowerModel
```

## Dependencies

See `Project.toml` in this directory.  Key packages: DecisionRules, DiffOpt,
Ipopt, Flux, JuMP, Wandb.
