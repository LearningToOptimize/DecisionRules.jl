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
# Train with the dedicated strict stage-wise entrypoint.
# Configure horizon with DR_NUM_STAGES, for example:
DR_NUM_STAGES=126 julia --project -t auto train_dr_hydropowermodels_strict.jl

# Evaluate on 100 held-out scenarios, 96 stages
julia --project -t auto eval_strict_rollout.jl \
    bolivia/ACPPowerModel/models/<checkpoint>.jld2
```

The training script logs to Weights & Biases and saves model checkpoints.
The evaluation script writes per-scenario costs, mean reservoir volumes,
and mean thermal generation to CSV files for comparison against SDDP.

## Reproducing the paper results (strict, Bolivia ACP)

Both stages below are ordinary configurations of the same entrypoint — no
side scripts. All knobs are environment variables documented in the header of
`train_dr_hydropowermodels_strict.jl`; defaults reproduce stage 1 exactly.

**Stage 1 — train from scratch** (h126 training horizon, r96 rollout
evaluation, LSTM(128,128) encoder, linear sigmoid head):

```bash
DR_NUM_STAGES=126 DR_NUM_ROLLOUT_STAGES=96 \
julia --project -t auto train_dr_hydropowermodels_strict.jl
```

**Stage 2 — fine-tune the best stage-1 checkpoint** (variance-reduced
gradients, warmup + cosine-decayed learning rate, checkpoints selected by the
held-out 96-stage rollout objective rather than the noisy training loss):

```bash
DR_PRETRAINED_MODEL=bolivia/ACPPowerModel/models/<stage1-best>.jld2 \
DR_NUM_STAGES=126 DR_NUM_ROLLOUT_STAGES=96 DR_NUM_EPOCHS=15 \
DR_NUM_TRAIN_PER_BATCH=16 DR_LR=1e-4 DR_LR_FINAL=1e-5 DR_LR_WARMUP=50 \
DR_SAVE_METRIC=rollout DR_NUM_EVAL_SCENARIOS=24 \
julia --project -t auto train_dr_hydropowermodels_strict.jl
```

Why stage 2 is configured this way: near convergence the single-sample
training loss (per-scenario std ≈ 5–7k) cannot rank checkpoints that differ by
a few hundred cost units, so selection switches to the deployment metric
(`DR_SAVE_METRIC=rollout`) over 24 fixed scenarios; a fresh Adam takes
full-size steps while its second-moment estimates are still zero, so the
warmup protects the loaded optimum; and the decayed learning rate lets the
policy re-anneal instead of random-walking at a fixed step size.

**Paired evaluation against SDDP** — both methods realize the same 500
inflow trajectories, generated from a documented seed
(`paired_scenario_indices` in `load_hydropowermodels.jl`, StableRNG, so the
protocol is reproducible from code alone):

```bash
julia --project -t auto eval_paired_tsddr.jl \
    bolivia/ACPPowerModel/models/<best>.jld2
cd sddp && julia --project -t auto eval_paired_sddp.jl
```

## Files in this folder

Every script, what it does, and when to use it. All commands run from this
folder with `julia --project`; configuration is via the env vars documented
in each script's header.

### Problem construction (used by everything else)

| File | Purpose |
|------|---------|
| `load_hydropowermodels.jl` | Case loader: builds the stage subproblems from the MOF file + hydro data, reads inflows (tiled cyclically beyond the 47-month record), returns hydro metadata; `strict=true` for hard target equalities. Also defines `paired_scenario_indices`, the seeded paired-evaluation protocol. |
| `hydro_reachable_policy.jl` | `HydroReachablePolicy`: reachable-set targets (sigmoid scaled to physics bounds), cascade-aware clamping, explicit LSTM state threading, optional stage-context inputs; plus checkpoint loaders. |

### Training

| File | Purpose |
|------|---------|
| `train_dr_hydropowermodels_strict.jl` | **The strict trainer** (stage-wise, Ipopt). Header documents the two-stage recipe that produces the paper policies (stage-1 from scratch, stage-2 fine-tune) and all env knobs incl. `DR_CONTEXT`. |
| `train_dr_hydropowermodels.jl` | Non-strict deterministic-equivalent training (penalty formulation). |
| `train_dr_hydropowermodels_subproblems.jl` | Non-strict stage-wise (single-shooting) training. |
| `train_dr_hydropowermodels_multipleshooting.jl` | Non-strict multiple-shooting training. |
| `train_ldr_hydropowermodels.jl` | TS-LDR (linear decision rule) baseline training. |
| `train_dr_l2O_supervised.jl` | Supervised learning-to-optimize baseline (needs `gen_inputs_l2O_hydropowermodels.jl` outputs). |

### Paired evaluation (produces the results tables)

| File | Purpose |
|------|---------|
| `eval_paired_tsddr.jl` | Paired rollout evaluation of a checkpoint (`ARGS[1]`) on the seeded 500-scenario protocol; writes `paired_costs[_<policy-tag>].csv` and trajectory CSVs. |
| `sddp/eval_paired_sddp.jl` | Paired SDDP simulation (`SDDP.Historical`) on the identical seeded scenarios; merges its column into the costs CSV. |
| `dump_paired_policy_reference.jl` | Dumps policy outputs + inflow values (JLD2) consumed by DecisionRulesExa.jl's cross-package equivalence evaluation. |

### Other evaluation & validation

| File | Purpose |
|------|---------|
| `eval_strict_rollout.jl` | Standalone 96-stage rollout evaluation of a strict checkpoint (unpaired scenario set). |
| `evaluate_hydro_policies.jl` | Batch evaluation of pre-trained TS-DDR/TS-LDR checkpoints on a fixed scenario set. |
| `eval_jump_de.jl` | Evaluates a policy through the full-horizon JuMP deterministic equivalent. |
| `check_consistent_state_paths.jl` | Consistency check: state paths agree across training formulations. |
| `test_strict_mode.jl` | Strict-mode construction checks (hard equalities, reachable targets). |
| `test_sampling_consistency.jl` | Verifies scenario sampling consistency across code paths. |
| `validate_sddp_vs_jump.jl` | Validates that the JuMP subproblems match SDDP's formulation (the audit that keeps the SDDP comparison fair). |

### Figures (regenerate the docs assets)

| File | Purpose |
|------|---------|
| `plot_hydro_strict_convergence.jl` | Wall-clock training-convergence figure (parses `sddp/SDDP.log`, pulls W&B histories). |
| `plot_hydro_paired_distributions.jl` | Paired cost-distribution figure (absolute densities + paired differences vs SDDP). |
| `compare_hydro_results.jl` | W&B comparison plots across training formulations. |

### SDDP baseline

| File | Purpose |
|------|---------|
| `sddp/run_sddp.jl` | Trains the SDDP baseline (SOC-WR cuts, ACP forward pass); writes cuts JSON + `SDDP.log`. |
| `sddp/run_sddp_inconsistent.jl` | SDDP with inconsistent backward/forward formulations (the published baseline configuration). |
| `sddp/simulate_sddp_policy.jl` | Simulates a trained SDDP policy from its cuts file. |
| `sddp/extract_sddp_trajectories.jl` | Extracts state/generation trajectories from SDDP simulations. |

### Utilities

| File | Purpose |
|------|---------|
| `export_subproblem_mof.jl` | Exports the single-stage OPF as `.mof.json` (how `bolivia/*.mof.json` was produced). |
| `gen_inputs_l2O_hydropowermodels.jl` | Generates supervised-learning inputs for the L2O baseline. |

### Data (`bolivia/`)

`PowerModels.json` + `hydro.json` (network and hydro data), `inflows.csv`
(47 monthly joint inflow scenarios), `ACPPowerModel.mof.json` etc. (exported
stage subproblems), `ACPPowerModel/models/` (trained checkpoints) and
`ACPPowerModel/results/` (evaluation outputs — data, not tracked).

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
| `train_dr_hydropowermodels_strict.jl` | Dedicated strict stage-wise training entrypoint with `DR_NUM_STAGES`, `DR_ENCODER_LAYERS`, and `DR_HEAD_LAYERS` environment knobs |

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
MadNLP+CUDSS on GPU when available. From this example environment, run:

```bash
cd examples/HydroPowerModels
julia --project train_dr_hydropowermodels.jl
```

If you want to submit through Slurm, use a site-local batch script that loads
Julia and requests the GPU resources required by MadNLP+CUDSS.

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
