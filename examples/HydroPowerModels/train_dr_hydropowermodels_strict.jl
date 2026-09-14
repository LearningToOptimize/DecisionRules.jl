# Train HydroPowerModels using strict subproblems with reachable policy
#
# Strict mode removes the deficit slack variables from the target constraint:
#   reservoir_out[r] = target[r]     (hard equality, no penalty)
# The dual λ_r is the clean shadow price ∂Q/∂target — pure economic signal.
#
# This requires HydroReachablePolicy, which guarantees every target is within
# the one-stage reachable set via sigmoid-bounded outputs scaled to physical
# reservoir limits. Stage-wise strict mode is closed-loop: each policy call sees
# the realized state from the previous strict stage solve.
#
# Usage:
#   julia --project -t auto train_dr_hydropowermodels_strict.jl
#
# Environment overrides:
#   DR_NUM_STAGES=126          number of training stages
#   DR_NUM_ROLLOUT_STAGES=96   number of rollout evaluation stages (default: 96,
#                              matching the Exa strict recipe: train 126 / roll out 96)
#   DR_LOAD_SCALER=1.0         demand scaler applied to demand.csv active demand and
#                              the nominal reactive demand (1.0 = real seasonal
#                              demand, no 0.6 scaler; parity with DecisionRulesExa)
#   DR_DEFICIT_COST=1e5        load-shedding cost per pu (paper recipe 1e5)
#   DR_NUM_EPOCHS=80           number of epochs
#   DR_NUM_BATCHES=100         gradient steps per epoch (total = epochs * batches)
#   DR_ENCODER_LAYERS=128,128  recurrent inflow encoder sizes
#   DR_HEAD_LAYERS=            nonrecurrent state-conditioned target head sizes
#   DR_GRAD_CLIP=0             gradient clipping (0 = disabled)
#   DR_NUM_TRAIN_PER_BATCH=1   sampled trajectories per gradient step (variance reduction)
#   DR_PRETRAINED_MODEL=path   warmstart from a StateConditionedPolicy checkpoint
#   DR_CONTEXT=                optional known context prepended to the policy input:
#                              ""/"none" = off, "phase" = seasonal sin/cos,
#                              "phase+progress" = seasonal sin/cos plus t/T
#   DR_NUM_EVAL_SCENARIOS=4    fixed held-out scenarios for rollout evaluation
#   DR_EVAL_EVERY=25           rollout-evaluate every this many batches
#   DR_SAVE_METRIC=training    checkpoint-selection metric:
#                              "training" — per-batch training loss (historical
#                              behavior; noisy for small DR_NUM_TRAIN_PER_BATCH)
#                              "rollout" — mean deficit-free objective of the
#                              fixed held-out rollout evaluation (the metric
#                              policies are ultimately judged on; evaluated
#                              every DR_EVAL_EVERY batches)
#   DR_LR=0.001                Adam learning rate
#   DR_LR_FINAL=DR_LR          final learning rate of a cosine decay across the
#                              full run (equal to DR_LR → constant, historical)
#   DR_LR_WARMUP=0             linear warmup iterations from DR_LR/100 to DR_LR.
#                              Protects a warmstarted policy from the initial
#                              full-size Adam steps taken while its second-moment
#                              estimates are still zero.
#
# Reproducible recipes (also listed in this folder's README):
#   From scratch (paper configuration — all defaults):
#     julia --project -t auto train_dr_hydropowermodels_strict.jl
#   Fine-tune from a converged checkpoint (variance-reduced, decayed LR,
#   rollout-selected checkpoints):
#     DR_PRETRAINED_MODEL=<best.jld2> DR_NUM_TRAIN_PER_BATCH=16 \
#     DR_LR=1e-4 DR_LR_FINAL=1e-5 DR_LR_WARMUP=50 \
#     DR_SAVE_METRIC=rollout DR_NUM_EVAL_SCENARIOS=24 DR_NUM_EPOCHS=15 \
#     julia --project -t auto train_dr_hydropowermodels_strict.jl
using DecisionRules
using Statistics
using Random
using Flux

using Ipopt
using Wandb, Dates, Logging
using JLD2
using DiffOpt

HydroPowerModels_dir = dirname(@__FILE__)
include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))
include(joinpath(HydroPowerModels_dir, "hydro_reachable_policy.jl"))

# ── Parameters ───────────────────────────────────────────────────────────────

case_name = "bolivia"
formulation = "ACPPowerModel"
num_stages = parse(Int, get(ENV, "DR_NUM_STAGES", "126"))
# Default rollout horizon 96 (train 126 / evaluate 96) — the paired-evaluation
# protocol shared with the SDDP baseline and DecisionRulesExa's strict trainer.
num_rollout_stages = parse(Int, get(ENV, "DR_NUM_ROLLOUT_STAGES", "96"))
# Demand scaler: 1.0 = real seasonal demand.csv, no historical 0.6 down-scaling.
load_scaler = parse(Float64, get(ENV, "DR_LOAD_SCALER", "1.0"))
# Load-shedding cost per pu; 1e5 is the paper recipe shared with DecisionRulesExa.
deficit_cost = parse(Float64, get(ENV, "DR_DEFICIT_COST", "1e5"))
model_dir = joinpath(HydroPowerModels_dir, case_name, formulation, "models")
mkpath(model_dir)
formulation_file = formulation * ".mof.json"
num_epochs = parse(Int, get(ENV, "DR_NUM_EPOCHS", "80"))
# Gradient steps per epoch. Configurable so the documented smoke test is
# genuinely small: the total update budget is num_epochs * num_batches, and with
# this fixed at 100 a "2-epoch" run was still 200 updates.
num_batches = parse(Int, get(ENV, "DR_NUM_BATCHES", "100"))
# Trajectories sampled per gradient step; >1 averages the per-sample dual
# gradients, reducing estimator variance at proportionally higher solve cost.
_num_train_per_batch = parse(Int, get(ENV, "DR_NUM_TRAIN_PER_BATCH", "1"))
"""
    parse_layers(s::AbstractString) -> Vector{Int64}

Parse comma-separated policy architecture settings from environment variables.

`DR_ENCODER_LAYERS` controls the recurrent inflow encoder. `DR_HEAD_LAYERS`
controls optional hidden layers in the nonrecurrent state-conditioned target
head. An empty string means no extra hidden head layers, preserving the
historical single sigmoid head.

# Arguments
- `s::AbstractString`: comma-separated layer widths.

# Returns
- `Vector{Int64}`: parsed hidden widths; `Int64[]` when `s` is empty.

# Examples
```julia
parse_layers("256, 256") == Int64[256, 256]
parse_layers("") == Int64[]
```
"""
parse_layers(s::AbstractString) =
    isempty(strip(s)) ? Int64[] : [parse(Int64, strip(x)) for x in split(s, ",") if !isempty(strip(x))]
layers = parse_layers(get(ENV, "DR_ENCODER_LAYERS", get(ENV, "DR_LAYERS", "128,128")))
head_layers = parse_layers(get(ENV, "DR_HEAD_LAYERS", ""))
grad_clip = parse(Float32, get(ENV, "DR_GRAD_CLIP", "0"))

function canonical_context_mode(raw_mode::AbstractString)
    mode = lowercase(strip(raw_mode))
    mode in ("", "none", "off", "false") && return ""
    mode in ("phase", "phase+progress") && return mode
    error("DR_CONTEXT must be \"\", \"phase\", or \"phase+progress\"; got \"$raw_mode\"")
end

function build_stage_context(mode::AbstractString, horizon::Int, period::Int)
    isempty(mode) && return nothing
    include_progress = mode == "phase+progress"
    return DecisionRules.stage_phase_context(
        horizon;
        period=period,
        include_progress=include_progress,
    )
end

function context_run_tag(mode::AbstractString)
    isempty(mode) && return ""
    return "-ctx" * replace(mode, "+" => "p")
end

context_mode = canonical_context_mode(get(ENV, "DR_CONTEXT", ""))
context_period = countlines(joinpath(HydroPowerModels_dir, case_name, "inflows.csv"))
stage_context = build_stage_context(context_mode, num_stages, context_period)
n_context = isnothing(stage_context) ? 0 : size(stage_context, 1)

# ── Learning-rate schedule ───────────────────────────────────────────────────
# lr(iter) = linear warmup from lr_init/100 over `lr_warmup` iterations, then
# cosine decay from lr_init to lr_final across the remaining budget. With the
# defaults (warmup = 0, lr_final = lr_init) this is a constant lr_init,
# reproducing the historical behavior exactly.
lr_init = parse(Float64, get(ENV, "DR_LR", "0.001"))
lr_final = parse(Float64, get(ENV, "DR_LR_FINAL", string(lr_init)))
lr_warmup = parse(Int, get(ENV, "DR_LR_WARMUP", "0"))

"""
    lr_schedule(iter, total_iters) -> Float64

Learning rate at one-based training iteration `iter`.

Linear warmup from `lr_init / 100` to `lr_init` over the first `lr_warmup`
iterations, then cosine decay from `lr_init` to `lr_final`:

```math
\\eta(k) = \\eta_f + \\tfrac{1}{2} (\\eta_0 - \\eta_f)
           \\bigl(1 + \\cos(\\pi \\rho_k)\\bigr),
```

where ``\\rho_k`` is the post-warmup progress fraction. Constant when
`lr_warmup == 0` and `lr_final == lr_init` (the defaults).
"""
function lr_schedule(iter, total_iters)
    if iter <= lr_warmup
        # Warmup guards a warmstarted policy against full-size Adam steps
        # taken while the optimizer's second-moment estimates are near zero.
        return lr_init * (0.01 + 0.99 * iter / max(lr_warmup, 1))
    end
    # Post-warmup progress in [0, 1] over the remaining iteration budget.
    ρ = clamp((iter - lr_warmup) / max(total_iters - lr_warmup, 1), 0.0, 1.0)
    return lr_final + 0.5 * (lr_init - lr_final) * (1 + cos(π * ρ))
end

optimizers = if grad_clip > 0
    [Flux.Optimisers.OptimiserChain(Flux.Optimisers.ClipGrad(grad_clip), Flux.Adam(lr_init))]
else
    [Flux.Adam(lr_init)]
end
pre_trained_model = get(ENV, "DR_PRETRAINED_MODEL", nothing)
clip_tag = grad_clip > 0 ? "-clip$(Int(grad_clip))" : ""
head_tag = isempty(head_layers) ? "-Hlinear" : "-H$(join(head_layers, "_"))"
_rollout_tag = num_rollout_stages != num_stages ? "-r$(num_rollout_stages)" : ""
# Tag runs with a non-default batch size so checkpoints are distinguishable.
nt_tag = _num_train_per_batch > 1 ? "-nt$(_num_train_per_batch)" : ""
# Tag warmstarted runs: their result is a fine-tune of another checkpoint, not
# a from-scratch training (the parent checkpoint is recorded in wandb config).
warm_tag = (isnothing(pre_trained_model) || pre_trained_model == "nothing") ? "" : "-warm"
save_file = "$(case_name)-$(formulation)-h$(num_stages)$(_rollout_tag)-subproblems-strict$(clip_tag)$(head_tag)$(nt_tag)$(context_run_tag(context_mode))$(warm_tag)-$(now())"
num_eval_scenarios = parse(Int, get(ENV, "DR_NUM_EVAL_SCENARIOS", "4"))
eval_every = parse(Int, get(ENV, "DR_EVAL_EVERY", "25"))
# Checkpoint-selection metric: "training" (historical; noisy at small batch
# sizes because a lucky scenario can look like a better policy) or "rollout"
# (deficit-free mean over the fixed held-out scenarios — the deployment metric).
save_metric = lowercase(get(ENV, "DR_SAVE_METRIC", "training"))
save_metric in ("training", "rollout") ||
    error("DR_SAVE_METRIC must be \"training\" or \"rollout\", got $save_metric")

# ── Build strict subproblems (no deficit, no penalty) ────────────────────────

# Define the DiffOpt optimizer for subproblems
diff_optimizer =
    () -> DiffOpt.diff_optimizer(
        optimizer_with_attributes(
            Ipopt.Optimizer,
            "print_level" => 0,
            "linear_solver" => "mumps",
        ),
    )

subproblems, state_params_in, state_params_out, uncertainty_samples,
    initial_state, max_volume, hydro_meta = build_hydropowermodels(
    joinpath(HydroPowerModels_dir, case_name),
    formulation_file;
    num_stages=num_stages,
    optimizer=diff_optimizer,
    strict=true,
    # Per-stage seasonal demand (bolivia/demand.csv, cyclically tiled) and the
    # paper deficit cost — parity with SDDP and DecisionRulesExa (see
    # build_hydropowermodels; demand_file=:auto picks up demand.csv).
    load_scaler=load_scaler,
    deficit_cost=deficit_cost,
)

num_hydro = length(initial_state)

# ── Logging ──────────────────────────────────────────────────────────────────

lg = WandbLogger(;
    project="RL",
    name=save_file,
    save_code=false,
    config=Dict(
        "layers" => layers,
        "head_layers" => head_layers,
        "activation" => "sigmoid (reachable)",
        "optimizer" => string(optimizers),
        "grad_clip" => grad_clip,
        "training_method" => "subproblems-strict",
        "penalty_schedule" => "none (strict)",
        "num_stages" => num_stages,
        "num_rollout_stages" => num_rollout_stages,
        "load_scaler" => load_scaler,
        "deficit_cost" => deficit_cost,
        "num_epochs" => string(num_epochs),
        "num_batches" => string(num_batches),
        "num_train_per_batch" => string(_num_train_per_batch),
        "pre_trained_model" => string(pre_trained_model),
        "context_mode" => isempty(context_mode) ? "none" : context_mode,
        "context_period" => context_period,
        "context_horizon" => num_stages,
        "n_context" => n_context,
        "num_eval_scenarios" => num_eval_scenarios,
        "eval_every" => eval_every,
        "save_metric" => save_metric,
        "lr" => lr_init,
        "lr_final" => lr_final,
        "lr_warmup" => lr_warmup,
    ),
)

# ── Build reachable policy ───────────────────────────────────────────────────

# HydroReachablePolicy: LSTM encoder over inflows + sigmoid feed-forward head
# over [encoded_inflow; reservoir_state], bounded to the one-stage reachable set.
base_model = hydro_reachable_policy(
    hydro_meta,
    layers;
    combiner_layers=head_layers,
    n_context=n_context,
)
models = isnothing(stage_context) ? base_model : ContextualPolicy(base_model, stage_context)
@info "Strict hydro policy context" context_mode=(isempty(context_mode) ? "none" : context_mode) context_period context_horizon=num_stages n_context

# ── Load pretrained model (warmstart from non-strict training) ───────────────

if !isnothing(pre_trained_model) && pre_trained_model != "nothing"
    model_save = JLD2.load(pre_trained_model)
    model_state = model_save["model_state"]
    # Load encoder/combiner weights; hydro bounds are preserved
    load_policy_weights!(models, model_state)
    @info "Loaded pretrained weights from $pre_trained_model"
end

# ── Initial evaluation and callbacks ─────────────────────────────────────────

Random.seed!(8788)
objective_values = [
    simulate_multistage(
        subproblems,
        state_params_in,
        state_params_out,
        initial_state,
        DecisionRules.sample(uncertainty_samples),
        models;
    ) for _ in 1:2
]
initial_training_obj = mean(objective_values)
convergence_criterium = StallingCriterium(num_epochs * num_batches, initial_training_obj, 0)

# Fixed held-out scenarios, materialized once so every evaluation uses the same set.
# Use num_rollout_stages for evaluation (may differ from training num_stages).
Random.seed!(8789)
rollout_uncertainty = uncertainty_samples[1:num_rollout_stages]
eval_scenarios = [DecisionRules.sample(rollout_uncertainty) for _ in 1:num_eval_scenarios]
rollout_evaluation = RolloutEvaluation(
    subproblems[1:num_rollout_stages],
    state_params_in[1:num_rollout_stages],
    state_params_out[1:num_rollout_stages],
    initial_state,
    eval_scenarios;
    stride=eval_every,
    policy_state=:realized,
)

# Checkpoint-selection baseline. With save_metric == "rollout" the incumbent
# is the current model's held-out rollout cost, so a warmstarted run only saves
# checkpoints that genuinely improve on the loaded policy under the metric it
# is ultimately judged on. iter = eval_every satisfies the stride gate.
best_obj = if save_metric == "rollout"
    rollout_evaluation(eval_every, models)
    @info "Initial rollout evaluation (checkpoint baseline)" rollout_evaluation.last_objective_no_deficit rollout_evaluation.last_violation_share
    rollout_evaluation.last_objective_no_deficit
else
    initial_training_obj
end
model_path = joinpath(model_dir, save_file * ".jld2")
save_control = SaveBest(best_obj, model_path)

# ── Train ────────────────────────────────────────────────────────────────────

# Total iteration budget, used by the learning-rate schedule.
total_iters = num_epochs * num_batches

# No penalty schedule needed — strict mode has no deficit to penalize.
train_multistage(
    models,
    initial_state,
    subproblems,
    state_params_in,
    state_params_out,
    uncertainty_samples;
    num_batches=total_iters,
    num_train_per_batch=_num_train_per_batch,
    optimizer=first(optimizers),
    # Apply the learning-rate schedule through the optimizer state. With the
    # default constant schedule adjust! is a no-op-equivalent every iteration.
    adjust_hyperparameters=(iter, opt_state, ntpb) -> begin
        Flux.Optimisers.adjust!(opt_state, lr_schedule(iter, total_iters))
        ntpb
    end,
    record=(sample_log, iter, model) -> begin
        # In strict mode: objectives == objectives_no_deficit (no penalty term)
        training_loss = mean(sample_log.objectives)
        metrics = Dict(
            "metrics/loss" => training_loss,
            "metrics/training_loss" => training_loss,
            "metrics/lr" => lr_schedule(iter, total_iters),
        )
        rollout_evaluation(iter, model)
        if iter % eval_every == 0
            metrics["metrics/rollout_objective_no_deficit"] =
                rollout_evaluation.last_objective_no_deficit
            metrics["metrics/rollout_target_violation_share"] =
                rollout_evaluation.last_violation_share
        end
        Wandb.log(lg, metrics)
        # Checkpoint selection: historical per-batch training loss, or the
        # held-out rollout objective (only refreshed at eval iterations).
        if save_metric == "rollout"
            iter % eval_every == 0 &&
                save_control(iter, model, rollout_evaluation.last_objective_no_deficit)
        else
            save_control(iter, model, training_loss)
        end
        return convergence_criterium(iter, model, training_loss)
    end,
    penalty_schedule=nothing,
)

# Finish the run
close(lg)
