# Train HydroPowerModels using strict subproblems with reachable policy
#
# Strict mode removes the deficit slack variables from the target constraint:
#   reservoir_out[r] = target[r]     (hard equality, no penalty)
# The dual λ_r is the clean shadow price ∂Q/∂target — pure economic signal.
#
# This requires HydroReachablePolicy, which guarantees every target is within
# the one-stage reachable set via sigmoid-bounded outputs scaled to physical
# reservoir limits.
#
# Usage:
#   julia --project -t auto train_dr_hydropowermodels_strict.jl
#
# Environment overrides:
#   DR_NUM_STAGES=126          number of training stages
#   DR_NUM_ROLLOUT_STAGES=96   number of rollout evaluation stages (default: DR_NUM_STAGES)
#   DR_NUM_EPOCHS=80           number of epochs
#   DR_GRAD_CLIP=0             gradient clipping (0 = disabled)
#   DR_PRETRAINED_MODEL=path   warmstart from a StateConditionedPolicy checkpoint
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
num_rollout_stages = parse(Int, get(ENV, "DR_NUM_ROLLOUT_STAGES", string(num_stages)))
model_dir = joinpath(HydroPowerModels_dir, case_name, formulation, "models")
mkpath(model_dir)
formulation_file = formulation * ".mof.json"
num_epochs = parse(Int, get(ENV, "DR_NUM_EPOCHS", "80"))
num_batches = 100
_num_train_per_batch = 1
layers = Int64[128, 128]
grad_clip = parse(Float32, get(ENV, "DR_GRAD_CLIP", "0"))
optimizers = if grad_clip > 0
    [Flux.Optimisers.OptimiserChain(Flux.Optimisers.ClipGrad(grad_clip), Flux.Adam())]
else
    [Flux.Adam()]
end
pre_trained_model = get(ENV, "DR_PRETRAINED_MODEL", nothing)
clip_tag = grad_clip > 0 ? "-clip$(Int(grad_clip))" : ""
_rollout_tag = num_rollout_stages != num_stages ? "-r$(num_rollout_stages)" : ""
save_file = "$(case_name)-$(formulation)-h$(num_stages)$(_rollout_tag)-subproblems-strict$(clip_tag)-$(now())"
num_eval_scenarios = 4                   # fixed held-out scenarios for rollout evaluation
eval_every = 25                          # rollout-evaluate every eval_every batches

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
)

num_hydro = length(initial_state)

# ── Logging ──────────────────────────────────────────────────────────────────

lg = WandbLogger(;
    project="RL",
    name=save_file,
    save_code=false,
    config=Dict(
        "layers" => layers,
        "activation" => "sigmoid (reachable)",
        "optimizer" => string(optimizers),
        "grad_clip" => grad_clip,
        "training_method" => "subproblems-strict",
        "penalty_schedule" => "none (strict)",
        "num_stages" => num_stages,
        "num_rollout_stages" => num_rollout_stages,
        "num_epochs" => string(num_epochs),
        "num_batches" => string(num_batches),
        "num_train_per_batch" => string(_num_train_per_batch),
        "pre_trained_model" => string(pre_trained_model),
    ),
)

# ── Build reachable policy ───────────────────────────────────────────────────

# HydroReachablePolicy: LSTM encoder + Dense(sigmoid) combiner, outputs bounded
# to the one-stage reachable set [lower, upper] for each reservoir.
models = hydro_reachable_policy(hydro_meta, layers)

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
best_obj = mean(objective_values)

model_path = joinpath(model_dir, save_file * ".jld2")
save_control = SaveBest(best_obj, model_path)
convergence_criterium = StallingCriterium(num_epochs * num_batches, best_obj, 0)

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

# ── Train ────────────────────────────────────────────────────────────────────

# No penalty schedule needed — strict mode has no deficit to penalize.
train_multistage(
    models,
    initial_state,
    subproblems,
    state_params_in,
    state_params_out,
    uncertainty_samples;
    num_batches=num_epochs * num_batches,
    num_train_per_batch=_num_train_per_batch,
    optimizer=first(optimizers),
    record=(sample_log, iter, model) -> begin
        # In strict mode: objectives == objectives_no_deficit (no penalty term)
        training_loss = mean(sample_log.objectives)
        metrics = Dict(
            "metrics/loss" => training_loss,
            "metrics/training_loss" => training_loss,
        )
        rollout_evaluation(iter, model)
        if iter % eval_every == 0
            metrics["metrics/rollout_objective_no_deficit"] =
                rollout_evaluation.last_objective_no_deficit
            metrics["metrics/rollout_target_violation_share"] =
                rollout_evaluation.last_violation_share
        end
        Wandb.log(lg, metrics)
        save_control(iter, model, training_loss)
        return convergence_criterium(iter, model, training_loss)
    end,
    penalty_schedule=nothing,
)

# Finish the run
close(lg)
