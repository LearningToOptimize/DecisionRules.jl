# Train HydroPowerModels using multiple shooting (windowed decomposition)
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

# Functions

function non_ensurance(x_out, x_in, uncertainty, max_volume)
    return x_out
end

# Parameters
case_name = "bolivia"                    # bolivia, case3
formulation = "ACPPowerModel"            # SOCWRConicPowerModel, DCPPowerModel, ACPPowerModel
num_stages = 96                          # 96, 48
window_size = 12                       # 12, 6
model_dir = joinpath(HydroPowerModels_dir, case_name, formulation, "models")
mkpath(model_dir)
formulation_file = formulation * ".mof.json"

# Training parameters
num_epochs = 30
num_batches = 100
_num_train_per_batch = 1
activation = sigmoid                     # tanh, identity, relu, sigmoid
layers = Int64[128, 128]
ensure_feasibility = non_ensurance
grad_clip = parse(Float32, get(ENV, "DR_GRAD_CLIP", "0"))
optimizers = if grad_clip > 0
    [Flux.Optimisers.OptimiserChain(Flux.Optimisers.ClipGrad(grad_clip), Flux.Adam())]
else
    [Flux.Adam()]
end
pre_trained_model = nothing
penalty_l2 = :auto
penalty_l1 = :auto
penalty_schedule = get(ENV, "DR_PENALTY_SCHEDULE", "annealed") == "annealed" ? :default_annealed : nothing
clip_tag = grad_clip > 0 ? "-clip$(Int(grad_clip))" : ""
sched_tag = isnothing(penalty_schedule) ? "-const" : "-anneal"
save_file = "$(case_name)-$(formulation)-h$(num_stages)-shooting-w$(window_size)$(clip_tag)$(sched_tag)-$(now())"
num_eval_scenarios = 4
eval_every = 25

# Build MSP using subproblems (not deterministic equivalent)

# Define the DiffOpt optimizer for subproblems and window models
diff_optimizer =
    () -> DiffOpt.diff_optimizer(
        optimizer_with_attributes(
            Ipopt.Optimizer,
            "print_level" => 0,
            "linear_solver" => "mumps",
        ),
    )

diff_model =
    () -> DiffOpt.nonlinear_diff_model(
        optimizer_with_attributes(
            Ipopt.Optimizer,
            "print_level" => 0,
            "linear_solver" => "mumps",
        ),
    )

subproblems, state_params_in, state_params_out, uncertainty_samples, initial_state, max_volume = build_hydropowermodels(
    joinpath(HydroPowerModels_dir, case_name),
    formulation_file;
    num_stages=num_stages,
    optimizer=diff_optimizer,
    penalty_l1=penalty_l1,
    penalty_l2=penalty_l2,
)

num_hydro = length(initial_state)

# Logging
lg = WandbLogger(;
    project="RL",
    name=save_file,
    save_code=false,
    config=Dict(
        "layers" => layers,
        "activation" => string(activation),
        "ensure_feasibility" => string(ensure_feasibility),
        "optimizer" => string(optimizers),
        "grad_clip" => grad_clip,
        "training_method" => "multiple_shooting",
        "window_size" => string(window_size),
        "penalty_l1" => string(penalty_l1),
        "penalty_l2" => string(penalty_l2),
        "penalty_schedule" => string(penalty_schedule),
        "num_epochs" => string(num_epochs),
        "num_batches" => string(num_batches),
        "num_train_per_batch" => string(_num_train_per_batch),
    ),
)

# Define Model
# Policy architecture: LSTM processes uncertainty, Dense combines with previous state
num_uncertainties = length(uncertainty_samples[1][1])
models = state_conditioned_policy(
    num_uncertainties,
    num_hydro,
    num_hydro,
    layers;
    activation=activation,
    encoder_type=Flux.LSTM,
)

# Load pretrained Model
if !isnothing(pre_trained_model)
    model_save = JLD2.load(pre_trained_model)
    model_state = model_save["model_state"]
    Flux.loadmodel!(models, model_state)
end

# Initial evaluation
Random.seed!(8788)
windows = DecisionRules.setup_shooting_windows(
    subproblems,
    state_params_in,
    state_params_out,
    Float64.(initial_state),
    uncertainty_samples;
    window_size=window_size,
    model_factory=diff_model,
)

objective_values = [
    begin
        uncertainty_sample = DecisionRules.sample(uncertainty_samples)
        uncertainties_vec = [
            [Float32(u[2]) for u in stage_u] for stage_u in uncertainty_sample
        ]
        DecisionRules.simulate_multiple_shooting(
            windows, models, Float32.(initial_state), uncertainty_sample, uncertainties_vec
        )
    end for _ in 1:2
]

best_obj = mean(objective_values)

model_path = joinpath(model_dir, save_file * ".jld2")
save_control = SaveBest(best_obj, model_path)
convergence_criterium = StallingCriterium(num_epochs * num_batches, best_obj, 0)

Random.seed!(8789)
eval_scenarios = [DecisionRules.sample(uncertainty_samples) for _ in 1:num_eval_scenarios]
rollout_evaluation = RolloutEvaluation(
    subproblems,
    state_params_in,
    state_params_out,
    initial_state,
    eval_scenarios;
    stride=eval_every,
    policy_state=:realized,
)
resolved_penalty_schedule = isnothing(penalty_schedule) ? nothing :
    DecisionRules._resolve_penalty_schedule(penalty_schedule, num_epochs * num_batches)
pending_metrics = Dict{String,Any}()

# Train Model using multiple shooting.
# A single call over num_epochs*num_batches batches so the penalty schedule spans the whole
# run (this also keeps one optimizer state throughout, and a `true` return from the record
# callback now stops the whole run).
train_multiple_shooting(
    models,
    initial_state,
    windows,
    uncertainty_samples;
    num_batches=num_epochs * num_batches,
    num_train_per_batch=_num_train_per_batch,
    optimizer=first(optimizers),
    record_loss=(iter, model, loss, tag) -> begin
        pending_metrics[tag] = loss
        if tag == "metrics/training_loss"
            rollout_evaluation(iter, model)
            if iter % eval_every == 0
                pending_metrics["metrics/rollout_objective_no_deficit"] =
                    rollout_evaluation.last_objective_no_deficit
                pending_metrics["metrics/rollout_target_violation_share"] =
                    rollout_evaluation.last_violation_share
            end
            if !isnothing(resolved_penalty_schedule)
                pending_metrics["metrics/target_penalty_multiplier"] =
                    DecisionRules._penalty_multiplier_for(resolved_penalty_schedule, iter)
            end
            Wandb.log(lg, copy(pending_metrics))
            empty!(pending_metrics)
            save_control(iter, model, loss)
            return convergence_criterium(iter, model, loss)
        end
        return false
    end,
    penalty_schedule=penalty_schedule,
)

# Finish the run
close(lg)
