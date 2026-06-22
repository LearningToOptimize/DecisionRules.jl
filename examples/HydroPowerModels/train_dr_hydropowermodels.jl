# Train HydroPowerModels using Deterministic Equivalent formulation (GPU-enabled)
using DecisionRules
using Statistics
using Random
using Flux

using Ipopt
using Wandb, Dates, Logging
using JLD2
using DiffOpt
using JuMP
using MadNLP

USE_GPU = try
    using CUDA, CUDSS, MadNLPGPU
    CUDA.functional()
catch
    @warn "GPU packages not available — running on CPU"
    false
end
@info "GPU status" USE_GPU

HydroPowerModels_dir = dirname(@__FILE__)
include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))

# Functions

function non_ensurance(x_out, x_in, uncertainty, max_volume)
    return x_out
end

# Parameters
case_name = "bolivia"                    # bolivia, case3
formulation = "ACPPowerModel"            # SOCWRConicPowerModel, DCPPowerModel, ACPPowerModel
num_stages = parse(Int, get(ENV, "DR_NUM_STAGES", "126"))
model_dir = joinpath(HydroPowerModels_dir, case_name, formulation, "models")
mkpath(model_dir)
solver_tag = USE_GPU ? "gpu" : "cpu"
formulation_file = formulation * ".mof.json"

# Training parameters
num_epochs = parse(Int, get(ENV, "DR_NUM_EPOCHS", "80"))
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
penalty_schedule = if get(ENV, "DR_PENALTY_SCHEDULE", "annealed") == "annealed"
    :default_annealed
else
    nothing
end
clip_tag = grad_clip > 0 ? "-clip$(Int(grad_clip))" : ""
sched_tag = isnothing(penalty_schedule) ? "-const" : "-anneal"
save_file = "$(case_name)-$(formulation)-h$(num_stages)-deteq-$(solver_tag)$(clip_tag)$(sched_tag)-$(now())"
num_eval_scenarios = 4
eval_every = 25

# Build MSP: subproblems for rollout evaluation (stage-wise, CPU, with DiffOpt)
diff_optimizer =
    () -> DiffOpt.diff_optimizer(
        optimizer_with_attributes(
            Ipopt.Optimizer,
            "print_level" => 0,
            "linear_solver" => "mumps",
        ),
    )
subproblems, state_params_in_sub, state_params_out_sub, uncertainty_samples_sub, initial_state, max_volume = build_hydropowermodels(
    joinpath(HydroPowerModels_dir, case_name),
    formulation_file;
    num_stages=num_stages,
    optimizer=diff_optimizer,
    penalty_l1=penalty_l1,
    penalty_l2=penalty_l2,
)

# Build det-eq for training
subproblems_de, state_params_in, state_params_out, uncertainty_samples, _, _ = build_hydropowermodels(
    joinpath(HydroPowerModels_dir, case_name),
    formulation_file;
    num_stages=num_stages,
    penalty_l1=penalty_l1,
    penalty_l2=penalty_l2,
)

det_equivalent = Model(MadNLP.Optimizer)

if USE_GPU
    set_optimizer_attribute(det_equivalent, "array_type", CUDA.CuArray)
    set_optimizer_attribute(det_equivalent, "linear_solver", MadNLPGPU.CUDSSSolver)
    set_optimizer_attribute(det_equivalent, "print_level", MadNLP.ERROR)
    set_optimizer_attribute(det_equivalent, "barrier", MadNLP.LOQOUpdate())
else
    set_optimizer_attribute(det_equivalent, "print_level", MadNLP.ERROR)
    set_optimizer_attribute(det_equivalent, "barrier", MadNLP.LOQOUpdate())
    # set_optimizer_attribute(det_equivalent, "linear_solver", MadNLPGPU.LapackCPUSolver())
end

det_equivalent, uncertainty_samples = DecisionRules.deterministic_equivalent!(
    det_equivalent,
    subproblems_de,
    state_params_in,
    state_params_out,
    initial_state,
    uncertainty_samples,
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
        "encoder_type" => "LSTM",
        "ensure_feasibility" => string(ensure_feasibility),
        "optimizer" => string(optimizers),
        "grad_clip" => grad_clip,
        "training_method" => "deterministic_equivalent",
        "solver" => USE_GPU ? "MadNLP+CUDSS (GPU)" : "MadNLP (CPU)",
        "penalty_l1" => string(penalty_l1),
        "penalty_l2" => string(penalty_l2),
        "penalty_schedule" => string(penalty_schedule),
        "num_epochs" => string(num_epochs),
        "num_batches" => string(num_batches),
        "num_train_per_batch" => string(_num_train_per_batch),
        "num_eval_scenarios" => num_eval_scenarios,
        "eval_every" => eval_every,
        "use_gpu" => USE_GPU,
    ),
)

# Define Model
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
@time objective_values = [
    simulate_multistage(
        det_equivalent,
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
stall_train = StallingCriterium(num_epochs * num_batches, best_obj, 0)
stall_rollout = StallingCriterium(num_epochs * num_batches, best_obj, 0)


# Rollout evaluation (stage-wise subproblems, CPU)
Random.seed!(8789)
eval_scenarios = [
    DecisionRules.sample(uncertainty_samples_sub) for _ in 1:num_eval_scenarios
]
rollout_evaluation = RolloutEvaluation(
    subproblems,
    state_params_in_sub,
    state_params_out_sub,
    initial_state,
    eval_scenarios;
    stride=eval_every,
    policy_state=:target,
)
realized_rollout_evaluation = RolloutEvaluation(
    subproblems,
    state_params_in_sub,
    state_params_out_sub,
    initial_state,
    eval_scenarios;
    stride=eval_every,
    policy_state=:realized,
)
resolved_penalty_schedule = isnothing(penalty_schedule) ? nothing :
    DecisionRules._resolve_penalty_schedule(penalty_schedule, num_epochs * num_batches)

# Train Model using deterministic equivalent.
train_multistage(
    models,
    initial_state,
    det_equivalent,
    state_params_in,
    state_params_out,
    uncertainty_samples;
    num_batches=num_epochs * num_batches,
    num_train_per_batch=_num_train_per_batch,
    optimizer=first(optimizers),
    record=(sample_log, iter, model) -> begin
        training_loss = mean(sample_log.objectives)
        loss_no_deficit = mean(sample_log.objectives_no_deficit)
        metrics = Dict(
            "metrics/loss" => loss_no_deficit,
            "metrics/training_loss" => training_loss,
        )
        rollout_evaluation(iter, model)
        realized_rollout_evaluation(iter, model)
        converged_training = stall_train(iter, model, training_loss)
        converged_rollout = false
        if iter % eval_every == 0
            converged_rollout = stall_rollout(
                iter, model, rollout_evaluation.last_objective_no_deficit
            )
            metrics["metrics/rollout_objective_no_deficit"] =
                rollout_evaluation.last_objective_no_deficit
            metrics["metrics/rollout_target_violation_share"] =
                rollout_evaluation.last_violation_share
            metrics["metrics/rollout_realized_objective_no_deficit"] =
                realized_rollout_evaluation.last_objective_no_deficit
            metrics["metrics/rollout_realized_target_violation_share"] =
                realized_rollout_evaluation.last_violation_share
        end
        if !isnothing(resolved_penalty_schedule)
            metrics["metrics/target_penalty_multiplier"] =
                DecisionRules._penalty_multiplier_for(resolved_penalty_schedule, iter)
        end
        Wandb.log(lg, metrics)
        save_control(iter, model, training_loss)
        return converged_training && converged_rollout && isapprox(training_loss, rollout_evaluation.last_objective_no_deficit; rtol=0.01)
    end,
    penalty_schedule=penalty_schedule,
)

# Finish the run
close(lg)
