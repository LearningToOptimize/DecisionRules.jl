# Train HydroPowerModels using Deterministic Equivalent formulation (GPU-enabled)
using DecisionRules
using Statistics
using Random
using Flux

using Ipopt, HSL_jll
using Wandb, Dates, Logging
using JLD2
using DiffOpt
using JuMP
using MadNLP
using CUDA
using CUDSS
using MadNLPGPU

USE_GPU = CUDA.functional()
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
num_stages = 96                          # 96, 48
model_dir = joinpath(HydroPowerModels_dir, case_name, formulation, "models")
mkpath(model_dir)
solver_tag = USE_GPU ? "gpu" : "cpu"
save_file = "$(case_name)-$(formulation)-h$(num_stages)-deteq-$(solver_tag)-$(now())"
formulation_file = formulation * ".mof.json"

# Training parameters
num_epochs = 20
num_batches = 100
_num_train_per_batch = 1
activation = sigmoid                     # tanh, identity, relu, sigmoid
layers = Int64[128, 128]
ensure_feasibility = non_ensurance
optimizers = [Flux.Adam()]
pre_trained_model = nothing
penalty_l2 = :auto
penalty_l1 = :auto
penalty_schedule = :default_annealed
num_eval_scenarios = 4
eval_every = 25

# Build MSP: subproblems for rollout evaluation (stage-wise, CPU, with DiffOpt)
diff_optimizer = () -> DiffOpt.diff_optimizer(optimizer_with_attributes(Ipopt.Optimizer,
    "print_level" => 0,
    "hsllib" => HSL_jll.libhsl_path,
    "linear_solver" => "ma27"
))
subproblems, state_params_in_sub, state_params_out_sub, uncertainty_samples_sub, initial_state, max_volume = build_hydropowermodels(
    joinpath(HydroPowerModels_dir, case_name), formulation_file;
    num_stages=num_stages,
    optimizer=diff_optimizer,
    penalty_l1=penalty_l1,
    penalty_l2=penalty_l2
)

# Build det-eq for training
subproblems_de, state_params_in, state_params_out, uncertainty_samples, _, _ = build_hydropowermodels(
    joinpath(HydroPowerModels_dir, case_name), formulation_file;
    num_stages=num_stages,
    penalty_l1=penalty_l1,
    penalty_l2=penalty_l2
)

if USE_GPU
    det_equivalent = Model(MadNLP.Optimizer)
    set_optimizer_attribute(det_equivalent, "array_type", CUDA.CuArray)
    set_optimizer_attribute(det_equivalent, "linear_solver", MadNLPGPU.CUDSSSolver)
    set_optimizer_attribute(det_equivalent, "print_level", MadNLP.ERROR)
    set_optimizer_attribute(det_equivalent, "barrier", MadNLP.LOQOUpdate())
else
    det_equivalent = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0,
        "hsllib" => HSL_jll.libhsl_path,
        "linear_solver" => "ma27"
    ))
end

det_equivalent, uncertainty_samples = DecisionRules.deterministic_equivalent!(
    det_equivalent, subproblems_de, state_params_in, state_params_out,
    initial_state, uncertainty_samples
)

num_hydro = length(initial_state)

# Logging
lg = WandbLogger(
    project = "RL",
    name = save_file,
    save_code = false,
    config = Dict(
        "layers" => layers,
        "activation" => string(activation),
        "encoder_type" => "LSTM",
        "ensure_feasibility" => string(ensure_feasibility),
        "optimizer" => string(optimizers),
        "training_method" => "deterministic_equivalent",
        "solver" => USE_GPU ? "MadNLP+CUDSS (GPU)" : "Ipopt+MA27 (CPU)",
        "penalty_l1" => string(penalty_l1),
        "penalty_l2" => string(penalty_l2),
        "penalty_schedule" => string(penalty_schedule),
        "num_epochs" => string(num_epochs),
        "num_batches" => string(num_batches),
        "num_train_per_batch" => string(_num_train_per_batch),
        "num_eval_scenarios" => num_eval_scenarios,
        "eval_every" => eval_every,
        "use_gpu" => USE_GPU,
    )
)

# Define Model
num_uncertainties = length(uncertainty_samples[1])
models = state_conditioned_policy(num_uncertainties, num_hydro, num_hydro, layers;
                                   activation=activation, encoder_type=Flux.LSTM)

# Load pretrained Model
if !isnothing(pre_trained_model)
    model_save = JLD2.load(pre_trained_model)
    model_state = model_save["model_state"]
    Flux.loadmodel!(models, model_state)
end

# Initial evaluation
Random.seed!(8788)
@time objective_values = [simulate_multistage(
    det_equivalent, state_params_in, state_params_out,
    initial_state, DecisionRules.sample(uncertainty_samples),
    models;
) for _ in 1:2]
best_obj = mean(objective_values)

model_path = joinpath(model_dir, save_file * ".jld2")
save_control = SaveBest(best_obj, model_path)
convergence_criterium = StallingCriterium(100, best_obj, 0)

# Rollout evaluation (stage-wise subproblems, CPU)
Random.seed!(8789)
eval_scenarios = [DecisionRules.sample(uncertainty_samples_sub) for _ in 1:num_eval_scenarios]
rollout_evaluation = RolloutEvaluation(subproblems, state_params_in_sub, state_params_out_sub,
    initial_state, eval_scenarios; stride=eval_every)

# Train Model using deterministic equivalent.
train_multistage(models, initial_state, det_equivalent, state_params_in, state_params_out, uncertainty_samples;
    num_batches=num_epochs * num_batches,
    num_train_per_batch=_num_train_per_batch,
    optimizer=first(optimizers),
    record=(sample_log, iter, model) -> begin
        training_loss = mean(sample_log.objectives)
        loss_no_deficit = mean(sample_log.objectives_no_deficit)
        Wandb.log(lg, Dict(
            "metrics/loss" => loss_no_deficit,
            "metrics/training_loss" => training_loss,
        ))
        rollout_evaluation(iter, model)
        if iter % eval_every == 0
            Wandb.log(lg, Dict(
                "metrics/rollout_objective_no_deficit" => rollout_evaluation.last_objective_no_deficit,
                "metrics/rollout_target_violation_share" => rollout_evaluation.last_violation_share,
            ))
        end
        save_control(iter, model, training_loss)
        return convergence_criterium(iter, model, training_loss)
    end,
    penalty_schedule=penalty_schedule
)

# Finish the run
close(lg)
