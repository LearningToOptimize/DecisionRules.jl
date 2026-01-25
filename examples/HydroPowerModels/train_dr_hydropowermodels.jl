# Train HydroPowerModels using Deterministic Equivalent formulation
using DecisionRules
using Statistics
using Random
using Flux

using Ipopt, HSL_jll
using Wandb, Dates, Logging
using JLD2
using DiffOpt
using JuMP

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
save_file = "$(case_name)-$(formulation)-h$(num_stages)-deteq-$(now())"
formulation_file = formulation * ".mof.json"

# Training parameters
num_epochs = 100
num_batches = 32
_num_train_per_batch = 1
dense = Flux.LSTM                        # RNN, Dense, LSTM
activation = sigmoid                     # tanh, identity, relu, sigmoid
layers = Int64[128, 128]
ensure_feasibility = non_ensurance
optimizers = [Flux.Adam()]
pre_trained_model = nothing

# Build MSP with deterministic equivalent formulation
subproblems, state_params_in, state_params_out, uncertainty_samples, initial_state, max_volume = build_hydropowermodels(    
    joinpath(HydroPowerModels_dir, case_name), formulation_file; 
    num_stages=num_stages
)

# det_equivalent = DiffOpt.diff_model(optimizer_with_attributes(Ipopt.Optimizer, 
#     "print_level" => 0,
#     "hsllib" => HSL_jll.libhsl_path,
#     "linear_solver" => "ma27"
# ))

det_equivalent = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer, 
    "print_level" => 0,
    "hsllib" => HSL_jll.libhsl_path,
    "linear_solver" => "ma27"
))

det_equivalent, uncertainty_samples = DecisionRules.deterministic_equivalent!(
    det_equivalent, subproblems, state_params_in, state_params_out, 
    initial_state, uncertainty_samples
)

num_hydro = length(initial_state)

# Logging
lg = WandbLogger(
    project = "RL",
    name = save_file,
    config = Dict(
        "layers" => layers,
        "activation" => string(activation),
        "dense" => string(dense),
        "ensure_feasibility" => string(ensure_feasibility),
        "optimizer" => string(optimizers),
        "training_method" => "deterministic_equivalent"
    )
)

function record_loss(iter, model, loss, tag)
    Wandb.log(lg, Dict(tag => loss))
    return false
end

# Define Model
# Policy architecture: LSTM processes uncertainty, Dense combines with previous state
num_uncertainties = length(uncertainty_samples[1])
models = state_conditioned_policy(num_uncertainties, num_hydro, num_hydro, layers; 
                                   activation=activation, encoder_type=dense)

# Load pretrained Model
if !isnothing(pre_trained_model)
    model_save = JLD2.load(pre_trained_model)
    model_state = model_save["model_state"]
    Flux.loadmodel!(models, model_state)
end

# Initial evaluation
Random.seed!(8788)
objective_values = [simulate_multistage(
    det_equivalent, state_params_in, state_params_out, 
    initial_state, DecisionRules.sample(uncertainty_samples), 
    models;
) for _ in 1:2]
best_obj = mean(objective_values)

model_path = joinpath(model_dir, save_file * ".jld2")
save_control = SaveBest(best_obj, model_path)
convergence_criterium = StallingCriterium(100, best_obj, 0)

adjust_hyperparameters = (iter, opt_state, num_train_per_batch) -> begin
    if iter % 2100 == 0
        num_train_per_batch = num_train_per_batch * 2
    end
    return num_train_per_batch
end

# Train Model using deterministic equivalent
for iter in 1:num_epochs
    num_train_per_batch = _num_train_per_batch
    train_multistage(models, initial_state, det_equivalent, state_params_in, state_params_out, uncertainty_samples; 
        num_batches=num_batches,
        num_train_per_batch=num_train_per_batch,
        optimizer=optimizers[floor(Int, min(iter, length(optimizers)))],
        record_loss= (iter, model, loss, tag) -> begin
            if tag == "metrics/training_loss"
                save_control(iter, model, loss)
                record_loss(iter, model, loss, tag)
                return convergence_criterium(iter, model, loss)
            end
            return record_loss(iter, model, loss, tag)
        end,
        adjust_hyperparameters=adjust_hyperparameters
    )
end

# Finish the run
close(lg)
