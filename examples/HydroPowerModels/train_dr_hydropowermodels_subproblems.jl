# import Pkg
# Pkg.activate(".")
using DecisionRules
using Statistics
using Random
using Flux

# using MosekTools
using Ipopt, HSL_jll # Gurobi, MosekTools, Ipopt, MadNLP
# using Gurobi # Gurobi, MosekTools, Ipopt, MadNLP
# import CUDA # if error run CUDA.set_runtime_version!(v"12.1.0")
# CUDA.set_runtime_version!(v"12.1.0")
# using MadNLP 
# using MadNLPGPU
# import ParametricOptInterface as POI
using Wandb, Dates, Logging
using JLD2
using DiffOpt
using MadDiff, MadNLP

HydroPowerModels_dir = dirname(@__FILE__)
include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))

# Functions

function non_ensurance(x_out, x_in, uncertainty, max_volume)
    return x_out
end

# Parameters
case_name = "bolivia" # bolivia, case3
formulation = "ACPPowerModel" # SOCWRConicPowerModel, DCPPowerModel, ACPPowerModel
num_stages = 96 # 96, 48
model_dir = joinpath(HydroPowerModels_dir, case_name, formulation, "models")
mkpath(model_dir)
save_file = "$(case_name)-$(formulation)-h$(num_stages)-subproblems-$(now())"
formulation_file = formulation * ".mof.json"
num_epochs=30
num_batches=100
_num_train_per_batch=1
# dense = Flux.LSTM # RNN, Dense, LSTM
activation = sigmoid # tanh, DecisionRules.identity, relu
layers = Int64[128, 128] # Int64[8, 8], Int64[]
ensure_feasibility = non_ensurance # ensure_feasibility_double_softplus
optimizers= [Flux.Adam()] # Flux.Adam(0.01), Flux.Descent(0.1), Flux.RMSProp(0.00001, 0.001)
pre_trained_model = nothing #joinpath(HydroPowerModels_dir, case_name, formulation, "models", "case3-ACPPowerModel-h48-2024-05-18T10:16:25.117.jld2")
penalty_l2 = :auto
penalty_l1 = :auto

# Build MSP using subproblems (not deterministic equivalent)

# Define the DiffOpt optimizer for subproblems
# diff_optimizer = () -> DiffOpt.diff_optimizer(optimizer_with_attributes(Ipopt.Optimizer, 
#     "print_level" => 0,
#     "hsllib" => HSL_jll.libhsl_path,
#     "linear_solver" => "ma27"
# ))

model_builder = () -> MadDiff.nonlinear_diff_model(MadNLP.Optimizer)

subproblems, state_params_in, state_params_out, uncertainty_samples, initial_state, max_volume = build_hydropowermodels(    
    joinpath(HydroPowerModels_dir, case_name), formulation_file; 
    num_stages=num_stages,
    model_builder=model_builder,
    penalty_l1=penalty_l1,
    penalty_l2=penalty_l2
)

num_hydro = length(initial_state)

# Logging

lg = WandbLogger(
    project = "RL",
    name = save_file,
    config = Dict(
        "layers" => layers,
        "activation" => string(activation),
        # "dense" => string(dense),
        "ensure_feasibility" => string(ensure_feasibility),
        "optimizer" => string(optimizers),
        "training_method" => "subproblems",
        "penalty_l1" => string(penalty_l1),
        "penalty_l2" => string(penalty_l2),
        "num_epochs" => string(num_epochs),
        "num_batches" => string(num_batches),
        "num_train_per_batch" => string(_num_train_per_batch),
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
                                   activation=activation, encoder_type=Flux.LSTM)

# Load pretrained Model
if !isnothing(pre_trained_model)
    model_save = JLD2.load(pre_trained_model)
    model_state = model_save["model_state"]
    Flux.loadmodel!(models, model_state)
end

Random.seed!(8788)
objective_values = [simulate_multistage(
    subproblems, state_params_in, state_params_out, 
    initial_state, DecisionRules.sample(uncertainty_samples), 
    models;
    # ensure_feasibility=(x_out, x_in, uncertainty) -> ensure_feasibility(x_out, x_in, uncertainty, max_volume)
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

# Train Model using subproblems (not deterministic equivalent)
for iter in 1:num_epochs
    num_train_per_batch = _num_train_per_batch
    train_multistage(models, initial_state, subproblems, state_params_in, state_params_out, uncertainty_samples; 
        num_batches=num_batches,
        num_train_per_batch=num_train_per_batch,
        optimizer=optimizers[floor(min(iter, length(optimizers)))],
        record_loss= (iter, model, loss, tag) -> begin
            if tag == "metrics/training_loss"
                save_control(iter, model, loss)
                record_loss(iter, model, loss, tag)
                return convergence_criterium(iter, model, loss)
            end
            return record_loss(iter, model, loss, tag)
        end,
        # ensure_feasibility=(x_out, x_in, uncertainty) -> ensure_feasibility(x_out, x_in, uncertainty, max_volume),
        adjust_hyperparameters=adjust_hyperparameters
    )
end

# Finish the run
close(lg)
