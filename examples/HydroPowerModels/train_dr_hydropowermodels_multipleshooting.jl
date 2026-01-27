# Train HydroPowerModels using multiple shooting (windowed decomposition)
using DecisionRules
using Statistics
using Random
using Flux

using Ipopt, HSL_jll
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
save_file = "$(case_name)-$(formulation)-h$(num_stages)-shooting-w$(window_size)-$(now())"
formulation_file = formulation * ".mof.json"

# Training parameters
num_epochs = 30
num_batches = 100
_num_train_per_batch = 1
activation = sigmoid                     # tanh, identity, relu, sigmoid
layers = Int64[128, 128]
ensure_feasibility = non_ensurance
optimizers = [Flux.Adam()]
pre_trained_model = nothing
penalty_l2 = :auto
penalty_l1 = :auto

# Build MSP using subproblems (not deterministic equivalent)

# Define the DiffOpt optimizer for subproblems and window models
diff_optimizer = () -> DiffOpt.diff_optimizer(optimizer_with_attributes(Ipopt.Optimizer,
    "print_level" => 0,
    "hsllib" => HSL_jll.libhsl_path,
    "linear_solver" => "ma27"
))

subproblems, state_params_in, state_params_out, uncertainty_samples, initial_state, max_volume = build_hydropowermodels(
    joinpath(HydroPowerModels_dir, case_name), formulation_file;
    num_stages=num_stages,
    optimizer=diff_optimizer,
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
        "ensure_feasibility" => string(ensure_feasibility),
        "optimizer" => string(optimizers),
        "training_method" => "multiple_shooting",
        "window_size" => string(window_size),
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

# Initial evaluation
Random.seed!(8788)
windows = DecisionRules.setup_shooting_windows(
    subproblems,
    state_params_in,
    state_params_out,
    Float64.(initial_state),
    uncertainty_samples;
    window_size=window_size,
    optimizer_factory=diff_optimizer,
)

objective_values = [begin
    uncertainty_sample = DecisionRules.sample(uncertainty_samples)
    uncertainties_vec = [[Float32(u[2]) for u in stage_u] for stage_u in uncertainty_sample]
    DecisionRules.simulate_multiple_shooting(
        windows,
        models,
        Float32.(initial_state),
        uncertainty_sample,
        uncertainties_vec
    )
end for _ in 1:2]

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

# Train Model using multiple shooting
for iter in 1:num_epochs
    num_train_per_batch = _num_train_per_batch
    train_multiple_shooting(
        models,
        initial_state,
        subproblems,
        state_params_in,
        state_params_out,
        () -> DecisionRules.sample(uncertainty_samples);
        window_size=window_size,
        num_batches=num_batches,
        num_train_per_batch=num_train_per_batch,
        optimizer=optimizers[floor(Int, min(iter, length(optimizers)))],
        record_loss=(iter, model, loss, tag) -> begin
            if tag == "metrics/training_loss"
                save_control(iter, model, loss)
                record_loss(iter, model, loss, tag)
                return convergence_criterium(iter, model, loss)
            end
            return record_loss(iter, model, loss, tag)
        end,
        adjust_hyperparameters=adjust_hyperparameters,
        optimizer_factory=diff_optimizer
    )
end

# Finish the run
close(lg)
