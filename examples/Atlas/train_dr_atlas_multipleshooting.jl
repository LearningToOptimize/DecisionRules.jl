# Train DecisionRules.jl policy for Atlas Robot Balancing
# Using multiple shooting (windowed decomposition)

using Flux
using DecisionRules
using Random
using Statistics
using JuMP
import Ipopt, HSL_jll
using Wandb, Dates, Logging
using JLD2
using DiffOpt

Atlas_dir = dirname(@__FILE__)
include(joinpath(Atlas_dir, "build_atlas_problem.jl"))

# ============================================================================
# Parameters
# ============================================================================

# Problem parameters
N = 11                          # Number of time steps
h = 0.01                        # Time step
perturbation_scale = 0.05       # Scale of random perturbations
num_scenarios = 10              # Number of uncertainty samples per stage
penalty = 1e3                   # Penalty for state deviation
perturbation_frequency = 5      # Frequency of perturbations (every k stages)
window_size = 5                 # Multiple shooting window length

# Training parameters
num_epochs = 1
num_batches = 100
_num_train_per_batch = 1
layers = Int64[64, 64]
activation = sigmoid
optimizers = [Flux.Adam(0.001)]

# Save paths
model_dir = joinpath(Atlas_dir, "models")
mkpath(model_dir)
save_file = "atlas-balancing-shooting-N$(N)-w$(window_size)-$(now())"

# ============================================================================
# Build Subproblems
# ============================================================================

println("Building Atlas subproblems...")

diff_optimizer = () -> DiffOpt.diff_optimizer(optimizer_with_attributes(Ipopt.Optimizer,
    "print_level" => 0,
    "hsllib" => HSL_jll.libhsl_path,
    "linear_solver" => "ma27"
))

@time subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples,
      _, _, x_ref, u_ref, atlas = build_atlas_subproblems(;
    N = N,
    h = h,
    perturbation_scale = perturbation_scale,
    num_scenarios = num_scenarios,
    penalty = penalty,
    perturbation_frequency = perturbation_frequency,
    optimizer = diff_optimizer,
)

nx = atlas.nx
nu = atlas.nu
n_perturb = length(uncertainty_samples[1])  # Number of perturbation parameters

println("Atlas state dimension: $nx")
println("Atlas control dimension: $nu")
println("Number of perturbations: $n_perturb")
println("Number of stages: $(N-1)")
println("Window size: $window_size")

# ============================================================================
# Logging
# ============================================================================

lg = WandbLogger(
    project = "DecisionRules-Atlas",
    name = save_file,
    config = Dict(
        "N" => N,
        "h" => h,
        "perturbation_scale" => perturbation_scale,
        "num_scenarios" => num_scenarios,
        "penalty" => penalty,
        "perturbation_frequency" => perturbation_frequency,
        "window_size" => window_size,
        "layers" => layers,
        "activation" => string(activation),
        "optimizer" => string(optimizers),
        "nx" => nx,
        "nu" => nu,
        "training_method" => "multiple_shooting",
    )
)

function record_loss(iter, model, loss, tag)
    Wandb.log(lg, Dict(tag => loss))
    return false
end

# ============================================================================
# Define Neural Network Policy
# ============================================================================

# Policy architecture: LSTM processes perturbations, Dense combines with previous state
# This design is memory-efficient and allows the LSTM to focus on temporal patterns
n_uncertainties = length(uncertainty_samples[1])
models = state_conditioned_policy(n_uncertainties, nx, nx, layers;
                                   activation=activation, encoder_type=Flux.LSTM)

println("Model architecture: StateConditionedPolicy")
println("  Encoder (LSTM): $n_uncertainties -> $(layers)")
println("  Combiner (Dense): $(layers[end]) + $nx -> $nx")

# ============================================================================
# Setup multiple shooting windows
# ============================================================================

windows = DecisionRules.setup_shooting_windows(
    subproblems,
    state_params_in,
    state_params_out,
    Float64.(initial_state),
    uncertainty_samples;
    window_size=window_size,
    optimizer_factory=diff_optimizer,
)

# ============================================================================
# Initial Evaluation
# ============================================================================

println("\nEvaluating initial policy...")
Random.seed!(8788)
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
println("Initial objective: $best_obj")

model_path = joinpath(model_dir, save_file * ".jld2")
save_control = SaveBest(best_obj, model_path)
convergence_criterium = StallingCriterium(200, best_obj, 0)

# ============================================================================
# Hyperparameter Adjustment
# ============================================================================

adjust_hyperparameters = (iter, opt_state, num_train_per_batch) -> begin
    if iter % 2100 == 0
        num_train_per_batch = num_train_per_batch * 2
    end
    return num_train_per_batch
end

# ============================================================================
# Training
# ============================================================================

println("\nStarting training with multiple shooting...")
println("Epochs: $num_epochs, Batches per epoch: $num_batches")

for iter in 1:num_epochs
    num_train_per_batch = _num_train_per_batch
    train_multiple_shooting(
        models,
        initial_state,
        windows,
        () -> uncertainty_samples;
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
        adjust_hyperparameters=adjust_hyperparameters
    )
end

# Finish logging
close(lg)

println("\nModel saved to: $model_path")
println("Training complete!")
