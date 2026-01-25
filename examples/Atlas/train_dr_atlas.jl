# Train DecisionRules.jl policy for Atlas Robot Balancing
#
# This script trains a neural network policy to control the Atlas robot
# under stochastic perturbations, similar to the Goddard Rocket example.

using Flux
using DecisionRules
using Random
using Statistics
using JuMP
import Ipopt, HSL_jll
using Wandb, Dates, Logging
using JLD2

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

# Training parameters
num_epochs = 1
num_batches = 100
num_train_per_batch = 1
layers = Int64[64, 64]
activation = sigmoid
optimizers = [Flux.Adam(0.001)]

# Save paths
model_dir = joinpath(Atlas_dir, "models")
mkpath(model_dir)
save_file = "atlas-balancing-N$(N)-$(now())"

# ============================================================================
# Build Problem
# ============================================================================

println("Building Atlas subproblems...")
@time subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples,
      X_vars, U_vars, x_ref, u_ref, atlas = build_atlas_subproblems(;
    N = N,
    h = h,
    perturbation_scale = perturbation_scale,
    num_scenarios = num_scenarios,
    penalty = penalty,
)

nx = atlas.nx
nu = atlas.nu
n_perturb = length(uncertainty_samples[1])  # Number of perturbation parameters

println("Atlas state dimension: $nx")
println("Atlas control dimension: $nu")
println("Number of perturbations: $n_perturb")
println("Number of stages: $(N-1)")

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
        "layers" => layers,
        "activation" => string(activation),
        "optimizer" => string(optimizers),
        "nx" => nx,
        "nu" => nu,
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
# Initial Evaluation
# ============================================================================

println("\nEvaluating initial policy...")
Random.seed!(8788)
objective_values = [simulate_multistage(
    subproblems, state_params_in, state_params_out, 
    initial_state, DecisionRules.sample(uncertainty_samples), 
    models;
) for _ in 1:2]
best_obj = mean(objective_values)
println("Initial objective: $best_obj")

model_path = joinpath(model_dir, save_file * ".jld2")
save_control = SaveBest(best_obj, model_path)

# ============================================================================
# Hyperparameter Adjustment
# ============================================================================

adjust_hyperparameters = (iter, opt_state, num_train_per_batch) -> begin
    if iter % 500 == 0
        num_train_per_batch = min(num_train_per_batch * 2, 8)
    end
    return num_train_per_batch
end

# ============================================================================
# Training
# ============================================================================

println("\nStarting training...")
println("Epochs: $num_epochs, Batches per epoch: $num_batches")

for epoch in 1:num_epochs
    println("\n=== Epoch $epoch ===")
    _num_train_per_batch = num_train_per_batch
    
    train_multistage(
        models, 
        initial_state, 
        subproblems, 
        state_params_in, 
        state_params_out, 
        uncertainty_samples;
        num_batches = num_batches,
        num_train_per_batch = _num_train_per_batch,
        optimizer = optimizers[min(epoch, length(optimizers))],
        record_loss = (iter, model, loss, tag) -> begin
            if tag == "metrics/training_loss"
                save_control(iter, model, loss)
            end
            return record_loss(iter, model, loss, tag)
        end,
        adjust_hyperparameters = adjust_hyperparameters
    )
end

# ============================================================================
# Final Evaluation
# ============================================================================

println("\n=== Final Evaluation ===")
Random.seed!(8788)
objective_values = [simulate_multistage(
    subproblems, state_params_in, state_params_out, 
    initial_state, DecisionRules.sample(uncertainty_samples), 
    models;
) for _ in 1:10]

println("Final objective: $(mean(objective_values)) ± $(std(objective_values))")
println("Best objective during training: $(save_control.best_loss)")

# Finish logging
close(lg)

println("\nModel saved to: $model_path")
println("Training complete!")
