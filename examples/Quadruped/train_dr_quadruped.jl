using Flux
using DecisionRules
using Random
using Statistics

using JuMP
import Ipopt, HSL_jll
using DiffOpt

include(joinpath(@__DIR__, "build_quadruped_subproblems.jl"))

# Build stage-wise subproblems and data
subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples =
    build_quadruped_subproblems(;N=5,
    solver = optimizer_with_attributes(Ipopt.Optimizer,
        # "print_level" => 0,
        "linear_solver" => "ma97",
        "hessian_approximation" => "limited-memory",
        "mu_target" => 1e-8,
    )
)

n = length(initial_state)

# Policy: maps (state) -> desired next state (same dimension)
# Ensure LSTM input dim matches the preceding Dense output (64)
hidden = 128
model = Chain(
    Dense(n, hidden, relu),
    Dense(hidden, hidden, relu),
    Dense(hidden, n),
)

# Try to load warm-start weights from sampling script
using JLD2
warmstart_path = joinpath(dirname(@__DIR__), "Dojo", "quadruped_warmstart_model.jld2")
if isfile(warmstart_path)
    println("Loading warm-start model from: $warmstart_path")
    try
        model_state = JLD2.load(warmstart_path, "model_state")
        Flux.loadmodel!(model, model_state)
        println("✓ Warm-start weights loaded successfully!")
    catch e
        println("⚠ Failed to load warm-start: $e")
        println("  Continuing with random initialization")
    end
else
    println("No warm-start model found at: $warmstart_path")
    println("Using random initialization (run sampling.jl first to create warm-start)")
end

# Quick baseline evaluation
Random.seed!(42)
objective_values = [simulate_multistage(
    subproblems, state_params_in, state_params_out,
    initial_state, sample(uncertainty_samples),
    model,
) for _ in 1:2]
best_obj = mean(objective_values)
println("Initial mean objective: $best_obj")

# Train with DecisionRules helper (differentiates through DiffOpt subproblems)
train_multistage(model, initial_state, subproblems, state_params_in, state_params_out, uncertainty_samples;
    num_batches = 5,
    num_train_per_batch = 2,
    optimizer = Flux.Adam(),
    record_loss = (iter, model, loss, tag) -> begin
        println("tag: $tag, Iter: $iter, Loss: $loss")
        false
    end,
)

# Post-train eval
Random.seed!(42)
objective_values = [simulate_multistage(
    subproblems, state_params_in, state_params_out,
    initial_state, sample(uncertainty_samples),
    model,
) for _ in 1:2]
println("Post-training mean objective: $(mean(objective_values))")
