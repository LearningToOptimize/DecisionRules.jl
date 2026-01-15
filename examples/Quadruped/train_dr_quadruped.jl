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
hidden = 64
model = Chain(
    Dense(n, hidden, relu),
    x -> reshape(x, :, 1),            # (hidden, 1)
    Flux.LSTM(hidden => hidden),      # input dim = hidden
    x -> x[:, end],
    Dense(hidden, n),
)

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
