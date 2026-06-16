using Flux
using DecisionRules
using Random
using Statistics

using JuMP
using Ipopt: Ipopt

include(joinpath(@__DIR__, "build_rocket_problem.jl"))

det_equivalent, state_params_in, state_params_out, initial_state_de, uncertainty_samples, x_v, x_h, x_m, u_t_max = build_rocket_problem(;
    penalty=1e-5
)

# NN maps [wind, v, h, m] → [target_v, target_h, target_m]
models = Chain(
    Dense(4, 32, sigmoid),
    x -> reshape(x, :, 1),
    Flux.LSTM(32 => 32),
    x -> x[:, end],
    Dense(32, 3),
)

# Pre-train

Random.seed!(8788)
objective_values = [
    simulate_multistage(
        det_equivalent,
        state_params_in,
        state_params_out,
        initial_state_de,
        sample(uncertainty_samples),
        models;
    ) for _ in 1:2
]
best_obj = mean(objective_values)

example_dir = @__DIR__

model_dir = joinpath(example_dir, "models")
save_file = "rocket_model"
mkpath(model_dir)

model_path = joinpath(model_dir, save_file * ".jld2")

save_control = SaveBest(best_obj, model_path)

train_multistage(
    models,
    initial_state_de,
    det_equivalent,
    state_params_in,
    state_params_out,
    uncertainty_samples;
    num_batches=10,
    num_train_per_batch=1,
    optimizer=Flux.Adam(),
    record_loss=(iter, model, loss, tag) -> begin
        if tag == "metrics/training_loss"
            save_control(iter, model, loss)
        end
        println("tag: $tag, Iter: $iter, Loss: $loss")
        return false
    end,
)

Random.seed!(8788)
objective_values = [
    simulate_multistage(
        det_equivalent,
        state_params_in,
        state_params_out,
        initial_state_de,
        sample(uncertainty_samples),
        models;
    ) for _ in 1:2
]
best_obj = mean(objective_values)

#####################################################################

subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples, velocities, heights, masses, u_t_max = build_rocket_subproblems(;
    penalty=1e-5
)

model_per_stage = Chain(
    Dense(4, 32, sigmoid),
    x -> reshape(x, :, 1),
    Flux.LSTM(32 => 32),
    x -> x[:, end],
    Dense(32, 3),
)

Random.seed!(8788)
objective_values = [
    simulate_multistage(
        subproblems,
        state_params_in,
        state_params_out,
        initial_state,
        sample(uncertainty_samples),
        model_per_stage;
    ) for _ in 1:2
]
best_obj = mean(objective_values)
train_multistage(
    model_per_stage,
    initial_state,
    subproblems,
    state_params_in,
    state_params_out,
    uncertainty_samples;
    num_batches=10,
    num_train_per_batch=1,
    optimizer=Flux.Adam(),
)

#####################################################################

# Finally, we plot the solution:

using Plots
using CSV
using DataFrames

example_dir = @__DIR__
dr_dir = joinpath(example_dir, "dr_results")
mkpath(dr_dir)

num_scenarios = 10
T = length(heights)

objective_values = Array{Float64}(undef, num_scenarios)
trajectories_h = Array{Float64}(undef, num_scenarios, T)

seeds = 1:num_scenarios
for (s, seed) in enumerate(seeds)
    Random.seed!(seed)
    objective_values[s] = simulate_multistage(
        subproblems,
        state_params_in,
        state_params_out,
        initial_state,
        sample(uncertainty_samples),
        model_per_stage,
    )
    trajectories_h[s, :] = value.(heights)
end

df_h = DataFrame(trajectories_h, [Symbol("$i") for i in 1:T])
df_h[!, :seed] = collect(seeds)
CSV.write(joinpath(dr_dir, "dr_h.csv"), df_h)

plt = Plots.plot(; xlabel="Time", ylabel="Height", legend=false);
for s in 1:num_scenarios
    Plots.plot!(1:T, trajectories_h[s, :]; color=:red);
end
Plots.savefig(plt, joinpath(example_dir, "dr_height_trajectories.png"))
println("Saved dr_results/dr_h.csv and dr_height_trajectories.png")
