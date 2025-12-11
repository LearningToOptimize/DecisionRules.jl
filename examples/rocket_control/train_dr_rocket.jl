using Flux
using DecisionRules
using Random
using Statistics

using JuMP
import Ipopt, HSL_jll

include("./examples/rocket_control/build_rocket_problem.jl")

det_equivalent, state_params_in, state_params_out, final_state, uncertainty_samples, x_v, x_h, x_m, u_t_max = build_rocket_problem(; penalty=1e-5)

# Create ML policy to solve the problem
models = Chain(Dense(1, 32, sigmoid), 
    x -> reshape(x, :, 1), 
    Flux.LSTM(32 => 32), 
    x -> x[:, end],
    Dense(32, 1, (x) -> sigmoid(x) .* u_t_max)
)

# Pre-train

Random.seed!(8788)
objective_values = [simulate_multistage(
    det_equivalent, state_params_in, state_params_out, 
    final_state, sample(uncertainty_samples), 
    models;
) for _ in 1:2]
best_obj = mean(objective_values)

example_dir = joinpath(pwd(), "examples", "rocket_control") #dirname(@__FILE__)

model_dir = joinpath(example_dir, "models")
save_file = "rocket_model"
mkpath(model_dir)

model_path = joinpath(model_dir, save_file * ".jld2")

save_control = SaveBest(best_obj, model_path, 0.003)

train_multistage(models, final_state, det_equivalent, state_params_in, state_params_out, uncertainty_samples; 
    num_batches=10,
    num_train_per_batch=1,
    optimizer=Flux.Adam(),
    record_loss= (iter, model, loss, tag) -> begin
        if tag == "metrics/training_loss"
            save_control(iter, model, loss)
        end
        println("tag: $tag, Iter: $iter, Loss: $loss")
        return false
    end,
)

Random.seed!(8788)
objective_values = [simulate_multistage(
    det_equivalent, state_params_in, state_params_out, 
    final_state, sample(uncertainty_samples), 
    models;
) for _ in 1:2]
best_obj = mean(objective_values)-


#####################################################################


# Finally, we plot the solution:

using Plots
using CSV
using DataFrames
dr_dir = joinpath(example_dir, "dr_results")
mkpath(dr_dir)

num_scenarios = 10

objective_values = Array{Float64}(undef, num_scenarios)
trajectories_h = Array{Float64}(undef, num_scenarios, length(x_h))

for s = 1:num_scenarios
    wind_sample = sample(uncertainty_samples)
    objective_values[s] = simulate_multistage(
        det_equivalent, state_params_in, state_params_out, 
        final_state, sample(uncertainty_samples), 
        models
    )
    
    trajectories_h[s, :] = value.(x_h)
end

plt = Plots.plot(; xlabel="Time", ylabel="Height", legend=false);
for s = 1:num_scenarios
    Plots.plot!(1:1000, trajectories_h[s, :], color=:red);
end