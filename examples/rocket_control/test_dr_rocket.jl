using Flux
using DecisionRules
using Random
using Statistics
using JuMP
using Ipopt: Ipopt
using JLD2
using CSV
using DataFrames

include(joinpath(@__DIR__, "build_rocket_problem.jl"))

example_dir = @__DIR__
model_dir = joinpath(example_dir, "models")
save_file = "rocket_model"
model_path = joinpath(model_dir, save_file * ".jld2")

# load model
det_equivalent, state_params_in, state_params_out, initial_state, uncertainty_samples, x_v, x_h, x_m, u_t_max = build_rocket_problem(;
    penalty=10
)

nn = Chain(Dense(4, 32, sigmoid), LSTM(32, 32), Dense(32, 3))
opt_state = Flux.setup(Flux.Adam(), nn)
x = randn(4, 1)
y = rand(3, 1)
train_set = [(x, y)]
Flux.train!(nn, train_set, opt_state) do m, x, y
    return Flux.mse(m(x), y)
end
model_state = JLD2.load(model_path, "model_state")
Flux.loadmodel!(nn, model_state)

# simulate
example_dir = @__DIR__
save_dir = joinpath(example_dir, "dr_results")
mkpath(save_dir)

# create two files to store the results for each seed. Each row will be a different seed. 
# We will have 1 column per time period
# 1. dr_h.csv: store hight (x_h) for each time period
df = DataFrame(zeros(0, 1000), [Symbol("$i") for i in 1:1000])
df[!, :seed] = Int[]
CSV.write(joinpath(save_dir, "dr_h.csv"), df)
# 2. dr_u.csv: store thrust (u_t) for each time period
CSV.write(joinpath(save_dir, "dr_u.csv"), df)

seeds = 1:10

for seed in seeds
    Random.seed!(seed)
    simulate_multistage(
        det_equivalent,
        state_params_in,
        state_params_out,
        initial_state,
        sample(uncertainty_samples),
        nn;
    )

    df_aux = DataFrame(
        [value.(det_equivalent[:x_h]); seed]'[:, :],
        vcat([Symbol("$i") for i in 1:1000], [:seed]),
    )
    CSV.write(joinpath(save_dir, "dr_h.csv"), df_aux; append=true)
    df_aux = DataFrame(
        [value.(det_equivalent[:u_t]); seed]'[:, :],
        vcat([Symbol("$i") for i in 1:1000], [:seed]),
    )
    CSV.write(joinpath(save_dir, "dr_u.csv"), df_aux; append=true)
end
