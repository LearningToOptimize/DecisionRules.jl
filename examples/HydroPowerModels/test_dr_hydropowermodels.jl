using Statistics
using Random
using Flux
using DecisionRules
using Gurobi
using MosekTools
using Ipopt, HSL_jll
import MathOptSymbolicAD
using JLD2
using HydroPowerModels
using DiffOpt

HydroPowerModels_dir = dirname(@__FILE__)
include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))

# Functions

function non_ensurance(x_out, x_in, uncertainty, max_volume)
    return x_out
end

# Parameters
case_name = "bolivia" # bolivia, case3
formulation = "DCPPowerModel" # SOCWRConicPowerModel, DCPPowerModel, ACPPowerModel
num_stages = 96 # 96, 48
model_dir = joinpath(HydroPowerModels_dir, case_name, formulation, "models")
model_file = readdir(model_dir, join=true)[end] # edit this for a specific model
save_name = split(split(model_file, "/")[end], ".")[1]
formulation_file = formulation * ".mof.json"
dense = Dense # RNN, Dense
activation = DecisionRules.identity # tanh, DecisionRules.identity, relu, sigmoid
layers = Int64[32, 32] # Int64[8, 8], Int64[]
num_models = num_stages # 1, num_stages
ensure_feasibility = non_ensurance # ensure_feasibility_double_softplus
optimizer = Flux.Adam(0.01)

data = HydroPowerModels.parse_folder(joinpath(HydroPowerModels_dir, case_name))[1];
HydroPowerModels.gather_useful_info!(data)
# Build MSP

subproblems, state_params_in, state_params_out, uncertainty_samples, initial_state, max_volume = build_hydropowermodels(    
    joinpath(HydroPowerModels_dir, case_name), formulation_file; num_stages=num_stages
)

det_equivalent = DiffOpt.diff_model(optimizer_with_attributes(Ipopt.Optimizer, 
    "print_level" => 0,
    "hsllib" => HSL_jll.libhsl_path,
    "linear_solver" => "ma27"
))

det_equivalent, uncertainty_samples = DecisionRules.deterministic_equivalent!(det_equivalent, subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples)

num_hydro = length(initial_state)

# Build Model
models = dense_multilayer_nn(num_models, num_hydro, num_hydro, layers; activation=activation, dense=dense)
model = if num_models > 1
    DecisionRules.make_single_network(models, num_hydro)
else
    models
end
# model = Chain(Dense(num_hydro, 32, sigmoid), LSTM(32, 32), Dense(32, num_hydro))
opt_state = Flux.setup(optimizer, model)
x = randn(num_hydro, 1)
y = rand(num_hydro, 1)
train_set = [(x, y)]
Flux.train!(model, train_set, opt_state) do m, x, y
    Flux.mse(m(x), y)
end
models = model
model_state = JLD2.load(model_file, "model_state")
Flux.loadmodel!(model, model_state)

Random.seed!(1221)
num_samples = 100
objective_values = Vector{Float64}(undef, num_samples)
states = Vector{Any}(undef, num_samples)
inflows = Array{Float64,3}(undef, num_samples, num_hydro, num_stages)
record_variables_names = ["0_pg", "norm_deficit"]
record_variables = Dict{String, Any}()
record = Dict{String, Array{Float64,3}}()
for _var in record_variables_names
    num_vars = length(find_variables(det_equivalent, [_var; r"#1$"]))
    record[_var] = Array{Float64,3}(undef, num_samples, num_vars, num_stages)
    record_variables[_var] = [find_variables(det_equivalent, [_var; Regex("#$i\$")]) for i in 1:num_stages]
end
for i in 1:num_samples
    Flux.reset!(models)
    uncertainty_s = sample(uncertainty_samples)
    for j in 1:num_hydro, t in 1:num_stages
        inflow_var = collect(keys(uncertainty_s[t]))
        inflow_var = inflow_var[findfirst(x -> occursin("_inflow[$j]", JuMP.name(x)), inflow_var)]
        inflows[i, j, t] = uncertainty_s[t][inflow_var]
    end
    simulate_multistage(
        det_equivalent, state_params_in, state_params_out, 
        initial_state, uncertainty_s, 
        models;
        ensure_feasibility=(x_out, x_in, _sa) -> ensure_feasibility(x_out, x_in, _sa, max_volume)
    )
    objective_values[i] = DecisionRules.get_objective_no_target_deficit(det_equivalent)
    for _var in record_variables_names
        num_vars = length(find_variables(det_equivalent, [_var; r"#1$"]))
        for j in 1:num_vars, t in 1:num_stages
            record[_var][i, j, t] = value(record_variables[_var][t][j])
        end
    end
    states[i] = Vector{Vector{Float64}}(undef, num_hydro)
    for j in 1:num_hydro
        states[i][j] = Vector{Float64}(undef, num_stages+1)
        states[i][j][1] = initial_state[j]
        for t in 1:num_stages
            states[i][j][t+1] = value(state_params_out[t][j][2])
        end
    end
end

# Plot Volumes

using Plots
using Statistics
using DataFrames
using CSV

plt = plot(1:num_stages+1, [sum([states[1][j][t] for j in 1:num_hydro]) for t in 1:num_stages+1], legend=false, xlabel="Stage", ylabel="Volume (Hm3)", title="$(case_name)-$(formulation)");
for i in 2:num_samples
    plot!(plt, 1:num_stages+1, [sum([states[i][j][t] for j in 1:num_hydro]) for t in 1:num_stages+1]);
end
savefig(plt, joinpath(HydroPowerModels_dir, case_name, formulation, save_name * "Volume.png"))

# Plot Mean Volume
volume_to_mw(volume, stage_hours; k = 0.0036) = volume / (k * stage_hours)

plt = plot(1:num_stages+1, [mean(sum([states[i][j][t] for j in 1:num_hydro]) for i in 1:num_samples) for t in 1:num_stages+1], xlabel="Stage", ylabel="Volume (Hm3)", label="Mean Volume", title="$(case_name)-$(formulation)");
savefig(plt, joinpath(HydroPowerModels_dir, case_name, formulation, save_name * "MeanVolume.png"))

df = DataFrame(ML_Rule=[mean(sum([volume_to_mw(states[i][j][t], 1) for j in 1:num_hydro]) for i in 1:num_samples) for t in 2:num_stages+1])

df = CSV.read(joinpath(HydroPowerModels_dir, case_name, formulation, "MeanVolume.csv"), DataFrame; header=true)
df[!, "TS-LDR"] = [mean(sum([volume_to_mw(states[i][j][t], 1) for j in 1:num_hydro]) for i in 1:num_samples) for t in 2:num_stages+1]

CSV.write(joinpath(HydroPowerModels_dir, case_name, formulation, "MeanVolume.csv"), df)

# Plot Mean Inflows

plt = plot(1:num_stages, [mean(sum(inflows[i, j, t] for j in 1:num_hydro) for i in 1:num_samples) for t in 1:num_stages], xlabel="Stage", ylabel="Inflow (Hm3)", label="Mean Inflow", title="$(case_name)-$(formulation)");
savefig(plt, joinpath(HydroPowerModels_dir, case_name, formulation, save_name * "MeanInflow.png"))

# Plot Generation
num_gen = size(record["0_pg"],2)
hydro_idx = HydroPowerModels.idx_hydro(data)

thermal_gen = [mean(sum(record["0_pg"][i, j, t] * 100 for j in 1:num_gen if !(j in hydro_idx)) for i in 1:num_samples) for t in 1:num_stages]

plt = plot(1:num_stages, thermal_gen, xlabel="Stage", ylabel="Generation (MW)", label="Mean Generation", title="$(case_name)-$(formulation)");
savefig(plt, joinpath(HydroPowerModels_dir, case_name, formulation, save_name * "MeanGeneration.png"))

df = DataFrame(ML_Rule=thermal_gen)

df = CSV.read(joinpath(HydroPowerModels_dir, case_name, formulation, "MeanGeneration.csv"), DataFrame; header=true)
df[!, "TS-LDR"] = thermal_gen

CSV.write(joinpath(HydroPowerModels_dir, case_name, formulation, "MeanGeneration.csv"), df)

# objective
df = CSV.read(joinpath(HydroPowerModels_dir, case_name, formulation, "costs.csv"), DataFrame; header=true)
df[!, "TS-LDR"] = objective_values

mean((df[!, "SDDP_SOC"] .- df[!, "TS-DDR"]) * 100 ./ df[!, "TS-DDR"])

CSV.write(joinpath(HydroPowerModels_dir, case_name, formulation, "costs.csv"), df)
# Plot deficit

# plt = plot(1:num_stages, [mean(sum(record["norm_deficit"][i, j, t] for j in 1:num_hydro) for i in 1:num_samples) for t in 1:num_stages], xlabel="Stage", ylabel="Deficit (Hm3)", label="Mean Deficit", title="$(case_name)-$(formulation)");
# savefig(plt, joinpath(HydroPowerModels_dir, case_name, formulation, save_name * "MeanDefict.png"))

