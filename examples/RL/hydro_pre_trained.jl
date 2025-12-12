using Distributed
using Random

@everywhere rl_path = @__DIR__

@everywhere dl_path = dirname(dirname(rl_path))

@everywhere import Pkg

@everywhere Pkg.activate(dl_path)

@everywhere Pkg.instantiate()

@everywhere using DecisionRules

@everywhere Pkg.activate(rl_path)

@everywhere Pkg.instantiate()

@everywhere import QuickPOMDPs: QuickPOMDP
@everywhere import POMDPTools: ImplicitDistribution, Deterministic
@everywhere import Distributions: Normal, Uniform
@everywhere using POMDPs
@everywhere using Flux
@everywhere using Crux
@everywhere import Crux: state_space
@everywhere using POMDPs
@everywhere import POMDPTools:FunctionPolicy
@everywhere using Random
@everywhere using Distributions
@everywhere using CommonRLInterface
@everywhere using CommonRLSpaces

@everywhere using Ipopt, HSL_jll
@everywhere HydroPowerModels_dir =joinpath(dirname(rl_path), "HydroPowerModels")
@everywhere include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))

@everywhere case_name = "bolivia" # bolivia, case3
@everywhere formulation = "ACPPowerModel" # SOCWRConicPowerModel, DCPPowerModel, ACPPowerModel
@everywhere num_stages = 96 # 96, 48

import Crux: new_ep_reset!
Crux.new_ep_reset!(π::ContinuousNetwork) = begin 
    Flux.reset!(π.network)
    @info π.network.layers[1].state
end

@everywhere function build_mdp(case_name, formulation, num_stages; solver=optimizer_with_attributes(Ipopt.Optimizer, 
        "print_level" => 0,
        "hsllib" => HSL_jll.libhsl_path,
        "linear_solver" => "ma27"
    ),
    penalty=1e6,
    _objective_value = objective_value
)
    formulation_file = formulation * ".mof.json"
    Random.seed!(1234)
    subproblems, state_params_in, state_params_out, uncertainty_samples, initial_state, max_volume = build_hydropowermodels(    
        joinpath(HydroPowerModels_dir, case_name), formulation_file; num_stages=num_stages, penalty=penalty
    )

    for subproblem in subproblems
        set_optimizer(subproblem, solver)
    end

    # test solve
    JuMP.optimize!(subproblems[1])
    termination_status(subproblems[1])

    Random.seed!(1234)
    uncertainty_sample = DecisionRules.sample(uncertainty_samples)[1]
    rain_state = [uncertainty_sample[i][2] for i in 1:length(uncertainty_sample)]

    num_a = length(state_params_in[1])
    mdp = QuickPOMDP(
        actions = Box(zeros(num_a), max_volume),
        obstype = Array{Float64,1},
        # discount = 0.995,

        gen = function (state, state_out, rng)
            state_in, rain_state, j = state[1:num_a], state[num_a+1:end-1], ceil(Int, state[end])
            rain = if j == num_stages
                DecisionRules.sample(uncertainty_samples[j])
            else
                DecisionRules.sample(uncertainty_samples[j+1])
            end
            rain = [rain[i][2] for i in 1:length(rain)]
            uncertainty_sample = [(uncertainty_samples[j][i][1], rain_state[i]) for i in 1:length(rain)]
            simulate_stage(subproblems[j], state_params_in[j], state_params_out[j], uncertainty_sample, state_in, state_out)
            r = _objective_value(subproblems[j])
            next_volume = DecisionRules.get_next_state(subproblems[j], state_params_in[j], state_params_out[j], state_in, state_out)
            @info "Stage t=$j" sum(state_in) sum(rain_state) sum(state_out) sum(next_volume) r
            sp = [next_volume; rain; j+1]
            o = rain #+ next_volume - state_out
            return (sp=sp, o=o, r=-r)
        end,

        initialstate = Deterministic([initial_state; rain_state; 1.0]),
        initialobs = (s) -> Deterministic(rain_state), # initial_state+rain_state
        isterminal = s -> s[end] > num_stages
    )
    return mdp, num_a, max_volume
end

# build the MDPs
@everywhere mdp, num_a, max_volume = build_mdp(case_name, formulation, num_stages;
    _objective_value=DecisionRules.get_objective_no_target_deficit
)
@everywhere S = state_space(mdp)

# Load TS-GDR model
using JLD2
dense = LSTM 
activation = sigmoid
layers = Int64[32, 32]
model = dense_multilayer_nn(1, num_a, num_a, layers; activation=activation, dense=dense)
model_dir = joinpath(HydroPowerModels_dir, case_name, "DCPPowerModel", "models")
model_file = readdir(model_dir, join=true)[end-1] # edit this for a specific model
opt_state = Flux.setup(Flux.Adam(0.01), model)
x = randn(num_a, 1)
y = rand(num_a, 1)
train_set = [(x, y)]
Flux.train!(model, train_set, opt_state) do m, x, y
    Flux.mse(m(x), y)
end
models = model
model_state = JLD2.load(model_file, "model_state")
Flux.loadmodel!(model, model_state)

# Sample expert data
Random.seed!(8788)
s = Sampler(mdp, ContinuousNetwork(model, num_a), max_steps=96, required_columns=[:t])
data = steps!(s, Nsteps=97)

####################### This code is for imitation learning #######################
# Save the expert data
data[:expert_val] = ones(Float32, 1, 10000)
data = ExperienceBuffer(data)
BSON.@save "./expert.bson" data

# Load the expert data and train the policy using Behavioral Cloning
expert_trajectories = BSON.load("./expert.bson")[:data]

𝒮_bc = BC(π=ActorCritic(SG(), DoubleNetwork(QSA(), QSA())), #ActorCritic(SG(), V()), 
          𝒟_demo=expert_trajectories, 
          S=S,
          opt=(epochs=10000, batch_size=1024), 
          log=(period=500,),
          max_steps=1000)
solve(𝒮_bc, mdp)


####################### The rest of this code is exploratory #######################
# This code is used to refine the policy with RL algorithms
# These policies only observe the uncertainty and not the rest of the state
####################################################################################

# Define the networks we will use
# @everywhere QSA() = ContinuousNetwork(Chain(Dense(34, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
# @everywhere V() = ContinuousNetwork(Chain(Dense(2*num_a+1, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
# @everywhere A() = ContinuousNetwork(Chain(Dense(2*num_a+1, 64, relu), Dense(64, 64, relu), Dense(64, num_a, tanh)))
# @everywhere SG() = SquashedGaussianPolicy(ContinuousNetwork(Chain(Dense(2*num_a+1, 64, relu), Dense(64, 64, relu), Dense(64, num_a, tanh))), zeros(Float32, 1), 1f0)

# Solve with REINFORCE
@everywhere 𝒮_reinforce = REINFORCE(π=GaussianPolicy(ContinuousNetwork(model, num_a), zeros(Float32, 1)), S=S, N=2000, ΔN=10, a_opt=(batch_size=1,))
@time π_reinforce = solve(𝒮_reinforce, mdp)

# Solve with PPO 
@everywhere 𝒮_ppo = PPO(π=ActorCritic(GaussianPolicy(ContinuousNetwork(model, num_a), zeros(Float32, 1)), V()), S=S, N=10000, ΔN=10, a_opt=(batch_size=10,))
@time π_ppo = solve(𝒮_ppo, mdp)

# Off-policy settings
@everywhere off_policy = (S=S,
              ΔN=10,
              N=1000,
              buffer_size=Int(200),
              buffer_init=200,
              c_opt=(batch_size=1, optimizer=Adam(1e-3)),
              a_opt=(batch_size=1, optimizer=Adam(1e-3)),
              π_explore=GaussianNoiseExplorationPolicy(0.5f0, a_min=[0.0], a_max=[1.0])
)
              
# # Solver with DDPG
@everywhere QSA() = ContinuousNetwork(Chain(Dense(22, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
@everywhere 𝒮_ddpg = DDPG(;π=ActorCritic(ContinuousNetwork(model, num_a), QSA()), off_policy...)
@time π_ddpg = solve(𝒮_ddpg, mdp)

# # Solve with TD3
# @everywhere 𝒮_td3 = TD3(;π=ActorCritic(A(), DoubleNetwork(QSA(), QSA())), off_policy...)
# @time π_td3 = solve(𝒮_td3, mdp)

# # Solve with SAC
@everywhere 𝒮_sac = SAC(;π=ActorCritic(GaussianPolicy(ContinuousNetwork(model, num_a), zeros(Float32, 1)), DoubleNetwork(QSA(), QSA())), off_policy...)
@time π_sac = solve(𝒮_sac, mdp)

# pmap(solve, [𝒮_reinforce, 𝒮_ppo, 𝒮_ddpg, 𝒮_td3, 𝒮_sac], [build_mdp(case_name, formulation, num_stages)[1] for _ in 1:5])

####################### This code is for Plotting #######################

using Plots
function plot_learning_mod(input; 
        title = "",
        ylabel = "Undiscounted Return (Costs)",  
        xlabel = "Training Steps", 
        values = :undiscounted_return, 
        labels = :default,
        legend = :bottomright,
        font = :palatino,
        p = plot(), 
        vertical_lines = [],
        vline_range = (0, 1), 
        thick_every = 1,
        yscale = :log10,
        xs=nothing
    )
    dirs = directories(input)
    
    N = length(dirs)
    values isa Symbol && (values = fill(values, N))
    if labels == :default
        labels = N == 1 ? [""] : ["Task $i" for i=1:N]
    end 
    
    # Plot the vertical lines (usually for multitask learning or to designate a point on the curve)
    for i = 1:length(vertical_lines)
        plot!(p, [vertical_lines[i], vertical_lines[i]], [vline_range...], color=:black, linewidth = i % thick_every == 0 ? 3 : 1, label = "")
    end
    
    # Plot the learning curves
    plot!(p, ylabel = ylabel, xlabel = xlabel, legend = legend, title = title, fontfamily = font, yscale=yscale)
    for i in 1:length(dirs)
        x, y = readtb(dirs[i], values[i])
        if xs != nothing
            x = x[xs]
            y = y[xs]
        end
        plot!(p, x, -y, alpha = 0.3, color=i, label = "", yscale=yscale)
        plot!(p, x, smooth(-y), color=i, label = labels[i], linewidth =2, yscale=yscale)
    end
    p
end

function Base.push!(
    history::Crux.ValueHistories.History{I,V},
    iteration::I,
    value::V) where {I,V}
    lastiter = history.lastiter
    # iteration > lastiter || throw(ArgumentError("Iterations must increase over time"))
    history.lastiter = iteration
    push!(history.iterations, iteration)
    push!(history.values, value)
    value
end

p = plot_learning_mod(
    [
        𝒮_reinforce, 
        𝒮_ppo, 
        𝒮_ddpg,
        𝒮_td3,
        𝒮_sac,
        𝒮_bc,
    ], 
    title="LTHD Training Curves", 
    labels=[
        "REINFORCE", 
        "PPO", 
        "DDPG",
        "TD3",
        "SAC",
        "BC"
    ], legend=:outertopright, yscale=:log10,
)

using CSV, DataFrames
tsgdr_df = CSV.read("../../tsgdr_train.csv", DataFrame)

plot!(p, tsgdr_df[:,"Step"], tsgdr_df[:,"bolivia-ACPPowerModel-h96-2024-05-18T17:06:09.485 - metrics/training_loss"], label="TS-GDR", linewidth=2, yscale=:log10)

Crux.savefig("./hydro_benchmark_warm.pdf")
