module DecisionRules

using JuMP
import ParametricOptInterface as POI
using Flux
using JLD2
using ChainRulesCore
import ChainRulesCore.rrule
using DiffOpt

export simulate_multistage, sample, train_multistage, simulate_states, simulate_stage, dense_multilayer_nn, variable_to_parameter, create_deficit!, 
    SaveBest, find_variables, identity

include("simulate_multistage.jl")
include("dense_multilayer_nn.jl")
include("utils.jl")

end
