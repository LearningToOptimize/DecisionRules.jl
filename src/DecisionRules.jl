module DecisionRules

using JuMP
import MathOptInterface as MOI
using Flux
using JLD2
using ChainRules: @ignore_derivatives
using ChainRulesCore
import ChainRulesCore.rrule
using DiffOpt
using Logging

export simulate_multistage, sample, train_multistage, simulate_states, simulate_stage, dense_multilayer_nn, variable_to_parameter, create_deficit!, 
    SaveBest, find_variables, compute_parameter_dual, StallingCriterium, policy_input_dim, 
    StateConditionedPolicy, state_conditioned_policy, materialize_tangent,
    # Multiple shooting exports
    train_multiple_shooting, setup_shooting_windows, solve_window, predict_window_targets,
    simulate_multiple_shooting, WindowData

include("parameter_duals.jl")
include("simulate_multistage.jl")
include("dense_multilayer_nn.jl")
include("utils.jl")
include("multiple_shooting.jl")

end
