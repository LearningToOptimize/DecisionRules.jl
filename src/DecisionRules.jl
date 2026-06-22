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
using Statistics: mean

export simulate_multistage,
    sample,
    train_multistage,
    simulate_states,
    simulate_stage,
    dense_multilayer_nn,
    variable_to_parameter,
    create_deficit!,
    default_annealed_schedule,
    SampleLog,
    default_record,
    RolloutEvaluation,
    SaveBest,
    find_variables,
    compute_parameter_dual,
    AbstractIntegerStrategy,
    NoIntegerStrategy,
    FixedDiscreteIntegerStrategy,
    ContinuousRelaxationIntegerStrategy,
    StallingCriterium,
    policy_input_dim,
    normalize_recur_state,
    StateConditionedPolicy,
    state_conditioned_policy,
    materialize_tangent,
    # Score-function gradient mixing
    ScoreFunctionConfig,
    ScoreFunctionSchedule,
    sf_params,
    # Multiple shooting exports
    train_multiple_shooting,
    setup_shooting_windows,
    solve_window,
    predict_window_targets,
    simulate_multiple_shooting,
    WindowData,
    # Gradient fallback
    AbstractGradientFallback,
    ZeroGradientFallback,
    ErrorGradientFallback

@doc raw"""
    AbstractGradientFallback

Abstract type governing what happens when a solver or differentiation error
occurs during training.

DecisionRules ships two concrete subtypes:

| Type | Behavior |
|------|----------|
| [`ZeroGradientFallback`](@ref) | Log a warning, return zero gradients, continue training |
| [`ErrorGradientFallback`](@ref) | Re-throw the error (useful in tests) |

## Extending

Implement your own subtype to customize recovery:

```julia
struct MyFallback <: DecisionRules.AbstractGradientFallback end

function DecisionRules.handle_gradient_error(::MyFallback, e, n_state_in, n_state_out)
    # e is the caught exception
    # Return a tuple of cotangents (same shape as the rrule pullback) or rethrow
    @error "Custom handler" exception=e
    return DecisionRules._zero_cotangents(n_state_in, n_state_out)
end

function DecisionRules.handle_training_error(::MyFallback, e, iter)
    # Return true to skip this iteration, false to rethrow
    @error "Custom training handler" exception=e
    return true  # skip
end

function DecisionRules.handle_rollout_error(::MyFallback, e, iter)
    # Return true to skip this scenario, false to rethrow
    return true
end
```

Then pass `gradient_fallback=MyFallback()` to [`train_multistage`](@ref) or
[`train_multiple_shooting`](@ref).
"""
abstract type AbstractGradientFallback end

"""
    ZeroGradientFallback()

Default fallback: log a warning and return zero gradients when the solver or
DiffOpt differentiation fails. Training continues with a skipped update for
that iteration.
"""
struct ZeroGradientFallback <: AbstractGradientFallback end

"""
    ErrorGradientFallback()

Strict fallback: re-throw any solver or differentiation error. Use this in
tests to ensure that controlled problems never silently produce zero gradients.
"""
struct ErrorGradientFallback <: AbstractGradientFallback end

_zero_cotangents(n_in, n_out) = (
    NoTangent(), NoTangent(), NoTangent(), NoTangent(),
    zeros(n_in), zeros(n_out), NoTangent(),
)

function handle_gradient_error(::ZeroGradientFallback, e, n_state_in, n_state_out)
    @warn "get_next_state pullback failed — returning zero gradients" exception=(e, catch_backtrace())
    return _zero_cotangents(n_state_in, n_state_out)
end

function handle_gradient_error(::ErrorGradientFallback, e, n_state_in, n_state_out)
    rethrow(e)
end

function handle_training_error(::ZeroGradientFallback, e, iter)
    @warn "Gradient computation failed at iter $iter — skipping update" exception=(e, catch_backtrace())
    return true
end

function handle_training_error(::ErrorGradientFallback, e, iter)
    rethrow(e)
end

function handle_rollout_error(::ZeroGradientFallback, e, iter)
    @warn "Rollout scenario failed at iter $iter — skipping" exception=(e, catch_backtrace())
    return true
end

function handle_rollout_error(::ErrorGradientFallback, e, iter)
    rethrow(e)
end

"""
    STRICT_GRADIENTS

Global flag controlling gradient fallback behavior in rrules.

When `false` (default), rrule pullbacks return zero gradients with a warning
when the solver terminates unsuccessfully — this keeps training alive when a
few samples hit numerical trouble.

When `true`, the same situation throws an error instead. Enable this in tests
to verify that controlled test cases never silently fall through to zero
gradients:

    DecisionRules.STRICT_GRADIENTS[] = true

!!! note
    This flag controls the **rrule-level** fallback for bad solver status.
    For the **training-loop-level** fallback (DiffOpt assertion errors, etc.),
    use the `gradient_fallback` keyword in [`train_multistage`](@ref) and
    [`train_multiple_shooting`](@ref).
"""
const STRICT_GRADIENTS = Ref(false)

const _DEFAULT_GRADIENT_FALLBACK = ZeroGradientFallback()

const _SUCCESSFUL_TERM_STATUSES = (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED)

include("integer_strategies.jl")
include("parameter_duals.jl")
include("score_function.jl")
include("simulate_multistage.jl")
include("dense_multilayer_nn.jl")
include("utils.jl")
include("multiple_shooting.jl")

end
