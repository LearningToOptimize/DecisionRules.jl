"""
    AbstractIntegerStrategy

Extension point for preparing models with discrete variables before reading duals
or solver sensitivities.
"""
abstract type AbstractIntegerStrategy end

"""
    NoIntegerStrategy()

Default integer strategy. Solves the model exactly as-is and preserves the
historical continuous-model behavior.
"""
struct NoIntegerStrategy <: AbstractIntegerStrategy end

"""
    FixedDiscreteIntegerStrategy()

Solve the original model, fix binary/integer variables to their incumbent values,
relax integrality, re-solve the fixed continuous model, and read duals or
sensitivities in that fixed-incumbent continuous state.

The returned derivative-like information is local to the incumbent integer
assignment and should be interpreted as a postprocessing surrogate, not as a full
differentiable MIP method.
"""
struct FixedDiscreteIntegerStrategy <: AbstractIntegerStrategy end

"""
    discrete_variables(model::JuMP.Model)

Return all binary or integer variables in `model`.
"""
function discrete_variables(model::JuMP.Model)
    return filter(JuMP.all_variables(model)) do variable
        JuMP.is_binary(variable) || JuMP.is_integer(variable)
    end
end

has_discrete_variables(model::JuMP.Model) = !isempty(discrete_variables(model))

function _assert_successful_solve(model::JuMP.Model; context::AbstractString="solve")
    status = JuMP.termination_status(model)
    status in _SUCCESSFUL_TERM_STATUSES && return status
    throw(
        ErrorException(
            "$context failed with termination status $status; expected one of " *
            "$(join(string.(_SUCCESSFUL_TERM_STATUSES), ", ")).",
        ),
    )
end

"""
    with_sensitivity_solution(f, model, integer_strategy)

Run `f(model)` while `model` is in a state where duals or DiffOpt sensitivities
can be read. Integer strategies that temporarily mutate the model must restore it
before returning, including when `f` throws.
"""
function with_sensitivity_solution(
    f::Function, model::JuMP.Model, ::NoIntegerStrategy
)
    optimize!(model)
    return f(model)
end

function with_sensitivity_solution(
    f::Function, model::JuMP.Model, ::FixedDiscreteIntegerStrategy
)
    optimize!(model)
    _assert_successful_solve(model; context="original integer solve")

    has_discrete_variables(model) || return f(model)

    undo = JuMP.fix_discrete_variables(model)
    try
        optimize!(model)
        _assert_successful_solve(model; context="fixed-discrete sensitivity solve")
        return f(model)
    finally
        undo()
    end
end

_with_current_or_sensitivity_solution(
    f::Function, model::JuMP.Model, ::NoIntegerStrategy
) = f(model)

function _with_current_or_sensitivity_solution(
    f::Function, model::JuMP.Model, integer_strategy::AbstractIntegerStrategy
)
    return with_sensitivity_solution(f, model, integer_strategy)
end

_sensitivity_forward_status(model::JuMP.Model, ::NoIntegerStrategy) =
    JuMP.termination_status(model)

function _sensitivity_forward_status(
    ::JuMP.Model, ::AbstractIntegerStrategy
)
    return MOI.OPTIMAL
end
