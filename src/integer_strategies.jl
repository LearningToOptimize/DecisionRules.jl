"""
    AbstractIntegerStrategy

Abstract supertype for strategies that prepare a JuMP model before reading
duals or solver sensitivities.

# Arguments
This abstract type has no fields. Concrete subtypes are passed as the
`integer_strategy::AbstractIntegerStrategy` keyword to simulation and training
functions.

# Examples
```julia
simulate_multistage(
    subproblems,
    state_params_in,
    state_params_out,
    initial_state,
    uncertainties,
    policy;
    integer_strategy = FixedDiscreteIntegerStrategy(),
)
```
"""
abstract type AbstractIntegerStrategy end

"""
    NoIntegerStrategy()

Solve the model exactly as written before reading duals or sensitivities.

Use this for continuous LP, conic, or nonlinear models whose derivative
information is available directly from the solved model.

# Arguments
This type has no fields.

# Examples
```julia
strategy = NoIntegerStrategy()
```
"""
struct NoIntegerStrategy <: AbstractIntegerStrategy end

"""
    FixedDiscreteIntegerStrategy()

Solve a mixed-integer model, fix discrete variables to their incumbent values,
relax integrality, re-solve, and read duals or sensitivities from the fixed
continuous model.

If ``z^*`` is the incumbent binary/integer solution, this strategy reads
derivative-like information from the continuous problem

```math
\\min_x f(x, z^*) \\quad \\text{subject to} \\quad g(x, z^*) \\le 0.
```

The result is local to the incumbent integer assignment. It is not a
differentiable MIP method.

# Arguments
This type has no fields.

# Examples
```julia
strategy = FixedDiscreteIntegerStrategy()
```
"""
struct FixedDiscreteIntegerStrategy <: AbstractIntegerStrategy end

"""
    discrete_variables(model::JuMP.Model)

Return the binary or integer variables in `model`.

# Arguments
- `model::JuMP.Model`: model to inspect.

# Examples
```julia
vars = DecisionRules.discrete_variables(model)
```
"""
function discrete_variables(model::JuMP.Model)
    # JuMP tracks binary and integer status on variables, not in one shared list.
    return filter(JuMP.all_variables(model)) do variable
        JuMP.is_binary(variable) || JuMP.is_integer(variable)
    end
end

"""
    has_discrete_variables(model::JuMP.Model) -> Bool

Return whether `model` contains at least one binary or integer variable.

# Arguments
- `model::JuMP.Model`: model to inspect.

# Examples
```julia
if DecisionRules.has_discrete_variables(model)
    @info "MIP model"
end
```
"""
has_discrete_variables(model::JuMP.Model) = !isempty(discrete_variables(model))

"""
    _assert_successful_solve(model::JuMP.Model; context::AbstractString = "solve")

Throw an error unless `model` terminated with an accepted success status.

# Arguments
- `model::JuMP.Model`: model whose termination status is checked.
- `context::AbstractString`: human-readable phrase included in the error.

# Examples
```julia
DecisionRules._assert_successful_solve(model; context = "fixed LP solve")
```
"""
function _assert_successful_solve(model::JuMP.Model; context::AbstractString="solve")
    # Keep the accepted statuses centralized in DecisionRules.jl.
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

Run `f(model)` while `model` is in a state suitable for reading duals or
DiffOpt sensitivities.

# Arguments
- `f::Function`: callback that reads values, duals, or sensitivities.
- `model::JuMP.Model`: model to solve and inspect.
- `integer_strategy::AbstractIntegerStrategy`: strategy used to prepare models
  with binary or integer variables.

# Examples
```julia
objective = with_sensitivity_solution(model, FixedDiscreteIntegerStrategy()) do m
    JuMP.objective_value(m)
end
```
"""
function with_sensitivity_solution(
    f::Function, model::JuMP.Model, ::NoIntegerStrategy
)
    # Continuous models can be solved directly.
    optimize!(model)
    return f(model)
end

function with_sensitivity_solution(
    f::Function, model::JuMP.Model, ::FixedDiscreteIntegerStrategy
)
    # First solve the original MIP to obtain an incumbent integer assignment.
    optimize!(model)
    _assert_successful_solve(model; context="original integer solve")

    # Models without discrete variables fall back to the direct solved state.
    has_discrete_variables(model) || return f(model)

    # JuMP returns an undo callback that restores integrality and bounds.
    undo = JuMP.fix_discrete_variables(model)
    try
        # Re-solve the fixed continuous problem before reading duals.
        optimize!(model)
        _assert_successful_solve(model; context="fixed-discrete sensitivity solve")
        return f(model)
    finally
        # Always restore the original model, even when the callback fails.
        undo()
    end
end

"""
    _with_current_or_sensitivity_solution(f, model, integer_strategy)

Run `f(model)` directly for continuous models and through
[`with_sensitivity_solution`](@ref) for integer strategies.

# Arguments
- `f::Function`: callback that reads values, duals, or sensitivities.
- `model::JuMP.Model`: model to inspect.
- `integer_strategy::AbstractIntegerStrategy`: current integer strategy.

# Examples
```julia
value = DecisionRules._with_current_or_sensitivity_solution(
    m -> JuMP.objective_value(m),
    model,
    strategy,
)
```
"""
_with_current_or_sensitivity_solution(
    f::Function, model::JuMP.Model, ::NoIntegerStrategy
) = f(model)

function _with_current_or_sensitivity_solution(
    f::Function, model::JuMP.Model, integer_strategy::AbstractIntegerStrategy
)
    return with_sensitivity_solution(f, model, integer_strategy)
end

"""
    ContinuousRelaxationIntegerStrategy()

Relax all binary/integer constraints to continuous bounds (binary → [0,1]),
solve the resulting LP, and read duals in that relaxed state.

Mathematically, this replaces ``z \\in \\{0,1\\}`` or integer restrictions with
continuous bounds before solving. The derivative signal belongs to the relaxed
problem, not to the original MIP.

Compared to [`FixedDiscreteIntegerStrategy`](@ref):
- **Faster**: one LP solve instead of MIP + LP.
- **Smoother gradients**: no integer fixing means no zero-gradient dead zones.
- **Less accurate**: the LP solution may have fractional integer variables,
  so the gradient does not correspond to any feasible integer assignment.

A practical pattern is to train with `ContinuousRelaxationIntegerStrategy`
during warmup (smooth landscape for initial learning) and switch to
`FixedDiscreteIntegerStrategy` later (integer-accurate gradients for
fine-tuning).

# Arguments
This type has no fields.

# Examples
```julia
strategy = ContinuousRelaxationIntegerStrategy()
```
"""
struct ContinuousRelaxationIntegerStrategy <: AbstractIntegerStrategy end

function with_sensitivity_solution(
    f::Function, model::JuMP.Model, ::ContinuousRelaxationIntegerStrategy
)
    # Continuous models need no relaxation step.
    has_discrete_variables(model) || begin
        optimize!(model)
        return f(model)
    end

    # JuMP returns an undo callback that restores integrality after the solve.
    undo = JuMP.relax_integrality(model)
    try
        # Solve the continuous relaxation before reading duals.
        optimize!(model)
        _assert_successful_solve(model; context="continuous relaxation sensitivity solve")
        return f(model)
    finally
        # Restore integer declarations before returning control to the caller.
        undo()
    end
end

"""
    _sensitivity_forward_status(model::JuMP.Model, strategy) -> MOI.TerminationStatusCode

Return the termination status that an rrule should use for gradient fallback.

# Arguments
- `model::JuMP.Model`: model inspected after the forward pass.
- `strategy::AbstractIntegerStrategy`: integer strategy used for the solve.

# Examples
```julia
status = DecisionRules._sensitivity_forward_status(model, strategy)
```
"""
_sensitivity_forward_status(model::JuMP.Model, ::NoIntegerStrategy) =
    JuMP.termination_status(model)

function _sensitivity_forward_status(
    ::JuMP.Model, ::AbstractIntegerStrategy
)
    # Integer strategies do their own solve checks inside the sensitivity pass.
    return MOI.OPTIMAL
end
