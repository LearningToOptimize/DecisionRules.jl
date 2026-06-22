"""
    simulate_states(initial_state, uncertainties, decision_rule) -> Vector{Vector}

Roll out `decision_rule` over `uncertainties` to produce a target-state trajectory.
At each stage the policy receives `[uncertainty..., previous_state...]` and outputs
the next target state.  Returns a length-`(T+1)` vector of states starting with
`initial_state`.
"""
function simulate_states(
    initial_state::Vector{T}, uncertainties, decision_rule::F;
) where {F,T<:Real}
    num_stages = length(uncertainties)
    states = Vector{Vector{T}}(undef, num_stages + 1)
    states[1] = initial_state
    for stage in 1:num_stages
        uncertainties_stage = [
            uncertainties[stage][i][2] for i in 1:length(uncertainties[stage])
        ]
        # Input: [uncertainty, previous_predicted_state]
        # For stage 1, previous_predicted_state = initial_state
        # For stage t > 1, previous_predicted_state = states[t] (output from previous stage)
        prev_state = states[stage]
        states[stage + 1] = decision_rule(vcat(uncertainties_stage, prev_state))
    end
    return states
end

function simulate_states(
    initial_state::Vector{T}, uncertainties, decision_rules::Vector{F};
) where {F,T<:Real}
    num_stages = length(uncertainties)
    states = Vector{Vector{T}}(undef, num_stages + 1)
    states[1] = initial_state
    for stage in 1:num_stages
        uncertainties_stage = [
            uncertainties[stage][i][2] for i in 1:length(uncertainties[stage])
        ]
        decision_rule = decision_rules[stage]
        # Input: [uncertainty, previous_predicted_state]
        prev_state = states[stage]
        states[stage + 1] = decision_rule(vcat(uncertainties_stage, prev_state))
    end
    return states
end

"""
    simulate_stage(subproblem, state_param_in, state_param_out, uncertainty,
                   state_in, state_out_target) -> Float64

Set parameter values on `subproblem` (incoming state, outgoing target, uncertainty),
solve it, and return the objective value.  Used as the inner solve in single-shooting
rollouts (Extension §2, Eq. 2.1).
"""
function simulate_stage(
    subproblem::JuMP.Model,
    state_param_in::Vector{Any},
    state_param_out::Vector{Tuple{Any,VariableRef}},
    uncertainty::Vector{Tuple{VariableRef,T}},
    state_in::Vector{Z},
    state_out_target::Vector{V},
    ;
    integer_strategy::AbstractIntegerStrategy=NoIntegerStrategy(),
) where {T<:Real,V<:Real,Z<:Real}
    return _simulate_stage(
        subproblem,
        state_param_in,
        state_param_out,
        uncertainty,
        state_in,
        state_out_target,
        integer_strategy,
    )
end

function _set_stage_parameters!(
    state_param_in,
    state_param_out,
    uncertainty,
    state_in,
    state_out_target,
)
    # Update state parameters
    for (i, state_var) in enumerate(state_param_in)
        set_parameter_value(state_var, state_in[i])
    end

    # Update uncertainty
    for (uncertainty_param, uncertainty_value) in uncertainty
        set_parameter_value(uncertainty_param, uncertainty_value)
    end

    # Update state parameters out
    for i in 1:length(state_param_out)
        state_var = state_param_out[i][1]
        set_parameter_value(state_var, state_out_target[i])
    end
    return nothing
end

function _simulate_stage(
    subproblem::JuMP.Model,
    state_param_in,
    state_param_out,
    uncertainty,
    state_in,
    state_out_target,
    integer_strategy::AbstractIntegerStrategy,
)
    _set_stage_parameters!(
        state_param_in, state_param_out, uncertainty, state_in, state_out_target
    )

    return with_sensitivity_solution(subproblem, integer_strategy) do sensitivity_model
        return objective_value(sensitivity_model)
    end
end

function _simulate_stage_with_parameter_duals(
    subproblem,
    state_param_in,
    state_param_out,
    uncertainty,
    state_in,
    state_out_target,
    integer_strategy::AbstractIntegerStrategy,
)
    _set_stage_parameters!(
        state_param_in, state_param_out, uncertainty, state_in, state_out_target
    )
    return with_sensitivity_solution(subproblem, integer_strategy) do sensitivity_model
        objective = objective_value(sensitivity_model)
        d_state_in = pdual.(state_param_in)
        d_state_out_target = pdual.([s[1] for s in state_param_out])
        return objective, d_state_in, d_state_out_target
    end
end

"""
    get_next_state(subproblem, state_param_in, state_param_out, state_in,
                   state_out_target) -> Vector

Return the realized state from the most recent solve of `subproblem` by reading
the values of the realized-state variables in `state_param_out`.
"""
function get_next_state(
    subproblem::JuMP.Model,
    state_param_in::Vector{Any},
    state_param_out::Vector{Tuple{Any,VariableRef}},
    state_in::Vector{T},
    state_out_target::Vector{Z},
    ;
    integer_strategy::AbstractIntegerStrategy=NoIntegerStrategy(),
) where {T<:Real,Z<:Real}
    return _get_next_state(
        subproblem,
        state_param_in,
        state_param_out,
        state_in,
        state_out_target,
        integer_strategy,
    )
end

function _get_next_state(
    subproblem::JuMP.Model,
    state_param_in,
    state_param_out,
    state_in,
    state_out_target,
    ::NoIntegerStrategy,
)
    return [value(state_param_out[i][2]) for i in 1:length(state_param_out)]
end

function _get_next_state(
    subproblem::JuMP.Model,
    state_param_in,
    state_param_out,
    state_in,
    state_out_target,
    integer_strategy::AbstractIntegerStrategy,
)
    _set_stage_parameters!(state_param_in, state_param_out, (), state_in, state_out_target)
    return with_sensitivity_solution(subproblem, integer_strategy) do sensitivity_model
        return [
            value(state_param_out[i][2]) for i in 1:length(state_param_out)
        ]
    end
end

"""
    ChainRulesCore.rrule(get_next_state, subproblem, state_param_in, state_param_out, state_in, state_out_target)

Correct reverse-mode rule using DiffOpt:
- Seeds reverse on realized output variables with Δstate_out
- Calls `DiffOpt.reverse_differentiate!`
- Reads sensitivities wrt parameter vars (state_param_in, state_param_out parameters)
- Returns VJP wrt the numeric inputs `state_in` and `state_out_target`
Assumptions:
- `subproblem` is a JuMP.Model constructed with `Model(() -> DiffOpt.diff_optimizer(...))`
- `state_param_in::Vector{JuMP.VariableRef}` are JuMP Parameter variables (incoming state parameters)
- `state_param_out::Vector{Tuple{JuMP.VariableRef,JuMP.VariableRef}}` holds
    (target-Parameter variable, realized-state Variable) per component
- `get_next_state(...)` updates parameter values, `optimize!`s, and returns a Vector matching the realized-state variables
"""
function ChainRulesCore.rrule(
    ::typeof(_get_next_state),
    subproblem::JuMP.Model,
    state_param_in,
    state_param_out,
    state_in,
    state_out_target,
    integer_strategy::AbstractIntegerStrategy,
)

    # Forward pass: run the solver via the user's function
    y = DecisionRules._get_next_state(
        subproblem,
        state_param_in,
        state_param_out,
        state_in,
        state_out_target,
        integer_strategy,
    )
    forward_status = _sensitivity_forward_status(subproblem, integer_strategy)

    function pullback(Δy)
        status = forward_status
        if !(status in _SUCCESSFUL_TERM_STATUSES)
            if STRICT_GRADIENTS[]
                error(
                    "get_next_state pullback: solver terminated with status $status; " *
                    "expected a successful solve.",
                )
            end
            @warn "get_next_state: solver status $status, returning zero gradients" status
            return (
                NoTangent(),
                NoTangent(),
                NoTangent(),
                NoTangent(),
                zeros(length(state_in)),
                zeros(length(state_out_target)),
                NoTangent(),
            )
        end

        try
            Δy = collect(Δy)  # ensure indexable, concrete element type

            return _with_current_or_sensitivity_solution(
                subproblem, integer_strategy
            ) do sensitivity_model
                # Best practice: clear previous seeds
                DiffOpt.empty_input_sensitivities!(sensitivity_model)

                # 1) Seed reverse on the realized output variables with Δy
                #    Each entry in state_param_out is (param_target, realized_state_var)
                @inbounds for i in eachindex(state_param_out)
                    realized_var = state_param_out[i][2]
                    # J' * Δ: set reverse seed on variable primal
                    DiffOpt.set_reverse_variable(sensitivity_model, realized_var, Δy[i])
                end

                # 2) Reverse differentiate
                DiffOpt.reverse_differentiate!(sensitivity_model)  # computes all needed products

                # 3) Read sensitivities w.r.t. parameter variables
                #    These are vector-Jacobian products dL/d(param) = (∂y/∂param)^T * Δy
                d_state_in = similar(
                    state_in, promote_type(eltype(state_in), eltype(Δy))
                )
                @inbounds for i in eachindex(state_param_in)
                    pin = state_param_in[i]  # JuMP.Parameter variable
                    d_state_in[i] = DiffOpt.get_reverse_parameter(sensitivity_model, pin)
                end

                d_state_out_target = similar(
                    state_out_target, promote_type(eltype(state_out_target), eltype(Δy))
                )
                @inbounds for i in eachindex(state_param_out)
                    pout = state_param_out[i][1]  # target Parameter variable
                    d_state_out_target[i] = DiffOpt.get_reverse_parameter(
                        sensitivity_model, pout
                    )
                end

                # Optional: clear seeds so they don't accumulate between calls
                DiffOpt.empty_input_sensitivities!(sensitivity_model)

                # Return cotangents for each primal argument, in order:
                #  (f, subproblem, state_param_in, state_param_out, state_in, state_out_target, integer_strategy)
                return (
                    NoTangent(),
                    NoTangent(),
                    NoTangent(),
                    NoTangent(),
                    d_state_in,
                    d_state_out_target,
                    NoTangent(),
                )
            end
        catch e
            return handle_gradient_error(
                _DEFAULT_GRADIENT_FALLBACK, e, length(state_in), length(state_out_target)
            )
        end
    end

    return y, pullback
end

function ChainRulesCore.rrule(
    ::typeof(get_next_state),
    subproblem::JuMP.Model,
    state_param_in,
    state_param_out,
    state_in,
    state_out_target,
    ;
    integer_strategy::AbstractIntegerStrategy=NoIntegerStrategy(),
)
    y, pullback = ChainRulesCore.rrule(
        _get_next_state,
        subproblem,
        state_param_in,
        state_param_out,
        state_in,
        state_out_target,
        integer_strategy,
    )
    function public_pullback(Δy)
        result = pullback(Δy)
        return result[1:6]
    end
    return y, public_pullback
end

function get_objective_no_target_deficit(
    subproblem::JuMP.Model; norm_deficit::AbstractString="norm_deficit"
)
    if subproblem.is_model_dirty
        return get(subproblem.ext, :_last_obj_no_deficit, 0.0)
    end
    try
        obj = JuMP.objective_function(subproblem)
        objective_val = objective_value(subproblem)
        for term in obj.terms
            if occursin(norm_deficit, JuMP.name(term[1]))
                objective_val -= term[2] * value(term[1])
            end
        end
        return objective_val
    catch
        return get(subproblem.ext, :_last_obj_no_deficit, 0.0)
    end
end

function get_objective_no_target_deficit(
    subproblems::Vector{JuMP.Model}; norm_deficit::AbstractString="norm_deficit"
)
    total_objective = 0.0
    for subproblem in subproblems
        total_objective += get_objective_no_target_deficit(
            subproblem; norm_deficit=norm_deficit
        )
    end
    return total_objective
end

# define ChainRulesCore.rrule of get_objective_no_target_deficit
function ChainRulesCore.rrule(
    ::typeof(get_objective_no_target_deficit), subproblem; norm_deficit="norm_deficit"
)
    objective_val = get_objective_no_target_deficit(subproblem; norm_deficit=norm_deficit)
    function _pullback(Δobjective_val)
        return (NoTangent(), NoTangent())
    end
    return objective_val, _pullback
end

function apply_rule(::Int, decision_rule::T, uncertainty, state_in) where {T}
    return decision_rule(vcat([uncertainty[i][2] for i in 1:length(uncertainty)], state_in))
end

function apply_rule(stage::Int, decision_rules::Vector{T}, uncertainty, state_in) where {T}
    return apply_rule(stage, decision_rules[stage], uncertainty, state_in)
end

"""
    simulate_multistage(subproblems, state_params_in, state_params_out,
                        initial_state, uncertainties, decision_rules) -> Float64

Stage-wise (single shooting) forward simulation.  Rolls `decision_rules` over
`uncertainties`, solving one subproblem per stage.  The realized state from each
stage feeds the next via `get_next_state`.  Returns the total objective across
all stages (Extension §2, Eq. 2.1–2.4).
"""
function simulate_multistage(
    subproblems::Vector{JuMP.Model},
    state_params_in::Vector{Vector{U}},
    state_params_out::Vector{Vector{Tuple{U,VariableRef}}},
    initial_state::Vector{T},
    uncertainties,
    decision_rules,
    ;
    integer_strategy::AbstractIntegerStrategy=NoIntegerStrategy(),
) where {T<:Real,U}
    @ignore_derivatives Flux.reset!(decision_rules)

    # Loop over stages
    objective_value = 0.0
    state_in = initial_state
    for stage in 1:length(subproblems)
        subproblem = subproblems[stage]
        state_param_in = state_params_in[stage]
        state_param_out = state_params_out[stage]
        uncertainty = uncertainties[stage]
        state_out = apply_rule(stage, decision_rules, uncertainty, state_in)
        objective_value += simulate_stage(
            subproblem,
            state_param_in,
            state_param_out,
            uncertainty,
            state_in,
            state_out;
            integer_strategy=integer_strategy,
        )
        state_in = DecisionRules.get_next_state(
            subproblem,
            state_param_in,
            state_param_out,
            state_in,
            state_out;
            integer_strategy=integer_strategy,
        )
    end

    # Return final objective value
    return objective_value
end

"""
    simulate_multistage(det_equivalent, state_params_in, state_params_out,
                        uncertainties, states) -> Float64

Deterministic-equivalent (direct transcription) forward pass.  Sets all parameter
values from `states` and `uncertainties` into the coupled `det_equivalent` model,
solves it, and returns the objective value (Extension §1, Eq. 1.1).
"""
function simulate_multistage(
    det_equivalent::JuMP.Model,
    state_params_in::Vector{Vector{Z}},
    state_params_out::Vector{Vector{Tuple{Z,VariableRef}}},
    uncertainties,
    states,
    ;
    integer_strategy::AbstractIntegerStrategy=NoIntegerStrategy(),
) where {Z}
    return _simulate_multistage_det(
        det_equivalent,
        state_params_in,
        state_params_out,
        uncertainties,
        states,
        integer_strategy,
    )
end

function _set_multistage_parameters!(
    state_params_in,
    state_params_out,
    uncertainties,
    states,
)
    for t in 1:length(state_params_in)
        state = states[t]
        # Update state parameters in
        if t == 1
            for (i, state_var) in enumerate(state_params_in[t])
                set_parameter_value(state_var, state[i])
            end
        end

        # Update uncertainty
        for (uncertainty_param, uncertainty_value) in uncertainties[t]
            set_parameter_value(uncertainty_param, uncertainty_value)
        end

        # Update state parameters out
        for i in 1:length(state_params_out[t])
            state_var = state_params_out[t][i][1]
            set_parameter_value(state_var, states[t + 1][i])
        end
    end
    return nothing
end

function _simulate_multistage_det(
    det_equivalent::JuMP.Model,
    state_params_in,
    state_params_out,
    uncertainties,
    states,
    integer_strategy::AbstractIntegerStrategy,
)
    _set_multistage_parameters!(state_params_in, state_params_out, uncertainties, states)

    return with_sensitivity_solution(det_equivalent, integer_strategy) do sensitivity_model
        obj = objective_value(sensitivity_model)
        sensitivity_model.ext[:_last_obj] = obj
        sensitivity_model.ext[:_last_obj_no_deficit] =
            get_objective_no_target_deficit(sensitivity_model)
        return obj
    end
end

function _simulate_multistage_det_with_parameter_duals(
    det_equivalent,
    state_params_in,
    state_params_out,
    uncertainties,
    states,
    integer_strategy::AbstractIntegerStrategy,
)
    _set_multistage_parameters!(state_params_in, state_params_out, uncertainties, states)

    return with_sensitivity_solution(det_equivalent, integer_strategy) do sensitivity_model
        objective = objective_value(sensitivity_model)
        sensitivity_model.ext[:_last_obj] = objective
        sensitivity_model.ext[:_last_obj_no_deficit] =
            get_objective_no_target_deficit(sensitivity_model)
        Δ_states = similar(states)
        Δ_states[1] = pdual.(state_params_in[1])
        for t in 1:length(state_params_out)
            Δ_states[t + 1] = pdual.([s[1] for s in state_params_out[t]])
        end
        return objective, Δ_states
    end
end

"""
    simulate_multistage(det_equivalent::JuMP.Model, state_params_in, state_params_out,
                        initial_state, uncertainties, decision_rules) -> Float64

Convenience overload: rolls out `decision_rules` to produce target states, then
calls the deterministic-equivalent `simulate_multistage` to solve the coupled problem.
"""
function simulate_multistage(
    subproblems::JuMP.Model,
    state_params_in::Vector{Vector{U}},
    state_params_out::Vector{Vector{Tuple{U,VariableRef}}},
    initial_state::Vector{T},
    uncertainties,
    decision_rules,
    ;
    integer_strategy::AbstractIntegerStrategy=NoIntegerStrategy(),
) where {T<:Real,U}
    Flux.reset!(decision_rules)
    states = simulate_states(initial_state, uncertainties, decision_rules)
    return simulate_multistage(
        subproblems,
        state_params_in,
        state_params_out,
        uncertainties,
        states;
        integer_strategy=integer_strategy,
    )
end

"""
    pdual(v::VariableRef) -> Float64

Compute ``∂Q/∂p`` for a JuMP parameter variable ``p`` in a solved model, where
``Q`` is the optimal objective value.  By the envelope theorem / Lagrangian
duality this equals the sum of ``-\\text{coef} \\times \\text{dual}`` over all
constraints where ``p`` appears, plus the objective coefficient of ``p``.

This is the key quantity in TS-DDR (arXiv:2405.14973): the dual ``λ_t`` of the
target constraint gives the sensitivity ``∂Q/∂\\hat{x}_t`` used in Eq. 1.2.
"""
function pdual(v::VariableRef)
    if is_parameter(v)
        return compute_parameter_dual(JuMP.owner_model(v), v)
    else
        error("Variable is not a parameter")
    end
end

pdual(vs::Vector) = [pdual(v) for v in vs]

"""
    ChainRulesCore.rrule(::typeof(simulate_stage), subproblem, state_param_in,
                         state_param_out, uncertainty, state_in, state_out)

Reverse-mode rule for a single-stage subproblem solve.

## Mathematical basis (TS-DDR, arXiv:2405.14973; Extension §2 Eq. 2.5)

For stage problem ``q_t(x_{t-1}, w_t; \\hat{x}_t)``, the sensitivities are:

    ∂q_t/∂(state_in)  = μ_t    (dual of dynamics constraint w.r.t. incoming state)
    ∂q_t/∂(target)     = λ_t    (dual of target constraint w.r.t. target x̂_t)

These are the Lagrange multipliers that [`compute_parameter_dual`](@ref) (`pdual`)
extracts from the solved model.  This is the **preferred path**: closed-form
and exact whenever the solver exposes constraint duals.

## Fallback strategy

1. **pdual** (parameter duals) — tried first.
2. **DiffOpt reverse differentiation** — if pdual raises (e.g. the optimizer
   wrapper does not expose conic duals).  Computes the same sensitivities via
   implicit differentiation of the KKT system.
3. **Zero gradients** — only when the solver terminated with an unsuccessful
   status (not OPTIMAL / ALMOST_OPTIMAL / LOCALLY_SOLVED).  A warning is
   emitted.  Set `DecisionRules.STRICT_GRADIENTS[] = true` to throw instead.
"""
function ChainRulesCore.rrule(
    ::typeof(_simulate_stage),
    subproblem,
    state_param_in,
    state_param_out,
    uncertainty,
    state_in,
    state_out,
    integer_strategy::AbstractIntegerStrategy,
)
    y = nothing
    d_state_in_pdual = nothing
    d_state_out_pdual = nothing
    pdual_available = false
    try
        y, d_state_in_pdual, d_state_out_pdual = _simulate_stage_with_parameter_duals(
            subproblem,
            state_param_in,
            state_param_out,
            uncertainty,
            state_in,
            state_out,
            integer_strategy,
        )
        pdual_available = true
    catch
        y = _simulate_stage(
            subproblem,
            state_param_in,
            state_param_out,
            uncertainty,
            state_in,
            state_out,
            integer_strategy,
        )
    end
    forward_status = _sensitivity_forward_status(subproblem, integer_strategy)
    function _pullback(Δy)
        status = forward_status
        if !(status in _SUCCESSFUL_TERM_STATUSES)
            if STRICT_GRADIENTS[]
                error(
                    "simulate_stage pullback: solver terminated with status $status; " *
                    "expected a successful solve. Set DecisionRules.STRICT_GRADIENTS[] " *
                    "= false to return zero gradients instead.",
                )
            end
            @warn "simulate_stage: solver status $status, returning zero gradients" status
            return (
                NoTangent(),
                NoTangent(),
                NoTangent(),
                NoTangent(),
                NoTangent(),
                zeros(length(state_param_in)),
                zeros(length(state_param_out)),
                NoTangent(),
            )
        end

        # Preferred: parameter duals (closed-form, Eq. 1.2 / 2.5)
        if pdual_available
            return (
                NoTangent(),
                NoTangent(),
                NoTangent(),
                NoTangent(),
                NoTangent(),
                d_state_in_pdual * Δy,
                d_state_out_pdual * Δy,
                NoTangent(),
            )
        end

        # Fallback: DiffOpt reverse differentiation (same math, implicit diff of KKT)
        return _with_current_or_sensitivity_solution(
            subproblem, integer_strategy
        ) do sensitivity_model
            DiffOpt.empty_input_sensitivities!(sensitivity_model)
            MOI.set(sensitivity_model, DiffOpt.ReverseObjectiveSensitivity(), Δy)
            DiffOpt.reverse_differentiate!(sensitivity_model)
            result = (
                NoTangent(),
                NoTangent(),
                NoTangent(),
                NoTangent(),
                NoTangent(),
                DiffOpt.get_reverse_parameter.(sensitivity_model, state_param_in),
                DiffOpt.get_reverse_parameter.(
                    sensitivity_model, [s[1] for s in state_param_out]
                ),
                NoTangent(),
            )
            DiffOpt.empty_input_sensitivities!(sensitivity_model)
            return result
        end
    end
    return y, _pullback
end

function ChainRulesCore.rrule(
    ::typeof(simulate_stage),
    subproblem,
    state_param_in,
    state_param_out,
    uncertainty,
    state_in,
    state_out,
    ;
    integer_strategy::AbstractIntegerStrategy=NoIntegerStrategy(),
)
    y, pullback = ChainRulesCore.rrule(
        _simulate_stage,
        subproblem,
        state_param_in,
        state_param_out,
        uncertainty,
        state_in,
        state_out,
        integer_strategy,
    )
    function public_pullback(Δy)
        result = pullback(Δy)
        return result[1:7]
    end
    return y, public_pullback
end

"""
    ChainRulesCore.rrule(::typeof(simulate_multistage), det_equivalent, state_params_in,
                         state_params_out, uncertainties, states)

Reverse-mode rule for the deterministic-equivalent (full-horizon) solve.

## Mathematical basis (TS-DDR, arXiv:2405.14973, Eq. 1.2; Extension §1)

For the coupled problem ``Q(w;θ) = \\min \\sum_t f_t + C_δ\\|δ_t\\|`` the gradient
estimator is:

    ∇_θ E[Q] ≈ (1/S) Σ_s  λ^s ⊙ ∇_θ π(·; θ)

where ``λ_t`` is the dual of the target constraint ``x_t + δ_t = \\hat{x}_t``.
The pullback returns ``Δ_{states}`` such that ``Δ_{states}[1]`` holds the
parameter duals of the initial-state parameters and ``Δ_{states}[t+1]`` holds
the target-constraint duals ``λ_t`` for each stage.

## Fallback strategy

Same as [`simulate_stage`](@ref): tries [`compute_parameter_dual`](@ref)
first, falls back to DiffOpt reverse differentiation if pdual raises.
Solver failure (bad termination status) returns zero gradients or throws
depending on [`STRICT_GRADIENTS`](@ref).
"""
function ChainRulesCore.rrule(
    ::typeof(_simulate_multistage_det),
    det_equivalent::JuMP.Model,
    state_params_in,
    state_params_out,
    uncertainties,
    states,
    integer_strategy::AbstractIntegerStrategy,
)
    y = nothing
    Δ_states_pdual = nothing
    pdual_available = false
    try
        y, Δ_states_pdual = _simulate_multistage_det_with_parameter_duals(
            det_equivalent,
            state_params_in,
            state_params_out,
            uncertainties,
            states,
            integer_strategy,
        )
        pdual_available = true
    catch
        y = _simulate_multistage_det(
            det_equivalent,
            state_params_in,
            state_params_out,
            uncertainties,
            states,
            integer_strategy,
        )
    end
    forward_status = _sensitivity_forward_status(det_equivalent, integer_strategy)
    function _pullback(Δy)
        status = forward_status
        Δ_states = similar(states)
        if !(status in _SUCCESSFUL_TERM_STATUSES)
            if STRICT_GRADIENTS[]
                error(
                    "simulate_multistage (det_eq) pullback: solver terminated with " *
                    "status $status; expected a successful solve.",
                )
            end
            @warn "simulate_multistage (det_eq): solver status $status, returning zero gradients" status
            Δ_states[1] = zeros(length(state_params_in[1]))
            for t in 1:length(state_params_out)
                Δ_states[t + 1] = zeros(length(state_params_out[t]))
            end
            return (
                NoTangent(),
                NoTangent(),
                NoTangent(),
                NoTangent(),
                NoTangent(),
                Δ_states,
                NoTangent(),
            )
        end

        # Preferred: parameter duals (closed-form, Eq. 1.2)
        if pdual_available
            return (
                NoTangent(),
                NoTangent(),
                NoTangent(),
                NoTangent(),
                NoTangent(),
                Δ_states_pdual * Δy,
                NoTangent(),
            )
        end

        # Fallback: DiffOpt reverse differentiation
        return _with_current_or_sensitivity_solution(
            det_equivalent, integer_strategy
        ) do sensitivity_model
            DiffOpt.empty_input_sensitivities!(sensitivity_model)
            MOI.set(sensitivity_model, DiffOpt.ReverseObjectiveSensitivity(), Δy)
            DiffOpt.reverse_differentiate!(sensitivity_model)
            Δ_states[1] = DiffOpt.get_reverse_parameter.(
                sensitivity_model, state_params_in[1]
            )
            for t in 1:length(state_params_out)
                Δ_states[t + 1] = DiffOpt.get_reverse_parameter.(
                    sensitivity_model, [s[1] for s in state_params_out[t]]
                )
            end
            DiffOpt.empty_input_sensitivities!(sensitivity_model)
            return (
                NoTangent(),
                NoTangent(),
                NoTangent(),
                NoTangent(),
                NoTangent(),
                Δ_states,
                NoTangent(),
            )
        end
    end
    return y, _pullback
end

function ChainRulesCore.rrule(
    ::typeof(simulate_multistage),
    det_equivalent::JuMP.Model,
    state_params_in,
    state_params_out,
    uncertainties,
    states,
    ;
    integer_strategy::AbstractIntegerStrategy=NoIntegerStrategy(),
)
    y, pullback = ChainRulesCore.rrule(
        _simulate_multistage_det,
        det_equivalent,
        state_params_in,
        state_params_out,
        uncertainties,
        states,
        integer_strategy,
    )
    function public_pullback(Δy)
        result = pullback(Δy)
        return result[1:6]
    end
    return y, public_pullback
end

@doc raw"""
    sample(uncertainty_pool) -> Vector{Vector{Tuple{VariableRef, T}}}

Draw one full uncertainty trajectory from a DecisionRules uncertainty pool.

The returned trajectory is a length-``T`` vector where each element is
`Vector{Tuple{VariableRef, Float64}}` — one realized value per uncertain
parameter for that stage. This is the format consumed by `simulate_multistage`,
`train_multistage`, and all other training/evaluation functions.

Three pool formats are supported, offering increasing levels of correlation:

## 1. Independent sampling (per-unit pools)

Each uncertain parameter has its own finite support; sampling draws
independently from each support at each stage.

    sample(multistage_pool::Vector{Vector{Tuple{VariableRef, Vector{T}}}})

`multistage_pool[t]` is `[(param₁, [v₁₁, v₁₂, …]), (param₂, [v₂₁, v₂₂, …]), …]`.
Each parameter picks one value uniformly at random from its own support.
**No spatial or temporal correlation is preserved.**

## 2. Joint-scenario sampling (spatial correlation)

Scenarios are pre-defined joint realizations across all parameters at each
stage. Sampling picks one complete scenario per stage uniformly, preserving
cross-parameter correlations (e.g., spatially correlated inflows across
hydro reservoirs). Stages are still drawn independently.

    sample(multistage_joint::Vector{Vector{Vector{Tuple{VariableRef, T}}}})

`multistage_joint[t]` is `[scenario₁, scenario₂, …]` where each scenario
is `[(param₁, val₁), (param₂, val₂), …]`.

## 3. Trajectory sampler (spatial + temporal correlation)

A callable `sampler(t, past) -> Vector{Tuple{VariableRef, T}}` that generates
stage `t`'s realization given the realized values from stages `1:t-1`. This
enables autoregressive, Markovian, or any custom temporal dependence.

    sample(sampler::Function, T::Int)

The callable receives:
- `t::Int` — the current stage (1-indexed)
- `past::Vector{Vector{Tuple{VariableRef, T}}}` — realized samples from
  stages `1:t-1` (empty vector for `t=1`)

and must return `Vector{Tuple{VariableRef, T}}` — the realized sample for
stage `t`.

## Output format

All three methods return `Vector{Vector{Tuple{VariableRef, T}}}` — a length-``T``
vector of per-stage realized samples. This is the universal input to
`simulate_multistage`, `train_multistage`, `simulate_multiple_shooting`, and all
evaluation functions.

# Examples
```julia
# 1. Independent sampling (each unit draws independently):
independent_pool = [
    [(inflow_1, [10.0, 15.0, 12.0]), (inflow_2, [8.0, 12.0, 9.0])],
    [(inflow_1, [11.0, 14.0, 13.0]), (inflow_2, [7.0, 11.0, 10.0])],
]
path = sample(independent_pool)

# 2. Joint-scenario sampling (preserves spatial correlation):
joint_pool = [
    [[(inflow_1, 10.0), (inflow_2, 8.0)],   # scenario 1
     [(inflow_1, 15.0), (inflow_2, 12.0)]],  # scenario 2 — stage 1
    [[(inflow_1, 11.0), (inflow_2, 7.0)],
     [(inflow_1, 14.0), (inflow_2, 11.0)]],  # stage 2
]
path = sample(joint_pool)

# 3. Trajectory sampler (preserves temporal + spatial correlation):
function my_sampler(t, past)
    if t == 1
        ω = rand(1:nScenarios)
        return [(inflow_params[t][r], data[r][t, ω]) for r in 1:nHyd]
    else
        # AR(1): next inflow depends on previous realized inflow
        prev_values = [pair[2] for pair in past[end]]
        noise = randn(nHyd) .* σ
        return [(inflow_params[t][r], ρ * prev_values[r] + noise[r]) for r in 1:nHyd]
    end
end
path = sample(my_sampler, T)
```

See the [Uncertainty Sampling](@ref) documentation page for a complete guide.
"""
function sample(uncertainty_samples::Vector{Tuple{VariableRef,Vector{T}}}) where {T<:Real}
    uncertainty_sample = Vector{Tuple{VariableRef,T}}(undef, length(uncertainty_samples))
    for i in 1:length(uncertainty_samples)
        uncertainty_sample[i] = (uncertainty_samples[i][1], rand(uncertainty_samples[i][2]))
    end
    return uncertainty_sample
end

function sample(joint_scenarios::Vector{Vector{Tuple{VariableRef,T}}}) where {T<:Real}
    return rand(joint_scenarios)
end

function sample(
    uncertainty_samples::Vector{Vector{Tuple{VariableRef,Vector{T}}}}
) where {T<:Real}
    return [sample(uncertainty_samples[t]) for t in 1:length(uncertainty_samples)]
end

function sample(
    uncertainty_samples::Vector{Vector{Vector{Tuple{VariableRef,T}}}}
) where {T<:Real}
    return [sample(uncertainty_samples[t]) for t in 1:length(uncertainty_samples)]
end

"""
    sample(sampler::Function, T::Int)

Draw a full trajectory using a callable trajectory sampler with temporal dependence.

`sampler(t, past)` receives the current stage `t` and a vector of all previously
realized samples `past[1:t-1]`, and returns the realized sample for stage `t`.

This enables autoregressive, Markovian, or any custom temporal correlation between
stages — something the data-based pool formats cannot express.

See [`sample`](@ref) for the full API and examples.
"""
function sample(sampler::Function, T::Int)
    trajectory = Vector{Vector{Tuple{VariableRef,Float64}}}(undef, T)
    past = Vector{Vector{Tuple{VariableRef,Float64}}}()
    for t in 1:T
        trajectory[t] = sampler(t, past)
        push!(past, trajectory[t])
    end
    return trajectory
end

"""
    sample(sampler::Function)

Call a zero-argument trajectory sampler that returns a complete trajectory.

This is the dispatch used by `train_multistage` and `train_multiple_shooting`
when `uncertainty_sampler` is a callable. Wrap a trajectory sampler as:

```julia
uncertainty_sampler = () -> sample(my_stage_sampler, T)
```
"""
function sample(sampler::Function)
    return sampler()
end

@doc raw"""
    train_multistage(model, initial_state, subproblems::Vector{JuMP.Model},
                     state_params_in, state_params_out, uncertainty_sampler;
                     kwargs...)

Train a target-state policy with stage-wise decomposition (single shooting).

For one sampled uncertainty trajectory ``w_{1:T}``, this overload solves one
optimization problem per stage. At stage ``t``, given the realized incoming
state ``x_{t-1}``, the policy predicts a target
``\hat{x}_t = \pi_\theta(w_t, x_{t-1})`` and the stage problem is

```math
\begin{aligned}
q_t(x_{t-1}, w_t; \hat{x}_t)
    = \min_{x_t, y_t, \delta_t}
    \quad & f_t(x_t, y_t) + C_\delta \|\delta_t\| \\
\text{s.t.}\quad
    & x_t = T_t(w_t, y_t, x_{t-1})                 && : \mu_t, \\
    & x_t + \delta_t = \hat{x}_t                   && : \lambda_t, \\
    & h_t(x_t, y_t) \ge 0 .
\end{aligned}
```

The rollout objective is the sum of stage values,

```math
Q(\theta; w) =
    \sum_{t=1}^{T} q_t(x_{t-1}, w_t; \hat{x}_t),
```

where each realized ``x_t`` is read from the previous stage solve. The gradient
therefore contains both the target duals ``\lambda_t`` and the sensitivity of
later realized states with respect to earlier targets. In the notation of the
extension note,

```math
\nabla_\theta Q(\theta; w)
=
\sum_{t=1}^{T}
\left[
    \frac{\partial q_t}{\partial \hat{x}_t}
    +
    \sum_{k=t+1}^{T}
    \frac{\partial q_k}{\partial x_{k-1}}
    \prod_{j=t+1}^{k-1}
    \frac{\partial x_j}{\partial x_{j-1}}
    \frac{\partial x_t}{\partial \hat{x}_t}
\right]
\nabla_\theta \pi_\theta(w_t, x_{t-1}).
```

The dual terms come from target and transition constraints; the state
sensitivities are computed through DiffOpt in the rrules for
[`simulate_stage`](@ref) and [`get_next_state`](@ref).

# Arguments
- `model`: differentiable Flux-compatible policy. It receives
  `vcat(stage_uncertainty, realized_state)` and returns the next target state.
- `initial_state::AbstractVector{<:Real}`: state ``x_0`` entering stage 1.
- `subproblems::Vector{JuMP.Model}`: one JuMP model per stage.
- `state_params_in`: stage input-state parameters.
- `state_params_out`: `(target_parameter, realized_state_variable)` pairs for
  each stage output state.
- `uncertainty_sampler`: source of uncertainty trajectories, passed to
  [`sample`](@ref). Three formats are accepted:
  1. **Per-unit pool** (`Vector{Vector{Tuple{VariableRef, Vector{T}}}}`):
     independent sampling per parameter per stage.
  2. **Joint-scenario pool** (`Vector{Vector{Vector{Tuple{VariableRef, T}}}}`):
     one scenario drawn per stage, preserving spatial correlation.
  3. **Callable** (`() -> Vector{Vector{Tuple{VariableRef, T}}}`): a zero-arg
     function returning a full trajectory. Use this for temporal correlation
     by wrapping a trajectory sampler:
     `() -> sample(my_stage_sampler, T)` where `my_stage_sampler(t, past)`
     generates stage `t` conditioned on past realizations.

# Keywords
- `num_batches::Integer`: number of SGD batches.
- `num_train_per_batch::Integer`: sampled trajectories per batch.
- `optimizer`: Flux optimizer used to update `model`.
- `adjust_hyperparameters::Function`: optional hook returning the batch size for
  the current iteration.
- `record_loss`: legacy logging callback.
- `sample_log::SampleLog`: per-batch objective cache.
- `record::Function`: callback called as `record(sample_log, iter, model)`.
- `penalty_schedule`: optional multiplier schedule for target-penalty terms.
- `integer_strategy::AbstractIntegerStrategy`: strategy used when a stage model
  has discrete variables and derivative information must be read.

# Examples
```julia
# With data pool (independent or joint):
train_multistage(
    policy, initial_state, subproblems,
    state_params_in, state_params_out, uncertainty_pool;
    num_batches=200, optimizer=Flux.Adam(1e-3),
)

# With trajectory sampler (temporal correlation):
ar_sampler(t, past) = my_ar1_model(t, past, inflow_params)
train_multistage(
    policy, initial_state, subproblems,
    state_params_in, state_params_out,
    () -> sample(ar_sampler, T);
    num_batches=200, optimizer=Flux.Adam(1e-3),
)
```
"""
function train_multistage(
    model,
    initial_state,
    subproblems::Vector{JuMP.Model},
    state_params_in,
    state_params_out,
    uncertainty_sampler;
    num_batches=100,
    num_train_per_batch=32,
    optimizer=Flux.Adam(0.01),
    adjust_hyperparameters=(iter, opt_state, num_train_per_batch) -> num_train_per_batch,
    record_loss=nothing,
    get_objective_no_target_deficit=get_objective_no_target_deficit,
    sample_log=SampleLog(objective_no_deficit_fn=get_objective_no_target_deficit),
    record=default_record,
    penalty_schedule=nothing,
    integer_strategy::AbstractIntegerStrategy=NoIntegerStrategy(),
    gradient_fallback::AbstractGradientFallback=ZeroGradientFallback(),
)
    if gradient_fallback isa ZeroGradientFallback
        @info "Training with ZeroGradientFallback: solver/differentiation errors will be " *
              "caught and the iteration skipped (zero gradient). Pass " *
              "`gradient_fallback=ErrorGradientFallback()` to throw instead, or implement " *
              "a custom `AbstractGradientFallback` subtype."
    end

    record = _resolve_record(record, record_loss)
    opt_state = Flux.setup(optimizer, model)

    schedule = _resolve_penalty_schedule(penalty_schedule, num_batches)
    penalty_bases = if isnothing(schedule)
        nothing
    else
        _check_deficit_penalty_bases(_deficit_penalty_bases(subproblems))
    end
    current_multiplier = NaN

    for iter in 1:num_batches
        if !isnothing(schedule)
            multiplier = _penalty_multiplier_for(schedule, iter)
            if multiplier != current_multiplier
                _apply_deficit_penalty_multiplier!(subproblems, penalty_bases, multiplier)
                current_multiplier = multiplier
            end
        end
        num_train_per_batch = adjust_hyperparameters(iter, opt_state, num_train_per_batch)

        uncertainty_samples = [sample(uncertainty_sampler) for _ in 1:num_train_per_batch]

        objective = 0.0
        _reset_sample_log!(sample_log)
        grads = try
            Flux.gradient(model) do m
                for s in 1:num_train_per_batch
                    Flux.reset!(m)
                    objective += simulate_multistage(
                        subproblems,
                        state_params_in,
                        state_params_out,
                        initial_state,
                        uncertainty_samples[s],
                        m;
                        integer_strategy=integer_strategy,
                    )
                    @ignore_derivatives sample_log(s, subproblems)
                end
                objective /= num_train_per_batch
                return objective
            end
        catch e
            if handle_training_error(gradient_fallback, e, iter)
                nothing
            end
        end
        record(sample_log, iter, model) && break

        if isnothing(grads)
            continue
        end

        grad = materialize_tangent(grads[1])
        Flux.update!(opt_state, model, grad)
    end

    return model
end

function sim_states(t, m, initial_state, uncertainty_sample_vec, prev_states)
    # Input: [uncertainty, previous_predicted_state]
    # For t=1: return initial_state (no prediction needed)
    # For t>1: policy receives [uncertainty[t-1], prev_states[t-1]]
    if t == 1
        return Float32.(initial_state)
    else
        uncertainties_t = uncertainty_sample_vec[t - 1]
        prev_state = prev_states[t - 1]
        return m(vcat(uncertainties_t, prev_state))
    end
end

@doc raw"""
    train_multistage(model, initial_state, det_equivalent::JuMP.Model,
                     state_params_in, state_params_out, uncertainty_sampler;
                     score_function=nothing, kwargs...)

Train a target-state policy with a deterministic equivalent (direct transcription).

For one sampled trajectory ``w_{1:T}``, the policy first produces the full target
trajectory

```math
\hat{x}_{1:T}(\theta) = \pi_\theta(w_{1:T}, x_0).
```

The coupled implementation problem is

```math
\begin{aligned}
Q(w; \theta)
    =
    \min_{\{x_t, y_t, \delta_t\}_{t=1}^{T}}
    \quad &
    \sum_{t=1}^{T} f_t(x_t, y_t)
    + C_\delta \sum_{t=1}^{T} \|\delta_t\| \\
\text{s.t.}\quad
    & x_t = T_t(w_t, y_t, x_{t-1})        && t=1,\ldots,T, \\
    & x_t + \delta_t = \hat{x}_t(\theta)  && : \lambda_t,\quad t=1,\ldots,T, \\
    & h_t(x_t, y_t) \ge 0                 && t=1,\ldots,T .
\end{aligned}
```

The target trajectory appears as right-hand-side parameters. If
``\lambda_t`` is the dual multiplier of the target constraint, the envelope
gradient used by this overload is

```math
\nabla_\theta \mathbb{E}[Q(w; \theta)]
\approx
\frac{1}{S}
\sum_{s=1}^{S}
\sum_{t=1}^{T}
\lambda_t^s \odot
\nabla_\theta \hat{x}_t^s(\theta),
```

where ``S`` is `num_train_per_batch` and ``\odot`` denotes componentwise
multiplication.

Pass a [`ScoreFunctionConfig`](@ref) or [`ScoreFunctionSchedule`](@ref) via
`score_function` to mix the dual gradient with a REINFORCE correction
estimated from rollouts under perturbed targets.

When `score_function` is used, there are two separate solve paths:

1. `integer_strategy` applies to `det_equivalent` and controls how local dual
   information is read for the differentiable dual-gradient term.
2. `score_function` owns separate rollout subproblems. Those models are solved
   exactly as they are built, and their realized costs define the Monte Carlo
   score-function term.

For a mixed-integer model, this usually means
`integer_strategy = FixedDiscreteIntegerStrategy()` for the dual path and
MIP rollout subproblems inside `ScoreFunctionConfig` for the score-function
path.

# Arguments
- `model`: differentiable Flux-compatible policy. It is rolled forward over
  uncertainty values to produce ``\hat{x}_{1:T}``.
- `initial_state::AbstractVector{<:Real}`: state ``x_0``.
- `det_equivalent::JuMP.Model`: full-horizon JuMP model for one sampled
  trajectory.
- `state_params_in`: input-state parameters in the deterministic equivalent.
- `state_params_out`: `(target_parameter, realized_state_variable)` pairs for
  each target state.
- `uncertainty_sampler`: source of uncertainty trajectories, passed to
  [`sample`](@ref). Three formats are accepted:
  1. **Per-unit pool** (`Vector{Vector{Tuple{VariableRef, Vector{T}}}}`):
     independent sampling per parameter per stage.
  2. **Joint-scenario pool** (`Vector{Vector{Vector{Tuple{VariableRef, T}}}}`):
     one scenario drawn per stage, preserving spatial correlation.
  3. **Callable** (`() -> Vector{Vector{Tuple{VariableRef, T}}}`): a zero-arg
     function returning a full trajectory. Use this for temporal correlation;
     see [`sample`](@ref).

# Keywords
- `num_batches::Integer`: number of SGD batches.
- `num_train_per_batch::Integer`: sampled trajectories per batch ``S``.
- `optimizer`: Flux optimizer used to update `model`.
- `adjust_hyperparameters::Function`: optional hook returning the batch size for
  the current iteration.
- `record_loss`: legacy logging callback.
- `sample_log::SampleLog`: per-batch objective cache.
- `record::Function`: callback called as `record(sample_log, iter, model)`.
- `penalty_schedule`: optional multiplier schedule for target-penalty terms.
- `integer_strategy::AbstractIntegerStrategy`: strategy used to read local dual
  information from `det_equivalent` when it has discrete variables.
- `score_function`: optional [`ScoreFunctionConfig`](@ref) or
  [`ScoreFunctionSchedule`](@ref) for mixed dual/score-function gradients.

# Examples
```julia
train_multistage(
    policy,
    initial_state,
    det_equivalent,
    state_params_in,
    state_params_out,
    uncertainty_sampler;
    num_batches = 200,
    num_train_per_batch = 16,
    optimizer = Flux.Adam(1.0e-3),
    integer_strategy = FixedDiscreteIntegerStrategy(),
    score_function = nothing,
)
```
"""
function train_multistage(
    model,
    initial_state,
    det_equivalent::JuMP.Model,
    state_params_in,
    state_params_out,
    uncertainty_sampler;
    num_batches=100,
    num_train_per_batch=32,
    optimizer=Flux.Adam(0.01),
    adjust_hyperparameters=(iter, opt_state, num_train_per_batch) -> num_train_per_batch,
    record_loss=nothing,
    get_objective_no_target_deficit=get_objective_no_target_deficit,
    sample_log=SampleLog(objective_no_deficit_fn=get_objective_no_target_deficit),
    record=default_record,
    penalty_schedule=nothing,
    integer_strategy::AbstractIntegerStrategy=NoIntegerStrategy(),
    score_function::Union{Nothing,ScoreFunctionConfig,ScoreFunctionSchedule}=nothing,
    gradient_fallback::AbstractGradientFallback=ZeroGradientFallback(),
)
    record = _resolve_record(record, record_loss)
    opt_state = Flux.setup(optimizer, model)
    num_stages = length(state_params_in)

    schedule = _resolve_penalty_schedule(penalty_schedule, num_batches)
    penalty_bases = isnothing(schedule) ? nothing :
        _check_deficit_penalty_bases(_deficit_penalty_bases(det_equivalent))
    current_multiplier = NaN

    sf_cfg = _sf_config(score_function)
    use_sf = !isnothing(sf_cfg)

    for iter in 1:num_batches
        if !isnothing(schedule)
            multiplier = _penalty_multiplier_for(schedule, iter)
            if multiplier != current_multiplier
                _apply_deficit_penalty_multiplier!(
                    det_equivalent, penalty_bases, multiplier)
                current_multiplier = multiplier
            end
        end
        num_train_per_batch = adjust_hyperparameters(iter, opt_state, num_train_per_batch)

        score_params = use_sf ? sf_params(score_function, iter) :
            (
                alpha = 1.0,
                score_weight = 0.0,
                perturbation_std = 0.0,
                num_rollouts = 0,
                active = false,
            )

        uncertainty_samples = [sample(uncertainty_sampler) for _ in 1:num_train_per_batch]
        num_uncertainties = length(uncertainty_samples[1][1])
        uncertainty_samples_vec = [
            [
                [uncertainty_samples[s][stage][i][2] for i in 1:num_uncertainties]
                for stage in 1:num_stages
            ] for s in 1:num_train_per_batch
        ]

        objective = 0.0
        _reset_sample_log!(sample_log)
        grads = try
            Flux.gradient(model) do m
                for s in 1:num_train_per_batch
                    Flux.reset!(m)
                    x0 = Float32.(initial_state)
                    states = vcat([x0], accumulate(
                        uncertainty_samples_vec[s]; init=x0
                    ) do prev, ξ
                        m(vcat(ξ, prev))
                    end)

                    dual_obj = simulate_multistage(
                        det_equivalent, state_params_in, state_params_out,
                        uncertainty_samples[s], states;
                        integer_strategy=integer_strategy)
                    @ignore_derivatives sample_log(s, det_equivalent)
                    objective += score_params.alpha * dual_obj

                    if score_params.active
                        advantages, perturbations = @ignore_derivatives(
                            _score_function_rollouts(
                                sf_cfg,
                                initial_state,
                                uncertainty_samples[s],
                                states;
                                perturbation_std = score_params.perturbation_std,
                                num_rollouts = score_params.num_rollouts,
                            )
                        )
                        for rollout in 1:score_params.num_rollouts
                            advantage = @ignore_derivatives advantages[rollout]
                            perturbation = @ignore_derivatives perturbations[rollout]
                            surrogate = _score_function_surrogate(
                                advantage,
                                perturbation,
                                states,
                                score_params.perturbation_std,
                            )
                            objective += score_params.score_weight *
                                surrogate / Float32(score_params.num_rollouts)
                        end
                    end
                end
                objective /= num_train_per_batch
                return objective
            end
        catch e
            if handle_training_error(gradient_fallback, e, iter)
                nothing
            end
        end
        record(sample_log, iter, model) && break

        if isnothing(grads)
            continue
        end

        Flux.update!(opt_state, model, materialize_tangent(grads[1]))
    end

    return model
end
