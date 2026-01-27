"""
Multiple Shooting (Windowed Decomposition) for DecisionRules.jl

Windowed decomposition (a.k.a. multiple shooting in control):

- Partition the horizon into windows I_k of size W.
- For each window k, solve a deterministic equivalent over all stages in I_k
  (direct transcription within the window).
- Between windows, pass the *realized* end state s_k (from the window solve)
  as the initial state parameter for window k+1.

Gradients:

- Within a window: gradients w.r.t. *all* targets in the window come “directly”
  from the solve (e.g., duals of target RHS constraints). We obtain these as
  reverse sensitivities w.r.t. the target Parameter variables.
- Between windows: chain rule through the realized end state s_k. We seed reverse
  differentiation on the realized end-state variables to propagate gradients to
  earlier targets and to the window start parameter.

Implementation detail: we rely on DiffOpt reverse differentiation on the *window
model* to obtain vector-Jacobian products w.r.t. window start parameters and all
target parameters.

Assumptions:
- `deterministic_equivalent!(window_model, window_subproblems, window_state_params_in,
   window_state_params_out, initial_state, window_uncertainties)` builds the window model
  and updates `window_state_params_in/out` to refer to parameters/variables in the new model.
- `window_state_params_out[t]` is a vector of tuples (target_param, realized_state_var) at stage t.
"""

using Base: accumulate

#=============================================================================
 Helpers
=============================================================================#

"""
    ensure_state_in_is_parameter!(m, state_in_params)

Ensure window-start state entries are MOI.Parameter variables so `set_parameter_value`
works on window models. If an entry is not a parameter, create a new parameter and
tie it to the original variable with an equality constraint.
"""
function ensure_state_in_is_parameter!(m::JuMP.Model, state_in_params::Vector)
    for i in eachindex(state_in_params)
        v = state_in_params[i]
        if v isa JuMP.VariableRef && !JuMP.is_parameter(v)
            p = @variable(m; base_name = "shooting_state_in[$i]", set = MOI.Parameter(0.0))
            @constraint(m, v == p)
            state_in_params[i] = p
        end
    end
    return state_in_params
end

"""
    ensure_target_params_are_parameters!(m, window_state_out_params)

Ensure the target parameters in window_state_out_params are MOI.Parameter variables.
If a target is not a parameter, create a new parameter and tie it to the original
variable with an equality constraint.
"""
function ensure_target_params_are_parameters!(m::JuMP.Model, window_state_out_params)
    for t in eachindex(window_state_out_params)
        stage_pairs = window_state_out_params[t]
        for i in eachindex(stage_pairs)
            target_param, realized_var = stage_pairs[i]
            if target_param isa JuMP.VariableRef && !JuMP.is_parameter(target_param)
                p = @variable(m; base_name = "shooting_target[$t,$i]", set = MOI.Parameter(0.0))
                @constraint(m, target_param == p)
                stage_pairs[i] = (p, realized_var)
            end
        end
    end
    return window_state_out_params
end

"""
    extract_uncertainty_params(window_uncertainties_new)

Normalize uncertainty data to a per-stage vector of parameter VariableRefs.

Accepts either:
- Vector{Vector{Tuple{VariableRef, Any}}} (common in this package), or
- Vector{Vector{VariableRef}}.
"""
function extract_uncertainty_params(window_uncertainties)
    if isempty(window_uncertainties)
        return Vector{Vector{VariableRef}}()
    end
    first_stage = window_uncertainties[1]
    if isempty(first_stage)
        return [VariableRef[] for _ in 1:length(window_uncertainties)]
    end
    if first(first_stage) isa VariableRef
        return window_uncertainties
    else
        # assume tuples (param, something)
        return [[u[1] for u in stage_u] for stage_u in window_uncertainties]
    end
end

"""
    set_window_uncertainties!(window, uncertainty_sample)

Set sampled uncertainty values into the window model parameters.

- `window.uncertainty_params[t][i]` is the parameter VariableRef in the window model
- `uncertainty_sample[global_t][i][2]` is the sampled numeric value (original structure)
"""
function set_window_uncertainties!(
    window,
    uncertainty_sample,  # per-stage: Vector{Vector{Tuple{VariableRef, <Real>}}}
)
    stage_range = window.stage_range
    for local_t in 1:length(stage_range)
        global_t = stage_range[local_t]
        params = window.uncertainty_params[local_t]
        @inbounds for i in eachindex(params)
            set_parameter_value(params[i], uncertainty_sample[global_t][i][2])
        end
    end
    return nothing
end

#=============================================================================
 Core primitive: solve_window + rrule
=============================================================================#

"""
    solve_window(window_model, window_state_in_params, window_state_out_params,
                 s_in, targets)

Solve a deterministic-equivalent window model.

Arguments
- `window_model`: JuMP model (DiffOpt-enabled) for the window
- `window_state_in_params`: Vector of MOI.Parameter vars for window initial state
- `window_state_out_params`: per-stage vector of tuples (target_param, realized_var)
- `s_in`: numeric initial state
- `targets`: vector of numeric targets, one per stage in the window

Returns
- (objective, s_out): objective value, realized end state (Float32 vector)
"""
function solve_window(
    window_model::JuMP.Model,
    window_state_in_params::AbstractVector,
    window_state_out_params::AbstractVector{<:AbstractVector{<:Tuple{<:Any, VariableRef}}},
    s_in::AbstractVector,
    targets::AbstractVector{<:AbstractVector},
)
    num_stages = length(window_state_out_params)

    # Set initial state parameters
    @inbounds for i in eachindex(window_state_in_params)
        set_parameter_value(window_state_in_params[i], s_in[i])
    end

    # Set target parameters for each stage
    @inbounds for t in 1:num_stages
        stage_pairs = window_state_out_params[t]
        tgt = targets[t]
        @inbounds for i in eachindex(stage_pairs)
            target_param = stage_pairs[i][1]
            set_parameter_value(target_param, tgt[i])
        end
    end

    optimize!(window_model)

    obj = objective_value(window_model)

    return obj
end


"""
ChainRulesCore.rrule for solve_window

Computes gradients w.r.t.:
- s_in (window start numeric state)
- targets (all target vectors in the window)
Given cotangents (Δobj_val), we:
- seed reverse variables (objective and realized end state vars) with Δobj_val
- reverse_differentiate!
- read reverse sensitivities w.r.t. window_state_in_params and all target params
"""
function ChainRulesCore.rrule(
    ::typeof(solve_window),
    window_model::JuMP.Model,
    window_state_in_params::AbstractVector,
    window_state_out_params::AbstractVector{<:AbstractVector{<:Tuple{<:Any, VariableRef}}},
    s_in::AbstractVector,
    targets::AbstractVector{<:AbstractVector},
)
    obj = solve_window(window_model, window_state_in_params, window_state_out_params, s_in, targets)
    @assert JuMP.owner_model(window_state_in_params[1]) === window_model "window_model must be DiffOpt-enabled"
    @assert JuMP.owner_model(window_state_out_params[1][1][1]) === window_model "window_model must be DiffOpt-enabled"
    dual_s_in = pdual.(window_state_in_params)
    dual_targets = [pdual.([s[1] for s in stage_pairs]) for stage_pairs in window_state_out_params]

    function pullback(Δobj_val)
        Δobj = (Δobj_val isa NoTangent || Δobj_val isa ZeroTangent) ? 0.0 : float(Δobj_val)
        d_s_in = Δobj .* dual_s_in
        d_targets = Δobj .* dual_targets
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), d_s_in, d_targets)
    end

    return obj, pullback
end

"""
    get_last_realized_state(window_model, window_state_in_params, window_state_out_params,
                            s_in, targets)

Get the realized end state from the window model after solving.
"""
function get_last_realized_state(
    window_model::JuMP.Model,
    window_state_in_params::AbstractVector,
    window_state_out_params::AbstractVector{<:AbstractVector{<:Tuple{<:Any, VariableRef}}},
    s_in::AbstractVector,
    targets::AbstractVector{<:AbstractVector},
)

    last_stage = window_state_out_params[end]
    s_out = Float32[value(pair[2]) for pair in last_stage]

    return s_out
end

"""
ChainRulesCore.rrule for get_last_realized_state

Computes gradients w.r.t.:
- s_in (window start numeric state)
- targets (all target vectors in the window)

Given cotangents (Δs_out), we:
- seed reverse variables (realized end state vars) with Δs_out
- reverse_differentiate!
- read reverse sensitivities w.r.t. window_state_in_params and all target params
"""
function ChainRulesCore.rrule(
    ::typeof(get_last_realized_state),
    window_model::JuMP.Model,
    window_state_in_params::AbstractVector,
    window_state_out_params::AbstractVector{<:AbstractVector{<:Tuple{<:Any, VariableRef}}},
    s_in::AbstractVector,
    targets::AbstractVector{<:AbstractVector},
)
    s_out = get_last_realized_state(window_model, window_state_in_params, window_state_out_params, s_in, targets)

    function pullback(Δs_out)

        Δs_out_vec =
            (Δs_out isa NoTangent || Δs_out isa ZeroTangent) ? zeros(Float32, length(s_out)) : Float32.(collect(Δs_out))

        num_stages = length(window_state_out_params)

        d_s_in = zeros(Float32, length(s_in))
        d_targets = [zeros(Float32, length(targets[t])) for t in 1:num_stages]

        DiffOpt.empty_input_sensitivities!(window_model)

        # (B) end-state contribution
        if !all(iszero, Δs_out_vec)
            last_stage = window_state_out_params[end]
            @inbounds for i in eachindex(last_stage)
                realized_var = last_stage[i][2]
                DiffOpt.set_reverse_variable(window_model, realized_var, Δs_out_vec[i])
            end
        end

        DiffOpt.reverse_differentiate!(window_model)

        # gradient w.r.t. window start parameters
        @inbounds for i in eachindex(window_state_in_params)
            d_s_in[i] = Float32(DiffOpt.get_reverse_parameter(window_model, window_state_in_params[i]))
        end

        # gradient w.r.t. all target parameters in all stages
        @inbounds for t in 1:num_stages
            stage_pairs = window_state_out_params[t]
            @inbounds for i in eachindex(stage_pairs)
                target_param = stage_pairs[i][1]
                d_targets[t][i] = Float32(DiffOpt.get_reverse_parameter(window_model, target_param))
            end
        end

        DiffOpt.empty_input_sensitivities!(window_model)

        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), d_s_in, d_targets)
    end

    return s_out, pullback
end

#=============================================================================
 Window data structure + setup
=============================================================================#

struct WindowData
    model::JuMP.Model
    state_in_params::Vector{Any}                         # window-start state parameters (MOI.Parameter vars)
    state_out_params::Vector{Vector{Tuple{Any, VariableRef}}} # per-stage (target_param, realized_var)
    uncertainty_params::Vector{Vector{VariableRef}}       # per-stage uncertainty parameters (in the window model)
    stage_range::UnitRange{Int}                           # global stages covered by this window
end

"""
ChainRulesCore.rrule(::typeof(set_window_uncertainties!), window::WindowData, uncertainty_sample)

Declare set_window_uncertainties! as non-differentiable (mutates solver state).
"""
ChainRulesCore.rrule(::typeof(set_window_uncertainties!), window::WindowData, uncertainty_sample) = begin
    set_window_uncertainties!(window, uncertainty_sample)
    function pullback(::Any)
        return (NoTangent(), NoTangent(), NoTangent())
    end
    return nothing, pullback
end

"""
    setup_shooting_windows(subproblems, state_params_in, state_params_out, initial_state,
                           uncertainties; window_size, optimizer_factory=nothing)

Build window models for multiple shooting.

Notes:
- We store only the uncertainty PARAMETER refs (not sample sets) in WindowData.
"""
function setup_shooting_windows(
    subproblems::Vector{JuMP.Model},
    state_params_in::Vector{Vector{U}},
    state_params_out::Vector{Vector{Tuple{U, VariableRef}}},
    initial_state::Vector{Float64},
    uncertainties;  # typically Vector{Vector{Tuple{VariableRef,Vector{Float64}}}} or similar
    window_size::Int,
    optimizer_factory=nothing,
) where {U}

    num_stages = length(subproblems)
    num_windows = ceil(Int, num_stages / window_size)

    windows = Vector{WindowData}(undef, num_windows)

    for w in 1:num_windows
        window_start = (w - 1) * window_size + 1
        window_end = min(w * window_size, num_stages)
        stage_range = window_start:window_end

        window_subproblems = subproblems[stage_range]
        window_state_params_in  = [state_params_in[t] for t in stage_range]
        window_state_params_out = [state_params_out[t] for t in stage_range]
        window_uncertainties = uncertainties[stage_range]

        window_model = JuMP.Model()
        if optimizer_factory !== nothing
            set_optimizer(window_model, optimizer_factory)
        end

        # Build deterministic equivalent window model.
        # This is expected to mutate window_state_params_in/out to refer to params/vars in window_model.
        window_model, window_uncertainties_new = deterministic_equivalent!(
            window_model,
            window_subproblems,
            window_state_params_in,
            window_state_params_out,
            initial_state,
            window_uncertainties,
        )

        # Only first stage has separate state_in params
        state_in_params = window_state_params_in[1]
        ensure_state_in_is_parameter!(window_model, state_in_params)
        ensure_target_params_are_parameters!(window_model, window_state_params_out)

        uncertainty_params = extract_uncertainty_params(window_uncertainties_new)

        windows[w] = WindowData(
            window_model,
            state_in_params,
            window_state_params_out,
            uncertainty_params,
            stage_range,
        )
    end

    return windows
end

#=============================================================================
 Policy rollouts inside a window
=============================================================================#

"""
    predict_window_targets(decision_rule, s_in, uncertainties_vec)

Predict one target per stage in a window. This is an AD-friendly scan:
target_1 = π([u1; s_in])
target_2 = π([u2; target_1])
...
"""
function predict_window_targets(
    decision_rule,
    s_in::AbstractVector{T},
    uncertainties_vec::Vector{<:AbstractVector},
) where {T}
    function step(prev_state, uncertainty)
        return decision_rule(vcat(uncertainty, prev_state))
    end
    return accumulate(step, uncertainties_vec; init=s_in)
end

#=============================================================================
 Simulation (forward) across windows
=============================================================================#

"""
    simulate_multiple_shooting(windows, decision_rule, initial_state, uncertainty_sample, uncertainties_vec)

- `uncertainty_sample`: per-stage sampled tuples (param, value) matching your existing sampler output:
    Vector{Vector{Tuple{VariableRef,<:Real}}}
- `uncertainties_vec`: per-stage vectors (Float32) used as policy inputs

Returns total objective across windows. Gradients flow through:
- targets within each window (via solve_window rrule)
- realized end state between windows (via solve_window rrule seeding on end vars)
"""
function simulate_multiple_shooting(
    windows::Vector{WindowData},
    decision_rule,
    initial_state::AbstractVector{T},
    uncertainty_sample,
    uncertainties_vec,
) where {T}
    total_objective = zero(T)
    current_real_state = initial_state

    for window in windows
        window_range = window.stage_range
        window_uncertainties_vec = uncertainties_vec[window_range]

        # Predict targets for this window
        targets = predict_window_targets(decision_rule, current_real_state, window_uncertainties_vec)

        # Set sampled uncertainty values into the window model (parameters in window model)
        @ignore_derivatives set_window_uncertainties!(window, uncertainty_sample)

        # Solve window
        window_obj = solve_window(
            window.model,
            window.state_in_params,
            window.state_out_params,
            current_real_state,
            targets,
        )

        realized_state = get_last_realized_state(window.model,
            window.state_in_params,
            window.state_out_params,
            current_real_state,
            targets,
        )

        total_objective += window_obj
        current_real_state = realized_state
    end

    return total_objective
end

#=============================================================================
 Training loop
=============================================================================#

"""
    train_multiple_shooting(model, initial_state, windows, uncertainty_sampler; ...)

This mirrors your other training loops:
- Reuse pre-built window models.
- For each SGD step, sample uncertainties, build uncertainties_vec for the policy,
  evaluate simulate_multiple_shooting, and update parameters.
"""
function train_multiple_shooting(
    model,
    initial_state::Vector{<:Real},
    windows::Vector{WindowData},
    uncertainty_sampler;
    num_batches::Int=100,
    num_train_per_batch::Int=32,
    optimizer=Flux.Adam(0.01),
    adjust_hyperparameters=(iter, opt_state, n) -> n,
    record_loss=(iter, model, loss, tag) -> begin
        println("tag: $tag, Iter: $iter, Loss: $loss")
        return false
    end,
    get_objective_no_target_deficit=get_objective_no_target_deficit,
)
    opt_state = Flux.setup(optimizer, model)

    # We only need the uncertainty *structure* here.
    base_uncertainty = uncertainty_sampler()
    # If uncertainty values are vectors (sample sets), draw realized values per iteration.
    has_sample_sets = !isempty(base_uncertainty) &&
        !isempty(base_uncertainty[1]) &&
        (base_uncertainty[1][1][2] isa AbstractVector)
    draw_uncertainty = has_sample_sets ? (() -> DecisionRules.sample(base_uncertainty)) : uncertainty_sampler

    initial_state_f32 = Float32.(initial_state)

    for iter in 1:num_batches
        num_train_per_batch = adjust_hyperparameters(iter, opt_state, num_train_per_batch)

        objective = 0.0
        eval_loss = 0.0

        grads = Flux.gradient(model) do m
            for _ in 1:num_train_per_batch
                @ignore_derivatives Flux.reset!(m)

                uncertainty_sample = @ignore_derivatives draw_uncertainty()
                uncertainties_vec = [[Float32(u[2]) for u in stage_u] for stage_u in uncertainty_sample]

                objective += simulate_multiple_shooting(
                    windows, m, initial_state_f32, uncertainty_sample, uncertainties_vec
                )

                # evaluation metric (no deficit)
                eval_loss += @ignore_derivatives begin
                    # windows already contain the window models; compute metric by summing each window model's
                    # deficit-free objective after a forward pass
                    total = 0.0
                    current_state = initial_state_f32
                    for win in windows
                        # set uncertainties for this metric pass
                        set_window_uncertainties!(win, uncertainty_sample)

                        win_range = win.stage_range
                        win_uvec = uncertainties_vec[win_range]
                        targs = predict_window_targets(m, current_state, win_uvec)

                        obj = solve_window(
                            win.model, win.state_in_params, win.state_out_params, current_state, targs
                        )
                        current_state = get_last_realized_state(
                            win.model, win.state_in_params, win.state_out_params, current_state, targs
                        )
                        total += get_objective_no_target_deficit(win.model)
                    end
                    total
                end
            end
            objective /= num_train_per_batch
            eval_loss /= num_train_per_batch
            return objective
        end

        record_loss(iter, model, eval_loss, "metrics/loss") && break
        record_loss(iter, model, objective, "metrics/training_loss") && break

        grad = materialize_tangent(grads[1])
        Flux.update!(opt_state, model, grad)
    end

    return model
end
