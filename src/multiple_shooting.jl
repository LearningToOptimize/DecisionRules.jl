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
- `windows_equivalent!(window_model, window_subproblems, window_state_params_in,
   window_state_params_out, initial_state, window_uncertainties)` builds the window model
  without mutating the original subproblems and updates `window_state_params_in/out`
  to refer to parameters/variables in the new model.
- `window_state_params_out[t]` is a vector of tuples (target_param, realized_state_var) at stage t.
"""

using Base: accumulate
import Zygote

function _print_window_status_and_params(window, status; context::AbstractString="")
    header = isempty(context) ? "solve_window status" : "solve_window status ($context)"
    println("[$header] status=$(status) stage_range=$(window.stage_range)")
    params = [v for v in all_variables(window.model) if JuMP.is_parameter(v)]
    sort!(params, by=JuMP.name)
    for v in params
        val = try
            JuMP.parameter_value(v)
        catch
            try
                value(v)
            catch
                missing
            end
        end
        println("  ", JuMP.name(v), " = ", val)
    end
end

#=============================================================================
 Helpers
=============================================================================#

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

function _param_init_value(src::JuMP.VariableRef)
    if JuMP.is_parameter(src)
        try
            return JuMP.parameter_value(src)
        catch
            return 0.0
        end
    end
    return 0.0
end

function _as_float64_vec(val)
    if val isa AbstractVector
        return val isa Vector{Float64} ? val : Float64.(val)
    end
    return [Float64(val)]
end

function _create_like_variable(
    m::JuMP.Model,
    src::JuMP.VariableRef,
    t::Int;
    force_parameter::Bool=false,
)
    if force_parameter || JuMP.is_parameter(src)
        dest = @variable(m, set=MOI.Parameter(_param_init_value(src)))
    else
        dest = @variable(m)
    end
    DecisionRules.var_set_name!(src, dest, t)
    return dest
end

"""
    windows_equivalent!(model, subproblems, state_params_in, state_params_out, initial_state, uncertainties)

Create a window equivalent without mutating the original subproblems and without
adding extra variables/constraints beyond those already present in the subproblems.
"""
function windows_equivalent!(
    model::JuMP.Model,
    subproblems::Vector{JuMP.Model},
    state_params_in::Vector,
    state_params_out::Vector,
    initial_state::Vector{Float64},
    uncertainties,
)
    num_stages = length(subproblems)
    set_objective_sense(model, objective_sense(subproblems[1]))

    var_src_to_dest = Dict{VariableRef, VariableRef}()
    state_in_new = Vector{Vector{Any}}(undef, num_stages)
    state_out_new = Vector{Vector{Tuple{Any, VariableRef}}}(undef, num_stages)
    uncertainties_new = Vector{Vector{Tuple{Any, Vector{Float64}}}}(undef, num_stages)

    for t in 1:num_stages
        subproblem = subproblems[t]

        state_in_vars = [v for v in state_params_in[t] if v isa VariableRef]
        state_out_targets = [pair[1] for pair in state_params_out[t] if pair[1] isa VariableRef]
        state_out_realized = [pair[2] for pair in state_params_out[t]]

        allvars = all_variables(subproblem)
        allvars = setdiff(allvars, state_in_vars)
        allvars = setdiff(allvars, state_out_targets)
        allvars = setdiff(allvars, state_out_realized)

        for src in allvars
            if !haskey(var_src_to_dest, src)
                dest = _create_like_variable(model, src, t)
                var_src_to_dest[src] = dest
            end
        end

        # state_out params (target param + realized var)
        state_out_new[t] = Vector{Tuple{Any, VariableRef}}(undef, length(state_params_out[t]))
        for (i, pair) in enumerate(state_params_out[t])
            target_src, realized_src = pair
            dest_realized = _create_like_variable(model, realized_src, t; force_parameter=false)
            var_src_to_dest[realized_src] = dest_realized

            if target_src isa VariableRef
                dest_target = _create_like_variable(model, target_src, t; force_parameter=true)
                var_src_to_dest[target_src] = dest_target
                state_out_new[t][i] = (dest_target, dest_realized)
            else
                state_out_new[t][i] = (target_src, dest_realized)
            end
        end

        # state_in params
        state_in_new[t] = Vector{Any}(undef, length(state_params_in[t]))
        if t == 1
            for (i, src) in enumerate(state_params_in[t])
                if src isa VariableRef
                    dest = _create_like_variable(model, src, t; force_parameter=true)
                    var_src_to_dest[src] = dest
                    state_in_new[t][i] = dest
                else
                    state_in_new[t][i] = src
                end
            end
        else
            for (i, src) in enumerate(state_params_in[t])
                state_in_new[t][i] = state_out_new[t - 1][i][2]
                if src isa VariableRef
                    var_src_to_dest[src] = state_in_new[t][i]
                end
            end
        end

        # uncertainties
        uncertainties_new[t] = Vector{Tuple{Any, Vector{Float64}}}(undef, length(uncertainties[t]))
        for (i, tup) in enumerate(uncertainties[t])
            u_src, u_vals = tup
            if u_src isa VariableRef
                dest = get(var_src_to_dest, u_src, nothing)
                if dest === nothing
                    dest = _create_like_variable(model, u_src, t; force_parameter=true)
                    var_src_to_dest[u_src] = dest
                end
                uncertainties_new[t][i] = (dest, _as_float64_vec(u_vals))
            else
                uncertainties_new[t][i] = (u_src, _as_float64_vec(u_vals))
            end
        end

        # constraints
        cons_to_cons = Dict{Any, Any}()
        for con in JuMP.all_constraints(subproblem; include_variable_in_set_constraints=true)
            obj = JuMP.constraint_object(con)
            if obj.func isa VariableRef && obj.set isa MOI.Parameter
                continue
            end
            c = DecisionRules.create_constraint(model, obj, var_src_to_dest)
            cons_to_cons[con] = c
        end

        # map any constraint-based targets/state_in/uncertainties
        for i in eachindex(state_out_new[t])
            target = state_out_new[t][i][1]
            if target isa ConstraintRef && haskey(cons_to_cons, target)
                state_out_new[t][i] = (cons_to_cons[target], state_out_new[t][i][2])
            end
        end
        if t == 1
            for i in eachindex(state_in_new[t])
                src = state_in_new[t][i]
                if src isa ConstraintRef && haskey(cons_to_cons, src)
                    state_in_new[t][i] = cons_to_cons[src]
                end
            end
        end
        for i in eachindex(uncertainties_new[t])
            src = uncertainties_new[t][i][1]
            if src isa ConstraintRef && haskey(cons_to_cons, src)
                uncertainties_new[t][i] = (cons_to_cons[src], uncertainties_new[t][i][2])
            end
        end

        # objective
        current = JuMP.objective_function(model)
        sub_obj = DecisionRules.copy_and_replace_variables(JuMP.objective_function(subproblem), var_src_to_dest)
        JuMP.set_objective_function(model, current + sub_obj)
    end

    return model, state_in_new, state_out_new, uncertainties_new
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
    targets::AbstractVector,
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
        tgt_vec = tgt isa AbstractVector ? tgt : (tgt,)
        @inbounds for i in eachindex(stage_pairs)
            target_param = stage_pairs[i][1]
            set_parameter_value(target_param, tgt_vec[i])
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
    targets::AbstractVector,
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
    targets::AbstractVector,
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
    targets::AbstractVector,
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

        # Build window equivalent model without mutating the originals.
        window_model,
        window_state_params_in,
        window_state_params_out,
        window_uncertainties_new = windows_equivalent!(
            window_model,
            window_subproblems,
            window_state_params_in,
            window_state_params_out,
            initial_state,
            window_uncertainties,
        )

        # Only first stage has separate state_in params
        state_in_params = window_state_params_in[1]

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
        @ignore_derivatives begin
            status = termination_status(window.model)
            if !(status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED))
                _print_window_status_and_params(window, status; context="simulate_multiple_shooting")
            end
        end

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

        grads = Flux.gradient(model) do m
            objective = 0.0
            for _ in 1:num_train_per_batch
                @ignore_derivatives Flux.reset!(m)

                uncertainty_sample = @ignore_derivatives draw_uncertainty()
                uncertainties_vec = [[Float32(u[2]) for u in stage_u] for stage_u in uncertainty_sample]

                objective += simulate_multiple_shooting(
                    windows, m, initial_state_f32, uncertainty_sample, uncertainties_vec
                )
            end
            objective /= num_train_per_batch
            return objective
        end

        eval_loss = @ignore_derivatives begin
            total = 0.0
            for _ in 1:num_train_per_batch
                Flux.reset!(model)
                uncertainty_sample = draw_uncertainty()
                uncertainties_vec = [[Float32(u[2]) for u in stage_u] for stage_u in uncertainty_sample]

                current_state = initial_state_f32
                for win in windows
                    set_window_uncertainties!(win, uncertainty_sample)

                    win_range = win.stage_range
                    win_uvec = uncertainties_vec[win_range]
                    targs = predict_window_targets(model, current_state, win_uvec)

                    solve_window(
                        win.model, win.state_in_params, win.state_out_params, current_state, targs
                    )
                    @ignore_derivatives begin
                        status = termination_status(win.model)
                        if !(status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED))
                            _print_window_status_and_params(win, status; context="train_multiple_shooting")
                        end
                    end
                    current_state = get_last_realized_state(
                        win.model, win.state_in_params, win.state_out_params, current_state, targs
                    )
                    total += get_objective_no_target_deficit(win.model)
                end
            end
            total / num_train_per_batch
        end

        record_loss(iter, model, eval_loss, "metrics/loss") && break
        record_loss(iter, model, objective, "metrics/training_loss") && break

        grad = materialize_tangent(grads[1])
        Flux.update!(opt_state, model, grad)
    end

    return model
end
