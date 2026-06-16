function simulate_states(
    initial_state::Vector{T},
    uncertainties,
    decision_rule::F;
) where {F, T <: Real}
    num_stages = length(uncertainties)
    states = Vector{Vector{T}}(undef, num_stages + 1)
    states[1] = initial_state
    for stage in 1:num_stages
        uncertainties_stage = [uncertainties[stage][i][2] for i in 1:length(uncertainties[stage])]
        # Input: [uncertainty, previous_predicted_state]
        # For stage 1, previous_predicted_state = initial_state
        # For stage t > 1, previous_predicted_state = states[t] (output from previous stage)
        prev_state = states[stage]
        states[stage + 1] = decision_rule(vcat(uncertainties_stage, prev_state))
    end
    return states
end

function simulate_states(
    initial_state::Vector{T},
    uncertainties,
    decision_rules::Vector{F};
) where {F, T <: Real}
    num_stages = length(uncertainties)
    states = Vector{Vector{T}}(undef, num_stages + 1)
    states[1] = initial_state
    for stage in 1:num_stages
        uncertainties_stage = [uncertainties[stage][i][2] for i in 1:length(uncertainties[stage])]
        decision_rule = decision_rules[stage]
        # Input: [uncertainty, previous_predicted_state]
        prev_state = states[stage]
        states[stage + 1] = decision_rule(vcat(uncertainties_stage, prev_state))
    end
    return states
end

function simulate_stage(subproblem::JuMP.Model, state_param_in::Vector{Any}, state_param_out::Vector{Tuple{Any, VariableRef}}, uncertainty::Vector{Tuple{VariableRef,T}}, state_in::Vector{Z}, state_out_target::Vector{V}
) where {T <: Real, V <: Real, Z <: Real}
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

    # Solve subproblem
    optimize!(subproblem)

    # objective value
    obj = objective_value(subproblem)

    return obj
end

function get_next_state(subproblem::JuMP.Model, state_param_in::Vector{Any}, state_param_out::Vector{Tuple{Any, VariableRef}}, state_in::Vector{T}, state_out_target::Vector{Z}) where {T <: Real, Z <: Real}
    state_out = [value(state_param_out[i][2]) for i in 1:length(state_param_out)]
    return state_out
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
function ChainRulesCore.rrule(::typeof(get_next_state),
    subproblem::JuMP.Model,
    state_param_in,
    state_param_out,
    state_in,
    state_out_target
)

    # Forward pass: run the solver via the user's function
    y = DecisionRules.get_next_state(subproblem, state_param_in, state_param_out, state_in, state_out_target)

    function pullback(Δy)
        try
            Δy = collect(Δy)  # ensure indexable, concrete element type

            # Best practice: clear previous seeds
            DiffOpt.empty_input_sensitivities!(subproblem)

            # 1) Seed reverse on the realized output variables with Δy
            #    Each entry in state_param_out is (param_target, realized_state_var)
            @inbounds for i in eachindex(state_param_out)
                realized_var = state_param_out[i][2]
                # J' * Δ: set reverse seed on variable primal
                DiffOpt.set_reverse_variable(subproblem, realized_var, Δy[i])
            end

            # 2) Reverse differentiate
            DiffOpt.reverse_differentiate!(subproblem)  # computes all needed products

            # 3) Read sensitivities w.r.t. parameter variables
            #    These are vector-Jacobian products dL/d(param) = (∂y/∂param)^T * Δy
            d_state_in = similar(state_in, promote_type(eltype(state_in), eltype(Δy)))
            @inbounds for i in eachindex(state_param_in)
                pin = state_param_in[i]  # JuMP.Parameter variable            
                d_state_in[i] = DiffOpt.get_reverse_parameter(subproblem, pin)
            end

            d_state_out_target = similar(state_out_target, promote_type(eltype(state_out_target), eltype(Δy)))
            @inbounds for i in eachindex(state_param_out)
                pout = state_param_out[i][1]  # target Parameter variable
                d_state_out_target[i] = DiffOpt.get_reverse_parameter(subproblem, pout)
            end

            # Optional: clear seeds so they don't accumulate between calls
            DiffOpt.empty_input_sensitivities!(subproblem)

            # Return cotangents for each primal argument, in order:
            #  (f, subproblem, state_param_in, state_param_out, state_in, state_out_target)
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), d_state_in, d_state_out_target)
        catch e
            msg = sprint(showerror, e)
            throw(ArgumentError(
                "Differentiating get_next_state requires a DiffOpt-enabled model " *
                "because the closed-loop rollout needs solution sensitivities of the " *
                "realized state variables. Use an appropriate DiffOpt wrapper for the " *
                "stage subproblems (for target-slack conic models, " *
                "`DiffOpt.conic_diff_model(...)`), or use the deterministic-equivalent " *
                "training path when only target duals are needed. Original error: $msg"
            ))
        end      
    end

    return y, pullback
end

function get_objective_no_target_deficit(subproblem::JuMP.Model; norm_deficit::AbstractString="norm_deficit")
    obj = JuMP.objective_function(subproblem)
    objective_val = objective_value(subproblem)
    for term in obj.terms
        if occursin(norm_deficit, JuMP.name(term[1]))
            objective_val -= term[2] * value(term[1])
        end
    end
    return objective_val
end

function get_objective_no_target_deficit(subproblems::Vector{JuMP.Model}; norm_deficit::AbstractString="norm_deficit")
    total_objective = 0.0
    for subproblem in subproblems
        total_objective += get_objective_no_target_deficit(subproblem, norm_deficit=norm_deficit)
    end
    return total_objective
end

# define ChainRulesCore.rrule of get_objective_no_target_deficit
function ChainRulesCore.rrule(::typeof(get_objective_no_target_deficit), subproblem; norm_deficit="norm_deficit")
    objective_val = get_objective_no_target_deficit(subproblem, norm_deficit=norm_deficit)
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

function simulate_multistage(
    subproblems::Vector{JuMP.Model},
    state_params_in::Vector{Vector{U}},
    state_params_out::Vector{Vector{Tuple{U, VariableRef}}},
    initial_state::Vector{T},
    uncertainties,
    decision_rules
) where {T <: Real, U}
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
        objective_value += simulate_stage(subproblem, state_param_in, state_param_out, uncertainty, state_in, state_out)
        state_in = DecisionRules.get_next_state(subproblem, state_param_in, state_param_out, state_in, state_out)
    end
    
    # Return final objective value
    return objective_value
end

function simulate_multistage(
    det_equivalent::JuMP.Model,
    state_params_in::Vector{Vector{Z}},
    state_params_out::Vector{Vector{Tuple{Z, VariableRef}}},
    uncertainties,
    states
    ) where {Z}
    
    for t in  1:length(state_params_in)
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

    # Solve det_equivalent
    optimize!(det_equivalent)

    return objective_value(det_equivalent)
end

function simulate_multistage(
    subproblems::JuMP.Model,
    state_params_in::Vector{Vector{U}},
    state_params_out::Vector{Vector{Tuple{U, VariableRef}}},
    initial_state::Vector{T},
    uncertainties,
    decision_rules
) where {T <: Real, U}
    Flux.reset!(decision_rules)
    states = simulate_states(initial_state, uncertainties, decision_rules)
    return simulate_multistage(subproblems, state_params_in, state_params_out, uncertainties, states)
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
function ChainRulesCore.rrule(::typeof(simulate_stage), subproblem, state_param_in, state_param_out, uncertainty, state_in, state_out)
    y = simulate_stage(subproblem, state_param_in, state_param_out, uncertainty, state_in, state_out)
    function _pullback(Δy)
        status = @ignore_derivatives JuMP.termination_status(subproblem)
        if !(status in _SUCCESSFUL_TERM_STATUSES)
            if STRICT_GRADIENTS[]
                error("simulate_stage pullback: solver terminated with status $status; " *
                      "expected a successful solve. Set DecisionRules.STRICT_GRADIENTS[] " *
                      "= false to return zero gradients instead.")
            end
            @warn "simulate_stage: solver status $status, returning zero gradients" status
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),
                    zeros(length(state_param_in)), zeros(length(state_param_out)))
        end

        # Preferred: parameter duals (closed-form, Eq. 1.2 / 2.5)
        try
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),
                    pdual.(state_param_in) * Δy,
                    pdual.([s[1] for s in state_param_out]) * Δy)
        catch
            # pdual unavailable (e.g. DiffOpt wrapper hides conic duals)
        end

        # Fallback: DiffOpt reverse differentiation (same math, implicit diff of KKT)
        DiffOpt.empty_input_sensitivities!(subproblem)
        MOI.set(subproblem, DiffOpt.ReverseObjectiveSensitivity(), Δy)
        DiffOpt.reverse_differentiate!(subproblem)
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),
                DiffOpt.get_reverse_parameter.(subproblem, state_param_in),
                DiffOpt.get_reverse_parameter.(subproblem, [s[1] for s in state_param_out]))
    end
    return y, _pullback
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
function ChainRulesCore.rrule(::typeof(simulate_multistage), det_equivalent::JuMP.Model, state_params_in, state_params_out, uncertainties, states)
    y = simulate_multistage(det_equivalent, state_params_in, state_params_out, uncertainties, states)
    function _pullback(Δy)
        status = @ignore_derivatives JuMP.termination_status(det_equivalent)
        Δ_states = similar(states)
        if !(status in _SUCCESSFUL_TERM_STATUSES)
            if STRICT_GRADIENTS[]
                error("simulate_multistage (det_eq) pullback: solver terminated with " *
                      "status $status; expected a successful solve.")
            end
            @warn "simulate_multistage (det_eq): solver status $status, returning zero gradients" status
            Δ_states[1] = zeros(length(state_params_in[1]))
            for t in 1:length(state_params_out)
                Δ_states[t + 1] = zeros(length(state_params_out[t]))
            end
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), Δ_states)
        end

        # Preferred: parameter duals (closed-form, Eq. 1.2)
        try
            Δ_states[1] = pdual.(state_params_in[1])
            for t in 1:length(state_params_out)
                Δ_states[t + 1] = pdual.([s[1] for s in state_params_out[t]])
            end
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), Δ_states * Δy)
        catch
            # pdual unavailable; fall back to DiffOpt
        end

        # Fallback: DiffOpt reverse differentiation
        DiffOpt.empty_input_sensitivities!(det_equivalent)
        MOI.set(det_equivalent, DiffOpt.ReverseObjectiveSensitivity(), Δy)
        DiffOpt.reverse_differentiate!(det_equivalent)
        Δ_states[1] = DiffOpt.get_reverse_parameter.(det_equivalent, state_params_in[1])
        for t in 1:length(state_params_out)
            Δ_states[t + 1] = DiffOpt.get_reverse_parameter.(det_equivalent, [s[1] for s in state_params_out[t]])
        end
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), Δ_states)
    end
    return y, _pullback
end

function sample(uncertainty_samples::Vector{Tuple{VariableRef, Vector{T}}}) where {T<:Real}
    uncertainty_sample = Vector{Tuple{VariableRef,T}}(undef, length(uncertainty_samples))
    for i in 1:length(uncertainty_samples)
        uncertainty_sample[i] = (uncertainty_samples[i][1], rand(uncertainty_samples[i][2]))
    end
    return uncertainty_sample
end

function sample(uncertainty_samples::Vector{Vector{Tuple{VariableRef, Vector{T}}}}) where {T<:Real}
    [sample(uncertainty_samples[t]) for t in 1:length(uncertainty_samples)]
end

function train_multistage(model, initial_state, subproblems::Vector{JuMP.Model},
    state_params_in, state_params_out, uncertainty_sampler;
    num_batches=100, num_train_per_batch=32, optimizer=Flux.Adam(0.01),
    adjust_hyperparameters=(iter, opt_state, num_train_per_batch) -> num_train_per_batch,
    record_loss=nothing,
    get_objective_no_target_deficit=get_objective_no_target_deficit,
    sample_log=SampleLog(objective_no_deficit_fn=get_objective_no_target_deficit),
    record=default_record,
    penalty_schedule=nothing
)
    record = _resolve_record(record, record_loss)
    # Initialise the optimiser for this model:
    opt_state = Flux.setup(optimizer, model)

    schedule = _resolve_penalty_schedule(penalty_schedule, num_batches)
    penalty_bases = isnothing(schedule) ? nothing : _check_deficit_penalty_bases(_deficit_penalty_bases(subproblems))
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
        # Sample uncertainties
        uncertainty_samples = [sample(uncertainty_sampler) for _ in 1:num_train_per_batch]
        objective = 0.0
        _reset_sample_log!(sample_log)
        grads = Flux.gradient(model) do m
            for s in 1:num_train_per_batch
                Flux.reset!(m)
                objective += simulate_multistage(subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples[s], m)
                @ignore_derivatives sample_log(s, subproblems)
            end
            objective /= num_train_per_batch
            return objective
        end
        record(sample_log, iter, model) && break

        # Update the parameters so as to reduce the objective,
        # according the chosen optimisation rule:
        # Convert gradients from MutableTangent to plain NamedTuples for Flux.update!
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

function train_multistage(model, initial_state, det_equivalent::JuMP.Model,
    state_params_in, state_params_out, uncertainty_sampler;
    num_batches=100, num_train_per_batch=32, optimizer=Flux.Adam(0.01),
    adjust_hyperparameters=(iter, opt_state, num_train_per_batch) -> num_train_per_batch,
    record_loss=nothing,
    get_objective_no_target_deficit=get_objective_no_target_deficit,
    sample_log=SampleLog(objective_no_deficit_fn=get_objective_no_target_deficit),
    record=default_record,
    penalty_schedule=nothing
)
    record = _resolve_record(record, record_loss)
    # Initialise the optimiser for this model:
    opt_state = Flux.setup(optimizer, model)
    num_stages = length(state_params_in)

    schedule = _resolve_penalty_schedule(penalty_schedule, num_batches)
    penalty_bases = isnothing(schedule) ? nothing : _check_deficit_penalty_bases(_deficit_penalty_bases(det_equivalent))
    current_multiplier = NaN

    for iter in 1:num_batches
        if !isnothing(schedule)
            multiplier = _penalty_multiplier_for(schedule, iter)
            if multiplier != current_multiplier
                _apply_deficit_penalty_multiplier!(det_equivalent, penalty_bases, multiplier)
                current_multiplier = multiplier
            end
        end
        num_train_per_batch = adjust_hyperparameters(iter, opt_state, num_train_per_batch)
        # Sample uncertainties
        uncertainty_samples = [sample(uncertainty_sampler) for _ in 1:num_train_per_batch]
        num_uncertainties = length(uncertainty_samples[1][1])
        uncertainty_samples_vec = [[[uncertainty_samples[s][stage][i][2] for i in 1:num_uncertainties] for stage in 1:length(uncertainty_samples[1])] for s in 1:num_train_per_batch]

        # Calculate the gradient of the objective
        # with respect to the parameters within the model:
        objective = 0.0
        _reset_sample_log!(sample_log)
        # try
            grads = Flux.gradient(model) do m
                for s in 1:num_train_per_batch
                    Flux.reset!(m)
                    # Compute states using accumulate (AD-friendly scan operation)
                    # Input to policy: [uncertainty, previous_predicted_state]
                    init_state = Float32.(initial_state)
                    # accumulate returns all intermediate states
                    predicted_states = accumulate(uncertainty_samples_vec[s]; init=init_state) do prev_state, uncertainties_t
                        m(vcat(uncertainties_t, prev_state))
                    end
                    # Combine initial state with predicted states
                    states = vcat([init_state], predicted_states)
                    objective += simulate_multistage(det_equivalent, state_params_in, state_params_out, uncertainty_samples[s], states)
                    @ignore_derivatives sample_log(s, det_equivalent)
                end
                objective /= num_train_per_batch
                return objective
            end
        # catch
            # continue;
        # end
        record(sample_log, iter, model) && break

        # Update the parameters so as to reduce the objective,
        # according the chosen optimisation rule:
        # Convert gradients from MutableTangent to plain NamedTuples for Flux.update!
        grad = materialize_tangent(grads[1])
        Flux.update!(opt_state, model, grad)
    end
    
    return model
end

# function make_single_network(models::Vector{F}, number_of_states::Int) where {F}
#     size_m = length(models)
#     return Parallel(permutedims ∘ hcat, [Chain(
#         x -> x[1:number_of_states * (i + 1)],
#         models[i]
#     ) for i in 1:size_m]...)
# end

# function train_multistage(models::Vector, initial_state, subproblems::Vector{JuMP.Model}, 
#     state_params_in, state_params_out, uncertainty_sampler; 
#     num_batches=100, num_train_per_batch=32, optimizer=Flux.Adam(0.01),
#     adjust_hyperparameters=(iter, opt_state, num_train_per_batch) -> num_train_per_batch,
#     record_loss=(iter, model, loss, tag) -> begin println("tag: $tag, Iter: $iter, Loss: $loss")
#         return false
#     end,
#     get_objective_no_target_deficit=get_objective_no_target_deficit
# )
#     num_states = length(initial_state)
#     model = make_single_network(models, num_states)
#     # Initialise the optimiser for this model:
#     opt_state = Flux.setup(optimizer, model)

#     for iter in 1:num_batches
#         num_train_per_batch = adjust_hyperparameters(iter, opt_state, num_train_per_batch)
#         # Sample uncertainties
#         uncertainty_samples = [sample(uncertainty_sampler) for _ in 1:num_train_per_batch]
#         uncertainty_samples_vecs = [[collect(values(uncertainty_sample[j])) for j in 1:length(uncertainty_sample)] for uncertainty_sample in uncertainty_samples]
#         uncertainty_samples_vec = [vcat(initial_state, uncertainty_samples_vecs[s]...) for s in 1:num_train_per_batch]

#         # Calculate the gradient of the objective
#         # with respect to the parameters within the model:
#         eval_loss = 0.0
#         objective = 0.0
#         grads = Flux.gradient(model) do m
#             for s in 1:num_train_per_batch
#                 states = m(uncertainty_samples_vec[s])
#                 state_in = initial_state
#                 for (j, subproblem) in enumerate(subproblems)
#                     state_out = states[j]
#                     objective += simulate_stage(subproblem, state_params_in[j], state_params_out[j], uncertainty_samples[s][j], state_in, state_out)
#                     eval_loss += get_objective_no_target_deficit(subproblem)
#                     state_in = get_next_state(subproblem, state_params_out[j], state_in, state_out)
#                 end
#             end
#             objective /= num_train_per_batch
#             eval_loss /= num_train_per_batch
#             return objective
#         end
#         record_loss(iter, model, eval_loss, "metrics/loss") && break
#         record_loss(iter, model, objective, "metrics/training_loss") && break

#         # Update the parameters so as to reduce the objective,
#         # according the chosen optimisation rule:
#         Flux.update!(opt_state, model, grads[1])
#     end
    
#     return model
# end


# function train_multistage(models::Vector, initial_state, det_equivalent::JuMP.Model, 
#     state_params_in, state_params_out, uncertainty_sampler; 
#     num_batches=100, num_train_per_batch=32, optimizer=Flux.Adam(0.01),
#     adjust_hyperparameters=(iter, opt_state, num_train_per_batch) -> num_train_per_batch,
#     record_loss=(iter, model, loss, tag) -> begin println("tag: $tag, Iter: $iter, Loss: $loss")
#         return false
#     end,
#     get_objective_no_target_deficit=get_objective_no_target_deficit
# )
#     num_states = length(initial_state)
#     num_stages = length(state_params_in)
#     # Initialise the optimiser for each model:
#     opt_states = [Flux.setup(optimizer, m) for m in models]

#     for iter in 1:num_batches
#         num_train_per_batch = adjust_hyperparameters(iter, opt_states, num_train_per_batch)
#         # Sample uncertainties
#         uncertainty_samples = [sample(uncertainty_sampler) for _ in 1:num_train_per_batch]
#         num_uncertainties = length(uncertainty_samples[1][1])
#         uncertainty_samples_vecs = [[[uncertainty_samples[s][stage][i][2] for i in 1:num_uncertainties] for stage in 1:length(uncertainty_samples[1])] for s in 1:num_train_per_batch]

#         # Calculate the gradient of the objective
#         # with respect to the parameters within the model:
#         eval_loss = 0.0
#         objective = 0.0
#         grads = Flux.gradient(models...) do ms...
#             for s in 1:num_train_per_batch
#                 # Compute states sequentially: each state depends on the previous predicted state
#                 # Input to policy: [uncertainty, previous_predicted_state]
#                 states = Vector{Any}(undef, num_stages + 1)
#                 states[1] = Float32.(initial_state)
#                 for t in 2:num_stages + 1
#                     uncertainties_t = uncertainty_samples_vecs[s][t - 1]
#                     prev_state = states[t - 1]
#                     states[t] = ms[t - 1](vcat(uncertainties_t, prev_state))
#                 end
#                 objective += simulate_multistage(det_equivalent, state_params_in, state_params_out, uncertainty_samples[s], states)
#                 @ignore_derivatives eval_loss += get_objective_no_target_deficit(det_equivalent)
#             end
#             objective /= num_train_per_batch
#             @ignore_derivatives eval_loss /= num_train_per_batch
#             return objective
#         end
#         record_loss(iter, models, eval_loss, "metrics/loss") && break
#         record_loss(iter, models, objective, "metrics/training_loss") && break

#         # Update the parameters so as to reduce the objective,
#         # according the chosen optimisation rule:
#         for (i, m) in enumerate(models)
#             Flux.update!(opt_states[i], m, grads[i])
#         end
#     end
    
#     return models
# end
