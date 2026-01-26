"""
Multiple Shooting Implementation for Decision Rules Training.

Multiple shooting is a hybrid approach between:
- Direct transcription (deterministic equivalent): Full horizon optimization, all stages coupled
- Single shooting (per-stage): Each stage solved independently, gradients via DiffOpt

Multiple shooting solves N windows of K stages each:
- Uses decision rule on REAL state once per window (start of each window)
- Uses `accumulate` to propagate predicted states within each window
- Backpropagation: Only uses DiffOpt for first state_in and last state_out per window

Benefits:
- More stable gradients than per-stage (fewer DiffOpt calls)
- More memory efficient than full det_eq (smaller coupled problems)
- Allows warmstarting from actual system state periodically
"""

# Note: This file is included in DecisionRules.jl module, so we have access to:
# - JuMP, Flux, ChainRulesCore, DiffOpt (from module using statements)
# - simulate_stage, sample, pdual, materialize_tangent, get_objective_no_target_deficit (from simulate_multistage.jl)
# - deterministic_equivalent!, find_variables (from utils.jl)

using Base: accumulate

"""
    create_shooting_window(subproblems, state_params_in, state_params_out, window_start, window_size)

Create a deterministic equivalent model for a window of K stages starting at `window_start`.
Returns the model, the state parameters mapped to the new model, and the uncertainty samples.
"""
function create_shooting_window!(
    model::JuMP.Model,
    subproblems::Vector{JuMP.Model},
    state_params_in::Vector{Vector{U}},
    state_params_out::Vector{Vector{Tuple{U, VariableRef}}},
    initial_state::Vector{Float64},
    uncertainties::Vector{Vector{Tuple{VariableRef, V}}},
    window_start::Int,
    window_size::Int
) where {U, V}
    window_end = min(window_start + window_size - 1, length(subproblems))
    actual_window_size = window_end - window_start + 1
    
    # Extract subproblems for this window
    window_subproblems = subproblems[window_start:window_end]
    window_state_params_in = state_params_in[window_start:window_end]
    window_state_params_out = state_params_out[window_start:window_end]
    window_uncertainties = uncertainties[window_start:window_end]
    
    # Use deterministic_equivalent! to create the window model
    model, uncertainties_new = deterministic_equivalent!(
        model,
        window_subproblems,
        window_state_params_in,
        window_state_params_out,
        initial_state,
        window_uncertainties
    )
    
    return model, uncertainties_new
end

"""
    simulate_shooting_window(window_model, state_params_in, state_params_out, uncertainties, states)

Simulate a shooting window with the given states (predicted by the decision rule).
Returns the objective value for the window.
"""
function simulate_shooting_window(
    window_model::JuMP.Model,
    state_params_in::Vector{Vector{Z}},
    state_params_out::Vector{Vector{Tuple{Z, VariableRef}}},
    uncertainties,
    states  # Length = num_stages_in_window + 1
) where {Z}
    
    num_stages = length(state_params_in)
    
    for t in 1:num_stages
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

    # Solve window model
    optimize!(window_model)

    return objective_value(window_model)
end

"""
ChainRulesCore.rrule for simulate_shooting_window.
Only computes gradients for the first state (window start) and last state (window end).
"""
function ChainRulesCore.rrule(
    ::typeof(simulate_shooting_window),
    window_model::JuMP.Model,
    state_params_in,
    state_params_out,
    uncertainties,
    states
)
    y = simulate_shooting_window(window_model, state_params_in, state_params_out, uncertainties, states)
    
    function _pullback(Δy)
        num_stages = length(state_params_in)
        Δ_states = [zeros(length(states[i])) for i in 1:length(states)]
        
        try
            # Only compute gradients for first and last states
            Δ_states[1] = pdual.(state_params_in[1]) * Δy
            Δ_states[end] = pdual.([s[1] for s in state_params_out[end]]) * Δy
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), Δ_states)
        catch
            DiffOpt.empty_input_sensitivities!(window_model)
            MOI.set(
                window_model,
                DiffOpt.ReverseObjectiveSensitivity(),
                Δy,
            )
            DiffOpt.reverse_differentiate!(window_model)
            
            # Only get gradients for first state_in and last state_out
            Δ_states[1] = DiffOpt.get_reverse_parameter.(window_model, state_params_in[1])
            Δ_states[end] = DiffOpt.get_reverse_parameter.(window_model, [s[1] for s in state_params_out[end]])
            
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), Δ_states)
        end
    end
    
    return y, _pullback
end

"""
    predict_window_states(decision_rule, window_start_state, window_uncertainties)

Predict states for a shooting window using the decision rule.
Uses `accumulate` to propagate states within the window (AD-friendly).

Returns: Vector of states of length (num_stages_in_window + 1)
"""
function predict_window_states(
    decision_rule,
    window_start_state::Vector{Float32},
    window_uncertainties_vec::Vector{Vector{Float32}}
)
    # Use accumulate to compute all states in the window
    # This is AD-friendly and computes: s[t+1] = decision_rule([u[t], s[t]])
    predicted_states = accumulate(window_uncertainties_vec; init=window_start_state) do prev_state, uncertainties_t
        decision_rule(vcat(uncertainties_t, prev_state))
    end
    
    # Combine initial state with predicted states
    return vcat([window_start_state], predicted_states)
end

"""
    simulate_multiple_shooting(window_models, state_params_in_windows, state_params_out_windows,
                               initial_state, uncertainties_windows, decision_rule)

Simulate multiple shooting across all windows.
Each window gets the REAL state from the previous window's optimization result.
"""
function simulate_multiple_shooting(
    window_models::Vector{JuMP.Model},
    state_params_in_windows::Vector{Vector{Vector{Z}}},
    state_params_out_windows::Vector{Vector{Vector{Tuple{Z, VariableRef}}}},
    initial_state::Vector{T},
    uncertainties_windows,  # Vector of uncertainties per window
    uncertainties_windows_vec,  # Vector of uncertainty values (for decision rule input)
    decision_rule
) where {T <: Real, Z}
    
    @ignore_derivatives Flux.reset!(decision_rule)
    
    num_windows = length(window_models)
    total_objective = zero(T)
    
    # Start with initial state
    current_real_state = Float32.(initial_state)
    
    for w in 1:num_windows
        # Predict states for this window starting from current real state
        window_states = predict_window_states(
            decision_rule,
            current_real_state,
            uncertainties_windows_vec[w]
        )
        
        # Simulate this window
        window_objective = simulate_shooting_window(
            window_models[w],
            state_params_in_windows[w],
            state_params_out_windows[w],
            uncertainties_windows[w],
            window_states
        )
        
        total_objective += window_objective
        
        # Get the REAL state from the optimization result (last stage of window)
        if w < num_windows
            current_real_state = get_window_final_real_state(
                window_models[w],
                state_params_out_windows[w]
            )
        end
    end
    
    return total_objective
end

"""
    get_window_final_real_state(subproblems, state_params_in, state_params_out, state_in, state_out_target)

Get the actual state values from the last stage of a window after optimization.
This is the "real" state that the next window will start from.

This follows the same pattern as `get_next_state` in simulate_multistage.jl:
- Forward: returns the realized state variables from the optimization
- Backward: uses DiffOpt to compute ∂(realized_state)/∂(target_state) and ∂(realized_state)/∂(state_in)

This enables the gradient chain:
  ∂(future_obj)/∂(target_states) = ∂(future_obj)/∂(next_state_in) × ∂(realized_state)/∂(target_states)
"""
function get_window_final_real_state(
    subproblem::JuMP.Model,
    state_params_in::Vector,
    state_params_out::Vector{Tuple{Z, VariableRef}},
    state_in::Vector{T},
    state_out_target::Vector{V}
) where {Z, T <: Real, V <: Real}
    # Get the realized state variable values (after optimization was run in simulate_stage)
    realized_state = Float32[value(var) for (param, var) in state_params_out]
    return realized_state
end

"""
ChainRulesCore.rrule for get_window_final_real_state.

Same pattern as get_next_state rrule:
- Seeds reverse on the realized output variables with Δy
- Calls DiffOpt.reverse_differentiate!
- Returns gradients w.r.t. state_in and state_out_target

This enables gradient flow between windows:
  ∂L/∂(target_states) += ∂L/∂(next_state_in) × ∂(realized_state)/∂(target_states)
"""
function ChainRulesCore.rrule(
    ::typeof(get_window_final_real_state),
    subproblem::JuMP.Model,
    state_params_in,
    state_params_out,
    state_in,
    state_out_target
)
    # Forward pass
    y = get_window_final_real_state(subproblem, state_params_in, state_params_out, state_in, state_out_target)
    
    function _pullback(Δy)
        try
            Δy = collect(Δy)  # ensure indexable
            
            # Clear previous seeds
            DiffOpt.empty_input_sensitivities!(subproblem)
            
            # Seed reverse on the realized output variables with Δy
            @inbounds for i in eachindex(state_params_out)
                realized_var = state_params_out[i][2]  # the variable (optimization output)
                DiffOpt.set_reverse_variable(subproblem, realized_var, Δy[i])
            end
            
            # Reverse differentiate
            DiffOpt.reverse_differentiate!(subproblem)
            
            # Get gradients w.r.t. state_in parameters
            d_state_in = similar(state_in, promote_type(eltype(state_in), eltype(Δy)))
            @inbounds for i in eachindex(state_params_in)
                d_state_in[i] = DiffOpt.get_reverse_parameter(subproblem, state_params_in[i])
            end
            
            # Get gradients w.r.t. state_out_target parameters
            d_state_out_target = similar(state_out_target, promote_type(eltype(state_out_target), eltype(Δy)))
            @inbounds for i in eachindex(state_params_out)
                target_param = state_params_out[i][1]  # the parameter (target)
                d_state_out_target[i] = DiffOpt.get_reverse_parameter(subproblem, target_param)
            end
            
            # Clear seeds
            DiffOpt.empty_input_sensitivities!(subproblem)
            
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), d_state_in, d_state_out_target)
            
        catch e
            @warn "get_window_final_real_state rrule fallback: $e"
            # Fallback: identity-like gradient
            d_state_in = zeros(eltype(Δy), length(state_in))
            d_state_out_target = collect(Δy)  # pass through to target
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), d_state_in, d_state_out_target)
        end
    end
    
    return y, _pullback
end

"""
    simulate_window_with_next_state(subproblems, state_params_in, state_params_out, 
                                     uncertainties, states)

Simulate a shooting window and return both the objective AND the realized final state.
This combined function enables proper gradient flow for the realized state.

Returns: (objective, realized_final_state)

The rrule computes gradients for both outputs:
- Gradient of objective w.r.t. states (for decision rule)
- Gradient of realized_final_state w.r.t. states (for next window's gradient flow)
"""
function simulate_window_with_next_state(
    subproblems::Vector{JuMP.Model},
    state_params_in,
    state_params_out,
    uncertainties,
    states  # Length = num_stages + 1
)
    num_stages = length(subproblems)
    total_objective = 0.0
    
    for t in 1:num_stages
        # Simulate this stage
        stage_objective = simulate_stage(
            subproblems[t],
            state_params_in[t],
            state_params_out[t],
            uncertainties[t],
            states[t],      # state_in
            states[t + 1]   # state_out (predicted by decision rule)
        )
        total_objective += stage_objective
    end
    
    # Get realized state from last stage
    last_stage_state_out = state_params_out[end]
    realized_final_state = Float32[value(var) for (param, var) in last_stage_state_out]
    
    return total_objective, realized_final_state
end

"""
ChainRulesCore.rrule for simulate_window_with_next_state.

Computes gradients for both the objective and the realized final state.
This enables gradient flow between windows when desired.

The key insight: the realized_final_state depends on the target states (from decision rule)
through the optimization. We use DiffOpt to compute ∂realized_state/∂target_state.
"""
function ChainRulesCore.rrule(
    ::typeof(simulate_window_with_next_state),
    subproblems,
    state_params_in,
    state_params_out,
    uncertainties,
    states
)
    obj, realized_state = simulate_window_with_next_state(
        subproblems, state_params_in, state_params_out, uncertainties, states
    )
    
    function _pullback(Δ)
        # Δ is a tuple: (Δobj, Δrealized_state)
        Δobj, Δrealized_state = Δ
        
        num_stages = length(subproblems)
        Δ_states = [zeros(Float32, length(states[i])) for i in 1:length(states)]
        
        # Part 1: Gradient from objective (same as simulate_window_stages)
        # Only compute for first state_in and last state_out
        if Δobj !== nothing && !iszero(Δobj)
            # First state gradient (from first subproblem)
            try
                Δ_states[1] .+= Float32.(pdual.(state_params_in[1])) * Δobj
            catch
                try
                    DiffOpt.empty_input_sensitivities!(subproblems[1])
                    MOI.set(subproblems[1], DiffOpt.ReverseObjectiveSensitivity(), Δobj)
                    DiffOpt.reverse_differentiate!(subproblems[1])
                    Δ_states[1] .+= Float32.(DiffOpt.get_reverse_parameter.(subproblems[1], state_params_in[1]))
                catch e
                    @warn "Failed to compute objective gradient for first state_in: $e"
                end
            end
            
            # Last state_out gradient (from last subproblem) 
            try
                Δ_states[end] .+= Float32.(pdual.([s[1] for s in state_params_out[end]])) * Δobj
            catch
                try
                    DiffOpt.empty_input_sensitivities!(subproblems[end])
                    MOI.set(subproblems[end], DiffOpt.ReverseObjectiveSensitivity(), Δobj)
                    DiffOpt.reverse_differentiate!(subproblems[end])
                    Δ_states[end] .+= Float32.(DiffOpt.get_reverse_parameter.(subproblems[end], [s[1] for s in state_params_out[end]]))
                catch e
                    @warn "Failed to compute objective gradient for last state_out: $e"
                end
            end
        end
        
        # Part 2: Gradient from realized_state (for inter-window gradient flow)
        # This is ∂realized_state/∂target_state - how the realized state depends on targets
        if Δrealized_state !== nothing && !all(iszero, Δrealized_state)
            Δrealized_state = collect(Δrealized_state)
            last_stage_state_out = state_params_out[end]
            
            try
                DiffOpt.empty_input_sensitivities!(subproblems[end])
                
                # Seed on realized variables (outputs of optimization)
                @inbounds for i in eachindex(last_stage_state_out)
                    realized_var = last_stage_state_out[i][2]
                    DiffOpt.set_reverse_variable(subproblems[end], realized_var, Δrealized_state[i])
                end
                
                DiffOpt.reverse_differentiate!(subproblems[end])
                
                # Get gradient w.r.t. target state parameters (last state in predicted states)
                @inbounds for i in eachindex(last_stage_state_out)
                    target_param = last_stage_state_out[i][1]
                    Δ_states[end][i] += Float32(DiffOpt.get_reverse_parameter(subproblems[end], target_param))
                end
                
                DiffOpt.empty_input_sensitivities!(subproblems[end])
            catch e
                @warn "Failed to compute realized_state gradient: $e"
            end
        end
        
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), Δ_states)
    end
    
    return (obj, realized_state), _pullback
end

"""
    setup_multiple_shooting(subproblems, state_params_in, state_params_out, initial_state, 
                            uncertainties, window_size; optimizer_factory)

Set up window models for multiple shooting.
Returns window_models, state_params_in_windows, state_params_out_windows, uncertainties_windows.
"""
function setup_multiple_shooting(
    subproblems::Vector{JuMP.Model},
    state_params_in::Vector{Vector{U}},
    state_params_out::Vector{Vector{Tuple{U, VariableRef}}},
    initial_state::Vector{Float64},
    uncertainties::Vector{Vector{Tuple{VariableRef, V}}};
    window_size::Int,
    optimizer_factory=nothing
) where {U, V}
    
    num_stages = length(subproblems)
    num_windows = ceil(Int, num_stages / window_size)
    
    window_models = Vector{JuMP.Model}(undef, num_windows)
    state_params_in_windows = Vector{Vector{Vector{Any}}}(undef, num_windows)
    state_params_out_windows = Vector{Vector{Vector{Tuple{Any, VariableRef}}}}(undef, num_windows)
    uncertainties_windows = Vector{Any}(undef, num_windows)
    
    for w in 1:num_windows
        window_start = (w - 1) * window_size + 1
        window_end = min(w * window_size, num_stages)
        actual_window_size = window_end - window_start + 1
        
        # Create a new model for this window
        window_model = JuMP.Model()
        if optimizer_factory !== nothing
            set_optimizer(window_model, optimizer_factory)
        end
        
        # Extract subproblems for this window
        window_subproblems = subproblems[window_start:window_end]
        window_state_params_in = state_params_in[window_start:window_end]
        window_state_params_out = state_params_out[window_start:window_end]
        window_uncertainties = uncertainties[window_start:window_end]
        
        # Build deterministic equivalent for this window
        window_model, uncertainties_new = deterministic_equivalent!(
            window_model,
            window_subproblems,
            window_state_params_in,
            window_state_params_out,
            initial_state,
            window_uncertainties
        )
        
        window_models[w] = window_model
        
        # Need to find the new state params in the combined model
        # For now, we'll store the structure needed
        state_params_in_windows[w] = [find_state_params_in(window_model, actual_window_size, t, length(initial_state)) for t in 1:actual_window_size]
        state_params_out_windows[w] = [find_state_params_out(window_model, actual_window_size, t, length(initial_state)) for t in 1:actual_window_size]
        uncertainties_windows[w] = uncertainties_new
    end
    
    return window_models, state_params_in_windows, state_params_out_windows, uncertainties_windows
end

"""
    find_state_params_in(model, num_stages, stage, state_dim)

Find state_in parameters for a given stage in a window model.
"""
function find_state_params_in(model::JuMP.Model, num_stages::Int, stage::Int, state_dim::Int)
    # State in parameters are named "state_in[i]" for stage 1, or linked to previous stage's state_out
    if stage == 1
        return find_variables(model, ["state_in", "[$stage]"])
    else
        # For later stages, state_in is linked to previous stage's state_out
        return find_variables(model, ["state_out", "[$(stage-1)]"])
    end
end

"""
    find_state_params_out(model, num_stages, stage, state_dim)

Find state_out parameters for a given stage in a window model.
"""
function find_state_params_out(model::JuMP.Model, num_stages::Int, stage::Int, state_dim::Int)
    # This is more complex as we need both parameter and variable
    # For simplicity, returning placeholder - actual implementation depends on model structure
    return find_variables(model, ["state_out", "[$stage]"])
end

"""
    extract_uncertainties_vec(uncertainties, num_stages)

Extract uncertainty values as vectors for decision rule input.
"""
function extract_uncertainties_vec(uncertainties)
    return [Float32[u[2] for u in stage_uncertainties] for stage_uncertainties in uncertainties]
end

"""
    train_multiple_shooting(model, initial_state, subproblems, state_params_in, state_params_out,
                            uncertainty_sampler; window_size, num_batches, num_train_per_batch, ...)

Train a decision rule using multiple shooting.

# Arguments
- `model`: The decision rule (neural network)
- `initial_state`: Initial state vector
- `subproblems`: Vector of JuMP models for each stage
- `state_params_in`: State input parameters for each stage
- `state_params_out`: State output parameters for each stage
- `uncertainty_sampler`: Function to sample uncertainties
- `window_size`: Number of stages per shooting window (default: 4)
- `num_batches`: Number of training iterations
- `num_train_per_batch`: Number of samples per batch
- `optimizer`: Flux optimizer
"""
function train_multiple_shooting(
    model,
    initial_state::Vector{<:Real},
    subproblems::Vector{JuMP.Model},
    state_params_in,
    state_params_out,
    uncertainty_sampler;
    window_size::Int=4,
    num_batches::Int=100,
    num_train_per_batch::Int=32,
    optimizer=Flux.Adam(0.01),
    adjust_hyperparameters=(iter, opt_state, num_train_per_batch) -> num_train_per_batch,
    record_loss=(iter, model, loss, tag) -> begin 
        println("tag: $tag, Iter: $iter, Loss: $loss")
        return false
    end,
    get_objective_no_target_deficit=get_objective_no_target_deficit,
    optimizer_factory=nothing
)
    # Initialise the optimiser
    opt_state = Flux.setup(optimizer, model)
    
    num_stages = length(subproblems)
    num_windows = ceil(Int, num_stages / window_size)
    num_uncertainties = length(uncertainty_sampler[1])
    
    for iter in 1:num_batches
        num_train_per_batch = adjust_hyperparameters(iter, opt_state, num_train_per_batch)
        
        # Sample uncertainties for this batch
        uncertainty_samples = [sample(uncertainty_sampler) for _ in 1:num_train_per_batch]
        
        # Pre-compute uncertainty vectors for decision rule input
        uncertainty_samples_vec = [
            [[uncertainty_samples[s][stage][i][2] for i in 1:num_uncertainties] 
             for stage in 1:num_stages] 
            for s in 1:num_train_per_batch
        ]
        
        # Split into windows
        uncertainty_samples_windows = [
            [uncertainty_samples[s][(w-1)*window_size+1 : min(w*window_size, num_stages)] 
             for w in 1:num_windows]
            for s in 1:num_train_per_batch
        ]
        uncertainty_samples_vec_windows = [
            [Float32.(uncertainty_samples_vec[s][(w-1)*window_size+1 : min(w*window_size, num_stages)])
             for w in 1:num_windows]
            for s in 1:num_train_per_batch
        ]
        
        objective = 0.0
        eval_loss = 0.0
        
        grads = Flux.gradient(model) do m
            for s in 1:num_train_per_batch
                Flux.reset!(m)
                sample_objective = train_multiple_shooting_sample(
                    m,
                    Float32.(initial_state),
                    subproblems,
                    state_params_in,
                    state_params_out,
                    uncertainty_samples[s],
                    uncertainty_samples_vec[s],
                    window_size,
                    optimizer_factory
                )
                objective += sample_objective
                eval_loss += @ignore_derivatives compute_eval_loss(subproblems, get_objective_no_target_deficit)
            end
            objective /= num_train_per_batch
            eval_loss /= num_train_per_batch
            return objective
        end
        
        record_loss(iter, model, eval_loss, "metrics/loss") && break
        record_loss(iter, model, objective, "metrics/training_loss") && break
        
        # Update the parameters
        grad = materialize_tangent(grads[1])
        Flux.update!(opt_state, model, grad)
    end
    
    return model
end

"""
    train_multiple_shooting_sample(model, initial_state, subproblems, ..., window_size)

Process a single sample for multiple shooting training.
"""
function train_multiple_shooting_sample(
    model,
    initial_state::Vector{Float32},
    subproblems::Vector{JuMP.Model},
    state_params_in,
    state_params_out,
    uncertainty_sample,
    uncertainty_sample_vec,
    window_size::Int,
    optimizer_factory
)
    num_stages = length(subproblems)
    num_windows = ceil(Int, num_stages / window_size)
    
    total_objective = 0.0f0
    current_real_state = initial_state
    
    for w in 1:num_windows
        window_start = (w - 1) * window_size + 1
        window_end = min(w * window_size, num_stages)
        actual_window_size = window_end - window_start + 1
        
        # Extract window data
        window_subproblems = subproblems[window_start:window_end]
        window_state_params_in = state_params_in[window_start:window_end]
        window_state_params_out = state_params_out[window_start:window_end]
        window_uncertainties = uncertainty_sample[window_start:window_end]
        window_uncertainties_vec = [Float32.(uncertainty_sample_vec[t]) for t in window_start:window_end]
        
        # Predict states for this window using decision rule
        window_states = predict_window_states(model, current_real_state, window_uncertainties_vec)
        
        # Simulate stages and get realized state for next window in one differentiable call
        # This returns (objective, realized_final_state) with proper gradient flow
        window_objective, realized_state = simulate_window_with_next_state(
            window_subproblems,
            window_state_params_in,
            window_state_params_out,
            window_uncertainties,
            window_states
        )
        
        total_objective += window_objective
        
        # Update real state for next window (now with gradient flow!)
        if w < num_windows
            current_real_state = realized_state
        end
    end
    
    return total_objective
end

"""
    simulate_window_stages(subproblems, state_params_in, state_params_out, uncertainties, states)

Simulate stages within a shooting window.
Gradients are computed only for first state_in and last state_out.
"""
function simulate_window_stages(
    subproblems::Vector{JuMP.Model},
    state_params_in,
    state_params_out,
    uncertainties,
    states  # Length = num_stages + 1
)
    num_stages = length(subproblems)
    total_objective = 0.0
    
    for t in 1:num_stages
        # Simulate this stage
        stage_objective = simulate_stage(
            subproblems[t],
            state_params_in[t],
            state_params_out[t],
            uncertainties[t],
            states[t],      # state_in
            states[t + 1]   # state_out (predicted by decision rule)
        )
        total_objective += stage_objective
    end
    
    return total_objective
end

"""
ChainRulesCore.rrule for simulate_window_stages.
Only backpropagates gradients through first state_in and last state_out.
"""
function ChainRulesCore.rrule(
    ::typeof(simulate_window_stages),
    subproblems,
    state_params_in,
    state_params_out,
    uncertainties,
    states
)
    y = simulate_window_stages(subproblems, state_params_in, state_params_out, uncertainties, states)
    
    function _pullback(Δy)
        num_stages = length(subproblems)
        Δ_states = [zeros(Float32, length(states[i])) for i in 1:length(states)]
        
        # Only compute gradients for first state_in (states[1]) and last state_out (states[end])
        # First state gradient
        try
            Δ_states[1] = Float32.(pdual.(state_params_in[1])) * Δy
        catch
            try
                DiffOpt.empty_input_sensitivities!(subproblems[1])
                MOI.set(subproblems[1], DiffOpt.ReverseObjectiveSensitivity(), Δy)
                DiffOpt.reverse_differentiate!(subproblems[1])
                Δ_states[1] = Float32.(DiffOpt.get_reverse_parameter.(subproblems[1], state_params_in[1]))
            catch e
                @warn "Failed to compute gradient for first state_in: $e"
            end
        end
        
        # Last state_out gradient
        try
            Δ_states[end] = Float32.(pdual.([s[1] for s in state_params_out[end]])) * Δy
        catch
            try
                DiffOpt.empty_input_sensitivities!(subproblems[end])
                MOI.set(subproblems[end], DiffOpt.ReverseObjectiveSensitivity(), Δy)
                DiffOpt.reverse_differentiate!(subproblems[end])
                Δ_states[end] = Float32.(DiffOpt.get_reverse_parameter.(subproblems[end], [s[1] for s in state_params_out[end]]))
            catch e
                @warn "Failed to compute gradient for last state_out: $e"
            end
        end
        
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), Δ_states)
    end
    
    return y, _pullback
end

"""
    compute_eval_loss(subproblems, get_objective_no_target_deficit)

Compute evaluation loss (objective without deficit penalties).
"""
function compute_eval_loss(subproblems, get_objective_no_target_deficit)
    total = 0.0
    for sp in subproblems
        total += get_objective_no_target_deficit(sp)
    end
    return total
end
