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
        if stage == 1
            uncertainties_stage = initial_state .+ uncertainties_stage
        end
        states[stage + 1] = decision_rule(uncertainties_stage)
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
        uncertainties_stage_vec = vcat(initial_state, [[uncertainties[j][i][2] for i in 1:length(uncertainties[j])] for j in 1:stage]...)
        uncertainties_stage = [uncertainties[stage][i][2] for i in 1:length(uncertainties[stage])]
        decision_rule = decision_rules[stage]
        states[stage + 1] = decision_rule(uncertainties_stage)
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
        catch
            # In case of any error (e.g. non-DiffOpt model, missing parameters, etc.), return identity gradients 1 * Δy for state_in and state_out_target.
            d_state_in = ones(length(state_in)) .* Δy  # simple fallback: sum Δy and return for all inputs
            d_state_out_target = ones(length(state_out_target)) .* Δy
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), d_state_in, d_state_out_target)
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
    @ignore_derivatives Flux.reset!.(decision_rules)
    
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
    Flux.reset!.(decision_rules)
    states = simulate_states(initial_state, uncertainties, decision_rules)
    return simulate_multistage(subproblems, state_params_in, state_params_out, uncertainties, states)
end

function pdual(v::VariableRef)
    if is_parameter(v)
        # Use our custom parameter dual computation that works with any JuMP model
        return compute_parameter_dual(JuMP.owner_model(v), v)
    else
        error("Variable is not a parameter")
    end
end

pdual(vs::Vector) = [pdual(v) for v in vs]

function ChainRulesCore.rrule(::typeof(simulate_stage), subproblem, state_param_in, state_param_out, uncertainty, state_in, state_out)
    y = simulate_stage(subproblem, state_param_in, state_param_out, uncertainty, state_in, state_out)
    function _pullback(Δy)
        try 
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), pdual.(state_param_in) * Δy, pdual.([s[1] for s in state_param_out]) * Δy)
        catch
            try
                DiffOpt.empty_input_sensitivities!(subproblem)
                MOI.set(
                    subproblem,
                    DiffOpt.ReverseObjectiveSensitivity(),
                    Δy,
                )
                DiffOpt.reverse_differentiate!(subproblem)

                return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), 
                    DiffOpt.get_reverse_parameter.(subproblem, state_param_in), 
                    DiffOpt.get_reverse_parameter.(subproblem, [s[1] for s in state_param_out])
                )
            catch e
                @warn "Failed to compute gradients via DiffOpt. Returning zero gradients for state_in and state_out_target."
                # print error
                @show e
                # print termination status
                @show JuMP.termination_status(subproblem)

                return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), zeros(length(state_param_in)), zeros(length(state_param_out)))
            end
        end
    end
    return y, _pullback
end

# Define ChainRulesCore.rrule of simulate_multistage
function ChainRulesCore.rrule(::typeof(simulate_multistage), det_equivalent::JuMP.Model, state_params_in, state_params_out, uncertainties, states)
    y = simulate_multistage(det_equivalent, state_params_in, state_params_out, uncertainties, states)
    function _pullback(Δy)
        Δ_states = similar(states)
        try
            Δ_states[1] = pdual.(state_params_in[1])
            for t in 1:length(state_params_out)
                Δ_states[t + 1] = pdual.([s[1] for s in state_params_out[t]])
            end
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), Δ_states * Δy)
        catch
            DiffOpt.empty_input_sensitivities!(det_equivalent)
            MOI.set(
                det_equivalent,
                DiffOpt.ReverseObjectiveSensitivity(),
                Δy,
            )
            DiffOpt.reverse_differentiate!(det_equivalent)
            Δ_states[1] = DiffOpt.get_reverse_parameter.(det_equivalent, state_params_in[1])
            for t in 1:length(state_params_out)
                Δ_states[t + 1] = DiffOpt.get_reverse_parameter.(det_equivalent, [s[1] for s in state_params_out[t]])
            end
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), Δ_states)
        end
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
    record_loss=(iter, model, loss, tag) -> begin println("tag: $tag, Iter: $iter, Loss: $loss")
        return false
    end,
    get_objective_no_target_deficit=get_objective_no_target_deficit
)
    # Initialise the optimiser for this model:
    opt_state = Flux.setup(optimizer, model)

    for iter in 1:num_batches
        num_train_per_batch = adjust_hyperparameters(iter, opt_state, num_train_per_batch)
        # Sample uncertainties
        uncertainty_samples = [sample(uncertainty_sampler) for _ in 1:num_train_per_batch]
        objective = 0.0
        eval_loss = 0.0
        grads = Flux.gradient(model) do m
            for s in 1:num_train_per_batch
                Flux.reset!(m)
                objective += simulate_multistage(subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples[s], m)
                eval_loss += get_objective_no_target_deficit(subproblems)
            end
            objective /= num_train_per_batch
            eval_loss /= num_train_per_batch
            return objective
        end
        record_loss(iter, model, eval_loss, "metrics/loss") && break
        record_loss(iter, model, objective, "metrics/training_loss") && break

        # Update the parameters so as to reduce the objective,
        # according the chosen optimisation rule:
        Flux.update!(opt_state, model, grads[1])
    end
    
    return model
end

function sim_states(t, m, initial_state, uncertainty_sample_vec)
    if t == 1
        return Float32.(initial_state)
    elseif t == 2
        return m(uncertainty_sample_vec[1] + initial_state)
    else
        return m(uncertainty_sample_vec[t - 1])
    end
end

function train_multistage(model, initial_state, det_equivalent::JuMP.Model, 
    state_params_in, state_params_out, uncertainty_sampler; 
    num_batches=100, num_train_per_batch=32, optimizer=Flux.Adam(0.01),
    adjust_hyperparameters=(iter, opt_state, num_train_per_batch) -> num_train_per_batch,
    record_loss=(iter, model, loss, tag) -> begin println("tag: $tag, Iter: $iter, Loss: $loss")
        return false
    end,
    get_objective_no_target_deficit=get_objective_no_target_deficit
)
    # Initialise the optimiser for this model:
    opt_state = Flux.setup(optimizer, model)

    for iter in 1:num_batches
        num_train_per_batch = adjust_hyperparameters(iter, opt_state, num_train_per_batch)
        # Sample uncertainties
        uncertainty_samples = [sample(uncertainty_sampler) for _ in 1:num_train_per_batch]
        num_uncertainties = length(uncertainty_samples[1][1])
        uncertainty_samples_vec = [[[uncertainty_samples[s][stage][i][2] for i in 1:num_uncertainties] for stage in 1:length(uncertainty_samples[1])] for s in 1:num_train_per_batch]

        # Calculate the gradient of the objective
        # with respect to the parameters within the model:
        objective = 0.0
        eval_loss = 0.0
        # try
            grads = Flux.gradient(model) do m
                for s in 1:num_train_per_batch
                    Flux.reset!(m)
                    # m.state = initial_state[:,:]
                    # m(initial_state) # Breaks Everything
                    states = [sim_states(t, m, initial_state, uncertainty_samples_vec[s]) for t = 1:length(state_params_in) + 1]
                    objective += simulate_multistage(det_equivalent, state_params_in, state_params_out, uncertainty_samples[s], states)
                    eval_loss += get_objective_no_target_deficit(det_equivalent)
                end
                objective /= num_train_per_batch
                eval_loss /= num_train_per_batch
                return objective
            end
        # catch
            # continue;
        # end
        record_loss(iter, model, eval_loss, "metrics/loss") && break
        record_loss(iter, model, objective, "metrics/training_loss") && break

        # Update the parameters so as to reduce the objective,
        # according the chosen optimisation rule:
        Flux.update!(opt_state, model, grads[1])
    end
    
    return model
end

function make_single_network(models::Vector{F}, number_of_states::Int) where {F}
    size_m = length(models)
    return Parallel(permutedims ∘ hcat, [Chain(
        x -> x[1:number_of_states * (i + 1)],
        models[i]
    ) for i in 1:size_m]...)
end

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


function train_multistage(models::Vector, initial_state, det_equivalent::JuMP.Model, 
    state_params_in, state_params_out, uncertainty_sampler; 
    num_batches=100, num_train_per_batch=32, optimizer=Flux.Adam(0.01),
    adjust_hyperparameters=(iter, opt_state, num_train_per_batch) -> num_train_per_batch,
    record_loss=(iter, model, loss, tag) -> begin println("tag: $tag, Iter: $iter, Loss: $loss")
        return false
    end,
    get_objective_no_target_deficit=get_objective_no_target_deficit
)
    num_states = length(initial_state)
    model = make_single_network(models, num_states)
    # Initialise the optimiser for this model:
    opt_state = Flux.setup(optimizer, model)

    for iter in 1:num_batches
        num_train_per_batch = adjust_hyperparameters(iter, opt_state, num_train_per_batch)
        # Sample uncertainties
        uncertainty_samples = [sample(uncertainty_sampler) for _ in 1:num_train_per_batch]
        uncertainty_samples_vecs = [[[uncertainty_samples[s][stage][i][2] for i in 1:num_uncertainties] for stage in 1:length(uncertainty_samples[1])] for s in 1:num_train_per_batch]
        uncertainty_samples_vec = [vcat(initial_state, uncertainty_samples_vecs[s]...) for s in 1:num_train_per_batch]

        # Calculate the gradient of the objective
        # with respect to the parameters within the model:
        eval_loss = 0.0
        objective = 0.0
        grads = Flux.gradient(model) do m
            for s in 1:num_train_per_batch
                states = [Vector(i) for i in eachrow([Float32.(initial_state)'; m(uncertainty_samples_vec[s])])]
                objective += simulate_multistage(det_equivalent, state_params_in, state_params_out, uncertainty_samples[s], states)
                @ignore_derivatives eval_loss += get_objective_no_target_deficit(det_equivalent)
            end
            objective /= num_train_per_batch
            @ignore_derivatives eval_loss /= num_train_per_batch
            return objective
        end
        record_loss(iter, model, eval_loss, "metrics/loss") && break
        record_loss(iter, model, objective, "metrics/training_loss") && break

        # Update the parameters so as to reduce the objective,
        # according the chosen optimisation rule:
        Flux.update!(opt_state, model, grads[1])
    end
    
    return model
end
