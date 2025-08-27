function simulate_states(
    initial_state::Vector{Float64},
    uncertainties,
    decision_rule::F;
    ensure_feasibility=(x_out, x_in, uncertainty) -> x_out
) where {F}
    num_stages = length(uncertainties)
    states = Vector{Vector{Float64}}(undef, num_stages + 1)
    states[1] = initial_state
    for stage in 1:num_stages
        uncertainties_stage = [uncertainties[stage][i][2] for i in 1:length(uncertainties[stage])]
        if stage == 1
            uncertainties_stage = initial_state .+ uncertainties_stage
        end
        state_out = decision_rule(uncertainties_stage)
        states[stage + 1] = ensure_feasibility(state_out, states[stage], uncertainties_stage)
    end
    return states
end

function simulate_states(
    initial_state::Vector{Float64},
    uncertainties,
    decision_rules::Vector{F};
    ensure_feasibility=(x_out, x_in, uncertainty) -> x_out
) where {F}
    num_stages = length(uncertainties)
    states = Vector{Vector{Float64}}(undef, num_stages + 1)
    states[1] = initial_state
    for stage in 1:num_stages
        uncertainties_stage_vec = vcat(initial_state, [[uncertainties[j][i][2] for i in 1:length(uncertainties[j])] for j in 1:stage]...)
        uncertainties_stage = [uncertainties[stage][i][2] for i in 1:length(uncertainties[stage])]
        decision_rule = decision_rules[stage]
        state_out = decision_rule(uncertainties_stage_vec)
        states[stage + 1] = ensure_feasibility(state_out, states[stage], uncertainties_stage)
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
    for i in 1:length(state_param_in)
        state_var = state_param_out[i][1]
        set_parameter_value(state_var, state_out_target[i])
    end

    # Solve subproblem
    optimize!(subproblem)

    # objective value
    obj = objective_value(subproblem)

    return obj
end

function get_next_state(subproblem::JuMP.Model, state_param_out::Vector{Tuple{Any, VariableRef}}, state_in::Vector{T}, state_out_target::Vector{Z}) where {T <: Real, Z <: Real}
    state_out = [value(state_param_out[i][2]) for i in 1:length(state_param_out)]
    return state_out
end

# Define rrule of get_next_state
# This is simplified. This actual jacobian will require DiffOpt
function rrule(::typeof(get_next_state), subproblem, state_param_out, state_in, state_out_target)
    state_out = get_next_state(subproblem, state_param_out, state_in, state_out_target)
    function _pullback(Δstate_out)
        d_state_in = zeros(length(state_in))
        d_state_out = zeros(length(state_out_target))
        for i in 1:length(state_in)
            s_out_target = state_out_target[i]
            s_in = state_in[i]
            if s_out_target < s_in && s_out_target >= 0.0
                d_state_out[i] = Δstate_out[i]
            elseif s_out_target > s_in && s_out_target >= 0.0
                d_state_in[i] = Δstate_out[i]
            end
        end
        return (NoTangent(), NoTangent(), NoTangent(), d_state_in , d_state_out)
    end
    return state_out, _pullback
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

# define rrule of get_objective_no_target_deficit
function rrule(::typeof(get_objective_no_target_deficit), subproblem; norm_deficit="norm_deficit")
    objective_val = get_objective_no_target_deficit(subproblem, norm_deficit=norm_deficit)
    function _pullback(Δobjective_val)
        return (NoTangent(), NoTangent())
    end
    return objective_val, _pullback
end

function simulate_multistage(
    subproblems::Vector{JuMP.Model},
    state_params_in::Vector{Vector{Any}},
    state_params_out::Vector{Vector{Tuple{Any, VariableRef}}},
    uncertainties,
    states::Vector{Vector{T}};
    _objective_value = objective_value
    ) where {T <: Real}
    
    # Loop over stages
    objective_value = 0.0
    state_in = states[1]
    for stage in 1:length(subproblems)
        state_out = states[stage + 1]
        subproblem = subproblems[stage]
        state_param_in = state_params_in[stage]
        state_param_out = state_params_out[stage]
        uncertainty = uncertainties[stage]
        simulate_stage(subproblem, state_param_in, state_param_out, uncertainty, state_in, state_out)
        objective_value += _objective_value(subproblem)
        state_in = DecisionRules.get_next_state(subproblem, state_param_out, state_in, state_out)
    end
    
    # Return final objective value
    return objective_value
end

function simulate_multistage(
    det_equivalent::JuMP.Model,
    state_params_in::Vector{Vector{Z}},
    state_params_out::Vector{Vector{Tuple{Z, VariableRef}}},
    uncertainties,
    states;
    _objective_value = objective_value #get_objective_no_target_deficit
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

    return _objective_value(det_equivalent)
end

function simulate_multistage(
    subproblems::Union{Vector{JuMP.Model}, JuMP.Model},
    state_params_in::Vector{Vector{U}},
    state_params_out::Vector{Vector{Tuple{U, VariableRef}}},
    initial_state::Vector{T},
    uncertainties,
    decision_rules;
    ensure_feasibility=(x_out, x_in, uncertainty) -> x_out,
    _objective_value=objective_value
) where {T <: Real, U}
    Flux.reset!.(decision_rules)
    states = simulate_states(initial_state, uncertainties, decision_rules, ensure_feasibility=ensure_feasibility)
    return simulate_multistage(subproblems, state_params_in, state_params_out, uncertainties, states; _objective_value=_objective_value)
end

function pdual(v::VariableRef)
    if is_parameter(v)
        return MOI.get(JuMP.owner_model(v), POI.ParameterDual(), v)
    else
        return dual(FixRef(v))
    end
end
pdual(v::ConstraintRef) = dual(v) # this needs to be fixed to not depend com how the constraint is created
pdual(vs::Vector) = [pdual(v) for v in vs]

function rrule(::typeof(simulate_stage), subproblem, state_param_in, state_param_out, uncertainty, state_in, state_out)
    y = simulate_stage(subproblem, state_param_in, state_param_out, uncertainty, state_in, state_out)
    function _pullback(Δy)
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), pdual.(state_param_in) * Δy, pdual.([s[1] for s in state_param_out]) * Δy)
    end
    return y, _pullback
end

# Define rrule of simulate_multistage
function rrule(::typeof(simulate_multistage), det_equivalent, state_params_in, state_params_out, uncertainties, states)
    y = simulate_multistage(det_equivalent, state_params_in, state_params_out, uncertainties, states)
    function _pullback(Δy)
        Δ_states = similar(states)
        Δ_states[1] = pdual.(state_params_in[1])
        for t in 1:length(state_params_out)
            Δ_states[t + 1] = pdual.([s[1] for s in state_params_out[t]])
        end
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), Δ_states * Δy)
    end
    return y, _pullback
end

function sample(uncertainty_samples::Vector{Tuple{VariableRef, Vector}})
    T = eltype(uncertainty_samples[1][2])
    uncertainty_sample = Vector{Tuple{VariableRef,T}}(undef, length(uncertainty_samples))
    for i in 1:length(uncertainty_samples)
        uncertainty_sample[i] = (uncertainty_samples[i][1], rand(uncertainty_samples[i][2]))
    end
    return uncertainty_sample
end

sample(uncertainty_samples::Vector{Vector{Tuple{VariableRef, Vector}}}) = [sample(uncertainty_samples[t]) for t in 1:length(uncertainty_samples)]

# function train_multistage(model, initial_state, subproblems::Vector{JuMP.Model}, 
#     state_params_in, state_params_out, uncertainty_sampler; 
#     num_batches=100, num_train_per_batch=32, optimizer=Flux.Adam(0.01), ensure_feasibility=(x_out, x_in, uncertainty) -> x_out,
#     adjust_hyperparameters=(iter, opt_state, num_train_per_batch) -> num_train_per_batch,
#     record_loss=(iter, model, loss, tag) -> begin println("tag: $tag, Iter: $iter, Loss: $loss")
#         return false
#     end,
#     get_objective_no_target_deficit=get_objective_no_target_deficit
# )
#     # Initialise the optimiser for this model:
#     opt_state = Flux.setup(optimizer, model)

#     for iter in 1:num_batches
#         num_train_per_batch = adjust_hyperparameters(iter, opt_state, num_train_per_batch)
#         # Sample uncertainties
#         uncertainty_samples = [sample(uncertainty_sampler) for _ in 1:num_train_per_batch]
#         uncertainty_samples_vec = [[collect(values(uncertainty_sample[j])) for j in 1:length(uncertainty_sample)] for uncertainty_sample in uncertainty_samples]

#         # Calculate the gradient of the objective
#         # with respect to the parameters within the model:
#         objective = 0.0
#         eval_loss = 0.0
#         grads = Flux.gradient(model) do m
#             for s in 1:num_train_per_batch
#                 Flux.reset!(m)
#                 # m(initial_state) # Breaks Everything
#                 # state_in = initial_state
#                 for (j, subproblem) in enumerate(subproblems)
#                     state_out = m(uncertainty_samples_vec[s][j])
#                     state_out = ensure_feasibility(state_out, state_in, uncertainty_samples_vec[s][j])
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

function sim_states(t, m, initial_state, uncertainty_sample_vec)
    if t == 1
        return Float32.(initial_state)
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
#     num_batches=100, num_train_per_batch=32, optimizer=Flux.Adam(0.01), ensure_feasibility=(x_out, x_in, uncertainty) -> x_out,
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
#                     state_out = ensure_feasibility(states[j], state_in, uncertainty_samples_vecs[s][j])
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
    num_batches=100, num_train_per_batch=32, optimizer=Flux.Adam(0.01), ensure_feasibility=(x_out, x_in, uncertainty) -> x_out,
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
