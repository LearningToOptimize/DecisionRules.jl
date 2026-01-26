function variable_to_parameter(model::JuMP.Model, variable::JuMP.VariableRef; initial_value=0.0, deficit=nothing)
    parameter = @variable(model; base_name = "_" * name(variable), set=MOI.Parameter(initial_value))
    # bind the parameter to the variable
    if isnothing(deficit)
        @constraint(model, variable == parameter)
        return parameter
    else
        @constraint(model, variable + deficit == parameter)
        return parameter, variable
    end
end

"""
    create_deficit!(model::JuMP.Model, len::Int; penalty_l1=nothing, penalty_l2=nothing, penalty=nothing)

Create deficit variables to penalize state deviations in a JuMP model.

Supports three modes:
- L1 norm only: Uses `MOI.NormOneCone` (default if no penalty specified)
- L2 squared norm only: Uses sum of squared deviations (solver-compatible alternative to SecondOrderCone)
- Both norms: Creates both constraints with separate penalties

# Arguments
- `model`: The JuMP model to add deficit variables to
- `len`: Number of deficit variables (typically dimension of state)
- `penalty_l1`: Penalty coefficient for L1 norm (NormOneCone). If `nothing` and L1 is used, defaults to max objective coefficient.
- `penalty_l2`: Penalty coefficient for L2 squared norm (sum of squares). If `nothing` and L2 is used, defaults to max objective coefficient.
- `penalty`: Legacy argument. If provided and penalty_l1/penalty_l2 are both `nothing`, uses this for L1 norm only.

# Returns
- `norm_deficit`: Single variable representing total penalized deviation (for logging compatibility)
- `_deficit`: Vector of deficit variables for each state dimension

# Examples
```julia
# L1 norm only (default behavior, backwards compatible)
norm_deficit, _deficit = create_deficit!(model, 3; penalty=1000.0)

# L2 norm only
norm_deficit, _deficit = create_deficit!(model, 3; penalty_l2=1000.0)

# Both L1 and L2 norms
norm_deficit, _deficit = create_deficit!(model, 3; penalty_l1=1000.0, penalty_l2=500.0)
```
"""
function create_deficit!(model::JuMP.Model, len::Int; penalty_l1=nothing, penalty_l2=nothing, penalty=nothing)
    # Handle legacy 'penalty' argument for backwards compatibility
    if isnothing(penalty_l1) && isnothing(penalty_l2)
        if !isnothing(penalty)
            penalty_l1 = penalty  # Use legacy penalty for L1 only
        else
            # Default: L1 norm with auto-computed penalty
            obj = objective_function(model)
            penalty_l1 = maximum(abs.(values(obj.terms)))
        end
    end

    # Auto-compute penalties if needed
    if !isnothing(penalty_l1) && penalty_l1 === :auto
        obj = objective_function(model)
        penalty_l1 = maximum(abs.(values(obj.terms)))
    end
    if !isnothing(penalty_l2) && penalty_l2 === :auto
        obj = objective_function(model)
        penalty_l2 = maximum(abs.(values(obj.terms)))
    end

    # Create deficit variables
    _deficit = @variable(model, _deficit[1:len])
    
    # Create individual norm variables for each cone type
    use_l1 = !isnothing(penalty_l1)
    use_l2 = !isnothing(penalty_l2)
    
    # Create norm_deficit as the total penalized deviation (for logging compatibility)
    @variable(model, norm_deficit >= 0.0)
    
    if use_l1 && use_l2
        # Both L1 and L2 squared norms
        @variable(model, norm_l1 >= 0.0)
        @variable(model, norm_l2_sq >= 0.0)  # L2 squared (sum of squares)
        @constraint(model, [norm_l1; _deficit] in MOI.NormOneCone(1 + len))
        @constraint(model, norm_l2_sq >= sum(_deficit[i]^2 for i in 1:len))
        # norm_deficit = penalty_l1 * norm_l1 + penalty_l2 * norm_l2_sq
        @constraint(model, norm_deficit >= penalty_l1 * norm_l1 + penalty_l2 * norm_l2_sq)
        set_objective_coefficient(model, norm_deficit, 1.0)
    elseif use_l1
        # L1 norm only
        @constraint(model, [norm_deficit; _deficit] in MOI.NormOneCone(1 + len))
        set_objective_coefficient(model, norm_deficit, penalty_l1)
    elseif use_l2
        # L2 squared norm only (sum of squares)
        @constraint(model, norm_deficit >= sum(_deficit[i]^2 for i in 1:len))
        set_objective_coefficient(model, norm_deficit, penalty_l2)
    else
        error("At least one of penalty_l1 or penalty_l2 must be specified")
    end
    
    return norm_deficit, _deficit
end

mutable struct SaveBest <: Function
    best_loss::Float64
    model_path::String
end
function (callback::SaveBest)(iter, model, loss)
    if loss < callback.best_loss
        m = model |> cpu
        @info "best model change" callback.best_loss loss
        callback.best_loss = loss
        model_state = Flux.state(m)
        jldsave(callback.model_path; model_state=model_state)
    end
    return false
end

mutable struct StallingCriterium <: Function
    patience::Int
    best_loss::Float64
    stall_count::Int
end
function (callback::StallingCriterium)(iter, model, loss)
    if loss < callback.best_loss
        callback.best_loss = loss
        callback.stall_count = 0
    else
        callback.stall_count += 1
    end
    if callback.stall_count >= callback.patience
        return true
    end
    return false
end

function var_set_name!(src::JuMP.VariableRef, dest::JuMP.VariableRef, t::Int)
    name = JuMP.name(src)
    if !isempty(name)
        # append node index to original variable name
        JuMP.set_name(dest, string(name, "#", t))
    else
        # append node index to original variable index
        var_name = string("_[", index(src).value, "]")
        JuMP.set_name(dest, string(var_name, "#", t))
    end
end

function add_child_model_vars!(model::JuMP.Model, subproblem::JuMP.Model, t::Int, state_params_in::Vector{Vector{Any}}, state_params_out::Vector{Vector{Tuple{Any, VariableRef}}}, initial_state::Vector{Float64}, var_src_to_dest::Dict{VariableRef, VariableRef})
    allvars = all_variables(subproblem)
    allvars = setdiff(allvars, state_params_in[t])
    if state_params_out[t][1][1] isa VariableRef # not MadNLP
        allvars = setdiff(allvars, [x[1] for x in state_params_out[t]])
    end
    allvars = setdiff(allvars, [x[2] for x in state_params_out[t]])
    x = @variable(model, [1:length(allvars)])
    for (src, dest) in zip(allvars, x)
        var_src_to_dest[src] = dest
        var_set_name!(src, dest, t)
    end

    for (i, src) in enumerate(state_params_out[t])
        dest_var = @variable(model)
        var_src_to_dest[src[2]] = dest_var
        var_set_name!(src[2], dest_var, t)
        
        if state_params_out[t][1][1] isa VariableRef
            dest_param = @variable(model)
            var_src_to_dest[src[1]] = dest_param
            var_set_name!(src[1], dest_param, t)
            state_params_out[t][i] = (dest_param, dest_var)
        else
            state_params_out[t][i] = (state_params_out[t][i][1], dest_var)
        end
    end
    if t == 1
        for (i, src) in enumerate(state_params_in[t])
            if src isa VariableRef
                dest = @variable(model)
                var_src_to_dest[src] = dest
                var_set_name!(src, dest, t)
                state_params_in[t][i] = dest
            end
        end
    else
        for (i, src) in enumerate(state_params_in[t])
            if src isa VariableRef
                var_src_to_dest[src] = state_params_out[t-1][i][2]
            end
            state_params_in[t][i] = state_params_out[t-1][i][2]
            # delete parameter constraint associated with src
            if src isa VariableRef
                for con in JuMP.all_constraints(subproblem, VariableRef, MOI.Parameter{Float64})
                    obj = JuMP.constraint_object(con)
                    if obj.func == src
                        JuMP.delete(subproblem, con)
                    end
                end
            end
        end
    end
    return var_src_to_dest
end

function copy_and_replace_variables(
    src::Vector,
    map::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return copy_and_replace_variables.(src, Ref(map))
end

function copy_and_replace_variables(
    src::Real,
    ::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return src
end

function copy_and_replace_variables(
    src::JuMP.VariableRef,
    src_to_dest_variable::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return src_to_dest_variable[src]
end

function copy_and_replace_variables(
    src::JuMP.GenericAffExpr,
    src_to_dest_variable::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return JuMP.GenericAffExpr(
        src.constant,
        Pair{VariableRef,Float64}[
            src_to_dest_variable[key] => val for (key, val) in src.terms
        ],
    )
end

function copy_and_replace_variables(
    src::JuMP.GenericQuadExpr,
    src_to_dest_variable::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return JuMP.GenericQuadExpr(
        copy_and_replace_variables(src.aff, src_to_dest_variable),
        Pair{UnorderedPair{VariableRef},Float64}[
            UnorderedPair{VariableRef}(
                src_to_dest_variable[pair.a],
                src_to_dest_variable[pair.b],
            ) => coef for (pair, coef) in src.terms
        ],
    )
end

function copy_and_replace_variables(
    src::JuMP.GenericNonlinearExpr{V},
    src_to_dest_variable::Dict{JuMP.VariableRef,JuMP.VariableRef},
) where {V}
    num_args = length(src.args)
    args = Vector{Any}(undef, num_args)
    for i = 1:num_args
        args[i] = copy_and_replace_variables(src.args[i], src_to_dest_variable)
    end

    return @expression(owner_model(first(src_to_dest_variable)[2]), eval(src.head)(args...))
end

function copy_and_replace_variables(
    src::Any,
    ::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return error(
        "`copy_and_replace_variables` is not implemented for functions like `$(src)`.",
    )
end

function create_constraint(model, obj, var_src_to_dest)
    new_func = copy_and_replace_variables(obj.func, var_src_to_dest)
    return @constraint(model, new_func in obj.set)
end

function create_constraint(model, obj::ScalarConstraint{NonlinearExpr, MOI.EqualTo{Float64}}, var_src_to_dest)
    new_func = copy_and_replace_variables(obj.func, var_src_to_dest)
    return @constraint(model, new_func == obj.set.value)
end

function create_constraint(model, obj::ScalarConstraint{NonlinearExpr, MOI.LessThan{Float64}}, var_src_to_dest)
    new_func = copy_and_replace_variables(obj.func, var_src_to_dest)
    return @constraint(model, new_func <= obj.set.upper)
end

function create_constraint(model, obj::ScalarConstraint{NonlinearExpr, MOI.GreaterThan{Float64}}, var_src_to_dest)
    new_func = copy_and_replace_variables(obj.func, var_src_to_dest)
    return @constraint(model, new_func >= obj.set.lower)
end

function add_child_model_exps!(model::JuMP.Model, subproblem::JuMP.Model, var_src_to_dest::Dict{VariableRef, VariableRef}, state_params_out, state_params_in, t)
    # Add constraints:
    # for (F, S) in JuMP.list_of_constraint_types(subproblem)
    cons_to_cons = Dict()
    for con in JuMP.all_constraints(subproblem; include_variable_in_set_constraints=true) #, F, S)
        obj = JuMP.constraint_object(con)
        c = create_constraint(model, obj, var_src_to_dest)
        cons_to_cons[con] = c
        if (state_params_out[t][1][1] isa ConstraintRef)
            for (i,_con) in enumerate(state_params_out[t])
                if con == _con[1]
                    state_params_out[t][i] = (c, state_params_out[t][i][2])
                    continue;
                end
            end
        end
        if (t==1) && (state_params_in[t][1] isa ConstraintRef)
            for (i,_con) in enumerate(state_params_in[t])
                if con == _con
                    state_params_in[t][i] = c
                    continue;
                end
            end
        end
    end
    # end
    # Add objective:
    current = JuMP.objective_function(model)
    subproblem_objective =
        copy_and_replace_variables(JuMP.objective_function(subproblem), var_src_to_dest)
    JuMP.set_objective_function(
        model,
        current + subproblem_objective,
    )
    return cons_to_cons
end

"Create Single JuMP.Model from subproblems. rename variables to avoid conflicts by adding [t] at the end of the variable name where t is the subproblem index"
function deterministic_equivalent!(model::JuMP.Model,
    subproblems::Vector{JuMP.Model},
    state_params_in::Vector{Vector{Any}},
    state_params_out::Vector{Vector{Tuple{Any, VariableRef}}},
    initial_state::Vector{Float64},
    uncertainties::Vector{Vector{Tuple{VariableRef, Vector{Float64}}}},
)
    set_objective_sense(model, objective_sense(subproblems[1]))
    uncertainties_new = Vector{Vector{Tuple{VariableRef, Vector{Float64}}}}(undef, length(uncertainties))
    var_src_to_dest = Dict{VariableRef, VariableRef}()
    for t in 1:length(subproblems)
        DecisionRules.add_child_model_vars!(model, subproblems[t], t, state_params_in, state_params_out, initial_state, var_src_to_dest)
    end

    cons_to_cons = Vector{Dict}(undef, length(subproblems))
    for t in 1:length(subproblems)
        cons_to_cons[t] = DecisionRules.add_child_model_exps!(model, subproblems[t], var_src_to_dest, state_params_out, state_params_in, t)
    end

    if uncertainties[1][1][1] isa VariableRef
        # use var_src_to_dest
        for t in 1:length(subproblems)
            uncertainties_new[t] = Vector{Tuple{VariableRef, Vector{Float64}}}(undef, length(uncertainties[t]))
            for (i, tup) in enumerate(uncertainties[t])
                ky, val = tup
                uncertainties_new[t][i] = (var_src_to_dest[ky],val)
            end
        end
    else
        # use cons_to_cons
        for t in 1:length(subproblems)
            uncertainties_new[t] = Vector{Tuple{VariableRef, Vector{Float64}}}(undef, length(uncertainties[t]))
            for (i, tup) in enumerate(uncertainties[t])
                ky, val = tup
                uncertainties_new[t] = (cons_to_cons[t][ky],val)
            end
        end
    end

    return model, uncertainties_new
end

function find_variables(model::JuMP.Model, variable_name_parts::Vector{S}) where {S}
    all_vars = all_variables(model)
    interest_vars = all_vars[findall(x -> all([occursin(part, JuMP.name(x)) for part in variable_name_parts]), all_vars)]
    if length(interest_vars) == 1
        return interest_vars
    end
    return [interest_vars[findfirst(x -> occursin(variable_name_parts[1] * "[$i]", JuMP.name(x)), interest_vars)] for i in 1:length(interest_vars)]
end
