using JuMP
using CSV
using Tables
using JSON

function find_reservoirs_and_inflow(model::JuMP.Model)
    reservoir_in = find_variables(model, ["reservoir", "_in"])
    reservoir_out = find_variables(model, ["reservoir", "_out"])
    inflow = find_variables(model, ["inflow"])
    return reservoir_in, reservoir_out, inflow
end

# function move_bounds_to_constrainits!(variable::JuMP.variableiableRef)
#     model = JuMP.owner_model(variable)
#     if has_lower_bound(variable)
#         @constraint(model, variable >= lower_bound(variable))
#         delete_lower_bound(variable)
#     end
#     if has_upper_bound(variable)
#         @constraint(model, variable <= upper_bound(variable))
#         delete_upper_bound(variable)
#     end
# end

# function add_deficit_constraints!(model::JuMP.Model; penalty=nothing)
#     if isnothing(penalty)
#         obj = objective_function(model)
#         # get the highest coefficient
#         penalty = maximum(abs.(values(obj.terms)))
#         penalty = penalty * 1.1
#     end
#     consrefs = [con for con in all_constraints(model, include_variable_in_set_constraints=false)]
#     @variable(model, _deficit[1:length(consrefs)])
#     @variable(model, norm_deficit)
#     for (i, eq) in enumerate(consrefs)
#         set_normalized_coefficient(eq, _deficit[i], 1)
#     end
#     @constraint(model, [norm_deficit; _deficit] in MOI.NormOneCone(1 + length(_deficit)))
#     set_objective_coefficient(model, norm_deficit, penalty)
#     return norm_deficit
# end

function read_inflow(file::String, nHyd::Int; num_stages=nothing)
    allinflows = CSV.read(file, Tables.matrix; header=false)
    nlin, ncol = size(allinflows)
    if isnothing(num_stages)
        num_stages = nlin
    elseif num_stages > nlin
        number_of_cycles = div(num_stages, nlin) + 1
        allinflows = vcat([allinflows for _ in 1:number_of_cycles]...)
    end
    nCen = Int(floor(ncol / nHyd))
    vector_inflows = Array{Array{Float64,2}}(undef, nHyd)
    for i in 1:nHyd
        vector_inflows[i] = allinflows[1:num_stages, ((i - 1) * nCen + 1):(i * nCen)]
    end
    return vector_inflows, nCen, num_stages
end

function build_hydropowermodels(case_folder::AbstractString, subproblem_file::AbstractString; num_stages=nothing, penalty=nothing, optimizer=nothing)
    hydro_file = JSON.parsefile(joinpath(case_folder, "hydro.json"))["Hydrogenerators"]
    nHyd = length(hydro_file)
    vector_inflows, nCen, num_stages = read_inflow(joinpath(case_folder, "inflows.csv"), nHyd; num_stages=num_stages)
    initial_state = [hydro["initial_volume"] for hydro in hydro_file]
    max_volume = [hydro["max_volume"] for hydro in hydro_file]

    subproblems = Vector{JuMP.Model}(undef, num_stages)
    state_params_in = Vector{Vector{Any}}(undef, num_stages)
    state_params_out = Vector{Vector{Tuple{Any, VariableRef}}}(undef, num_stages)
    uncertainty_samples = Vector{Vector{Tuple{VariableRef, Vector{Float64}}}}(undef, num_stages)
    
    for t in 1:num_stages
        subproblems[t] = JuMP.read_from_file(joinpath(case_folder, subproblem_file); use_nlp_block = false)
        # Set optimizer if provided (for DiffOpt support)
        if !isnothing(optimizer)
            set_optimizer(subproblems[t], optimizer)
        end
        norm_deficit, _deficit = create_deficit!(subproblems[t], nHyd; penalty=penalty)
        # delete fix constraints
        for con in JuMP.all_constraints(subproblems[t], VariableRef, MOI.EqualTo{Float64})
            delete(subproblems[t], con)
        end
        state_params_in[t], state_param_out, inflow = find_reservoirs_and_inflow(subproblems[t])
        state_params_in[t] = variable_to_parameter.(subproblems[t], state_params_in[t])
        state_params_out[t] = [variable_to_parameter(subproblems[t], state_param_out[i]; deficit=_deficit[i]) for i in 1:nHyd]
        inflow = [(variable_to_parameter(subproblems[t], inflow[i]),  vector_inflows[i][t, :] .+ 0.0) for i in 1:nHyd]
        uncertainty_samples[t] = inflow
    end

    return subproblems, state_params_in, state_params_out, uncertainty_samples, initial_state, max_volume
end

function ensure_feasibility_cap(state_out, state_in, uncertainty, max_volume)
    state_out = max.(state_out, 0)
    state_out = min.(state_out, state_in .+ uncertainty)
    state_out = min.(state_out, max_volume)
    return state_out
end

function ensure_feasibility_double_softplus(state_out, state_in, uncertainty, max_volume)
    actual_max = min.(max_volume, state_in .+ uncertainty)
    return softplus.(state_out .- 0.0) - softplus.(state_out .- actual_max)
end

function ensure_feasibility_sigmoid(state_out, state_in, uncertainty, max_volume)
    return sigmoid.(state_out) .* min.(max_volume, state_in .+ uncertainty)
end