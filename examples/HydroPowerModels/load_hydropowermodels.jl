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

"""
    build_hydropowermodels(case_folder, subproblem_file; num_stages, penalty, penalty_l1,
                           penalty_l2, optimizer, strict) -> (subproblems, state_params_in,
                           state_params_out, uncertainty_samples, initial_state, max_volume,
                           hydro_meta)

Build multi-stage hydro power subproblems from a case folder containing `hydro.json`,
`inflows.csv`, and a MOF subproblem file. Each stage gets its own JuMP model with
parameterized incoming state, outgoing target (with or without deficit slack), and
uncertainty (inflow) samples.

When `strict=true`, the outgoing state is bound to the target via a hard equality
constraint (`reservoir_out == target`) with **no deficit variables** and **no penalty
term**. The dual of this equality is the clean shadow price ∂Q/∂target — pure economic
signal without penalty noise. This requires a feasibility-guaranteeing policy (e.g.
[`HydroReachablePolicy`]) to avoid infeasible subproblems.

When `strict=false` (default), deficit variables are created via [`create_deficit!`](@ref)
and penalized in the objective, allowing the solver to deviate from the target.

# Arguments
- `case_folder::AbstractString`: path to the case directory (must contain `hydro.json`
  and `inflows.csv`)
- `subproblem_file::AbstractString`: MOF filename for the stage subproblem (e.g.
  `"ACPPowerModel.mof.json"`)
- `num_stages`: number of stages (default: number of rows in `inflows.csv`)
- `penalty`: legacy L1 penalty coefficient (use `penalty_l1`/`penalty_l2` instead)
- `penalty_l1`: L1 norm penalty coefficient, or `:auto`
- `penalty_l2`: L2 squared norm penalty coefficient, or `:auto`
- `optimizer`: optimizer factory for DiffOpt, e.g. `() -> DiffOpt.diff_optimizer(...)`
- `strict::Bool=false`: if `true`, use hard equality target constraints (no deficit)

# Returns
A 7-tuple `(subproblems, state_params_in, state_params_out, uncertainty_samples,
initial_state, max_volume, hydro_meta)` where:
- `subproblems::Vector{JuMP.Model}`: one JuMP model per stage
- `state_params_in::Vector{Vector{Any}}`: incoming state parameters per stage
- `state_params_out::Vector{Vector{Tuple{Any,VariableRef}}}`: `(parameter, variable)`
  tuples for outgoing state per stage
- `uncertainty_samples`: joint inflow scenarios per stage
- `initial_state::Vector{Float64}`: initial reservoir volumes
- `max_volume::Vector{Float64}`: maximum reservoir volumes
- `hydro_meta::NamedTuple`: hydro system metadata for policy construction (see below)

## `hydro_meta` fields
- `nHyd::Int`: number of hydro units
- `min_vol`, `max_vol`: per-unit volume bounds
- `min_turn`, `max_turn`: per-unit turbine outflow bounds
- `initial_volume`: initial reservoir volumes
- `downstream_turn`, `downstream_spill`: downstream connectivity (by hydro index)
- `upstream_turn`: `Vector{Vector{Tuple{Int,Float64}}}` — for each unit, list of
  `(upstream_array_pos, upstream_max_turn)` pairs feeding into it
- `upstream_spill`: same structure for spill connections
- `K::Float64`: stage duration in hours
- `production_factor`: per-unit production factors

See also: [`create_deficit!`](@ref), [`variable_to_parameter`](@ref)
"""
function build_hydropowermodels(
    case_folder::AbstractString,
    subproblem_file::AbstractString;
    num_stages=nothing,
    penalty=nothing,
    penalty_l1=nothing,
    penalty_l2=nothing,
    optimizer=nothing,
    strict::Bool=false,
)
    # Parse the hydro system data file
    hydro_json = JSON.parsefile(joinpath(case_folder, "hydro.json"))
    hydro_file = hydro_json["Hydrogenerators"]
    nHyd = length(hydro_file)
    # Extract water-balance conversion factor K from the MOF model's hydro_balance
    # constraint. K converts flow units (m³/s) to volume units (hm³) per stage.
    # hydro.json["stage_hours"] is the stage duration, NOT the water-balance K.
    _tmp_model = JuMP.read_from_file(
        joinpath(case_folder, subproblem_file); use_nlp_block=false
    )
    _hb_con = JuMP.constraint_by_name(_tmp_model, "hydro_balance[1]")
    _hb_func = JuMP.constraint_object(_hb_con).func
    _inflow_var = first(filter(
        v -> occursin("inflow", JuMP.name(v)), JuMP.all_variables(_tmp_model)
    ))
    K = abs(JuMP.coefficient(_hb_func, _inflow_var))
    # Read historical inflow scenarios from CSV
    vector_inflows, nCen, num_stages = read_inflow(
        joinpath(case_folder, "inflows.csv"), nHyd; num_stages=num_stages
    )
    # Extract initial volumes and volume bounds
    initial_state = [hydro["initial_volume"] for hydro in hydro_file]
    max_volume = [hydro["max_volume"] for hydro in hydro_file]
    min_volume = [hydro["min_volume"] for hydro in hydro_file]

    # Build upstream connectivity: for each unit, who feeds into it?
    # The hydro.json stores downstream references; we invert them here.
    # index_to_pos maps the hydro "index" field to the array position (1-based)
    index_to_pos = Dict(hydro["index"] => i for (i, hydro) in enumerate(hydro_file))
    # upstream_turn[r] = [(upstream_array_pos, upstream_max_turn), ...]
    upstream_turn = [Tuple{Int,Float64}[] for _ in 1:nHyd]
    # upstream_spill[r] = [(upstream_array_pos, Inf), ...] — spill is unbounded
    upstream_spill = [Tuple{Int,Float64}[] for _ in 1:nHyd]
    for (i, hydro) in enumerate(hydro_file)
        # Turbine outflow from unit i feeds into each downstream unit
        for ds_idx in hydro["downstream_turn"]
            ds_pos = index_to_pos[ds_idx]
            push!(upstream_turn[ds_pos], (i, hydro["max_turn"]))
        end
        # Spillage from unit i feeds into each downstream unit
        for ds_idx in hydro["downstream_spill"]
            ds_pos = index_to_pos[ds_idx]
            push!(upstream_spill[ds_pos], (i, Inf))
        end
    end

    # Assemble hydro metadata for policy construction (e.g. HydroReachablePolicy)
    hydro_meta = (
        nHyd = nHyd,
        min_vol = min_volume,
        max_vol = max_volume,
        min_turn = [hydro["min_turn"] for hydro in hydro_file],
        max_turn = [hydro["max_turn"] for hydro in hydro_file],
        initial_volume = initial_state,
        downstream_turn = [hydro["downstream_turn"] for hydro in hydro_file],
        downstream_spill = [hydro["downstream_spill"] for hydro in hydro_file],
        upstream_turn = upstream_turn,
        upstream_spill = upstream_spill,
        K = K,
        production_factor = [hydro["production_factor"] for hydro in hydro_file],
    )

    # Allocate per-stage containers
    subproblems = Vector{JuMP.Model}(undef, num_stages)
    state_params_in = Vector{Vector{Any}}(undef, num_stages)
    state_params_out = Vector{Vector{Tuple{Any,VariableRef}}}(undef, num_stages)
    uncertainty_samples = Vector{Vector{Vector{Tuple{VariableRef,Float64}}}}(
        undef, num_stages
    )

    for t in 1:num_stages
        # Read the stage subproblem from MOF file
        subproblems[t] = JuMP.read_from_file(
            joinpath(case_folder, subproblem_file); use_nlp_block=false
        )
        # Set optimizer if provided (for DiffOpt support)
        if !isnothing(optimizer)
            set_optimizer(subproblems[t], optimizer)
        end

        if strict
            # Strict mode: no deficit variables, no penalty — hard equality
            # reservoir_out[i] == target[i] enforced directly
        else
            # Default mode: create deficit variables with penalty
            norm_deficit, _deficit = create_deficit!(
                subproblems[t],
                nHyd;
                penalty=penalty,
                penalty_l1=penalty_l1,
                penalty_l2=penalty_l2,
            )
        end

        # Delete fix constraints (fixed-value equality constraints on variables)
        for con in JuMP.all_constraints(subproblems[t], VariableRef, MOI.EqualTo{Float64})
            delete(subproblems[t], con)
        end
        # Identify reservoir and inflow variables by name pattern
        state_params_in[t], state_param_out, inflow = find_reservoirs_and_inflow(
            subproblems[t]
        )
        # Convert incoming state variables to parameters
        state_params_in[t] = variable_to_parameter.(subproblems[t], state_params_in[t])

        if strict
            # Strict mode: hard equality constraint (no deficit slack)
            # variable_to_parameter without deficit creates: reservoir_out[i] == parameter
            # Returns just the parameter; we manually pair it with the variable
            state_params_out[t] = [
                let param = variable_to_parameter(subproblems[t], state_param_out[i])
                    (param, state_param_out[i])
                end
                for i in 1:nHyd
            ]
        else
            # Default mode: variable_to_parameter with deficit returns (parameter, variable)
            state_params_out[t] = [
                variable_to_parameter(
                    subproblems[t], state_param_out[i]; deficit=_deficit[i]
                )
                for i in 1:nHyd
            ]
        end

        # Joint scenarios: all hydro units share the same scenario index ω,
        # preserving the spatial correlation in the historical inflow data.
        inflow_params = [variable_to_parameter(subproblems[t], inflow[i]) for i in 1:nHyd]
        joint_scenarios = [
            [(inflow_params[i], vector_inflows[i][t, ω] + 0.0) for i in 1:nHyd]
            for ω in 1:nCen
        ]
        uncertainty_samples[t] = joint_scenarios
    end

    return subproblems,
    state_params_in, state_params_out, uncertainty_samples, initial_state,
    max_volume, hydro_meta
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
