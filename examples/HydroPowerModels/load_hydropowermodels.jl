using JuMP
using CSV
using Tables
using JSON
using StableRNGs

# Paired evaluation protocol: at stage t of paired scenario s, EVERY method
# (SDDP.Historical, TS-DDR CPU/GPU rollouts) realizes joint inflow scenario
# `paired_scenario_indices(N, nCen)[t, s]`. StableRNG streams are stable
# across Julia versions, so the protocol is fully defined by this seed — no
# data file to distribute or track.
const PAIRED_SCENARIO_SEED = 20260706
# The generated matrix ALWAYS has this many stage rows (the full SDDP horizon:
# 96 reported + 30 end-of-horizon buffer). Consumers that need fewer stages
# slice rows — they must never generate a smaller matrix, because arrays of
# different shapes consume the RNG stream differently and pairing across
# methods would silently break.
const PAIRED_NUM_STAGES = 126

"""
    paired_scenario_indices(num_scenarios, nCen;
                            seed = PAIRED_SCENARIO_SEED) -> Matrix{Int}

Deterministic paired-evaluation index matrix of fixed shape
`PAIRED_NUM_STAGES × num_scenarios`: entry `[t, s]` is uniform on `1:nCen`
(the per-stage joint inflow support, `nCen = ncol(inflows.csv) ÷ nHyd`).
Slice rows for shorter horizons; never regenerate at a different shape.
"""
function paired_scenario_indices(
    num_scenarios::Integer, nCen::Integer;
    seed::Integer = PAIRED_SCENARIO_SEED,
)
    return rand(StableRNG(seed), 1:Int(nCen), PAIRED_NUM_STAGES, Int(num_scenarios))
end

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

# ── Per-stage demand support ──────────────────────────────────────────────────
#
# The stage subproblem MOF files ship with a single (historically 0.6-scaled)
# demand baked into the per-bus power-balance constraints. The functions below
# replace those baked constants stage by stage with the real seasonal demand
# from `demand.csv`, matching the SDDP baseline (`sddp/run_sddp_inconsistent.jl`
# `load_case_data`) and the GPU companion package DecisionRulesExa.jl
# (`load_demand` + `set_demand!` in its HydroPowerModels example).

"""
    read_load_data(pm_file) -> NamedTuple

Parse the case's `PowerModels.json` for the load-to-bus mapping and the
default (nominal) load values needed for per-stage demand substitution.

Loads are sorted by their PowerModels `index`, so column ``j`` of `demand.csv`
corresponds to the load with `index == j` (for Bolivia, load ``j`` sits at bus
``j``, ``j = 1, \\dots, 26``).

# Arguments
- `pm_file::AbstractString`: path to `PowerModels.json`.

# Returns
A `NamedTuple` with fields:
- `nbus::Int`: number of buses.
- `load_bus::Vector{Int}`: bus of each load, sorted by load index.
- `load_pd::Vector{Float64}`: nominal active demand per load (pu).
- `load_qd::Vector{Float64}`: nominal reactive demand per load (pu).
"""
function read_load_data(pm_file::AbstractString)
    # Parse the PowerModels network description
    pm = JSON.parsefile(pm_file)
    # Number of buses (bus indices are 1-based and consecutive for Bolivia)
    nbus = length(pm["bus"])
    # Loads sorted by PowerModels index so demand.csv column j == load index j
    loads = sort!(collect(values(pm["load"])); by=l -> l["index"])
    load_bus = [Int(l["load_bus"]) for l in loads]
    load_pd = [Float64(get(l, "pd", 0.0)) for l in loads]
    load_qd = [Float64(get(l, "qd", 0.0)) for l in loads]
    return (nbus=nbus, load_bus=load_bus, load_pd=load_pd, load_qd=load_qd)
end

"""
    read_demand(file, num_loads; num_stages) -> Matrix{Float64}

Read a per-stage demand CSV (rows = stages of one annual cycle, columns =
loads by PowerModels index) and tile it cyclically over `num_stages` stages:

```math
D^{\\mathrm{tiled}}_{t,j} = D_{((t-1) \\bmod n_{\\mathrm{rows}}) + 1,\\; j},
\\qquad t = 1, \\dots, T.
```

This is the same cyclic tiling used by the SDDP baseline
(`sddp/run_sddp_inconsistent.jl`) and DecisionRulesExa.jl's `load_demand`,
so all methods see identical stage demands.

# Arguments
- `file::AbstractString`: path to `demand.csv` (no header; pu units).
- `num_loads::Int`: expected number of columns (loads); a mismatch warns.
- `num_stages::Int`: horizon length ``T`` to tile to.

# Returns
- `Matrix{Float64}` of size `num_stages × num_loads`.
"""
function read_demand(file::AbstractString, num_loads::Int; num_stages::Int)
    # Read the raw stage × load demand table (no header)
    raw = CSV.read(file, Tables.matrix; header=false)
    nrows, ncols = size(raw)
    # Guard against a stale demand file that does not match the network
    ncols == num_loads ||
        @warn "demand file has $ncols columns but the case has $num_loads loads"
    # Cyclic tiling: stage t uses annual-cycle row ((t-1) mod nrows) + 1
    demand = Matrix{Float64}(undef, num_stages, ncols)
    for t in 1:num_stages, j in 1:ncols
        demand[t, j] = Float64(raw[((t - 1) % nrows) + 1, j])
    end
    return demand
end

"""
    find_bus_balance_constraints(model) -> (active, reactive)

Locate the per-bus active and reactive power-balance constraints of an OPF
stage subproblem read from a MOF file, keyed by bus index.

PowerModels writes both nodal balances as scalar equality constraints whose
normalized right-hand side carries (minus) the bus demand:

```math
\\sum_{a \\in A_b} p_a - \\sum_{g \\in G_b} pg_g - \\mathrm{deficit}_b
    = -pd_b, \\qquad
\\sum_{a \\in A_b} q_a - \\sum_{g \\in G_b} qg_g \\; (+\\, b^{sh}_b vm_b^2)
    = -qd_b,
```

so per-stage demand substitution reduces to `set_normalized_rhs`. Constraints
are identified structurally (constraint names are not preserved by the MOF
round-trip):
- **active** balance at bus ``b``: the unique affine/quadratic equality
  containing the load-shedding variable `deficit[b]` (HydroPowerModels adds
  one per bus);
- **reactive** balance at bus ``b``: the unique affine/quadratic equality
  containing a reactive branch-flow variable `0_q[(l, b, j)]` (the second
  tuple element of a PowerModels arc is the bus whose balance it enters) and
  no `deficit[...]` variable.

The active-balance orientation is validated: the coefficient of `deficit[b]`
must be ``-1`` (the PowerModels/JuMP canonical form above); otherwise an
error is thrown rather than silently writing a wrong-signed demand.

# Arguments
- `model::JuMP.Model`: a stage subproblem read from the MOF file.

# Returns
- `(active, reactive)`: two `Dict{Int,JuMP.ConstraintRef}` mapping bus index
  to its balance constraint. `reactive` is empty for formulations without
  reactive balances (e.g. DC).
"""
function find_bus_balance_constraints(model::JuMP.Model)
    active = Dict{Int,JuMP.ConstraintRef}()
    reactive = Dict{Int,JuMP.ConstraintRef}()
    # Reactive branch-flow (arc) variable name: 0_q[(line, from_bus, to_bus)]
    arc_regex = r"^0_q\[\((\d+), (\d+), (\d+)\)\]$"
    # Scan affine and quadratic equalities (nonlinear AC flow-definition rows
    # are ScalarNonlinearFunction and are correctly excluded by these types)
    for F in (JuMP.AffExpr, JuMP.QuadExpr)
        for con in JuMP.all_constraints(model, F, MOI.EqualTo{Float64})
            func = JuMP.constraint_object(con).func
            # Affine part of the function (QuadExpr wraps an AffExpr)
            aff = F === JuMP.QuadExpr ? func.aff : func
            # Active balance: contains the bus load-shedding variable deficit[b]
            found_deficit = false
            for (var, coef) in aff.terms
                vname = JuMP.name(var)
                if startswith(vname, "deficit[")
                    # Bus index from the variable name "deficit[b]"
                    bus = parse(Int, vname[(length("deficit[") + 1):(end - 1)])
                    # Validate canonical orientation so RHS = -pd is correct
                    coef ≈ -1.0 || error(
                        "deficit[$bus] enters its balance with coefficient " *
                        "$coef (expected -1); demand substitution would be " *
                        "wrong-signed — regenerate the MOF file",
                    )
                    active[bus] = con
                    found_deficit = true
                    break
                end
            end
            found_deficit && continue
            # Reactive balance: contains a q-arc variable; its second tuple
            # element is the bus whose balance this constraint expresses
            for (var, _) in aff.terms
                m = match(arc_regex, JuMP.name(var))
                if !isnothing(m)
                    reactive[parse(Int, m.captures[2])] = con
                    break
                end
            end
        end
    end
    return active, reactive
end

"""
    set_bus_demand!(active, reactive, pd_bus, qd_bus) -> Nothing

Overwrite the demand baked into the per-bus balance constraints of one stage
subproblem. For each bus ``b`` the normalized right-hand side is set to

```math
\\mathrm{rhs}^{P}_b = -pd_b, \\qquad \\mathrm{rhs}^{Q}_b = -qd_b,
```

matching the PowerModels canonical balance orientation validated by
[`find_bus_balance_constraints`](@ref). Buses absent from a dictionary (e.g.
no reactive balance in DC formulations) are skipped.

# Arguments
- `active::Dict{Int,JuMP.ConstraintRef}`: bus → active balance constraint.
- `reactive::Dict{Int,JuMP.ConstraintRef}`: bus → reactive balance constraint.
- `pd_bus::Vector{Float64}`: per-bus active demand (pu) for this stage.
- `qd_bus::Vector{Float64}`: per-bus reactive demand (pu) for this stage.
"""
function set_bus_demand!(
    active::Dict{Int,JuMP.ConstraintRef},
    reactive::Dict{Int,JuMP.ConstraintRef},
    pd_bus::Vector{Float64},
    qd_bus::Vector{Float64},
)
    # Active balance: rhs = -pd (canonical orientation, validated at discovery)
    for (bus, con) in active
        JuMP.set_normalized_rhs(con, -pd_bus[bus])
    end
    # Reactive balance: rhs = -qd (same exporter, same orientation)
    for (bus, con) in reactive
        JuMP.set_normalized_rhs(con, -qd_bus[bus])
    end
    return nothing
end

"""
    set_load_deficit_cost!(model, deficit_cost) -> Nothing

Set the objective coefficient of every per-bus load-shedding variable
`deficit[b]` to `deficit_cost`:

```math
\\text{objective} \\mathrel{+}= c_{\\mathrm{def}} \\sum_b \\mathrm{deficit}_b,
```

replacing the cost baked into the MOF file (historically
``60\\,\\$/\\mathrm{MWh} \\times \\mathrm{baseMVA}\\,100 = 6{,}000`` per pu).
The paper recipe uses ``c_{\\mathrm{def}} = 10^5``, matching
DecisionRulesExa.jl's `deficit_cost` (a low shedding cost lets the solver
serve the seasonal peak by shedding load instead of storing water).

Must be called **after** [`create_deficit!`](@ref) in non-strict mode: that
function derives `:auto` target penalties from the maximum objective
coefficient, which must keep its historical (pre-override) value.

# Arguments
- `model::JuMP.Model`: stage subproblem containing `deficit[b]` variables.
- `deficit_cost::Real`: load-shedding cost per pu (``10^5`` in the paper).
"""
function set_load_deficit_cost!(model::JuMP.Model, deficit_cost::Real)
    for var in JuMP.all_variables(model)
        # Only the per-bus load-shedding variables named "deficit[b]"
        if startswith(JuMP.name(var), "deficit[")
            JuMP.set_objective_coefficient(model, var, Float64(deficit_cost))
        end
    end
    return nothing
end

"""
    build_hydropowermodels(case_folder, subproblem_file; num_stages, penalty, penalty_l1,
                           penalty_l2, optimizer, strict, demand_file, load_scaler,
                           deficit_cost) -> (subproblems, state_params_in,
                           state_params_out, uncertainty_samples, initial_state, max_volume,
                           hydro_meta)

Build multi-stage hydro power subproblems from a case folder containing `hydro.json`,
`inflows.csv`, and a MOF subproblem file. Each stage gets its own JuMP model with
parameterized incoming state, outgoing target (with or without deficit slack), and
uncertainty (inflow) samples.

# Per-stage demand and deficit cost (parity with SDDP / DecisionRulesExa.jl)

When the case folder contains `demand.csv` (or `demand_file` points to one),
the demand baked into the MOF file (historically ``0.6 \\times`` the
`PowerModels.json` loads) is **replaced stage by stage**: the active demand of
load ``j`` at stage ``t`` is the cyclically tiled CSV entry

```math
pd_{t,j} = s \\cdot D_{((t-1) \\bmod n_{\\mathrm{rows}}) + 1,\\; j},
```

with `load_scaler` ``s`` (default 1 — the real seasonal demand, no 0.6
scaler), and the reactive demand is the nominal `PowerModels.json` value
``qd_j`` scaled by the same ``s``. This matches the SDDP baseline
(`sddp/run_sddp_inconsistent.jl` `load_case_data`) and DecisionRulesExa.jl
(`load_demand` with `load_scaler=1.0`). Without a demand file the baked MOF
demand is left untouched (historical behavior).

Independently, `deficit_cost` (default ``10^5``, the paper recipe shared with
DecisionRulesExa.jl) overrides the objective coefficient of the per-bus
load-shedding variables `deficit[b]`; pass `nothing` to keep the baked
coefficient (historically 6,000 per pu). The override is applied **after**
[`create_deficit!`](@ref) so `:auto` target penalties keep their historical
value.

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
- `demand_file=:auto`: per-stage demand CSV. `:auto` uses
  `case_folder/demand.csv` when it exists; `nothing` disables the substitution
  (keep the demand baked into the MOF); a path string forces a specific file
- `load_scaler::Real=1.0`: scaler ``s`` applied to both the per-stage active
  demand and the nominal reactive demand when a demand file is in effect
  (1.0 = real seasonal demand; only used with a demand file)
- `deficit_cost=1e5`: objective coefficient of the per-bus load-shedding
  variables `deficit[b]`, or `nothing` to keep the baked MOF coefficient

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
- `K::Float64`: water-balance conversion factor from flow units to volume units
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
    demand_file=:auto,
    load_scaler::Real=1.0,
    deficit_cost=1e5,
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
    # Extract volume bounds
    max_volume = [hydro["max_volume"] for hydro in hydro_file]
    min_volume = [hydro["min_volume"] for hydro in hydro_file]
    # Initial volumes clamped into [min_vol, max_vol] (parity with
    # DecisionRulesExa.jl): Bolivia's CHJ unit has max_volume = 0 and a
    # denormal ~1e-316 initial_volume in hydro.json, which would sit above its
    # upper bound; clamping maps it (and any other out-of-bounds value) onto
    # the feasible box, so x0 is always a feasible reservoir state.
    initial_state = [
        clamp(hydro["initial_volume"], min_volume[i], max_volume[i])
        for (i, hydro) in enumerate(hydro_file)
    ]

    # ── Per-stage demand (parity with SDDP + DecisionRulesExa.jl) ─────────────
    # Resolve the demand file: :auto uses case_folder/demand.csv when present.
    resolved_demand_file = if demand_file === :auto
        _f = joinpath(case_folder, "demand.csv")
        isfile(_f) ? _f : nothing
    else
        demand_file
    end
    # Per-stage per-bus active demand [num_stages × nbus] and constant per-bus
    # reactive demand [nbus], both already scaled by load_scaler; or nothing.
    pd_bus, qd_bus = if isnothing(resolved_demand_file)
        nothing, nothing
    else
        # Load → bus mapping and nominal reactive demand from PowerModels.json
        load_data = read_load_data(joinpath(case_folder, "PowerModels.json"))
        # Active demand per load, tiled cyclically over the horizon
        demand = read_demand(
            resolved_demand_file, length(load_data.load_bus); num_stages=num_stages
        )
        # Aggregate loads onto buses: pd_bus[t, b] = s · Σ_{j: bus(j)=b} D[t, j]
        _pd = zeros(Float64, num_stages, load_data.nbus)
        for (j, bus) in enumerate(load_data.load_bus)
            j > size(demand, 2) && break
            for t in 1:num_stages
                _pd[t, bus] += load_scaler * demand[t, j]
            end
        end
        # Reactive demand stays at the (scaled) nominal PowerModels.json value:
        # qd_bus[b] = s · Σ_{j: bus(j)=b} qd_j (SDDP's set_active_demand!
        # touches only pd; DecisionRulesExa uses default_bus_reactive_demand)
        _qd = zeros(Float64, load_data.nbus)
        for (j, bus) in enumerate(load_data.load_bus)
            _qd[bus] += load_scaler * load_data.load_qd[j]
        end
        _pd, _qd
    end

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

        # Replace the baked MOF demand with this stage's seasonal demand:
        # active balance rhs = -pd_bus[t, b], reactive balance rhs = -qd_bus[b]
        if !isnothing(pd_bus)
            active_cons, reactive_cons = find_bus_balance_constraints(subproblems[t])
            set_bus_demand!(active_cons, reactive_cons, vec(pd_bus[t, :]), qd_bus)
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

        # Override the load-shedding cost AFTER create_deficit! so that :auto
        # target penalties (max |objective coefficient| at call time) keep the
        # historical baked value instead of picking up the 1e5 override.
        if !isnothing(deficit_cost)
            set_load_deficit_cost!(subproblems[t], deficit_cost)
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
