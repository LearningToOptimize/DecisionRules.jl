#!/usr/bin/env julia

# Frozen case contract for the public Bolivia hydro example.
#
# This file is the single machine-readable description of the case that the
# published TS-DDR-vs-SDDP comparison was actually run on, plus the verifier
# that proves a checkout still matches it. It is byte-identical in
# DecisionRules.jl and DecisionRulesExa.jl so that either engine can assert the
# same contract without depending on the other.
#
# It VERIFIES; it does not repair. The three case inputs are consumed exactly as
# committed — no initial-volume repair, no demand atoms, no re-export of the
# reference MathOptFormat models. Every quantity below is either read from those
# bytes or is a constant of the evaluation protocol.
#
# Usage:
#   julia --project generate_canonical_case_artifacts.jl            # verify + (re)write bolivia/case_manifest.json
#   julia --project generate_canonical_case_artifacts.jl --verify   # verify only; fails if the manifest disagrees
#   julia --project generate_canonical_case_artifacts.jl --exa-root=/path/to/DecisionRulesExa.jl
#                                                                   # additionally mirror the frozen bytes to the other engine

module HydroCanonicalCase

using JSON
using SHA
using StableRNGs

# ── Frozen case inputs ────────────────────────────────────────────────────────
# The three files below are the WHOLE case. They are byte-identical in both
# public packages; `verify_inputs` refuses to proceed on any other bytes.
#
# `hydro.json` is the RAW upstream file. Its `initial_volume` entries are
# denormal doubles near 9e-316 — i.e. zero to ~300 orders of magnitude below the
# smallest reservoir capacity in the case (0.042 pu). Both engines clamp the
# initial state into `[min_volume, max_volume]` and then use it at the engine's
# working precision, which leaves every reservoir empty at stage 1. That empty
# start is the state the published result was produced from; see
# `INITIAL_STATE_NOTE`.
const INPUT_HASHES = Dict(
    "PowerModels.json" => "1ff598447957f9fc17ca570415bf5b9b5b14e1292ea3bd3163db0ad79911a782",
    "hydro.json" => "b25ce1c7bafcfaf907091dcd1007949c79a79974c9a33020b2587400d756b29a",
    "inflows.csv" => "5afb275dff3fc879e3e93b6510b81295834faad0bcd2bd1fc070a8e3e6653c77",
)

# Files whose PRESENCE would mean a different case. `demand.csv` /
# `demand_scenarios.csv` would introduce demand uncertainty; the `bolivia_*`
# variants are the abandoned grid/demand case-design candidates. The frozen
# experiment uses deterministic demand and inflow uncertainty only, so the
# verifier fails closed if any of these reappear.
const FORBIDDEN_CASE_FILES = (
    "demand.csv",
    "demand_scenarios.csv",
    "demand_noise.csv",
)

# ── Load and cost conventions ─────────────────────────────────────────────────
const ACTIVE_LOAD_FACTOR = 0.6      # pd  <- 0.6 * PowerModels.json pd, every stage
const REACTIVE_LOAD_FACTOR = 0.6    # qd  <- 0.6 * PowerModels.json qd, every stage
const DEMAND_IS_DETERMINISTIC = true
const ACTIVE_DEFICIT_COST = 6000.0  # USD per pu of shed active power per stage
const ACTIVE_DEFICIT_COST_DERIVATION = "cost_deficit 60 USD/MWh * baseMVA 100"
const REACTIVE_BALANCE = "hard"     # no reactive slack variable anywhere

# ── Water balance ─────────────────────────────────────────────────────────────
# The stage water balance converts flow (m^3/s) to stored volume with
# K = 0.0036 * stage_hours. Weekly stages give K = 0.6048; a run that leaves
# stage_hours at its default of 1 silently models a week as an hour, so both
# numbers are asserted against `hydro.json` rather than assumed.
const STAGE_HOURS = 168
const HYDRO_CONVERSION_K = 0.6048

# ── Horizon ───────────────────────────────────────────────────────────────────
# 126 stages are simulated; costs are reported over the first 96. The 30-stage
# tail is a look-ahead buffer that keeps the reported window free of end-of-
# horizon reservoir dumping.
const REPORTING_STAGES = 96
const LOOKAHEAD_STAGES = 30
const TOTAL_STAGES = REPORTING_STAGES + LOOKAHEAD_STAGES

# ── Evaluation protocol ───────────────────────────────────────────────────────
# The paired protocol is reproducible by construction rather than stored: entry
# [t, s] of `rand(StableRNG(20260706), 1:nCen, 126, 500)` is the inflow scenario
# realized at stage t of paired column s. Array SHAPE is part of the contract —
# an RNG stream consumed into a differently shaped array yields different draws,
# so every consumer must generate exactly 126 x 500 and slice what it needs.
const INFLOW_PROTOCOL_SEED = 20260706
const PROTOCOL_STAGES = 126
const PROTOCOL_SCENARIOS = 500
const PROTOCOL_SCENARIO_IDS = "1:500 (global column ids; shards must preserve them)"

# ── Method configuration held fixed across both policies ──────────────────────
const TSDDR_FORMULATION = "ACPPowerModel"          # true AC, polar; training AND evaluation
const TSDDR_TARGET_MODE = "strict"                 # reservoir targets are equalities, no slack
const TSDDR_TARGET_ACTIVATION = "stretchedsigmoid" # maps onto [0, 1 - 1e-3] of the reachable set
const SDDP_BACKWARD_FORMULATION = "SOCWRConicPowerModel"
const SDDP_FORWARD_FORMULATION = "ACPPowerModel"

# ── Serialized stage subproblems (MathOptFormat) ──────────────────────────────
# `<formulation>.mof.json` is the one-stage OPF subproblem, serialized by JuMP's
# MathOptFormat writer from the model HydroPowerModels builds out of the three
# frozen inputs above. It is a GENERATED artifact, never hand-edited:
# `export_subproblem_mof.jl` is the only supported producer, and it regenerates
# all three formulations from the unchanged case in one pass.
#
# These files are load-bearing. The JuMP/MAIN workflow — `build_hydropowermodels`
# in `load_hydropowermodels.jl`, and therefore `train_dr_hydropowermodels_strict.jl`,
# `eval_paired_tsddr.jl` and `eval_jump_de.jl` — reads one copy per stage and
# re-derives the water-balance coefficient from it, failing closed unless that
# coefficient equals `0.0036 * stage_hours`. The SDDP baseline instead builds
# through HydroPowerModels directly, and the ExaModels engine builds its own
# `ExaModel`; all three therefore have to agree, which is what the full-solution
# parity gate checks.
const FORMULATIONS = ("ACPPowerModel", "DCPPowerModel", "SOCWRConicPowerModel")
const GENERATED_MODEL_NOTE =
    "One-stage subproblem exports generated by export_subproblem_mof.jl from the " *
    "frozen inputs through HydroPowerModels with stage_hours = $(STAGE_HOURS), so " *
    "the hydro-balance inflow coefficient is the case's K = $(HYDRO_CONVERSION_K). " *
    "The JuMP/MAIN workflow loads these files as its stage subproblems; SDDP builds " *
    "through HydroPowerModels and the ExaModels engine builds its own model."

# A reservoir volume below this is zero for every purpose in this case: the
# smallest nonzero capacity is 0.042 pu, ~314 orders of magnitude larger.
const EMPTY_VOLUME_TOL = 1e-300

const INITIAL_STATE_NOTE =
    "Empty start. hydro.json carries denormal initial_volume values near 9e-316; " *
    "both engines clamp the initial state into [min_volume, max_volume] and " *
    "evaluate it at working precision, which leaves every reservoir at zero. " *
    "No 70%-of-capacity repair is applied — the published result was produced " *
    "from the raw bytes."

"""
    sha256_file(path) -> String

Hex-encoded SHA-256 of the file at `path`.
"""
sha256_file(path::AbstractString) = bytes2hex(open(SHA.sha256, path))

"""
    initial_state(case_dir) -> Vector{Float64}

Reservoir volumes at stage 1, computed the way both engines compute them:
`clamp(initial_volume, min_volume, max_volume)` per unit.

# Arguments
- `case_dir::AbstractString`: directory holding `hydro.json`.

# Returns
- `Vector{Float64}`: one clamped initial volume per hydro unit, in file order.
"""
function initial_state(case_dir::AbstractString)
    hydro = JSON.parsefile(joinpath(case_dir, "hydro.json"))["Hydrogenerators"]
    return [
        clamp(
            Float64(unit["initial_volume"]),
            Float64(unit["min_volume"]),
            Float64(unit["max_volume"]),
        )
        for unit in hydro
    ]
end

"""
    scale_main_loads!(alldata) -> typeof(alldata)

Apply the canonical demand convention to a parsed HydroPowerModels case, in
place: every load's `pd` is multiplied by `ACTIVE_LOAD_FACTOR` and every load's
`qd` by `REACTIVE_LOAD_FACTOR`, in every stage's power-system dictionary.

`PowerModels.json` is kept byte-exact on disk, so the 0.6 factor exists only as
this runtime step. Applying it symmetrically to active AND reactive demand is
part of the frozen contract: an asymmetric scaling changes the reactive balance
and therefore the AC feasible set. Every consumer of the case — the MOF
exporter, the SDDP baseline, and the ExaModels builder — must call this or
reproduce it exactly.

The per-stage active load-shedding price is re-asserted at the same time:
`cost_deficit` is stored per MVA, so `ACTIVE_DEFICIT_COST / baseMVA` is the
value that makes the objective coefficient of `deficit[b]` equal
`ACTIVE_DEFICIT_COST`. `verify_inputs` has already checked that the committed
bytes imply exactly that, so this is a no-op assertion in normal operation and a
loud one otherwise.

# Arguments
- `alldata`: the vector of per-stage dictionaries returned by
  `HydroPowerModels.parse_folder`.

# Returns
- `alldata`, mutated in place.
"""
function scale_main_loads!(alldata)
    for data in alldata
        for load in values(data["powersystem"]["load"])
            load["pd"] *= ACTIVE_LOAD_FACTOR
            load["qd"] *= REACTIVE_LOAD_FACTOR
        end
        data["powersystem"]["cost_deficit"] =
            ACTIVE_DEFICIT_COST / Float64(data["powersystem"]["baseMVA"])
    end
    return alldata
end

"""
    verify_inputs(case_dir) -> Dict

Assert every property of the frozen case that can be checked from the committed
bytes, and return the topology counts.

Checks, in order: the three input hashes; the absence of any file that would
introduce demand uncertainty; `stage_hours` and the derived water-balance `K`;
`baseMVA`; the active load-shedding coefficient; and that the clamped initial
state is empty in both Float64 and Float32.

# Arguments
- `case_dir::AbstractString`: directory holding the three frozen inputs.

# Returns
- `Dict{String,Int}`: bus / branch / generator / load / hydro-unit counts.

# Throws
- `ErrorException` on the first violated invariant, naming the expected and the
  observed value.
"""
function verify_inputs(case_dir::AbstractString)
    for (name, expected) in INPUT_HASHES
        path = joinpath(case_dir, name)
        isfile(path) || error("missing frozen case input: $path")
        actual = sha256_file(path)
        actual == expected ||
            error("frozen input hash mismatch for $path: expected $expected, got $actual")
    end

    for name in FORBIDDEN_CASE_FILES
        path = joinpath(case_dir, name)
        isfile(path) && error(
            "$path is present. The frozen case has DETERMINISTIC demand and " *
            "inflow uncertainty only; a demand file means a different experiment.",
        )
    end

    hydro = JSON.parsefile(joinpath(case_dir, "hydro.json"))
    Int(hydro["stage_hours"]) == STAGE_HOURS ||
        error("stage_hours must be $STAGE_HOURS, got $(hydro["stage_hours"])")
    k = 0.0036 * Int(hydro["stage_hours"])
    k == HYDRO_CONVERSION_K ||
        error("water-balance K must be $HYDRO_CONVERSION_K, got $k")

    power = JSON.parsefile(joinpath(case_dir, "PowerModels.json"))
    Float64(power["baseMVA"]) == 100.0 ||
        error("frozen baseMVA must be 100, got $(power["baseMVA"])")
    coefficient = Float64(power["cost_deficit"]) * Float64(power["baseMVA"])
    coefficient == ACTIVE_DEFICIT_COST ||
        error("active load-shedding coefficient must be $ACTIVE_DEFICIT_COST, got $coefficient")

    # Empty start. Checking BOTH precisions matters: MAIN runs the state in
    # Float64 (where the denormal survives as ~9e-316, i.e. zero to within
    # EMPTY_VOLUME_TOL) while the ExaModels engine runs it in Float32 (where the
    # same denormal underflows to an exact zero). Either way stage 1 starts dry.
    x0 = initial_state(case_dir)
    bad = findall(v -> !(v <= EMPTY_VOLUME_TOL), x0)
    isempty(bad) || error(
        "frozen case must start from empty reservoirs (<= $EMPTY_VOLUME_TOL); " *
        "units $bad hold $(x0[bad]). A 70%-of-capacity initial-volume repair is " *
        "NOT part of this case.",
    )
    all(Float32.(x0) .== 0.0f0) ||
        error("clamped initial state must be exactly zero in Float32; got $(Float32.(x0))")

    return Dict(
        "buses" => length(power["bus"]),
        "branches" => length(power["branch"]),
        "generators" => length(power["gen"]),
        "loads" => length(power["load"]),
        "hydro_units" => length(hydro["Hydrogenerators"]),
    )
end

"""
    protocol_indices(ncen) -> Matrix{Int}

The frozen paired protocol: `[t, s]` is the inflow scenario realized at stage
`t` of paired column `s`.

# Arguments
- `ncen::Integer`: number of inflow scenarios available in `inflows.csv`.

# Returns
- `Matrix{Int}`: a `PROTOCOL_STAGES` x `PROTOCOL_SCENARIOS` index matrix.

# Notes
Generated with `StableRNGs.StableRNG(INFLOW_PROTOCOL_SEED)`. The shape is part
of the contract: a differently shaped `rand` call consumes the same stream
differently and yields a different protocol.
"""
function protocol_indices(ncen::Integer)
    return rand(
        StableRNG(INFLOW_PROTOCOL_SEED),
        1:ncen,
        PROTOCOL_STAGES,
        PROTOCOL_SCENARIOS,
    )
end

"""
    inflow_scenario_count(case_dir) -> Int

Number of inflow scenarios in `inflows.csv`, derived as columns / hydro units.
"""
function inflow_scenario_count(case_dir::AbstractString)
    header = split(first(eachline(joinpath(case_dir, "inflows.csv"))), ',')
    nhyd = length(JSON.parsefile(joinpath(case_dir, "hydro.json"))["Hydrogenerators"])
    length(header) % nhyd == 0 ||
        error("inflows.csv has $(length(header)) columns, not a multiple of $nhyd units")
    return length(header) ÷ nhyd
end

"""
    protocol_digest(case_dir) -> (ncen, sha256)

SHA-256 of the frozen protocol index matrix, serialized column-major as
comma-separated decimal integers.

This is the reproducibility artifact for the protocol: it is small, it is
independent of any stored CSV, and it fails if either the seed, the shape, or
the inflow-scenario count changes.
"""
function protocol_digest(case_dir::AbstractString)
    ncen = inflow_scenario_count(case_dir)
    indices = protocol_indices(ncen)
    context = SHA.SHA256_CTX()
    for (i, v) in enumerate(indices)
        SHA.update!(context, codeunits(i == 1 ? string(v) : "," * string(v)))
    end
    return (ncen = ncen, sha256 = bytes2hex(SHA.digest!(context)))
end

"""
    verify_generated_model(path, formulation) -> Dict

Check the structural invariants of a generated MathOptFormat stage export and
return its summary.

Asserted: minimization sense; 28 active-deficit objective terms all priced at
`ACTIVE_DEFICIT_COST`; 11 hydro balances each with exactly one inflow term, all
agreeing on that term's coefficient, whose magnitude must be the case's
`HYDRO_CONVERSION_K`; no mixing of operational active deficit with
reservoir-target slack; and, for `ACPPowerModel`, 28 hard reactive-balance
equations plus both-ended apparent-power limits.

The water-balance assertion is the one that catches the failure this case has
already suffered once: an export taken with `stage_hours` left at its default of
1 carries K = 0.0036 and silently models a week as an hour. Such a file is
rejected here, and again by `build_hydropowermodels` when it loads a stage.
"""
function verify_generated_model(path::AbstractString, formulation::AbstractString)
    isfile(path) || error("missing reference model export: $path")
    model = JSON.parsefile(path)

    model["objective"]["sense"] == "min" ||
        error("$formulation objective is not a minimization")

    terms = get(model["objective"]["function"], "terms", Any[])
    deficit_terms = [t for t in terms if startswith(get(t, "variable", ""), "deficit[")]
    length(deficit_terms) == 28 ||
        error("$formulation must price 28 active-deficit terms, found $(length(deficit_terms))")
    all(Float64(t["coefficient"]) == ACTIVE_DEFICIT_COST for t in deficit_terms) ||
        error("$formulation active-deficit coefficient is not $ACTIVE_DEFICIT_COST")

    balances = [
        c for c in model["constraints"]
        if startswith(get(c, "name", ""), "hydro_balance[")
    ]
    length(balances) == 11 ||
        error("$formulation must have 11 hydro balances, found $(length(balances))")
    coefficients = Float64[]
    for constraint in balances
        inflow_terms = [
            t for t in constraint["function"]["terms"]
            if startswith(get(t, "variable", ""), "inflow[")
        ]
        length(inflow_terms) == 1 ||
            error("$formulation hydro balance does not have exactly one inflow term")
        push!(coefficients, abs(Float64(only(inflow_terms)["coefficient"])))
    end
    length(unique(coefficients)) == 1 ||
        error("$formulation hydro balances disagree on the inflow coefficient: $(unique(coefficients))")
    only(unique(coefficients)) == HYDRO_CONVERSION_K || error(
        "$formulation hydro-balance inflow coefficient is " *
        "$(only(unique(coefficients))), not the case's K = $HYDRO_CONVERSION_K. " *
        "Regenerate with export_subproblem_mof.jl, which passes " *
        "stage_hours = $STAGE_HOURS into HydroPowerModels.",
    )

    any(startswith(get(v, "name", ""), "target_deficit") for v in model["variables"]) &&
        error("$formulation export mixes operational active deficit with target slack")

    if formulation == "ACPPowerModel"
        reactive = [
            c for c in model["constraints"]
            if get(c["set"], "type", "") == "EqualTo" && any(
                startswith(get(t, "variable", ""), "0_q[")
                for t in get(c["function"], "terms", Any[])
            )
        ]
        length(reactive) >= 28 ||
            error("ACP export is missing hard reactive-balance equations (found $(length(reactive)))")
        quadratic = [
            c for c in model["constraints"]
            if get(c["function"], "type", "") == "ScalarQuadraticFunction" &&
               get(c["set"], "type", "") == "LessThan"
        ]
        length(quadratic) >= 62 ||
            error("ACP export is missing both-ended apparent-power limits (found $(length(quadratic)))")
    end

    return Dict(
        "sha256" => sha256_file(path),
        "variables" => length(model["variables"]),
        "constraints" => length(model["constraints"]),
        "objective_sense" => model["objective"]["sense"],
        "active_deficit_terms" => length(deficit_terms),
        "hydro_balances" => length(balances),
        "hydro_balance_inflow_coefficient" => only(unique(coefficients)),
        "matches_case_K" => true,
    )
end

"""
    build_manifest(case_dir) -> Dict

Verify the case and assemble the full frozen-contract manifest.
"""
function build_manifest(case_dir::AbstractString)
    counts = verify_inputs(case_dir)
    digest = protocol_digest(case_dir)
    x0 = initial_state(case_dir)

    return Dict(
        "schema_version" => 2,
        "case" => "Bolivia (upstream case, unmodified)",
        "frozen_on" => "2026-08-02",
        "input_hashes" => INPUT_HASHES,
        "forbidden_case_files" => collect(FORBIDDEN_CASE_FILES),
        "topology_counts" => counts,

        "initial_state" => Dict(
            "effective" => "empty (all reservoirs at zero)",
            "mechanism" => "clamp(initial_volume, min_volume, max_volume) at engine precision",
            "raw_initial_volume_max" => maximum(x0),
            "empty_volume_tolerance" => EMPTY_VOLUME_TOL,
            "float32_is_exactly_zero" => all(Float32.(x0) .== 0.0f0),
            "note" => INITIAL_STATE_NOTE,
        ),

        "water_balance" => Dict(
            "stage_hours" => STAGE_HOURS,
            "K" => HYDRO_CONVERSION_K,
            "K_derivation" => "0.0036 * stage_hours",
        ),

        "demand" => Dict(
            "active_load_factor" => ACTIVE_LOAD_FACTOR,
            "reactive_load_factor" => REACTIVE_LOAD_FACTOR,
            "deterministic" => DEMAND_IS_DETERMINISTIC,
            "uncertainty" => "none; inflow uncertainty only",
        ),

        "costs" => Dict(
            "active_deficit_cost_usd_per_pu_stage" => ACTIVE_DEFICIT_COST,
            "active_deficit_cost_derivation" => ACTIVE_DEFICIT_COST_DERIVATION,
            "reactive_balance" => REACTIVE_BALANCE,
        ),

        "horizon" => Dict(
            "reporting_stages" => REPORTING_STAGES,
            "lookahead_stages" => LOOKAHEAD_STAGES,
            "total_stages" => TOTAL_STAGES,
        ),

        "protocol" => Dict(
            "seed" => INFLOW_PROTOCOL_SEED,
            "rng" => "StableRNG(seed); rand(1:nCen, $(PROTOCOL_STAGES), $(PROTOCOL_SCENARIOS))",
            "stages" => PROTOCOL_STAGES,
            "scenarios" => PROTOCOL_SCENARIOS,
            "scenario_ids" => PROTOCOL_SCENARIO_IDS,
            "inflow_scenarios" => digest.ncen,
            "indices_sha256" => digest.sha256,
            "uncertainty" => "inflow only",
        ),

        "method" => Dict(
            "tsddr_formulation" => TSDDR_FORMULATION,
            "tsddr_target_mode" => TSDDR_TARGET_MODE,
            "tsddr_target_activation" => TSDDR_TARGET_ACTIVATION,
            "sddp_backward_formulation" => SDDP_BACKWARD_FORMULATION,
            "sddp_forward_formulation" => SDDP_FORWARD_FORMULATION,
        ),

        "stage_models" => Dict{String,Any}(
            "note" => GENERATED_MODEL_NOTE,
            "generator" => "export_subproblem_mof.jl",
            "consumed_by" => "build_hydropowermodels (JuMP/MAIN stage subproblems)",
            "exports" => Dict(
                formulation => verify_generated_model(
                    joinpath(case_dir, formulation * ".mof.json"),
                    formulation,
                )
                for formulation in FORMULATIONS
            ),
        ),
    )
end

"""
    manifest_path(case_dir) -> String

Location of the frozen-contract manifest inside `case_dir`.
"""
manifest_path(case_dir::AbstractString) = joinpath(case_dir, "case_manifest.json")

"""
    manifest_json(manifest) -> String

Serialize `manifest` as JSON with every object's keys in sorted order.

Written out rather than delegated to `JSON.print` because the manifest is a
CHECKED artifact: two runs, and the two packages, must produce byte-identical
files, and dictionary iteration order is not a stable contract. Sorting the keys
here makes the bytes a function of the content alone.
"""
manifest_json(manifest) = sprint(io -> _write_json(io, manifest, 0))

"""
    _write_json(io, value, level) -> Nothing

Recursive pretty-printer with two-space indentation and sorted object keys.
Scalars are delegated to `JSON.json` so escaping and number formatting stay
consistent with the parser.
"""
function _write_json(io::IO, value, level::Int)
    pad = " " ^ (2 * level)
    inner = " " ^ (2 * (level + 1))
    if value isa AbstractDict
        isempty(value) && return print(io, "{}")
        ks = sort(collect(keys(value)); by = string)
        print(io, "{\n")
        for (i, k) in enumerate(ks)
            print(io, inner, JSON.json(string(k)), ": ")
            _write_json(io, value[k], level + 1)
            print(io, i == length(ks) ? "\n" : ",\n")
        end
        print(io, pad, "}")
    elseif value isa AbstractVector
        isempty(value) && return print(io, "[]")
        print(io, "[\n")
        for (i, v) in enumerate(value)
            print(io, inner)
            _write_json(io, v, level + 1)
            print(io, i == length(value) ? "\n" : ",\n")
        end
        print(io, pad, "]")
    else
        print(io, JSON.json(value))
    end
    return nothing
end

"""
    write_manifest(case_dir, manifest) -> String

Write `manifest` as sorted, indented JSON so the file is byte-reproducible
across runs and across the two packages. Returns the path written.
"""
function write_manifest(case_dir::AbstractString, manifest)
    path = manifest_path(case_dir)
    open(path, "w") do io
        print(io, manifest_json(manifest))
        write(io, '\n')
    end
    return path
end

"""
    verify(case_dir) -> Dict

Full verification: rebuild the contract from the committed bytes and require the
stored manifest to be identical to it.

# Throws
- `ErrorException` if the manifest is missing or if any field differs, naming
  the differing key path.
"""
function verify(case_dir::AbstractString)
    expected = build_manifest(case_dir)
    path = manifest_path(case_dir)
    isfile(path) || error("missing frozen-contract manifest: $path (run this script with no arguments to write it)")

    # Primary check is on BYTES: the serializer is deterministic, so the stored
    # file must equal what the committed inputs serialize to. Anything else is a
    # difference, including formatting drift.
    rendered = manifest_json(expected) * "\n"
    read(path, String) == rendered && return expected

    # Bytes differ: parse both and report every differing key path, so the
    # failure names WHAT changed rather than only that something did.
    differences = String[]
    compare_manifest!(differences, "", JSON.parsefile(path), JSON.parse(rendered))
    isempty(differences) && push!(differences, "(values agree; only formatting differs)")
    return error(
        "case manifest does not describe the committed bytes:\n  " *
        join(differences, "\n  "),
    )
end

"""
    _is_json_object(value) -> Bool

Whether `value` behaves like a JSON object: keyed, and not a string or array.
Duck-typed because a JSON parser may return its own dictionary-like type rather
than an `AbstractDict`.
"""
_is_json_object(value) =
    !(value isa AbstractString) && !(value isa AbstractVector) &&
    applicable(keys, value) && applicable(getindex, value, "")

"""
    compare_manifest!(differences, prefix, stored, expected) -> Nothing

Depth-first structural comparison that appends a human-readable `key: stored vs
expected` line to `differences` for every mismatch instead of stopping at the
first one.
"""
function compare_manifest!(differences, prefix, stored, expected)
    if _is_json_object(expected) && _is_json_object(stored)
        for key in union(keys(expected), keys(stored))
            path = isempty(prefix) ? String(key) : "$prefix.$key"
            if !haskey(stored, key)
                push!(differences, "$path: missing from manifest")
            elseif !haskey(expected, key)
                push!(differences, "$path: unexpected key in manifest")
            else
                compare_manifest!(differences, path, stored[key], expected[key])
            end
        end
    elseif expected isa AbstractVector && stored isa AbstractVector && !(expected isa AbstractString)
        length(expected) == length(stored) ||
            return push!(differences, "$prefix: length $(length(stored)) vs $(length(expected))")
        for i in eachindex(expected)
            compare_manifest!(differences, "$prefix[$i]", stored[i], expected[i])
        end
    elseif stored != expected
        push!(differences, "$prefix: $(repr(stored)) vs expected $(repr(expected))")
    end
    return nothing
end

"""
    mirror_case!(case_dir, other_case_dir) -> Vector{String}

Copy the frozen inputs, the reference exports and the manifest from `case_dir`
into `other_case_dir`, then assert byte-identity. Returns the file names copied.

This is the only supported way to synchronize the two engines' copies of the
case; it never regenerates anything.
"""
function mirror_case!(case_dir::AbstractString, other_case_dir::AbstractString)
    mkpath(other_case_dir)
    names = vcat(
        collect(keys(INPUT_HASHES)),
        [formulation * ".mof.json" for formulation in FORMULATIONS],
        ["case_manifest.json"],
    )
    for name in names
        source = joinpath(case_dir, name)
        isfile(source) || error("cannot mirror missing file: $source")
        cp(source, joinpath(other_case_dir, name); force = true)
        sha256_file(source) == sha256_file(joinpath(other_case_dir, name)) ||
            error("mirrored copy of $name is not byte-identical")
    end
    return names
end

export INPUT_HASHES, FORBIDDEN_CASE_FILES, ACTIVE_LOAD_FACTOR, REACTIVE_LOAD_FACTOR,
       DEMAND_IS_DETERMINISTIC, ACTIVE_DEFICIT_COST, REACTIVE_BALANCE,
       STAGE_HOURS, HYDRO_CONVERSION_K, REPORTING_STAGES, LOOKAHEAD_STAGES,
       TOTAL_STAGES, INFLOW_PROTOCOL_SEED, PROTOCOL_STAGES, PROTOCOL_SCENARIOS,
       TSDDR_FORMULATION, TSDDR_TARGET_MODE, TSDDR_TARGET_ACTIVATION,
       SDDP_BACKWARD_FORMULATION, SDDP_FORWARD_FORMULATION, FORMULATIONS,
       GENERATED_MODEL_NOTE,
       EMPTY_VOLUME_TOL, sha256_file, initial_state, scale_main_loads!,
       verify_inputs,
       inflow_scenario_count, protocol_indices, protocol_digest,
       verify_generated_model, build_manifest, manifest_path, manifest_json,
       write_manifest,
       verify, mirror_case!

end # module

using .HydroCanonicalCase
using JSON

function argument_value(prefix)
    index = findfirst(value -> startswith(value, prefix), ARGS)
    return isnothing(index) ? nothing : split(ARGS[index], '='; limit = 2)[2]
end

function main()
    case_dir = joinpath(@__DIR__, "bolivia")

    if "--verify" in ARGS
        manifest = HydroCanonicalCase.verify(case_dir)
        println(JSON.json(Dict(
            "status" => "verified",
            "case_dir" => case_dir,
            "manifest_sha256" => sha256_file(HydroCanonicalCase.manifest_path(case_dir)),
            "protocol_indices_sha256" => manifest["protocol"]["indices_sha256"],
        )))
        return
    end

    manifest = HydroCanonicalCase.build_manifest(case_dir)
    path = HydroCanonicalCase.write_manifest(case_dir, manifest)
    HydroCanonicalCase.verify(case_dir)

    mirrored = String[]
    exa_root = argument_value("--exa-root=")
    if exa_root !== nothing
        mirrored = HydroCanonicalCase.mirror_case!(
            case_dir,
            joinpath(exa_root, "examples", "HydroPowerModels", "bolivia"),
        )
    end

    println(JSON.json(Dict(
        "status" => "written",
        "manifest" => path,
        "manifest_sha256" => sha256_file(path),
        "protocol_indices_sha256" => manifest["protocol"]["indices_sha256"],
        "mirrored" => mirrored,
    )))
end

(abspath(PROGRAM_FILE) == @__FILE__) && main()
