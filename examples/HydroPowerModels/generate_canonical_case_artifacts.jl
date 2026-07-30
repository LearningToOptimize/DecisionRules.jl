#!/usr/bin/env julia

module HydroCanonicalCase

using JSON
using SHA

const ACTIVE_LOAD_FACTOR = 0.6
const REACTIVE_LOAD_FACTOR = 0.6
const STAGE_HOURS = 168
const HYDRO_CONVERSION_K = 0.6048
const REPORTING_STAGES = 96
const LOOKAHEAD_STAGES = 30
const TOTAL_STAGES = REPORTING_STAGES + LOOKAHEAD_STAGES
const ACTIVE_DEFICIT_COST = 6000.0
const INFLOW_PROTOCOL_SEED = 20260706
const DEMAND_PROTOCOL_SEED = 20260714
const PROTOCOL_SCENARIOS = 500
const DEMAND_ATOMS = (0.9, 1.0, 1.1)
const FORMULATIONS = ("ACPPowerModel", "DCPPowerModel", "SOCWRConicPowerModel")

const INPUT_HASHES = Dict(
    "PowerModels.json" => "1ff598447957f9fc17ca570415bf5b9b5b14e1292ea3bd3163db0ad79911a782",
    "inflows.csv" => "5afb275dff3fc879e3e93b6510b81295834faad0bcd2bd1fc070a8e3e6653c77",
    "hydro.json" => "1014b90ac06afb36f136a34c66fbd12d45a81c1e700324dcda398a5438d4d98b",
    "demand_scenarios.csv" => "abeb664b606ba17b830398e8d891d4d23dd7739d5b221088ca7a30bcfab4e8df",
)

sha256_file(path::AbstractString) = bytes2hex(open(SHA.sha256, path))

function verify_inputs(case_dir::AbstractString)
    for (name, expected) in INPUT_HASHES
        path = joinpath(case_dir, name)
        isfile(path) || error("missing canonical input: $path")
        actual = sha256_file(path)
        actual == expected ||
            error("canonical input hash mismatch for $path: expected $expected, got $actual")
    end

    hydro = JSON.parsefile(joinpath(case_dir, "hydro.json"))
    Int(hydro["stage_hours"]) == STAGE_HOURS ||
        error("stage_hours must be $STAGE_HOURS")
    isapprox(0.0036 * Int(hydro["stage_hours"]), HYDRO_CONVERSION_K; atol=0, rtol=1e-14) ||
        error("hydro conversion K must be $HYDRO_CONVERSION_K")

    power = JSON.parsefile(joinpath(case_dir, "PowerModels.json"))
    Float64(power["baseMVA"]) == 100.0 || error("canonical baseMVA must be 100")
    Float64(power["cost_deficit"]) * Float64(power["baseMVA"]) == ACTIVE_DEFICIT_COST ||
        error("canonical active-deficit coefficient must be $ACTIVE_DEFICIT_COST")
    return Dict(
        "buses" => length(power["bus"]),
        "branches" => length(power["branch"]),
        "generators" => length(power["gen"]),
        "loads" => length(power["load"]),
        "hydro_units" => length(hydro["Hydrogenerators"]),
    )
end

function scale_main_loads!(alldata)
    for stage in alldata
        for load in values(stage["powersystem"]["load"])
            load["pd"] = Float64(load["pd"]) * ACTIVE_LOAD_FACTOR
            load["qd"] = Float64(load["qd"]) * REACTIVE_LOAD_FACTOR
        end
        stage["powersystem"]["cost_deficit"] =
            ACTIVE_DEFICIT_COST / Float64(stage["powersystem"]["baseMVA"])
    end
    return alldata
end

function base_manifest(case_dir::AbstractString)
    counts = verify_inputs(case_dir)
    return Dict(
        "schema_version" => 1,
        "case" => "Bolivia MAIN",
        "input_hashes" => INPUT_HASHES,
        "initial_volume_repair" => "70% of capacity for corrupted MAIN denormals",
        "active_load_factor" => ACTIVE_LOAD_FACTOR,
        "reactive_load_factor" => REACTIVE_LOAD_FACTOR,
        "stage_hours" => STAGE_HOURS,
        "hydro_conversion_K" => HYDRO_CONVERSION_K,
        "reporting_stages" => REPORTING_STAGES,
        "lookahead_stages" => LOOKAHEAD_STAGES,
        "total_stages" => TOTAL_STAGES,
        "active_deficit_cost_usd_per_pu_stage" => ACTIVE_DEFICIT_COST,
        "active_deficit_cost_derivation" => "60 USD/MWh * 100 MVA",
        "strict_targets_primary" => true,
        "reactive_balance" => "hard",
        "branch_thermal_limits" => "both ends",
        "reachable_activation" => "stretchedsigmoid_safe_upper_margin_1e-3",
        "demand_atoms" => collect(DEMAND_ATOMS),
        "demand_atom_probabilities" => fill(1 / 3, length(DEMAND_ATOMS)),
        "topology_counts" => counts,
        "mofs" => Dict{String,Any}(),
    )
end

function write_manifest(case_dir::AbstractString, manifest)
    path = joinpath(case_dir, "case_manifest.json")
    open(path, "w") do io
        JSON.print(io, manifest, 2)
        write(io, '\n')
    end
    return path
end

function read_manifest(case_dir::AbstractString)
    path = joinpath(case_dir, "case_manifest.json")
    isfile(path) || error("missing canonical case manifest: $path")
    manifest = JSON.parsefile(path)
    manifest["active_load_factor"] == ACTIVE_LOAD_FACTOR ||
        error("manifest active_load_factor must be $ACTIVE_LOAD_FACTOR")
    manifest["reactive_load_factor"] == REACTIVE_LOAD_FACTOR ||
        error("manifest reactive_load_factor must be $REACTIVE_LOAD_FACTOR")
    manifest["stage_hours"] == STAGE_HOURS ||
        error("manifest stage_hours must be $STAGE_HOURS")
    manifest["hydro_conversion_K"] == HYDRO_CONVERSION_K ||
        error("manifest hydro_conversion_K must be $HYDRO_CONVERSION_K")
    manifest["active_deficit_cost_usd_per_pu_stage"] == ACTIVE_DEFICIT_COST ||
        error("manifest active deficit cost must be $ACTIVE_DEFICIT_COST")
    return manifest
end

function objective_terms(mof)
    function_object = mof["objective"]["function"]
    return get(function_object, "terms", Any[])
end

function verify_mof(path::AbstractString, formulation::AbstractString)
    mof = JSON.parsefile(path)
    mof["objective"]["sense"] == "min" || error("$formulation objective is not Min")
    deficit_terms = [
        term for term in objective_terms(mof)
        if startswith(get(term, "variable", ""), "deficit[")
    ]
    length(deficit_terms) == 28 ||
        error("$formulation must have 28 operational active-deficit objective terms")
    all(Float64(term["coefficient"]) == ACTIVE_DEFICIT_COST for term in deficit_terms) ||
        error("$formulation active-deficit coefficient is not $ACTIVE_DEFICIT_COST")

    hydro_balance = [
        constraint for constraint in mof["constraints"]
        if startswith(get(constraint, "name", ""), "hydro_balance[")
    ]
    length(hydro_balance) == 11 || error("$formulation must have 11 hydro balances")
    for constraint in hydro_balance
        inflow_terms = [
            term for term in constraint["function"]["terms"]
            if startswith(get(term, "variable", ""), "inflow[")
        ]
        length(inflow_terms) == 1 || error("$formulation hydro balance lacks one inflow")
        abs(Float64(only(inflow_terms)["coefficient"])) == HYDRO_CONVERSION_K ||
            error("$formulation MOF has stale hydro K")
    end

    variable_names = String[get(v, "name", "") for v in mof["variables"]]
    any(startswith(name, "target_deficit") for name in variable_names) &&
        error("$formulation MOF mixes operational active deficit with target slack")

    if formulation == "ACPPowerModel"
        reactive_balances = [
            constraint for constraint in mof["constraints"]
            if any(
                startswith(get(term, "variable", ""), "0_q[")
                for term in get(constraint["function"], "terms", Any[])
            ) && get(constraint["set"], "type", "") == "EqualTo"
        ]
        length(reactive_balances) >= 28 ||
            error("ACP MOF is missing hard reactive balance equations")
        quadratic_limits = [
            constraint for constraint in mof["constraints"]
            if get(constraint["function"], "type", "") == "ScalarQuadraticFunction" &&
               get(constraint["set"], "type", "") == "LessThan"
        ]
        length(quadratic_limits) >= 62 ||
            error("ACP MOF is missing both-ended apparent-power thermal limits")
    end

    return Dict(
        "sha256" => sha256_file(path),
        "variables" => length(mof["variables"]),
        "constraints" => length(mof["constraints"]),
        "objective_sense" => mof["objective"]["sense"],
        "active_deficit_terms" => length(deficit_terms),
        "hydro_balance_terms" => length(hydro_balance),
    )
end

function finalize_manifest(case_dir::AbstractString)
    manifest = read_manifest(case_dir)
    manifest["mofs"] = Dict(
        formulation => verify_mof(
            joinpath(case_dir, formulation * ".mof.json"),
            formulation,
        )
        for formulation in FORMULATIONS
    )
    write_manifest(case_dir, manifest)
    return manifest
end

export ACTIVE_LOAD_FACTOR, REACTIVE_LOAD_FACTOR, STAGE_HOURS,
       HYDRO_CONVERSION_K, REPORTING_STAGES, LOOKAHEAD_STAGES, TOTAL_STAGES,
       ACTIVE_DEFICIT_COST, INFLOW_PROTOCOL_SEED, DEMAND_PROTOCOL_SEED,
       PROTOCOL_SCENARIOS, DEMAND_ATOMS, FORMULATIONS, INPUT_HASHES,
       sha256_file, verify_inputs, scale_main_loads!, base_manifest,
       write_manifest, read_manifest, verify_mof, finalize_manifest

end

using .HydroCanonicalCase
using JSON

function argument_value(prefix)
    arg = findfirst(value -> startswith(value, prefix), ARGS)
    return isnothing(arg) ? nothing : split(ARGS[arg], '='; limit=2)[2]
end

function main()
    case_dir = joinpath(@__DIR__, "bolivia")
    verify_inputs(case_dir)

    if "--verify-only" in ARGS
        manifest = read_manifest(case_dir)
        for formulation in FORMULATIONS
            verify_mof(joinpath(case_dir, formulation * ".mof.json"), formulation)
        end
        println(JSON.json(Dict("status" => "verified", "manifest" => manifest)))
        return
    end

    if !("--finalize-only" in ARGS)
        write_manifest(case_dir, base_manifest(case_dir))
    end
    "--prepare-only" in ARGS && return

    if !("--finalize-only" in ARGS)
        exporter = joinpath(@__DIR__, "export_subproblem_mof.jl")
        for formulation in FORMULATIONS
            run(`$(Base.julia_cmd()) $exporter bolivia $formulation`)
        end
    end

    manifest = finalize_manifest(case_dir)
    protocol_generator = joinpath(@__DIR__, "generate_joint_protocol.jl")
    run(`$(Base.julia_cmd()) $protocol_generator`)
    manifest = read_manifest(case_dir)

    exa_root = argument_value("--exa-root=")
    if exa_root !== nothing
        exa_dir = joinpath(exa_root, "examples", "HydroPowerModels")
        exa_case = joinpath(exa_dir, "bolivia")
        mkpath(exa_case)
        for name in keys(INPUT_HASHES)
            cp(joinpath(case_dir, name), joinpath(exa_case, name); force=true)
        end
        for formulation in FORMULATIONS
            name = formulation * ".mof.json"
            cp(joinpath(case_dir, name), joinpath(exa_case, name); force=true)
        end
        for name in ("case_manifest.json", "joint_protocol_500.csv")
            cp(joinpath(case_dir, name), joinpath(exa_case, name); force=true)
        end
        for name in ("generate_canonical_case_artifacts.jl", "generate_joint_protocol.jl")
            cp(joinpath(@__DIR__, name), joinpath(exa_dir, name); force=true)
        end
    end

    println(JSON.json(Dict(
        "status" => "generated",
        "manifest_sha256" => sha256_file(joinpath(case_dir, "case_manifest.json")),
        "mofs" => manifest["mofs"],
        "protocol" => manifest["protocol"],
    )))
end

(abspath(PROGRAM_FILE) == @__FILE__) && main()
