#!/usr/bin/env julia

using CSV
using DataFrames
using JSON
using SHA
using StableRNGs

include(joinpath(@__DIR__, "generate_canonical_case_artifacts.jl"))
using .HydroCanonicalCase

const CASE_DIR = joinpath(@__DIR__, "bolivia")
const OUTPUT = joinpath(CASE_DIR, "joint_protocol_500.csv")

function inflow_scenario_count(case_dir)
    first_row = split(first(eachline(joinpath(case_dir, "inflows.csv"))), ',')
    nhyd = length(JSON.parsefile(joinpath(case_dir, "hydro.json"))["Hydrogenerators"])
    length(first_row) % nhyd == 0 || error("inflow columns are not divisible by nHyd")
    return length(first_row) ÷ nhyd
end

function generate_indices(ncen)
    inflow = rand(
        StableRNG(INFLOW_PROTOCOL_SEED),
        1:ncen,
        TOTAL_STAGES,
        PROTOCOL_SCENARIOS,
    )
    demand = reduce(
        hcat,
        [
            rand(
                StableRNG(DEMAND_PROTOCOL_SEED + scenario),
                1:length(DEMAND_ATOMS),
                TOTAL_STAGES,
            )
            for scenario in 1:PROTOCOL_SCENARIOS
        ],
    )
    return inflow, demand
end

function protocol_table(inflow, demand)
    rows = TOTAL_STAGES * PROTOCOL_SCENARIOS
    stage = Vector{Int}(undef, rows)
    scenario = Vector{Int}(undef, rows)
    inflow_index = Vector{Int}(undef, rows)
    demand_index = Vector{Int}(undef, rows)
    k = 1
    for t in 1:TOTAL_STAGES, s in 1:PROTOCOL_SCENARIOS
        stage[k] = t
        scenario[k] = s
        inflow_index[k] = inflow[t, s]
        demand_index[k] = demand[t, s]
        k += 1
    end
    return DataFrame(; stage, scenario, inflow_index, demand_index)
end

function verify_protocol(path=OUTPUT)
    verify_inputs(CASE_DIR)
    ncen = inflow_scenario_count(CASE_DIR)
    inflow, demand = generate_indices(ncen)
    saved = CSV.read(path, DataFrame)
    expected = protocol_table(inflow, demand)
    names(saved) == names(expected) || error("joint protocol columns do not match")
    saved == expected || error("joint protocol does not reconstruct exactly")
    manifest = read_manifest(CASE_DIR)
    if haskey(manifest, "protocol")
        manifest["protocol"]["csv_sha256"] == sha256_file(path) ||
            error("joint protocol hash does not match the case manifest")
        manifest["protocol"]["source_hashes"] == INPUT_HASHES ||
            error("joint protocol source hashes do not match the canonical inputs")
    end
    return (ncen=ncen, protocol_sha256=sha256_file(path))
end

if "--verify" in ARGS
    result = verify_protocol()
    println(JSON.json(Dict(
        "status" => "verified",
        "inflow_scenarios" => result.ncen,
        "protocol_sha256" => result.protocol_sha256,
    )))
else
    counts = verify_inputs(CASE_DIR)
    ncen = inflow_scenario_count(CASE_DIR)
    inflow, demand = generate_indices(ncen)
    CSV.write(OUTPUT, protocol_table(inflow, demand))
    protocol_hash = sha256_file(OUTPUT)
    metadata = Dict(
        "schema_version" => 1,
        "layout" => "stage-major rows; stage outer, scenario inner",
        "dimensions" => Dict(
            "stages" => TOTAL_STAGES,
            "scenarios" => PROTOCOL_SCENARIOS,
            "inflow_scenarios" => ncen,
            "demand_atoms" => length(DEMAND_ATOMS),
        ),
        "seeds" => Dict(
            "inflow_matrix" => INFLOW_PROTOCOL_SEED,
            "demand_column_base" => DEMAND_PROTOCOL_SEED,
            "demand_column_rule" => "StableRNG(demand_column_base + scenario_id)",
        ),
        "demand_atom_values" => collect(DEMAND_ATOMS),
        "source_hashes" => INPUT_HASHES,
        "csv_sha256" => protocol_hash,
        "topology_counts" => Dict(string(k) => v for (k, v) in pairs(counts)),
    )
    manifest_path = joinpath(CASE_DIR, "case_manifest.json")
    manifest = read_manifest(CASE_DIR)
    manifest["protocol"] = metadata
    open(manifest_path, "w") do io
        JSON.print(io, manifest, 2)
        write(io, '\n')
    end
    result = verify_protocol()
    result.protocol_sha256 == protocol_hash || error("protocol hash changed during verification")
    println(JSON.json(Dict(
        "status" => "generated",
        "rows" => TOTAL_STAGES * PROTOCOL_SCENARIOS,
        "protocol_sha256" => protocol_hash,
        "manifest_sha256" => sha256_file(manifest_path),
    )))
end
