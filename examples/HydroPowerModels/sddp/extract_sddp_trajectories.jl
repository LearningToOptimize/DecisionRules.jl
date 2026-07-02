# Extract per-stage SDDP trajectory data for cross-validation with JuMP subproblems.
#
# Simulates a small number of scenarios under the trained SDDP policy, then saves
# per-stage reservoir states, inflows, thermal generation, and stage costs to CSV.
# The companion script `validate_sddp_vs_jump.jl` replays these trajectories through
# the DecisionRules JuMP subproblems and compares stage-by-stage costs.
#
# Runs in the sddp/ project environment.
using MadNLP
using HydroPowerModels
using JuMP
using PowerModels
using Statistics
using SDDP: SDDP
using CSV, DataFrames
using DelimitedFiles
using Random

const CASE = "bolivia"
const SDDP_DIR = dirname(@__FILE__)
const HYDRO_DIR = dirname(SDDP_DIR)
const CASE_DIR = joinpath(HYDRO_DIR, CASE)
const RM_STAGES = 30
const REPORT_STAGES = parse(Int, get(ENV, "DR_VALIDATE_STAGES", "96"))
const NUM_STAGES = REPORT_STAGES + RM_STAGES
const FORMULATION = ACPPowerModel
const FORMULATION_B = SOCWRConicPowerModel
const NUM_SCENARIOS = parse(Int, get(ENV, "DR_VALIDATE_SCENARIOS", "4"))
const SEED = parse(Int, get(ENV, "DR_VALIDATE_SEED", "42"))

println("=" ^ 60)
println("Extract SDDP trajectories for cross-validation")
println("  Stages:     $REPORT_STAGES (of $NUM_STAGES total)")
println("  Scenarios:  $NUM_SCENARIOS")
println("  Seed:       $SEED")
println("=" ^ 60)

# ── Build SDDP model and load cuts ────────────────────────────────────────
alldata = HydroPowerModels.parse_folder(CASE_DIR)
for load in values(alldata[1]["powersystem"]["load"])
    load["qd"] = load["qd"] * 0.6
    load["pd"] = load["pd"] * 0.6
end

params = create_param(;
    stages=NUM_STAGES,
    model_constructor_grid=FORMULATION,
    post_method=PowerModels.build_opf,
    optimizer=() -> MadNLP.Optimizer(; print_level=0),
)

m = hydro_thermal_operation(alldata, params)

cuts_file = joinpath(
    CASE_DIR, string(FORMULATION),
    string(FORMULATION_B) * "-" * string(FORMULATION) * ".cuts.json",
)
SDDP.read_cuts_from_file(m.forward_graph, cuts_file)
println("Loaded cuts: $cuts_file")

# ── Simulate ───────────────────────────────────────────────────────────────
Random.seed!(SEED)
results = HydroPowerModels.simulate(m, NUM_SCENARIOS)

nhyd = alldata[1]["hydro"]["nHyd"]
hydro_data = alldata[1]["hydro"]
n_inflow_rows = hydro_data["size_inflow"][1]
num_gen = length(results[:simulations][1][1][:powersystem]["solution"]["gen"])
hydro_idx = HydroPowerModels.idx_hydro(results[:data][1])
baseMVA_val = alldata[1]["powersystem"]["baseMVA"]

function cidx(i, n)
    return mod(i, n) == 0 ? n : mod(i, n)
end

# ── Extract per-stage data ─────────────────────────────────────────────────
rows = NamedTuple[]
for s in 1:NUM_SCENARIOS
    for t in 1:REPORT_STAGES
        stage = results[:simulations][s][t]
        noise_term = stage[:noise_term]
        res = stage[:reservoirs]

        # Reservoir states
        res_in = [Float64(res[:reservoir][r].in) for r in 1:nhyd]
        res_out = [Float64(res[:reservoir][r].out) for r in 1:nhyd]
        outflow = [Float64(res[:outflow][r]) for r in 1:nhyd]
        spill = [Float64(res[:spill][r]) for r in 1:nhyd]

        # Inflows: read from the same data that SDDP used
        row_idx = cidx(t, n_inflow_rows)
        inflows = [Float64(hydro_data["Hydrogenerators"][r]["inflow"][row_idx, noise_term]) for r in 1:nhyd]

        # Generator dispatch
        sol = stage[:powersystem]["solution"]
        num_gen = length(sol["gen"])
        pg_vals = [Float64(sol["gen"]["$g"]["pg"]) for g in 1:num_gen]

        row = (
            scenario = s,
            stage = t,
            noise_term = noise_term,
            stage_objective = Float64(stage[:stage_objective]),
        )

        # Add per-hydro fields
        for r in 1:nhyd
            row = merge(row, NamedTuple{(
                Symbol("res_in_$r"), Symbol("res_out_$r"),
                Symbol("inflow_$r"), Symbol("outflow_$r"), Symbol("spill_$r"),
            )}((res_in[r], res_out[r], inflows[r], outflow[r], spill[r])))
        end

        # Add per-gen pg fields
        for g in 1:num_gen
            row = merge(row, NamedTuple{(Symbol("pg_$g"),)}((pg_vals[g],)))
        end

        push!(rows, row)
    end
end

# ── Save ───────────────────────────────────────────────────────────────────
out_file = joinpath(CASE_DIR, string(FORMULATION), "sddp_trajectories_validate.csv")
CSV.write(out_file, DataFrame(rows))
println("Saved: $out_file  ($(length(rows)) rows)")

# Summary
for s in 1:NUM_SCENARIOS
    total = sum(r.stage_objective for r in rows if r.scenario == s)
    println("  Scenario $s: total cost = $(round(total; digits=1))")
end

# Also save initial state for reference
initial_volumes = [Float64(hydro_data["Hydrogenerators"][r]["initial_volume"]) for r in 1:nhyd]
println("\nInitial volumes: $initial_volumes")

load_data = alldata[1]["powersystem"]["load"]
println("baseMVA: $baseMVA_val")
println("Number of loads: $(length(load_data))")
println("\nHydro grid indices: $hydro_idx")
println("nHyd: $nhyd, num_gen: $num_gen")
println("Inflow rows: $n_inflow_rows")

# Save metadata
meta_file = joinpath(CASE_DIR, string(FORMULATION), "sddp_validate_meta.csv")
meta = DataFrame(
    key = ["baseMVA", "nhyd", "n_inflow_rows", "num_gen", "load_scaler",
           [string("initial_vol_", r) for r in 1:nhyd]...,
           [string("hydro_grid_idx_", r) for r in 1:nhyd]...],
    value = [Float64(baseMVA_val), Float64(nhyd), Float64(n_inflow_rows), Float64(num_gen), 0.6,
             initial_volumes...,
             [Float64(hydro_idx[r]) for r in 1:nhyd]...],
)
CSV.write(meta_file, meta)
println("Saved metadata: $meta_file")

# Save per-bus demand (after 0.6 scaling) for comparison with MOF
demand_file = joinpath(CASE_DIR, string(FORMULATION), "sddp_validate_demands.csv")
demand_rows = NamedTuple[]
for (bus_key, ld) in load_data
    push!(demand_rows, (
        bus = bus_key,
        pd = Float64(ld["pd"]),
        qd = Float64(ld["qd"]),
        status = get(ld, "status", 1),
    ))
end
CSV.write(demand_file, DataFrame(demand_rows))
println("Saved demands (post-scaling): $demand_file")
