# Paired SDDP simulation using pre-sampled scenario indices via SDDP.Historical.
#
# Reads scenario indices from paired_scenario_indices.csv (the same file used by
# eval_paired_tsddr.jl) and simulates the trained SDDP policy under those exact
# inflow realizations using the Historical sampling scheme.
#
# Usage:
#   julia --project -t auto eval_paired_sddp.jl
using MadNLP
using HydroPowerModels
using JuMP
using PowerModels
using Statistics
using SDDP: SDDP
using DelimitedFiles
using CSV, DataFrames

const CASE = "bolivia"
const SDDP_DIR = dirname(@__FILE__)
const HYDRO_DIR = dirname(SDDP_DIR)
const CASE_DIR = joinpath(HYDRO_DIR, CASE)
const RM_STAGES = 30
const REPORT_STAGES = 96
const NUM_STAGES = REPORT_STAGES + RM_STAGES
const FORMULATION = ACPPowerModel
const FORMULATION_B = SOCWRConicPowerModel

# ── Load pre-sampled scenario indices ──────────────────────────────────────
indices_file = joinpath(HYDRO_DIR, CASE, "paired_scenario_indices.csv")
all_indices = Int.(readdlm(indices_file, ','))
num_scenarios = size(all_indices, 2)
@assert size(all_indices, 1) >= NUM_STAGES "Need $NUM_STAGES rows, got $(size(all_indices, 1))"
println("Loaded scenario indices: $(size(all_indices))")
println("Evaluating $num_scenarios scenarios, $REPORT_STAGES reported stages (of $NUM_STAGES total)")

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
    CASE_DIR,
    string(FORMULATION),
    string(FORMULATION_B) * "-" * string(FORMULATION) * ".cuts.json",
)
SDDP.read_cuts_from_file(m.forward_graph, cuts_file)
println("Loaded cuts: $cuts_file")

# ── Build SDDP.Historical sampling scheme ──────────────────────────────────
# Each scenario is a vector of (node, noise_term) pairs.
# node = stage index, noise_term = scenario column ω ∈ 1:nCen
historical_scenarios = [
    [(t, all_indices[t, s]) for t in 1:NUM_STAGES]
    for s in 1:num_scenarios
]

sampling_scheme = SDDP.Historical(historical_scenarios)

# ── Simulate ───────────────────────────────────────────────────────────────
println("\nSimulating $num_scenarios scenarios with SDDP.Historical...")
results = HydroPowerModels.simulate(
    m, num_scenarios;
    sampling_scheme=sampling_scheme,
)

# Verify noise terms match our indices
for s in 1:min(3, num_scenarios), t in 1:min(5, REPORT_STAGES)
    recorded_ω = results[:simulations][s][t][:noise_term]
    expected_ω = all_indices[t, s]
    if recorded_ω != expected_ω
        error("Mismatch at scenario $s, stage $t: got ω=$recorded_ω, expected $expected_ω")
    end
end
println("Noise term verification passed (spot-checked)")

# ── Extract results ────────────────────────────────────────────────────────
nhyd = alldata[1]["hydro"]["nHyd"]
volume_to_mw(volume; k=0.0036) = volume / k

objective_values = [
    sum(results[:simulations][i][t][:stage_objective] for t in 1:REPORT_STAGES)
    for i in 1:num_scenarios
]

hydro_vol = [
    mean(
        sum(
            volume_to_mw(results[:simulations][i][t][:reservoirs][:reservoir][j].out) for
            j in 1:nhyd
        ) for i in 1:num_scenarios
    ) for t in 1:REPORT_STAGES
]

num_gen = length(results[:simulations][1][1][:powersystem]["solution"]["gen"])
hydro_idx = HydroPowerModels.idx_hydro(results[:data][1])
thermal_gen = [
    mean(
        sum(
            results[:simulations][i][t][:powersystem]["solution"]["gen"]["$j"]["pg"] *
            results[:data][1]["powersystem"]["baseMVA"] for
            j in 1:num_gen if !(j in hydro_idx)
        ) for i in 1:num_scenarios
    ) for t in 1:REPORT_STAGES
]

# ── Report ─────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 60)
println("Results: Paired SDDP ($REPORT_STAGES stages, $num_scenarios scenarios)")
println("=" ^ 60)
println("  Mean cost:   $(round(mean(objective_values); digits=1))")
println("  Std:         $(round(std(objective_values); digits=1))")
println("  Min:         $(round(minimum(objective_values); digits=1))")
println("  Max:         $(round(maximum(objective_values); digits=1))")
println("  Median:      $(round(median(objective_values); digits=1))")
println("=" ^ 60)

# ── Save results ───────────────────────────────────────────────────────────
out_dir = joinpath(CASE_DIR, string(FORMULATION))

const COL_NAME = "SDDP-SOC (paired)"
costs_file = joinpath(out_dir, "paired_costs.csv")
if isfile(costs_file)
    df = CSV.read(costs_file, DataFrame)
    df[!, COL_NAME] = objective_values
else
    df = DataFrame(Symbol(COL_NAME) => objective_values)
end
CSV.write(costs_file, df)
println("Updated: $costs_file")

vol_file = joinpath(out_dir, "paired_MeanVolume.csv")
if isfile(vol_file)
    df_vol = CSV.read(vol_file, DataFrame; header=true)
    df_vol[!, COL_NAME] = hydro_vol
else
    df_vol = DataFrame(Symbol(COL_NAME) => hydro_vol)
end
CSV.write(vol_file, df_vol)
println("Updated: $vol_file")

gen_file = joinpath(out_dir, "paired_MeanGeneration.csv")
if isfile(gen_file)
    df_gen = CSV.read(gen_file, DataFrame; header=true)
    df_gen[!, COL_NAME] = thermal_gen
else
    df_gen = DataFrame(Symbol(COL_NAME) => thermal_gen)
end
CSV.write(gen_file, df_gen)
println("Updated: $gen_file")

# Per-scenario paired costs for direct comparison
println("\nPer-scenario cost comparison (first 10):")
if isfile(costs_file)
    df_all = CSV.read(costs_file, DataFrame)
    if hasproperty(df_all, Symbol("TS-DDR (strict, paired)"))
        tsddr_costs = df_all[!, "TS-DDR (strict, paired)"]
        sddp_costs = df_all[!, COL_NAME]
        diffs = sddp_costs .- tsddr_costs
        println("  Scenario | SDDP      | TS-DDR    | Diff")
        for s in 1:min(10, num_scenarios)
            println("  $(lpad(s, 7)) | $(lpad(round(sddp_costs[s]; digits=1), 9)) | $(lpad(round(tsddr_costs[s]; digits=1), 9)) | $(round(diffs[s]; digits=1))")
        end
        println("\n  Mean diff (SDDP - TS-DDR): $(round(mean(diffs); digits=1))")
        println("  Std diff:                  $(round(std(diffs); digits=1))")
        println("  Paired t-test p-value:     (compute externally)")
    end
end
