# Paired SDDP simulation on the seeded paired protocol via SDDP.Historical.
#
# Scenario indices are generated from PAIRED_SCENARIO_SEED (identical to
# eval_paired_tsddr.jl's protocol), so the SDDP policy is simulated under the
# exact inflow realizations every TS-DDR evaluation uses.
#
# Usage:
#   julia --project -t auto eval_paired_sddp.jl
using MadNLP
using StableRNGs
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

# ── Paired scenario indices (seeded protocol) ──────────────────────────────
# Identical generation to load_hydropowermodels.jl's paired_scenario_indices:
# entry [t, s] is uniform on 1:nCen from StableRNG(PAIRED_SCENARIO_SEED), so
# this script and every TS-DDR evaluation realize the same inflow at the same
# stage of the same paired scenario — with no shared data file. nCen is
# derived from the inflow data (columns ÷ hydro units), never hardcoded.
const PAIRED_SCENARIO_SEED = 20260706
# Fixed generated shape shared by ALL consumers (see load_hydropowermodels.jl:
# arrays of different shapes consume the RNG stream differently, so every
# script must generate exactly this shape and slice what it needs).
const PAIRED_NUM_STAGES = 126
const N_HYDRO = 11
@assert NUM_STAGES == PAIRED_NUM_STAGES "SDDP horizon must equal the paired protocol shape"
num_scenarios = parse(Int, get(ENV, "DR_NUM_SCENARIOS", "500"))
nCen = div(size(readdlm(joinpath(HYDRO_DIR, CASE, "inflows.csv"), ','), 2), N_HYDRO)
# ALWAYS generate the full protocol matrix (fixed shape — see the note above),
# then optionally simulate only a shard of its columns so the 500 scenarios
# can run in parallel across nodes (DR_SCENARIO_FIRST/LAST, 1-based inclusive).
all_indices = rand(StableRNG(PAIRED_SCENARIO_SEED), 1:nCen, PAIRED_NUM_STAGES, num_scenarios)
scen_first = parse(Int, get(ENV, "DR_SCENARIO_FIRST", "1"))
scen_last = parse(Int, get(ENV, "DR_SCENARIO_LAST", string(num_scenarios)))
@assert 1 <= scen_first <= scen_last <= num_scenarios
scen_range = scen_first:scen_last
println("Paired protocol: seed=$PAIRED_SCENARIO_SEED, $(PAIRED_NUM_STAGES)×$(num_scenarios), nCen=$nCen")
println("Evaluating scenarios $scen_first:$scen_last, $REPORT_STAGES reported stages (of $NUM_STAGES total)")

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
    # Hardened solver settings for the 500-scenario simulation: with bare
    # defaults one nonconvex ACP node failed to converge on one seeded draw
    # (SDDP aborts the whole simulation on any node failure). Larger
    # iteration budget + explicit tolerance make every node solvable; the
    # cuts themselves are unaffected (they were trained separately).
    optimizer=() -> MadNLP.Optimizer(;
        print_level=0,
        max_iter=9000,
        tol=1e-6,
    ),
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
    for s in scen_range
]
n_sim = length(scen_range)

sampling_scheme = SDDP.Historical(historical_scenarios)

# ── Simulate ───────────────────────────────────────────────────────────────
println("\nSimulating $n_sim scenarios with SDDP.Historical...")
results = HydroPowerModels.simulate(
    m, n_sim;
    sampling_scheme=sampling_scheme,
)

# Verify noise terms match our indices
for (si, s) in enumerate(first(scen_range, min(3, n_sim))), t in 1:min(5, REPORT_STAGES)
    recorded_ω = results[:simulations][si][t][:noise_term]
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
    for i in 1:n_sim
]

hydro_vol = [
    mean(
        sum(
            volume_to_mw(results[:simulations][i][t][:reservoirs][:reservoir][j].out) for
            j in 1:nhyd
        ) for i in 1:n_sim
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
        ) for i in 1:n_sim
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

# Shard mode: emit only this shard's per-scenario costs (merged afterwards by
# merge_sddp_shards.jl); the full-run outputs below are skipped.
if n_sim != num_scenarios
    shard_file = joinpath(out_dir, "sddp_shard_$(scen_first)_$(scen_last).csv")
    CSV.write(shard_file, DataFrame(scenario = collect(scen_range), cost = objective_values))
    println("Shard written: $shard_file")
    exit(0)
end

# Optional output tag (mirrors eval_paired_tsddr.jl): when DR_OUTPUT_TAG is
# set, every output filename gets an _<tag> suffix so re-evaluations at a
# different scenario count never overwrite existing result files. Note that a
# tagged run starts fresh tagged files; the merge-with-existing-columns logic
# only applies within the same tag.
tag_suffix = let tag = get(ENV, "DR_OUTPUT_TAG", "")
    isempty(tag) ? "" : "_$(tag)"
end
isempty(tag_suffix) || println("Output tag suffix: $tag_suffix")

const COL_NAME = "SDDP-SOC (paired)"
costs_file = joinpath(out_dir, "paired_costs$(tag_suffix).csv")
if isfile(costs_file)
    df = CSV.read(costs_file, DataFrame)
    df[!, COL_NAME] = objective_values
else
    df = DataFrame(Symbol(COL_NAME) => objective_values)
end
CSV.write(costs_file, df)
println("Updated: $costs_file")

vol_file = joinpath(out_dir, "paired_MeanVolume$(tag_suffix).csv")
if isfile(vol_file)
    df_vol = CSV.read(vol_file, DataFrame; header=true)
    df_vol[!, COL_NAME] = hydro_vol
else
    df_vol = DataFrame(Symbol(COL_NAME) => hydro_vol)
end
CSV.write(vol_file, df_vol)
println("Updated: $vol_file")

gen_file = joinpath(out_dir, "paired_MeanGeneration$(tag_suffix).csv")
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
