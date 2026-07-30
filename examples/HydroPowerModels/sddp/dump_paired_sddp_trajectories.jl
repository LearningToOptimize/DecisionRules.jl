# Dump SDDP per-scenario trajectories on the seeded paired protocol.
#
# This is the data leg of the imitation-capacity test:
#   1. Simulate the saved SDDP cuts on the exact paired-500 inflow matrix.
#   2. Save per-stage teacher-forcing data:
#        input  = (inflow_t, res_in_t)
#        target = res_out_t
#   3. Assert the resulting paired mean is the known SDDP score, so the dumped
#      decisions provably come from the same policy used for the 303,665 baseline.
#
# Runs in examples/HydroPowerModels/sddp Project.toml.
#
# Usage:
#   julia --project=. dump_paired_sddp_trajectories.jl
#
# Useful knobs:
#   DR_NUM_SCENARIOS=500
#   DR_OUTPUT_TAG=sddp_paired500
#   DR_EXPECTED_MEAN=303665.4
#   DR_EXPECTED_TOL=5.0

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

const PAIRED_SCENARIO_SEED = 20260706
const PAIRED_NUM_STAGES = 126
const N_HYDRO = 11
@assert NUM_STAGES == PAIRED_NUM_STAGES

const NUM_SCENARIOS = parse(Int, get(ENV, "DR_NUM_SCENARIOS", "500"))
const EXPECTED_MEAN = parse(Float64, get(ENV, "DR_EXPECTED_MEAN", "303665.4"))
const EXPECTED_TOL = parse(Float64, get(ENV, "DR_EXPECTED_TOL", "5.0"))
const TAG = strip(get(ENV, "DR_OUTPUT_TAG", "sddp_paired500"))
const TAG_SUFFIX = isempty(TAG) ? "" : "_$(TAG)"

println("=" ^ 72)
println("Dump paired SDDP trajectories for imitation-capacity test")
println("  stages:       $REPORT_STAGES reported / $NUM_STAGES simulated")
println("  scenarios:    $NUM_SCENARIOS")
println("  expected mean $(round(EXPECTED_MEAN; digits=1)) ± $EXPECTED_TOL")
println("=" ^ 72)

# ── Paired protocol ───────────────────────────────────────────────────────────

nCen = div(size(readdlm(joinpath(CASE_DIR, "inflows.csv"), ','), 2), N_HYDRO)
scenario_indices = rand(
    StableRNG(PAIRED_SCENARIO_SEED),
    1:nCen,
    PAIRED_NUM_STAGES,
    NUM_SCENARIOS,
)

# ── Build SDDP model and load cuts ────────────────────────────────────────────

alldata = HydroPowerModels.parse_folder(CASE_DIR)
for data in alldata
    for load in values(data["powersystem"]["load"])
        load["pd"] *= 0.6
        load["qd"] *= 0.6
    end
    data["powersystem"]["cost_deficit"] = 6000.0 / data["powersystem"]["baseMVA"]
end

stage_hours = Int(get(alldata[1]["hydro"], "stage_hours", 1))  # K = 0.0036·stage_hours
params = create_param(;
    stages = NUM_STAGES,
    stage_hours = stage_hours,
    model_constructor_grid = FORMULATION,
    post_method = PowerModels.build_opf,
    optimizer = () -> MadNLP.Optimizer(;
        print_level = 0,
        max_iter = 9000,
        tol = 1e-6,
    ),
)

model = hydro_thermal_operation(alldata, params)
cuts_file = joinpath(
    CASE_DIR,
    string(FORMULATION),
    string(FORMULATION_B) * "-" * string(FORMULATION) * ".cuts.json",
)
SDDP.read_cuts_from_file(model.forward_graph, cuts_file)
println("Loaded cuts: $cuts_file")

historical_scenarios = [
    [(t, scenario_indices[t, s]) for t in 1:NUM_STAGES]
    for s in 1:NUM_SCENARIOS
]

println("Simulating paired SDDP policy...")
results = HydroPowerModels.simulate(
    model,
    NUM_SCENARIOS;
    sampling_scheme = SDDP.Historical(historical_scenarios),
)

# Spot-check the Historical path is exactly the paired matrix we generated.
for s in 1:min(3, NUM_SCENARIOS), t in 1:min(5, REPORT_STAGES)
    got = results[:simulations][s][t][:noise_term]
    expected = scenario_indices[t, s]
    got == expected || error("Noise mismatch at scenario=$s stage=$t: got $got expected $expected")
end
println("Noise term verification passed")

# ── Extract teacher-forcing rows ──────────────────────────────────────────────

nhyd = alldata[1]["hydro"]["nHyd"]
@assert nhyd == N_HYDRO
hydro_data = alldata[1]["hydro"]
n_inflow_rows = hydro_data["size_inflow"][1]
cidx(i, n) = mod(i - 1, n) + 1

rows = NamedTuple[]
costs = zeros(Float64, NUM_SCENARIOS)

for s in 1:NUM_SCENARIOS
    for t in 1:REPORT_STAGES
        stage = results[:simulations][s][t]
        noise_term = Int(stage[:noise_term])
        reservoirs = stage[:reservoirs]
        costs[s] += Float64(stage[:stage_objective])

        row = (
            scenario = s,
            stage = t,
            noise_term = noise_term,
            stage_objective = Float64(stage[:stage_objective]),
        )

        raw_row = cidx(t, n_inflow_rows)
        for r in 1:nhyd
            inflow = Float64(hydro_data["Hydrogenerators"][r]["inflow"][raw_row, noise_term])
            row = merge(row, NamedTuple{(
                Symbol("inflow_$r"),
                Symbol("res_in_$r"),
                Symbol("res_out_$r"),
                Symbol("outflow_$r"),
                Symbol("spill_$r"),
            )}((
                inflow,
                Float64(reservoirs[:reservoir][r].in),
                Float64(reservoirs[:reservoir][r].out),
                Float64(reservoirs[:outflow][r]),
                Float64(reservoirs[:spill][r]),
            )))
        end
        push!(rows, row)
    end
end

mean_cost = mean(costs)
std_cost = std(costs)
println("SDDP paired mean = $(round(mean_cost; digits=1)), std = $(round(std_cost; digits=1))")
abs(mean_cost - EXPECTED_MEAN) <= EXPECTED_TOL ||
    error("Dumped SDDP mean $(mean_cost) is not within $EXPECTED_TOL of expected $EXPECTED_MEAN")

# ── Save outputs ──────────────────────────────────────────────────────────────

out_dir = joinpath(CASE_DIR, string(FORMULATION), "results")
mkpath(out_dir)

traj_file = joinpath(out_dir, "sddp_paired_trajectories$(TAG_SUFFIX).csv")
CSV.write(traj_file, DataFrame(rows))
println("Saved trajectories: $traj_file")

cost_file = joinpath(out_dir, "sddp_paired_costs$(TAG_SUFFIX).csv")
CSV.write(cost_file, DataFrame(scenario = 1:NUM_SCENARIOS, cost = costs))
println("Saved costs: $cost_file")

meta_file = joinpath(out_dir, "sddp_paired_trajectories_meta$(TAG_SUFFIX).csv")
meta = DataFrame(
    key = [
        "paired_seed",
        "num_scenarios",
        "report_stages",
        "num_stages",
        "nhyd",
        "n_inflow_rows",
        "ncen",
        "mean_cost",
        "std_cost",
        "expected_mean",
        "expected_tol",
        "cuts_file",
    ],
    value = string.([
        PAIRED_SCENARIO_SEED,
        NUM_SCENARIOS,
        REPORT_STAGES,
        NUM_STAGES,
        nhyd,
        n_inflow_rows,
        nCen,
        mean_cost,
        std_cost,
        EXPECTED_MEAN,
        EXPECTED_TOL,
        cuts_file,
    ]),
)
CSV.write(meta_file, meta)
println("Saved metadata: $meta_file")

idx_file = joinpath(out_dir, "sddp_paired_scenario_indices$(TAG_SUFFIX).csv")
idx_rows = (
    (; scenario = s, stage = t, noise_term = scenario_indices[t, s])
    for s in 1:NUM_SCENARIOS for t in 1:NUM_STAGES
)
CSV.write(idx_file, DataFrame(collect(idx_rows)))
println("Saved scenario indices: $idx_file")
