# Simulate a pre-trained SDDP policy (cuts from run_sddp_inconsistent.jl) under
# the ACP formulation and produce comparison plots/CSVs against TS-DDR baselines.
using MadNLP
using HydroPowerModels
using JuMP
using PowerModels
using Statistics
using SDDP: SDDP

using Random
seed = 1221

# Load case
case = "bolivia"
case_dir = joinpath(dirname(@__DIR__), case)
alldata = HydroPowerModels.parse_folder(case_dir);
for load in values(alldata[1]["powersystem"]["load"])
    load["qd"] = load["qd"] * 0.6
    load["pd"] = load["pd"] * 0.6
end
rm_stages = 30
num_stages = 96 + rm_stages
formulation = ACPPowerModel
formulation_b = SOCWRConicPowerModel

params = create_param(;
    stages=num_stages,
    model_constructor_grid=formulation,
    post_method=PowerModels.build_opf,
    optimizer=() -> MadNLP.Optimizer(; print_level=0),
);

m = hydro_thermal_operation(alldata, params);

# Load pre-trained cuts
SDDP.read_cuts_from_file(
    m.forward_graph,
    joinpath(
        case_dir,
        string(formulation),
        string(formulation_b)*"-"*string(formulation)*".cuts.json",
    ),
)

# Simulate
Random.seed!(seed)
num_sim = 100
results = HydroPowerModels.simulate(m, num_sim);

# Plotting
nhyd = alldata[1]["hydro"]["nHyd"]
using Plots
using CSV
using DataFrames
volume_to_mw(volume, stage_hours; k=0.0036) = volume / (k * stage_hours)

const SDDP_COL = "SDDP-SOC"
labels = ["TS-DDR"; "TS-LDR"; "SDDP-DCLL"; SDDP_COL]
colors = [:black :purple :red :orange]
markers = [:hline :+ :pixel :diamond]

const DOCS_ASSETS = joinpath(dirname(@__DIR__), "..", "..", "docs", "src", "assets")
mkpath(DOCS_ASSETS)
out_dir = joinpath(case_dir, string(formulation))

# Volume trajectory
hydro_gen = [
    mean(
        sum(
            volume_to_mw(results[:simulations][i][t][:reservoirs][:reservoir][j].out, 1) for
            j in 1:nhyd
        ) for i in 1:num_sim
    ) for t in 1:(num_stages - rm_stages)
]

savefig(
    plot(
        hydro_gen;
        legend=false,
        xlabel="Stage",
        ylabel="Volume (Hm3)",
        title="$(case)-$(formulation_b)-$(formulation)",
    ),
    joinpath(out_dir, "SDDP-$(case)-$(formulation_b)-$(formulation)-Volume.png"),
)

df = CSV.read(joinpath(out_dir, "MeanVolume.csv"), DataFrame; header=true)
df[!, SDDP_COL] = hydro_gen
CSV.write(joinpath(out_dir, "MeanVolume.csv"), df)

savefig(
    plot(
        Matrix(df[!, labels]);
        labels=permutedims(labels),
        xlabel="Stage",
        ylabel="Expected Volume (MWh)",
        color=colors,
        shape=markers,
        title="Reservoir Volume Comparison",
    ),
    joinpath(DOCS_ASSETS, "hydro_volume_comparison.png"),
)

# Thermal generation
num_gen = length(results[:simulations][1][1][:powersystem]["solution"]["gen"])
hydro_idx = HydroPowerModels.idx_hydro(results[:data][1])
thermal_gen = [
    mean(
        sum(
            results[:simulations][i][t][:powersystem]["solution"]["gen"]["$j"]["pg"] *
            results[:data][1]["powersystem"]["baseMVA"] for
            j in 1:num_gen if !(j in hydro_idx)
        ) for i in 1:num_sim
    ) for t in 1:(num_stages - rm_stages)
]

savefig(
    plot(
        thermal_gen;
        legend=false,
        xlabel="Stage",
        ylabel="Mwh",
        title="Thermal-Generation $(case)-$(formulation_b)-$(formulation)",
    ),
    joinpath(out_dir, "SDDP-$(case)-$(formulation_b)-$(formulation)-thermal.png"),
)

df = CSV.read(joinpath(out_dir, "MeanGeneration.csv"), DataFrame)
df[!, SDDP_COL] = thermal_gen
CSV.write(joinpath(out_dir, "MeanGeneration.csv"), df)

savefig(
    plot(
        Matrix(df[!, labels]);
        labels=permutedims(labels),
        xlabel="Stage",
        ylabel="Expected Thermal Generation (MWh)",
        color=colors,
        shape=markers,
        title="Thermal Generation Comparison",
    ),
    joinpath(DOCS_ASSETS, "hydro_generation_comparison.png"),
)

# Objective costs
objective_values = [
    sum(results[:simulations][i][t][:stage_objective] for t in 1:(num_stages - rm_stages))
    for i in 1:length(results[:simulations])
]

costs_file = joinpath(out_dir, "costs.csv")
if isfile(costs_file)
    df = CSV.read(costs_file, DataFrame)
    df[!, SDDP_COL] = objective_values
else
    df = DataFrame(Symbol(SDDP_COL) => objective_values)
end
CSV.write(costs_file, df)

println("Mean Sim: ", mean(objective_values))
println("Std  Sim: ", std(objective_values))
