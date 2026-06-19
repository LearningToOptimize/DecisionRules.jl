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

labels = ["TS-DDR"; "TS-LDR"; "SDDP-DCLL"]
colors = [:black :purple :red]
markers = [:hline :+ :pixel]

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
    joinpath(
        case_dir,
        string(formulation),
        "SDDP-$(case)-$(formulation_b)-$(formulation)-Volume.png",
    ),
)

df = CSV.read(
    joinpath(case_dir, string(formulation), "MeanVolume.csv"), DataFrame; header=true
)
df[!, "$(string(formulation_b))"] = hydro_gen

CSV.write(joinpath(case_dir, string(formulation), "MeanVolume.csv"), df)

savefig(
    plot(
        Matrix(df[!, labels]);
        labels=permutedims(names(df[!, labels])),
        xlabel="Stage",
        ylabel="Expected Volume (MWh)",
        color=colors,
        shape=markers,
    ),
    joinpath(case_dir, string(formulation), "DCLL-Comparison-$(case)-Volume.png"),
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
    joinpath(
        case_dir,
        string(formulation),
        "SDDP-$(case)-$(formulation_b)-$(formulation)-thermal.png",
    ),
)

df = CSV.read(joinpath(case_dir, string(formulation), "MeanGeneration.csv"), DataFrame)
df[!, "SOC"] = thermal_gen

CSV.write(joinpath(case_dir, string(formulation), "MeanGeneration.csv"), df)

savefig(
    plot(
        Matrix(df[!, labels]);
        labels=permutedims(names(df[!, labels])),
        xlabel="Stage",
        ylabel="Expected Thermal Generation (MWh)",
        color=colors,
        shape=markers,
    ),
    joinpath(case_dir, string(formulation), "DCLL-Comparison-$(case)-thermal.png"),
)

# Objective costs
objective_values = [
    sum(results[:simulations][i][t][:stage_objective] for t in 1:(num_stages - rm_stages))
    for i in 1:length(results[:simulations])
]

df = CSV.read(joinpath(case_dir, string(formulation), "costs.csv"), DataFrame)
df[!, "SDDP_SOC"] = objective_values

CSV.write(joinpath(case_dir, string(formulation), "costs.csv"), df)

println("Mean Sim: ", mean(objective_values))
