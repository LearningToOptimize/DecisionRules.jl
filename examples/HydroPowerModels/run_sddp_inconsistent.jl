# SDDP baseline with inconsistent formulations: train SDDP with a convex
# backward-pass formulation (SOCWRConic) and an AC forward-pass formulation,
# then simulate the resulting policy under the AC model.
# Requires HydroPowerModels.jl, Gurobi, Mosek, and MadNLP.
using Gurobi
using MosekTools
using MadNLP
using HydroPowerModels
using JuMP
using Statistics
import SDDP: stopping_rule_status, convergence_test, PolicyGraph, AbstractStoppingRule, Log
using Wandb, Dates, Logging

using Random
seed = 1221

# Load case
case = "bolivia"
case_dir = joinpath(dirname(@__FILE__), case)
alldata = HydroPowerModels.parse_folder(case_dir);
for load in values(alldata[1]["powersystem"]["load"])
    load["qd"] = load["qd"] * 0.6
    load["pd"] = load["pd"] * 0.6
end
rm_stages = 30
num_stages = 96 + rm_stages
formulation_b = SOCWRConicPowerModel
formulation = ACPPowerModel

# Wandb stopping rule
mutable struct WandBLog <: SDDP.AbstractStoppingRule
    lg
end

SDDP.stopping_rule_status(::WandBLog) = :not_solved

save_file = "SDDP-$(case)-$(formulation)-$(formulation_b)-h$(num_stages)-$(now())"

cuts_file = joinpath(
    case_dir,
    string(formulation),
    string(formulation_b)*"-"*string(formulation)*".cuts.json",
)

function SDDP.convergence_test(
    policy::SDDP.PolicyGraph, log::Vector{SDDP.Log}, rule::WandBLog
)
    SDDP.write_cuts_to_file(policy, cuts_file)
    Wandb.log(
        rule.lg,
        Dict(
            "iteration" => length(log),
            "bound" => log[end].bound,
            "metrics/loss" => log[end].simulation_value,
        ),
    )
    return false
end

lg = WandbLogger(; project="HydroPowerModels", name=save_file, save_code=false)

# Train: SOCWRConic backward pass, ACP forward pass via MadNLP
Random.seed!(seed)

params = create_param(;
    stages=num_stages,
    model_constructor_grid=formulation_b,
    model_constructor_grid_forward=formulation,
    post_method=PowerModels.build_opf,
    optimizer=Mosek.Optimizer,
    optimizer_forward=() -> MadNLP.Optimizer(; print_level=MadNLP.INFO),
);

m = hydro_thermal_operation(alldata, params);

if isfile(cuts_file)
    SDDP.read_cuts_from_file(m.forward_graph, cuts_file)
end

start_time = time()
HydroPowerModels.train(
    m;
    iteration_limit=2000,
    stopping_rules=[
        WandBLog(lg); SDDP.Statistical(; num_replications=300, iteration_period=200)
    ],
);
end_time = time() - start_time

(SDDP.termination_status(m.forward_graph), end_time)

SDDP.write_cuts_to_file(m.forward_graph, cuts_file)

# Simulation
using Random: Random
Random.seed!(seed)
results = HydroPowerModels.simulate(m, 300);

# Objective
objective_values = [
    sum(results[:simulations][i][t][:stage_objective] for t in 1:(num_stages - rm_stages))
    for i in 1:length(results[:simulations])
]
println("Mean Sim: ", mean(objective_values))

Wandb.log(
    lg,
    Dict(
        "bound" => SDDP.calculate_bound(m.forward_graph),
        "metrics/final_loss" => mean(objective_values),
    ),
)

close(lg)
