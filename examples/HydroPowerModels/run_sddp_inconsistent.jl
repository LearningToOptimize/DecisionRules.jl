# ## Importing package and optimizer
using Gurobi # Gurobi, MosekTools
using MosekTools
# using Ipopt, HSL_jll
using MadNLP
using HydroPowerModels
using JuMP
using Statistics
import SDDP: stopping_rule_status, convergence_test, PolicyGraph, AbstractStoppingRule, Log
using Wandb, Dates, Logging

# ## Initialization
using Random
seed = 1221

# ## Load Case Specifications

# Data
case = "bolivia" # bolivia, case3
case_dir = joinpath(dirname(@__FILE__), case)
alldata = HydroPowerModels.parse_folder(case_dir);
for load in values(alldata[1]["powersystem"]["load"])
    load["qd"] = load["qd"] * 0.6
    load["pd"] = load["pd"] * 0.6
end
rm_stages = 30 # 0, 12
num_stages = 96 + rm_stages # 96, 48
formulation_backup = SOCWRConicPowerModel
formulation_b = SOCWRConicPowerModel # SOCWRConicPowerModel, DCPPowerModel, DCPLLPowerModel
formulation = ACPPowerModel

# # Parameters
# params = create_param(;
#     stages = num_stages,
#     model_constructor_grid = formulation_b,
#     model_constructor_grid_forward = formulation,
#     post_method = PowerModels.build_opf,
#     optimizer = Gurobi.Optimizer,
#     optimizer_forward = Ipopt.Optimizer, 
#     #     "print_level" => 0,
#     #     "hsllib" => HSL_jll.libhsl_path,
#     #     "linear_solver" => "ma97"
#     # ),
#     # discount_factor=0.99502487562
# );

# ## Build Model
# m = hydro_thermal_operation(alldata, params);

# # Save subproblem
# results = HydroPowerModels.simulate(m, 2);
# model = m.forward_graph[1].subproblem
# delete(model, all_variables(model)[findfirst(x -> x == "",  name.(all_variables(model)))])
# JuMP.write_to_file(model, joinpath(case_dir, string(formulation)) * ".mof.json")

# Wandb logger

"""
    WandBLog(limit::Int)

Teriminate the algorithm after `limit` number of iterations.
"""
mutable struct WandBLog <: SDDP.AbstractStoppingRule
    lg
end

SDDP.stopping_rule_status(::WandBLog) = :not_solved

save_file = "SDDP-$(case)-$(formulation)-$(formulation_b)-h$(num_stages)-$(now())"

cuts_file = joinpath(case_dir, string(formulation), string(formulation_b)*"-"*string(formulation)*".cuts.json")

function SDDP.convergence_test(policy::SDDP.PolicyGraph, log::Vector{SDDP.Log}, rule::WandBLog)
    SDDP.write_cuts_to_file(policy,joinpath(case_dir, string(formulation), string(formulation_b)*"-"*string(formulation)*".cuts.json"))

    Wandb.log(rule.lg, Dict(
        "iteration" => length(log),
        "bound" => log[end].bound,
        "metrics/loss" => log[end].simulation_value,
    ))
    return false
end

lg = WandbLogger(
    project = "HydroPowerModels",
    name = save_file,
    save_code = false,
)

# global num_trys = 0
# max_num_trys = 60

# ## Train
Random.seed!(seed)
# while num_trys <= max_num_trys
    # try
        # Parameters
        params = create_param(;
            stages = num_stages,
            model_constructor_grid = formulation_b,
            model_constructor_grid_forward = formulation,
            post_method = PowerModels.build_opf,
            optimizer = Mosek.Optimizer,
            optimizer_forward = ()->MadNLP.Optimizer(print_level=MadNLP.INFO),
            # optimizer_with_attributes(Ipopt.Optimizer, 
            #     "print_level" => 0,
            #     "hsllib" => HSL_jll.libhsl_path,
            #     "linear_solver" => "ma97"
            # ),
            # discount_factor=0.99502487562
        );

        # ## Build Model
        m = hydro_thermal_operation(alldata, params);

        # ## Load Policy
        if isfile(cuts_file)
            SDDP.read_cuts_from_file(m.forward_graph,cuts_file)
        end

        start_time = time()
        HydroPowerModels.train(
            m;
            iteration_limit = 2000,
            stopping_rules = [WandBLog(lg); SDDP.Statistical(; num_replications = 300, iteration_period = 200)],
        );
        end_time = time() - start_time

        # Termination Status and solve time (s)
        (SDDP.termination_status(m.forward_graph), end_time)

        # save cuts
        SDDP.write_cuts_to_file(m.forward_graph,cuts_file)

        # global num_trys += max_num_trys + 1

    # catch
    #     # Parameters
    #     params = create_param(;
    #         stages = num_stages,
    #         model_constructor_grid = formulation_b,
    #         model_constructor_grid_forward = formulation_backup,
    #         post_method = PowerModels.build_opf,
    #         optimizer = Gurobi.Optimizer,
    #         optimizer_forward = Mosek.Optimizer,
    #         # discount_factor=0.99502487562
    #     );

    #     # ## Build Model
    #     m = hydro_thermal_operation(alldata, params);

    #     # ## Load Policy
    #     if isfile(cuts_file)
    #         SDDP.read_cuts_from_file(m.forward_graph,cuts_file)
    #     end

    #     start_time = time()
    #     HydroPowerModels.train(
    #         m;
    #         iteration_limit = 10,
    #         stopping_rules = [WandBLog(lg)],
    #     );
    #     end_time = time() - start_time

    #     # Termination Status and solve time (s)
    #     (SDDP.termination_status(m.forward_graph), end_time)

    #     # save cuts
    #     SDDP.write_cuts_to_file(m.forward_graph,cuts_file)

    #     global num_trys += 1
    # end
# end

# ## Simulation
using Random: Random
Random.seed!(seed)
results = HydroPowerModels.simulate(m, 300);

# ## Objective
objective_values = [sum(results[:simulations][i][t][:stage_objective] for t=1:num_stages-rm_stages) for i=1:length(results[:simulations])]
println("Mean Sim: ", mean(objective_values))

Wandb.log(lg, Dict(
    "bound" => SDDP.calculate_bound(m.forward_graph),
    "metrics/final_loss" => mean(objective_values),
))

# Finish the run
close(lg)