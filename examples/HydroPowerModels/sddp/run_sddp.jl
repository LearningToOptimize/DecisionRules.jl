# SDDP baseline: train and simulate SDDP policy on the Bolivia LTHD problem
# using a consistent convex SOCWRConic formulation.

using Clarabel
using HydroPowerModels
using JuMP
using Logging
using PowerModels
using Random
using SDDP
using Statistics
using Wandb, Dates

const SEED = parse(Int, get(ENV, "DR_SDDP_SEED", "1221"))
const CASE = get(ENV, "DR_SDDP_CASE", "bolivia")
const HYDRO_DIR = dirname(@__DIR__)
const CASE_DIR = joinpath(HYDRO_DIR, CASE)
const RM_STAGES = parse(Int, get(ENV, "DR_SDDP_RM_STAGES", "30"))
const NUM_STAGES = parse(Int, get(ENV, "DR_SDDP_NUM_STAGES", string(96 + RM_STAGES)))
const ITERATION_LIMIT = parse(Int, get(ENV, "DR_SDDP_ITERATION_LIMIT", "200"))
const NUM_SIMULATIONS = parse(Int, get(ENV, "DR_SDDP_SIMULATIONS", "300"))
const STAT_REPLICATIONS = parse(Int, get(ENV, "DR_SDDP_STAT_REPLICATIONS", "300"))
const STAT_PERIOD = parse(Int, get(ENV, "DR_SDDP_STAT_PERIOD", "50"))
const FORMULATION = SOCWRConicPowerModel
const save_file = "SDDP-$(CASE)-$(FORMULATION)-$(FORMULATION)-h$(NUM_STAGES)-$(Dates.now())"
const CUTS_FILE = joinpath(
    CASE_DIR,
    string(FORMULATION),
    string(FORMULATION) * "-" * string(FORMULATION) * ".cuts.json",
)

function clarabel_optimizer()
    return Clarabel.Optimizer(;
        verbose=false,
        max_iter=parse(Int, get(ENV, "DR_SDDP_CLARABEL_MAX_ITER", "1000")),
        tol_gap_abs=parse(Float64, get(ENV, "DR_SDDP_CLARABEL_TOL", "1e-7")),
        tol_gap_rel=parse(Float64, get(ENV, "DR_SDDP_CLARABEL_TOL", "1e-7")),
        tol_feas=parse(Float64, get(ENV, "DR_SDDP_CLARABEL_TOL", "1e-7")),
    )
end

mutable struct WandBLog <: SDDP.AbstractStoppingRule
    cuts_file::String
    lg
end

SDDP.stopping_rule_status(::WandBLog) = :not_solved

function SDDP.convergence_test(
    policy::SDDP.PolicyGraph,
    log::Vector{SDDP.Log},
    rule::WandBLog,
)
    mkpath(dirname(rule.cuts_file))
    SDDP.write_cuts_to_file(policy, rule.cuts_file)
    latest = log[end]
    Wandb.log(
        rule.lg,
        Dict(
            "batch" => length(log),
            "metrics/loss" => latest.bound,
            "metrics/rollout_realized_objective_no_deficit" => latest.simulation_value,
        ),
    )
    println(
        "iteration=$(length(log)) bound=$(latest.bound) simulation_value=$(latest.simulation_value)",
    )
    flush(stdout)
    return false
end

function load_case_data()
    alldata = HydroPowerModels.parse_folder(CASE_DIR)
    for load in values(alldata[1]["powersystem"]["load"])
        load["qd"] *= 0.6
        load["pd"] *= 0.6
    end
    return alldata
end

function main()
    println("Run: ", save_file)
    println("Case directory: ", CASE_DIR)
    println("Formulation: ", FORMULATION, " with Clarabel")

    Random.seed!(SEED)
    mkpath(dirname(CUTS_FILE))
    alldata = load_case_data()
    lg = WandbLogger(;
        project="RL",
        name=save_file,
        save_code=false,
        config=Dict(
            "case_name" => CASE,
            "training_method" => "sddp_consistent",
            "formulation" => string(FORMULATION),
            "solver" => "Clarabel",
            "num_stages" => NUM_STAGES,
            "rm_stages" => RM_STAGES,
            "iteration_limit" => ITERATION_LIMIT,
            "num_simulations" => NUM_SIMULATIONS,
            "stat_replications" => STAT_REPLICATIONS,
            "stat_period" => STAT_PERIOD,
            "seed" => SEED,
        ),
    )
    params = create_param(;
        stages=NUM_STAGES,
        model_constructor_grid=FORMULATION,
        post_method=PowerModels.build_opf,
        optimizer=clarabel_optimizer,
    )
    model = hydro_thermal_operation(alldata, params)

    if isfile(CUTS_FILE)
        println("Loading existing cuts: ", CUTS_FILE)
        SDDP.read_cuts_from_file(model.forward_graph, CUTS_FILE)
    end

    stopping_rules = SDDP.AbstractStoppingRule[WandBLog(CUTS_FILE, lg)]
    if STAT_REPLICATIONS > 0
        push!(
            stopping_rules,
            SDDP.Statistical(;
                num_replications=STAT_REPLICATIONS,
                iteration_period=STAT_PERIOD,
            ),
        )
    end

    start_time = time()
    HydroPowerModels.train(
        model;
        iteration_limit=ITERATION_LIMIT,
        stopping_rules=stopping_rules,
    )
    elapsed = time() - start_time
    bound = SDDP.calculate_bound(model.forward_graph)
    println("Termination status: ", SDDP.termination_status(model.forward_graph))
    println("Elapsed seconds: ", elapsed)
    println("Bound: ", bound)

    SDDP.write_cuts_to_file(model.forward_graph, CUTS_FILE)
    println("Saved cuts: ", CUTS_FILE)

    Random.seed!(SEED)
    results = HydroPowerModels.simulate(model, NUM_SIMULATIONS)
    objective_values = [
        sum(results[:simulations][i][t][:stage_objective] for t in 1:(NUM_STAGES - RM_STAGES))
        for i in 1:length(results[:simulations])
    ]
    final_loss = mean(objective_values)
    println("Mean Sim: ", final_loss)
    Wandb.log(
        lg,
        Dict(
            "batch" => ITERATION_LIMIT,
            "metrics/loss" => bound,
            "metrics/final_loss" => final_loss,
            "metrics/rollout_realized_objective_no_deficit" => final_loss,
            "metrics/final_rollout_realized_objective_no_deficit" => final_loss,
            "metrics/elapsed_seconds" => elapsed,
        ),
    )
    close(lg)
end

main()
