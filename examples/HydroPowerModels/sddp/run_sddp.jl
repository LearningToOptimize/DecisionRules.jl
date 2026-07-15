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
using CSV, DataFrames

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

# Stochastic demand (bolivia/demand_scenarios.csv): overrides
# HydroPowerModels.rainfall_noises with inflow × demand product atoms and
# defines DEMAND_SPREAD / DEMAND_TAG. No-op (deterministic demand, original
# method reproduced verbatim) when the file is absent. Must be included BEFORE
# hydro_thermal_operation builds the policy graphs.
include(joinpath(@__DIR__, "sddp_demand_noise.jl"))

const save_file = "SDDP-$(CASE)-$(FORMULATION)-$(FORMULATION)-h$(NUM_STAGES)$(DEMAND_TAG)-$(Dates.now())"
# DEMAND_TAG keeps demand-noise cuts in a separate file: cuts computed for the
# deterministic-demand program are not valid lower bounds for the noisy one
# (and vice versa), so they must never be warm-start-mixed.
const CUTS_FILE = joinpath(
    CASE_DIR,
    string(FORMULATION),
    string(FORMULATION) * "-" * string(FORMULATION) * DEMAND_TAG * ".cuts.json",
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
    rule.lg === nothing || Wandb.log(
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
    demand_file = joinpath(CASE_DIR, "demand.csv")
    if isfile(demand_file)
        # Bolivia-specific real SEASONAL per-stage demand. demand.csv is
        # [rows × nLoad]; in the Bolivia case load id == load_bus so column j is
        # the demand of load j on bus j. The rows are an annual (48-week)
        # counter-seasonal cycle (demand peaks in the dry, low-inflow winter);
        # they are tiled cyclically across the full horizon. Because SDDP's
        # PolicyGraph builds stage t from alldata[t] (build_model.jl:
        # `data = alldata[min(t, size(alldata,1))]`), alldata is expanded to the
        # full horizon here so each week gets its own demand. This REPLACES the
        # old artificial `pd,qd *= 0.6` down-scaling — the demand is now the real
        # un-scaled load, a materially harder problem.
        demand = Matrix(CSV.read(demand_file, DataFrame; header = false))
        nrows, nload = size(demand)
        alldata = HydroPowerModels.parse_folder(CASE_DIR; stages = NUM_STAGES)
        nbus = length(alldata[1]["powersystem"]["bus"])
        T = length(alldata)
        demand_all = zeros(Float64, T, nbus)
        for t in 1:T
            r = ((t - 1) % nrows) + 1
            @inbounds for j in 1:nload
                demand_all[t, j] = demand[r, j]   # load j sits on bus j (Bolivia)
            end
        end
        HydroPowerModels.set_active_demand!(alldata, demand_all)
        @info "Applied real seasonal per-stage demand from demand.csv" nrows nload T
        return alldata
    else
        # Legacy fallback (non-Bolivia / missing file): the old 0.6 down-scaling.
        alldata = HydroPowerModels.parse_folder(CASE_DIR)
        for load in values(alldata[1]["powersystem"]["load"])
            load["qd"] *= 0.6
            load["pd"] *= 0.6
        end
        return alldata
    end
end

function main()
    println("Run: ", save_file)
    println("Case directory: ", CASE_DIR)
    println("Formulation: ", FORMULATION, " with Clarabel")

    Random.seed!(SEED)
    mkpath(dirname(CUTS_FILE))
    alldata = load_case_data()
    # W&B is best-effort: on a compute node its Python service can fail to start
    # (ServicePollForTokenError). Never let that kill a 24h SDDP run — fall back
    # to stdout-only logging (bound/sim are printed every iteration anyway).
    # DR_SDDP_WANDB=false disables it outright.
    lg = nothing
    if lowercase(get(ENV, "DR_SDDP_WANDB", "true")) in ("true", "1", "yes")
        try
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
                    # Demand-noise provenance: "none" = deterministic demand.
                    "demand_spread" => DEMAND_SPREAD === nothing ? "none" : DEMAND_SPREAD,
                ),
            )
        catch err
            @warn "W&B init failed; continuing with stdout-only logging" exception=(err, catch_backtrace())
            lg = nothing
        end
    end
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
    # DR_SDDP_TIME_LIMIT (seconds) lets a wall-clock budget bind instead of the
    # iteration count, so a 24h SLURM job trains as long as possible and still
    # stops cleanly in time to run the final simulation. 0 => no time limit.
    _tl = parse(Float64, get(ENV, "DR_SDDP_TIME_LIMIT", "0"))
    train_kwargs = _tl > 0 ?
        (iteration_limit = ITERATION_LIMIT, stopping_rules = stopping_rules, time_limit = _tl) :
        (iteration_limit = ITERATION_LIMIT, stopping_rules = stopping_rules)
    HydroPowerModels.train(model; train_kwargs...)
    elapsed = time() - start_time
    bound = SDDP.calculate_bound(model.forward_graph)
    println("Termination status: ", SDDP.termination_status(model.forward_graph))
    println("Elapsed seconds: ", elapsed)
    println("Bound: ", bound)

    SDDP.write_cuts_to_file(model.forward_graph, CUTS_FILE)
    println("Saved cuts: ", CUTS_FILE)

    Random.seed!(SEED)
    # Record the actual load-shed variable so we log a REAL deficit metric, not
    # an inference from the bound/sim gap.
    results = HydroPowerModels.simulate(model, NUM_SIMULATIONS;
        custom_recorders = Dict{Symbol,Function}(
            :deficit => (sp::JuMP.Model) -> sum(JuMP.value.(sp[:deficit]))))
    Teval = NUM_STAGES - RM_STAGES
    objective_values = [
        sum(results[:simulations][i][t][:stage_objective] for t in 1:Teval)
        for i in 1:length(results[:simulations])
    ]
    final_loss = mean(objective_values)
    # Deficit: MW-stage per scenario and % of served energy (baseMVA ≈ 100).
    def_per_scen = [sum(results[:simulations][i][t][:deficit] for t in 1:Teval)
                    for i in 1:length(results[:simulations])]
    served = sum(sum(l["pd"] for l in values(alldata[min(t, length(alldata))]["powersystem"]["load"]))
                 for t in 1:Teval)
    mean_def = mean(def_per_scen)
    pct_def = 100 * mean_def / served
    println("Mean Sim: ", final_loss)
    println("FINAL bound=$(bound) mean_sim=$(final_loss) gap=$(round(100*(final_loss-bound)/abs(bound); digits=3))% ",
            "mean_deficit=$(round(mean_def*100; digits=2))MW-stage pct_deficit=$(round(pct_def; digits=4))% elapsed=$(elapsed)")
    if lg !== nothing
        Wandb.log(
            lg,
            Dict(
                "batch" => ITERATION_LIMIT,
                "metrics/loss" => bound,
                "metrics/final_loss" => final_loss,
                "metrics/rollout_realized_objective_no_deficit" => final_loss,
                "metrics/final_rollout_realized_objective_no_deficit" => final_loss,
                "metrics/elapsed_seconds" => elapsed,
                "metrics/deficit_MWstage" => mean_def * 100,
                "metrics/deficit_pct" => pct_def,
            ),
        )
        close(lg)
    end
end

main()
