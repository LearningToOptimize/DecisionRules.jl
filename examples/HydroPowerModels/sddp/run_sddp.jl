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
# Multi-rung conic recovery (SLOW_PROGRESS stalls once cuts accumulate)
include(joinpath(@__DIR__, "sddp_ac_starts.jl"))

const save_file = "SDDP-$(CASE)-$(FORMULATION)-$(FORMULATION)-h$(NUM_STAGES)$(DEMAND_TAG)-$(Dates.now())"
# DEMAND_TAG keeps demand-noise cuts in a separate file: cuts computed for the
# deterministic-demand program are not valid lower bounds for the noisy one
# (and vice versa), so they must never be warm-start-mixed.
const CUTS_FILE = get(ENV, "DR_SDDP_CUTS_FILE", joinpath(
    CASE_DIR,
    string(FORMULATION),
    string(FORMULATION) * "-" * string(FORMULATION) * DEMAND_TAG * ".cuts.json",
))

function clarabel_optimizer()
    return Clarabel.Optimizer(;
        verbose=false,
        max_iter=parse(Int, get(ENV, "DR_SDDP_CLARABEL_MAX_ITER", "200000")),
        tol_gap_abs=parse(Float64, get(ENV, "DR_SDDP_CLARABEL_TOL", "1e-8")),
        tol_gap_rel=parse(Float64, get(ENV, "DR_SDDP_CLARABEL_TOL", "1e-8")),
        tol_feas=parse(Float64, get(ENV, "DR_SDDP_CLARABEL_TOL", "1e-8")),
        equilibrate_enable=false,
        static_regularization_constant=parse(Float64,
            get(ENV, "DR_SDDP_CLARABEL_STATREG", "1e-8")),
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
    alldata = HydroPowerModels.parse_folder(CASE_DIR; stages=NUM_STAGES)
    for data in alldata
        for load in values(data["powersystem"]["load"])
            load["pd"] *= 0.6
            load["qd"] *= 0.6
        end
        data["powersystem"]["cost_deficit"] = 6000.0 / data["powersystem"]["baseMVA"]
    end
    @info "Canonical MAIN demand" pd_scale=0.6 qd_scale=0.6 deficit_cost=6000.0
    return alldata
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
    # Water balance: K = 0.0036·stage_hours. stage_hours comes from the case's
    # hydro.json (parsed into alldata); default 1 for cases predating the field.
    stage_hours = Int(get(alldata[1]["hydro"], "stage_hours", 1))
    @info "SDDP water balance" stage_hours K_eff = 0.0036 * stage_hours
    params = create_param(;
        stages=NUM_STAGES,
        stage_hours=stage_hours,
        model_constructor_grid=FORMULATION,
        post_method=PowerModels.build_opf,
        optimizer=clarabel_optimizer,
    )
    model = hydro_thermal_operation(alldata, params)
    # SOC-SOC graphs share one recovery (no ACP nodes in the warmup)
    recovery = make_robust_recovery(clarabel_optimizer, clarabel_optimizer)
    model.forward_graph.ext[:numerical_difficulty_callback] = recovery
    model.backward_graph.ext[:numerical_difficulty_callback] = recovery

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
    # Cut generation is stock SDDP: no custom duality_handler. Gating which backward
    # duals become cuts is an intervention in cut creation, so SDDP's own ConicDuality
    # is used instead.
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
    sims = SDDP.simulate(model.forward_graph, NUM_SIMULATIONS;
        custom_recorders = Dict{Symbol,Function}(
            :deficit => (sp::JuMP.Model) -> sum(JuMP.value.(sp[:deficit]))))
    Teval = NUM_STAGES - RM_STAGES
    full_objective_values = [sum(sims[i][t][:stage_objective] for t in 1:NUM_STAGES)
                             for i in eachindex(sims)]
    full_loss = mean(full_objective_values)
    objective_values = [sum(sims[i][t][:stage_objective] for t in 1:Teval)
                        for i in eachindex(sims)]
    final_loss = mean(objective_values)
    # Deficit: MW-stage per scenario and % of served energy (baseMVA ≈ 100).
    def_per_scen = [sum(sims[i][t][:deficit] for t in 1:Teval)
                    for i in eachindex(sims)]
    served = sum(sum(l["pd"] for l in values(alldata[min(t, length(alldata))]["powersystem"]["load"]))
                 for t in 1:Teval)
    mean_def = mean(def_per_scen)
    pct_def = 100 * mean_def / served
    println("Mean Sim: ", final_loss)
    println("FINAL bound_T$(NUM_STAGES)=$(bound) fwd_T$(NUM_STAGES)=$(full_loss) ",
            "gap_T$(NUM_STAGES)=$(round(100*(full_loss-bound)/abs(bound); digits=3))% ",
            "fwd_first$(Teval)=$(final_loss) mean_deficit=$(round(mean_def*100; digits=2))MW-stage ",
            "pct_deficit=$(round(pct_def; digits=4))% elapsed=$(elapsed)")
    if lg !== nothing
        Wandb.log(
            lg,
            Dict(
                "batch" => ITERATION_LIMIT,
                "metrics/loss" => bound,
                "metrics/final_loss" => full_loss,
                "metrics/rollout_realized_objective_no_deficit" => full_loss,
                "metrics/final_rollout_realized_objective_no_deficit" => full_loss,
                "metrics/elapsed_seconds" => elapsed,
                "metrics/deficit_MWstage" => mean_def * 100,
                "metrics/deficit_pct" => pct_def,
            ),
        )
        close(lg)
    end
end

main()
