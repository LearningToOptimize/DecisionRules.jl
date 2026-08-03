# SDDP baseline with inconsistent formulations: train SDDP with a convex
# SOCWRConic backward-pass formulation and an AC forward-pass formulation,
# then simulate the resulting policy under the AC model.
#
# Environment overrides for local smoke tests:
#   DR_SDDP_ITERATION_LIMIT=2
#   DR_SDDP_SIMULATIONS=2
#   DR_SDDP_STAT_REPLICATIONS=2
#   DR_SDDP_STAT_PERIOD=1

using Clarabel
using HydroPowerModels
using JuMP
using Logging
using MadNLP
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
const ITERATION_LIMIT = parse(Int, get(ENV, "DR_SDDP_ITERATION_LIMIT", "2000"))
const NUM_SIMULATIONS = parse(Int, get(ENV, "DR_SDDP_SIMULATIONS", "300"))
const STAT_REPLICATIONS = parse(Int, get(ENV, "DR_SDDP_STAT_REPLICATIONS", "300"))
const STAT_PERIOD = parse(Int, get(ENV, "DR_SDDP_STAT_PERIOD", "200"))
# Reactive-demand scaler (historical value 0.6; see load_case_data comment)
const QD_SCALER = parse(Float64, get(ENV, "DR_SDDP_QD_SCALER", "0.6"))

const FORMULATION_BACKWARD = SOCWRConicPowerModel
const FORMULATION_FORWARD = ACPPowerModel

# Robust flat-voltage primal starts for the ACP forward graph (vm = 0 default
# start is singular for polar AC and crashes MadNLP at stressed load levels).
include(joinpath(@__DIR__, "sddp_ac_starts.jl"))

# The frozen case has DETERMINISTIC demand and inflow-only uncertainty, so the
# stage noise is HydroPowerModels' own `rainfall_noises` — no override, no
# product atoms. A demand file appearing in the case directory would change the
# stochastic program underneath the cuts, so its absence is asserted before any
# model is built.
for name in ("demand.csv", "demand_scenarios.csv", "demand_noise.csv")
    isfile(joinpath(CASE_DIR, name)) && error(
        "$name is present in $CASE_DIR. Cuts trained against a different " *
        "stochastic program are not valid lower bounds for this one.",
    )
end

const save_file = "SDDP-$(CASE)-$(FORMULATION_FORWARD)-$(FORMULATION_BACKWARD)-h$(NUM_STAGES)-$(Dates.now())"
const CUTS_DIR = joinpath(CASE_DIR, string(FORMULATION_FORWARD))
const CUTS_FILE = get(ENV, "DR_SDDP_CUTS_FILE", joinpath(
    CUTS_DIR,
    string(FORMULATION_BACKWARD) * "-" * string(FORMULATION_FORWARD) * ".cuts.json",
))

function clarabel_optimizer()
    return Clarabel.Optimizer(;
        verbose=false,
        # Bound integrity is more important than continuing a damaged run.
        # This configuration independently solved the cut-loaded node-14 dump
        # to OPTIMAL; stronger 1e-5/1e-4 regularization falsely reported that
        # same feasible problem INFEASIBLE (cutval_11220957.out).
        max_iter=parse(Int, get(ENV, "DR_SDDP_CLARABEL_MAX_ITER", "200000")),
        tol_gap_abs=parse(Float64, get(ENV, "DR_SDDP_CLARABEL_TOL", "1e-8")),
        tol_gap_rel=parse(Float64, get(ENV, "DR_SDDP_CLARABEL_TOL", "1e-8")),
        tol_feas=parse(Float64, get(ENV, "DR_SDDP_CLARABEL_TOL", "1e-8")),
        equilibrate_enable=false,
        static_regularization_constant=parse(Float64,
            get(ENV, "DR_SDDP_CLARABEL_STATREG", "1e-8")),
    )
end

function madnlp_optimizer()
    return MadNLP.Optimizer(;
        print_level=parse(Int, get(ENV, "DR_SDDP_MADNLP_PRINT_LEVEL", "0")),
    )
end

# Regularization variants for the recovery ladder (rung 3+). Identical to
# `clarabel_optimizer` in every respect EXCEPT static regularization, and in
# particular at the same strict 1e-8 feasibility/gap tolerances — this is not a
# tolerance ladder. Rungs 1 and 2 of the recovery are numerically identical to
# each other, so an ALMOST_OPTIMAL caused by a knife-edge KKT resonance
# reproduces on both; node-122 diagnosis (PROJECT.md §19.12) showed 1e-11,
# 3e-11, 3e-10 and 1e-9 all reach literal OPTIMAL on the instance where 1e-10
# does not, with objectives agreeing to ~1e-7 relative. Ordered nearest-first
# from the default so the mildest change is tried first.
const CLARABEL_STATREG_VARIANTS = let raw = get(ENV, "DR_SDDP_CLARABEL_STATREG_VARIANTS",
                                                "3e-10,3e-11,1e-9,1e-11")
    [parse(Float64, strip(x)) for x in split(raw, ",") if !isempty(strip(x))]
end

function clarabel_variant_optimizers()
    base = parse(Float64, get(ENV, "DR_SDDP_CLARABEL_STATREG", "1e-8"))
    tol = parse(Float64, get(ENV, "DR_SDDP_CLARABEL_TOL", "1e-8"))
    maxit = parse(Int, get(ENV, "DR_SDDP_CLARABEL_MAX_ITER", "200000"))
    # Skip any variant equal to the base setting: retrying it would repeat the
    # failing solve exactly, which is what rungs 1-2 already do.
    return Function[
        () -> Clarabel.Optimizer(; verbose=false, max_iter=maxit,
            tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol,
            equilibrate_enable=false, static_regularization_constant=sr)
        for sr in CLARABEL_STATREG_VARIANTS if !isapprox(sr, base; rtol=1e-12)
    ]
end

mutable struct WandBLog <: SDDP.AbstractStoppingRule
    cuts_file::String
    lg
end

# Out-of-sample metrics produced by StatisticalMetrics (below) and picked up by WandBLog
# on the next iteration's log line. A Ref rather than a second simulation: SDDP's own
# convergence check already simulates the policy out-of-sample, so these numbers are a
# by-product of work that is happening anyway.
const OUT_OF_SAMPLE_METRICS = Ref{Dict{String,Any}}(Dict{String,Any}())

"""
    StatisticalMetrics(; num_replications, iteration_period, z_score, lg)

Drop-in REPLACEMENT for `SDDP.Statistical` — not an addition. It performs the identical
convergence test (simulate `num_replications` out-of-sample paths every
`iteration_period` iterations; stop when the bound enters the simulation's confidence
interval) but keeps the simulation results instead of discarding them, and records the
physical quantities that decide whether the reported AC-SOC gap is real:
deficit (is the AC policy shedding load?), storage, and spill.

`SDDP.Statistical` cannot be reused for this: its `results` is a local variable inside
`convergence_test` and is never exposed. Since this rule replaces it, the number of
simulations is UNCHANGED — the same 300 replications every 200 iterations, simply not
thrown away. Running this alongside `SDDP.Statistical` would double the simulation cost
and must not be done.

The recorders read values off models the simulation has already solved, so they add no
solves — only field reads.
"""
struct StatisticalMetrics <: SDDP.AbstractStoppingRule
    num_replications::Int
    iteration_period::Int
    z_score::Float64
    lg
    baseMVA::Float64   # pu -> MW for the deficit report; taken from the parsed case
end

SDDP.stopping_rule_status(::StatisticalMetrics) = :statistical

function SDDP.convergence_test(
    graph::SDDP.PolicyGraph,
    log::Vector{SDDP.Log},
    rule::StatisticalMetrics,
)
    if length(log) % rule.iteration_period != 0
        return false
    end
    results = SDDP.simulate(graph, rule.num_replications;
        custom_recorders = Dict{Symbol,Function}(
            :deficit => sp -> sum(JuMP.value.(sp[:deficit])),
            :volume  => sp -> sum(JuMP.value(sp[:reservoir][i].out)
                                  for i in 1:length(sp[:reservoir])),
            :spill   => sp -> sum(JuMP.value.(sp[:spill])),
        ))
    objectives = map(sim -> sum(s[:stage_objective] for s in sim), results)
    sample_mean = mean(objectives)
    sample_ci = rule.z_score * std(objectives) / sqrt(rule.num_replications)

    # Physical audit over the SAME simulations. baseMVA converts pu -> MW so the deficit
    # is reported in physical units and is directly comparable to the case's served load.
    baseMVA = rule.baseMVA
    T = length(results[1])
    defs = [sum(s[:deficit] for s in sim) * baseMVA for sim in results]
    vols = [mean(s[:volume] for s in sim) for sim in results]
    spil = [sum(s[:spill] for s in sim) for sim in results]
    nshed = sum(count(s -> s[:deficit] > 1e-6, sim) for sim in results)
    bound = log[end].bound
    OUT_OF_SAMPLE_METRICS[] = Dict{String,Any}(
        "metrics/oos_cost_mean" => sample_mean,
        "metrics/oos_cost_ci95" => sample_ci,
        "metrics/oos_gap_pct" => abs(bound) > 0 ? 100 * (sample_mean - bound) / abs(bound) : NaN,
        "metrics/oos_deficit_mw_stages_mean" => mean(defs),
        "metrics/oos_deficit_mw_stages_max" => maximum(defs),
        "metrics/oos_scenarios_with_deficit" => count(>(1e-6), defs),
        "metrics/oos_stages_with_deficit" => nshed,
        "metrics/oos_stages_total" => T * length(results),
        "metrics/oos_mean_storage" => mean(vols),
        "metrics/oos_spill_total" => mean(spil),
        "metrics/oos_replications" => rule.num_replications,
    )
    println("  [oos] it=$(length(log)) cost=$(round(sample_mean;digits=1))±$(round(sample_ci;digits=1)) " *
            "gap=$(round(100*(sample_mean-bound)/abs(bound);digits=3))% " *
            "deficit_MW=$(round(mean(defs);digits=4)) shed_stages=$nshed/$(T*length(results)) " *
            "storage=$(round(mean(vols);digits=3)) spill=$(round(mean(spil);digits=3))")
    flush(stdout)
    rule.lg === nothing || Wandb.log(rule.lg,
        merge(Dict{String,Any}("batch" => length(log)), OUT_OF_SAMPLE_METRICS[]))

    return graph.objective_sense == MOI.MIN_SENSE ?
        sample_mean - sample_ci <= bound : bound <= sample_mean + sample_ci
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
    gap = abs(latest.bound) > 0 ?
        100 * (latest.simulation_value - latest.bound) / abs(latest.bound) : NaN
    # NOTE: no simulation is run here. The out-of-sample metrics come from the
    # StatisticalMetrics rule below, which extracts them from the simulations SDDP's own
    # convergence check ALREADY performs — adding a second simulation here would slow
    # training to re-derive information that already exists.
    extra = OUT_OF_SAMPLE_METRICS[]
    rule.lg === nothing || Wandb.log(
        rule.lg,
        merge(Dict{String,Any}(
            "batch" => length(log),
            "metrics/loss" => latest.bound,
            "metrics/bound" => latest.bound,
            "metrics/simulation_value" => latest.simulation_value,
            "metrics/gap_pct" => gap,
            "metrics/rollout_realized_objective_no_deficit" => latest.simulation_value,
            "metrics/elapsed_seconds" => latest.time,
            "metrics/total_solves" => latest.total_solves,
        ), extra),
    )
    println(
        "iteration=$(length(log)) bound=$(latest.bound) simulation_value=$(latest.simulation_value) gap=$(round(gap;digits=3))%",
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
    println("Stages: ", NUM_STAGES, " (reporting first ", NUM_STAGES - RM_STAGES, ")")
    println("Backward formulation: ", FORMULATION_BACKWARD, " with Clarabel")
    println("Forward formulation: ", FORMULATION_FORWARD, " with MadNLP")
    println("Iteration limit: ", ITERATION_LIMIT)
    println("Simulations: ", NUM_SIMULATIONS)

    Random.seed!(SEED)
    mkpath(CUTS_DIR)
    alldata = load_case_data()
    lg = nothing
    if lowercase(get(ENV, "DR_SDDP_WANDB", "true")) in ("true","1","yes")
      try
        lg = WandbLogger(;
        project="RL",
        name=save_file,
        save_code=false,
        config=Dict(
            "case_name" => CASE,
            "training_method" => "sddp_inconsistent",
            "backward_formulation" => string(FORMULATION_BACKWARD),
            "forward_formulation" => string(FORMULATION_FORWARD),
            "backward_solver" => "Clarabel",
            "forward_solver" => "MadNLP",
            "num_stages" => NUM_STAGES,
            "rm_stages" => RM_STAGES,
            "iteration_limit" => ITERATION_LIMIT,
            "num_simulations" => NUM_SIMULATIONS,
            "stat_replications" => STAT_REPLICATIONS,
            "stat_period" => STAT_PERIOD,
            "seed" => SEED,
            "demand" => "deterministic (0.6 x PowerModels.json, active and reactive)",
        ),
      )
      catch err; @warn "W&B init failed; stdout-only" err; lg=nothing; end
    end

    # Water balance: K = 0.0036·stage_hours (from hydro.json; default 1).
    stage_hours = Int(get(alldata[1]["hydro"], "stage_hours", 1))
    @info "SDDP water balance" stage_hours K_eff = 0.0036 * stage_hours
    params = create_param(;
        stages=NUM_STAGES,
        stage_hours=stage_hours,
        model_constructor_grid=FORMULATION_BACKWARD,
        model_constructor_grid_forward=FORMULATION_FORWARD,
        post_method=PowerModels.build_opf,
        optimizer=clarabel_optimizer,
        optimizer_forward=madnlp_optimizer,
    )

    model = hydro_thermal_operation(alldata, params)
    # ACP forward subproblems need non-singular voltage starts (see sddp_ac_starts.jl)
    preset_ac_starts!(model.forward_graph)
    # multi-attempt numerical recovery on BOTH graphs (intermittent MadNLP
    # failures at stressed operating points survive SDDP's single-retry default;
    # rung 2+ re-attaches a brand-new solver instance — see sddp_ac_starts.jl)
    # NOTE: the conic_variants regularization ladder is deliberately NOT used. Retrying
    # a failed backward solve at a different regularization until one returns OPTIMAL
    # changes which cuts get built, i.e. it influences cut creation. Cut generation must
    # be stock SDDP behaviour, so the recovery is the historical 2-arg form only.
    recovery = make_robust_recovery(madnlp_optimizer, clarabel_optimizer)
    model.forward_graph.ext[:numerical_difficulty_callback] = recovery
    model.backward_graph.ext[:numerical_difficulty_callback] = recovery

    if isfile(CUTS_FILE)
        println("Loading existing cuts: ", CUTS_FILE)
        SDDP.read_cuts_from_file(model.forward_graph, CUTS_FILE)
    end

    stopping_rules = SDDP.AbstractStoppingRule[WandBLog(CUTS_FILE, lg)]
    if STAT_REPLICATIONS > 0
        # StatisticalMetrics REPLACES SDDP.Statistical (identical convergence test, same
        # number of simulations) and additionally records deficit/storage/spill from
        # those same out-of-sample paths. Never push both: that would simulate twice.
        push!(
            stopping_rules,
            StatisticalMetrics(
                STAT_REPLICATIONS,
                STAT_PERIOD,
                1.96,
                lg,
                Float64(alldata[1]["powersystem"]["baseMVA"]),
            ),
        )
    end

    start_time = time()
    _tl = parse(Float64, get(ENV, "DR_SDDP_TIME_LIMIT", "0"))
    # Cut generation is stock SDDP: no custom duality_handler. StrictConicDuality was
    # removed because rejecting/accepting backward duals on a status test is an
    # intervention in how cuts are created; SDDP's own ConicDuality is used instead.
    _kw = _tl>0 ? (iteration_limit=ITERATION_LIMIT, stopping_rules=stopping_rules, time_limit=_tl) :
                  (iteration_limit=ITERATION_LIMIT, stopping_rules=stopping_rules)
    HydroPowerModels.train(model; _kw...)
    elapsed = time() - start_time

    status = SDDP.termination_status(model.forward_graph)
    bound = SDDP.calculate_bound(model.forward_graph)
    println("Termination status: ", status)
    println("Elapsed seconds: ", elapsed)
    println("Bound: ", bound)

    SDDP.write_cuts_to_file(model.forward_graph, CUTS_FILE)
    println("Saved cuts: ", CUTS_FILE)

    Random.seed!(SEED)
    # SDDP.simulate directly: HydroPowerModels.simulate hardcodes its own
    # custom_recorders and silently drops this one (the :deficit reads below
    # would KeyError after the full training otherwise).
    sims = SDDP.simulate(model.forward_graph, NUM_SIMULATIONS;
        custom_recorders = Dict{Symbol,Function}(
            :deficit => (sp::JuMP.Model) -> sum(JuMP.value.(sp[:deficit]))))
    Teval = NUM_STAGES - RM_STAGES
    # Full-horizon forward cost: the ONLY number comparable to the bound
    # (bounds are horizon-specific — a T-stage bound does not bound a T'-stage
    # metric). The Teval truncation below is a separate policy-quality report.
    full_objective_values = [sum(sims[i][t][:stage_objective] for t in 1:NUM_STAGES)
        for i in 1:length(sims)]
    full_loss = mean(full_objective_values)
    objective_values = [sum(sims[i][t][:stage_objective] for t in 1:Teval)
        for i in 1:length(sims)]
    final_loss = mean(objective_values)
    def_per = [sum(sims[i][t][:deficit] for t in 1:Teval) for i in 1:length(sims)]
    served = sum(sum(l["pd"] for l in values(alldata[min(t,length(alldata))]["powersystem"]["load"])) for t in 1:Teval)
    mean_def = mean(def_per); pct_def = 100*mean_def/served
    println("Mean Sim: ", final_loss)
    # THE gap: ACP forward cost vs SOCP bound — SAME horizon (NUM_STAGES) on
    # both sides. final_loss (first Teval stages) is reported separately and
    # must never be compared to the bound.
    println("FINAL bound_T$(NUM_STAGES)=$(bound) fwd_T$(NUM_STAGES)=$(full_loss) ",
            "GAP_T$(NUM_STAGES)=$(round(100*(full_loss-bound)/abs(bound);digits=3))% ",
            "fwd_first$(Teval)=$(final_loss) ",
            "deficit=$(round(mean_def*100;digits=2))MW-stage pct_deficit=$(round(pct_def;digits=4))% elapsed=$(elapsed)")
    if lg !== nothing
        Wandb.log(lg, Dict("batch"=>ITERATION_LIMIT, "metrics/loss"=>bound, "metrics/final_loss"=>full_loss,
            "metrics/rollout_realized_objective_no_deficit"=>full_loss,
            "metrics/final_rollout_realized_objective_no_deficit"=>full_loss,
            "metrics/elapsed_seconds"=>elapsed, "metrics/gap_pct"=>100*(full_loss-bound)/abs(bound),
            "metrics/deficit_pct"=>pct_def))
        close(lg)
    end
end

main()
