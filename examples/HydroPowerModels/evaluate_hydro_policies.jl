# Evaluate pre-trained TS-DDR and TS-LDR policies on the Bolivia LTHD problem
# using stage-wise rollout under the ACP formulation with a fixed scenario set.
#
# This produces an apples-to-apples comparison across all methods using the
# same evaluation protocol:
#   - stage-wise AC-OPF subproblems (Ipopt)
#   - realized-state feedback (closed-loop / deployment semantics)
#   - same seed and number of out-of-sample scenarios
#   - operational cost excluding target-deficit penalty
#
# The script auto-discovers saved .jld2 checkpoints and reconstructs the
# correct policy architecture (LDR vs DDR) from the filename.
# Results are written to eval_costs.csv.
#
# Usage:
#   julia --project=. evaluate_hydro_policies.jl [NUM_SIMULATIONS]
#
# Environment overrides:
#   DR_EVAL_SIMULATIONS=100   number of out-of-sample scenarios
#   DR_EVAL_SEED=1221         random seed for scenario generation

using DecisionRules
using Statistics
using Random
using Flux
using Ipopt
using DiffOpt
using JLD2
using JuMP
using CSV
using DataFrames

const HYDRO_DIR = dirname(@__FILE__)
include(joinpath(HYDRO_DIR, "load_hydropowermodels.jl"))

const CASE_NAME = "bolivia"
const FORMULATION = "ACPPowerModel"
const FORMULATION_FILE = FORMULATION * ".mof.json"
const NUM_STAGES = 96
const NUM_SIMULATIONS = parse(Int, get(ENV, "DR_EVAL_SIMULATIONS",
    length(ARGS) >= 1 ? ARGS[1] : "100"))
const SEED = parse(Int, get(ENV, "DR_EVAL_SEED", "1221"))

const CASE_DIR = joinpath(HYDRO_DIR, CASE_NAME)
const OUT_DIR = joinpath(CASE_DIR, FORMULATION)
const MODEL_DIR = joinpath(OUT_DIR, "models")

println("="^60)
println("Policy Evaluation (TS-DDR + TS-LDR)")
println("="^60)
println("Case:         ", CASE_NAME)
println("Formulation:  ", FORMULATION)
println("Stages:       ", NUM_STAGES)
println("Simulations:  ", NUM_SIMULATIONS)
println("Seed:         ", SEED)
println("="^60)

# ── Build stage-wise subproblems ─────────────────────────────────────────────

diff_optimizer = () -> DiffOpt.diff_optimizer(
    optimizer_with_attributes(
        Ipopt.Optimizer, "print_level" => 0, "linear_solver" => "mumps",
    ),
)

subproblems, state_params_in, state_params_out, uncertainty_samples, initial_state, max_volume =
    build_hydropowermodels(
        CASE_DIR, FORMULATION_FILE;
        num_stages=NUM_STAGES,
        optimizer=diff_optimizer,
        penalty_l1=:auto, penalty_l2=:auto,
    )

num_hydro = length(initial_state)
num_uncertainties = length(uncertainty_samples[1][1])
num_inputs = DecisionRules.policy_input_dim(num_uncertainties, num_hydro)

# ── Generate fixed scenario set ──────────────────────────────────────────────

Random.seed!(SEED)
eval_scenarios = [DecisionRules.sample(uncertainty_samples) for _ in 1:NUM_SIMULATIONS]

# ── Discover saved models ────────────────────────────────────────────────────
#
# Model files encode the training method and policy type in their filename:
#   *-deteq-*   → DDR trained with deterministic equivalent
#   *-subproblems-* → DDR trained with stage-wise decomposition
#   *-shooting-* → DDR trained with multiple shooting
#   *-ldr-*     → LDR (linear decision rule)
#
# DDR models use state_conditioned_policy (LSTM [128,128], sigmoid).
# LDR models use dense_multilayer_nn (identity activation, [64,64]).
# The most recent file (by lexicographic sort on timestamps) is selected
# for each method.

struct PolicySpec
    label::String
    model_file::String
    is_ldr::Bool
end

function _method_variant(base)
    method = if contains(base, "ldr")
        "ldr"
    elseif contains(base, "shooting")
        "shooting"
    elseif contains(base, "subproblems")
        "subproblems"
    elseif contains(base, "deteq")
        "deteq"
    else
        return nothing
    end
    clip_tag = contains(base, "clip") ? "-clip" : ""
    sched_tag = contains(base, "anneal") ? "-anneal" :
                contains(base, "const") ? "-const" : ""
    return method * clip_tag * sched_tag
end

function _variant_label(variant)
    labels = Dict(
        "subproblems-anneal" => "Subproblems (anneal)",
        "subproblems-clip-anneal" => "Subproblems (clip, anneal)",
        "subproblems-const" => "Subproblems (const)",
        "subproblems-clip-const" => "Subproblems (clip, const)",
        "subproblems" => "Subproblems",
        "shooting-anneal" => "Shooting w=12 (anneal)",
        "shooting-clip-anneal" => "Shooting w=12 (clip, anneal)",
        "shooting" => "Shooting w=12",
        "deteq-anneal" => "DE (anneal)",
        "deteq-clip-anneal" => "DE (clip, anneal)",
        "deteq" => "DE",
        "ldr" => "TS-LDR",
    )
    return get(labels, variant, variant)
end

function discover_policies(model_dir)
    files = sort(filter(f -> endswith(f, ".jld2"), readdir(model_dir; join=true)))
    best = Dict{String,Tuple{String,Bool}}()
    for f in files
        base = basename(f)
        variant = _method_variant(base)
        isnothing(variant) && continue
        is_ldr = contains(base, "ldr")
        best[variant] = (f, is_ldr)
    end
    specs = PolicySpec[]
    for (variant, (path, is_ldr)) in sort(collect(best); by=first)
        push!(specs, PolicySpec(_variant_label(variant), path, is_ldr))
    end
    return specs
end

function build_policy(spec::PolicySpec, num_inputs, num_hydro, num_uncertainties)
    if spec.is_ldr
        return dense_multilayer_nn(num_inputs, num_hydro, Int64[64, 64]; activation=identity)
    else
        return state_conditioned_policy(
            num_uncertainties, num_hydro, num_hydro, Int64[128, 128];
            activation=sigmoid, encoder_type=Flux.LSTM,
        )
    end
end

policies = discover_policies(MODEL_DIR)
println("\nDiscovered policies:")
for p in policies
    tag = p.is_ldr ? " (LDR)" : " (DDR)"
    println("  ", p.label, tag, " → ", basename(p.model_file))
end

# ── Evaluate each policy ─────────────────────────────────────────────────────

results = DataFrame()

for spec in policies
    println("\nEvaluating: ", spec.label)

    models = build_policy(spec, num_inputs, num_hydro, num_uncertainties)
    model_state = JLD2.load(spec.model_file, "model_state")
    Flux.loadmodel!(models, model_state)

    objectives_no_deficit = Vector{Float64}(undef, NUM_SIMULATIONS)
    objectives_total = Vector{Float64}(undef, NUM_SIMULATIONS)

    for i in 1:NUM_SIMULATIONS
        Flux.reset!(models)

        objectives_total[i] = simulate_multistage(
            subproblems,
            state_params_in,
            state_params_out,
            initial_state,
            eval_scenarios[i],
            models;
        )

        objectives_no_deficit[i] = DecisionRules.get_objective_no_target_deficit(subproblems)
    end

    violation_share = 1.0 - mean(objectives_no_deficit) / mean(objectives_total)

    println("  Mean cost (no deficit): ", round(mean(objectives_no_deficit); digits=1))
    println("  Std:                    ", round(std(objectives_no_deficit); digits=1))
    println("  Violation share:        ", round(violation_share * 100; digits=2), "%")

    results[!, spec.label] = objectives_no_deficit
end

# ── Write results ────────────────────────────────────────────────────────────

costs_file = joinpath(OUT_DIR, "eval_costs.csv")
CSV.write(costs_file, results)
println("\nSaved: ", costs_file)

# ── Summary table ────────────────────────────────────────────────────────────

println("\n", "="^70)
println(rpad("Method", 35), rpad("Mean", 12), rpad("Std", 12), "N")
println("-"^70)
for col in names(results)
    vals = results[!, col]
    println(
        rpad(col, 35),
        rpad(string(round(mean(vals); digits=1)), 12),
        rpad(string(round(std(vals); digits=1)), 12),
        length(vals),
    )
end
println("="^70)
println("\nNote: SDDP results are from sddp/simulate_sddp_policy.jl")
println("SDDP uses 126 stages (96 + 30 margin) to avoid end-of-horizon effects,")
println("while TS-DDR/TS-LDR use 96 stages. This gives SDDP a structural advantage.")
