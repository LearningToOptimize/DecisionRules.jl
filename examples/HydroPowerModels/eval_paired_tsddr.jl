# Paired TS-DDR strict rollout evaluation on the seeded paired protocol.
#
# Scenario indices are a pure function of PAIRED_SCENARIO_SEED (see
# paired_scenario_indices in load_hydropowermodels.jl), so this script and the
# SDDP Historical simulation realize the exact same inflows with no shared
# data file.
#
# Usage:
#   julia --project -t auto eval_paired_tsddr.jl MODEL_PATH
#
# Environment overrides:
#   DR_NUM_EVAL_STAGES=96
#   DR_NUM_SCENARIOS=500
#   DR_ENCODER_LAYERS=128,128
#   DR_HEAD_LAYERS=
#   DR_CONTEXT=                ""/"none", "phase", or "phase+progress"
#   DR_CONTEXT_HORIZON=126     denominator/horizon used for progress context
#   DR_OUTPUT_TAG=""   (when set, ALL output filenames are suffixed with
#                       "_<tag>" — paired_costs_<tag>.csv, etc. — so evaluating
#                       a new checkpoint never clobbers the untagged
#                       ground-truth results; unset → historical filenames)
using DecisionRules
using Flux
using Statistics
using Random
using JuMP, DiffOpt, Ipopt
using JLD2
using CSV, DataFrames
using JSON
using DelimitedFiles

HydroPowerModels_dir = dirname(@__FILE__)
include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))
include(joinpath(HydroPowerModels_dir, "hydro_reachable_policy.jl"))

model_path = ARGS[1]
num_eval_stages = parse(Int, get(ENV, "DR_NUM_EVAL_STAGES", "96"))
num_scenarios = parse(Int, get(ENV, "DR_NUM_SCENARIOS", "500"))
# Optional output tag: suffixes every output filename with "_<tag>" so a new
# checkpoint's evaluation cannot overwrite the untagged ground-truth files.
output_tag = strip(get(ENV, "DR_OUTPUT_TAG", ""))
tag_suffix = isempty(output_tag) ? "" : "_" * output_tag

parse_layers(s::AbstractString) =
    isempty(strip(s)) ? Int64[] : [parse(Int64, strip(x)) for x in split(s, ",") if !isempty(strip(x))]

function canonical_context_mode(raw_mode::AbstractString)
    mode = lowercase(strip(raw_mode))
    mode in ("", "none", "off", "false") && return ""
    mode in ("phase", "phase+progress") && return mode
    error("DR_CONTEXT must be \"\", \"phase\", or \"phase+progress\"; got \"$raw_mode\"")
end

function build_stage_context(mode::AbstractString, horizon::Int, period::Int)
    isempty(mode) && return nothing
    include_progress = mode == "phase+progress"
    return DecisionRules.stage_phase_context(
        horizon;
        period=period,
        include_progress=include_progress,
    )
end

layers = parse_layers(get(ENV, "DR_ENCODER_LAYERS", get(ENV, "DR_LAYERS", "128,128")))
head_layers = parse_layers(get(ENV, "DR_HEAD_LAYERS", ""))
context_mode = canonical_context_mode(get(ENV, "DR_CONTEXT", ""))
context_horizon = parse(Int, get(ENV, "DR_CONTEXT_HORIZON", "126"))
context_period = countlines(joinpath(HydroPowerModels_dir, "bolivia", "inflows.csv"))
context_horizon >= num_eval_stages ||
    error("DR_CONTEXT_HORIZON=$context_horizon must cover DR_NUM_EVAL_STAGES=$num_eval_stages")
stage_context = build_stage_context(context_mode, context_horizon, context_period)
n_context = isnothing(stage_context) ? 0 : size(stage_context, 1)

println("=" ^ 60)
println("Paired TS-DDR Strict Rollout Evaluation")
println("  Model:      $model_path")
println("  Stages:     $num_eval_stages")
println("  Scenarios:  $num_scenarios")
println("  Layers:     $layers")
println("  Head:       $head_layers")
println("  Context:    $(isempty(context_mode) ? "none" : context_mode)")
isempty(output_tag) || println("  Output tag: $output_tag")
println("=" ^ 60)

# ── Build strict subproblems ───────────────────────────────────────────────
case_name = "bolivia"
formulation = "ACPPowerModel"
formulation_file = formulation * ".mof.json"

diff_optimizer =
    () -> DiffOpt.diff_optimizer(
        optimizer_with_attributes(
            Ipopt.Optimizer,
            "print_level" => 0,
            "linear_solver" => "mumps",
        ),
    )

subproblems, state_params_in, state_params_out, uncertainty_samples,
    initial_state, max_volume, hydro_meta = build_hydropowermodels(
    joinpath(HydroPowerModels_dir, case_name),
    formulation_file;
    num_stages=num_eval_stages,
    optimizer=diff_optimizer,
    strict=true,
)

num_hydro = length(initial_state)
nCen = length(uncertainty_samples[1])
println("nHyd=$num_hydro, nCen=$nCen")

# Paired scenario indices: a pure function of the protocol seed (see
# paired_scenario_indices in load_hydropowermodels.jl). Fixed 126-row shape;
# this evaluation uses rows 1:num_eval_stages.
@assert num_eval_stages <= PAIRED_NUM_STAGES
all_indices = paired_scenario_indices(num_scenarios, nCen)
println("Paired protocol: seed=$(PAIRED_SCENARIO_SEED), rows 1:$(num_eval_stages) of $(PAIRED_NUM_STAGES)×$(num_scenarios), nCen=$nCen")

# ── Identify thermal generators ────────────────────────────────────────────
hydro_data = JSON.parsefile(joinpath(HydroPowerModels_dir, case_name, "hydro.json"))
power_data = JSON.parsefile(joinpath(HydroPowerModels_dir, case_name, "PowerModels.json"))
baseMVA = power_data["baseMVA"]
hydro_grid_idx = Set(hg["index_grid"] for hg in hydro_data["Hydrogenerators"])
num_gen = length(power_data["gen"])
thermal_idx = [i for i in 1:num_gen if !(i in hydro_grid_idx)]

volume_to_mw(volume; k=0.0036) = volume / k

pg_vars_per_stage = [DecisionRules.find_variables(subproblems[t], ["pg"]) for t in 1:num_eval_stages]

# ── Build policy and load weights ──────────────────────────────────────────
base_model = hydro_reachable_policy(
    hydro_meta,
    layers;
    combiner_layers=head_layers,
    n_context=n_context,
)
models = isnothing(stage_context) ? base_model : ContextualPolicy(base_model, stage_context)
model_save = JLD2.load(model_path)
model_state = model_save["model_state"]
load_policy_weights!(models, model_state)
println("Loaded model weights from $model_path")

# ── Construct scenarios from pre-sampled indices ───────────────────────────
eval_scenarios = Vector{Vector{Vector{Tuple{eltype(uncertainty_samples[1][1][1][1]), eltype(uncertainty_samples[1][1][1][2])}}}}(undef, num_scenarios)
for s in 1:num_scenarios
    eval_scenarios[s] = [uncertainty_samples[t][all_indices[t, s]] for t in 1:num_eval_stages]
end

# Verify: print first scenario's first stage inflows
println("\nFirst scenario, stage 1 inflows:")
for (param, val) in eval_scenarios[1][1]
    println("  $(JuMP.name(param)) = $val")
end

# ── Run rollout evaluation ─────────────────────────────────────────────────
println("\nEvaluating $num_scenarios scenarios on $num_eval_stages stages...")

# Scenario sharding: build the paired index matrix for ALL `num_scenarios`
# (so the StableRNG pairing is identical across shards) but only roll out the
# slice [DR_SCEN_FIRST, DR_SCEN_LAST]. Rollouts are per-scenario independent, so
# sharding across SLURM tasks is exact and embarrassingly parallel.
scen_first = parse(Int, get(ENV, "DR_SCEN_FIRST", "1"))
scen_last  = parse(Int, get(ENV, "DR_SCEN_LAST", string(num_scenarios)))
scen_ids   = scen_first:min(scen_last, num_scenarios)
nshard     = length(scen_ids)
println("  Shard: scenarios $(first(scen_ids))..$(last(scen_ids))  ($nshard of $num_scenarios)")

"""
    accepted(model) -> Bool

Whether a JuMP solve converged to a usable point.

`LOCALLY_SOLVED` is the normal Ipopt outcome for this nonconvex ACP problem;
`OPTIMAL` and `ALMOST_LOCALLY_SOLVED` are accepted for the same reason MadNLP's
acceptable level is accepted on the ExaModels side.
"""
accepted(model) = JuMP.termination_status(model) in (
    MOI.LOCALLY_SOLVED, MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED,
)

"""
    solve_stage!(model) -> (ok::Bool, retried::Bool, status)

Solve one stage subproblem, retrying ONCE from a cold start if the first attempt
does not converge.

The retry clears every variable's start value, so the second attempt does not
inherit the failed iterate. A stage that fails twice is REPORTED — the scenario
is marked unsolved rather than having a meaningless `objective_value` folded
into the mean.
"""
function solve_stage!(model)
    optimize!(model)
    accepted(model) && return (true, false, JuMP.termination_status(model))
    for variable in JuMP.all_variables(model)
        JuMP.set_start_value(variable, nothing)
    end
    optimize!(model)
    return (accepted(model), true, JuMP.termination_status(model))
end

# Per-scenario cost AND per-scenario solve provenance. A scenario is never
# dropped or renumbered: `scen_done` carries the GLOBAL protocol column id for
# every scenario attempted, and `scen_solved` says whether its cost is usable.
costs = Float64[]
scen_done = Int[]
scen_solved = Bool[]
scen_retried = Int[]        # stages that needed the cold retry
scen_status = String[]      # terminal status of the first failing stage, or ""
vol_trajectories = zeros(num_eval_stages, nshard)
gen_trajectories = zeros(num_eval_stages, nshard)

for (i, s) in enumerate(scen_ids)
    scenario = eval_scenarios[s]
    Flux.reset!(models)

    state = Float64.(initial_state)
    scenario_cost = 0.0
    scenario_ok = true
    scenario_retries = 0
    scenario_status = ""

    for t in 1:num_eval_stages
        for (j, param) in enumerate(state_params_in[t])
            set_parameter_value(param, state[j])
        end

        w_t = scenario[t]
        for (param, val) in w_t
            set_parameter_value(param, val)
        end

        w_vals = Float32.([val for (_, val) in w_t])
        x_hat = models(vcat(w_vals, Float32.(state)))

        for j in 1:num_hydro
            target_param = state_params_out[t][j][1]
            set_parameter_value(target_param, Float64(x_hat[j]))
        end

        ok, retried, status = solve_stage!(subproblems[t])
        retried && (scenario_retries += 1)
        if !ok
            # A non-converged stage makes every later stage of this scenario
            # meaningless, so the rollout stops here and the scenario is
            # recorded as unsolved. Its cost is retained for inspection but is
            # excluded from every statistic below.
            scenario_ok = false
            scenario_status = string(status)
            @warn "Stage solve failed after cold retry" scenario = s stage = t status
            break
        end
        scenario_cost += objective_value(subproblems[t])

        for j in 1:num_hydro
            state[j] = value(state_params_out[t][j][2])
        end

        vol_trajectories[t, i] = sum(volume_to_mw(state[j]) for j in 1:num_hydro)
        gen_trajectories[t, i] = sum(
            value(pg_vars_per_stage[t][j]) * baseMVA for j in thermal_idx
        )
    end

    push!(costs, scenario_cost)
    push!(scen_done, s)
    push!(scen_solved, scenario_ok)
    push!(scen_retried, scenario_retries)
    push!(scen_status, scenario_status)
    if i % 10 == 0 || i == nshard
        solved_costs = costs[scen_solved]
        running = isempty(solved_costs) ? NaN : round(mean(solved_costs); digits=1)
        println("  [$i/$nshard] (scen $s) cost = $(round(scenario_cost; digits=1))" *
                (scenario_ok ? "" : " [UNSOLVED]") * ", running mean = $running")
    end
end

# ── Report results ─────────────────────────────────────────────────────────
# Statistics are computed over SOLVED scenarios only, and the count is printed
# next to them, so a partial evaluation can never be read as a complete one.
solved_costs = costs[scen_solved]
n_solved = length(solved_costs)
n_retried = count(>(0), scen_retried)
println("\n" * "=" ^ 60)
println("Results: Paired TS-DDR Strict ($num_eval_stages stages, shard $(first(scen_ids)):$(last(scen_ids)))")
println("=" ^ 60)
println("  Solved:      $n_solved / $nshard" *
        (n_solved == nshard ? "  (COMPLETE)" : "  ** INCOMPLETE — statistics are over a SUBSET **"))
n_retried == 0 || println("  Retried:     $n_retried scenario(s) needed a cold retry")
if n_solved < nshard
    println("  Unsolved:    $(scen_done[.!scen_solved])")
    println("  Statuses:    $(scen_status[.!scen_solved])")
end
if n_solved > 0
    println("  Mean cost:   $(round(mean(solved_costs); digits=1))")
    println("  Std:         $(round(std(solved_costs); digits=1))")
    println("  Min:         $(round(minimum(solved_costs); digits=1))")
    println("  Max:         $(round(maximum(solved_costs); digits=1))")
    println("  Median:      $(round(median(solved_costs); digits=1))")
end
println("  Violation:   0.0% (strict mode)")
println("=" ^ 60)

# ── Save results ───────────────────────────────────────────────────────────
out_dir = joinpath(HydroPowerModels_dir, case_name, formulation)

const COL_NAME = "TS-DDR (strict, paired)"
# When sharding, suffix the filename with the scenario range so shards never
# collide; a full (1..num_scenarios) run keeps the historical name. Always write
# a `scenario` column so shard CSVs merge unambiguously by scenario id.
is_shard = !(scen_first == 1 && last(scen_ids) == num_scenarios)
shard_suffix = is_shard ? "_s$(first(scen_ids))_$(last(scen_ids))" : ""
costs_file = joinpath(out_dir, "paired_costs$(tag_suffix)$(shard_suffix).csv")
# Solve provenance travels WITH the cost: a merge downstream can then reject an
# incomplete set instead of averaging a garbage objective from a failed solve.
df = DataFrame(
    :scenario => collect(scen_done),
    Symbol(COL_NAME) => costs,
    :all_stages_solved => scen_solved,
    :stages_retried => scen_retried,
    :failure_status => scen_status,
)
CSV.write(costs_file, df)
println("Saved: $costs_file")

mean_vol = vec(mean(vol_trajectories; dims=2))
vol_file = joinpath(out_dir, "paired_MeanVolume$(tag_suffix)$(shard_suffix).csv")
df_vol = DataFrame(Symbol(COL_NAME) => mean_vol)
CSV.write(vol_file, df_vol)
println("Saved: $vol_file")

mean_gen = vec(mean(gen_trajectories; dims=2))
gen_file = joinpath(out_dir, "paired_MeanGeneration$(tag_suffix)$(shard_suffix).csv")
df_gen = DataFrame(Symbol(COL_NAME) => mean_gen)
CSV.write(gen_file, df_gen)
println("Saved: $gen_file")

results_dir = joinpath(out_dir, "results")
mkpath(results_dir)
results_file = joinpath(results_dir, "paired_strict_rollout$(tag_suffix)$(shard_suffix).jld2")
jldsave(results_file;
    costs=costs,
    scenarios=collect(scen_done),
    vol_trajectories=vol_trajectories,
    gen_trajectories=gen_trajectories,
    scenario_indices=all_indices[1:num_eval_stages, scen_ids],
)
println("Saved: $results_file")
