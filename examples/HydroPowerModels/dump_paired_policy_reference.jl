# Dump the paired-evaluation policy reference for cross-package equivalence checks.
#
# This script produces the "ground truth" artifacts that the ExaModels-based
# companion package (DecisionRulesExa.jl) uses to verify that its policy,
# scenario indexing, and units reproduce this repository's behavior exactly:
#
#   1. The paired scenarios' inflow VALUE arrays, reconstructed from the
#      seeded protocol (`paired_scenario_indices` in load_hydropowermodels.jl)
#      and `build_hydropowermodels`'s `uncertainty_samples` — the same
#      construction as `eval_paired_tsddr.jl`.
#   2. The OPEN-LOOP policy target trajectories
#
#         x̂_0 = x_0,   x̂_t = π_θ(w_t, x̂_{t-1}),
#
#      computed WITHOUT any solver. In strict mode the stage subproblem enforces
#      the hard equality `reservoir_out == x̂_t`, so the closed-loop realized
#      state equals the target up to solver tolerance (~1e-8): the open-loop
#      recursion doubles as the closed-loop state-path reference AND the
#      policy-parity reference.
#   3. Three deterministic probe evaluations of the policy (single call after
#      `Flux.reset!`), which test weight loading independently of any
#      recurrent-state threading across stages.
#   4. The policy's frozen hydro metadata (bounds, upstream contribution, K),
#      to validate the companion package's bounds construction.
#
# No subproblem is ever solved — only built (needed for `uncertainty_samples`
# and `hydro_meta`, whose `K` is extracted from the MOF hydro_balance row).
#
# Usage:
#   julia --project -t auto dump_paired_policy_reference.jl [MODEL_PATH]
#
# Environment overrides:
#   DR_NUM_EVAL_STAGES=96
#   DR_NUM_SCENARIOS=500
#   DR_ENCODER_LAYERS=128,128
#   DR_HEAD_LAYERS=
#   DR_CONTEXT=                ""/"none", "phase", or "phase+progress"
#   DR_CONTEXT_HORIZON=126     denominator/horizon used for progress context
#   DR_OUTPUT_TAG=             suffix for paired_policy_reference_<tag>.jld2
#
# Output:
#   bolivia/ACPPowerModel/results/paired_policy_reference.jld2
using DecisionRules
using Flux
using Statistics
using Random
using JuMP
using JLD2
using DelimitedFiles

HydroPowerModels_dir = dirname(@__FILE__)
include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))
include(joinpath(HydroPowerModels_dir, "hydro_reachable_policy.jl"))

# ── Configuration ──────────────────────────────────────────────────────────────
# Default to the stage-wise strict TS-DDR checkpoint evaluated by
# eval_paired_tsddr.jl (the run that nearly matched SDDP on paired scenarios).
default_model_path = joinpath(
    HydroPowerModels_dir,
    "bolivia",
    "ACPPowerModel",
    "models",
    "bolivia-ACPPowerModel-h126-r96-subproblems-strict-2026-07-01T09:41:53.026.jld2",
)
model_path = length(ARGS) >= 1 ? ARGS[1] : default_model_path
num_eval_stages = parse(Int, get(ENV, "DR_NUM_EVAL_STAGES", "96"))
num_scenarios = parse(Int, get(ENV, "DR_NUM_SCENARIOS", "500"))
output_tag = strip(get(ENV, "DR_OUTPUT_TAG", ""))

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
println("Paired Policy Reference Dump (no solves)")
println("  Model:      $model_path")
println("  Stages:     $num_eval_stages")
println("  Scenarios:  $num_scenarios")
println("  Layers:     $layers")
println("  Head:       $head_layers")
println("  Context:    $(isempty(context_mode) ? "none" : context_mode)")
isempty(output_tag) || println("  Output tag: $output_tag")
println("=" ^ 60)

# ── Build strict subproblems (structure only — no optimizer, no solves) ────────
# Built exactly like eval_paired_tsddr.jl (num_stages=96, strict=true) so that
# `uncertainty_samples` and `hydro_meta` (including K parsed from the MOF
# hydro_balance coefficient) are bit-identical to the ground-truth evaluation.
# `optimizer=nothing` skips `set_optimizer` — safe because nothing is solved.
case_name = "bolivia"
formulation = "ACPPowerModel"
formulation_file = formulation * ".mof.json"

subproblems, state_params_in, state_params_out, uncertainty_samples,
    initial_state, max_volume, hydro_meta = build_hydropowermodels(
    joinpath(HydroPowerModels_dir, case_name),
    formulation_file;
    num_stages=num_eval_stages,
    optimizer=nothing,
    strict=true,
)

num_hydro = length(initial_state)
nCen = length(uncertainty_samples[1])
println("nHyd=$num_hydro, nCen=$nCen, K=$(hydro_meta.K)")

# Paired scenario indices: pure function of the protocol seed (entry [t, s]
# indexes `uncertainty_samples[t][ω]`; see load_hydropowermodels.jl). Fixed
# 126-row shape; this dump uses rows 1:num_eval_stages.
@assert num_eval_stages <= PAIRED_NUM_STAGES
all_indices = paired_scenario_indices(num_scenarios, nCen)
println("Paired protocol: seed=$(PAIRED_SCENARIO_SEED), rows 1:$(num_eval_stages) of $(PAIRED_NUM_STAGES)×$(num_scenarios), nCen=$nCen")

# ── Build policy and load checkpoint weights ───────────────────────────────────
# Same construction as eval_paired_tsddr.jl lines 86-91: encoder+combiner
# weights are restored; hydro bounds come from hydro_meta.
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

# ── Reconstruct inflow VALUE arrays for the paired scenarios ───────────────────
# inflow_values[t, r, s] is the inflow of hydro unit r at stage t under paired
# scenario s, i.e. the second element of `uncertainty_samples[t][idx[t,s]][r]`.
# This is exactly the value fed to both the policy and the subproblem parameter
# in eval_paired_tsddr.jl.
inflow_values = zeros(Float64, num_eval_stages, num_hydro, num_scenarios)
for s in 1:num_scenarios
    for t in 1:num_eval_stages
        w_t = uncertainty_samples[t][all_indices[t, s]]
        for (r, (_, val)) in enumerate(w_t)
            inflow_values[t, r, s] = val
        end
    end
end

# ── Open-loop policy target trajectories ───────────────────────────────────────
# Recursion (no solver):
#
#     x̂_0 = x_0,   x̂_t = π_θ(Float32.(w_t), Float32.(x̂_{t-1})).
#
# In strict mode the closed-loop rollout of eval_paired_tsddr.jl realizes
# x_t = x̂_t exactly (hard equality), so this trajectory is the reference for
# both the policy outputs and the realized reservoir path (up to the Ipopt
# solver tolerance with which the closed loop reads back the realized state).
xhat_trajectories = zeros(Float64, num_eval_stages, num_hydro, num_scenarios)
for s in 1:num_scenarios
    # Fresh recurrent state per scenario, matching the closed-loop evaluation.
    Flux.reset!(models)
    state = Float64.(initial_state)
    for t in 1:num_eval_stages
        w_vals = Float32.(inflow_values[t, :, s])
        # π_θ(w_t, x̂_{t-1}) — identical input construction to eval_paired_tsddr.jl
        x_hat = models(vcat(w_vals, Float32.(state)))
        xhat_trajectories[t, :, s] = Float64.(x_hat)
        # Open-loop: feed the target back as the next previous state.
        state = Float64.(x_hat)
    end
    if s % 20 == 0 || s == num_scenarios
        println("  open-loop trajectories: [$s/$num_scenarios]")
    end
end

# ── Deterministic probe evaluations ────────────────────────────────────────────
# Each probe is a SINGLE policy call after Flux.reset!, i.e. from the zero
# initial recurrent state. These test pure weight parity: any cross-package
# difference in recurrent-state threading across stages cannot affect them.
#
#   probe 1: w = stage-1 inflows of paired scenario 1,  x = initial_state
#   probe 2: w = zeros,                                 x = initial_state
#   probe 3: w = mean over ω of stage-1 inflows,        x = initial_state
w_probe_1 = Float32.(inflow_values[1, :, 1])
w_probe_2 = zeros(Float32, num_hydro)
w_probe_3 = Float32.([
    mean(uncertainty_samples[1][ω][r][2] for ω in 1:nCen) for r in 1:num_hydro
])
x_probe = Float32.(initial_state)

probe_inputs = zeros(Float32, 2 * num_hydro, 3)
probe_inputs[:, 1] = vcat(w_probe_1, x_probe)
probe_inputs[:, 2] = vcat(w_probe_2, x_probe)
probe_inputs[:, 3] = vcat(w_probe_3, x_probe)

probe_outputs = zeros(Float64, num_hydro, 3)
for p in 1:3
    # Reset before every probe: single call from the zero recurrent state.
    Flux.reset!(models)
    probe_outputs[:, p] = Float64.(models(probe_inputs[:, p]))
end
println("Probe outputs computed.")

# ── Save reference JLD2 ────────────────────────────────────────────────────────
out_dir = joinpath(HydroPowerModels_dir, case_name, formulation, "results")
mkpath(out_dir)
out_suffix = isempty(output_tag) ? "" : "_" * output_tag
out_file = joinpath(out_dir, "paired_policy_reference$(out_suffix).jld2")
metadata_policy = models isa ContextualPolicy ? models.policy : models
jldsave(out_file;
    # Scenario data (authoritative values for the cross-package rollout)
    inflow_values=inflow_values,                 # [T × nHyd × S]
    scenario_indices=all_indices[1:num_eval_stages, 1:num_scenarios],
    # Policy references
    xhat_trajectories=xhat_trajectories,         # [T × nHyd × S] open-loop targets
    probe_inputs=probe_inputs,                   # [2 nHyd × 3] Float32
    probe_outputs=probe_outputs,                 # [nHyd × 3]
    # System data for metadata cross-checks
    initial_state=Float64.(initial_state),
    max_volume=Float64.(max_volume),
    # Frozen hydro metadata exactly as used by the policy forward pass
    policy_K=metadata_policy.K,
    policy_min_vol=Float64.(metadata_policy.min_vol),
    policy_max_vol=Float64.(metadata_policy.max_vol),
    policy_min_turn=Float64.(metadata_policy.min_turn),
    policy_max_turn=Float64.(metadata_policy.max_turn),
    policy_upstream_max=Float64.(metadata_policy.upstream_max),
    # Provenance
    model_path=model_path,
    num_eval_stages=num_eval_stages,
    num_scenarios=num_scenarios,
    encoder_layers=layers,
    head_layers=head_layers,
    context_mode=isempty(context_mode) ? "none" : context_mode,
    context_period=context_period,
    context_horizon=context_horizon,
    n_context=n_context,
)
println("Saved: $out_file")
