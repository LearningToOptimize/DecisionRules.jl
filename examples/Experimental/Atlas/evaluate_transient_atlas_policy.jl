# Evaluate a trained Atlas policy under single-shot transient perturbations.
#
# Designed for SLURM array execution:
# - each task runs one rollout of length N
# - perturbation is applied only at stage 1
# - perturbation sign/magnitude are mapped from array task id
#
# Results are saved to:
#   transient_eval_results/<policy_name>/sim_XXX_<sign>_mag_<value>.jld2
#
# Usage examples:
#   julia --project=. evaluate_transient_atlas_policy.jl
#   ATLAS_POLICY_PATH=./models/atlas-balancing-deteq-...jld2 \
#   SLURM_ARRAY_TASK_ID=4 julia --project=. evaluate_transient_atlas_policy.jl

using Flux
using DecisionRules
using LinearAlgebra
using JuMP
import Ipopt, HSL_jll
using JLD2
using Dates
using Printf

Atlas_dir = dirname(@__FILE__)
include(joinpath(Atlas_dir, "build_atlas_problem.jl"))

function parse_env(T::Type, key::String, default)
    if haskey(ENV, key) && !isempty(ENV[key])
        return parse(T, ENV[key])
    end
    return default
end

function resolve_policy_path(policy_hint::Union{Nothing, String}, model_dir::String)
    if !isnothing(policy_hint) && !isempty(policy_hint)
        candidates = String[policy_hint]
        push!(candidates, joinpath(model_dir, policy_hint))
        if !endswith(policy_hint, ".jld2")
            push!(candidates, policy_hint * ".jld2")
            push!(candidates, joinpath(model_dir, policy_hint * ".jld2"))
        end
        for path in candidates
            isfile(path) && return path
        end
        error("Could not resolve ATLAS_POLICY_PATH: $policy_hint")
    end

    model_files = filter(f -> endswith(f, ".jld2") && startswith(f, "atlas-balancing"), readdir(model_dir))
    isempty(model_files) && error("No .jld2 models found in $model_dir and no ATLAS_POLICY_PATH provided.")
    model_files_full = [joinpath(model_dir, f) for f in model_files]
    return model_files_full[argmax([mtime(f) for f in model_files_full])]
end

function perturbation_from_task(task_id::Int, n_levels::Int, min_mag::Float64, max_mag::Float64)
    n_levels < 1 && error("ATLAS_TRANSIENT_N_LEVELS must be >= 1.")
    (min_mag <= 0 || max_mag < min_mag) &&
        error("Expected 0 < min_mag <= max_mag, got min_mag=$min_mag, max_mag=$max_mag.")

    magnitudes = collect(range(min_mag, max_mag, length=n_levels))
    n_tasks = 2 * n_levels
    (task_id < 1 || task_id > n_tasks) &&
        error("Task id $task_id is out of range 1:$n_tasks for n_levels=$n_levels.")

    mag_idx = fld(task_id - 1, 2) + 1
    sign = isodd(task_id) ? -1.0 : 1.0
    mag = magnitudes[mag_idx]
    return sign * mag, mag, sign, n_tasks
end

function build_fixed_perturbation_sample(uncertainty_samples, perturbation_value::Float64)
    # Start from DecisionRules.sample so tuple types match simulate_multistage expectations.
    perturbation_sample = DecisionRules.sample(uncertainty_samples)
    for t in eachindex(perturbation_sample)
        for i in eachindex(perturbation_sample[t])
            var = perturbation_sample[t][i][1]
            perturbation_sample[t][i] = (var, 0.0)
        end
    end

    if !isempty(perturbation_sample) && !isempty(perturbation_sample[1])
        first_var = perturbation_sample[1][1][1]
        perturbation_sample[1][1] = (first_var, perturbation_value)
    end

    return perturbation_sample
end

function main()
    model_dir = joinpath(Atlas_dir, "models")
    output_root = joinpath(Atlas_dir, "transient_eval_results")
    mkpath(output_root)

    # CLI arg takes precedence; otherwise use env var.
    policy_hint = !isempty(ARGS) ? ARGS[1] : get(ENV, "ATLAS_POLICY_PATH", "")
    policy_path = resolve_policy_path(policy_hint, model_dir)
    policy_name = splitext(basename(policy_path))[1]
    result_dir = joinpath(output_root, policy_name)
    mkpath(result_dir)

    N = parse_env(Int, "ATLAS_TRANSIENT_HORIZON", 300)
    h = parse_env(Float64, "ATLAS_TRANSIENT_TIMESTEP", 0.01)
    n_levels = parse_env(Int, "ATLAS_TRANSIENT_N_LEVELS", 10)
    max_mag = parse_env(Float64, "ATLAS_TRANSIENT_MAX_MAG", 1.0)
    min_mag_default = max_mag / max(n_levels, 1)
    min_mag = parse_env(Float64, "ATLAS_TRANSIENT_MIN_MAG", min_mag_default)
    task_id = parse_env(Int, "SLURM_ARRAY_TASK_ID", parse_env(Int, "TASK_ID", 1))

    atlas = Atlas()
    perturbation_idx_default = atlas.nq + 5
    perturbation_idx = parse_env(Int, "ATLAS_TRANSIENT_PERTURBATION_INDEX", perturbation_idx_default)
    (perturbation_idx < 1 || perturbation_idx > atlas.nx) &&
        error("ATLAS_TRANSIENT_PERTURBATION_INDEX=$perturbation_idx is outside 1:$(atlas.nx).")

    perturbation_value, perturbation_mag, perturbation_sign, n_tasks =
        perturbation_from_task(task_id, n_levels, min_mag, max_mag)

    println("Transient eval task:")
    println("  policy: $policy_name")
    println("  task: $task_id / $n_tasks")
    println("  perturbation value: $perturbation_value")
    println("  perturbation index: $perturbation_idx")
    println("  horizon: N=$N, h=$h")

    subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples,
    X_vars, U_vars, x_ref, u_ref, _ = build_atlas_subproblems(;
        atlas = atlas,
        N = N,
        h = h,
        perturbation_scale = 0.0,
        perturbation_frequency = N, # stage 1 only for N-step rollout
        perturbation_indices = [perturbation_idx],
        num_scenarios = 1,
    )

    nx = atlas.nx
    nu = atlas.nu
    n_uncertainties = length(uncertainty_samples[1])
    layers = Int64[64, 64]
    activation = sigmoid
    models = state_conditioned_policy(n_uncertainties, nx, nx, layers;
        activation = activation, encoder_type = Flux.LSTM)

    model_data = JLD2.load(policy_path)
    haskey(model_data, "model_state") || error("Model file does not contain `model_state`: $policy_path")
    Flux.loadmodel!(models, normalize_recur_state(model_data["model_state"]))
    Flux.reset!(models)

    perturbation_sample = build_fixed_perturbation_sample(uncertainty_samples, perturbation_value)
    perturbation_series = zeros(Float64, N - 1)
    perturbation_series[1] = perturbation_value

    status = "success"
    error_message = ""
    objective_value = NaN
    states = zeros(Float64, nx, N)
    actions = zeros(Float64, nu, N - 1)
    rollout_state_shift_l2 = fill(NaN, N)
    state_change_l2 = fill(NaN, N - 1)
    time = collect(0:N-1) .* h

    try
        objective_value = simulate_multistage(
            subproblems,
            state_params_in,
            state_params_out,
            initial_state,
            perturbation_sample,
            models,
        )

        states[:, 1] .= initial_state
        for t in 1:N-1
            states[:, t + 1] .= value.(X_vars[t])
            actions[:, t] .= value.(U_vars[t])
        end
        rollout_state_shift_l2 .= [norm(states[:, t] .- x_ref) for t in 1:N]
        state_change_l2 .= [norm(states[:, t + 1] .- states[:, t]) for t in 1:N-1]
    catch err
        status = "failure"
        error_message = sprint(showerror, err, catch_backtrace())
        @warn "Transient evaluation failed." exception=(err, catch_backtrace())
    end

    mag_token = replace(@sprintf("%.4f", perturbation_mag), "." => "p")
    sign_token = perturbation_sign > 0 ? "pos" : "neg"
    result_file = joinpath(
        result_dir,
        @sprintf("sim_%03d_%s_mag_%s.jld2", task_id, sign_token, mag_token),
    )

    jldsave(
        result_file;
        policy_name,
        policy_path,
        status,
        error_message,
        task_id,
        n_tasks,
        N,
        h,
        perturbation_idx,
        perturbation_value,
        perturbation_mag,
        perturbation_sign,
        objective_value,
        states,
        actions,
        rollout_state_shift_l2,
        state_change_l2,
        perturbation_series,
        time,
        x_ref,
        u_ref,
    )

    println("Saved transient result: $result_file")
    println("Status: $status")
    if status != "success"
        error(error_message)
    end
end

main()
