# Plot transient evaluation study for Atlas policy rollouts.
#
# Expected input directory structure:
#   transient_eval_results/<policy_name>/sim_*.jld2
#
# Produces:
# 1) perturbation vs time-to-equilibrium scatter
# 2) rollout state-shift L2 vs time for all simulations, color-coded by perturbation
#
# Usage:
#   julia --project=. plot_transient_eval_results.jl
#   julia --project=. plot_transient_eval_results.jl <policy_name_or_model_path>

using JLD2
using Statistics
using LinearAlgebra
using Plots
using Printf

Atlas_dir = dirname(@__FILE__)
results_root = joinpath(Atlas_dir, "transient_eval_results")

function parse_env(T::Type, key::String, default)
    if haskey(ENV, key) && !isempty(ENV[key])
        return parse(T, ENV[key])
    end
    return default
end

function resolve_policy_name(policy_hint::Union{Nothing, String})
    if !isnothing(policy_hint) && !isempty(policy_hint)
        # If a full model path is given, use stem as policy folder name.
        if endswith(policy_hint, ".jld2")
            return splitext(basename(policy_hint))[1]
        end
        return basename(policy_hint)
    end

    # Fallback to most recently modified policy result folder.
    isdir(results_root) || error("No transient_eval_results directory found at $results_root")
    dirs = filter(d -> isdir(joinpath(results_root, d)), readdir(results_root))
    isempty(dirs) && error("No policy subfolders found in $results_root")
    full = [joinpath(results_root, d) for d in dirs]
    return dirs[argmax([mtime(d) for d in full])]
end

function time_to_equilibrium(state_change_l2::Vector{Float64}, h::Float64; stable_steps::Int=50, delta_tol::Float64=1e-3)
    if length(state_change_l2) < stable_steps
        return NaN, nothing
    end
    for start_step in 1:(length(state_change_l2) - stable_steps + 1)
        if all(@view(state_change_l2[start_step:start_step + stable_steps - 1]) .<= delta_tol)
            # Declare equilibrium at the end of the stable window.
            eq_step = start_step + stable_steps
            return (eq_step - 1) * h, eq_step
        end
    end
    return NaN, nothing
end

function color_for_perturbation(value::Float64, max_abs::Float64, gradient)
    alpha = max_abs > 0 ? clamp((value + max_abs) / (2 * max_abs), 0.0, 1.0) : 0.5
    return get(gradient, alpha)
end

function main()
    policy_hint = !isempty(ARGS) ? ARGS[1] : get(ENV, "ATLAS_POLICY_PATH", "")
    policy_name = resolve_policy_name(policy_hint)
    result_dir = joinpath(results_root, policy_name)
    isdir(result_dir) || error("Result directory does not exist: $result_dir")

    sim_files = sort(filter(
        f -> endswith(f, ".jld2") && startswith(basename(f), "sim_"),
        readdir(result_dir; join=true),
    ))
    isempty(sim_files) && error("No sim_*.jld2 files found in $result_dir")

    stable_steps = parse_env(Int, "ATLAS_EQUILIBRIUM_STEPS", 50)
    delta_tol = parse_env(Float64, "ATLAS_EQUILIBRIUM_DELTA_TOL", 1e-3)

    perturbations = Float64[]
    equilibrium_times = Float64[]
    equilibrium_steps = Union{Nothing, Int}[]
    rollout_series = Vector{Vector{Float64}}()
    time_series = Vector{Vector{Float64}}()
    task_ids = Int[]
    failed_files = String[]
    h_ref = NaN
    N_ref = 0

    for file in sim_files
        data = JLD2.load(file)
        status = get(data, "status", "success")
        if status != "success"
            push!(failed_files, file)
            continue
        end

        perturbation_value = Float64(data["perturbation_value"])
        h = Float64(data["h"])
        N = Int(data["N"])
        state_change_l2 = Vector{Float64}(data["state_change_l2"])
        rollout_state_shift_l2 = Vector{Float64}(data["rollout_state_shift_l2"])
        time = haskey(data, "time") ? Vector{Float64}(data["time"]) : collect(0:N-1) .* h
        task_id = Int(data["task_id"])

        eq_time, eq_step = time_to_equilibrium(state_change_l2, h; stable_steps=stable_steps, delta_tol=delta_tol)

        push!(perturbations, perturbation_value)
        push!(equilibrium_times, eq_time)
        push!(equilibrium_steps, eq_step)
        push!(rollout_series, rollout_state_shift_l2)
        push!(time_series, time)
        push!(task_ids, task_id)
        h_ref = h
        N_ref = N
    end

    isempty(perturbations) && error("No successful simulation files found in $result_dir")

    # Sort by perturbation value for coherent plotting and summary.
    order = sortperm(perturbations)
    perturbations = perturbations[order]
    equilibrium_times = equilibrium_times[order]
    equilibrium_steps = equilibrium_steps[order]
    rollout_series = rollout_series[order]
    time_series = time_series[order]
    task_ids = task_ids[order]

    max_abs_pert = maximum(abs.(perturbations))
    color_grad = cgrad([:blue, :white, :red])

    reached = findall(!isnan, equilibrium_times)
    unreached = findall(isnan, equilibrium_times)
    fallback_time = N_ref > 0 ? (N_ref - 1) * h_ref : maximum(vcat(time_series...))

    p1 = plot(
        title = "Perturbation vs Time To Equilibrium",
        xlabel = "Signed Perturbation Magnitude",
        ylabel = "Time To Equilibrium (s)",
        legend = :top,
        grid = true,
    )
    if !isempty(reached)
        scatter!(
            p1,
            perturbations[reached],
            equilibrium_times[reached];
            marker_z = perturbations[reached],
            color = color_grad,
            clims = (-max_abs_pert, max_abs_pert),
            ms = 8,
            label = "Reached equilibrium",
            colorbar_title = "Perturbation",
        )
    end
    if !isempty(unreached)
        scatter!(
            p1,
            perturbations[unreached],
            fill(fallback_time, length(unreached));
            marker = :x,
            ms = 8,
            color = :black,
            label = "Not reached by horizon",
        )
    end

    p2 = plot(
        title = "Rollout State Shift L2 vs Time",
        xlabel = "Time (s)",
        ylabel = "||x_t - x_ref||\u2082",
        legend = false,
        grid = true,
    )
    for i in eachindex(perturbations)
        perturb = perturbations[i]
        t = time_series[i]
        shift = rollout_series[i]
        c = color_for_perturbation(perturb, max_abs_pert, color_grad)
        plot!(
            p2,
            t,
            shift;
            color = c,
            lw = 2,
            alpha = 0.9,
        )
    end

    # Add a compact color scale legend.
    p_scale = scatter(
        [0.0, 1.0],
        [0.0, 0.0];
        marker_z = [-max_abs_pert, max_abs_pert],
        color = color_grad,
        clims = (-max_abs_pert, max_abs_pert),
        markersize = 0.001,
        markerstrokewidth = 0,
        legend = false,
        colorbar_title = "Perturbation",
        framestyle = :none,
        xshowaxis = false,
        yshowaxis = false,
    )

    combined = plot(p1, p2, p_scale; layout = @layout([a; b; c{0.06h}]), size = (1200, 1100))

    plots_dir = joinpath(result_dir, "plots")
    mkpath(plots_dir)
    plot_path = joinpath(plots_dir, "transient_student_study_plots.png")
    savefig(combined, plot_path)

    summary_path = joinpath(result_dir, "transient_equilibrium_summary.tsv")
    open(summary_path, "w") do io
        println(io, "task_id\tperturbation\tequilibrium_time_s\tequilibrium_step")
        for i in eachindex(perturbations)
            eq_step = isnothing(equilibrium_steps[i]) ? "NA" : string(equilibrium_steps[i])
            eq_time = isnan(equilibrium_times[i]) ? "NA" : @sprintf("%.6f", equilibrium_times[i])
            println(io, "$(task_ids[i])\t$(perturbations[i])\t$(eq_time)\t$(eq_step)")
        end
    end

    println("Saved plots to: $plot_path")
    println("Saved equilibrium summary to: $summary_path")
    println("Successful simulations: $(length(perturbations))")
    println("Failed simulations skipped: $(length(failed_files))")
    if !isempty(failed_files)
        println("Failed files:")
        for f in failed_files
            println("  - $f")
        end
    end
    println("Equilibrium definition: ||x_{t+1} - x_t||₂ <= $delta_tol for $stable_steps consecutive steps.")
end

main()
