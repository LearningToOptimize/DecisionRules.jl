# compare_hydro_results.jl
#
# Pull training histories from W&B and generate comparison plots for the docs.
# Saves plots to ../../docs/src/assets/ and prints a summary table.
#
# Usage:
#   julia --project compare_hydro_results.jl [--run-names name1,name2,name3]
#
# By default, uses the 3 most recent "running" or "finished" runs in the RL project
# that match the three training methods.

using Plots
using Statistics

import Wandb
const wb = Wandb.wandb
const PC = parentmodule(typeof(wb))

const DOCS_ASSETS = joinpath(@__DIR__, "..", "..", "docs", "src", "assets")
mkpath(DOCS_ASSETS)

api = wb.Api()
all_runs = api.runs("RL", order="-created_at")

methods_wanted = ["deterministic_equivalent", "subproblems", "multiple_shooting"]
method_labels = Dict(
    "deterministic_equivalent" => "Deterministic Equivalent",
    "subproblems" => "Stage-wise Subproblems",
    "multiple_shooting" => "Multiple Shooting (w=12)",
)
method_colors = Dict(
    "deterministic_equivalent" => :blue,
    "subproblems" => :red,
    "multiple_shooting" => :green,
)

runs = Dict{String,Any}()
for i in 0:29
    try
        r = all_runs[i]
        method = PC.pyconvert(String, get(r.config, "training_method", "?"))
        state = PC.pyconvert(String, r.state)
        if method in methods_wanted && !haskey(runs, method) && state in ("running", "finished", "crashed")
            runs[method] = r
        end
    catch
        break
    end
    length(runs) == 3 && break
end

println("Using runs:")
for (m, r) in runs
    println("  $(method_labels[m]): $(PC.pyconvert(String, r.name)) ($(PC.pyconvert(String, r.state)))")
end

function get_history(r, metric)
    keys_list = PC.pylist([metric])
    hist = r.scan_history(keys=keys_list)
    vals = Float64[]
    for row in hist
        v = try PC.pyconvert(Float64, get(row, metric, nothing)) catch; nothing end
        !isnothing(v) && push!(vals, v)
    end
    return vals
end

# ── Plot 1: Training convergence (in-sample loss) ────────────────────────────

plt1 = plot(; xlabel="Iteration", ylabel="Operational Cost (no deficit)",
    title="Training Convergence", legend=:topright)
for m in methods_wanted
    haskey(runs, m) || continue
    vals = get_history(runs[m], "metrics/loss")
    isempty(vals) && continue
    plot!(plt1, 1:length(vals), vals; label=method_labels[m], color=method_colors[m], alpha=0.7)
end
savefig(plt1, joinpath(DOCS_ASSETS, "hydro_training_convergence.png"))
println("Saved hydro_training_convergence.png")

# ── Plot 2: Out-of-sample rollout ────────────────────────────────────────────

plt2 = plot(; xlabel="Iteration", ylabel="Rollout Cost (no deficit)",
    title="Out-of-Sample Rollout", legend=:topright)
for m in methods_wanted
    haskey(runs, m) || continue
    vals = get_history(runs[m], "metrics/rollout_objective_no_deficit")
    isempty(vals) && continue
    eval_every = try
        PC.pyconvert(Int, get(runs[m].config, "eval_every", 25))
    catch
        25
    end
    iters = eval_every .* (1:length(vals))
    plot!(plt2, iters, vals; label=method_labels[m], color=method_colors[m],
        marker=:circle, markersize=3)
end
savefig(plt2, joinpath(DOCS_ASSETS, "hydro_cost_comparison.png"))
println("Saved hydro_cost_comparison.png")

# ── Plot 3: Target violation share ───────────────────────────────────────────

plt3 = plot(; xlabel="Iteration", ylabel="Violation Share",
    title="Target Violation Share", legend=:topright, ylims=(0, 0.3))
for m in methods_wanted
    haskey(runs, m) || continue
    vals = get_history(runs[m], "metrics/rollout_target_violation_share")
    isempty(vals) && continue
    eval_every = try
        PC.pyconvert(Int, get(runs[m].config, "eval_every", 25))
    catch
        25
    end
    iters = eval_every .* (1:length(vals))
    plot!(plt3, iters, vals; label=method_labels[m], color=method_colors[m],
        marker=:circle, markersize=3)
end
savefig(plt3, joinpath(DOCS_ASSETS, "hydro_violation_share.png"))
println("Saved hydro_violation_share.png")

# ── Summary table ────────────────────────────────────────────────────────────

println("\n" * "="^80)
println("Summary Table")
println("="^80)
println(rpad("Method", 30), rpad("Last Loss", 15), rpad("Rollout", 15),
    rpad("Violation", 12), rpad("Steps", 8))
println("-"^80)
for m in methods_wanted
    haskey(runs, m) || continue
    r = runs[m]
    summ = r.summary
    loss = try round(PC.pyconvert(Float64, get(summ, "metrics/loss", nothing)); digits=0) catch; nothing end
    rollout = try round(PC.pyconvert(Float64, get(summ, "metrics/rollout_objective_no_deficit", nothing)); digits=0) catch; nothing end
    violation = try round(PC.pyconvert(Float64, get(summ, "metrics/rollout_target_violation_share", nothing)); digits=4) catch; nothing end
    steps = try PC.pyconvert(Int, get(summ, "_step", nothing)) catch; nothing end
    println(rpad(method_labels[m], 30),
        rpad(isnothing(loss) ? "-" : string(loss), 15),
        rpad(isnothing(rollout) ? "-" : string(rollout), 15),
        rpad(isnothing(violation) ? "-" : string(violation), 12),
        rpad(isnothing(steps) ? "-" : string(steps), 8))
end
println("="^80)
