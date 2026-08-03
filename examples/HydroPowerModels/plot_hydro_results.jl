#!/usr/bin/env julia

# Publication figures for the Bolivia hydro case study.
#
# Every figure is generated from the compact evidence committed under
# `results/`, so `julia --project plot_hydro_results.jl` reproduces the
# published assets deterministically without re-running training, evaluation, or
# anything on a GPU.
#
# Produces, in `docs/src/assets/`:
#
#   hydro_training_history.png       the from-scratch training run
#   hydro_cost_distributions.png     absolute cost densities, both policies
#   hydro_paired_differences.png     paired TS-DDR - SDDP differences, zero marked
#   hydro_stagewise_physical.png     where the cost difference is actually incurred
#
# ── The training figure, and the mistakes it is built to avoid ────────────────
#
# Three quantities live on this run and they are NOT interchangeable:
#
#   * the STOCHASTIC TRAINING LOSS — one noisy sample of the 126-stage
#     deterministic-equivalent objective per update, over `nt` freshly sampled
#     inflow trajectories;
#   * the 126-stage TRAINING OBJECTIVE per epoch — the same quantity averaged
#     over an epoch's samples;
#   * the 96-stage FIXED-PANEL ROLLOUT — the ten common-random-number scenarios
#     on which checkpoints are actually SELECTED.
#
# They differ in horizon (126 vs 96), in sampling (fresh vs fixed), and in level
# by tens of thousands. Plotting them on one axis invites exactly the error of
# reading a drop in a noisy training sample as progress on the selection metric,
# so they are drawn in SEPARATE PANELS with their own axes.
#
# The raw training loss is drawn faintly and smoothed. The smoothing window is a
# fixed number of SAMPLED TRAJECTORIES, not of updates: `nt` changes between
# stages (16 -> 24), so a fixed-update window would average four thousand
# trajectories in one stage and six thousand in another and the curve's noise
# level would change for a reason that has nothing to do with learning. The
# smoothing RESETS at each restart boundary — a restart re-initialises the
# optimizer and the learning-rate phase, and carrying an average across it would
# invent a transition that did not happen.
#
# The x-axis is ACTIVE WALL TIME, cumulative across the selected lineage. Update
# count would hide that the stages have different per-update costs.
#
# Only the SELECTED lineage is drawn (coldB -> C1 -> C3). The rejected C2 branch
# is deliberately absent from the curve; it is reported in the text and in
# `results/rejected_branch_C2.csv` as discarded search work. The final paired-500
# result is NOT a point on this figure: it is a different, larger evaluation and
# appears in its own figures.

using CSV, DataFrames
using JSON
using Plots
using Printf
using Statistics

const HYDRO_DIR = @__DIR__
const RESULTS = joinpath(HYDRO_DIR, "results")
const ASSETS = normpath(joinpath(HYDRO_DIR, "..", "..", "docs", "src", "assets"))
mkpath(ASSETS)

# Okabe-Ito, a validated colour-vision-deficiency-safe set. The assignment
# follows the ENTITY and never changes between figures: SDDP is green,
# TS-DDR blue, and neutral ink is grey.
const C_SDDP = colorant"#009E73"
const C_TSDDR = colorant"#0072B2"
const C_RAW = colorant"#0072B2"
const C_INK = colorant"#4D4D4D"
const C_ACCENT = colorant"#D55E00"

# Smoothing window for the stochastic training loss, in SAMPLED TRAJECTORIES.
# 8,000 is ~20 updates at nt = 16 and ~13 at nt = 24: enough to see through the
# sample noise, short enough to keep a real change visible.
const SMOOTH_TRAJECTORIES = 8_000

"""
    gaussian_kde(samples; npoints=512) -> (xs, density)

Gaussian kernel density estimate with Silverman's rule-of-thumb bandwidth

```math
h = 0.9 \\, \\min(\\hat\\sigma, \\mathrm{IQR}/1.34) \\, n^{-1/5},
```

evaluated on an even grid spanning the sample range padded by three bandwidths
so the tails fall smoothly to zero inside the plot.
"""
function gaussian_kde(samples::AbstractVector{<:Real}; npoints::Int = 512)
    n = length(samples)
    sigma = std(samples)
    iqr = quantile(samples, 0.75) - quantile(samples, 0.25)
    h = max(0.9 * min(sigma, iqr / 1.34) * n^(-1 / 5), 1e-9 * max(abs(mean(samples)), 1.0))
    xs = range(minimum(samples) - 3h, maximum(samples) + 3h; length = npoints)
    dens = [sum(exp(-0.5 * ((x - s) / h)^2) for s in samples) / (n * h * sqrt(2pi)) for x in xs]
    return collect(xs), dens
end

"""
    trajectory_smooth(times, values, nt, window) -> (times, smoothed)

Trailing mean of `values` over the most recent `window` SAMPLED TRAJECTORIES,
where update `i` contributes `nt[i]` trajectories.

Returned aligned with `times`. The first points average fewer trajectories than
the window, which is unavoidable and visible: the curve simply starts where the
data do.
"""
function trajectory_smooth(times, values, nt, window::Real)
    n = length(values)
    out = similar(values, Float64)
    for i in 1:n
        total = 0.0
        acc = 0.0
        weight = 0.0
        j = i
        while j >= 1 && total < window
            total += nt[j]
            acc += values[j] * nt[j]
            weight += nt[j]
            j -= 1
        end
        out[i] = acc / weight
    end
    return times, out
end

"""
    load_training() -> (DataFrame, Dict)

The selected lineage's training history and its time accounting.

`_runtime` in the raw history is per-STAGE. The lineage's active time is
cumulative, so each stage is offset by the total active time of the stages
before it, taken from `lineage_accounting.json` — the same numbers the reported
time-to-policy uses, rather than a second derivation of them.
"""
function load_training()
    history = CSV.read(joinpath(RESULTS, "training_history.csv"), DataFrame)
    accounting = JSON.parsefile(joinpath(RESULTS, "lineage_accounting.json"))
    selected = [s for s in accounting["selected_ancestry"] if s != "random initialisation"]
    per_stage = accounting["components"]["per_stage"]

    offset = 0.0
    frames = DataFrame[]
    for stage in selected
        rows = history[history.stage .== stage, :]
        rows = copy(rows)
        rows.active_seconds = rows[!, "_runtime"] .+ offset
        push!(frames, rows)
        offset += per_stage[stage]["active_seconds"]
    end
    return vcat(frames...), accounting
end

# ── Figure 1: the from-scratch training run ───────────────────────────────────

function figure_training()
    history, accounting = load_training()
    selected = [s for s in accounting["selected_ancestry"] if s != "random initialisation"]
    per_stage = accounting["components"]["per_stage"]

    # Restart boundaries: the cumulative active time at which each stage ended.
    boundaries = Float64[]
    running = 0.0
    for stage in selected[1:(end - 1)]
        running += per_stage[stage]["active_seconds"]
        push!(boundaries, running / 3600)
    end

    loss = dropmissing(history, "metrics/training_loss")
    panel = dropmissing(history, "metrics/rollout_objective_no_target_penalty")
    epochs = dropmissing(history, "metrics/epoch_objective")

    top = plot(;
        ylabel = "126-stage objective",
        title = "From-scratch TS-DDR training — selected lineage " *
                join(selected, " → "),
        legend = :topright, grid = :y, gridalpha = 0.15,
    )
    # Raw stochastic samples, faint. Smoothing is per STAGE so it resets at each
    # restart, and per trajectory so its noise level does not track nt.
    for stage in selected
        rows = loss[loss.stage .== stage, :]
        isempty(rows) && continue
        hours = rows.active_seconds ./ 3600
        plot!(top, hours, rows[!, "metrics/training_loss"];
              color = C_RAW, alpha = 0.18, linewidth = 1,
              label = stage == first(selected) ? "stochastic training loss (per update)" : "")
        xs, ys = trajectory_smooth(hours, Float64.(rows[!, "metrics/training_loss"]),
                                   Float64.(rows[!, "metrics/num_train_per_batch"]),
                                   SMOOTH_TRAJECTORIES)
        plot!(top, xs, ys; color = C_RAW, linewidth = 2.5,
              label = stage == first(selected) ?
                      "smoothed over $(SMOOTH_TRAJECTORIES) sampled trajectories" : "")
    end
    if !isempty(epochs)
        scatter!(top, epochs.active_seconds ./ 3600, epochs[!, "metrics/epoch_objective"];
                 color = C_INK, markershape = :diamond, markersize = 3,
                 markerstrokewidth = 0, label = "126-stage epoch objective")
    end
    for (i, b) in enumerate(boundaries)
        vline!(top, [b]; color = C_ACCENT, linestyle = :dash, linewidth = 1.2,
               label = i == 1 ? "restart (new stage: optimizer, LR phase, warm-up reset)" : "")
    end

    bottom = plot(;
        xlabel = "active wall time (h)",
        ylabel = "96-stage fixed-panel cost",
        legend = :topright, grid = :y, gridalpha = 0.15,
    )
    hours = panel.active_seconds ./ 3600
    plot!(bottom, hours, panel[!, "metrics/rollout_objective_no_target_penalty"];
          color = C_TSDDR, linewidth = 2, markershape = :circle, markersize = 4,
          markerstrokewidth = 0, label = "fixed 10-scenario panel (selection metric)")
    # Only complete evaluations are selectable; incomplete ones are marked so the
    # curve is not read as if every point were a candidate.
    incomplete = panel[panel[!, "metrics/rollout_n_ok"] .< 10, :]
    isempty(incomplete) || scatter!(bottom,
        incomplete.active_seconds ./ 3600,
        incomplete[!, "metrics/rollout_objective_no_target_penalty"];
        color = C_ACCENT, markershape = :xcross, markersize = 7, markerstrokewidth = 2,
        label = "incomplete evaluation — refused for selection")
    statistics = JSON.parsefile(joinpath(RESULTS, "statistics.json"))["statistics"]
    hline!(bottom, [313331.56404]; color = C_SDDP, linewidth = 2, linestyle = :dash,
           label = "SDDP on the same panel")
    for (i, b) in enumerate(boundaries)
        vline!(bottom, [b]; color = C_ACCENT, linestyle = :dash, linewidth = 1.2, label = "")
    end

    # Stage annotations: nt and the learning-rate band actually used.
    running = 0.0
    for stage in selected
        rows = history[history.stage .== stage, :]
        nt = Int(first(skipmissing(rows[!, "metrics/num_train_per_batch"])))
        lrs = collect(skipmissing(rows[!, "metrics/lr"]))
        mid = (running + per_stage[stage]["active_seconds"] / 2) / 3600
        annotate!(top, mid, maximum(skipmissing(loss[!, "metrics/training_loss"])),
                  text(@sprintf("%s\nnt=%d\nLR %.0e→%.0e", stage, nt, maximum(lrs), minimum(lrs)),
                       7, C_INK, :center))
        running += per_stage[stage]["active_seconds"]
    end

    figure = plot(top, bottom; layout = grid(2, 1; heights = [0.55, 0.45]),
                  size = (1000, 760), left_margin = 8Plots.mm, bottom_margin = 6Plots.mm)
    path = joinpath(ASSETS, "hydro_training_history.png")
    savefig(figure, path)
    println("Saved: $path")
    return nothing
end

# ── Figures 2 and 3: the paired 500-scenario evaluation ───────────────────────

function figure_distributions(paired, statistics)
    tsddr = Float64.(paired.tsddr_cost)
    sddp = Float64.(paired.sddp_cost)

    figure = plot(;
        xlabel = "96-stage true-ACP operating cost (USD)",
        ylabel = "density",
        title = @sprintf("Cost over %d paired inflow scenarios", nrow(paired)),
        legend = :topright, grid = :y, gridalpha = 0.15, size = (1000, 420),
    )
    for (label, costs, colour) in (("SDDP", sddp, C_SDDP), ("TS-DDR (C3)", tsddr, C_TSDDR))
        xs, dens = gaussian_kde(costs)
        plot!(figure, xs, dens; label = @sprintf("%s   mean %.0f, sd %.0f", label,
                                                 mean(costs), std(costs)),
              color = colour, linewidth = 2, fill = (0, 0.25, colour))
        vline!(figure, [mean(costs)]; color = colour, linestyle = :dash,
               linewidth = 1, alpha = 0.8, label = "")
    end
    # The two distributions overlap almost completely; saying so on the figure
    # keeps it from being read as two separated populations.
    annotate!(figure, mean(sddp), 0.0,
              text("the two distributions differ in mean by " *
                   @sprintf("%.0f (%.4f%%), against a spread of ~%.0f",
                            statistics["paired_difference"]["mean"],
                            statistics["paired_difference"]["relative_gap_pct"],
                            std(sddp)),
                   8, C_INK, :bottom))
    path = joinpath(ASSETS, "hydro_cost_distributions.png")
    savefig(figure, path)
    println("Saved: $path")
    return nothing
end

function figure_paired_differences(paired, statistics)
    diffs = Float64.(paired.paired_difference)
    d = statistics["paired_difference"]
    wins = statistics["wins_losses_ties"]["tsddr_wins"]

    figure = plot(;
        xlabel = "paired difference, TS-DDR − SDDP, same scenario (USD)",
        ylabel = "density",
        title = "Paired differences — the comparison the evaluation is designed to make",
        legend = :topright, grid = :y, gridalpha = 0.15, size = (1000, 440),
    )
    xs, dens = gaussian_kde(diffs)
    plot!(figure, xs, dens; color = C_TSDDR, linewidth = 2,
          fill = (0, 0.25, C_TSDDR), label = "")
    # Mass to the LEFT of zero is where TS-DDR is cheaper. Shaded so the reader
    # sees the sign of the result without decoding the axis.
    left = xs .<= 0
    any(left) && plot!(figure, xs[left], dens[left]; color = C_SDDP, linewidth = 0,
                       fill = (0, 0.45, C_SDDP),
                       label = @sprintf("TS-DDR cheaper: %d of %d scenarios",
                                        wins, nrow(paired)))
    vline!(figure, [0.0]; color = C_INK, linewidth = 1.5, label = "zero")
    vline!(figure, [d["mean"]]; color = C_ACCENT, linewidth = 2, linestyle = :dash,
           label = @sprintf("mean +%.1f  (95%% CI [+%.1f, +%.1f])",
                            d["mean"], d["ci95_cost"][1], d["ci95_cost"][2]))
    path = joinpath(ASSETS, "hydro_paired_differences.png")
    savefig(figure, path)
    println("Saved: $path")
    return nothing
end

# ── Figure 4: where the difference is incurred ────────────────────────────────

function figure_stagewise()
    stagewise = CSV.read(joinpath(RESULTS, "stagewise_physical.csv"), DataFrame)
    reported = stagewise[stagewise.stage .<= 96, :]

    top = plot(;
        ylabel = "cumulative cost difference (USD)",
        title = "Where the cost difference is incurred — means over 500 paired scenarios",
        legend = :bottomright, grid = :y, gridalpha = 0.15,
    )
    plot!(top, reported.stage, reported.cum_cost_difference;
          color = C_TSDDR, linewidth = 2.5, label = "cumulative TS-DDR − SDDP")
    hline!(top, [0.0]; color = C_INK, linewidth = 1, label = "")

    middle = plot(; ylabel = "reservoir 2 storage (pu)",
                  legend = :topright, grid = :y, gridalpha = 0.15)
    plot!(middle, reported.stage, reported.sddp_reservoir2;
          color = C_SDDP, linewidth = 2, label = "SDDP")
    plot!(middle, reported.stage, reported.tsddr_reservoir2;
          color = C_TSDDR, linewidth = 2, label = "TS-DDR (C3)")

    bottom = plot(; xlabel = "stage (week)", ylabel = "thermal generation (MW)",
                  legend = :topright, grid = :y, gridalpha = 0.15)
    plot!(bottom, reported.stage, reported.sddp_thermal_MW;
          color = C_SDDP, linewidth = 2, label = "SDDP")
    plot!(bottom, reported.stage, reported.tsddr_thermal_MW;
          color = C_TSDDR, linewidth = 2, label = "TS-DDR (C3)")

    figure = plot(top, middle, bottom; layout = (3, 1), size = (1000, 900),
                  left_margin = 9Plots.mm, bottom_margin = 6Plots.mm)
    path = joinpath(ASSETS, "hydro_stagewise_physical.png")
    savefig(figure, path)
    println("Saved: $path")
    return nothing
end

function main()
    paired = CSV.read(joinpath(RESULTS, "paired_500.csv"), DataFrame)
    statistics = JSON.parsefile(joinpath(RESULTS, "statistics.json"))["statistics"]

    # The published statistics must be the ones these rows imply; a figure drawn
    # from rows that disagree with the reported table would be worse than none.
    @assert nrow(paired) == statistics["n_scenarios"]
    @assert isapprox(mean(paired.tsddr_cost), statistics["tsddr"]["mean"]; rtol = 1e-9)
    @assert isapprox(mean(paired.sddp_cost), statistics["sddp"]["mean"]; rtol = 1e-9)
    @assert isapprox(mean(paired.paired_difference),
                     statistics["paired_difference"]["mean"]; rtol = 1e-9)
    @assert all(paired.tsddr_all_solved) && all(paired.sddp_all_solved)

    figure_training()
    figure_distributions(paired, statistics)
    figure_paired_differences(paired, statistics)
    figure_stagewise()

    println("\nPaired evaluation, n = ", nrow(paired))
    @printf("  TS-DDR  %.5f  (sd %.5f)\n", statistics["tsddr"]["mean"], statistics["tsddr"]["sd"])
    @printf("  SDDP    %.5f  (sd %.5f)\n", statistics["sddp"]["mean"], statistics["sddp"]["sd"])
    d = statistics["paired_difference"]
    @printf("  paired difference +%.5f, SE %.5f, t = %.2f\n", d["mean"], d["se"], d["t"])
    @printf("  relative %+.6f%%, 95%% CI [%+.6f%%, %+.6f%%]\n",
            d["relative_gap_pct"], d["ci95_pct"][1], d["ci95_pct"][2])
end

(abspath(PROGRAM_FILE) == @__FILE__) && main()
