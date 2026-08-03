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
#   hydro_energy_price.png           the mean nodal energy price, week by week
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
    phase_label(stage) -> String

Display name for a training phase.

The recorded evidence keys phases by the identifiers the original run used.
Those are lab-notebook tags: they carry no meaning to a reader and, worse, they
read as a search over many attempts. A published figure names phases by their
position in the schedule; the identifiers stay in the record, where they are
what actually indexes the data.
"""
phase_label(stage) = get(PHASE_LABELS, stage, stage)

const PHASE_LABELS = Dict("coldB" => "phase 1", "C1" => "phase 2", "C3" => "phase 3")

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

    # Reserve empty bands above and below the data: the phase annotations live in
    # the upper one and the legend in the lower right, so neither can land on the
    # curves or on each other. Placing both at :topright previously overlapped the
    # annotations with the legend box and clipped the last phase at the frame.
    loss_lo = minimum(skipmissing(loss[!, "metrics/training_loss"]))
    loss_hi = maximum(skipmissing(loss[!, "metrics/training_loss"]))
    loss_span = loss_hi - loss_lo
    annotation_y = loss_hi + 0.11 * loss_span

    # Pad the time axis so the last phase's centred annotation cannot run into the
    # frame. Both panels get the SAME limits: they are stacked and read as one
    # time axis, so padding only the top one would misalign the restart lines.
    hours_max = maximum(loss.active_seconds) / 3600
    xlimits = (-0.02 * hours_max, 1.05 * hours_max)

    top = plot(;
        ylabel = "126-stage objective",
        title = "Training from random initialisation — " *
                join(phase_label.(selected), " → "),
        legend = :bottomright, grid = :y, gridalpha = 0.15,
        ylims = (loss_lo - 0.20 * loss_span, loss_hi + 0.22 * loss_span),
        xlims = xlimits,
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
               label = i == 1 ? "restart (optimiser, LR schedule and warm-up reset)" : "")
    end

    bottom = plot(;
        xlabel = "active wall time (h)",
        ylabel = "96-stage fixed-panel cost",
        legend = :topright, grid = :y, gridalpha = 0.15,
        xlims = xlimits,
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
    #
    # The schedule RAMPS UP from LR/100 across the warm-up before the cosine
    # decay begins, so a plain `minimum` over the logged rate returns the warm-up's
    # first step rather than the schedule's floor — for phase 1 that is
    # 1e-3 * (0.01 + 0.99/20) = 6e-5, which reads as a decay target it never was.
    # The floor after the peak is the band the phase actually descended through,
    # and it stays truthful when a phase stops before the cosine completes.
    running = 0.0
    for stage in selected
        rows = history[history.stage .== stage, :]
        nt = Int(first(skipmissing(rows[!, "metrics/num_train_per_batch"])))
        lrs = collect(skipmissing(rows[!, "metrics/lr"]))
        lr_peak = maximum(lrs)
        lr_floor = minimum(@view lrs[argmax(lrs):end])
        mid = (running + per_stage[stage]["active_seconds"] / 2) / 3600
        annotate!(top, mid, annotation_y,
                  text(@sprintf("%s\nsample %d\nLR %.0e→%.0e",
                                phase_label(stage), nt, lr_peak, lr_floor),
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
        xlabel = "96-stage true-ACP operating cost (objective units)",
        ylabel = "density",
        title = @sprintf("Cost over %d paired inflow scenarios", nrow(paired)),
        legend = :topright, grid = :y, gridalpha = 0.15, size = (1000, 460),
        left_margin = 12Plots.mm, bottom_margin = 10Plots.mm,
    )
    for (label, costs, colour) in (("SDDP", sddp, C_SDDP), ("TS-DDR", tsddr, C_TSDDR))
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
        xlabel = "paired difference, TS-DDR − SDDP, same scenario (objective units)",
        ylabel = "density",
        title = "Paired differences — the comparison the evaluation is designed to make",
        # The mass sits to the RIGHT of zero, so :topright puts the legend on top
        # of the peak. The left half of this axis is empty by construction.
        legend = :topleft, grid = :y, gridalpha = 0.15, size = (1000, 470),
        left_margin = 12Plots.mm, bottom_margin = 10Plots.mm,
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
        ylabel = "cumulative cost difference\n(objective units)",
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
          color = C_TSDDR, linewidth = 2, label = "TS-DDR")

    bottom = plot(; xlabel = "stage (week)", ylabel = "thermal generation (MW)",
                  legend = :topright, grid = :y, gridalpha = 0.15)
    plot!(bottom, reported.stage, reported.sddp_thermal_MW;
          color = C_SDDP, linewidth = 2, label = "SDDP")
    plot!(bottom, reported.stage, reported.tsddr_thermal_MW;
          color = C_TSDDR, linewidth = 2, label = "TS-DDR")

    figure = plot(top, middle, bottom; layout = (3, 1), size = (1000, 900),
                  left_margin = 14Plots.mm, bottom_margin = 8Plots.mm)
    path = joinpath(ASSETS, "hydro_stagewise_physical.png")
    savefig(figure, path)
    println("Saved: $path")
    return nothing
end

"""
    price_series() -> Union{Nothing,DataFrame}

Per-stage nodal energy price for each policy, averaged over the panel columns.

Prefers the compact `results/stagewise_prices.csv`. If that is absent but the
raw solution dumps are, it reduces them and writes the compact file, so the
figure is reproducible from `results/` alone thereafter. Returns `nothing` when
neither exists, and the price figure is then skipped rather than faked.

The reduction is a mean of `price_active` over buses and over the scenarios the
two policies have IN COMMON — averaging one policy over ten columns and the other
over one would compare scenario sets, not policies. The number of paired
scenarios is carried in the output and shown on the figure.

It is a *system* price, not a locational one: the point of the figure is when
energy is expensive, and how the two policies' water decisions move that in
time. Per-bus detail is in the dumps for anyone who wants it.

`price_active` is the dual of a bus's active-power balance, in objective units
per per-unit power per stage; dividing by `baseMVA = 100` would put it per MW.
The objective's own unit is whatever the case's cost coefficients are denominated
in, which the case does not state, so it is not called a currency here — the
load-shedding price of 6000 is the reference that makes the level interpretable.
"""
function price_series()
    compact = joinpath(RESULTS, "stagewise_prices.csv")
    isfile(compact) && return CSV.read(compact, DataFrame)

    audit = joinpath(HYDRO_DIR, "bolivia", "ACPPowerModel", "audit")
    isdir(audit) || return nothing
    sources = Dict(
        "sddp" => filter(f -> occursin(r"^sddp_\d+_\d+_solution\.csv$", f), readdir(audit)),
        "tsddr" => filter(f -> occursin(r"solution.*\.csv$", f),
                          readdir(joinpath(HYDRO_DIR, "bolivia", "ACPPowerModel"))),
    )
    isempty(sources["sddp"]) && return nothing

    # Accumulate per (policy, scenario, stage) so the two policies can be
    # restricted to the SAME scenarios before averaging. A mean over ten columns
    # on one side and one column on the other would not be a comparison of
    # policies — it would be a comparison of scenario sets.
    price = Dict{String,Dict{Tuple{Int,Int},Vector{Float64}}}()
    for (policy, files) in sources
        acc = Dict{Tuple{Int,Int},Vector{Float64}}()
        base = policy == "sddp" ? audit : joinpath(HYDRO_DIR, "bolivia", "ACPPowerModel")
        for f in files
            path = joinpath(base, f)
            isfile(path) || continue
            for row in CSV.Rows(path;
                                types = Dict(:scenario => Int, :stage => Int,
                                             :value => Float64))
                row.class == "price_active" || continue
                push!(get!(acc, (row.scenario, row.stage), Float64[]), row.value)
            end
        end
        isempty(acc) || (price[policy] = acc)
    end
    haskey(price, "sddp") || return nothing

    scenarios = Set(k[1] for k in keys(price["sddp"]))
    for policy in keys(price)
        intersect!(scenarios, Set(k[1] for k in keys(price[policy])))
    end
    isempty(scenarios) && return nothing
    stages = sort(unique(k[2] for k in keys(price["sddp"]) if k[1] in scenarios))
    @info "nodal prices" policies = sort(collect(keys(price))) n_scenarios =
        length(scenarios) scenarios = sort(collect(scenarios))

    frame = DataFrame(stage = stages)
    for policy in ("sddp", "tsddr")
        haskey(price, policy) || continue
        frame[!, Symbol(policy * "_price")] = [
            mean(vcat((get(price[policy], (s, t), Float64[]) for s in scenarios)...))
            for t in stages
        ]
    end
    frame[!, :n_scenarios] .= length(scenarios)
    CSV.write(compact, frame)
    println("Wrote: $compact")
    return frame
end

"""
    price_bands(prices; min_run=3) -> Vector{NamedTuple}

Contiguous runs of constant sign in the per-stage price difference
`TS-DDR − SDDP`, keeping only runs of at least `min_run` stages.

This exists because the obvious summary is misleading. Bucketing the horizon
into equal segments and averaging reports a smooth drift from cheaper to dearer;
the difference actually alternates in bands tied to the reservoir cycle, and
fixed buckets straddle them. Runs of constant sign are the structure that is
there, so the prose quotes this rather than a bucketing chosen in advance.

`min_run` drops one- and two-stage sign flips, which are sampling noise on a
ten-scenario mean rather than a change in behaviour.
"""
function price_bands(prices::DataFrame; min_run::Int = 3)
    delta = (.-prices.tsddr_price) .- (.-prices.sddp_price)
    stages = prices.stage
    bands = NamedTuple[]
    i = 1
    while i <= length(delta)
        j = i
        while j < length(delta) && (delta[j + 1] > 0) == (delta[i] > 0)
            j += 1
        end
        if j - i + 1 >= min_run
            push!(bands, (first_stage = stages[i], last_stage = stages[j],
                          n = j - i + 1, dearer = delta[i] > 0,
                          mean = mean(view(delta, i:j))))
        end
        i = j + 1
    end
    return bands
end

"""
    report_price_bands()

Print the sign bands of the price difference.

The case study quotes these numbers, so they are printed by the script that
draws the figure rather than derived once by hand: a table transcribed into prose
has no way to notice when the evidence beneath it changes.
"""
function report_price_bands()
    prices = price_series()
    prices === nothing && return nothing
    all(c -> Symbol(c) in propertynames(prices), ("sddp_price", "tsddr_price")) ||
        return nothing
    println("\nPrice difference (TS-DDR − SDDP), contiguous sign bands of >= 3 stages:")
    for b in price_bands(prices)
        @printf("  stages %3d-%-3d (%2d stages)  %-14s  mean %+7.1f\n",
                b.first_stage, b.last_stage, b.n,
                b.dearer ? "TS-DDR dearer" : "TS-DDR cheaper", b.mean)
    end
    return nothing
end

function figure_prices()
    prices = price_series()
    if prices === nothing
        @warn "no nodal-price data found; skipping the price figure. Produce it with " *
              "DR_SOLUTION_DUMP=1 on either paired evaluator."
        return nothing
    end
    # The recorded dual is of the balance AS STORED, which is the NEGATIVE of the
    # conventional price (see PRICE_CLASSES). Negating puts "expensive" up, where
    # a reader expects it.
    n_paired = :n_scenarios in propertynames(prices) ?
               Int(first(prices.n_scenarios)) : 0
    suffix = n_paired == 0 ? "" :
             "  ($n_paired paired scenario$(n_paired == 1 ? "" : "s"))"

    has_both = all(c -> Symbol(c) in propertynames(prices),
                   ("sddp_price", "tsddr_price"))

    # LEVELS. Drawn on the data's own scale. Earlier versions put the
    # load-shedding price (6000) on this axis as a reference line, which set the
    # y-range to [0, 6000] and squeezed the entire signal — a band about 120 wide
    # — into an unreadable sliver at the bottom. The reference belongs in words:
    # serving load costs roughly a quarter of what shedding it does, which is why
    # nothing is shed anywhere in this study.
    # Negate once, here: the recorded dual is of the balance as stored, and every
    # panel below plots the conventional price.
    series = Dict(col => .-prices[!, Symbol(col)]
                  for col in ("sddp_price", "tsddr_price")
                  if Symbol(col) in propertynames(prices))

    levels = plot(;
        ylabel = "marginal cost of load\n(objective units per pu per stage)",
        title = "What energy is worth, week by week" * suffix,
        legend = :topright, grid = :y, gridalpha = 0.15,
    )
    for (col, label, colour) in (("sddp_price", "SDDP", C_SDDP),
                                 ("tsddr_price", "TS-DDR", C_TSDDR))
        haskey(series, col) || continue
        plot!(levels, prices.stage, series[col];
              color = colour, linewidth = 2, label = label)
    end
    annotate!(levels, first(prices.stage), maximum(series["sddp_price"]),
              text("load shedding is priced at 6000, far above this axis",
                   7, C_INK, :left, :top))

    # DIFFERENCE. The levels differ by well under a percent, so the sign of the
    # difference — cheaper early, dearer at the end — is not legible from two
    # overlaid curves however they are scaled. It is the claim the text makes, so
    # it gets its own panel rather than a reader's benefit of the doubt.
    panels = Any[levels]
    if has_both
        delta = series["tsddr_price"] .- series["sddp_price"]
        gap = plot(; xlabel = "stage (week)",
                   ylabel = "TS-DDR − SDDP",
                   legend = :topleft, grid = :y, gridalpha = 0.15)
        plot!(gap, prices.stage, delta; color = C_TSDDR, linewidth = 2,
              fill = (0, 0.20, C_TSDDR), label = "difference in marginal cost")
        hline!(gap, [0.0]; color = C_INK, linewidth = 1.2, label = "")
        push!(panels, gap)
    else
        plot!(levels; xlabel = "stage (week)")
    end

    figure = length(panels) == 1 ? plot(panels[1]; size = (1000, 480)) :
             plot(panels...; layout = grid(2, 1; heights = [0.62, 0.38]),
                  size = (1000, 700))
    plot!(figure; left_margin = 14Plots.mm, bottom_margin = 10Plots.mm)
    path = joinpath(ASSETS, "hydro_energy_price.png")
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
    figure_prices()

    println("\nPaired evaluation, n = ", nrow(paired))
    @printf("  TS-DDR  %.5f  (sd %.5f)\n", statistics["tsddr"]["mean"], statistics["tsddr"]["sd"])
    @printf("  SDDP    %.5f  (sd %.5f)\n", statistics["sddp"]["mean"], statistics["sddp"]["sd"])
    d = statistics["paired_difference"]
    @printf("  paired difference +%.5f, SE %.5f, t = %.2f\n", d["mean"], d["se"], d["t"])
    @printf("  relative %+.6f%%, 95%% CI [%+.6f%%, %+.6f%%]\n",
            d["relative_gap_pct"], d["ci95_pct"][1], d["ci95_pct"][2])
    report_price_bands()
end

(abspath(PROGRAM_FILE) == @__FILE__) && main()
