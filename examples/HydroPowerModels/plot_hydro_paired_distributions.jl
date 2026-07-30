# plot_hydro_paired_distributions.jl
#
# Distributional comparison of per-scenario 96-stage rollout costs on the
# PAIRED scenario set (all methods solve the exact same inflow trajectories;
# see paired_scenario_indices in load_hydropowermodels.jl and the
# eval_paired_* scripts).
#
# Produces docs/src/assets/hydro_paired_cost_distributions.png with two panels:
#
#   (a) Absolute cost densities — one transparent filled density per method,
#       overlaid. Shows location and spread of each method's cost
#       distribution over the shared scenarios.
#   (b) Paired-difference densities — density of (method − SDDP) per
#       scenario, with a reference line at 0. Because the evaluation is
#       paired, this panel removes the common between-scenario variance
#       (per-scenario cost std ≈ 5.6k vs paired-difference std ≈ 0.5k) and
#       makes the comparison decidable at a glance: probability mass left of
#       zero is scenarios where the method is cheaper than SDDP.
#
# Densities are Gaussian kernel density estimates computed directly from the
# per-scenario samples (Silverman's bandwidth), no extra packages required.
#
# Data sources (outputs of the paired evaluation scripts; tags identify
# policies; any missing source is skipped so the plot degrades gracefully):
#   MAIN  bolivia/ACPPowerModel/paired_costs.csv          (SDDP column)
#   MAIN  bolivia/ACPPowerModel/paired_costs_ft16.csv     (CPU ft16 column)
#   EXA   .../results/paired_exa_strict_warm_*.jld2       (costs_stagewise)
#
# Usage (inside a SLURM job; examples/HydroPowerModels project):
#   julia --project plot_hydro_paired_distributions.jl
#
# Colors: fixed Okabe–Ito assignments, consistent with
# plot_hydro_strict_convergence.jl (SDDP green, CPU subproblems vermillion,
# GPU DE blue, second GPU variant orange). Identity is encoded by color +
# legend + direct mean annotations, never color alone.

using CSV, DataFrames
using JLD2
using Plots
using Statistics

const HPM_DIR = dirname(@__FILE__)
const OUT_DIR = joinpath(HPM_DIR, "bolivia", "ACPPowerModel")
const EXA_RESULTS = "/storage/scratch1/9/arosemberg3/DecisionRulesExa.jl/" *
                    "examples/HydroPowerModels/bolivia/ACPPowerModel/results"
const DOCS_ASSETS = joinpath(HPM_DIR, "..", "..", "docs", "src", "assets")
mkpath(DOCS_ASSETS)

# Tags identify POLICIES (the paired protocol itself is fixed by seed;
# see paired_scenario_indices in load_hydropowermodels.jl).

# Fixed categorical assignments (Okabe–Ito; validated CVD-safe set). The
# mapping follows the entity, matching the convergence plot, and is never
# reassigned when a series is missing.
const COLORS = Dict(
    "SDDP (SOC-WR / ACP)"              => colorant"#009E73",  # green
    "TS-DDR strict subproblems (CPU)"  => colorant"#D55E00",  # vermillion
    "TS-DDR strict DE (GPU, H256x2)"   => colorant"#0072B2",  # blue
    "TS-DDR strict DE (GPU, H128x3)"   => colorant"#E69F00",  # orange
)

"""
    gaussian_kde(samples; npoints=400) -> (xs, density)

Gaussian kernel density estimate with Silverman's rule-of-thumb bandwidth

```math
h = 0.9 \\, \\min(\\hat\\sigma, \\mathrm{IQR}/1.34) \\; n^{-1/5}.
```

Evaluated on an even grid spanning the sample range padded by three
bandwidths, so the tails fall smoothly to zero inside the plot.
"""
function gaussian_kde(samples::AbstractVector{<:Real}; npoints::Int=400)
    n = length(samples)
    σ = std(samples)
    iqr = quantile(samples, 0.75) - quantile(samples, 0.25)
    # Guard degenerate spread (all-equal samples) with a tiny bandwidth.
    h = max(0.9 * min(σ, iqr / 1.34) * n^(-1/5), eps(Float64) + 1e-9 * max(abs(mean(samples)), 1.0))
    lo, hi = minimum(samples) - 3h, maximum(samples) + 3h
    xs = range(lo, hi; length=npoints)
    dens = [sum(exp(-0.5 * ((x - s) / h)^2) for s in samples) / (n * h * sqrt(2π)) for x in xs]
    return collect(xs), dens
end

# ── Load per-scenario cost vectors (skip missing sources gracefully) ─────────

"""
    load_series() -> Vector{Pair{String,Vector{Float64}}}

Collect every available method's per-scenario paired cost vector, in the
fixed display order. SDDP first (it is the comparison baseline for panel b).
"""
function load_series()
    series = Pair{String,Vector{Float64}}[]

    # MAIN tagged CSV: SDDP column + TS-DDR (.026/ft16) columns.
    main_csv = joinpath(OUT_DIR, "paired_costs.csv")
    if isfile(main_csv)
        df = CSV.read(main_csv, DataFrame)
        for name in names(df)
            if occursin("SDDP", name)
                push!(series, "SDDP (SOC-WR / ACP)" => Float64.(df[!, name]))
            end
        end
    else
        @warn "missing $main_csv — SDDP/anchor series skipped"
    end

    ft_csv = joinpath(OUT_DIR, "paired_costs_ft16.csv")
    if isfile(ft_csv)
        df = CSV.read(ft_csv, DataFrame)
        col = first(names(df))
        push!(series, "TS-DDR strict subproblems (CPU)" => Float64.(df[!, col]))
    else
        @warn "missing $ft_csv — CPU subproblems series skipped"
    end

    for (label, exa_tag) in (
        "TS-DDR strict DE (GPU, H256x2)" => "warm_H256x2",
        "TS-DDR strict DE (GPU, H128x3)" => "warm_H128x3",
    )
        f = joinpath(EXA_RESULTS, "paired_exa_strict_$(exa_tag).jld2")
        if isfile(f)
            d = JLD2.load(f)
            key = haskey(d, "costs_stagewise") ? "costs_stagewise" : "costs"
            push!(series, label => Float64.(vec(d[key])))
        else
            @warn "missing $f — $label skipped"
        end
    end

    return series
end

series = load_series()
isempty(series) && error("No paired cost sources found")

# All series must share the paired scenario count for panel (b) to be valid.
n_scen = length(last(first(series)))
@assert all(length(v) == n_scen for (_, v) in series) "Series lengths differ — not the same paired protocol"
println("Loaded $(length(series)) methods × $n_scen paired scenarios")

# ── Panel (a): absolute cost densities ────────────────────────────────────────

plt_abs = plot(;
    xlabel="96-stage rollout cost",
    ylabel="Density",
    title="Per-scenario cost distributions ($n_scen paired scenarios)",
    legend=:topright,
    grid=:y, gridalpha=0.15,
    size=(900, 400),
)
for (label, costs) in series
    xs, dens = gaussian_kde(costs)
    c = COLORS[label]
    plot!(plt_abs, xs, dens; label=label, color=c, linewidth=2,
        fill=(0, 0.30, c))
    # Direct mean annotation as a short tick, text in muted ink (not series
    # color), placed above the curve peak region.
    vline!(plt_abs, [mean(costs)]; color=c, linestyle=:dash, linewidth=1,
        alpha=0.7, label="")
end

# ── Panel (b): paired differences vs SDDP ─────────────────────────────────────

sddp = last(first(series))
plt_diff = plot(;
    xlabel="Paired cost difference vs SDDP  (method − SDDP, same scenario)",
    ylabel="Density",
    title="Paired differences (mass left of 0 = scenarios beating SDDP)",
    legend=:topright,
    grid=:y, gridalpha=0.15,
    size=(900, 400),
)
vline!(plt_diff, [0.0]; color=:gray30, linestyle=:solid, linewidth=1, label="")
for (label, costs) in series[2:end]   # skip SDDP itself
    diffs = costs .- sddp
    xs, dens = gaussian_kde(diffs)
    c = COLORS[label]
    winrate = round(100 * count(<(0), diffs) / n_scen; digits=1)
    plot!(plt_diff, xs, dens;
        label="$label  (win $winrate%)",
        color=c, linewidth=2, fill=(0, 0.30, c))
    vline!(plt_diff, [mean(diffs)]; color=c, linestyle=:dash, linewidth=1,
        alpha=0.7, label="")
end

final = plot(plt_abs, plt_diff; layout=(2, 1), size=(900, 800),
    left_margin=8Plots.mm, bottom_margin=6Plots.mm)
outfile = joinpath(DOCS_ASSETS, "hydro_paired_cost_distributions.png")
savefig(final, outfile)
println("Saved: $outfile")

# ── Summary table ─────────────────────────────────────────────────────────────

println("\n", "="^78)
println(rpad("Method", 36), rpad("Mean", 12), rpad("Std", 10),
        rpad("Δ vs SDDP", 12), "SEM(Δ)")
println("-"^78)
for (label, costs) in series
    d = costs .- sddp
    println(rpad(label, 36),
        rpad(string(round(mean(costs); digits=1)), 12),
        rpad(string(round(std(costs); digits=1)), 10),
        rpad(label == first(first(series)) ? "—" : string(round(mean(d); digits=1)), 12),
        label == first(first(series)) ? "—" : string(round(std(d) / sqrt(n_scen); digits=1)))
end
println("="^78)
