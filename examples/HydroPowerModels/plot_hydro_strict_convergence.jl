# plot_hydro_strict_convergence.jl
#
# Regenerate the docs convergence figure for the 126-stage Bolivia hydro case,
# comparing the final training schedules against WALL-CLOCK TIME:
#
#   1. SDDP (SOC-WR relaxation of the AC OPF), CPU:
#        - forward-pass simulation cost per iteration (statistical upper estimate)
#        - lower bound per iteration, plus a horizontal dashed line at its final
#          value ("SDDP lower bound (SOC-WR)") — no policy can go below it.
#      Source: OFFLINE parse of sddp/SDDP.log (SDDP.jl iteration table).
#   2. TS-DDR strict, stage-wise subproblems (CPU, Ipopt):
#        phase 1 uses single-sample gradients and training-loss checkpointing;
#        phase 2 switches to 16-sample gradients, decayed LR, and held-out
#        rollout checkpointing. Phase-2 runtime is offset by phase-1 runtime.
#   3. TS-DDR strict, full-horizon deterministic equivalent (GPU, ExaModels+MadNLP):
#        phase 1 uses the strict GPU DE trainer; phase 2 switches to
#        hard-reactive physics, larger mini-batches, decayed LR, and held-out
#        rollout checkpointing. Phase-2 runtime is offset by phase-1 runtime.
#
# Outputs (both saved into docs/src/assets/, path relative to this script):
#   - hydro_training_convergence_by_time.png  (x = wall-clock hours, log10)
#   - hydro_training_convergence_by_step.png  (x = training iteration, log10)
#
# Usage:
#   cd examples/HydroPowerModels
#   julia --project plot_hydro_strict_convergence.jl
#
#   The script must run under the examples/HydroPowerModels environment
#   (examples/HydroPowerModels/Project.toml), which carries both `Plots` and
#   `Wandb` — the root DecisionRules Project.toml does NOT have them.
#   Network access + W&B credentials (~/.netrc or WANDB_API_KEY) are required
#   for series 2 and 3; series 1 (SDDP) is parsed fully offline.
#
# Environment knobs:
#   DR_PLOT_RUNS  Override the W&B run-name substrings, format:
#                     "main=<substr1>,<substr2>;exa=<substr1>,<substr2>,..."
#                 Either part may be omitted to keep its default.
#   DR_PLOT_SMOOTH  Rolling-mean window for the noisy single-sample losses
#                   (default 25).
#
# The script is idempotent: it only reads the log/W&B histories and overwrites
# the two PNGs; re-running it produces the same figures for the same data.

using Plots        # plotting backend (GR) — writes the PNGs
using Statistics   # mean/median/quantile for smoothing and axis limits
using Printf       # formatted summary table

# W&B access follows the exact pattern of compare_hydro_results.jl:
# Wandb.jl wraps the Python `wandb` module via PythonCall; `PC` is PythonCall
# (recovered as the parent module of the wrapped Python object type) and gives
# us `pyconvert`/`pylist` for Julia<->Python conversion.
import Wandb
const wb = Wandb.wandb
const PC = parentmodule(typeof(wb))

# ── Paths ─────────────────────────────────────────────────────────────────────

# Docs asset directory, relative to this script (same convention as
# compare_hydro_results.jl) so the script works from any checkout location.
const DOCS_ASSETS = joinpath(@__DIR__, "..", "..", "docs", "src", "assets")
mkpath(DOCS_ASSETS)

# SDDP.jl training log (offline source for series 1).
const SDDP_LOG = joinpath(@__DIR__, "sddp", "SDDP.log")

# ── Run-name configuration (overridable via DR_PLOT_RUNS) ─────────────────────

# Series 2: strict stage-wise subproblems (CPU, Ipopt), represented as a
# two-phase schedule. Runtime is cumulative across phases.
DEFAULT_MAIN_RUNS = [
    "bolivia-ACPPowerModel-h126-r96-subproblems-strict-2026-07-01T09:41:53.026",
    "bolivia-ACPPowerModel-h126-r96-subproblems-strict-Hlinear-nt16-warm-2026-07-04T21:13:01.712",
]

# Series 3: strict full-horizon DE (GPU), represented as a two-phase schedule.
# Runtime is cumulative across phases.
DEFAULT_EXA_RUNS = [
    "bolivia-ACPPowerModel-h126-r96-deteq-strict-gpu-H256_256-20260704-131003",
    "bolivia-ACPPowerModel-h126-r96-deteq-strict-gpu-H256_256-nt4-rollout-warm-rqhard-20260705-121353",
]

"""
    parse_run_overrides(spec) -> (main::Vector{String}, exa::Vector{String})

Parse the `DR_PLOT_RUNS` override string of the form
`"main=<substr1>,<substr2>;exa=<substr1>,<substr2>,..."`. Each `key=value`
segment is optional; omitted keys keep the defaults above. Unknown keys raise
an error so typos fail loudly instead of silently plotting the default runs.
"""
function parse_run_overrides(spec::AbstractString)
    main = copy(DEFAULT_MAIN_RUNS)                # start from defaults …
    exa  = copy(DEFAULT_EXA_RUNS)
    isempty(strip(spec)) && return main, exa      # empty knob → defaults
    for part in split(spec, ';'; keepempty=false) # each "key=value" segment
        kv = split(part, '='; limit=2)
        length(kv) == 2 || error("DR_PLOT_RUNS segment '$part' is not key=value")
        key, val = strip(kv[1]), strip(kv[2])
        if key == "main"
            main = String.(strip.(split(val, ','; keepempty=false)))
        elseif key == "exa"
            exa = String.(strip.(split(val, ','; keepempty=false)))  # comma list
        else
            error("DR_PLOT_RUNS: unknown key '$key' (expected 'main' or 'exa')")
        end
    end
    return main, exa
end

const MAIN_RUN_SUBSTRS, EXA_RUN_SUBSTRS = parse_run_overrides(get(ENV, "DR_PLOT_RUNS", ""))

# Rolling-mean window for the single-sample (batch size 1) training losses.
const SMOOTH_WINDOW = parse(Int, get(ENV, "DR_PLOT_SMOOTH", "25"))

# ── SDDP.log parsing (offline) ────────────────────────────────────────────────

"""
    parse_sddp_log(path) -> (iteration, simulation, bound, time_s)

Parse the standard SDDP.jl iteration table from a training log. The table
header is `iteration  simulation  bound  time (s)  solves  pid` and each data
row is `iter  simulation  bound  cumulative_time_s  total_solves  pid`
(times are CUMULATIVE seconds since training start; values are in the model's
objective units, here \$).

The log may contain several training blocks (restarted jobs each print a fresh
header). This file (sddp/SDDP.log) contains THREE blocks: the first two are
short aborted trial runs (2 and 1 iterations); the LAST block is the real
441-iteration training run whose final bound is ≈ 378,207. We therefore locate
the LAST table header and parse the data rows that follow it. Note the last
block has no closing "status/total time" footer (the job hit its wall limit),
which is fine: parsing simply stops at the first non-data line or EOF.
"""
function parse_sddp_log(path::AbstractString)
    lines = readlines(path)                                  # whole log in memory (small)
    # Locate every iteration-table header; we will use the LAST one.
    header_re = r"^\s*iteration\s+simulation\s+bound\s+time"
    header_idxs = findall(l -> occursin(header_re, l), lines)
    isempty(header_idxs) && error("No SDDP iteration table found in $path")
    start = header_idxs[end] + 2                             # +1 header, +1 dashed rule
    # Data row: iter (int), simulation (float), bound (float), time (float), solves (int), pid.
    row_re = r"^\s*(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+(\d+)\s+(\d+)\s*$"
    iters, sims, bounds, times = Int[], Float64[], Float64[], Float64[]
    for l in lines[start:end]                                # scan until table ends
        m = match(row_re, l)
        m === nothing && break                               # dashed rule / footer / EOF
        push!(iters,  parse(Int,     m.captures[1]))
        push!(sims,   parse(Float64, m.captures[2]))
        push!(bounds, parse(Float64, m.captures[3]))
        push!(times,  parse(Float64, m.captures[4]))
    end
    isempty(iters) && error("SDDP iteration table at line $(start-2) had no data rows")
    return iters, sims, bounds, times
end

# ── Smoothing ─────────────────────────────────────────────────────────────────

"""
    rolling_mean(v, w) -> Vector{Float64}

Trailing (causal) rolling mean: `out[i] = mean(v[max(1, i-w+1) : i])`, i.e. the
average of the last `w` samples up to and including `i`. A trailing window is
used (rather than centered) so the smoothed curve never uses future samples —
appropriate for a convergence trace. The first `w-1` points average over the
shorter available prefix.
"""
function rolling_mean(v::AbstractVector{<:Real}, w::Int)
    n = length(v)
    out = Vector{Float64}(undef, n)
    for i in 1:n                                             # O(n·w); n ≲ 10⁴ so fine
        lo = max(1, i - w + 1)
        out[i] = mean(@view v[lo:i])
    end
    return out
end

# ── W&B helpers (patterned on compare_hydro_results.jl) ──────────────────────

"""
    find_runs_by_substring(api, substrs; project="RL", max_scan=400) -> Dict

Scan the project's runs in `-created_at` order (most recent first) and return a
`Dict(substring => run)` mapping each requested name-substring to the MOST
RECENT run whose name contains it. Substrings with no match within `max_scan`
runs are absent from the result (caller warns + skips).
"""
function find_runs_by_substring(api, substrs::Vector{String}; project::String="RL", max_scan::Int=400)
    all_runs = api.runs(project, order="-created_at")        # lazy paginated iterator
    found = Dict{String,Any}()
    for i in 0:(max_scan - 1)                                # python 0-based indexing
        r = try
            all_runs[i]
        catch
            break                                            # ran off the end of the project
        end
        name = try
            PC.pyconvert(String, r.name)
        catch
            continue                                         # unreadable run entry — skip
        end
        for s in substrs                                     # first (= newest) match wins
            if !haskey(found, s) && occursin(s, name)
                found[s] = r
            end
        end
        length(found) == length(substrs) && break            # all found — stop scanning
    end
    return found
end

"""
    get_history_with_runtime(r, metric) -> (t_seconds, values)

Fetch the full logged history of `metric` from W&B run `r` via `scan_history`,
together with the built-in `_runtime` field (wall-clock seconds since run
start, attached by W&B to every logged row). Rows missing either field, or
with non-finite metric values, are dropped. Returns two aligned vectors.
"""
function get_history_with_runtime(r, metric::String)
    keys_list = PC.pylist([metric, "_runtime"])              # request both columns
    hist = r.scan_history(keys=keys_list)
    ts, vals = Float64[], Float64[]
    for row in hist                                          # rows stream from the API
        v = try PC.pyconvert(Float64, get(row, metric, nothing))    catch; nothing end
        t = try PC.pyconvert(Float64, get(row, "_runtime", nothing)) catch; nothing end
        (v === nothing || t === nothing || !isfinite(v)) && continue
        push!(ts, t); push!(vals, v)
    end
    return ts, vals
end

"""
    prefix_to_best_training_loss(t, v) -> (t, v)

Return the prefix ending at the best raw training-loss row. The phase-1 strict
CPU checkpoint is selected by this metric.
"""
function prefix_to_best_training_loss(t::Vector{Float64}, v::Vector{Float64})
    isempty(v) && return t, v
    stop = argmin(v)
    return t[1:stop], v[1:stop]
end

"""
    prefix_to_best_rollout(r, t, v) -> (t, v)

Second phases use `DR_SAVE_METRIC=rollout`, so the selected checkpoint is the
best held-out rollout objective rather than the noisy training loss. If the
rollout metric is available, cut the trace at that runtime.
"""
function prefix_to_best_rollout(r, t::Vector{Float64}, v::Vector{Float64})
    rt, rv = get_history_with_runtime(r, "metrics/rollout_objective_no_deficit")
    isempty(rv) && return t, v
    cutoff = rt[argmin(rv)]
    keep = findall(x -> x <= cutoff, t)
    isempty(keep) && return t[1:1], v[1:1]
    return t[keep], v[keep]
end

"""
    append_continuation!(all_t, all_v, t, v)

Append a later phase after the current last wall-clock time. Each W&B run
records runtime from its own start; this converts phases to total runtime.
"""
function append_continuation!(all_t::Vector{Float64}, all_v::Vector{Float64},
                              t::Vector{Float64}, v::Vector{Float64})
    isempty(v) && return all_t, all_v
    offset = isempty(all_t) ? 0.0 : all_t[end]
    append!(all_t, offset .+ t)
    append!(all_v, v)
    return all_t, all_v
end

# ── Load all series ───────────────────────────────────────────────────────────

@info "Parsing SDDP log (offline): $SDDP_LOG"
sddp_iter, sddp_sim, sddp_bound, sddp_time = parse_sddp_log(SDDP_LOG)
sddp_hours  = sddp_time ./ 3600                              # cumulative s → h
bound_final = sddp_bound[end]                                # ≈ 3.782e5 — the SOC-WR lower bound
@info "  $(length(sddp_iter)) SDDP iterations, final bound = $(round(bound_final; digits=1)), " *
      "total time = $(round(sddp_hours[end]; digits=2)) h"

@info "Connecting to W&B (project RL)..."
api = wb.Api()
wanted = vcat(MAIN_RUN_SUBSTRS, EXA_RUN_SUBSTRS)             # all substrings in one scan
found  = find_runs_by_substring(api, wanted)

# Series 2: MAIN strict stage-wise subproblems (CPU). `metrics/training_loss`
# is the per-batch mean 126-stage objective. The default is a continuation
# curve: stage 1 up to its selected training-loss checkpoint, then phase 2 up
# to its best held-out rollout checkpoint, with runtimes offset.
main_hours = Float64[]; main_loss = Float64[]
for (i, s) in enumerate(MAIN_RUN_SUBSTRS)
    if !haskey(found, s)
        @warn "MAIN subproblems-strict run not found (substring '$s') — continuation segment skipped"
        continue
    end
    r = found[s]
    name = PC.pyconvert(String, r.name)
    @info "  MAIN run segment $i: $name ($(PC.pyconvert(String, r.state)))"
    t, v = get_history_with_runtime(r, "metrics/training_loss")
    if isempty(v)
        @warn "  no metrics/training_loss rows yet for $name — segment skipped"
        continue
    end
    if i == 1
        t, v = prefix_to_best_training_loss(t, v)
    else
        t, v = prefix_to_best_rollout(r, t, v)
    end
    append_continuation!(main_hours, main_loss, t ./ 3600, v)
end

# Series 3: strict-DE GPU schedule. `metrics/training_loss` is the
# highest-resolution loss logged by DecisionRulesExa.
exa_hours = Float64[]; exa_loss = Float64[]
for (i, s) in enumerate(EXA_RUN_SUBSTRS)
    if !haskey(found, s)
        @warn "Exa GPU run not found (substring '$s') — continuation segment skipped"
        continue
    end
    r = found[s]
    name = PC.pyconvert(String, r.name)
    @info "  Exa run segment $i: $name ($(PC.pyconvert(String, r.state)))"
    t, v = get_history_with_runtime(r, "metrics/training_loss")
    if isempty(v)
        @warn "  no metrics/training_loss rows yet for $name — segment skipped"
        continue
    end
    if i > 1
        t, v = prefix_to_best_rollout(r, t, v)
    end
    append_continuation!(exa_hours, exa_loss, t ./ 3600, v)
end

main_smooth = rolling_mean(main_loss, SMOOTH_WINDOW)
exa_smooth = rolling_mean(exa_loss, SMOOTH_WINDOW)

# ── Axis limits & styling ─────────────────────────────────────────────────────

# Colors: Okabe–Ito colorblind-safe hues, kept in the same blue/red/green
# family order as the existing docs plots (compare_hydro_results.jl uses
# :blue/:red/:green for DE/subproblems/other):
#   SDDP simulation  → blue, SDDP bound → neutral dark gray,
#   MAIN subproblems (CPU) → vermillion, Exa DE (GPU) family → bluish green.
const C_SDDP  = "#0072B2"   # blue
const C_BOUND = "#4D4D4D"   # neutral dark gray (bound is a reference, not a competitor)
const C_MAIN  = "#D55E00"   # vermillion
const C_EXA   = "#009E73"   # bluish green

# Y-limits: the interesting band is near the SDDP lower bound. Early-training
# TS-DDR losses can be orders of magnitude larger (deficit-penalty transients);
# cropping them keeps the convergence region readable. Lower limit sits just
# below the bound (nothing can be below it); upper limit covers the SDDP
# forward-pass scatter and every method's converged level (robust tail median),
# hard-capped at 3× the bound so a diverged run cannot blow up the scale.
"""
    tail_level(v; frac=0.1) -> Float64

Robust "final level" of a series: the median of its last `max(1, frac·n)`
samples. Used only for choosing the y-axis upper limit.
"""
tail_level(v::AbstractVector{<:Real}; frac::Float64=0.1) =
    isempty(v) ? NaN : median(@view v[max(1, end - max(1, round(Int, frac*length(v))) + 1):end])

finals = Float64[tail_level(sddp_sim)]                       # SDDP forward-pass level
isempty(main_smooth) || push!(finals, tail_level(main_smooth))
isempty(exa_smooth) || push!(finals, tail_level(exa_smooth))
y_lo = 0.985 * bound_final
y_hi = min(3.0 * bound_final,
           1.10 * max(maximum(sddp_sim), maximum(filter(isfinite, finals))))

# X-axis choice: log10. Justification: the series span ~3 orders of magnitude
# of wall time (GPU DE runs log their first losses within minutes, while the
# SDDP and CPU subproblem schedules run much longer). On a
# linear axis the fast-GPU story is squashed into an invisible sliver at the
# left edge; log10 lets both the early GPU descent and the long CPU tails read.
# Same reasoning for the by-iteration plot (441 SDDP iterations vs ~8000
# TS-DDR batches). Zero/near-zero times are clamped to 10 s to stay on-axis.
clamp_hours(h) = max.(h, 10 / 3600)

# ── Plotting ──────────────────────────────────────────────────────────────────

"""
    convergence_plot(xsel, xlabel) -> Plots.Plot

Build one convergence figure. `xsel(series_kind, i)` is inlined below instead —
this helper simply takes, per series, precomputed x-vectors, so both figures
(by-time and by-step) share identical styling and differ only in x data.
`xs` is a NamedTuple with fields `sddp`, `main`, `exa::Vector` holding the
x-vectors for each series.
"""
function convergence_plot(xs, xlabel::String)
    plt = plot(; size = (900, 540),                          # docs figure size
        xlabel = xlabel,
        ylabel = "Operational Cost (126 stages)",            # docs-style cost label
        title  = "Training Convergence (126-stage Bolivia AC): GPU vs CPU",
        legend = :topright,
        xscale = :log10,                                     # see justification above
        ylims  = (y_lo, y_hi),
        left_margin = 5Plots.mm, bottom_margin = 4Plots.mm)

    # SDDP forward-pass simulation cost: per-iteration Monte-Carlo estimate of
    # the current policy's cost (an upper, noisy companion to the bound).
    plot!(plt, xs.sddp, sddp_sim; color = C_SDDP, alpha = 0.55, lw = 1.2,
          label = "SDDP forward simulation (SOC-WR, CPU)")
    # SDDP lower bound: monotone deterministic series.
    plot!(plt, xs.sddp, sddp_bound; color = C_BOUND, lw = 2.0,
          label = "SDDP lower bound (SOC-WR)")
    # TS-DDR curves use a trailing rolling mean (window $SMOOTH_WINDOW) to make
    # small-sample training losses readable without using future samples.
    if !isempty(main_loss)
        plot!(plt, xs.main, main_smooth; color = C_MAIN, lw = 2.0,
              label = "TS-DDR strict subproblems (CPU, Ipopt)")
    end

    if !isempty(exa_loss)
        plot!(plt, xs.exa, exa_smooth; color = C_EXA, lw = 2.3,
              label = "TS-DDR strict DE (GPU, MadNLP/cuDSS)")
    end
    return plt
end

# Figure 1 — by wall-clock time (hours). This is the headline GPU-vs-CPU plot.
xs_time = (sddp = clamp_hours(sddp_hours),
           main = clamp_hours(main_hours),
           exa  = clamp_hours(exa_hours))
plt_time = convergence_plot(xs_time, "Wall-clock time (hours, log scale)")
savefig(plt_time, joinpath(DOCS_ASSETS, "hydro_training_convergence_by_time.png"))
println("Saved hydro_training_convergence_by_time.png")

# Figure 2 — by training iteration (SDDP iterations vs TS-DDR batches).
# Iteration costs differ wildly across methods (one SDDP iteration = 1890
# solves; one TS-DDR batch = 1 DE solve or 126 stage solves) — this companion
# shows per-iteration progress, while Figure 1 shows the honest time story.
xs_step = (sddp = Float64.(sddp_iter),
           main = collect(1.0:length(main_loss)),
           exa  = collect(1.0:length(exa_loss)))
plt_step = convergence_plot(xs_step, "Training iteration (log scale)")
savefig(plt_step, joinpath(DOCS_ASSETS, "hydro_training_convergence_by_step.png"))
println("Saved hydro_training_convergence_by_step.png")

# ── Summary table ─────────────────────────────────────────────────────────────

println("\n", "="^88)
println("Summary — final value & wall time per series")
println("="^88)
@printf("%-48s %14s %10s %8s\n", "Series", "Final value", "Hours", "Points")
println("-"^88)
@printf("%-48s %14.1f %10.2f %8d\n", "SDDP lower bound (SOC-WR)",
        bound_final, sddp_hours[end], length(sddp_bound))
@printf("%-48s %14.1f %10.2f %8d\n", "SDDP forward simulation",
        tail_level(sddp_sim), sddp_hours[end], length(sddp_sim))
if !isempty(main_loss)
    @printf("%-48s %14.1f %10.2f %8d\n", "TS-DDR strict subproblems (CPU) [smoothed]",
            main_smooth[end], main_hours[end], length(main_loss))
end
if !isempty(exa_loss)
    @printf("%-48s %14.1f %10.2f %8d\n", "TS-DDR strict DE GPU [smoothed]",
            exa_smooth[end], exa_hours[end], length(exa_loss))
end
println("="^88)
