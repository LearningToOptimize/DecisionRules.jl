# battery_analysis.jl
#
# Tables and figures for the battery-storage study.
#
# Everything here consumes the SHARED SOLUTION SCHEMA — the long
# `(scenario, stage, class, index, value)` format defined in
# `battery_solution_schema.jl` — or the named tuples the diagnostics return.
# Nothing reads a solver-specific internal layout, which is what lets one plot
# overlay a JuMP trajectory and an ExaModels trajectory without either engine
# knowing about the other.
#
# Figures are written to files rather than displayed: this study runs on cluster
# nodes without a display, and a figure that only exists in a REPL cannot be
# regenerated from the evidence.

using DataFrames
using Plots
using Printf
using Statistics

@isdefined(SolutionRecorder) || include(joinpath(@__DIR__, "battery_solution_schema.jl"))

# Headless rendering: GR opens no window and writes straight to the file.
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

# ─────────────────────────────────────────────────────────────────────────────
# The schema as a DataFrame
# ─────────────────────────────────────────────────────────────────────────────

"""
    solution_frame(source) -> DataFrame

Read the shared solution schema into a long `DataFrame` with columns
`scenario, stage, class, index, value`.

# Arguments
- `source`: a path to a solution CSV, a [`SolutionRecorder`](@ref), or the
  dictionary [`read_solution`](@ref) returns.

# Notes
Long format is kept all the way to the plotting layer. Widening happens per
figure, through [`pivot_class`](@ref), because different figures widen on
different keys and a single wide table would have to guess which.
"""
function solution_frame(source)
    rows = if source isa SolutionRecorder
        source.rows
    elseif source isa AbstractString
        [(k[1], k[2], k[3], k[4], v) for (k, v) in read_solution(source)]
    elseif source isa AbstractDict
        [(k[1], k[2], k[3], k[4], v) for (k, v) in source]
    else
        error("solution_frame: unsupported source of type $(typeof(source))")
    end
    df = DataFrame(scenario = [r[1] for r in rows], stage = [r[2] for r in rows],
                   class = [r[3] for r in rows], index = [r[4] for r in rows],
                   value = [r[5] for r in rows])
    return sort!(df, [:scenario, :stage, :class, :index])
end

"""
    pivot_class(df, class; scenario=nothing) -> DataFrame

One row per stage, one column per component identifier, for a single class.

# Notes
Column names are the component IDENTIFIERS as strings, never positions, so a
table built from a case with nonconsecutive identifiers still names the right
component.
"""
function pivot_class(df::DataFrame, class::AbstractString; scenario = nothing)
    sub = scenario === nothing ? df[df.class .== class, :] :
                                 df[(df.class .== class) .& (df.scenario .== scenario), :]
    isempty(sub) && return DataFrame(stage = Int[])
    return unstack(sub, :stage, :index, :value; renamecols = i -> Symbol(string(i)))
end

# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────

"""
    stage_cost_table(result) -> DataFrame

Per-stage cost decomposition of a [`deterministic_equivalent`](@ref) result: the
generation, throughput, deficit and surplus components, the stage total and the
running total.

# Notes
The cumulative column is what makes a stagewise comparison readable: two policies
that differ by a fraction of a percent per stage are indistinguishable stage by
stage and obvious cumulatively.
"""
function stage_cost_table(result)
    T = result.horizon
    return DataFrame(
        stage = 1:T,
        generation = [result.stages[t].cost_generation for t in 1:T],
        throughput = [result.stages[t].cost_throughput for t in 1:T],
        deficit = [result.stages[t].cost_deficit for t in 1:T],
        surplus = [result.stages[t].cost_surplus for t in 1:T],
        total = [result.stages[t].cost_stage for t in 1:T],
        cumulative = cumsum([result.stages[t].cost_stage for t in 1:T]),
    )
end

"""
    battery_table(result, case) -> DataFrame

Per-stage, per-battery energy, charging, discharging, throughput and the
simultaneous-operation measure.

# Notes
`simultaneous = min(p_ch, p_dis)` is reported because a continuous relaxation of
the charge/discharge complementarity CAN return a solution that does both at
once, and a positive throughput cost is the only thing making that suboptimal. A
column of zeros here is what licenses reading `p_bat` as a physical injection.
"""
function battery_table(result, case::BatteryCase)
    rows = DataFrame(stage = Int[], battery = Int[], bus = Int[],
                     energy_in = Float64[], energy_out = Float64[],
                     p_ch = Float64[], p_dis = Float64[], p_bat = Float64[],
                     throughput = Float64[], simultaneous = Float64[])
    Δt = stage_hours(case)
    for t in 1:result.horizon, b in case.batteries
        s = result.stages[t]
        push!(rows, (t, b.index, b.bus, s.energy_in[b.index], s.energy_out[b.index],
                     s.p_ch[b.index], s.p_dis[b.index], s.p_bat[b.index],
                     Δt * (s.p_ch[b.index] + s.p_dis[b.index]),
                     min(s.p_ch[b.index], s.p_dis[b.index])))
    end
    return rows
end

"""
    price_table(result, case; buses=nothing) -> DataFrame

Per-stage active and reactive nodal prices.

# Notes
Reactive prices are reported alongside active ones because on a case where
storage matters for VOLTAGE support rather than for energy arbitrage, the
reactive price is where the mechanism shows — and a study that only tabulates
active prices would report that nothing is happening.
"""
function price_table(result, case::BatteryCase; buses = nothing)
    ids = buses === nothing ? sort!([Int(b["index"]) for (_, b) in case.network["bus"]]) :
                              collect(Int.(buses))
    rows = DataFrame(stage = Int[], bus = Int[], price_active = Float64[],
                     price_reactive = Float64[], vm = Float64[],
                     deficit = Float64[], surplus = Float64[])
    for t in 1:result.horizon, i in ids
        s = result.stages[t]
        push!(rows, (t, i, get(s.price_active, i, NaN), get(s.price_reactive, i, NaN),
                     s.vm[i], s.deficit[i], s.surplus[i]))
    end
    return rows
end

"""
    value_curve_table(cmp) -> DataFrame

The ACP-versus-SOC-WR value curve comparison of
[`compare_value_curves`](@ref), one row per grid point.

# Notes
The column that carries the finding is `dlambda`: the difference in the MARGINAL
value of stored energy. `dcost` — the difference in the value curves themselves —
is reported beside it and is expected to be nonzero everywhere, because a
relaxation is below the true model by construction. Reading the level difference
as the mechanism is the mistake this table is laid out to prevent.
"""
function value_curve_table(cmp)
    n = length(cmp.energy)
    return DataFrame(
        energy = cmp.energy,
        reachable = cmp.acp.reachable,
        endpoint = cmp.acp.endpoint,
        nonsmooth = cmp.acp.nonsmooth,
        value_acp = cmp.acp.value,
        value_soc = cmp.soc.value,
        dcost = cmp.acp.value .- cmp.soc.value,
        lambda_acp = cmp.acp.lambda,
        lambda_soc = cmp.soc.lambda,
        dlambda = cmp.dlambda,
        fd_acp = cmp.acp.fd,
        fd_error_acp = cmp.acp.fd_error,
        recourse_acp = cmp.acp.worst_recourse,
    )
end

"""
    marginal_value_table(comparisons) -> DataFrame

One row per probed battery: the interior ACP and SOC-WR marginal stored-energy
values and their disagreement, with the LOCATIONAL RANK each formulation assigns.

# Arguments
- `comparisons`: an iterable of [`compare_value_curves`](@ref) results.

# Notes
Ranking is the point. A uniform level shift in the relaxation's marginal values
changes nothing about where a policy puts energy; a change in the ORDER of buses
does. `rank_acp != rank_soc` is the decision-level evidence that a cut built on
the relaxation would steer storage to a different place from the truth.

Interior points only — see [`energy_value_curve`](@ref) on why a multiplier at a
reachable-interval endpoint is not comparable across solvers.
"""
function marginal_value_table(comparisons)
    rows = DataFrame(battery = Int[], bus = Int[], stage = Int[], atom = Int[],
                     lambda_acp = Float64[], lambda_soc = Float64[],
                     dlambda = Float64[], rel_dlambda = Float64[],
                     reversal = Bool[], num_interior = Int[],
                     num_recourse_excluded = Int[])
    for c in comparisons
        idx = c.interior
        isempty(idx) && continue
        λa = mean(c.acp.lambda[idx])
        λs = mean(c.soc.lambda[idx])
        push!(rows, (c.acp.battery, c.acp.bus, c.acp.stage, c.acp.atom,
                     λa, λs, λs - λa, (λs - λa) / max(abs(λa), 1e-8),
                     c.reversal, length(idx), c.num_recourse_excluded))
    end
    isempty(rows) && return rows
    rows.rank_acp = competerank_desc(rows.lambda_acp)
    rows.rank_soc = competerank_desc(rows.lambda_soc)
    rows.rank_changed = rows.rank_acp .!= rows.rank_soc
    return rows
end

"""
    competerank_desc(v) -> Vector{Int}

Descending competition rank of `v`: the largest value gets rank 1, ties share the
smaller rank.

# Notes
Written out rather than pulled from StatsBase, which this example does not
otherwise need, and ties are handled explicitly because two buses with equal
marginal value are not "differently ranked" in any meaningful sense.
"""
function competerank_desc(v::AbstractVector)
    order = sortperm(v; rev = true)
    r = Vector{Int}(undef, length(v))
    prev = NaN
    prev_rank = 0
    for (pos, i) in enumerate(order)
        if !(v[i] ≈ prev)
            prev_rank = pos
            prev = v[i]
        end
        r[i] = prev_rank
    end
    return r
end

"""
    binding_table(probe; limit=20) -> DataFrame

The binding and nearly binding limits of a probe, most binding first.
"""
function binding_table(probe; limit::Integer = 20)
    rows = first(probe.binding, limit)
    return DataFrame(kind = [r.kind for r in rows], index = [r.index for r in rows],
                     side = [r.side for r in rows], value = [r.value for r in rows],
                     limit = [r.limit for r in rows], slack = [r.slack for r in rows])
end

"""
    panel_table(panel) -> DataFrame

One row per scenario of a [`perfect_foresight_panel`](@ref), with the statuses
kept as strings so a failed scenario survives a CSV round-trip.
"""
function panel_table(panel)
    return DataFrame(scenario = [r.scenario for r in panel.rows],
                     solved = [r.solved for r in panel.rows],
                     first_status = [string(r.first_status) for r in panel.rows],
                     status = [string(r.status) for r in panel.rows],
                     cost = [r.cost for r in panel.rows],
                     worst_deficit = [r.worst_deficit for r in panel.rows],
                     worst_surplus = [r.worst_surplus for r in panel.rows],
                     simultaneous = [r.simultaneous for r in panel.rows],
                     terminal_energy = [r.terminal_energy for r in panel.rows],
                     throughput = [r.throughput for r in panel.rows])
end

# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_value_curves(cmp, path) -> String

Two stacked panels: the ACP and SOC-WR value curves, and the marginal value of
stored energy under each.

# Notes
The marginal values go on their OWN panel rather than a twin axis. The two levels
routinely differ by less than a percent while the quantity being studied is that
difference, and overlaying them on one axis makes a difference of interest look
like a coincident pair of lines.

Reachable-interval endpoints are drawn as open markers, because a multiplier
there is a subgradient and comparing two solvers' subgradients is not evidence.
"""
function plot_value_curves(cmp, path::AbstractString)
    e = cmp.energy
    ok = cmp.acp.solved .& cmp.soc.solved
    interior = ok .& .!cmp.acp.endpoint

    top = plot(e[ok], cmp.acp.value[ok]; label = "ACP (true)", lw = 2,
               ylabel = "stage cost", legend = :topleft)
    plot!(top, e[ok], cmp.soc.value[ok]; label = "SOC-WR (relaxation)", lw = 2, ls = :dash)

    bot = plot(e[interior], cmp.acp.lambda[interior]; label = "∂Q/∂e  ACP", lw = 2,
               xlabel = "outgoing energy of battery $(cmp.acp.battery) at bus $(cmp.acp.bus) (pu·h)",
               ylabel = "marginal value", legend = :topleft)
    plot!(bot, e[interior], cmp.soc.lambda[interior]; label = "∂Q/∂e  SOC-WR", lw = 2, ls = :dash)
    endpoints = ok .& cmp.acp.endpoint
    any(endpoints) && scatter!(bot, e[endpoints], cmp.acp.lambda[endpoints];
                               label = "endpoint (subgradient)", markershape = :circle,
                               markercolor = :white)

    fig = plot(top, bot; layout = (2, 1), size = (860, 620),
               title = ["stage $(cmp.acp.stage), atom $(cmp.acp.atom)" ""])
    savefig(fig, path)
    return path
end

"""
    plot_energy_trajectory(result, case, path) -> String

Battery energy, charging and discharging over the horizon of one deterministic
equivalent.
"""
function plot_energy_trajectory(result, case::BatteryCase, path::AbstractString)
    T = result.horizon
    top = plot(; ylabel = "stored energy (pu·h)", legend = :topright)
    bot = plot(; ylabel = "power (pu)", xlabel = "stage", legend = :topright)
    for b in case.batteries
        plot!(top, 1:T, [result.stages[t].energy_out[b.index] for t in 1:T];
              label = "battery $(b.index) @ bus $(b.bus)", lw = 2)
        plot!(bot, 1:T, [result.stages[t].p_bat[b.index] for t in 1:T];
              label = "net injection $(b.index)", lw = 2)
    end
    hline!(bot, [0.0]; label = "", lc = :black, ls = :dot)
    fig = plot(top, bot; layout = (2, 1), size = (860, 620))
    savefig(fig, path)
    return path
end

"""
    plot_stage_costs(results, labels, path) -> String

Stage and cumulative cost of one or more deterministic equivalents.

# Notes
Cumulative cost is plotted as a DIFFERENCE from the first series when more than
one is given: two trajectories whose cumulative costs differ by a fraction of a
percent are one line at any readable scale, and the difference is the quantity.
"""
function plot_stage_costs(results, labels, path::AbstractString)
    T = results[1].horizon
    top = plot(; ylabel = "stage cost", legend = :topleft)
    for (r, l) in zip(results, labels)
        plot!(top, 1:T, r.stage_cost; label = l, lw = 2)
    end
    bot = if length(results) == 1
        plot(1:T, results[1].cumulative_cost; label = labels[1], lw = 2,
             ylabel = "cumulative cost", xlabel = "stage", legend = :topleft)
    else
        p = plot(; ylabel = "cumulative Δcost vs $(labels[1])", xlabel = "stage",
                 legend = :topleft)
        for (r, l) in zip(results[2:end], labels[2:end])
            plot!(p, 1:T, r.cumulative_cost .- results[1].cumulative_cost; label = l, lw = 2)
        end
        hline!(p, [0.0]; label = "", lc = :black, ls = :dot)
        p
    end
    fig = plot(top, bot; layout = (2, 1), size = (860, 620))
    savefig(fig, path)
    return path
end

"""
    plot_prices(result, case, path; buses=nothing) -> String

Active and reactive nodal prices over the horizon, plus the voltage magnitude at
the same buses.
"""
function plot_prices(result, case::BatteryCase, path::AbstractString; buses = nothing)
    ids = buses === nothing ? sort!(unique(b.bus for b in case.batteries)) : collect(Int.(buses))
    T = result.horizon
    pa = plot(; ylabel = "active price", legend = :topleft)
    pr = plot(; ylabel = "reactive price", legend = :topleft)
    pv = plot(; ylabel = "|V| (pu)", xlabel = "stage", legend = :topleft)
    for i in ids
        plot!(pa, 1:T, [get(result.stages[t].price_active, i, NaN) for t in 1:T];
              label = "bus $i", lw = 2)
        plot!(pr, 1:T, [get(result.stages[t].price_reactive, i, NaN) for t in 1:T];
              label = "bus $i", lw = 2)
        plot!(pv, 1:T, [result.stages[t].vm[i] for t in 1:T]; label = "bus $i", lw = 2)
    end
    fig = plot(pa, pr, pv; layout = (3, 1), size = (860, 860))
    savefig(fig, path)
    return path
end

"""
    plot_paired_costs(a, b, labels, path) -> String

A cost-distribution overlay and the paired-difference histogram for two policies
evaluated on the SAME scenarios.

# Notes
Both panels are drawn because they answer different questions and are routinely
confused. The overlay shows that the two distributions are nearly identical — a
property of the problem, not of the comparison. The paired differences show
whether one policy is systematically cheaper, which the overlay cannot resolve
when the between-scenario spread dwarfs the between-policy difference.
"""
function plot_paired_costs(a::AbstractVector, b::AbstractVector, labels, path::AbstractString)
    d = Float64.(a) .- Float64.(b)
    top = histogram(Float64.(a); label = labels[1], alpha = 0.5, bins = 30,
                    ylabel = "scenarios", legend = :topright)
    histogram!(top, Float64.(b); label = labels[2], alpha = 0.5, bins = 30)
    bot = histogram(d; label = "$(labels[1]) − $(labels[2])", bins = 30,
                    xlabel = "paired cost difference", ylabel = "scenarios",
                    legend = :topright)
    vline!(bot, [0.0]; label = "", lc = :black, ls = :dot)
    vline!(bot, [mean(d)]; label = @sprintf("mean %+.2f", mean(d)), lc = :red, lw = 2)
    fig = plot(top, bot; layout = (2, 1), size = (860, 620))
    savefig(fig, path)
    return path
end

"""
    plot_demand_support(case, path; stages=nothing) -> String

The frozen demand support: the total system demand of every atom at every stage.

# Notes
This is the figure that shows what the uncertainty actually is — how wide the
support is, whether it widens in the stressed window, and whether the
deterministic profile or the multiplier is doing the work. A study that never
plots its own uncertainty tends to discover late that it has almost none.
"""
function plot_demand_support(case::BatteryCase, path::AbstractString; stages = nothing)
    ts = stages === nothing ? (1:horizon(case.demand)) : collect(Int.(stages))
    fig = plot(; xlabel = "stage", ylabel = "total active demand (pu)", legend = :topleft)
    maxK = maximum(num_atoms(case.demand, t) for t in ts)
    for k in 1:maxK
        xs = Int[]
        ys = Float64[]
        for t in ts
            k <= num_atoms(case.demand, t) || continue
            pd, _ = realized_bus_demand(case, t, k)
            push!(xs, t)
            push!(ys, sum(values(pd)))
        end
        plot!(fig, xs, ys; label = "atom $k", lw = 2, seriestype = :steppost)
    end
    savefig(fig, path)
    return path
end
