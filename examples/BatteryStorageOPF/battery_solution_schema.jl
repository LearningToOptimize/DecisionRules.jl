# battery_solution_schema.jl
#
# The shared, engine-neutral description of a solved battery-storage AC-OPF
# trajectory. This file is shipped BYTE-IDENTICALLY in both public engines, so
# a solution written by the JuMP/PowerModels engine and a solution written by
# the ExaModels engine are the same object and can be differenced by name
# rather than by position.
#
# Format: one long CSV with header
#
#     scenario,stage,class,index,value
#
# `class` names a physical quantity (see `SOLUTION_CLASSES`), `index` is the
# NETWORK identifier of the component it belongs to (bus id, generator id,
# branch id, battery id) — never a positional offset — and `value` is a Float64
# printed with full round-tripping precision. Scalars per stage use index 0.
#
# Why long format and why identifiers. Objective agreement between two engines
# can hide a different feasible set, a null-space variable, or a solver barrier
# offset; only a per-variable comparison catches those, and a per-variable
# comparison is only trustworthy when both sides agree what "variable 7" means.
# Component identifiers in PGLib cases are arbitrary integers and need not be
# consecutive, so positional indexing is not merely fragile — it is wrong.
#
# This file deliberately has no package dependencies beyond `Printf` and the
# standard library: it must be copyable into either engine without dragging a
# resolver conflict behind it.

using Printf

"""
Physical classes a battery-storage solution may record.

Battery-layer classes (index = battery identifier):

- `"energy_in"`   incoming energy ``e_{b,t-1}`` (pu·h)
- `"energy_out"`  outgoing energy ``e_{b,t}`` (pu·h)
- `"target"`      the strict target ``\\hat e_{b,t}`` (pu·h); absent in the
                  targetless SDDP formulation
- `"target_dual"` the multiplier ``\\lambda_{b,t}`` of the strict target
                  equality, i.e. ``\\partial Q/\\partial \\hat e_{b,t}``
- `"p_ch"`        charging power (pu)
- `"p_dis"`       discharging power (pu)
- `"p_bat"`       net active injection ``p^{dis}-p^{ch}`` (pu)

Nodal classes (index = bus identifier):

- `"deficit"`     the uncapped nonnegative recourse injection ``d_{i,t}`` (pu)
- `"surplus"`     the uncapped nonnegative recourse sink ``s_{i,t}`` (pu)
- `"pd"`, `"qd"`  REALIZED active/reactive demand at the bus (pu)
- `"vm"`, `"va"`  voltage magnitude (pu) and angle (rad)
- `"pg_bus"`, `"qg_bus"` generation aggregated to the bus (pu)
- `"price_active"`, `"price_reactive"` nodal duals of the balance, where the
                  engine has them

Generator classes (index = generator identifier): `"pg"`, `"qg"` (pu).

Branch classes (index = branch identifier): `"p_fr"`, `"q_fr"`, `"p_to"`,
`"q_to"` (pu, at the respective ends).

Scalar classes (index 0):

- `"cost_generation"`, `"cost_throughput"`, `"cost_deficit"`, `"cost_surplus"`
- `"cost_stage"`      their sum for the stage
- `"objective"`       the engine's own reported stage objective
- `"residual_equality"`  worst absolute equality-constraint residual
- `"solved"`          1.0 when the engine accepted the solve, 0.0 otherwise
"""
const SOLUTION_CLASSES = (
    "energy_in", "energy_out", "target", "target_dual",
    "p_ch", "p_dis", "p_bat",
    "deficit", "surplus", "pd", "qd", "vm", "va", "pg_bus", "qg_bus",
    "price_active", "price_reactive",
    "pg", "qg",
    "p_fr", "q_fr", "p_to", "q_to",
    "cost_generation", "cost_throughput", "cost_deficit", "cost_surplus",
    "cost_stage", "objective", "residual_equality", "solved",
)

"Header line of every solution CSV."
const SOLUTION_HEADER = "scenario,stage,class,index,value"

"""
    SolutionRecorder

Accumulator for solution records in the shared long format.

# Fields
- `rows::Vector{Tuple{Int,Int,String,Int,Float64}}`: `(scenario, stage, class,
  index, value)` in insertion order.

# Notes
Rows are appended in whatever order an engine produces them; nothing downstream
depends on the order, because comparison is by `(scenario, stage, class,
index)`. Insertion order IS preserved on write so that a diff of two files from
the same engine stays readable.
"""
struct SolutionRecorder
    rows::Vector{Tuple{Int,Int,String,Int,Float64}}
end

SolutionRecorder() = SolutionRecorder(Tuple{Int,Int,String,Int,Float64}[])

"""
    record!(rec, scenario, stage, class, index, value)

Append one record, validating the class name.

# Notes
An unknown class is an ERROR rather than a silently-written row: a typo in a
class name would make the corresponding quantity vanish from a cross-engine
comparison and the comparison would still report "all classes agree".
"""
function record!(rec::SolutionRecorder, scenario::Integer, stage::Integer,
                 class::AbstractString, index::Integer, value::Real)
    class in SOLUTION_CLASSES || error("unknown solution class \"$class\"")
    push!(rec.rows, (Int(scenario), Int(stage), String(class), Int(index), Float64(value)))
    return rec
end

"""
    record_map!(rec, scenario, stage, class, values::AbstractDict)

Append one record per `(identifier => value)` pair, in sorted identifier order.
"""
function record_map!(rec::SolutionRecorder, scenario::Integer, stage::Integer,
                     class::AbstractString, values::AbstractDict)
    for k in sort!(collect(keys(values)))
        record!(rec, scenario, stage, class, k, values[k])
    end
    return rec
end

"""
    write_solution(path, rec::SolutionRecorder)

Write the accumulated records to `path` in the shared long format.

# Notes
Values are printed with `%.17g`, which round-trips every `Float64` exactly, so a
cross-engine difference read back from these files is a difference between the
engines and never a difference introduced by printing.
"""
function write_solution(path::AbstractString, rec::SolutionRecorder)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, SOLUTION_HEADER)
        for (s, t, c, i, v) in rec.rows
            @printf(io, "%d,%d,%s,%d,%.17g\n", s, t, c, i, v)
        end
    end
    return path
end

"""
    physical_residuals(network, batteries, Δt, sol) -> NamedTuple

Recompute the physics of one solved stage from its reported values and return
the worst violation in each class.

# Arguments
- `network::AbstractDict`: the frozen PGLib network (per-unit).
- `batteries`: the case's batteries; each must expose `index`, `bus`,
  `self_discharge`, `charge_efficiency`, `discharge_efficiency`.
- `Δt::Real`: stage duration in hours.
- `sol`: a named tuple or dictionary exposing, keyed by NETWORK identifier,
  `vm`, `va`, `pg`, `qg`, `p_fr`, `q_fr`, `p_to`, `q_to`, `deficit`, `surplus`,
  `pd`, `qd`, `p_ch`, `p_dis`, `energy_in`, `energy_out`.

# Returns
A `NamedTuple` of worst absolute violations:
`(branch_flow, active_balance, reactive_balance, thermal, angle, voltage,
  transition)`.

# Notes
This is deliberately INDEPENDENT of both engines: it re-derives the AC branch
flows from the reported voltages, re-adds the nodal balances from the reported
injections, and re-applies the battery transition to the reported controls. An
engine can therefore be wrong in a way its own solver is happy with and still be
caught here — which is the only kind of check worth running against a manually
written formulation.

Angles are only meaningful for a polar solution. A solution whose `va` entries
are `NaN` (a W-space relaxation has no angle variable) yields `NaN` in the
branch-flow and angle classes, which is honest rather than silently zero.
"""
function physical_residuals(network::AbstractDict, batteries, Δt::Real, sol)
    vm, va = sol.vm, sol.va
    worst_flow = 0.0
    worst_thermal = 0.0
    worst_angle = 0.0

    inj_p = Dict{Int,Float64}(k => 0.0 for k in keys(vm))
    inj_q = Dict{Int,Float64}(k => 0.0 for k in keys(vm))

    for (_, br) in network["branch"]
        Int(get(br, "br_status", 1)) == 0 && continue
        l = Int(br["index"])
        haskey(sol.p_fr, l) || continue
        f, t = Int(br["f_bus"]), Int(br["t_bus"])
        r, x = Float64(get(br, "br_r", 0.0)), Float64(br["br_x"])
        r2x2 = r^2 + x^2
        g = r2x2 > 0 ? r / r2x2 : 0.0
        b = r2x2 > 0 ? -x / r2x2 : 0.0
        tap = Float64(get(br, "tap", 1.0)); tap = tap ≈ 0 ? 1.0 : tap
        shift = Float64(get(br, "shift", 0.0))
        tr, ti = tap * cos(shift), tap * sin(shift)
        ttm = tr^2 + ti^2; ttm = ttm > 0 ? ttm : 1.0
        g_fr, b_fr = Float64(get(br, "g_fr", 0.0)), Float64(get(br, "b_fr", 0.0))
        g_to, b_to = Float64(get(br, "g_to", 0.0)), Float64(get(br, "b_to", 0.0))

        vf, vt = vm[f], vm[t]
        θ = va[f] - va[t]
        pfr = (g + g_fr) / ttm * vf^2 + (-g * tr + b * ti) / ttm * vf * vt * cos(θ) +
              (-b * tr - g * ti) / ttm * vf * vt * sin(θ)
        qfr = -(b + b_fr) / ttm * vf^2 - (-b * tr - g * ti) / ttm * vf * vt * cos(θ) +
              (-g * tr + b * ti) / ttm * vf * vt * sin(θ)
        pto = (g + g_to) * vt^2 + (-g * tr - b * ti) / ttm * vt * vf * cos(-θ) +
              (-b * tr + g * ti) / ttm * vt * vf * sin(-θ)
        qto = -(b + b_to) * vt^2 - (-b * tr + g * ti) / ttm * vt * vf * cos(-θ) +
              (-g * tr - b * ti) / ttm * vt * vf * sin(-θ)

        worst_flow = max(worst_flow, abs(pfr - sol.p_fr[l]), abs(qfr - sol.q_fr[l]),
                         abs(pto - sol.p_to[l]), abs(qto - sol.q_to[l]))

        rate = Float64(get(br, "rate_a", Inf))
        if isfinite(rate)
            worst_thermal = max(worst_thermal,
                                sol.p_fr[l]^2 + sol.q_fr[l]^2 - rate^2,
                                sol.p_to[l]^2 + sol.q_to[l]^2 - rate^2)
        end
        amin = Float64(get(br, "angmin", -pi)); amax = Float64(get(br, "angmax", pi))
        worst_angle = max(worst_angle, amin - θ, θ - amax)

        inj_p[f] -= sol.p_fr[l]; inj_q[f] -= sol.q_fr[l]
        inj_p[t] -= sol.p_to[l]; inj_q[t] -= sol.q_to[l]
    end

    for (_, gen) in network["gen"]
        Int(get(gen, "gen_status", 1)) == 0 && continue
        gi = Int(gen["index"])
        haskey(sol.pg, gi) || continue
        bus = Int(gen["gen_bus"])
        inj_p[bus] += sol.pg[gi]
        inj_q[bus] += sol.qg[gi]
    end

    for (_, sh) in get(network, "shunt", Dict{String,Any}())
        Int(get(sh, "status", 1)) == 0 && continue
        bus = Int(sh["shunt_bus"])
        haskey(inj_p, bus) || continue
        inj_p[bus] -= Float64(get(sh, "gs", 0.0)) * vm[bus]^2
        inj_q[bus] += Float64(get(sh, "bs", 0.0)) * vm[bus]^2
    end

    worst_transition = 0.0
    for b in batteries
        haskey(sol.p_ch, b.index) || continue
        inj_p[b.bus] += sol.p_dis[b.index] - sol.p_ch[b.index]
        lhs = sol.energy_out[b.index] - b.self_discharge * sol.energy_in[b.index] -
              b.charge_efficiency * Δt * sol.p_ch[b.index] +
              (Δt / b.discharge_efficiency) * sol.p_dis[b.index]
        worst_transition = max(worst_transition, abs(lhs))
    end

    worst_p = 0.0
    worst_q = 0.0
    worst_v = 0.0
    for (_, bus) in network["bus"]
        i = Int(bus["index"])
        haskey(inj_p, i) || continue
        worst_p = max(worst_p, abs(inj_p[i] + sol.deficit[i] - sol.surplus[i] - sol.pd[i]))
        worst_q = max(worst_q, abs(inj_q[i] - sol.qd[i]))
        worst_v = max(worst_v, Float64(get(bus, "vmin", 0.0)) - vm[i],
                      vm[i] - Float64(get(bus, "vmax", Inf)))
    end

    return (branch_flow = worst_flow, active_balance = worst_p,
            reactive_balance = worst_q, thermal = worst_thermal,
            angle = worst_angle, voltage = worst_v, transition = worst_transition)
end

"""
    read_solution(path) -> Dict{Tuple{Int,Int,String,Int},Float64}

Read a solution file into a lookup keyed by `(scenario, stage, class, index)`.

# Notes
Duplicate keys are an ERROR. Two rows claiming the same physical quantity mean
the writer lost track of what it was recording, and silently keeping the last
one would make a parity comparison depend on file order.
"""
function read_solution(path::AbstractString)
    out = Dict{Tuple{Int,Int,String,Int},Float64}()
    open(path, "r") do io
        header = readline(io)
        header == SOLUTION_HEADER ||
            error("$path: unexpected header \"$header\"; expected \"$SOLUTION_HEADER\"")
        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = split(line, ',')
            length(parts) == 5 || error("$path: malformed row \"$line\"")
            key = (parse(Int, parts[1]), parse(Int, parts[2]), String(parts[3]), parse(Int, parts[4]))
            haskey(out, key) && error("$path: duplicate record for $key")
            out[key] = parse(Float64, parts[5])
        end
    end
    return out
end
