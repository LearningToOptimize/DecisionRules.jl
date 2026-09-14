#!/usr/bin/env julia

# One schema for a full physical solution of the Bolivia stage problem, shared
# by every engine so their solutions can be differenced variable by variable.
#
# This file is byte-identical in DecisionRules.jl and DecisionRulesExa.jl. It
# depends on nothing but `Printf` and the standard library, so either package can
# include it without pulling in the other's environment.
#
# ── Why a schema at all ───────────────────────────────────────────────────────
#
# Three different pieces of code build the same stage problem: HydroPowerModels
# (which the SDDP baseline drives, and from which `export_subproblem_mof.jl`
# serializes the JuMP/MAIN stage subproblems), the JuMP path that re-reads those
# serialized models, and the ExaModels builder used for GPU training. Agreement
# on the objective is not evidence that they agree as models — two different
# feasible sets can price the same operating point identically. The only
# sufficient check is that, given the SAME incoming state, the SAME inflow and
# the SAME reservoir target, every engine returns the SAME value for EVERY
# physical variable. That requires a common name for every variable, which is
# what this file fixes.
#
# ── Long format ───────────────────────────────────────────────────────────────
#
#   scenario,stage,class,index,value
#
# One row per scalar. Long rather than wide because the classes have different
# lengths (11 hydro units, 28 buses, 34 generators, 31 branches) and because a
# missing row is then an explicit absence rather than a silently empty column.
#
# ── Index conventions ─────────────────────────────────────────────────────────
#
# `index` is ALWAYS the case's own 1-based index, never a solver-internal
# position:
#
# * hydro classes are indexed 1..11 in `hydro.json` `Hydrogenerators` order;
# * bus, generator and branch classes use the `PowerModels.json` `index` field.
#   For this case those run 1..28, 1..34 and 1..31 with no gaps, and the
#   ExaModels builder sorts by the same index, so position and index coincide —
#   `verify_index_convention` asserts that rather than assuming it;
# * scalar, per-stage quantities use index 0.
#
# Branch flows are split by ORIENTATION, not by the tuple JuMP happens to print:
# `p_fr` is the flow measured at the branch's `f_bus` end and `p_to` at its
# `t_bus` end. The serialized JuMP model names both ends `0_p[(l, i, j)]`, so
# the reader has to resolve `(i, j)` against the branch's own endpoints; getting
# this backwards would compare a branch's two ends to each other and could hide
# a real disagreement behind an apparent one, or vice versa.

module HydroSolutionSchema

using Printf

# ── Classes ───────────────────────────────────────────────────────────────────

"""
Hydro-indexed classes, 1..nHyd in `hydro.json` order.

`reservoir_in` is the incoming state of the stage and `reservoir_out` the
outgoing one; in strict mode `reservoir_out` is pinned to `target` by an
equality whose dual is `target_multiplier` — the quantity TS-DDR differentiates
through, so it is compared like any physical variable rather than treated as
diagnostics.
"""
const HYDRO_CLASSES = (
    "reservoir_in", "reservoir_out", "target", "inflow", "outflow", "spill",
    "min_volume_violation", "min_outflow_violation", "target_multiplier",
)

"""
Bus-indexed classes: voltage polar coordinates, the active-power slack, and the
NODAL generation aggregates.

`pg_bus` / `qg_bus` are derived, not read from a solver: they are the sums of
`pg` / `qg` over the generators sitting at a bus (`augment_nodal!`). They are
carried because they, and not the per-generator dispatch, are what the AC nodal
balance determines. Six of this case's buses host several generators — bus 1
hosts ten — and the balance constrains only their sum, so per-unit `qg` has a
null space that two solvers can land in differently while describing the same
operating point. Comparing the aggregates separates "the engines disagree about
the network" from "the engines split an indeterminate quantity differently".
"""
const BUS_CLASSES = (
    "vm", "va", "deficit", "pg_bus", "qg_bus", "price_active", "price_reactive",
)

"""
Bus-indexed DUAL quantities: the locational marginal prices.

`price_active` is the dual of a bus's active-power balance — the marginal cost
of serving one more per-unit of load there for one stage — and `price_reactive`
the dual of its reactive balance. They are the economic read-out of the
dispatch: what the policy's water decisions are actually worth on the network,
and where scarcity is binding.

They exist ONLY as duals, so no primal recording substitutes for them, and they
come from the JuMP engine, which evaluates both policies against the same stage
model.

UNITS AND SIGN, because both are easy to get wrong. The value is the dual of the
balance constraint AS STORED, whose normalized form is
`sum(p_arcs) - sum(pg) + gs*vm^2 == -sum(pd)`; the dual of that is the NEGATIVE
of the conventional locational marginal price, so a more negative number means
energy is more expensive. Its unit is objective units per per-unit injection per
stage — the case's own cost units, not USD/MWh. The meaningful reference on the
same scale is `ACTIVE_DEFICIT_COST = 6000`, the price of shedding a per-unit of
load for one stage: on this case the marginal energy price runs around 1600, so
shedding is roughly four times the cost of serving, which is why the deficit
variables sit at their bound.
"""
const PRICE_CLASSES = ("price_active", "price_reactive")

"""Generator-indexed classes: active and reactive dispatch, per unit."""
const GEN_CLASSES = ("pg", "qg")

"""
Classes whose value is NOT determined by the model.

`min_volume_violation` and `min_outflow_violation` are HydroPowerModels' free
relaxation slacks on the minimum-volume and minimum-outflow bounds. They carry
no objective coefficient and appear in no constraint other than their own
one-sided bound, so any sufficiently large value is optimal and two solvers can
return wildly different ones for the identical model. For this case both
minimums are zero, which makes the slacks inert as well as unpriced.

They are still recorded and still compared — silently dropping a variable is
what this schema exists to prevent — but a difference in them is not evidence of
a model difference, and the parity report says so rather than letting a 1e12
entry sit unexplained in a table of 1e-4 residuals.
"""
const UNPRICED_SLACK_CLASSES = ("min_volume_violation", "min_outflow_violation")

"""Branch-indexed classes: active and reactive flow at each of the two ends."""
const BRANCH_CLASSES = ("p_fr", "p_to", "q_fr", "q_to")

"""Per-stage scalars, written with index 0."""
const SCALAR_CLASSES = ("stage_objective", "cum_objective")

"""Every class this schema knows about, in report order."""
const ALL_CLASSES = (
    HYDRO_CLASSES..., BUS_CLASSES..., GEN_CLASSES..., BRANCH_CLASSES...,
    SCALAR_CLASSES...,
)

"""
    class_group(class) -> String

Which index convention a class uses: `"hydro"`, `"bus"`, `"gen"`, `"branch"` or
`"scalar"`. Used to label the parity report and to check that an index is in
range for its class.
"""
function class_group(class::AbstractString)
    class in HYDRO_CLASSES && return "hydro"
    class in BUS_CLASSES && return "bus"
    class in GEN_CLASSES && return "gen"
    class in BRANCH_CLASSES && return "branch"
    class in SCALAR_CLASSES && return "scalar"
    return error("unknown solution class: $class")
end

# ── Writing ───────────────────────────────────────────────────────────────────

"""
    SolutionWriter(path)

Append-only writer for the long-format solution CSV at `path`.

Floats are written with `repr`, i.e. the shortest decimal string that round-trips
to the same `Float64`. A parity gate that compares two engines at 1e-9 cannot
afford a printf-rounded value: the printed difference would be an artifact of
the formatting rather than of the models.

Close it with `close(writer)`.
"""
mutable struct SolutionWriter
    io::IOStream
    rows::Int
end

function SolutionWriter(path::AbstractString)
    io = open(path, "w")
    println(io, "scenario,stage,class,index,value")
    return SolutionWriter(io, 0)
end

Base.close(writer::SolutionWriter) = close(writer.io)

"""
    record!(writer, scenario, stage, class, index, value) -> Nothing

Write one scalar. `class` must be a known class and `index` its case index
(0 for per-stage scalars).
"""
function record!(
    writer::SolutionWriter,
    scenario::Integer,
    stage::Integer,
    class::AbstractString,
    index::Integer,
    value::Real,
)
    class_group(class)  # validates
    println(writer.io, scenario, ",", stage, ",", class, ",", index, ",", repr(Float64(value)))
    writer.rows += 1
    return nothing
end

"""
    record_vector!(writer, scenario, stage, class, values) -> Nothing

Write a whole class at once, taking `index` from the position in `values`. This
is correct exactly because the case's PowerModels indices are contiguous from 1
(see `verify_index_convention`).
"""
function record_vector!(
    writer::SolutionWriter,
    scenario::Integer,
    stage::Integer,
    class::AbstractString,
    values,
)
    for (i, v) in enumerate(values)
        record!(writer, scenario, stage, class, i, v)
    end
    return nothing
end

"""
    record_scalar!(writer, scenario, stage, class, value) -> Nothing

Write a per-stage scalar at index 0.
"""
record_scalar!(writer, scenario, stage, class, value) =
    record!(writer, scenario, stage, class, 0, value)

# ── Reading ───────────────────────────────────────────────────────────────────

"""
    read_solution(path) -> Dict{Tuple{Int,Int,String,Int},Float64}

Load a long-format solution keyed by `(scenario, stage, class, index)`.

A duplicate key is an error: it would mean the producer wrote the same variable
twice, and silently keeping the last value would hide whichever run was wrong.
"""
function read_solution(path::AbstractString)
    isfile(path) || error("missing solution file: $path")
    out = Dict{Tuple{Int,Int,String,Int},Float64}()
    open(path) do io
        header = readline(io)
        strip(header) == "scenario,stage,class,index,value" ||
            error("$path is not a long-format solution file (header: $header)")
        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = split(line, ',')
            length(parts) == 5 || error("malformed row in $path: $line")
            key = (
                parse(Int, parts[1]), parse(Int, parts[2]),
                String(parts[3]), parse(Int, parts[4]),
            )
            haskey(out, key) && error("duplicate entry $key in $path")
            out[key] = parse(Float64, parts[5])
        end
    end
    return out
end

# ── Comparison ────────────────────────────────────────────────────────────────

"""
One class's worst disagreement between two solutions.

`max_abs` / `max_rel` are the largest absolute and relative differences over
every `(scenario, stage, index)` the class covers; `at` names where the absolute
maximum occurred, and `left` / `right` are the two values there. `scale` is the
denominator used for the relative difference, `max(|a|, |b|, rel_floor)`, so a
variable that is zero in both engines cannot manufacture a relative difference.
"""
struct ClassDifference
    class::String
    group::String
    n::Int
    max_abs::Float64
    max_rel::Float64
    at::Tuple{Int,Int,Int}
    left::Float64
    right::Float64
end

"""
    compare_solutions(left, right; rel_floor=1.0, classes=ALL_CLASSES)
        -> (differences, missing_keys)

Difference two solutions class by class.

`rel_floor` is the smallest magnitude used as a relative-difference denominator.
It defaults to 1.0 because the case's variables are per-unit quantities of order
1 (voltages ~1.0, flows and dispatch < 10, storage < 1); dividing a 1e-9
disagreement by a 1e-12 spill value would report a "relative difference" of
1000 that means nothing physical.

`missing_keys` lists every key present in exactly one of the two solutions. It
is returned rather than tolerated: the gate requires that no physical variable
be silently omitted, and an omitted variable shows up here rather than as a
smaller maximum.
"""
function compare_solutions(
    left::AbstractDict, right::AbstractDict;
    rel_floor::Real = 1.0,
    classes = ALL_CLASSES,
)
    wanted = Set(classes)
    keys_left = Set(k for k in keys(left) if k[3] in wanted)
    keys_right = Set(k for k in keys(right) if k[3] in wanted)
    missing_keys = sort(collect(symdiff(keys_left, keys_right)))

    differences = ClassDifference[]
    for class in classes
        shared = [k for k in keys_left if k[3] == class && haskey(right, k)]
        isempty(shared) && continue
        max_abs = -Inf
        max_rel = 0.0
        at = (0, 0, 0)
        best_left = 0.0
        best_right = 0.0
        for k in shared
            a = left[k]
            b = right[k]
            d = abs(a - b)
            r = d / max(abs(a), abs(b), rel_floor)
            max_rel = max(max_rel, r)
            if d > max_abs
                max_abs = d
                at = (k[1], k[2], k[4])
                best_left = a
                best_right = b
            end
        end
        push!(differences, ClassDifference(
            class, class_group(class), length(shared),
            max_abs, max_rel, at, best_left, best_right,
        ))
    end
    return differences, missing_keys
end

"""
    difference_table(differences) -> String

Render `differences` as a fixed-width table: class, index group, number of
compared scalars, maximum absolute and relative difference, and the
`(scenario, stage, index)` at which the absolute maximum occurred together with
both values there.
"""
function difference_table(differences)
    io = IOBuffer()
    @printf(io, "%-22s %-7s %8s %14s %14s  %-18s %-16s %-16s\n",
            "class", "group", "n", "max|Δ|", "max relΔ", "at (scen,stage,idx)",
            "left", "right")
    println(io, "-"^126)
    for d in differences
        marker = d.class in UNPRICED_SLACK_CLASSES ? " *" : ""
        @printf(io, "%-22s %-7s %8d %14.6e %14.6e  (%5d,%4d,%4d)  %16.9g %16.9g%s\n",
                d.class, d.group, d.n, d.max_abs, d.max_rel,
                d.at[1], d.at[2], d.at[3], d.left, d.right, marker)
    end
    if any(d.class in UNPRICED_SLACK_CLASSES for d in differences)
        println(io)
        println(io, "  * unpriced free slack: zero objective coefficient, no other " *
                    "constraint, no upper bound —")
        println(io, "    the model does not determine its value, so a difference here " *
                    "is not a model difference.")
    end
    return String(take!(io))
end

"""
    generator_bus(case_dir, parsefile) -> Dict{Int,Int}

Generator index to the bus it injects at, from `PowerModels.json`.
"""
function generator_bus(case_dir::AbstractString, parsefile)
    gens = parsefile(joinpath(case_dir, "PowerModels.json"))["gen"]
    return Dict(Int(g["index"]) => Int(g["gen_bus"]) for g in values(gens))
end

"""
    augment_nodal!(solution, gen_bus) -> solution

Add the derived `pg_bus` / `qg_bus` entries to a loaded solution, in place.

Derived here rather than written by each engine so the aggregation is one
implementation shared by both sides: an aggregate that disagreed only because
two producers summed differently would be worse than no aggregate at all.
"""
function augment_nodal!(solution::AbstractDict, gen_bus::AbstractDict)
    totals = Dict{Tuple{Int,Int,String,Int},Float64}()
    for ((scenario, stage, class, index), value) in solution
        (class == "pg" || class == "qg") || continue
        haskey(gen_bus, index) || error("generator $index is not in PowerModels.json")
        key = (scenario, stage, class * "_bus", gen_bus[index])
        totals[key] = get(totals, key, 0.0) + value
    end
    merge!(solution, totals)
    return solution
end

"""
    branch_orientation(case_dir, parsefile) -> Dict{Tuple{Int,Int,Int},Tuple{Bool,Int}}

Map a serialized branch-flow subscript `(l, i, j)` onto `(is_from_end, l)`.

HydroPowerModels/PowerModels names both ends of branch `l` `0_p[(l, i, j)]`,
distinguished only by whether `(i, j)` is `(f_bus, t_bus)` or its reverse.
Resolving that against `PowerModels.json` is what keeps a branch's two ends from
being silently compared to each other.

`parsefile` is passed in (rather than importing JSON here) so this module stays
dependency-free and includable from either package.
"""
function branch_orientation(case_dir::AbstractString, parsefile)
    branches = parsefile(joinpath(case_dir, "PowerModels.json"))["branch"]
    out = Dict{Tuple{Int,Int,Int},Tuple{Bool,Int}}()
    for branch in values(branches)
        l = Int(branch["index"])
        f = Int(branch["f_bus"])
        t = Int(branch["t_bus"])
        out[(l, f, t)] = (true, l)
        out[(l, t, f)] = (false, l)
    end
    return out
end

"""
    solution_class(name, orientation) -> Union{Nothing,Tuple{String,Int}}

Map a serialized JuMP variable name onto `(class, index)` in this schema.

Returns `nothing` for names that are not physical state: the `_`-prefixed JuMP
parameters the DecisionRules loader introduces for the incoming state and the
target, and SDDP's own `_subproblem`-internal bookkeeping variables (the
`theta`/`bellman` cost-to-go term, which is a value-function surrogate and not a
physical quantity — it exists in the SDDP subproblem and cannot exist in a
one-stage model).

Every OTHER name raises: an unrecognized variable must be classified
deliberately, not dropped, or the gate's promise that no physical variable is
silently omitted would be void.
"""
function solution_class(name::AbstractString, orientation)
    startswith(name, "_") && return nothing
    name in ("bellman_term", "theta", "_theta") && return nothing
    m = match(
        r"^(0_va|0_vm|0_pg|0_qg|deficit|inflow|outflow|spill|min_volume_violation|min_outflow_violation)\[(\d+)\]$",
        name,
    )
    if m !== nothing
        class = Dict(
            "0_va" => "va", "0_vm" => "vm", "0_pg" => "pg", "0_qg" => "qg",
            "deficit" => "deficit", "inflow" => "inflow", "outflow" => "outflow",
            "spill" => "spill",
            "min_volume_violation" => "min_volume_violation",
            "min_outflow_violation" => "min_outflow_violation",
        )[m.captures[1]]
        return (class, parse(Int, m.captures[2]))
    end
    m = match(r"^reservoir\[(\d+)\]_(in|out)$", name)
    m !== nothing && return ("reservoir_" * m.captures[2], parse(Int, m.captures[1]))
    m = match(r"^0_(p|q)\[\((\d+), (\d+), (\d+)\)\]$", name)
    if m !== nothing
        key = (
            parse(Int, m.captures[2]), parse(Int, m.captures[3]),
            parse(Int, m.captures[4]),
        )
        haskey(orientation, key) ||
            error("branch subscript $key is not a branch of PowerModels.json")
        from_end, l = orientation[key]
        return (m.captures[1] * (from_end ? "_fr" : "_to"), l)
    end
    return error("unclassified variable name: $name")
end

"""
    verify_index_convention(case_dir) -> Nothing

Assert that the `PowerModels.json` bus, generator and branch indices are exactly
`1:n`, so that a class written by position (`record_vector!`) carries the case's
own index.

If a future case breaks this, every engine's solution would still be written,
but the comparison would silently align different physical objects. Failing here
is the alternative.
"""
function verify_index_convention(case_dir::AbstractString, parsefile)
    power = parsefile(joinpath(case_dir, "PowerModels.json"))
    for section in ("bus", "gen", "branch")
        indices = sort([Int(v["index"]) for v in values(power[section])])
        indices == collect(1:length(indices)) || error(
            "$section indices in PowerModels.json are not 1:$(length(indices)); " *
            "the position-based solution schema would misalign them",
        )
    end
    return nothing
end

export HYDRO_CLASSES, BUS_CLASSES, GEN_CLASSES, BRANCH_CLASSES, SCALAR_CLASSES,
       PRICE_CLASSES, ALL_CLASSES, UNPRICED_SLACK_CLASSES, class_group, SolutionWriter, record!,
       record_vector!, record_scalar!, read_solution, ClassDifference,
       compare_solutions, difference_table, branch_orientation, solution_class,
       generator_bus, augment_nodal!, verify_index_convention

end # module
