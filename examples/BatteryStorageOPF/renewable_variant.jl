# Renewable/storage variants of a frozen battery case.
#
# A variant is a TRANSFORMATION of a frozen case, not a new case family: the
# host network, its battery buses, its temporal profile and its cost data all
# come from the base case, and this file only applies the edits named in
# `renewable_variants.toml`. That is what lets the same recipe be carried to
# another host without rewriting anything, and what lets a reader reconstruct a
# variant from the base case plus one TOML file.
#
# Four edits, in this order:
#
#   1. assign every bus a REGION (a role, assigned by the recipe — not a
#      property discovered in the case);
#   2. apply a GENERATION TRANSITION, deleting the units the recipe retires;
#   3. add RENEWABLE units with a per-stage availability schedule;
#   4. RESIZE the battery fleet by region, preserving the base case's total
#      power and total energy exactly.
#
# and one optional convention: a matched initial/terminal inventory.
#
# Commands:
#   julia --project=. renewable_variant.jl --base DIR --out DIR --template reliable

using JSON
using Printf
using TOML

@isdefined(read_battery_case) || include(joinpath(@__DIR__, "battery_case.jl"))

"Schema tag of the renewable-variant recipe."
const RENEWABLE_VARIANT_SCHEMA = "battery_storage_opf/renewable_variants/1"

"Default location of the recipe, beside this file."
renewable_variants_path(dir::AbstractString = @__DIR__) =
    joinpath(dir, "renewable_variants.toml")

# ─────────────────────────────────────────────────────────────────────────────
# A — the recipe
# ─────────────────────────────────────────────────────────────────────────────

"""
    renewable_variant_recipe(path = renewable_variants_path()) -> Dict

Read and check the variant recipe.

# Notes
The schema tag is checked here and nowhere else, so a recipe written for a later
version of this file fails at the first call rather than silently producing a
case built from half of its fields.
"""
function renewable_variant_recipe(path::AbstractString = renewable_variants_path())
    isfile(path) || error("no renewable-variant recipe at $path")
    rec = TOML.parsefile(path)
    get(rec, "schema", nothing) == RENEWABLE_VARIANT_SCHEMA ||
        error("recipe schema $(get(rec, "schema", nothing)); expected $RENEWABLE_VARIANT_SCHEMA")
    haskey(rec, "regions") || error("recipe has no [regions]")
    haskey(rec, "transition") || error("recipe has no [transition]")
    haskey(rec, "renewables") || error("recipe has no [renewables]")
    return rec
end

# ─────────────────────────────────────────────────────────────────────────────
# B — regions
# ─────────────────────────────────────────────────────────────────────────────

"""
    variant_bus_region(network, recipe) -> Dict{Int,String}

Map every bus identifier of `network` to the name of the region it belongs to.

# Notes
A zone named by no region is an error rather than a silent assignment to the
rest-of-system region: a recipe that forgot a zone has forgotten part of the
host, and the resulting case would carry a region map nobody wrote down.
"""
function variant_bus_region(network::AbstractDict, recipe::AbstractDict)
    zone_of_region = Dict{Int,String}()
    for (name, spec) in recipe["regions"], z in spec["zones"]
        haskey(zone_of_region, z) &&
            error("zone $z is claimed by both $(zone_of_region[z]) and $name")
        zone_of_region[z] = name
    end
    out = Dict{Int,String}()
    for (_, bus) in network["bus"]
        i = Int(bus["bus_i"])
        z = Int(bus["zone"])
        haskey(zone_of_region, z) || error("zone $z (bus $i) is named by no region")
        out[i] = zone_of_region[z]
    end
    return out
end

"""
    variant_region_names(recipe) -> Vector{String}

Region names in a deterministic order: the recipe's own key order is a `Dict`
order and must never reach a result.
"""
variant_region_names(recipe::AbstractDict) = sort!(collect(keys(recipe["regions"])))

# ─────────────────────────────────────────────────────────────────────────────
# C — generator economics
# ─────────────────────────────────────────────────────────────────────────────

"""
    gen_cost_coefficients(gen) -> (c2, c1, c0)

The generator's cost polynomial padded to degree two, in the per-unit basis
PowerModels stores it in:

```math
C(p) = c_2 p^2 + c_1 p + c_0 ,\\qquad p \\text{ in per unit}.
```

# Notes
`ncost` is the number of coefficients actually present, highest order first, so
a linear cost is `[c1, c0]` and a constant cost is `[c0]`. Padding here means no
caller has to branch on `ncost`, which is exactly the branch that gets forgotten.
"""
function gen_cost_coefficients(gen::AbstractDict)
    c = gen["cost"]
    n = Int(gen["ncost"])
    n == 3 && return (Float64(c[1]), Float64(c[2]), Float64(c[3]))
    n == 2 && return (0.0, Float64(c[1]), Float64(c[2]))
    n == 1 && return (0.0, 0.0, Float64(c[1]))
    return (0.0, 0.0, 0.0)
end

"""
    gen_marginal_cost(gen, p, baseMVA) -> Float64

Marginal cost ``dC/dp`` at output `p` (per unit), converted to currency per MWh:

```math
\\frac{dC}{dp}(p) = 2 c_2 p + c_1
\\quad\\text{[currency per pu·h]},\\qquad
\\text{divide by } \\texttt{baseMVA} \\text{ for currency per MWh.}
```
"""
function gen_marginal_cost(gen::AbstractDict, p::Real, baseMVA::Real)
    c2, c1, _ = gen_cost_coefficients(gen)
    return (2 * c2 * p + c1) / baseMVA
end

"""
    gen_average_variable_cost(gen, baseMVA) -> Float64

Average variable cost at RATED output, in currency per MWh:

```math
\\mathrm{avc} = \\frac{c_2 \\overline p^2 + c_1 \\overline p}{\\overline p}
             \\Big/ \\texttt{baseMVA}
             = \\frac{c_2 \\overline p + c_1}{\\texttt{baseMVA}} .
```

# Notes
This is the measure the transition rule uses, and it is not interchangeable with
the linear coefficient. On a host where many units carry a nonzero quadratic
term, a unit with `c1 = 0` can have an average variable cost of tens of currency
units per MWh; ordering by `c1` would retire it as if it were free. The constant
term is deliberately absent: it is charged whether or not the unit generates and
so says nothing about the cost of energy from it.

Returns `0.0` for a unit with `pmax == 0`, which has no rated output to average
over and is never the marginal unit anyway.
"""
function gen_average_variable_cost(gen::AbstractDict, baseMVA::Real)
    pmax = Float64(gen["pmax"])
    pmax > 0 || return 0.0
    c2, c1, _ = gen_cost_coefficients(gen)
    return (c2 * pmax + c1) / baseMVA
end

# ─────────────────────────────────────────────────────────────────────────────
# D — the generation transition
# ─────────────────────────────────────────────────────────────────────────────

"""
    variant_retirements(network, recipe) -> Vector{Int}

Generator indices the transition retires, sorted ascending.

# Notes
Only IN-SERVICE units are candidates: PGLib files carry out-of-service
generators that PowerModels never puts in `ref`, and retiring one of those would
be a retirement that changes nothing while appearing in the record as if it had.

The two exception rules are applied AFTER the threshold, so the record shows a
one-line rule plus a named list of units kept in spite of it, rather than a
threshold tuned until the right units fell out.
"""
function variant_retirements(network::AbstractDict, recipe::AbstractDict)
    tr = recipe["transition"]
    tr["measure"] == "average_variable_cost_at_pmax" ||
        error("unknown transition measure $(tr["measure"])")
    baseMVA = Float64(network["baseMVA"])
    θ = Float64(tr["threshold_usd_per_mwh"])

    live = [g for (_, g) in network["gen"] if Int(get(g, "gen_status", 1)) == 1]
    sort!(live; by = g -> Int(g["index"]))

    # Geographic exemption, applied first and by PLACE: a generator standing in
    # an exempt zone is not a candidate at all, whatever its cost. Resolving it
    # through the bus's own `zone` field rather than through a list of indices is
    # what makes it survive a change of host or of generator numbering.
    exempt_zones = Set{Int}(Int.(get(tr, "exempt_zones", Int[])))
    zone_of_bus = Dict{Int,Int}(Int(b["bus_i"]) => Int(b["zone"]) for (_, b) in network["bus"])
    in_exempt_zone(g) = get(zone_of_bus, Int(g["gen_bus"]), -1) in exempt_zones

    keep = Set{Int}(Int.(get(tr, "keep_extra_gen_indices", Int[])))
    # Restored-for-service units are excluded from the retirement exactly as the
    # adequacy exceptions are, so they keep their own bounds, cost curve and
    # reactive limits. They are listed separately in the recipe and recorded
    # separately in the manifest so the two reasons never blur together.
    union!(keep, Set{Int}(Int.(get(tr, "restore_gen_indices", Int[]))))
    if get(tr, "keep_reference_bus_unit", false)
        refs = sort!([Int(b["bus_i"]) for (_, b) in network["bus"] if Int(b["bus_type"]) == 3])
        length(refs) == 1 || error("expected exactly one reference bus, found $(length(refs))")
        at_ref = [g for g in live if Int(g["gen_bus"]) == refs[1] &&
                                     gen_average_variable_cost(g, baseMVA) <= θ &&
                                     !in_exempt_zone(g)]
        if !isempty(at_ref)
            # Lowest average variable cost, ties broken by index: two identical
            # units at the reference bus must not make the case depend on which
            # one a JSON reader happened to hand back first.
            best = first(sort(at_ref; by = g -> (gen_average_variable_cost(g, baseMVA),
                                                 Int(g["index"]))))
            push!(keep, Int(best["index"]))
        end
    end
    return sort!([Int(g["index"]) for g in live
                  if gen_average_variable_cost(g, baseMVA) <= θ &&
                     !(Int(g["index"]) in keep) && !in_exempt_zone(g)])
end

# ─────────────────────────────────────────────────────────────────────────────
# E — the renewable build
# ─────────────────────────────────────────────────────────────────────────────

"""
    variant_availability(recipe, region, template, horizon) -> Vector{Float64}

The per-stage availability multipliers of `region` under `template`.

# Notes
A key `"<region>_<template>"` overrides the plain `"<region>"` key, which is how
one region carries a template-dependent profile while the others do not. Every
multiplier is checked to lie in ``[0, 1]``: an availability is a fraction of
installed capacity, and a value above one would install capacity the recipe never
declared.
"""
function variant_availability(recipe::AbstractDict, region::AbstractString,
                              template::AbstractString, horizon::Integer)
    av = recipe["renewables"]["availability"]
    key = "$(region)_$(template)"
    haskey(av, key) || (key = region)
    haskey(av, key) || error("no availability for region $region (template $template)")
    v = Float64.(av[key])
    length(v) == horizon ||
        error("availability for $key covers $(length(v)) stages, case horizon is $horizon")
    all(x -> 0.0 <= x <= 1.0, v) ||
        error("availability for $key leaves [0, 1]")
    return v
end

"""
    variant_renewable_units(network, recipe, bus_region, template) -> Vector{Dict}

Build the renewable generators, one per host generator bus of each region that
the recipe gives installed capacity.

# Notes
**Placement.** Capacity is split over the buses of the region's existing
in-service generators in proportion to their original `pmax`. No interconnection
point is invented, which keeps the build on buses the host already energises at
transmission level, and the proportional split keeps a single small generator bus
from receiving a plant larger than its surroundings.

**Reactive capability** is `qmax = -qmin = qmax_fraction * pmax` and is scaled by
availability along with the active limits, because `apply_stage_availability!`
scales all four bounds by one multiplier. A unit at zero availability therefore
supplies no reactive power. That is a MODELLING ASSUMPTION recorded in the
recipe, not a property of the resource: an inverter can be specified to hold
reactive capability at zero irradiance, and a study that wanted that would set a
separate reactive schedule.

**Cost** is identically zero, including the constant term, so curtailment is free
and is measured as `availability * pmax - pg` rather than inferred from a price.

New indices continue from the host's largest generator index, so no existing
unit's identity changes and a variant's generator indices can be compared with
the base case's.
"""
function variant_renewable_units(network::AbstractDict, recipe::AbstractDict,
                                 bus_region::AbstractDict, template::AbstractString,
                                 horizon::Integer)
    rb = recipe["renewables"]
    rb["placement_rule"] == "proportional_to_regional_gen_pmax" ||
        error("unknown placement rule $(rb["placement_rule"])")
    qfrac = Float64(rb["qmax_fraction"])
    cost = Float64.(rb["cost"])

    live = [g for (_, g) in network["gen"] if Int(get(g, "gen_status", 1)) == 1]
    next_index = maximum(Int(g["index"]) for (_, g) in network["gen"]) + 1

    units = Dict{String,Any}[]
    for region in sort!(collect(keys(rb["installed_pu"])))
        total = Float64(rb["installed_pu"][region])
        total > 0 || continue
        sched = variant_availability(recipe, region, template, horizon)

        # Weight per BUS, not per generator: two units at one bus are one
        # interconnection point and must not receive twice the share.
        weight = Dict{Int,Float64}()
        for g in live
            b = Int(g["gen_bus"])
            get(bus_region, b, "") == region || continue
            weight[b] = get(weight, b, 0.0) + Float64(g["pmax"])
        end
        isempty(weight) && error("region $region has no in-service generator bus to build on")
        wsum = sum(values(weight))

        for b in sort!(collect(keys(weight)))
            pmax = total * weight[b] / wsum
            push!(units, Dict{String,Any}(
                "index" => next_index,
                "gen_bus" => b,
                "gen_status" => 1,
                "pg" => 0.0, "qg" => 0.0,
                "pmax" => pmax, "pmin" => Float64(rb["pmin_pu"]),
                "qmax" => qfrac * pmax, "qmin" => -qfrac * pmax,
                "vg" => 1.0, "mbase" => Float64(network["baseMVA"]),
                "model" => 2, "ncost" => length(cost), "cost" => cost,
                "startup" => 0.0, "shutdown" => 0.0,
                "source_id" => Any["gen", next_index],
                "rs_region" => region,
                STAGE_AVAILABILITY_KEY => sched,
            ))
            next_index += 1
        end
    end
    return units
end

# ─────────────────────────────────────────────────────────────────────────────
# F — the fleet
# ─────────────────────────────────────────────────────────────────────────────

"""
    variant_fleet(batteries, bus_region, recipe) -> (Vector{BatterySpec}, Dict)

Resize the fleet by region, preserving the base case's TOTAL power and TOTAL
energy exactly.

# Notes
The fleet is resized, never relocated: unit count, identities and buses are the
base case's, and only the ratings change. Relocation would change two things at
once — where storage is and how much of it there is — and the comparison this
case exists for is about the first.

Each region carries two independent factors, ``\\phi^P_r`` and ``\\phi^E_r``,
so a region's DURATION changes with its split:

```math
\\phi^P_r = \\frac{P^{target}_r}{\\sum_{b \\in r} \\overline p^{dis}_b},\\qquad
\\phi^E_r = \\frac{E^{target}_r}{\\sum_{b \\in r} \\overline e_b}.
```

`energy_min` and `energy_initial` scale with ``\\phi^E_r`` so the reserve
fraction and the initial state of charge are unchanged by the resizing; the
boundary convention, if the recipe declares one, then overwrites
`energy_initial`.

The region named by `storage.balance` receives the exact arithmetic remainder of
the totals rather than a figure written in the recipe, so the budgets close to
floating-point equality instead of to the number of decimals someone typed.
"""
function variant_fleet(batteries::AbstractVector{BatterySpec},
                       bus_region::AbstractDict, recipe::AbstractDict)
    st = recipe["storage"]
    base_p = sum(b.discharge_max for b in batteries)
    base_e = sum(b.energy_max for b in batteries)

    targets_p = Dict{String,Float64}()
    targets_e = Dict{String,Float64}()
    for (r, spec) in st["region"]
        targets_p[r] = Float64(spec["power_pu"])
        targets_e[r] = Float64(spec["energy_puh"])
    end
    bal = String(st["balance"])
    haskey(targets_p, bal) &&
        error("the balancing region $bal must not also carry an explicit target")
    targets_p[bal] = base_p - sum(values(targets_p))
    targets_e[bal] = base_e - sum(values(targets_e))
    targets_p[bal] > 0 && targets_e[bal] > 0 ||
        error("the explicit regional targets exhaust the frozen budget; $bal would receive $(targets_p[bal]) pu")

    have_p = Dict{String,Float64}(); have_e = Dict{String,Float64}()
    for b in batteries
        r = bus_region[b.bus]
        have_p[r] = get(have_p, r, 0.0) + b.discharge_max
        have_e[r] = get(have_e, r, 0.0) + b.energy_max
    end
    for r in keys(targets_p)
        get(have_p, r, 0.0) > 0 ||
            error("region $r is given a storage target but hosts no battery of the base case")
    end
    # ...and the converse, which is the silent one: a region the recipe forgot
    # would get a factor of zero and its batteries would vanish from a fleet the
    # totals still say is intact, because the balancing region would absorb
    # their power and energy.
    for r in keys(have_p)
        haskey(targets_p, r) ||
            error("region $r hosts batteries of the base case but the recipe gives it no storage target")
    end

    out = BatterySpec[]
    for b in batteries
        r = bus_region[b.bus]
        φp = get(targets_p, r, 0.0) / have_p[r]
        φe = get(targets_e, r, 0.0) / have_e[r]
        push!(out, BatterySpec(b.index, b.bus,
                               b.energy_min * φe, b.energy_max * φe, b.energy_initial * φe,
                               b.charge_max * φp, b.discharge_max * φp,
                               b.charge_efficiency, b.discharge_efficiency,
                               b.self_discharge, b.throughput_cost))
    end
    report = Dict{String,Any}(
        "rule" => "resize by region; unit count and buses unchanged",
        "balance_region" => bal,
        "target_power_pu" => Dict{String,Any}(targets_p),
        "target_energy_puh" => Dict{String,Any}(targets_e),
        "base_power_pu" => base_p, "base_energy_puh" => base_e,
    )
    return out, report
end

"""
    apply_boundary_convention(batteries, recipe) -> (Vector{BatterySpec}, Dict)

Set each battery's initial energy to a declared fraction of its USABLE energy
above the floor, and report the terminal requirement that must accompany it:

```math
e^{0}_b = \\underline e_b + f\\,(\\overline e_b - \\underline e_b),
\\qquad e^{T}_b \\ge e^{0}_b .
```

# Notes
Opt-in. A recipe without a `[boundary]` section leaves the fleet alone, which is
the behaviour every case frozen before this file existed relies on.

Returning the terminal requirement rather than imposing it here is deliberate:
the requirement belongs to every model, policy layer and comparator built on the
case, so it is recorded in the case manifest and read from there, not rebuilt by
each caller from a fraction each caller has to remember.
"""
function apply_boundary_convention(batteries::AbstractVector{BatterySpec},
                                   recipe::AbstractDict)
    haskey(recipe, "boundary") || return collect(batteries), Dict{String,Any}()
    bd = recipe["boundary"]
    f = Float64(bd["initial_fraction_of_usable"])
    0.0 <= f <= 1.0 || error("initial_fraction_of_usable must lie in [0, 1], got $f")
    out = BatterySpec[]
    floor_of = Dict{String,Any}()
    for b in batteries
        e0 = b.energy_min + f * (b.energy_max - b.energy_min)
        push!(out, BatterySpec(b.index, b.bus, b.energy_min, b.energy_max, e0,
                               b.charge_max, b.discharge_max,
                               b.charge_efficiency, b.discharge_efficiency,
                               b.self_discharge, b.throughput_cost))
        floor_of[string(b.index)] = e0
    end
    report = Dict{String,Any}(
        "initial_fraction_of_usable" => f,
        "initial_energy" => "e0 = e_min + f*(e_max - e_min)",
        "terminal_at_least_initial" => Bool(get(bd, "terminal_at_least_initial", false)),
        "terminal_energy_min" => Bool(get(bd, "terminal_at_least_initial", false)) ?
                                 floor_of : nothing,
    )
    return out, report
end

# ─────────────────────────────────────────────────────────────────────────────
# G — demand
# ─────────────────────────────────────────────────────────────────────────────

"""
    deterministic_support(support) -> DemandSupport

Collapse a finite demand support to ONE atom of multiplier 1 at every stage,
keeping the temporal profile, the load order, the stage duration and the
protocol seed.

# Notes
The realized demand of load ``j`` at stage ``t`` becomes
``p^0_j h_{j,t}`` exactly — the base case's profile with no shock — so a variant
built this way is deterministic in demand while remaining, in every other
respect, the same artifact the stochastic recipes will extend. The protocol seed
is carried through unchanged so that a later stochastic version of the same host
draws the same evaluation scenario indices.
"""
function deterministic_support(support::DemandSupport)
    n = num_loads(support)
    atoms = [ones(Float64, n, 1) for _ in 1:support.horizon]
    probs = [[1.0] for _ in 1:support.horizon]
    source = Dict{String,Any}(support.source)
    source["method"] = "deterministic_profile"
    source["collapsed_from"] = Dict{String,Any}(
        "atoms_per_stage" => [size(a, 2) for a in support.atoms],
        "note" => "every stage carries one atom of multiplier 1; the profile is unchanged",
    )
    return DemandSupport(support.stage_hours, support.horizon, copy(support.load_ids),
                         copy(support.profile), atoms, probs, support.protocol_seed, source)
end

# ─────────────────────────────────────────────────────────────────────────────
# H — the builder
# ─────────────────────────────────────────────────────────────────────────────

"""
    colocate_fleet(fleet, units, bus_region, recipe) -> (Vector{BatterySpec}, Dict)

Move the source regions' batteries onto their own region's renewable injection
buses, changing nothing about the batteries but where they stand.

# Notes
Every rating, efficiency, cost, initial inventory and terminal requirement is
carried across: `BatterySpec` is rebuilt with a new `bus` and identical
everything else, and the terminal floor is keyed by battery index, so it follows
the unit rather than the site. Regions outside `storage.colocate.regions` are
untouched.

The assignment minimises, for each battery in turn, the ratio of assigned
battery energy to installed renewable capacity at the candidate bus. Sorting the
batteries by descending energy first makes the result independent of the order
the fleet happens to be stored in, and the bus-identifier tie-break makes it
independent of dictionary order — the same two properties every other
construction rule in this study is built to have.
"""
function colocate_fleet(fleet::AbstractVector{BatterySpec}, units::AbstractVector,
                        bus_region::AbstractDict, recipe::AbstractDict)
    spec = get(get(recipe, "storage", Dict()), "colocate", nothing)
    spec === nothing && return collect(fleet), Dict{String,Any}("applied" => false)
    spec["rule"] == "greedy_energy_per_installed_capacity" ||
        error("unknown colocation rule $(spec["rule"])")
    regions = Set{String}(String.(spec["regions"]))

    # Installed renewable capacity per bus, per region.
    cap = Dict{String,Dict{Int,Float64}}()
    for u in units
        r = String(u["rs_region"])
        r in regions || continue
        d = get!(cap, r, Dict{Int,Float64}())
        d[Int(u["gen_bus"])] = get(d, Int(u["gen_bus"]), 0.0) + Float64(u["pmax"])
    end

    assigned = Dict{String,Dict{Int,Float64}}(r => Dict(b => 0.0 for b in keys(d))
                                              for (r, d) in cap)
    out = Vector{BatterySpec}(undef, length(fleet))
    moves = Dict{String,Any}[]
    order = sortperm(collect(fleet); by = b -> (-b.energy_max, b.index))
    for i in order
        b = fleet[i]
        r = bus_region[b.bus]
        if !(r in regions) || !haskey(cap, r) || isempty(cap[r])
            out[i] = b
            continue
        end
        buses = sort!(collect(keys(cap[r])))
        best = argmin([( (assigned[r][u] + b.energy_max) / cap[r][u], u) for u in buses])
        u = buses[best]
        assigned[r][u] += b.energy_max
        out[i] = BatterySpec(b.index, u, b.energy_min, b.energy_max, b.energy_initial,
                             b.charge_max, b.discharge_max, b.charge_efficiency,
                             b.discharge_efficiency, b.self_discharge, b.throughput_cost)
        b.bus == u || push!(moves, Dict{String,Any}(
            "battery" => b.index, "region" => r, "from_bus" => b.bus, "to_bus" => u,
            "energy_max" => b.energy_max, "power_max" => b.discharge_max,
            "installed_renewable_pmax_at_target" => cap[r][u]))
    end
    report = Dict{String,Any}("applied" => true, "rule" => spec["rule"],
        "regions" => sort!(collect(regions)), "moved" => length(moves),
        "unmoved" => length(fleet) - length(moves), "mapping" => moves,
        "energy_per_bus" => Dict(r => Dict(string(k) => v for (k, v) in d)
                                 for (r, d) in assigned),
        "installed_per_bus" => Dict(r => Dict(string(k) => v for (k, v) in d)
                                    for (r, d) in cap))
    return out, report
end

"""
    build_renewable_variant(base_dir, out_dir; recipe, template, verify) -> Dict

Build one renewable/storage variant of the frozen case in `base_dir` and write it
to `out_dir`. Returns the variant's case manifest.

# Arguments
- `base_dir::AbstractString`: a directory written by [`write_battery_case`](@ref).
- `out_dir::AbstractString`: where the variant is written. Created if absent.

# Keywords
- `recipe`: a parsed recipe, or the path of one. Defaults to the recipe beside
  this file.
- `template::AbstractString`: which weather template's availability to freeze
  into the case. Defaults to the recipe's `templates.nominal`.
- `verify::Bool`: passed to [`read_battery_case`](@ref) for the base case.

# Notes
The availability schedule is frozen INTO the case, so a template is a case, not a
runtime flag. Two templates are two directories that differ in exactly one
artifact, which is what makes a comparison between them a comparison of the
weather and of nothing else.
"""
function build_renewable_variant(base_dir::AbstractString, out_dir::AbstractString;
                                 recipe = renewable_variants_path(),
                                 template::Union{Nothing,AbstractString} = nothing,
                                 colocate::Bool = false,
                                 verify::Bool = true)
    rec = recipe isa AbstractDict ? recipe : renewable_variant_recipe(recipe)
    tpl = template === nothing ? String(rec["templates"]["nominal"]) : String(template)
    tpl in String.(rec["templates"]["names"]) ||
        error("template $tpl is not one of $(rec["templates"]["names"])")

    base = read_battery_case(base_dir; verify = verify)
    net = deepcopy(base.network)
    H = horizon(base.demand)
    bus_region = variant_bus_region(net, rec)

    # 1. transition
    retired = variant_retirements(net, rec)
    # The build is placed on the host's ORIGINAL in-service generator buses,
    # computed BEFORE the deletions below. That is what "renewables replace the
    # retired plants" means physically — the retired sites keep their
    # interconnection — and it also keeps the placement defined for a region
    # that retires every unit it has.
    units = variant_renewable_units(net, rec, bus_region, tpl, H)
    retired_record = Dict{String,Any}[]
    # One pass over the generator dictionary, keyed by the generator's OWN
    # index: the dictionary key is a string that usually equals it and is not
    # required to, and a variant that deleted the wrong unit would be a case
    # nobody could reconstruct from the recipe.
    key_of = Dict{Int,String}()
    for (k, g) in net["gen"]
        i = Int(g["index"])
        haskey(key_of, i) && error("two generators carry index $i")
        key_of[i] = k
    end
    # Record what the restoration actually puts back, in full: a restored unit
    # with a positive minimum output is a commitment, not an option.
    restored_record = Dict{String,Any}[]
    for idx in Int.(get(rec["transition"], "restore_gen_indices", Int[]))
        haskey(key_of, idx) || error("restored generator $idx is not in the network")
        g = net["gen"][key_of[idx]]
        c2, c1, c0 = gen_cost_coefficients(g)
        push!(restored_record, Dict{String,Any}(
            "index" => idx, "bus" => Int(g["gen_bus"]),
            "region" => bus_region[Int(g["gen_bus"])],
            "pmax" => Float64(g["pmax"]), "pmin" => Float64(g["pmin"]),
            "qmax" => Float64(g["qmax"]),
            "avc_usd_per_mwh" => gen_average_variable_cost(g, Float64(net["baseMVA"])),
            "marginal_at_pmin_usd_per_mwh" => gen_marginal_cost(g, Float64(g["pmin"]), Float64(net["baseMVA"])),
            "marginal_at_pmax_usd_per_mwh" => gen_marginal_cost(g, Float64(g["pmax"]), Float64(net["baseMVA"])),
            "constant_cost" => c0,
            "note" => "restored intact: original active bounds, cost curve and reactive limits"))
    end

    # What the geographic exemption preserved: the units the threshold WOULD have
    # retired had their zone not been exempt. Recorded in full, because an
    # exemption that quietly preserves cheap capacity is an economic change.
    exempt_record = Dict{String,Any}[]
    let ez = Set{Int}(Int.(get(rec["transition"], "exempt_zones", Int[]))),
        zb = Dict{Int,Int}(Int(b["bus_i"]) => Int(b["zone"]) for (_, b) in net["bus"]),
        θ = Float64(rec["transition"]["threshold_usd_per_mwh"]),
        base = Float64(net["baseMVA"])
        for (_, g) in sort!(collect(net["gen"]); by = kv -> Int(kv[2]["index"]))
            Int(get(g, "gen_status", 1)) == 1 || continue
            get(zb, Int(g["gen_bus"]), -1) in ez || continue
            gen_average_variable_cost(g, base) <= θ || continue
            c2, c1, c0 = gen_cost_coefficients(g)
            push!(exempt_record, Dict{String,Any}(
                "index" => Int(g["index"]), "bus" => Int(g["gen_bus"]),
                "zone" => zb[Int(g["gen_bus"])],
                "region" => bus_region[Int(g["gen_bus"])],
                "pmax" => Float64(g["pmax"]), "pmin" => Float64(g["pmin"]),
                "qmax" => Float64(g["qmax"]), "qmin" => Float64(g["qmin"]),
                "avc_usd_per_mwh" => gen_average_variable_cost(g, base),
                "marginal_at_pmin_usd_per_mwh" => gen_marginal_cost(g, Float64(g["pmin"]), base),
                "marginal_at_pmax_usd_per_mwh" => gen_marginal_cost(g, Float64(g["pmax"]), base),
                "constant_cost" => c0,
                "note" => "exempt by zone: untouched, original bounds and full cost polynomial"))
        end
    end

    disposition = String(get(rec["transition"], "retired_disposition", "delete"))
    disposition in ("delete", "synchronous_condenser") ||
        error("unknown retired_disposition $disposition")
    for idx in retired
        haskey(key_of, idx) || error("generator $idx is not in the network")
        key = key_of[idx]
        g = net["gen"][key]
        push!(retired_record, Dict{String,Any}(
            "index" => idx, "bus" => Int(g["gen_bus"]),
            "region" => bus_region[Int(g["gen_bus"])],
            "pmax" => Float64(g["pmax"]), "qmax" => Float64(g["qmax"]),
            "avc_usd_per_mwh" => gen_average_variable_cost(g, Float64(net["baseMVA"])),
            "constant_cost" => gen_cost_coefficients(g)[3],
            "disposition" => disposition))
        if disposition == "delete"
            delete!(net["gen"], key)
        else
            # A synchronous condenser: no active capability at all, reactive
            # limits untouched, and NO cost of any order — including the
            # constant term, which a machine that burns no fuel does not incur
            # and which would otherwise re-enter the objective as a
            # decision-independent charge for a plant that generates nothing.
            g["pmin"] = 0.0
            g["pmax"] = 0.0
            g["pg"] = 0.0
            g["cost"] = [0.0, 0.0, 0.0]
            g["ncost"] = 3
            g["model"] = 2
            g["rs_retired"] = true
        end
    end

    # 2. renewables
    for u in units
        net["gen"][string(u["index"])] = u
    end

    # 3. fleet
    fleet, fleet_report = variant_fleet(base.batteries, bus_region, rec)
    colo_report = Dict{String,Any}("applied" => false)
    if colocate
        fleet, colo_report = colocate_fleet(fleet, units, bus_region, rec)
    end
    fleet, boundary_report = apply_boundary_convention(fleet, rec)

    # 4. demand
    support = get(get(rec, "demand", Dict()), "mode", "") == "deterministic_profile" ?
              deterministic_support(base.demand) : base.demand

    name = "$(base.name)__rs_$(tpl)" * (colocate ? "_colo" : "")
    variant = Dict{String,Any}(
        "schema" => RENEWABLE_VARIANT_SCHEMA,
        "recipe_version" => rec["recipe_version"],
        "base_case" => base.name,
        "template" => tpl,
        "regions" => Dict{String,Any}(r => Dict{String,Any}(
            "zones" => rec["regions"][r]["zones"],
            "role" => rec["regions"][r]["role"],
            "buses" => count(==(r), values(bus_region)),
        ) for r in variant_region_names(rec)),
        "transition" => Dict{String,Any}(
            "measure" => rec["transition"]["measure"],
            "threshold_usd_per_mwh" => rec["transition"]["threshold_usd_per_mwh"],
            "retired" => retired_record,
            "retired_count" => length(retired),
            "retired_pmax_pu" => sum(r["pmax"] for r in retired_record; init = 0.0),
            "retired_qmax_pu" => sum(r["qmax"] for r in retired_record; init = 0.0),
            "retired_constant_cost" => sum(r["constant_cost"] for r in retired_record; init = 0.0),
            "disposition" => disposition,
            "exempt_zones" => sort!(collect(Int.(get(rec["transition"], "exempt_zones", Int[])))),
            "exempt_zone_generators" => exempt_record,
            "exempt_pmax_pu" => sum(r["pmax"] for r in exempt_record; init = 0.0),
            "exempt_pmin_pu" => sum(r["pmin"] for r in exempt_record; init = 0.0),
            "exempt_constant_cost" => sum(r["constant_cost"] for r in exempt_record; init = 0.0),
            "restored_for_service" => restored_record,
            "restored_pmax_pu" => sum(r["pmax"] for r in restored_record; init = 0.0),
            "restored_pmin_pu" => sum(r["pmin"] for r in restored_record; init = 0.0),
            "restored_constant_cost" => sum(r["constant_cost"] for r in restored_record; init = 0.0),
        ),
        "renewables" => Dict{String,Any}(
            "installed_pu" => Dict{String,Any}(rec["renewables"]["installed_pu"]),
            "qmax_fraction" => rec["renewables"]["qmax_fraction"],
            "units" => length(units),
            "reactive_capability" => "qmax = qmax_fraction * pmax, scaled by availability; " *
                                     "a modelling assumption of this recipe, not a property of the resource",
        ),
        "storage" => fleet_report,
        "colocation" => colo_report,
        "boundary" => boundary_report,
    )

    manifest = write_battery_case(out_dir;
        name = name, network = net, batteries = fleet,
        recourse = base.recourse, demand = support,
        source_version = string(get(get(base.manifest, "source", Dict()), "version", "unknown")),
        placement = Dict{String,Any}("inherited_from" => base.name,
                                     "rule" => "base-case buses, resized by region"),
        protocol_stages = Int(base.manifest["protocol"]["num_stages"]),
        protocol_scenarios = Int(base.manifest["protocol"]["num_scenarios"]),
        boundary = get(boundary_report, "terminal_energy_min", nothing) === nothing ?
                   nothing : Dict{String,Any}(
                       "terminal_energy_min" => boundary_report["terminal_energy_min"],
                       "initial_fraction_of_usable" => boundary_report["initial_fraction_of_usable"]),
        variant = variant)
    return manifest
end

if abspath(PROGRAM_FILE) == @__FILE__
    iseven(length(ARGS)) || error("arguments come in --name value pairs")
    args = Dict{String,String}()
    for k in 1:2:length(ARGS)
        startswith(ARGS[k], "--") || error("unexpected argument $(ARGS[k])")
        args[ARGS[k][3:end]] = ARGS[k + 1]
    end
    m = build_renewable_variant(args["base"], args["out"];
                                template = get(args, "template", nothing))
    @printf("wrote %s: %d gens, %d batteries, horizon %d\n",
            m["case"], m["counts"]["gen"], m["counts"]["battery"], m["counts"]["horizon"])
end
