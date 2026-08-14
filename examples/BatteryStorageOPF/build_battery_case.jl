# build_battery_case.jl
#
# The public case-construction API, and the script that builds the frozen cases
# of this study with it.
#
# This is the only file in the project that reaches out to PGLib.jl in order to
# CREATE case data; everything downstream reads the four frozen files through
# `battery_case.jl`. The user-facing workflow is five calls:
#
#   src   = acquire_pglib_case("pglib_opf_case30_ieee")            # A1
#   buses, prec = select_battery_buses(src.network, SampledPlacement(3; seed = 7))
#   fleet, crec = battery_fleet(src.network, buses; power = ..., energy_hours = ...)
#   supp  = freeze_demand_support(sampler, src.network, 24; seed = 11, ...)   # A3
#   build_case(src; dir = ..., batteries = fleet, support = supp, ...)
#
# and `verify(dir)` re-checks everything the manifest claims.
#
# Usage
#   julia --project=. build_battery_case.jl                 # build (and mirror)
#   julia --project=. build_battery_case.jl --verify        # re-verify only
#   DR_BAT_CASE=pglib_opf_case30_ieee julia ... build_battery_case.jl
#
# Environment overrides (all optional; defaults are the frozen correctness case):
#   DR_BAT_CASE            PGLib case name
#   DR_BAT_DIR             output directory (default: case/<case name>)
#   DR_BAT_NUM             number of batteries
#   DR_BAT_BUSES           comma-separated explicit bus override
#   DR_BAT_SEED            placement seed
#   DR_BAT_HORIZON         frozen horizon
#   DR_BAT_MIRROR          path of the DecisionRulesExa.jl battery example to
#                          mirror the case and the shared sources into
#
# Reproducibility note. PowerModels MUTATES the network dictionary it is handed
# (per-unit conversion, status propagation, angle-bound defaults). The exporter
# therefore parses the PGLib case FRESH here and writes those bytes before any
# model is instantiated, so the artifact can never depend on which formulation
# happened to be built first in the same session. Running this script twice
# produces byte-identical files.

using PGLib
using PowerModels
using Pkg
using SHA
using Distributions

@isdefined(BatterySpec) || include(joinpath(@__DIR__, "battery_case.jl"))
@isdefined(DemandSampler) || include(joinpath(@__DIR__, "battery_demand.jl"))

PowerModels.silence()

# ─────────────────────────────────────────────────────────────────────────────
# A1 — PGLib case acquisition
# ─────────────────────────────────────────────────────────────────────────────

"""
    acquire_pglib_case(name; variant=nothing, validate=true) -> NamedTuple

Obtain a canonical PGLib benchmark and check it is usable as a study case.

# Arguments
- `name::AbstractString`: PGLib case name, e.g. `"pglib_opf_case30_ieee"` or
  `"pglib_opf_case30_ieee__api"`.

# Keywords
- `variant`: `nothing` (inferred from the name), `"api"`, `"sad"` or `""` for the
  typical operating condition. PGLib ships each benchmark in three libraries and
  they are DIFFERENT SYSTEMS, not different settings of one: `__api` is the
  congested "active power increase" condition and `__sad` the small-angle
  difference one. Selecting one is a choice of benchmark, not a modification of a
  case, which is why it goes through the acquisition entry point and is recorded
  in the manifest.
- `validate::Bool`: run [`validate_network`](@ref) before returning.

# Returns
A `NamedTuple` with

- `name::String` — the case name as given;
- `network::Dict{String,Any}` — the parsed, per-unit PowerModels network
  dictionary, VERBATIM, with the benchmark's original component identifiers;
- `versions::Dict{String,Any}` — the PGLib, PowerModels and Julia versions the
  bytes came from;
- `report::NamedTuple` — the validation summary (component counts, the load and
  generation totals, the largest connected component).

# Notes
Component identifiers are the benchmark's own and are never renumbered. PGLib
cases contain nonconsecutive identifiers, isolated buses and out-of-service
components; every consumer in this study keys on identifiers rather than on
positions, and this entry point is where that is checked rather than assumed.

Recording the PGLib version matters because a benchmark revised upstream must not
silently become "the same case": the version travels into the manifest and a
rebuild against a different release changes the artifact hashes.
"""
function acquire_pglib_case(name::AbstractString; variant = nothing,
                            validate::Bool = true)
    v = variant === nothing ? _infer_variant(name) : String(variant)
    network = isempty(v) ? pglib(String(name)) : pglib(String(name), v)
    # `PGLib.pglib` WARNS and returns an empty dictionary for a name it cannot
    # find. A study that accepted that would silently build a case out of
    # nothing, so an empty result is an error here.
    isempty(get(network, "bus", Dict())) &&
        error("PGLib returned no network for \"$name\"" *
              (isempty(v) ? "" : " (variant \"$v\")") *
              "; check the name with PGLib.find_pglib_case")
    versions = Dict{String,Any}(
        "PGLib" => string(_package_version("PGLib")),
        "PowerModels" => string(_package_version("PowerModels")),
        "julia" => string(VERSION),
        "variant" => isempty(v) ? "typical" : v,
    )
    report = validate ? validate_network(network) : nothing
    return (name = String(name), variant = v, network = network,
            versions = versions, report = report)
end

"""
    _infer_variant(name) -> String

The PGLib library a case name belongs to, from its `__api` / `__sad` suffix.

# Notes
Inferring rather than requiring the caller to pass it twice removes the one way
this can go silently wrong: asking for `"…__api"` out of the typical-operating-
condition library, getting a name miss, and building the study on whatever the
fuzzy name filter happened to return.
"""
function _infer_variant(name::AbstractString)
    endswith(name, "__api") && return "api"
    endswith(name, "__sad") && return "sad"
    return ""
end

"""
    validate_network(network) -> NamedTuple

Fail closed on the network properties this study depends on, and summarize what
was found.

# Checks
1. the mandatory component tables exist and are non-empty;
2. every component identifier equals its own `"index"` field, so the dictionary
   key and the identifier can never disagree;
3. every in-service generator, branch and load refers to an existing bus;
4. every in-service bus is reachable from the reference bus over in-service
   branches — a second electrical island has its own reference angle and its own
   power balance, and a study that quietly spans two of them is not the study it
   says it is;
5. at least one reference bus and at least one positively-priced generator exist;
6. total nominal active load is positive.

# Returns
Counts, totals, and the number of buses in the reference bus's island.

# Notes
Out-of-service components are reported but not rejected: PGLib ships them and
PowerModels handles them. What is rejected is an in-service component the model
cannot place.
"""
function validate_network(network::AbstractDict)
    for tbl in ("bus", "gen", "branch", "load")
        haskey(network, tbl) || error("network has no \"$tbl\" table")
        isempty(network[tbl]) && error("network table \"$tbl\" is empty")
    end
    for tbl in ("bus", "gen", "branch", "load", "shunt")
        haskey(network, tbl) || continue
        for (key, comp) in network[tbl]
            parse(Int, string(key)) == Int(comp["index"]) ||
                error("$tbl entry \"$key\" carries index $(comp["index"])")
        end
    end

    bus_ids = Set(Int(b["index"]) for (_, b) in network["bus"])
    active_bus = Set(Int(b["index"]) for (_, b) in network["bus"]
                     if Int(get(b, "bus_type", 1)) != 4)
    for (_, g) in network["gen"]
        Int(get(g, "gen_status", 1)) == 0 && continue
        Int(g["gen_bus"]) in bus_ids || error("generator $(g["index"]) sits at unknown bus $(g["gen_bus"])")
    end
    for (_, l) in network["load"]
        Int(get(l, "status", 1)) == 0 && continue
        Int(l["load_bus"]) in bus_ids || error("load $(l["index"]) sits at unknown bus $(l["load_bus"])")
    end
    adjacency = Dict{Int,Vector{Int}}(i => Int[] for i in bus_ids)
    n_branch_off = 0
    for (_, br) in network["branch"]
        if Int(get(br, "br_status", 1)) == 0
            n_branch_off += 1
            continue
        end
        f, t = Int(br["f_bus"]), Int(br["t_bus"])
        (f in bus_ids && t in bus_ids) ||
            error("branch $(br["index"]) connects unknown buses $f-$t")
        push!(adjacency[f], t)
        push!(adjacency[t], f)
    end

    refs = [Int(b["index"]) for (_, b) in network["bus"] if Int(get(b, "bus_type", 1)) == 3]
    isempty(refs) && error("network has no reference (slack) bus")
    # Breadth-first search from the first reference bus over in-service branches.
    seen = Set{Int}([first(sort!(refs))])
    queue = collect(seen)
    while !isempty(queue)
        i = pop!(queue)
        for j in adjacency[i]
            if !(j in seen)
                push!(seen, j)
                push!(queue, j)
            end
        end
    end
    stranded = setdiff(active_bus, seen)
    isempty(stranded) ||
        error("in-service buses $(sort!(collect(stranded))) are not connected to the reference bus")

    priced = 0
    scheduled = 0
    for (_, g) in network["gen"]
        # An availability schedule is checked on EVERY generator, in service or
        # not: a malformed schedule on a unit that is switched on later is still
        # a malformed case, and this is the last point at which the case exists
        # as data rather than as a solved stage.
        if haskey(g, STAGE_AVAILABILITY_KEY)
            sched = g[STAGE_AVAILABILITY_KEY]
            (sched isa AbstractVector && !isempty(sched)) ||
                error("generator $(g["index"]): \"$STAGE_AVAILABILITY_KEY\" must be a non-empty vector of multipliers")
            all(x -> isfinite(Float64(x)) && Float64(x) >= 0, sched) ||
                error("generator $(g["index"]): \"$STAGE_AVAILABILITY_KEY\" must hold finite nonnegative multipliers, got $sched")
            scheduled += 1
        end
        Int(get(g, "gen_status", 1)) == 0 && continue
        c = Float64.(g["cost"])
        any(!=(0), c) && (priced += 1)
    end
    priced > 0 || error("network has no positively-priced generator")

    total_pd = sum(Float64(l["pd"]) for (_, l) in network["load"]
                   if Int(get(l, "status", 1)) != 0; init = 0.0)
    total_pd > 0 || error("network has no positive nominal active load")
    total_pmax = sum(Float64(g["pmax"]) for (_, g) in network["gen"]
                     if Int(get(g, "gen_status", 1)) != 0; init = 0.0)

    return (bus = length(network["bus"]), gen = length(network["gen"]),
            branch = length(network["branch"]), load = length(network["load"]),
            shunt = length(get(network, "shunt", Dict())),
            branches_out_of_service = n_branch_off,
            buses_out_of_service = length(bus_ids) - length(active_bus),
            connected_component = length(seen),
            total_load_pu = total_pd, total_pmax_pu = total_pmax,
            priced_generators = priced,
            scheduled_generators = scheduled,
            headroom = total_pmax / total_pd)
end

"Version string of an installed package, by name."
function _package_version(name::AbstractString)
    for (_, dep) in Pkg.dependencies()
        dep.name == name && return something(dep.version, "unknown")
    end
    return "unknown"
end

# ─────────────────────────────────────────────────────────────────────────────
# Recourse prices
# ─────────────────────────────────────────────────────────────────────────────

"""
    recourse_prices(network; factor=50) -> RecourseCosts

Derive the two-sided recourse prices from the host case.

# Arguments
- `network::AbstractDict`: parsed PowerModels network (per-unit).

# Keywords
- `factor::Real`: multiple of the largest generator marginal cost.

# Returns
- [`RecourseCosts`](@ref) with `deficit == surplus`.

# Notes
The price is `factor` times the largest generator MARGINAL cost evaluated at that
generator's `pmax`, rounded up to an integer. Deriving it from the case rather
than hard-coding a number is what lets the same API move to another PGLib
benchmark without a hidden re-tuning, and the factor puts recourse far outside
any economic trade-off against dispatch: a policy that uses it is rejected, not
priced.

**Why the factor is LARGE, measured.** It is tempting to shrink this price on
conditioning grounds: it is the biggest coefficient in the objective by orders of
magnitude, and on `pglib_opf_case300_ieee` a factor of 50 gives 584,698 per pu
against a worst measured NODAL price of 4,612. A failing SOC-WR subproblem duly
produced a `DUAL_INFEASIBLE` certificate whose "unbounded ray" had recourse
components of order `1e-7` — numerical noise that, multiplied by 584,698, moves
the objective by about 2. That reads as a scaling problem, and it is a trap.

A controlled measurement on `pglib_opf_case118_ieee` says the opposite. Holding
everything else fixed and running ten SDDP iterations:

| conic tolerance | factor | result |
|---|---|---|
| `1e-8` | 50 | 10 iterations, bound 2,152,035 |
| `1e-6` | 50 | 10 iterations, bound 2,149,769 |
| `1e-8` | **5** | **fails at node 19** |

The factor, not the tolerance, decides whether SDDP runs, and it is the LOW price
that fails. The reason is admissibility, not scaling: as the price falls the
solution starts to USE recourse, and the base ACP sweep of the research case
shows exactly that — worst recourse 0.00 at factors 50 and 20, then 2.3e-3,
4.5e-3 and 7.5e-3 at 5, 2 and 1. Once recourse is a cheap generator rather than a
last resort, the stage problem's dual structure changes and the conic solves
degrade.

So the price must beat every real alternative by a wide margin, and the apparent
tension with conditioning is resolved in favour of the large price.

Deficit and surplus are priced EQUALLY. An asymmetric price would make the
recourse a one-sided economic signal and could bias which side of a reachable
interval a target is pushed towards; the recourse must be neutral, because its
purpose is feasibility, not incentive.
"""
function recourse_prices(network::AbstractDict; factor::Real = 50)
    marginal = 0.0
    for (_, gen) in network["gen"]
        Int(get(gen, "gen_status", 1)) == 0 && continue
        cost = Float64.(gen["cost"])
        ncost = Int(gen["ncost"])
        pmax = Float64(gen["pmax"])
        # PowerModels stores the polynomial highest-order-first. The marginal
        # cost of a quadratic model at pmax is 2 c2 pmax + c1; a linear model's
        # is just c1; a constant model's is 0.
        m = if ncost >= 3
            2 * cost[end - 2] * pmax + cost[end - 1]
        elseif ncost == 2
            cost[end - 1]
        else
            0.0
        end
        marginal = max(marginal, m)
    end
    marginal > 0 || error("recourse_prices: case has no positively-priced generator")
    price = ceil(factor * marginal)
    return RecourseCosts(price, price)
end

# ─────────────────────────────────────────────────────────────────────────────
# The general case constructor
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_case(source; dir, batteries, support, placement=Dict(), recourse=nothing,
               protocol_stages, protocol_scenarios, mirror=nothing) -> BatteryCase

Write the four frozen artifacts of a case and read them back through the
verifier.

# Arguments
- `source`: the named tuple returned by [`acquire_pglib_case`](@ref).

# Keywords
- `dir::AbstractString`: output directory.
- `batteries::Vector{BatterySpec}`: the fleet, from [`battery_fleet`](@ref).
- `support::DemandSupport`: the FROZEN finite support, from
  [`freeze_demand_support`](@ref).
- `placement::AbstractDict`: the placement and capacity records, merged into the
  battery artifact.
- `recourse`: `nothing` to derive prices from the case, or explicit
  [`RecourseCosts`](@ref).
- `protocol_stages`, `protocol_scenarios`: the FINAL paired protocol the manifest
  freezes a digest for.
- `screening_seed`, `screening_scenarios`: the INDEPENDENT screening protocol.
  Leaving the seed `nothing` records no screening protocol, which is right for a
  correctness case that has no selection decision to make.
- `mirror`: an Exa example directory to mirror the case and shared sources into.

# Notes
Everything the two engines must agree on is decided HERE, once, and hashed. The
function returns the case as READ BACK from disk, not the objects it was handed,
so a case that cannot be re-read is a build failure rather than a later mystery.
"""
function build_case(source;
                    dir::AbstractString,
                    batteries::AbstractVector{BatterySpec},
                    support::DemandSupport,
                    placement::AbstractDict = Dict{String,Any}(),
                    recourse::Union{Nothing,RecourseCosts} = nothing,
                    protocol_stages::Integer,
                    protocol_scenarios::Integer,
                    screening_seed::Union{Nothing,Integer} = nothing,
                    screening_scenarios::Integer = 0,
                    mirror::Union{Nothing,AbstractString} = nothing,
                    quiet::Bool = false)
    prices = recourse === nothing ? recourse_prices(source.network) : recourse
    record = Dict{String,Any}(placement)
    record["versions"] = source.versions

    manifest = write_battery_case(dir;
                                  name = source.name,
                                  network = source.network,
                                  batteries = batteries,
                                  recourse = prices,
                                  demand = support,
                                  source_version = string(source.versions["PGLib"]),
                                  placement = record,
                                  protocol_stages = protocol_stages,
                                  protocol_scenarios = protocol_scenarios,
                                  screening_seed = screening_seed,
                                  screening_scenarios = screening_scenarios)

    built = read_battery_case(dir)             # read back through the verifier
    if !quiet
        print(describe(built))
        println("manifest artifacts:")
        for (k, v) in sort!(collect(manifest["artifacts"]); by = first)
            println("  ", rpad(k, 18), v)
        end
        println("  ", rpad("support", 18), manifest["support"]["sha256"])
        println("  ", rpad("protocol", 18), manifest["protocol"]["sha256"])
        manifest["screening"] === nothing ||
            println("  ", rpad("screening", 18), manifest["screening"]["sha256"])
    end
    mirror === nothing || mirror_to_exa(dir, mirror; quiet = quiet)
    return built
end

"""
    mirror_to_exa(case_dir, exa_example_dir; quiet=false)

Copy the frozen case and the byte-shared source files into the ExaModels
example, then assert byte identity of everything copied.

# Notes
The two engines are independent packages: neither may `include` a file from the
other. They are nonetheless required to agree exactly on the case contract and
on the solution schema, so those files are shipped as COPIES whose identity is
asserted here. The assertion is the point — a copy that has drifted is precisely
the failure this guards against.
"""
function mirror_to_exa(case_dir::AbstractString, exa_example_dir::AbstractString;
                       quiet::Bool = false)
    isdir(exa_example_dir) || error("mirror target $exa_example_dir does not exist")
    target_case = joinpath(exa_example_dir, "case", basename(case_dir))
    mkpath(target_case)
    for f in ("network.json", "batteries.json", "demand.json", "case_manifest.json")
        cp(joinpath(case_dir, f), joinpath(target_case, f); force = true)
        sha256_file(joinpath(case_dir, f)) == sha256_file(joinpath(target_case, f)) ||
            error("mirrored case file $f is not byte-identical")
    end
    for f in ("battery_case.jl", "battery_solution_schema.jl")
        src = joinpath(@__DIR__, f)
        dst = joinpath(exa_example_dir, f)
        cp(src, dst; force = true)
        sha256_file(src) == sha256_file(dst) ||
            error("mirrored source file $f is not byte-identical")
    end
    quiet || println("mirrored case + shared sources into ", exa_example_dir)
    return nothing
end

"""
    verify(dir) -> BatteryCase

Re-verify a frozen case in place: hashes, schemas, the stage-duration contract,
the battery bound consistency checks, the support digest and the regenerated
protocol digest.
"""
function verify(dir::AbstractString = get(ENV, "DR_BAT_DIR",
                                          joinpath(@__DIR__, "case", DEFAULT_CASE)))
    case = read_battery_case(dir)
    print(describe(case))
    println("case verified: ", dir)
    return case
end

# ─────────────────────────────────────────────────────────────────────────────
# The correctness case
#
# These are the constants of the Phase-1 correctness case. They deliberately
# describe a SMALL, comfortably feasible system: the case exists to establish
# that the two engines agree and that the strict formulation is well posed, not
# to expose a scientific mechanism. The research case is constructed from the
# same API by `case_design.jl`, and is not this section's business.
# ─────────────────────────────────────────────────────────────────────────────

"PGLib benchmark used by the correctness case."
const DEFAULT_CASE = get(ENV, "DR_BAT_CASE", "pglib_opf_case14_ieee")

"Number of batteries placed when no explicit bus list is given."
const DEFAULT_NUM_BATTERIES = parse(Int, get(ENV, "DR_BAT_NUM", "3"))

"Seed of the deterministic placement draw."
const DEFAULT_PLACEMENT_SEED = parse(Int, get(ENV, "DR_BAT_SEED", "20260804"))

"Seed of the paired evaluation protocol."
const DEFAULT_PROTOCOL_SEED = parse(Int, get(ENV, "DR_BAT_PROTOCOL_SEED", "20260804"))

"Frozen horizon of the correctness case."
const DEFAULT_HORIZON = parse(Int, get(ENV, "DR_BAT_HORIZON", "48"))

"Shape and scenario dimensions the manifest freezes a protocol digest for."
const PROTOCOL_STAGES = parse(Int, get(ENV, "DR_BAT_PROTOCOL_STAGES", "48"))
const PROTOCOL_SCENARIOS = parse(Int, get(ENV, "DR_BAT_PROTOCOL_SCENARIOS", "200"))

"""
Fraction of stored energy a battery retains across one one-hour stage.

A physical 0.1 %/h standing loss. It is deliberately NOT 1: with ``\\alpha = 1``
the multiplier transform that turns stage duals into the actor signal reduces to
a plain difference and a sign or scaling error in the ``\\alpha`` term would go
unnoticed by every test.
"""
const SELF_DISCHARGE = 0.999

"""
Degradation price charged on battery throughput, per pu·h of
``\\Delta t (p^{ch} + p^{dis})``.

Small against the case's marginal generation cost — about 0.2 % of it — so it
does not distort dispatch, but STRICTLY positive, which is what makes
simultaneous charging and discharging strictly suboptimal in the continuous
relaxation instead of merely unattractive.
"""
const THROUGHPUT_COST = 5.0

"""
    build(; case, dir, num_batteries, buses, seed, horizon, mirror) -> BatteryCase

Build the correctness case: a system-wide three-atom demand support on a mild
diurnal profile, with batteries drawn from the load buses.

# Notes
The demand sampler is written out in full rather than hidden behind a constant,
because this function is also the smallest complete EXAMPLE of the public
workflow: acquire, place, rate, freeze, build.
"""
function build(; case::AbstractString = DEFAULT_CASE,
                 dir::AbstractString = get(ENV, "DR_BAT_DIR", joinpath(@__DIR__, "case", DEFAULT_CASE)),
                 num_batteries::Integer = DEFAULT_NUM_BATTERIES,
                 buses = _env_buses(),
                 seed::Integer = DEFAULT_PLACEMENT_SEED,
                 horizon::Integer = DEFAULT_HORIZON,
                 mirror::Union{Nothing,AbstractString} = get(ENV, "DR_BAT_MIRROR", nothing),
                 quiet::Bool = false)

    source = acquire_pglib_case(case)

    # ── batteries ────────────────────────────────────────────────────────────
    strategy = buses === nothing ? SampledPlacement(num_batteries; seed = seed) :
                                   ExplicitPlacement(buses)
    chosen, placement_record = select_battery_buses(source.network, strategy)
    total_load = sum(values(nominal_load_at_bus(source.network)))
    fleet, capacity_record = battery_fleet(source.network, chosen;
                                           power = 0.10 * total_load,
                                           energy_hours = 2.0,
                                           self_discharge = SELF_DISCHARGE,
                                           throughput_cost = THROUGHPUT_COST,
                                           initial_fraction = 0.5)

    # ── demand ───────────────────────────────────────────────────────────────
    # Three symmetric system-wide atoms with equal probability. Symmetry means
    # the process has mean 1, so the deterministic shape alone describes expected
    # demand and the uncertainty is a pure spread around it. Three atoms is small
    # enough that an SDDP backward pass enumerates it exactly and large enough
    # that a policy must hedge rather than track a single forecast.
    sampler = SystemMultiplier(DiscreteNonParametric([0.95, 1.0, 1.05],
                                                     [1 / 3, 1 / 3, 1 / 3]))
    support = freeze_demand_support(sampler, source.network, horizon;
                                    seed = DEFAULT_PROTOCOL_SEED,
                                    method = :exact,
                                    profile = diurnal_profile(horizon),
                                    protocol_seed = DEFAULT_PROTOCOL_SEED,
                                    stage_hours = 1.0)

    return build_case(source;
                      dir = dir,
                      batteries = fleet,
                      support = support,
                      placement = Dict{String,Any}("buses" => placement_record,
                                                   "capacity" => capacity_record,
                                                   "capacity_rule" =>
                                                       "power = 10% of nominal system load; energy = 2 h"),
                      protocol_stages = PROTOCOL_STAGES,
                      protocol_scenarios = PROTOCOL_SCENARIOS,
                      mirror = mirror,
                      quiet = quiet)
end

"""
    ensure_case(dir; kwargs...) -> BatteryCase

Read the frozen case at `dir`, BUILDING it first if it is not there.

# Notes
Cases are constructed, never committed: the artifacts are a pure function of this
file's builder and its recorded seeds. Anything that needs a case — a test, a
diagnostic, an engine — calls this rather than assuming someone checked one in.
"""
function ensure_case(dir::AbstractString = joinpath(@__DIR__, "case", DEFAULT_CASE);
                     quiet::Bool = true, kwargs...)
    isfile(joinpath(dir, "case_manifest.json")) && return read_battery_case(dir)
    return build(; dir = dir, quiet = quiet, kwargs...)
end

# ─────────────────────────────────────────────────────────────────────────────
# The research case
#
# Constructed from the same public API as the correctness case, with three
# differences, each of which was MEASURED rather than chosen:
#
#   • the benchmark is a congested PGLib operating condition, selected because
#     its SOC-WR relaxation misprices stored energy at a decision level;
#   • the deterministic profile has a quiet window where charging is cheap and a
#     stressed window where stored energy is valuable;
#   • the uncertainty is a small JOINT REGIONAL support that stresses the
#     constrained pocket and the rest of the system independently, so WHERE
#     energy is stored matters and not only HOW MUCH.
#
# Every constant below is part of the frozen benchmark and travels into the
# manifest. The evidence that selected them is in the Phase-2 record.
# ─────────────────────────────────────────────────────────────────────────────

"""
    research_profile(horizon; low, high, peak_hour, period) -> Vector{Float64}

The deterministic hourly profile of the research case.

# Notes
A raised cosine on `[low, high]` rather than a mean-1 profile: the LEVEL matters
here, because the case is built on a congested operating condition where the
network's ability to deliver is the binding physics. The quiet hours sit low
enough that charging is cheap and uncongested; the peak sits at the highest level
the base ACP problem still serves with NO recourse under every atom, which is the
admissibility precondition of the whole strict-target construction.
"""
function research_profile(horizon::Integer; low::Real = 0.78, high::Real = 1.00,
                          peak_hour::Integer = 19, period::Integer = 24)
    mid = (low + high) / 2
    amp = (high - low) / 2
    return [mid + amp * cos(2π * (t - peak_hour) / period) for t in 1:horizon]
end

"""
    regional_demand_sampler(network; pocket_buses, quiet_stages, horizon,
                            pocket_atoms, rest_atoms) -> DemandSampler

The research case's authoring sampler: deterministic quiet hours, and a joint
two-region support in the stressed hours.

# Arguments
- `pocket_buses`: the buses of the constrained region, measured from the nodal
  prices and the binding limits of the base dispatch.

# Keywords
- `quiet_stages`: a predicate `t -> Bool` marking the stages with no uncertainty.
- `pocket_atoms`, `rest_atoms`: the finite supports of the two regional
  multipliers.

# Notes
Two regions, two atoms each, is the smallest construction that makes the
LOCATION of stored energy a hedging decision rather than a bookkeeping detail:
the pocket and the rest move independently, so a policy that has put its energy
in the wrong place cannot move it in time. A system-wide sampler with the same
marginal spread would produce a purely temporal problem, and every battery in it
would be interchangeable.

The quiet stages carry a DEGENERATE support (one atom, probability 1). That is
deliberate and is not the same as having no stages there: the quiet hours are
where the policy charges, and the value of doing so is exactly what the stressed
hours reveal.
"""
function regional_demand_sampler(network::AbstractDict;
                                 pocket_buses,
                                 quiet_stages,
                                 horizon::Integer,
                                 pocket_atoms = ([0.96, 1.04], [0.5, 0.5]),
                                 rest_atoms = ([0.99, 1.01], [0.5, 0.5]))
    meta = demand_meta(network)
    pocket = Set(Int.(pocket_buses))
    pocket_loads = [meta.load_ids[j] for j in 1:meta.num_loads if meta.load_bus[j] in pocket]
    rest_loads = [meta.load_ids[j] for j in 1:meta.num_loads if !(meta.load_bus[j] in pocket)]
    isempty(pocket_loads) && error("regional_demand_sampler: the pocket contains no load")
    isempty(rest_loads) && error("regional_demand_sampler: every load is in the pocket")

    stressed = GroupMultiplier([pocket_loads, rest_loads],
                               [DiscreteNonParametric(pocket_atoms...),
                                DiscreteNonParametric(rest_atoms...)])
    quiet = DeterministicMultiplier(1.0)
    return StageMultiplier([quiet_stages(t) ? quiet : stressed for t in 1:horizon])
end

"""
    nearest_bus_regions(network, centers) -> Vector{Vector{Int}}

Partition every load bus among `centers` by electrical hop distance: each load
bus joins the center it is closest to over the branch graph.

# Notes
Automatic, and it applies to any PGLib case and any set of centers. Passing the
battery buses as the centers gives each battery its OWN region, which is the
precondition for the demand process to say anything locational: if two batteries
sit in one region they see the same signal and one of them is redundant.

Ties go to the lower-numbered center, so the partition is a deterministic
function of the network and the centers. Buses unreachable from every center
(an islanded component) join the first center; a PGLib case is connected, so
this is a guard rather than a case that arises.

The manual path is to skip this and hand region load lists straight to
[`JointRegionMultiplier`](@ref) — which is what to do when a measured region
(nodal prices, binding limits) is wanted instead of a distance partition.
"""
function nearest_bus_regions(network::AbstractDict, centers)
    ctr = Int.(collect(centers))
    isempty(ctr) && throw(ArgumentError("nearest_bus_regions: no centers given"))

    adj = Dict{Int,Vector{Int}}()
    for (_, br) in network["branch"]
        get(br, "br_status", 1) == 0 && continue
        f, t = Int(br["f_bus"]), Int(br["t_bus"])
        push!(get!(adj, f, Int[]), t)
        push!(get!(adj, t, Int[]), f)
    end

    # One multi-source BFS rather than one BFS per center: the frontier carries
    # its owner, so every bus is settled at its true nearest center in one sweep.
    owner = Dict{Int,Int}()
    for (i, c) in enumerate(ctr)
        haskey(owner, c) || (owner[c] = i)
    end
    frontier = copy(ctr)
    while !isempty(frontier)
        nxt = Int[]
        for b in frontier, nb in get(adj, b, Int[])
            haskey(owner, nb) && continue
            owner[nb] = owner[b]
            push!(nxt, nb)
        end
        frontier = nxt
    end

    regions = [Int[] for _ in ctr]
    for b in sort!(collect(keys(nominal_load_at_bus(network))))
        push!(regions[get(owner, b, 1)], b)
    end
    return regions
end

"""
    rotating_regime_sampler(network; centers, horizon, low, high, period,
                            peak_hour, stress_min, stress_max, joint_share,
                            spread, min_probability, groups, modes,
                            mode_probabilities) -> DemandSampler

A demand process whose regional means, tail and inter-region CORRELATION all
change with the stage.

# Keywords
- `centers`: the region centers, normally the battery buses. Ignored when
  `groups` is given.
- `groups`: explicit region bus lists — the manual path past the automatic
  [`nearest_bus_regions`](@ref) partition.
- `low`, `high`: the slack and stressed regional multipliers.
- `period`, `peak_hour`: the daily cycle the regime rotates on.
- `stress_min`, `stress_max`: total probability that SOME region is stressed, at
  the trough and at the peak of the cycle.
- `joint_share`: at the cycle peak, the share of that stress probability going
  to the all-regions-stressed mode. Below the peak it is scaled down, so heavy
  hours are also the CORRELATED hours.
- `spread`: `0` makes every region equally likely to be the stressed one at
  every stage; `1` makes the stressed region rotate sharply with the cycle.
- `min_probability`: modes below this at a stage are dropped and the rest
  renormalized.
- `modes`, `mode_probabilities`: fully manual overrides — an explicit
  `(num_regions, K)` mode matrix and a `t -> probabilities` callable.

# Notes
The modes are the joint outcomes worth naming: everything slack, exactly one
region stressed (one mode per region), and everything stressed. So `R` regions
cost `R + 2` atoms rather than the `2^R` of independent regions.

What makes the process hard to operate is that the three things a policy would
want to know move independently:

- the MEAN moves, because the probability of any stress at all follows the daily
  cycle between `stress_min` and `stress_max`;
- the TAIL moves, because the all-regions-stressed mode is scaled by the cycle
  on top of that, so the heavy outcome is concentrated in a few hours;
- the CORRELATION moves, and it changes SIGN. When mass sits on the single-region
  modes the regions are negatively correlated — one is stressed exactly when the
  others are slack, and energy stored in the right place is worth much more than
  the same energy stored elsewhere. When mass sits on the calm and all-stressed
  modes they are positively correlated, and location buys nothing.

A policy therefore cannot reduce the problem to one storage schedule copied
across sites, nor to a per-site schedule computed independently: the right
charge in region `r` depends on the phase of the cycle and on what the other
regions are expected to do. That is the multi-dimensional structure the study
needs, and it is also precisely what a perfect-foresight solution exploits —
knowing WHICH region gets stressed, not merely how much total demand arrives.

`spread` is the knob that sets how much of that is locational: at `spread=0` the
identity of the stressed region is pure coin-flip noise with no time structure,
and the process degenerates to a temporal one.
"""
function rotating_regime_sampler(network::AbstractDict;
                                 centers = Int[],
                                 horizon::Integer,
                                 low::Real = 0.85,
                                 high::Real = 1.15,
                                 period::Integer = 24,
                                 peak_hour::Integer = 19,
                                 stress_min::Real = 0.10,
                                 stress_max::Real = 0.90,
                                 joint_share::Real = 0.35,
                                 spread::Real = 0.85,
                                 min_probability::Real = 1e-3,
                                 groups = nothing,
                                 modes = nothing,
                                 mode_probabilities = nothing)
    region_buses = groups === nothing ? nearest_bus_regions(network, centers) :
                   [sort!(collect(Int.(g))) for g in groups]
    R = length(region_buses)
    R >= 2 || throw(ArgumentError("rotating_regime_sampler needs at least 2 regions"))
    any(isempty, region_buses) &&
        throw(ArgumentError("rotating_regime_sampler: a region contains no load bus"))

    # Columns: calm, then one per region, then all-stressed.
    M = modes === nothing ?
        hcat(fill(Float64(low), R),
             [[r == c ? Float64(high) : Float64(low) for r in 1:R] for c in 1:R]...,
             fill(Float64(high), R)) :
        (modes isa AbstractMatrix ? Float64.(Matrix(modes)) :
         reduce(hcat, [Float64.(collect(m)) for m in modes]))
    K = size(M, 2)

    "Mode probabilities at stage `t`: the cycle sets how much stress and how correlated."
    function default_probs(t)
        # Cycle intensity in [0, 1], peaking at `peak_hour`.
        s = (1 + cos(2π * (t - peak_hour) / period)) / 2
        total = stress_min + (stress_max - stress_min) * s
        joint = total * joint_share * s          # tail concentrates at the peak
        local_total = total - joint
        # Which region is the stressed one rotates through the cycle: region r's
        # turn is offset by r/R of a period.
        w = [(1 - spread) + spread *
             max(0.0, cos(2π * (t - peak_hour) / period - 2π * (r - 1) / R))
             for r in 1:R]
        sw = sum(w)
        sw > 0 || (w = fill(1.0, R); sw = R)
        return vcat(1 - total, local_total .* (w ./ sw), joint)
    end

    probs_at = mode_probabilities === nothing ? default_probs : mode_probabilities

    stages = Vector{DemandSampler}(undef, horizon)
    for t in 1:horizon
        p = Float64.(collect(probs_at(t)))
        length(p) == K || error("mode probabilities at stage $t have length $(length(p)), expected $K")
        # Strictly positive as well as above the threshold: at the trough of the
        # cycle the all-stressed mode has probability exactly zero, and a
        # zero-probability atom is not a mode of the law, it is an absent one.
        keep = [k for k in 1:K if p[k] >= min_probability && p[k] > 0]
        isempty(keep) && (keep = [argmax(p)])
        q = p[keep] ./ sum(p[keep])
        # A single surviving mode stays a JointRegionMultiplier carrying one
        # atom: it keys by BUS, where DeterministicMultiplier's dict form keys
        # by LOAD, and mixing the two keyings is a silent way to get it wrong.
        stages[t] = JointRegionMultiplier(region_buses, M[:, keep], q; by = :bus)
    end
    return StageMultiplier(stages)
end

"""
    build_research_case(; case, dir, pocket_buses, battery_buses, horizon,
                          report_stages, power_fraction, energy_hours,
                          profile_low, profile_high, pocket_atoms, rest_atoms,
                          protocol_scenarios, mirror) -> BatteryCase

Build a multiperiod research case from the public API.

# Keywords
- `case::AbstractString`: PGLib benchmark, including its `__api` / `__sad` suffix.
- `pocket_buses`: the constrained region, from the base-dispatch measurement.
- `battery_buses`: where the batteries go, from the storage-value measurement.
- `horizon::Integer`: stages FROZEN, i.e. reported window plus look-ahead tail.
- `report_stages::Integer`: the reported window, recorded in the manifest.
- `power_fraction::Real`: each battery's power rating as a fraction of nominal
  active load — of the WHOLE SYSTEM under `power_basis = :system`, or of the
  battery's OWN region under `:region`; `energy_hours` its duration.
- `power_basis::Symbol`: `:system` or `:region`. Use `:region` on a large network
  with concentrated regions, where a system-wide fraction can make one battery
  large enough to dominate its own neighbourhood.
- `profile_low`, `profile_high`: the deterministic profile's band.
- `pocket_atoms`, `rest_atoms`: `(values, probabilities)` of the two regional
  multipliers in the stressed hours. `:pocket` regime only.
- `demand_regime`: `:pocket` for the two-region product process, or `:rotating`
  for the time-varying joint process of [`rotating_regime_sampler`](@ref), whose
  regions carry independent means, tails and a correlation that changes sign
  through the cycle.
- `region_centers`: the `:rotating` regions' centers; defaults to the battery
  buses, which is what gives each battery a region of its own.
- `region_groups`: explicit region bus lists — the manual path past the
  automatic partition.
- `regime`: a NamedTuple of [`rotating_regime_sampler`](@ref) keywords
  (`low`, `high`, `stress_min`, `stress_max`, `joint_share`, `spread`, …).

# Notes
Everything the benchmark IS is an argument here, and every argument lands in the
manifest, so the frozen case can be rebuilt from the recorded values alone.

**Horizon and tail.** The reported window is a whole number of daily cycles, so a
storage decision taken in it is completed inside it. The tail exists for one
reason: with a finite horizon and no terminal value, the optimal thing to do in
the last stages is to empty every battery, and if the reported window ended at
the horizon that dump would be a large part of the reported cost. Reporting a
prefix and letting the tail absorb the terminal effect is what keeps the
comparison about the policy rather than about the boundary condition.

**Quiet hours.** Stages in the first half of each daily cycle carry a degenerate
one-atom support. That is where charging happens, and it is deterministic on
purpose: the study is about valuing STORED energy under uncertainty about when it
will be needed, not about a policy's ability to forecast the hour it charges in.
"""
function build_research_case(; case::AbstractString,
                               dir::AbstractString,
                               pocket_buses,
                               battery_buses,
                               horizon::Integer = 36,
                               report_stages::Integer = 24,
                               power_fraction::Real = 0.04,
                               energy_hours::Real = 2.0,
                               profile_low::Real = 0.78,
                               profile_high::Real = 1.00,
                               pocket_atoms = ([0.96, 1.04], [0.5, 0.5]),
                               rest_atoms = ([0.99, 1.01], [0.5, 0.5]),
                               pocket_fraction::Real = 0.0,
                               demand_regime::Symbol = :pocket,
                               power_basis::Symbol = :system,
                               region_centers = Int[],
                               region_groups = nothing,
                               regime = NamedTuple(),
                               self_discharge::Real = SELF_DISCHARGE,
                               throughput_cost::Real = THROUGHPUT_COST,
                               initial_fraction::Real = 0.5,
                               reserve_fraction::Real = 0.0,
                               protocol_seed::Integer = DEFAULT_PROTOCOL_SEED,
                               protocol_scenarios::Integer = 500,
                               screening_seed::Integer = DEFAULT_PROTOCOL_SEED + 1,
                               screening_scenarios::Integer = 64,
                               recourse_factor::Real = 50,
                               quiet_stages = t -> mod1(t, 24) <= 12,
                               mirror::Union{Nothing,AbstractString} = nothing,
                               quiet::Bool = false)
    source = acquire_pglib_case(case)

    # A pocket given as a FRACTION selects that share of the load buses, largest
    # demand first. It is the automatic path; an explicit `pocket_buses` list is
    # the manual one, for when a measured region is wanted instead.
    pocket = pocket_fraction > 0 ?
             (l = nominal_load_at_bus(source.network);
              r = sort!(collect(keys(l)); by = b -> -max(0.0, l[b]));
              r[1:max(1, round(Int, pocket_fraction * length(r)))]) : pocket_buses

    buses, placement_record = select_battery_buses(source.network,
                                                   ExplicitPlacement(battery_buses))
    load_at = nominal_load_at_bus(source.network)
    total_load = sum(max(0.0, v) for v in values(load_at))

    # `:system` rates every battery at a fraction of TOTAL system load; `:region`
    # rates it at a fraction of its OWN region's load.
    #
    # `:system` is a trap on a large network with concentrated regions. On
    # case300 a 0.04 system fraction is 9.54 pu, and the region around bus 138
    # holds 24 pu of load — so charging that battery is a 40 % local demand
    # spike, and there are states where it is the only thing between the region
    # and a shortfall. The dual of stored energy is then the RECOURSE price
    # rather than an energy price, and SDDP builds cuts whose coefficients span
    # ten orders of magnitude, which is what makes later subproblems report false
    # infeasibility. Sizing against the battery's own region keeps the machine
    # proportionate to the load it serves at every site.
    power_rule = if power_basis === :region
        regions = region_groups === nothing ?
                  nearest_bus_regions(source.network,
                                      isempty(region_centers) ? buses : region_centers) :
                  [sort!(collect(Int.(g))) for g in region_groups]
        region_load = Dict{Int,Float64}()
        for (i, r) in enumerate(regions)
            l = sum(max(0.0, get(load_at, b, 0.0)) for b in r)
            for b in r
                region_load[b] = l
            end
            i <= length(buses) || continue
        end
        bus -> power_fraction * get(region_load, bus, total_load)
    elseif power_basis === :system
        power_fraction * total_load
    else
        throw(ArgumentError("power_basis must be :system or :region, got :$power_basis"))
    end

    fleet, capacity_record = battery_fleet(source.network, buses;
                                           power = power_rule,
                                           energy_hours = energy_hours,
                                           self_discharge = self_discharge,
                                           throughput_cost = throughput_cost,
                                           initial_fraction = initial_fraction,
                                           reserve_fraction = reserve_fraction)

    # `:pocket` is the two-region product process; `:rotating` is the
    # time-varying joint one, whose regions are the batteries' own neighbourhoods.
    sampler = if demand_regime === :rotating
        rotating_regime_sampler(source.network;
                                centers = isempty(region_centers) ? buses : region_centers,
                                horizon = horizon,
                                groups = region_groups,
                                regime...)
    elseif demand_regime === :pocket
        regional_demand_sampler(source.network;
                                pocket_buses = pocket,
                                quiet_stages = quiet_stages,
                                horizon = horizon,
                                pocket_atoms = pocket_atoms,
                                rest_atoms = rest_atoms)
    else
        throw(ArgumentError("demand_regime must be :pocket or :rotating, got :$demand_regime"))
    end
    support = freeze_demand_support(sampler, source.network, horizon;
                                    seed = protocol_seed,
                                    method = :exact,
                                    profile = research_profile(horizon;
                                                               low = profile_low,
                                                               high = profile_high),
                                    profile_period = 24,
                                    protocol_seed = protocol_seed,
                                    stage_hours = 1.0)

    return build_case(source;
                      dir = dir,
                      batteries = fleet,
                      support = support,
                      recourse = recourse_prices(source.network; factor = recourse_factor),
                      placement = Dict{String,Any}(
                          "recourse_factor" => Float64(recourse_factor),
                          "buses" => placement_record,
                          "capacity" => capacity_record,
                          "capacity_rule" => "power = $(power_fraction) × nominal $(power_basis === :region ? "region" : "system") load; energy = $(energy_hours) h",
                          "power_basis" => String(power_basis),
                          "pocket_buses" => sort!(collect(Int.(pocket))),
                          "pocket_fraction" => Float64(pocket_fraction),
                          "profile" => Dict{String,Any}("low" => Float64(profile_low),
                                                        "high" => Float64(profile_high),
                                                        "peak_hour" => 19,
                                                        "period" => 24),
                          "pocket_atoms" => Dict{String,Any}("values" => Float64.(pocket_atoms[1]),
                                                             "probabilities" => Float64.(pocket_atoms[2])),
                          "rest_atoms" => Dict{String,Any}("values" => Float64.(rest_atoms[1]),
                                                           "probabilities" => Float64.(rest_atoms[2])),
                          "demand_regime" => String(demand_regime),
                          "regime" => Dict{String,Any}(String(k) => v for (k, v) in pairs(regime)),
                          "sampler" => sampler_to_dict(sampler),
                          "report_stages" => Int(report_stages),
                          "lookahead_stages" => Int(horizon) - Int(report_stages)),
                      protocol_stages = Int(horizon),
                      protocol_scenarios = Int(protocol_scenarios),
                      screening_seed = Int(screening_seed),
                      screening_scenarios = Int(screening_scenarios),
                      mirror = mirror,
                      quiet = quiet)
end

"Explicit battery buses from `DR_BAT_BUSES`, or `nothing` for the seeded draw."
function _env_buses()
    raw = get(ENV, "DR_BAT_BUSES", "")
    isempty(strip(raw)) && return nothing
    return [parse(Int, strip(x)) for x in split(raw, ",") if !isempty(strip(x))]
end

# ─────────────────────────────────────────────────────────────────────────────
# Engineered relaxation-gap modules (Phase 3A)
# ─────────────────────────────────────────────────────────────────────────────

"""
    CycleModuleSpec(; kwargs...)

The parameters of one meshed cycle module. Every field is recorded in the
manifest and enters the case digest, so a module is reproducible from numbers
alone.

# Fields
- `load_p`, `load_q`: the remote bus's demand, pu.
- `local_pmax`, `local_cost`: the remote generator's capacity and LINEAR cost.
  `pmin` is zero, so it is a convex, always-feasible fallback and never a
  commitment decision.
- `local_qmin`, `local_qmax`: its reactive band, sized so reactive feasibility
  at the remote bus is never the binding physics.
- `r_hm, x_hm, r_mr, x_mr, r_hr, x_hr`: the three cycle branches' impedances.
- `rate_hm, rate_mr, rate_hr`: their apparent-power ratings.
- `vmin`, `vmax`: voltage band of the two new buses.
- `base_kv`: carried from the host bus when zero.

# Notes
The module exists to make ONE omitted condition matter. `SOCWRConicPowerModel`
constrains each branch's voltage-product block by
`wr^2 + wi^2 <= w_fr * w_to` but never requires the recovered angle differences
to sum to zero around a cycle, so on a loop it admits `W` matrices that no real
voltage profile realises. On a radial network the condition is vacuous, which is
why the relaxation is exact there.

The amplifier is loop RESISTANCE. Around a cycle the relaxation can settle on a
`W` whose implied flows do not pay the full `I^2 R` a consistent voltage profile
would, so SOC systematically believes the remote bus can be served more cheaply
through the loop than AC can actually serve it. The expensive local generator
turns that belief into money: where SOC imports cheap power around the cycle,
AC must dispatch the local unit instead.

This is a hypothesis about mechanism, and the point of the Phase 3A gate is to
measure whether it produces a different STORAGE ACTION rather than merely a
different objective level. A relaxation gap in cost alone changes nothing about
what a policy should do.
"""
Base.@kwdef struct CycleModuleSpec
    load_p::Float64 = 0.60
    load_q::Float64 = 0.20
    local_pmax::Float64 = 1.20
    local_cost::Float64 = 260.0
    local_qmin::Float64 = -0.80
    local_qmax::Float64 = 0.80
    r_hm::Float64 = 0.030
    x_hm::Float64 = 0.030
    r_mr::Float64 = 0.030
    x_mr::Float64 = 0.030
    r_hr::Float64 = 0.010
    x_hr::Float64 = 0.060
    rate_hm::Float64 = 0.50
    rate_mr::Float64 = 0.50
    rate_hr::Float64 = 0.50
    vmin::Float64 = 0.94
    vmax::Float64 = 1.06
    base_kv::Float64 = 0.0
end

"""
    attach_cycle_module!(network, host_bus, spec; tag) -> NamedTuple

Attach one triangular meshed module to `host_bus` IN PLACE, and report every
identifier and parameter it added.

# Arguments
- `network`: a parsed PowerModels network, mutated in place.
- `host_bus`: any existing in-service bus. The module is generic; nothing here
  is specific to one PGLib case.
- `spec::CycleModuleSpec`: the parameters.

# Returns
A `NamedTuple` with the new `mid_bus`, `remote_bus`, the three `branches`, the
`gen` and `load` identifiers, and `record`, a plain dictionary of everything
added — which is what travels into the manifest and the digest.

# Notes
Identifiers are allocated above the current maximum of EVERY table, so they
cannot collide with the backbone's own nonconsecutive numbering. Units are
untouched: the network stays in whatever per-unit convention it arrived in, and
`base_kv` is inherited from the host bus unless overridden.

The topology is deliberately the smallest object that carries a cycle: host,
one intermediate bus, one remote bus, three branches. The battery and the
uncertainty attach at the remote bus, so the storage state sits BEHIND the loop
and its value depends on how the loop is modelled — which is the whole point.
"""
function attach_cycle_module!(network::AbstractDict, host_bus::Integer,
                              spec::CycleModuleSpec = CycleModuleSpec();
                              tag::AbstractString = "cyc")
    haskey(network["bus"], string(host_bus)) ||
        throw(ArgumentError("host bus $host_bus is not in the network"))
    host = network["bus"][string(host_bus)]
    Int(get(host, "bus_type", 1)) == 4 &&
        throw(ArgumentError("host bus $host_bus is out of service"))

    nextid(tbl) = isempty(get(network, tbl, Dict())) ? 1 :
                  maximum(parse(Int, k) for k in keys(network[tbl])) + 1
    bkv = spec.base_kv > 0 ? spec.base_kv : Float64(get(host, "base_kv", 1.0))

    mid = nextid("bus")
    rem = mid + 1
    for (id, nm) in ((mid, "$(tag)_mid"), (rem, "$(tag)_remote"))
        network["bus"][string(id)] = Dict{String,Any}(
            "index" => id, "bus_i" => id, "bus_type" => 1,
            "vmin" => spec.vmin, "vmax" => spec.vmax,
            "vm" => 1.0, "va" => 0.0, "base_kv" => bkv,
            "zone" => Int(get(host, "zone", 1)), "area" => Int(get(host, "area", 1)),
            "name" => nm, "source_id" => Any["bus", id])
    end

    b1, b2, b3 = nextid("branch"), nextid("branch") + 1, nextid("branch") + 2
    function addbranch!(id, f, t, r, x, rate)
        network["branch"][string(id)] = Dict{String,Any}(
            "index" => id, "f_bus" => f, "t_bus" => t,
            "br_r" => r, "br_x" => x,
            "g_fr" => 0.0, "b_fr" => 0.0, "g_to" => 0.0, "b_to" => 0.0,
            "tap" => 1.0, "shift" => 0.0, "br_status" => 1,
            "angmin" => -pi / 3, "angmax" => pi / 3,
            "rate_a" => rate, "rate_b" => rate, "rate_c" => rate,
            "transformer" => false, "source_id" => Any["branch", id])
    end
    addbranch!(b1, Int(host_bus), mid, spec.r_hm, spec.x_hm, spec.rate_hm)
    addbranch!(b2, mid, rem, spec.r_mr, spec.x_mr, spec.rate_mr)
    addbranch!(b3, Int(host_bus), rem, spec.r_hr, spec.x_hr, spec.rate_hr)

    gid = nextid("gen")
    network["gen"][string(gid)] = Dict{String,Any}(
        "index" => gid, "gen_bus" => rem,
        "pg" => 0.0, "qg" => 0.0,
        "pmin" => 0.0, "pmax" => spec.local_pmax,
        "qmin" => spec.local_qmin, "qmax" => spec.local_qmax,
        "vg" => 1.0, "mbase" => Float64(network["baseMVA"]), "gen_status" => 1,
        # Linear and convex on purpose: an expensive fallback, never a
        # commitment decision, so nothing here makes the problem nonconvex.
        "model" => 2, "ncost" => 2, "cost" => [spec.local_cost, 0.0],
        "startup" => 0.0, "shutdown" => 0.0,
        "source_id" => Any["gen", gid])

    lid = nextid("load")
    network["load"][string(lid)] = Dict{String,Any}(
        "index" => lid, "load_bus" => rem,
        "pd" => spec.load_p, "qd" => spec.load_q, "status" => 1,
        "source_id" => Any["load", lid])

    record = Dict{String,Any}(
        "tag" => String(tag), "host_bus" => Int(host_bus),
        "mid_bus" => mid, "remote_bus" => rem,
        "branch_hm" => b1, "branch_mr" => b2, "branch_hr" => b3,
        "gen" => gid, "load" => lid, "base_kv" => bkv,
        "spec" => Dict{String,Any}(string(f) => getfield(spec, f)
                                   for f in fieldnames(CycleModuleSpec)))
    return (mid_bus = mid, remote_bus = rem, branches = (b1, b2, b3),
            gen = gid, load = lid, record = record)
end

"""
    attach_scheduled_generator!(network, bus; pmax, cost, pmin=0.0, qmin=0.0,
                                qmax=0.0, availability=Float64[], tag="sched")
        -> NamedTuple

Attach one dispatchable generator to an existing bus IN PLACE, optionally
available in only some stages, and report everything it added.

# Arguments
- `network`: a parsed PowerModels network, mutated in place.
- `bus`: any existing in-service bus.

# Keywords
- `pmax::Real`, `pmin::Real`: active limits (pu).
- `cost`: polynomial coefficients HIGHEST ORDER FIRST, PowerModels' own
  convention — `[c2, c1, c0]` for `c2 pg^2 + c1 pg + c0`.
- `qmin::Real`, `qmax::Real`: reactive limits (pu). Both default to zero, which
  is a unit that supplies active power only and leaves the reactive picture of
  the case exactly as it was.
- `availability`: per-stage multiplier vector written to
  [`STAGE_AVAILABILITY_KEY`](@ref); empty means available in every stage.
- `tag::AbstractString`: recorded, and used in the generator's name.

# Returns
A `NamedTuple` with the new `gen` identifier and `record`, a plain dictionary of
every parameter — which is what travels into the placement artifact and is
therefore hashed with the case.

# Notes
Generic: any parsed PGLib case, any bus, and every value is an argument, so a
specific unit can be constructed by hand exactly as an automatic placement rule
would construct it. The identifier is allocated above the current maximum of the
generator table, so it cannot collide with a backbone's nonconsecutive numbering.

The cost is required to be CONVEX and nondecreasing over the unit's own operating
interval — a negative quadratic coefficient, or a marginal cost that goes
negative inside `[pmin, pmax]`, would make a stage problem either nonconvex in
its relaxed form or paid to generate, and both would be found much later as a
strange dispatch rather than here as a rejected case.
"""
function attach_scheduled_generator!(network::AbstractDict, bus::Integer;
                                     pmax::Real,
                                     cost::AbstractVector,
                                     pmin::Real = 0.0,
                                     qmin::Real = 0.0,
                                     qmax::Real = 0.0,
                                     availability::AbstractVector = Float64[],
                                     tag::AbstractString = "sched")
    haskey(network["bus"], string(bus)) ||
        throw(ArgumentError("bus $bus is not in the network"))
    Int(get(network["bus"][string(bus)], "bus_type", 1)) == 4 &&
        throw(ArgumentError("bus $bus is out of service"))
    0 <= pmin <= pmax || throw(ArgumentError("need 0 <= pmin <= pmax, got [$pmin, $pmax]"))
    pmax > 0 || throw(ArgumentError("pmax must be strictly positive, got $pmax"))
    qmin <= qmax || throw(ArgumentError("need qmin <= qmax, got [$qmin, $qmax]"))
    c = Float64.(collect(cost))
    (!isempty(c) && all(isfinite, c)) ||
        throw(ArgumentError("cost must be a non-empty vector of finite coefficients"))
    length(c) <= 3 ||
        throw(ArgumentError("only constant, linear and quadratic costs are supported, got $(length(c)) coefficients"))
    # Highest order first: the quadratic coefficient is c[end-2] when present.
    c2 = length(c) >= 3 ? c[end - 2] : 0.0
    c1 = length(c) >= 2 ? c[end - 1] : 0.0
    c2 >= 0 || throw(ArgumentError("quadratic cost coefficient $c2 is negative, so the cost is not convex"))
    for p in (Float64(pmin), Float64(pmax))
        2 * c2 * p + c1 >= 0 ||
            throw(ArgumentError("marginal cost $(2 * c2 * p + c1) at pg = $p is negative"))
    end
    av = Float64.(collect(availability))
    all(x -> isfinite(x) && x >= 0, av) ||
        throw(ArgumentError("availability multipliers must be finite and nonnegative, got $av"))

    gid = isempty(get(network, "gen", Dict())) ? 1 :
          maximum(parse(Int, k) for k in keys(network["gen"])) + 1
    gen = Dict{String,Any}(
        "index" => gid, "gen_bus" => Int(bus),
        "pg" => 0.0, "qg" => 0.0,
        "pmin" => Float64(pmin), "pmax" => Float64(pmax),
        "qmin" => Float64(qmin), "qmax" => Float64(qmax),
        "vg" => 1.0, "mbase" => Float64(network["baseMVA"]), "gen_status" => 1,
        "model" => 2, "ncost" => length(c), "cost" => c,
        "startup" => 0.0, "shutdown" => 0.0,
        "name" => "$(tag)_gen", "source_id" => Any["gen", gid])
    isempty(av) || (gen[STAGE_AVAILABILITY_KEY] = av)
    network["gen"][string(gid)] = gen

    record = Dict{String,Any}(
        "tag" => String(tag), "gen" => gid, "bus" => Int(bus),
        "pmin" => Float64(pmin), "pmax" => Float64(pmax),
        "qmin" => Float64(qmin), "qmax" => Float64(qmax),
        "cost" => c, "availability" => av)
    return (gen = gid, record = record)
end

"""
    module_digest(backbone_hash, seed, records) -> String

A deterministic digest over the backbone, the seed and every module parameter.

# Notes
The generated case is never committed, so the digest is what makes a run
identifiable: two runs agree if and only if they built the same network from the
same backbone bytes.
"""
function module_digest(backbone_hash::AbstractString, seed::Integer, records)
    io = IOBuffer()
    print(io, backbone_hash, "|", seed)
    for r in records
        print(io, "|", r["tag"], ":", r["host_bus"], ":", r["mid_bus"], ":", r["remote_bus"])
        for k in sort!(collect(keys(r["spec"])))
            print(io, ",", k, "=", r["spec"][k])
        end
    end
    return bytes2hex(SHA.sha256(take!(io)))[1:16]
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--verify" in ARGS
        verify()
    else
        build()
    end
end
