# The preregistered PGLib battery portfolio.
#
# The study is a PANEL of canonical PGLib systems rather than one engineered
# case, so every construction choice has to be a documented FUNCTION of the
# benchmark's own bytes: given a case name and this file, any reader must land on
# the same network, the same storage buses, the same regions, the same finite
# demand support and the same calibrated demand level, byte for byte, on any
# machine and without the private campaign repository.
#
# That constraint is what shapes the code below.
#
#   * Nothing consults a Julia RNG for a structural choice. `StableRNG` is stable
#     across versions, but "stable" is a property of one library's stream and the
#     panel has to survive a reader who reimplements the rule. Placement keys come
#     from SHA-256 of a documented string, which is a specification, not an
#     implementation.
#   * Nothing iterates a `Dict` and keeps the order. Every loop that can affect a
#     result runs over an explicitly sorted vector of identifiers, so a network
#     parsed twice — or parsed by a different JSON reader — gives one answer.
#   * Nothing is calibrated per case by hand. One rule, one margin, one search,
#     frozen before any method is run, and the margin is a constant in this file
#     rather than an argument a caller could tune case by case.
#
# The public entry points are `build_portfolio_case` (construct one frozen case),
# `freeze_portfolio` (construct the whole panel and write the manifest),
# `materialize_portfolio_case` (regenerate one case from the manifest and check
# it against the recorded digests) and `verify_portfolio_manifest`.
#
# Commands:
#   julia --project=. battery_portfolio.jl --list
#   julia --project=. battery_portfolio.jl --case pglib_opf_case118_ieee --out DIR
#   julia --project=. battery_portfolio.jl --verify
#   julia --project=. battery_portfolio.jl --freeze --out DIR

using PGLib
using PowerModels
using HiGHS
using JSON
using LinearAlgebra
using Printf
using SHA
using Statistics
import MathOptInterface as MOI

@isdefined(acquire_pglib_case) || include(joinpath(@__DIR__, "build_battery_case.jl"))
@isdefined(base_feasibility) || include(joinpath(@__DIR__, "battery_diagnostics.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# A — the preregistered constants
#
# Everything in this section was fixed before any method was run on any case.
# Changing one of them invalidates the panel rather than adjusting it, which is
# why they are `const` in the public file and not keyword arguments.
# ─────────────────────────────────────────────────────────────────────────────

"Schema tag of the portfolio manifest."
const PORTFOLIO_SCHEMA = "battery_storage_opf/portfolio/1"

"""
The one global seed. It enters every derived quantity only through SHA-256,
together with the canonical digest of the network the quantity belongs to, so
two cases never share a draw and no quantity depends on the order the panel was
built in.
"""
const PORTFOLIO_SEED = 20260814

"Horizon of every frozen case, in stages."
const PORTFOLIO_HORIZON = 24

"Duration of one stage, in hours. The profile below is hourly, so this is 1."
const PORTFOLIO_STAGE_HOURS = 1.0

"""
The common normalized daily profile ``h_t``, multiplying the canonical PGLib
`pd` AND `qd` so every realization keeps each load's own power factor.

# Notes
It is the same 24 numbers for every case in the panel: the panel varies the
NETWORK, and a per-case profile would confound the two. Its maximum is 1.0 and
is first attained at stage 11, which is the stage the headroom calibration
searches on.
"""
const PORTFOLIO_PROFILE = [
    0.72, 0.68, 0.65, 0.64, 0.66, 0.72,
    0.80, 0.88, 0.94, 0.98, 1.00, 0.99,
    0.97, 0.95, 0.94, 0.96, 1.00, 1.00,
    0.98, 0.94, 0.90, 0.84, 0.79, 0.75,
]

"Number of PTDF-sensitivity demand regions per case."
const PORTFOLIO_REGIONS = 6

"The multiplier a region carries in its OWN atom."
const PORTFOLIO_REGION_HIGH = 1.15

"The multiplier a region carries in every other region's atom."
const PORTFOLIO_REGION_LOW = 0.97

"""
Lower bound on a region's share of the case's nominal active demand.

# Notes
Six regions and an unconstrained clustering do not make six LEVERS. The atoms
move demand by region, so a region carrying 0.2 % of the load is an atom that
moves nothing: measured on the first freeze, `case1951_rte` put 80.8 % of demand
in one region and `case300_ieee` 61.5 %, and on those cases the six atoms
collapse toward two directions — one region high, everything else low.

The bounds below are what make all six atoms carry comparable demand mass. They
are a CONSTRUCTION gate, fixed before any method result existed, and applied to
every case of the panel rather than to the ones that looked worst.
"""
const PORTFOLIO_REGION_SHARE_MIN = 0.08

"Upper bound on a region's share of the case's nominal active demand."
const PORTFOLIO_REGION_SHARE_MAX = 0.28

"""
Smallest admissible effective region count,
``N_{eff} = 1/\\sum_r s_r^2`` over the region demand shares.

# Notes
The inverse Simpson index of the demand shares: the number of EQUALLY weighted
regions that would produce the same concentration. Six exactly equal regions give
6.0; one region carrying everything gives 1.0. The share bounds imply this gate
on their own — shares in `[0.08, 0.28]` cannot concentrate below about 4.6 — so
it is a redundant check, which is exactly what makes it worth asserting.
"""
const PORTFOLIO_MIN_EFFECTIVE_REGIONS = 4.5

"Maximum balanced-assignment refinement sweeps."
const PORTFOLIO_BALANCE_ITERATIONS = 20

"""
The largest number of branch dimensions a load bus's sensitivity signature is
built from. A signature over every limited corridor of a 2000-bus system is
mostly noise from corridors no demand can move; the cap keeps the clustering
looking at the corridors the demand actually reaches.
"""
const PORTFOLIO_MAX_CORRIDORS = 64

"Fraction of the calibrated peak active demand the whole storage fleet can discharge."
const PORTFOLIO_POWER_SHARE = 0.10

"""
Cap on one bus's raw sizing weight, as a multiple of the median weight over the
SELECTED buses. Nominal demand is heavy-tailed on most PGLib systems and an
uncapped proportional split puts a third of the fleet on one bus, which measures
that bus rather than the network.
"""
const PORTFOLIO_WEIGHT_CAP = 3.0

"Storage duration at full discharge power, in hours."
const PORTFOLIO_DURATION_HOURS = 8.0

"Energy floor, as a fraction of energy capacity."
const PORTFOLIO_RESERVE_FRACTION = 0.05

"Initial energy, as a fraction of energy capacity."
const PORTFOLIO_INITIAL_FRACTION = 0.5

"Charging efficiency ``\\eta^{ch}``."
const PORTFOLIO_CHARGE_EFFICIENCY = 0.95

"Discharging efficiency ``\\eta^{dis}``."
const PORTFOLIO_DISCHARGE_EFFICIENCY = 0.95

"Self-discharge ``\\alpha`` per stage."
const PORTFOLIO_SELF_DISCHARGE = 0.999

"Throughput cost per unit of energy moved, in the case's own objective units."
const PORTFOLIO_THROUGHPUT_COST = 5.0

"Lower end of the headroom search. Below this a case is replaced, not rescaled."
const PORTFOLIO_KAPPA_LO = 0.50

"Upper end of the headroom search."
const PORTFOLIO_KAPPA_HI = 1.25

"""
Spacing of the bracketing scan that precedes the bisection.

# Notes
The admissible set is NOT an interval anchored at `PORTFOLIO_KAPPA_LO`. On
several PGLib systems the base ACP has to spill at low demand — generators whose
`pmin` is positive cannot be switched off in an OPF, so at half load the network
is over-generating and the surplus injection carries it — while the same system
is perfectly clean at 0.9. A bisection that assumed admissibility at the bottom
of the bracket would report those systems as failures of the data gate, which
they are not.

So the search scans this grid DOWNWARD from `PORTFOLIO_KAPPA_HI` first, stops at
the largest admissible grid point, and bisects between it and the inadmissible
point above it. The grid is a constant of this file, identical for every case.
"""
const PORTFOLIO_KAPPA_STEP = 0.05

"Bisection stops when the bracket is narrower than this."
const PORTFOLIO_KAPPA_TOL = 1e-3

"""
The margin below the largest admissible demand level. Fixed before results and
identical for every case: a per-case margin is a per-case tuning knob wearing a
safety argument.
"""
const PORTFOLIO_KAPPA_MARGIN = 0.95

"""
How many grid steps the frozen level may retreat when the full `24 × 6`
verification rejects it.

# Notes
The search gates on the profile's peak and trough, which are the two extremes of
the day — but admissibility is not monotone in the demand level, so an INTERIOR
profile value can fail at a level both extremes accept. Measured on
`case2000_goc`: the search returned `κ_max = 1.159375` and the verification then
rejected `3` of the `144` stage/atom combinations.

Gating the search on all eighteen distinct profile values instead would be
exact, but it multiplies the search by nine and a single 2000-bus ACP solve
takes about twenty-four seconds; the search alone would outlast the job.

So the frozen level RETREATS instead: one grid step down, re-verify, up to this
many times, and the case fails the gate if none of them survives. Every attempt
is recorded. It is a fixed rule with a fixed bound, applied identically to every
case — not a per-case search for a level that happens to work.
"""
const PORTFOLIO_KAPPA_RETREATS = 3

"Largest physical residual a solve may leave and still count as complete (pu)."
const PORTFOLIO_RESIDUAL_TOL = 1e-7

"""
Largest nodal deficit or surplus injection that still counts as none (pu).

# Notes
Deliberately looser than the residual tolerance. The recourse variables are
priced far above any generator, so an interior-point method leaves them at a
small positive value rather than exactly zero even when nothing is short; what
must not happen is a solve that USES them.
"""
const PORTFOLIO_RECOURSE_TOL = 1e-6

"Columns of the screening protocol."
const PORTFOLIO_SCREENING_SCENARIOS = 32

"Columns of the final protocol."
const PORTFOLIO_FINAL_SCENARIOS = 500

"The preregistered primary panel, in order."
const PORTFOLIO_PRIMARY = [
    "pglib_opf_case118_ieee",
    "pglib_opf_case162_ieee_dtc",
    "pglib_opf_case179_goc",
    "pglib_opf_case200_activ",
    "pglib_opf_case240_pserc",
    "pglib_opf_case300_ieee",
    "pglib_opf_case500_goc",
    "pglib_opf_case588_sdet",
    "pglib_opf_case793_goc",
    "pglib_opf_case1354_pegase",
    "pglib_opf_case1888_rte",
    "pglib_opf_case2000_goc",
]

"""
The reserve panel, in order. A primary case is replaced only when the canonical
case is unavailable from the pinned PGLib version, or when its UNMODIFIED base
ACP fails at the conservative level `PORTFOLIO_KAPPA_LO`. A failure of SOC, of
DC or of any method is never a reason to replace a case.
"""
const PORTFOLIO_RESERVE = [
    "pglib_opf_case1951_rte",
    "pglib_opf_case2312_goc",
    "pglib_opf_case2383wp_k",
    "pglib_opf_case2736sp_k",
]

"Smallest panel that may be frozen."
const PORTFOLIO_MIN_CASES = 10

"The public command a reader runs to regenerate one case."
const PORTFOLIO_COMMAND =
    "julia --project=. battery_portfolio.jl --case <PGLIB CASE> --out <OUTPUT DIR>"

# ─────────────────────────────────────────────────────────────────────────────
# B — deterministic keys
#
# Every structural draw in this file is a pure function of three things: the
# portfolio seed, the canonical digest of the network being drawn on, and the
# identifier being drawn for. No RNG object, no stream position, no iteration
# order. The rule below is the specification; a reader who reimplements it in
# another language reproduces the panel exactly.
# ─────────────────────────────────────────────────────────────────────────────

"""
    network_digest(network) -> String

SHA-256 of the canonical JSON encoding of a parsed PGLib network.

# Notes
`canonical_json` sorts object keys and prints numbers in a fixed form, so the
digest is a function of the network's CONTENT and not of the order a `Dict`
happened to iterate in. This digest is the network's identity everywhere in the
portfolio: placement keys, region labels and protocol seeds all consume it, so
two cases that differ by one branch cannot accidentally share a draw.
"""
network_digest(network::AbstractDict) = bytes2hex(sha256(canonical_json(plain(network))))

"""
    portfolio_key(tag, network_sha, id) -> String

The exact byte string whose SHA-256 is the uniform draw for `id`.

# Arguments
- `tag::AbstractString`: what is being drawn — `"placement"`, `"region-label"`,
  `"protocol/final"`, … Distinct tags give independent draws from the same seed.
- `network_sha::AbstractString`: [`network_digest`](@ref) of the case.
- `id`: the identifier being drawn for; printed with `string`.

# Notes
Written as its own function so the specification is one readable line and the
tests can assert the literal bytes. The trailing newline matters: without a
separator that cannot appear in a field, `("a", "bc")` and `("ab", "c")` would
hash the same.
"""
portfolio_key(tag::AbstractString, network_sha::AbstractString, id) =
    string(PORTFOLIO_SCHEMA, "\n", tag, "\n", PORTFOLIO_SEED, "\n",
           network_sha, "\n", string(id), "\n")

"""
    portfolio_uniform(tag, network_sha, id) -> Float64

A uniform draw on ``(0,1)`` derived from SHA-256 of [`portfolio_key`](@ref).

# Notes
The first eight digest bytes are read big-endian into a `UInt64`, the low 11 bits
are discarded and the remaining 53 are placed on the ``2^{-53}`` grid offset by
half a step. Discarding the low bits is what makes the value exactly
representable, so the draw is the same number on any platform's floating point
rather than the same number up to rounding; the half-step offset keeps it
strictly inside ``(0,1)``, which `-log(u)` requires.
"""
function portfolio_uniform(tag::AbstractString, network_sha::AbstractString, id)
    h = sha256(portfolio_key(tag, network_sha, id))
    x = zero(UInt64)
    for i in 1:8
        x = (x << 8) | UInt64(h[i])
    end
    return (Float64(x >> 11) + 0.5) * 2.0^-53
end

"""
    weighted_selection(candidates, weights, k; tag, network_sha) -> Vector{Int}

Weighted sampling WITHOUT replacement, keyed by hash rather than by an RNG.

# Arguments
- `candidates::AbstractVector{Int}`: the pool, any order; the result does not
  depend on it.
- `weights::AbstractDict{Int,<:Real}`: strictly positive weight per candidate.
- `k::Integer`: how many to select.

# Returns
- The `k` selected identifiers, sorted ascending.

# Notes
The rule is Efraimidis–Spirakis with exponential keys: draw
``u_b \\sim U(0,1)`` and give `b` the key

```math
\\kappa_b = \\frac{-\\log u_b}{w_b},
```

then take the `k` SMALLEST keys. Because ``-\\log u_b`` is a unit exponential,
``\\kappa_b`` is exponential with rate ``w_b``, the smallest of independent
exponentials is bus `b` with probability ``w_b / \\sum w``, and the same holds
recursively on what is left — so this is exactly sampling proportional to weight
without replacement, not an approximation of it.

Ties are broken by ascending bus identifier. A tie needs two 53-bit draws to
coincide, but "break ties by bus id" is a one-line rule that makes the output
independent of the sort algorithm's stability, and that is worth more than the
probability argument.
"""
function weighted_selection(candidates::AbstractVector{<:Integer},
                            weights::AbstractDict{<:Integer,<:Real},
                            k::Integer;
                            tag::AbstractString, network_sha::AbstractString)
    pool = sort!(collect(Int.(candidates)))
    allunique(pool) || error("weighted selection pool contains duplicates")
    0 <= k <= length(pool) ||
        throw(ArgumentError("cannot select $k of $(length(pool)) candidates"))
    keyed = Tuple{Float64,Int}[]
    for b in pool
        w = Float64(weights[b])
        w > 0 || error("bus $b has non-positive selection weight $w")
        u = portfolio_uniform(tag, network_sha, b)
        push!(keyed, (-log(u) / w, b))
    end
    sort!(keyed; by = x -> (x[1], x[2]))
    return sort!([b for (_, b) in keyed[1:Int(k)]])
end

"""
    portfolio_seed_for(tag, network_sha) -> Int

A per-case integer seed for a `StableRNG`-driven protocol, derived from the same
key rule.

# Notes
Reduced into `1:2^31-1` so it is a valid seed on every platform and prints as a
small integer in the manifest. Two cases get independent protocols because the
network digest is part of the key; two tags get independent protocols on one case
for the same reason.
"""
function portfolio_seed_for(tag::AbstractString, network_sha::AbstractString)
    h = sha256(portfolio_key(tag, network_sha, "seed"))
    x = zero(UInt64)
    for i in 1:8
        x = (x << 8) | UInt64(h[i])
    end
    return Int(x % UInt64(2147483647)) + 1
end

"""
    digest_of(parts...) -> String

SHA-256 over a newline-joined list of already-stringified parts.

# Notes
Used for the derived digests the manifest records — placement, regions, ratings.
Each caller prints its own fields, so what is hashed is visible at the call site
rather than hidden in a serializer.
"""
digest_of(parts...) = bytes2hex(sha256(join(string.(parts), "\n") * "\n"))

# ─────────────────────────────────────────────────────────────────────────────
# C — eligibility, placement and ratings
# ─────────────────────────────────────────────────────────────────────────────

"""
    connected_component(network) -> Set{Int}

Bus identifiers reachable from the reference bus over in-service branches.

# Notes
`eligible_buses` already rejects a bus with no in-service branch, but a PGLib
case can contain a whole ISLAND of interconnected buses that the reference bus
cannot reach. Such an island is dropped by the solved network, so a battery
placed on it would be storage the study never dispatches.
"""
function connected_component(network::AbstractDict)
    refs = sort!([Int(b["index"]) for (_, b) in network["bus"]
                  if Int(get(b, "bus_type", 1)) == 3])
    isempty(refs) && error("the network declares no reference bus")
    adj = Dict{Int,Vector{Int}}()
    for k in sort!(collect(keys(network["branch"])))
        br = network["branch"][k]
        Int(get(br, "br_status", 1)) == 0 && continue
        f, t = Int(br["f_bus"]), Int(br["t_bus"])
        push!(get!(adj, f, Int[]), t)
        push!(get!(adj, t, Int[]), f)
    end
    seen = Set{Int}(refs)
    stack = copy(refs)
    while !isempty(stack)
        v = pop!(stack)
        for w in sort!(get(adj, v, Int[]))
            w in seen && continue
            push!(seen, w)
            push!(stack, w)
        end
    end
    return seen
end

"""
    portfolio_eligible_buses(network) -> Vector{Int}

The storage-eligible buses of a case: in service, positive nominal active
demand, and retained in the connected solved network.
"""
function portfolio_eligible_buses(network::AbstractDict)
    load_at = nominal_load_at_bus(network)
    component = connected_component(network)
    pool = [b for b in eligible_buses(network)
            if get(load_at, b, 0.0) > 0 && b in component]
    isempty(pool) && error("no storage-eligible bus remains")
    return sort!(pool)
end

"""
    portfolio_battery_count(network) -> Int

The preregistered fleet size,

```math
n_{bat} = \\min\\Bigl(|\\mathcal B_{elig}|,\\;
    \\mathrm{clamp}\\bigl(\\mathrm{round}(0.20\\, n_{bus}),\\, 24,\\, 240\\bigr)\\Bigr).
```
"""
function portfolio_battery_count(network::AbstractDict)
    n_bus = length(network["bus"])
    return min(length(portfolio_eligible_buses(network)),
               clamp(round(Int, 0.20 * n_bus), 24, 240))
end

"""
    portfolio_placement(network, network_sha) -> (buses, record)

The reproducible storage placement: `portfolio_battery_count` buses drawn from
the eligible pool without replacement, with probability proportional to nominal
active demand, using [`weighted_selection`](@ref).
"""
function portfolio_placement(network::AbstractDict, network_sha::AbstractString)
    pool = portfolio_eligible_buses(network)
    load_at = nominal_load_at_bus(network)
    weights = Dict{Int,Float64}(b => load_at[b] for b in pool)
    k = min(length(pool), clamp(round(Int, 0.20 * length(network["bus"])), 24, 240))
    buses = weighted_selection(pool, weights, k;
                               tag = "placement", network_sha = network_sha)
    record = Dict{String,Any}(
        "algorithm" => "sha256-exponential-key weighted sampling without replacement",
        "key" => "SHA256(\"$PORTFOLIO_SCHEMA\\nplacement\\n$PORTFOLIO_SEED\\n<network sha256>\\n<bus>\\n\")",
        "weight" => "nominal in-service active demand at the bus (pu)",
        "seed" => PORTFOLIO_SEED,
        "network_sha256" => network_sha,
        "eligible" => pool,
        "count" => k,
        "buses" => buses,
        "digest" => digest_of("portfolio/placement/1", network_sha, PORTFOLIO_SEED,
                              join(buses, ",")),
    )
    return buses, record
end

"""
    portfolio_ratings(network, buses, peak_demand) -> (power, budget, record)

Split the fleet's discharge-power budget across the selected buses.

# Arguments
- `peak_demand::Real`: the case's CALIBRATED peak active demand
  ``\\kappa \\max_t h_t \\sum_i p^{d,0}_i`` (pu).

# Returns
- `power::Dict{Int,Float64}`: discharge (and charge) rating per bus, pu.
- `budget::Float64`: ``0.10\\,\\times`` the calibrated peak, pu.
- `record::Dict`: the manifest record, including the digest of the ratings.

# Notes
Raw weight is the bus's nominal active demand, capped at
`PORTFOLIO_WEIGHT_CAP` times the MEDIAN over the selected buses — the median of
the selection, not of the eligible pool, because the cap is there to keep one
selected bus from dominating the selected fleet.

The budget is distributed by normalized weight and the LAST battery absorbs the
remainder, so the fleet's aggregate power equals the declared budget to within a
few ulps rather than to an accumulated relative error over up to 240 divisions.
It cannot be made to agree BITWISE and be checked as such, because the check
would then depend on the order the sum was taken in; the tolerance is a few ulps
of the budget per battery and is asserted here. Both the unrounded weights and
the resulting ratings are recorded.
"""
function portfolio_ratings(network::AbstractDict, buses::AbstractVector{<:Integer},
                           peak_demand::Real)
    load_at = nominal_load_at_bus(network)
    ordered = sort!(collect(Int.(buses)))
    raw = [max(0.0, get(load_at, b, 0.0)) for b in ordered]
    all(>(0), raw) || error("a selected storage bus has non-positive nominal demand")
    cap = PORTFOLIO_WEIGHT_CAP * Statistics.median(raw)
    capped = min.(raw, cap)
    budget = PORTFOLIO_POWER_SHARE * Float64(peak_demand)
    budget > 0 || error("the calibrated peak demand must be positive, got $peak_demand")
    total = sum(capped)
    power = Dict{Int,Float64}()
    running = 0.0
    for (i, b) in enumerate(ordered)
        p = i == length(ordered) ? budget - running : budget * capped[i] / total
        p > 0 || error("bus $b received a non-positive power rating $p")
        power[b] = p
        running += p
    end
    # Summed in the order the split accumulated in, because floating-point
    # addition is not associative and a `Dict`'s iteration order is not that one.
    total = sum(power[b] for b in ordered)
    abs(total - budget) <= 8 * eps(budget) * length(ordered) ||
        error("fleet power $total does not equal the budget $budget")
    record = Dict{String,Any}(
        "power_budget_pu" => budget,
        "peak_demand_pu" => Float64(peak_demand),
        "system_power_share" => PORTFOLIO_POWER_SHARE,
        "weight" => "nominal active demand, capped at $(PORTFOLIO_WEIGHT_CAP)× the selected-bus median",
        "weight_cap_pu" => cap,
        "raw_weight_pu" => Dict(string(b) => raw[i] for (i, b) in enumerate(ordered)),
        "capped_weight_pu" => Dict(string(b) => capped[i] for (i, b) in enumerate(ordered)),
        "power_pu" => Dict(string(b) => power[b] for b in ordered),
        "duration_hours" => PORTFOLIO_DURATION_HOURS,
        "reserve_fraction" => PORTFOLIO_RESERVE_FRACTION,
        "initial_fraction" => PORTFOLIO_INITIAL_FRACTION,
        "charge_efficiency" => PORTFOLIO_CHARGE_EFFICIENCY,
        "discharge_efficiency" => PORTFOLIO_DISCHARGE_EFFICIENCY,
        "self_discharge" => PORTFOLIO_SELF_DISCHARGE,
        "throughput_cost" => PORTFOLIO_THROUGHPUT_COST,
        "digest" => digest_of("portfolio/ratings/1", budget,
                              join((@sprintf("%d:%.17g", b, power[b]) for b in ordered), ",")),
    )
    return power, budget, record
end

"""
    portfolio_fleet(network, buses, peak_demand) -> (fleet, record)

Build the `BatterySpec` fleet from [`portfolio_ratings`](@ref) and the common
technology constants.
"""
function portfolio_fleet(network::AbstractDict, buses::AbstractVector{<:Integer},
                         peak_demand::Real)
    power, _, record = portfolio_ratings(network, buses, peak_demand)
    fleet, capacity = battery_fleet(network, sort!(collect(Int.(buses)));
                                    power = power,
                                    energy_hours = PORTFOLIO_DURATION_HOURS,
                                    charge_efficiency = PORTFOLIO_CHARGE_EFFICIENCY,
                                    discharge_efficiency = PORTFOLIO_DISCHARGE_EFFICIENCY,
                                    self_discharge = PORTFOLIO_SELF_DISCHARGE,
                                    throughput_cost = PORTFOLIO_THROUGHPUT_COST,
                                    initial_fraction = PORTFOLIO_INITIAL_FRACTION,
                                    reserve_fraction = PORTFOLIO_RESERVE_FRACTION)
    merged = Dict{String,Any}(record)
    merged["capacity_record"] = capacity
    return fleet, merged
end

# ─────────────────────────────────────────────────────────────────────────────
# D — PTDF sensitivity regions
#
# Uniform demand scaling is the one direction that cannot switch a constraint:
# every per-bus sensitivity is averaged out and the network stays qualitatively
# where it was. To make demand a lever on the network's own binding structure the
# regions have to separate buses by HOW they load the corridors, and that map is
# the PTDF matrix.
#
# DC sensitivities are used only to CHOOSE where demand goes. Every number the
# study reports still comes from the true ACP model.
# ─────────────────────────────────────────────────────────────────────────────

"""
    ptdf_matrix(network) -> NamedTuple

DC power transfer distribution factors of the in-service network.

# Returns
A named tuple with
- `M::Matrix{Float64}`: `M[l, i]` is the flow induced on live branch `l` by a
  unit injection at bus position `i`, withdrawn at the reference bus;
- `live::Vector{NTuple{4,Any}}`: `(branch id, f_bus, t_bus, reactance)` per row of
  `M`, in ascending branch-id order;
- `pos::Dict{Int,Int}`: bus identifier → column of `M`;
- `buses::Vector{Int}`: the in-service bus identifiers, ascending;
- `ref::Int`: the reference bus identifier.

# Notes
Built from the DC susceptance matrix ``B'`` and the branch-flow map ``B_f`` as
``M = B_f B'^{-1}`` with the reference row and column removed and the reference
column left at zero. A load increase at bus `i` is a NEGATIVE injection, so
demand at buses with large positive `M[l, i]` unloads branch `l` and demand at
large negative entries loads it — which is why the sensitivity signature below
carries a minus sign.

Rows and columns are ordered by identifier, never by `Dict` iteration, so the
matrix is a function of the network's content alone.
"""
function ptdf_matrix(network::AbstractDict)
    buses = sort!([Int(b["index"]) for (_, b) in network["bus"]
                   if Int(get(b, "bus_type", 1)) != 4])
    isempty(buses) && error("the network has no in-service bus")
    pos = Dict(b => i for (i, b) in enumerate(buses))
    nb = length(buses)
    refbus = minimum(Int(b["index"]) for (_, b) in network["bus"]
                     if Int(get(b, "bus_type", 1)) == 3)
    ref = pos[refbus]

    live = Tuple{Int,Int,Int,Float64}[]
    for k in sort!(collect(keys(network["branch"])); by = x -> Int(network["branch"][x]["index"]))
        br = network["branch"][k]
        Int(get(br, "br_status", 1)) == 0 && continue
        x = Float64(br["br_x"])
        abs(x) > 1e-8 || continue
        f, t = Int(br["f_bus"]), Int(br["t_bus"])
        (haskey(pos, f) && haskey(pos, t)) || continue
        push!(live, (Int(br["index"]), f, t, x))
    end
    nl = length(live)
    nl > 0 || error("the network has no in-service branch with a usable reactance")

    B = zeros(nb, nb)
    Bf = zeros(nl, nb)
    for (e, (_, f, t, x)) in enumerate(live)
        i, j, b = pos[f], pos[t], 1 / x
        B[i, i] += b; B[j, j] += b; B[i, j] -= b; B[j, i] -= b
        Bf[e, i] += b; Bf[e, j] -= b
    end
    keep = setdiff(1:nb, ref)
    M = zeros(nl, nb)
    M[:, keep] = Bf[:, keep] / B[keep, keep]
    return (M = M, live = live, pos = pos, buses = buses, ref = refbus)
end

"""
    select_corridors(network, ptdf; limit=PORTFOLIO_MAX_CORRIDORS) -> Vector{Int}

The branch dimensions a load bus's sensitivity signature is measured on.

# Returns
- Row indices into `ptdf.M`, ascending.

# Notes
One deterministic rule, applied identically to every case: among in-service
branches carrying a finite positive thermal rating, score

```math
\\rho_l = \\frac{1}{\\overline s_l}
    \\sum_{b\\,\\in\\,\\text{load buses}} \\max\\bigl(0,\\, -M_{l,b}\\bigr)\\, p^{d,0}_b ,
```

the loading this corridor would take on if all the demand that pushes power
THROUGH it were scaled by one, relative to its own rating. It is high exactly
when a corridor is both sensitive and has load behind it — a corridor with a
large sensitivity and no demand on the sending side is not a lever — and it is a
pure function of the network data, so no solve and therefore no solver-dependent
tie enters the panel's identity.

The `limit` highest scores are kept, ties broken by ascending branch identifier.
"""
function select_corridors(network::AbstractDict, ptdf; limit::Integer = PORTFOLIO_MAX_CORRIDORS)
    load_at = nominal_load_at_bus(network)
    loadbus = sort!([b for b in keys(load_at) if load_at[b] > 0 && haskey(ptdf.pos, b)])
    scored = Tuple{Float64,Int,Int}[]
    for (e, (id, _, _, _)) in enumerate(ptdf.live)
        rate = Float64(get(network["branch"][string(id)], "rate_a", Inf))
        (isfinite(rate) && rate > 0) || continue
        ρ = sum(max(0.0, -ptdf.M[e, ptdf.pos[b]]) * load_at[b] for b in loadbus; init = 0.0) / rate
        ρ > 0 || continue
        push!(scored, (ρ, id, e))
    end
    isempty(scored) && error("no in-service corridor carries a finite rating and reachable load")
    sort!(scored; by = x -> (-x[1], x[2]))
    return sort!([e for (_, _, e) in scored[1:min(Int(limit), length(scored))]])
end

"""
    sensitivity_signatures(network, ptdf, corridors) -> (buses, S, weights)

Normalized load-bus sensitivity signatures over the selected corridors.

# Returns
- `buses::Vector{Int}`: the load buses, ascending; row order of `S`.
- `S::Matrix{Float64}`: row `i` is the unit-norm signature of bus `buses[i]`.
- `weights::Vector{Float64}`: nominal active demand per row (pu).

# Notes
The raw entry is ``-M_{l,b}/\\overline s_l`` — the loading a unit of demand at
`b` puts on corridor `l`, relative to that corridor's own rating, so corridors of
very different ratings are commensurable. Rows are then scaled to unit norm, so
clustering sees the SHAPE of a bus's influence and not its size; size enters the
clustering through the demand WEIGHT instead, which is what keeps a region from
being defined by a crowd of tiny loads.
"""
function sensitivity_signatures(network::AbstractDict, ptdf, corridors::AbstractVector{<:Integer})
    load_at = nominal_load_at_bus(network)
    buses = sort!([b for b in keys(load_at) if load_at[b] > 0 && haskey(ptdf.pos, b)])
    S = zeros(length(buses), length(corridors))
    for (j, e) in enumerate(corridors)
        rate = Float64(get(network["branch"][string(ptdf.live[e][1])], "rate_a", Inf))
        for (i, b) in enumerate(buses)
            S[i, j] = -ptdf.M[e, ptdf.pos[b]] / rate
        end
    end
    for i in axes(S, 1)
        n = sqrt(sum(abs2, view(S, i, :)))
        n > 1e-12 && (S[i, :] ./= n)
    end
    return buses, S, [load_at[b] for b in buses]
end

"""
    weighted_kmeans(S, weights, k; iterations=100) -> Vector{Int}

Demand-weighted k-means with deterministic initialization and explicit tie
breaking.

# Arguments
- `S::AbstractMatrix`: one signature per ROW.
- `weights::AbstractVector`: nonnegative weight per row.
- `k::Integer`: number of clusters.

# Returns
- `assign::Vector{Int}`: cluster index per row, in `1:k`.

# Notes
Determinism is the whole point, so nothing here is left to a default:

- **initialization** is farthest-point. The first centre is the row of largest
  weight (ties: lowest row index); each further centre is the row farthest from
  the centres already chosen (ties: lowest row index). No RNG is consulted, so
  there is no seed to record and no stream to depend on;
- **assignment** takes the lowest centre index among equal distances;
- **update** is the weight-weighted mean of the members, which is what makes a
  region follow the demand rather than the count of buses;
- an **empty** cluster is reseeded to the row that is farthest from its own
  centre among clusters that still have at least two members (ties: lowest row
  index), so `k` regions are always returned and every region is nonempty;
- the loop stops when no assignment changes, and in any case after `iterations`
  sweeps, so it terminates on a case where two configurations alternate.
"""
function weighted_kmeans(S::AbstractMatrix{Float64}, weights::AbstractVector{<:Real},
                         k::Integer; iterations::Integer = 100)
    n = size(S, 1)
    n >= k || error("cannot form $k clusters from $n signatures")
    d2(i, c) = sum(abs2, view(S, i, :) .- c)

    centres = Vector{Vector{Float64}}()
    first_row = argmax([(Float64(weights[i]), -i) for i in 1:n])
    push!(centres, collect(view(S, first_row, :)))
    while length(centres) < k
        best_i, best_d = 0, -Inf
        for i in 1:n
            di = minimum(d2(i, c) for c in centres)
            di > best_d && (best_d = di; best_i = i)
        end
        push!(centres, collect(view(S, best_i, :)))
    end

    assign = zeros(Int, n)
    for _ in 1:Int(iterations)
        changed = false
        for i in 1:n
            best_j, best_d = 1, Inf
            for j in 1:k
                dj = d2(i, centres[j])
                dj < best_d && (best_d = dj; best_j = j)
            end
            best_j == assign[i] || (assign[i] = best_j; changed = true)
        end
        for j in 1:k
            members = findall(==(j), assign)
            if isempty(members)
                donor, donor_d = 0, -Inf
                for i in 1:n
                    count(==(assign[i]), assign) >= 2 || continue
                    di = d2(i, centres[assign[i]])
                    di > donor_d && (donor_d = di; donor = i)
                end
                donor == 0 && error("cannot repair an empty cluster")
                assign[donor] = j
                centres[j] = collect(view(S, donor, :))
                changed = true
                continue
            end
            w = sum(Float64(weights[i]) for i in members)
            centres[j] = w > 0 ?
                vec(sum(Float64(weights[i]) .* view(S, i, :) for i in members) ./ w) :
                vec(sum(view(S, i, :) for i in members) ./ length(members))
        end
        changed || break
    end
    return assign
end

"""
    region_dispersion(S, weights, assign, centres) -> Float64

Demand-weighted mean squared PTDF-signature distance to the assigned
representative,

```math
D = \\frac{\\sum_i w_i \\lVert s_i - c_{a(i)} \\rVert^2}{\\sum_i w_i}.
```

# Notes
This is the quantity the balance constraints trade against. Reporting it for the
constrained AND the unconstrained assignment is what keeps the tradeoff visible:
if balancing demand made the regions arbitrary bus partitions, this number would
jump, and the point of the regions — that they are levers on the network's own
binding structure — would be gone.
"""
function region_dispersion(S::AbstractMatrix, weights::AbstractVector,
                           assign::AbstractVector{<:Integer},
                           centres::AbstractVector)
    num = 0.0
    den = 0.0
    for i in axes(S, 1)
        w = Float64(weights[i])
        num += w * sum(abs2, view(S, i, :) .- centres[assign[i]])
        den += w
    end
    return num / den
end

"Demand-weighted centroid of each cluster, in cluster-index order."
function region_centres(S::AbstractMatrix, weights::AbstractVector,
                        assign::AbstractVector{<:Integer}, k::Integer)
    return [begin
                m = findall(==(j), assign)
                w = sum(Float64(weights[i]) for i in m; init = 0.0)
                isempty(m) ? zeros(size(S, 2)) :
                w > 0 ? vec(sum(Float64(weights[i]) .* view(S, i, :) for i in m) ./ w) :
                        vec(sum(view(S, i, :) for i in m) ./ length(m))
            end for j in 1:Int(k)]
end

"""
    balanced_assignment(S, weights, k; share_min, share_max, iterations, optimizer)
        -> NamedTuple

Demand-balanced assignment of signatures to regions.

# Returns
`(assign, centres, iterations, objective, exception, cap)`.

# Notes
The unconstrained k-means of the first freeze minimizes signature distance and
lets the demand fall where it may. This keeps the same objective and the same
features, and adds the constraint that makes the regions usable as levers:

```math
\\min_{x} \\sum_{i,r} w_i \\lVert s_i - c_r \\rVert^2 x_{ir}
\\quad\\text{s.t.}\\quad
\\sum_r x_{ir} = 1,\\;
\\underline s\\,W \\le \\sum_i w_i x_{ir} \\le \\overline s\\,W,\\;
x_{ir} \\in \\{0,1\\}.
```

Representatives are refined the way Lloyd's algorithm refines them — assign,
recompute demand-weighted centroids, repeat — but the assignment step is this
integer program rather than a nearest-centre rule, so the demand bounds hold at
every sweep and not merely at the end. Initialization is the same deterministic
farthest-point seeding the unconstrained clustering used. The sweep with the
lowest objective wins, ties going to the earliest, and the loop stops as soon as
an assignment repeats.

**The single-heavy-bus exception.** A bus is indivisible. If one bus alone
carries more than `share_max` of the demand, no assignment can respect the upper
bound and the program is infeasible as written. The minimum necessary relaxation
is then applied and recorded: the cap rises to that bus's own share, and a
`Σ y_r ≤ 1` constraint permits exactly ONE region to use it. The other five stay
under `share_max`.

Solved with HiGHS through JuMP — a maintained open-source solver, no proprietary
reproduction requirement. Determinism rests on HiGHS being deterministic for
fixed input, options and thread count, which is why `threads` is pinned to one
and both MIP gaps to zero; the manifest's digests catch any drift.
"""
function balanced_assignment(S::AbstractMatrix{Float64}, weights::AbstractVector{<:Real},
                             k::Integer;
                             share_min::Real = PORTFOLIO_REGION_SHARE_MIN,
                             share_max::Real = PORTFOLIO_REGION_SHARE_MAX,
                             iterations::Integer = PORTFOLIO_BALANCE_ITERATIONS,
                             optimizer = HiGHS.Optimizer)
    n = size(S, 1)
    K = Int(k)
    n >= K || error("cannot form $K regions from $n signatures")
    w = Float64.(collect(weights))
    W = sum(w)
    W > 0 || error("the total assignment weight is zero")

    # The indivisible-bus exception, decided from the data before any solve.
    heaviest = maximum(w) / W
    exception = heaviest > share_max
    cap = exception ? heaviest : Float64(share_max)
    share_min * (K - 1) + cap <= 1.0 + 1e-12 ||
        error("a single bus carries $(round(100 * heaviest; digits = 2)) % of demand; " *
              "$K regions cannot also each hold $(share_min) of it")

    # Deterministic farthest-point seeding, identical to the unconstrained rule.
    d2(i, c) = sum(abs2, view(S, i, :) .- c)
    centres = Vector{Vector{Float64}}()
    push!(centres, collect(view(S, argmax([(w[i], -i) for i in 1:n]), :)))
    while length(centres) < K
        best_i, best_d = 0, -Inf
        for i in 1:n
            di = minimum(d2(i, c) for c in centres)
            di > best_d && (best_d = di; best_i = i)
        end
        push!(centres, collect(view(S, best_i, :)))
    end

    seen = Set{Vector{Int}}()
    best_assign, best_centres, best_obj, sweeps = Int[], centres, Inf, 0
    for it in 1:Int(iterations)
        sweeps = it
        model = JuMP.Model(optimizer)
        JuMP.set_silent(model)
        for (opt, val) in ("threads" => 1, "mip_rel_gap" => 0.0,
                           "mip_abs_gap" => 0.0, "random_seed" => 0)
            try
                JuMP.set_attribute(model, opt, val)
            catch
                # An option this solver build does not expose is not a reason to
                # stop; determinism is asserted by the regression suite.
            end
        end
        JuMP.@variable(model, x[1:n, 1:K], Bin)
        JuMP.@constraint(model, [i = 1:n], sum(x[i, r] for r in 1:K) == 1)
        JuMP.@constraint(model, [r = 1:K], sum(w[i] * x[i, r] for i in 1:n) >= share_min * W)
        if exception
            JuMP.@variable(model, y[1:K], Bin)
            JuMP.@constraint(model, sum(y) <= 1)
            JuMP.@constraint(model, [r = 1:K],
                             sum(w[i] * x[i, r] for i in 1:n) <=
                             share_max * W + y[r] * (cap - share_max) * W)
        else
            JuMP.@constraint(model, [r = 1:K], sum(w[i] * x[i, r] for i in 1:n) <= cap * W)
        end
        cost = [w[i] * d2(i, centres[r]) for i in 1:n, r in 1:K]
        JuMP.@objective(model, Min, sum(cost[i, r] * x[i, r] for i in 1:n, r in 1:K))
        JuMP.optimize!(model)
        JuMP.termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED) ||
            error("the balanced assignment did not solve: $(JuMP.termination_status(model))")

        assign = [argmax([JuMP.value(x[i, r]) for r in 1:K]) for i in 1:n]
        obj = JuMP.objective_value(model)
        if obj < best_obj - 1e-12
            best_obj, best_assign, best_centres = obj, copy(assign), copy(centres)
        end
        assign in seen && break
        push!(seen, copy(assign))
        centres = region_centres(S, w, assign, K)
    end
    return (assign = best_assign, centres = best_centres, iterations = sweeps,
            objective = best_obj, exception = exception, cap = cap)
end

"""
    portfolio_regions(network; nregions=PORTFOLIO_REGIONS) -> NamedTuple

The case's demand regions, from PTDF sensitivity signatures.

# Returns
A named tuple with `regions::Vector{Vector{Int}}` (bus identifiers per region,
ascending), `sizes`, `demand_pu`, `demand_share`, `corridors` (branch
IDENTIFIERS, ascending) and `digest`.

# Notes
Region IDENTITY is canonicalized after clustering: regions are relabelled in
descending total nominal demand, ties broken by the lowest bus identifier they
contain. Without that step the labels would carry the order the initialization
happened to pick centres in, and "region 3" would not mean the same thing to a
reader who reparsed the network.

Every region is nonempty and contains load by construction — the rows being
clustered are exactly the buses with positive nominal active demand — and the
regions partition those buses, which is what makes each region's support mean
exactly one.
"""
function portfolio_regions(network::AbstractDict; nregions::Integer = PORTFOLIO_REGIONS)
    ptdf = ptdf_matrix(network)
    corridors = select_corridors(network, ptdf)
    buses, S, weights = sensitivity_signatures(network, ptdf, corridors)

    balanced = balanced_assignment(S, weights, nregions)
    assign = balanced.assign

    # The unconstrained clustering is still computed, for one reason: it is the
    # baseline the balance constraints are traded against, and a tradeoff nobody
    # measured is a tradeoff nobody made.
    free_assign = weighted_kmeans(S, weights, nregions)
    free_centres = region_centres(S, weights, free_assign, nregions)
    dispersion = region_dispersion(S, weights, assign, balanced.centres)
    free_dispersion = region_dispersion(S, weights, free_assign, free_centres)

    raw = [[buses[i] for i in 1:length(buses) if assign[i] == j] for j in 1:Int(nregions)]
    load = [sum(weights[i] for i in 1:length(buses) if assign[i] == j; init = 0.0)
            for j in 1:Int(nregions)]
    all(!isempty, raw) || error("the balanced assignment left a region empty")
    all(>(0), load) || error("a region carries no demand")
    order = sort(1:Int(nregions); by = j -> (-load[j], minimum(raw[j])))
    regions = [sort!(raw[j]) for j in order]
    demand = [load[j] for j in order]
    total = sum(demand)
    share = demand ./ total
    n_eff = 1 / sum(abs2, share)
    ids = [ptdf.live[e][1] for e in corridors]

    # Fail closed on the gate, not on a report of it. The bounds are checked
    # against the SOLVED shares rather than against the constraints that were
    # written, because a constraint the solver satisfied to its own feasibility
    # tolerance is not the same statement as a share that is inside the band.
    all(s -> s >= PORTFOLIO_REGION_SHARE_MIN - 1e-9, share) ||
        error("a region share fell below $PORTFOLIO_REGION_SHARE_MIN: $share")
    over = findall(s -> s > PORTFOLIO_REGION_SHARE_MAX + 1e-9, share)
    isempty(over) || balanced.exception ||
        error("a region share exceeded $PORTFOLIO_REGION_SHARE_MAX without a " *
              "single-bus exception: $share")
    length(over) <= 1 ||
        error("more than one region exceeded $PORTFOLIO_REGION_SHARE_MAX: $share")
    n_eff >= PORTFOLIO_MIN_EFFECTIVE_REGIONS - 1e-9 || balanced.exception ||
        error("effective region count $n_eff is below $PORTFOLIO_MIN_EFFECTIVE_REGIONS")

    return (regions = regions,
            sizes = [length(r) for r in regions],
            demand_pu = demand,
            demand_share = share,
            effective_regions = n_eff,
            dispersion = dispersion,
            unconstrained_dispersion = free_dispersion,
            unconstrained_share = sort(
                [sum(weights[i] for i in 1:length(buses) if free_assign[i] == j; init = 0.0)
                 for j in 1:Int(nregions)] ./ total; rev = true),
            exception = balanced.exception,
            cap = balanced.cap,
            sweeps = balanced.iterations,
            corridors = ids,
            # Digest tag bumped to /2: the assignment rule changed, so a region
            # digest from the first freeze must not be mistaken for one of these.
            digest = digest_of("portfolio/regions/2", nregions, join(ids, ","),
                               join((join(r, ",") for r in regions), ";")))
end

# ─────────────────────────────────────────────────────────────────────────────
# E — the finite joint demand support
# ─────────────────────────────────────────────────────────────────────────────

"""
    portfolio_mode_matrix(nregions=PORTFOLIO_REGIONS) -> Matrix{Float64}

The `(nregions × nregions)` matrix of joint multiplier atoms: atom `r` gives
region `r` the high multiplier and every other region the low one.

# Notes
With `PORTFOLIO_REGION_HIGH = 1.15`, `PORTFOLIO_REGION_LOW = 0.97` and six
equiprobable atoms every region's support mean is exactly one,

```math
\\frac{1.15 + 5 \\times 0.97}{6} = \\frac{6.00}{6} = 1 ,
```

so the uncertainty adds no expected load and the deterministic profile alone
carries the level. The atoms are locationally NEGATIVELY correlated — a region is
high precisely when the other five are low — which is the structure a single
system-wide multiplier cannot express and the reason a policy has to know WHERE
demand went, not only how much of it there is.
"""
function portfolio_mode_matrix(nregions::Integer = PORTFOLIO_REGIONS)
    M = fill(PORTFOLIO_REGION_LOW, Int(nregions), Int(nregions))
    for r in 1:Int(nregions)
        M[r, r] = PORTFOLIO_REGION_HIGH
    end
    return M
end

"""
    portfolio_sampler(regions) -> JointRegionMultiplier

The authoring sampler of the frozen support: one equiprobable joint atom per
region, applied by BUS.
"""
portfolio_sampler(regions::AbstractVector) =
    JointRegionMultiplier(regions, portfolio_mode_matrix(length(regions)),
                          fill(1 / length(regions), length(regions)); by = :bus)

"""
    portfolio_support(network, regions, κ; horizon=PORTFOLIO_HORIZON, seed) -> DemandSupport

Freeze the panel's finite demand support at demand level `κ`.

# Notes
The temporal profile handed to `freeze_demand_support` is `κ * PORTFOLIO_PROFILE`
— the calibration scales the COMMON profile rather than the atoms, so every case
faces the identically shaped day and the same six multiplier vectors at every
stage, and only the level differs. The freeze is `:exact`: the sampler declares a
finite support, so the atoms are its own and nothing is resampled or reweighted.
"""
function portfolio_support(network::AbstractDict, regions::AbstractVector, κ::Real;
                           horizon::Integer = PORTFOLIO_HORIZON, seed::Integer)
    return freeze_demand_support(portfolio_sampler(regions), network, Int(horizon);
                                 seed = Int(seed),
                                 method = :exact,
                                 profile = Float64(κ) .* PORTFOLIO_PROFILE,
                                 profile_period = length(PORTFOLIO_PROFILE),
                                 protocol_seed = Int(seed),
                                 stage_hours = PORTFOLIO_STAGE_HOURS)
end

# ─────────────────────────────────────────────────────────────────────────────
# F — method-independent ACP headroom calibration
#
# The level is calibrated on the UNMODIFIED network with no batteries and no
# policy target, so nothing a method does can move it. What is being asked is a
# property of the benchmark alone: how much of this common profile can this
# system serve, on true ACP, at every atom, without reaching for the recourse
# injections.
# ─────────────────────────────────────────────────────────────────────────────

"""
    portfolio_probe_case(name, network, support) -> BatteryCase

An in-memory `BatteryCase` carrying no batteries, for the headroom gate.

# Notes
Built in memory rather than through `build_case` because the gate evaluates a
dozen demand levels and writing a 2000-bus network to disk for each of them
would dominate the calibration. Nothing is frozen here and nothing is hashed:
the FROZEN case still goes through `build_case`, which reads it back through the
verifier. The fleet is empty and every gate solve runs with `mode = :none`, so
this is literally the PGLib network plus the two nodal recourse injections.
"""
function portfolio_probe_case(name::AbstractString, network::AbstractDict,
                              support::DemandSupport)
    manifest = Dict{String,Any}("schema" => BATTERY_MANIFEST_SCHEMA,
                                "case" => String(name),
                                "stage_hours" => support.stage_hours)
    return BatteryCase(".", String(name), network, BatterySpec[],
                       recourse_prices(network), support, manifest)
end

"""
    physical_residual(r) -> Float64

Collapse the named tuple `physical_residuals` returns into one number.

# Notes
Its fields are not all of one kind: `branch_flow`, `active_balance`,
`reactive_balance` and `transition` are absolute equation errors and are
nonnegative, while `thermal`, `angle` and `voltage` are signed SLACKS — a
satisfied limit reports a negative number. Taking an absolute value over all
seven would turn the healthiest possible voltage margin into the largest possible
residual, so the slacks are clamped at zero and only a genuine VIOLATION
contributes.
"""
physical_residual(r) = max(r.branch_flow, r.active_balance, r.reactive_balance,
                           r.transition, max(0.0, r.thermal), max(0.0, r.angle),
                           max(0.0, r.voltage))

"""
    admissible_stage(case, t, atom; optimizer=nothing) -> NamedTuple

Solve one base ACP stage and report whether it clears the headroom gate.

# Returns
`(ok, solved, status, cost, residual, recourse)`.

# Notes
Three conditions, all of them fail-closed. The solve must COMPLETE
(`OPTIMAL`/`LOCALLY_SOLVED`); the physical residuals — recomputed from the
solution by `physical_residuals`, independently of whatever the solver reported —
must be at most `PORTFOLIO_RESIDUAL_TOL`; and the worst nodal deficit or surplus
must be at most `PORTFOLIO_RECOURSE_TOL`. A level that needs recourse is a level
at which the case measures the recourse price rather than the network.
"""
function admissible_stage(case::BatteryCase, t::Integer, atom::Integer; optimizer = nothing)
    b = base_feasibility(case, PowerModels.ACPPowerModel;
                         stage = t, atom = atom, optimizer = optimizer)
    if !b.solved
        return (ok = false, solved = false, status = string(b.status),
                cost = NaN, residual = NaN, recourse = NaN)
    end
    r = physical_residual(b.residuals)
    rec = b.worst_recourse
    return (ok = r <= PORTFOLIO_RESIDUAL_TOL && rec <= PORTFOLIO_RECOURSE_TOL,
            solved = true, status = string(b.status), cost = b.cost_stage,
            residual = r, recourse = rec)
end

"""
    peak_stage(profile=PORTFOLIO_PROFILE) -> Int

The FIRST stage attaining the profile's maximum. Three stages of the common
profile reach 1.00; taking the first makes the choice a rule rather than a
preference.
"""
peak_stage(profile::AbstractVector = PORTFOLIO_PROFILE) = argmax(profile)

"""
    trough_stage(profile=PORTFOLIO_PROFILE) -> Int

The FIRST stage attaining the profile's minimum.
"""
trough_stage(profile::AbstractVector = PORTFOLIO_PROFILE) = argmin(profile)

"""
    gate_stages(profile=PORTFOLIO_PROFILE) -> Vector{Int}

The stages the headroom search evaluates: the profile's peak AND its trough.

# Notes
The search cannot run on the peak alone. The profile takes the day down to
`0.64` of the calibrated level, and several PGLib systems cannot follow it: their
in-service generators carry a positive `pmin`, an OPF has no unit commitment to
switch one off, and at low demand the system over-generates and the SURPLUS
injection takes the difference. That is exactly the "meaningful recourse" the
gate exists to forbid, and a level chosen on the peak alone would fail the
`24 × 6` verification at the trough instead.

Both extremes, then, and both at every atom. The interior stages are not implied
by the extremes — admissibility is not monotone in the level — which is why the
full grid is still verified afterwards; what the two extremes buy is that the
verification almost always passes, rather than rejecting a level the search had
no way of knowing was bad.
"""
gate_stages(profile::AbstractVector = PORTFOLIO_PROFILE) =
    sort!(unique([peak_stage(profile), trough_stage(profile)]))

"""
    admissible_level(name, network, regions, κ; stages, seed, optimizer=nothing)
        -> NamedTuple

Whether EVERY atom of every gate stage clears the gate at demand level `κ`.

# Notes
Short-circuits on the first combination that fails, because the gate is a
conjunction and the search only needs the verdict. The stage and atom that failed
are returned, which is what a replacement decision has to be able to cite.
"""
function admissible_level(name::AbstractString, network::AbstractDict,
                          regions::AbstractVector, κ::Real;
                          stages::AbstractVector{<:Integer} = gate_stages(),
                          seed::Integer, optimizer = nothing)
    support = portfolio_support(network, regions, κ; seed = seed)
    case = portfolio_probe_case(name, network, support)
    for t in stages, a in 1:num_atoms(support, Int(t))
        r = admissible_stage(case, t, a; optimizer = optimizer)
        r.ok || return (ok = false, stage = Int(t), atom = a, detail = r)
    end
    return (ok = true, stage = 0, atom = 0,
            detail = (ok = true, solved = true, status = "OPTIMAL",
                      cost = NaN, residual = NaN, recourse = NaN))
end

"""
    calibrate_kappa(name, network, regions; seed, optimizer=nothing, log=true)
        -> NamedTuple

The case's method-independent demand-level calibration.

# Returns
`(ok, kappa_max, kappa_case, evaluations, trace, reason)`. `ok = false` means the
case fails the base-ACP data gate at `PORTFOLIO_KAPPA_LO` and is replaced from
the reserve panel.

# Notes
The search is [`search_kappa_max`](@ref): a downward scan of a fixed grid to
bracket the largest admissible level, then bisection to `PORTFOLIO_KAPPA_TOL`.
The reported `kappa_max` is the largest level actually PROVEN admissible, never
the midpoint of a bracket, so the frozen level is backed by a solve rather than
by an interpolation.

The frozen level is then

```math
\\kappa_{case} = 0.95\\, \\kappa_{\\max},
```

with the margin a constant of this file. No solver setting is touched: the same
optimizer, the same tolerances and the same formulation are used for every case
and every level, because a per-case solver adjustment would make the calibration
a property of the solver rather than of the benchmark.
"""
function calibrate_kappa(name::AbstractString, network::AbstractDict,
                         regions::AbstractVector; seed::Integer,
                         optimizer = nothing, log::Bool = true)
    ts = gate_stages()
    trace = Dict{String,Any}[]
    evals = Ref(0)
    function gate(κ)
        r = admissible_level(name, network, regions, κ; stages = ts, seed = seed,
                             optimizer = optimizer)
        evals[] += 1
        push!(trace, Dict{String,Any}("kappa" => Float64(κ), "admissible" => r.ok,
                                      "first_failing_stage" => r.stage,
                                      "first_failing_atom" => r.atom,
                                      "status" => r.detail.status,
                                      "residual" => r.detail.residual,
                                      "recourse" => r.detail.recourse))
        log && @printf("    κ=%.6f  %s%s\n", κ, r.ok ? "admissible" : "REJECTED",
                       r.ok ? "" : @sprintf("  (stage %d atom %d, %s, residual %.2e, recourse %.2e)",
                                            r.stage, r.atom, r.detail.status,
                                            r.detail.residual, r.detail.recourse))
        return r.ok
    end

    r = search_kappa_max(gate)
    r.ok || return (ok = false, kappa_max = NaN, kappa_case = NaN,
                    evaluations = evals[], trace = trace, reason = r.reason)
    return (ok = true, kappa_max = r.kappa_max,
            kappa_case = PORTFOLIO_KAPPA_MARGIN * r.kappa_max,
            evaluations = evals[], trace = trace, reason = "")
end

"""
    kappa_grid(lo=PORTFOLIO_KAPPA_LO, hi=PORTFOLIO_KAPPA_HI,
               step=PORTFOLIO_KAPPA_STEP) -> Vector{Float64}

The bracketing grid, DESCENDING from `hi`, with `lo` exactly as its last point.

# Notes
Built by subtracting whole multiples of `step` from `hi` rather than by
accumulating additions, so the same values come out on any machine, and the last
point is written as `lo` rather than computed, so the bottom of the bracket is
the number the constant says it is and not one ulp away from it.
"""
function kappa_grid(lo::Real = PORTFOLIO_KAPPA_LO, hi::Real = PORTFOLIO_KAPPA_HI,
                    step::Real = PORTFOLIO_KAPPA_STEP)
    n = round(Int, (hi - lo) / step)
    g = [Float64(hi) - k * Float64(step) for k in 0:n]
    g[end] = Float64(lo)
    return g
end

"""
    search_kappa_max(gate; lo=PORTFOLIO_KAPPA_LO, hi=PORTFOLIO_KAPPA_HI,
                     step=PORTFOLIO_KAPPA_STEP, tol=PORTFOLIO_KAPPA_TOL)
        -> NamedTuple

The panel's deterministic headroom search, isolated from what it is searching on.

# Arguments
- `gate`: a predicate `κ -> Bool` saying whether the case is admissible at that
  demand level.

# Returns
`(ok, kappa_max, reason)`. `ok = false` means NO point of the grid is
admissible, which is the preregistered replacement condition.

# Notes
Two steps, both deterministic:

1. **bracket** — walk [`kappa_grid`](@ref) downward from `hi` and stop at the
   first admissible point. Downward, because the quantity wanted is the LARGEST
   admissible level; and by scan rather than by bisection, because admissibility
   is not monotone in the demand level (see `PORTFOLIO_KAPPA_STEP`).
2. **refine** — bisect between that point and the inadmissible grid point
   immediately above it, until the bracket is narrower than `tol`.

The returned level is always one the gate ACCEPTED, never a midpoint that was
merely bracketed. Separated from [`calibrate_kappa`](@ref) so the search itself
can be tested against a predicate with a known threshold rather than only
through hundreds of ACP solves.
"""
function search_kappa_max(gate; lo::Real = PORTFOLIO_KAPPA_LO,
                          hi::Real = PORTFOLIO_KAPPA_HI,
                          step::Real = PORTFOLIO_KAPPA_STEP,
                          tol::Real = PORTFOLIO_KAPPA_TOL)
    grid = kappa_grid(lo, hi, step)
    for (i, κ) in enumerate(grid)
        gate(κ) || continue
        i == 1 && return (ok = true, kappa_max = κ, reason = "")
        a, b = κ, grid[i - 1]
        while b - a > tol
            mid = 0.5 * (a + b)
            gate(mid) ? (a = mid) : (b = mid)
        end
        return (ok = true, kappa_max = a, reason = "")
    end
    return (ok = false, kappa_max = NaN,
            reason = "the unmodified base ACP clears the gate at no level of " *
                     "[$lo, $hi]; the conservative level $lo fails")
end

"""
    verify_kappa(name, network, regions, κ; seed, optimizer=nothing, log=true)
        -> NamedTuple

Check EVERY stage/atom combination at the frozen level.

# Returns
`(ok, checked, failures)` where `failures` lists `(stage, atom, status, residual,
recourse)` for each combination that did not clear the gate.

# Notes
The search runs on the peak stage alone; this runs on all `24 × 6` of them. Lower
profile stages are not automatically easier — a lightly loaded system can be the
one that has to spill — so the verification is exhaustive rather than argued.
"""
function verify_kappa(name::AbstractString, network::AbstractDict,
                      regions::AbstractVector, κ::Real; seed::Integer,
                      optimizer = nothing, log::Bool = true)
    support = portfolio_support(network, regions, κ; seed = seed)
    case = portfolio_probe_case(name, network, support)
    failures = Dict{String,Any}[]
    checked = 0
    for t in 1:support.horizon, a in 1:num_atoms(support, t)
        r = admissible_stage(case, t, a; optimizer = optimizer)
        checked += 1
        r.ok || push!(failures, Dict{String,Any}("stage" => t, "atom" => a,
                                                 "status" => r.status,
                                                 "residual" => r.residual,
                                                 "recourse" => r.recourse))
    end
    log && @printf("    verified %d stage/atom combinations, %d failures\n",
                   checked, length(failures))
    return (ok = isempty(failures), checked = checked, failures = failures)
end

# ─────────────────────────────────────────────────────────────────────────────
# G — building one frozen portfolio case
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_portfolio_case(name; dir, kappa=nothing, verify_all=true,
                         optimizer=nothing, quiet=false) -> NamedTuple

Construct one frozen portfolio case from its PGLib name.

# Keywords
- `dir::AbstractString`: where the four case artifacts are written.
- `kappa`: `nothing` to calibrate the demand level here, or the frozen
  `κ_case` from the portfolio manifest to reproduce a case without repeating the
  search.
- `verify_all::Bool`: check all `24 × 6` stage/atom combinations at the frozen
  level.

# Returns
A named tuple with the read-back `case`, the `record` the portfolio manifest
stores, and the intermediate `regions`, `placement` and `calibration` results.

# Notes
The order of construction is forced by what depends on what: regions are needed
before the support, the support before the headroom gate, the gate before the
peak demand, and the peak demand before the storage ratings. Placement does not
depend on any of them — it is a function of the network alone — but the fleet's
ratings do, which is why the buses are drawn early and rated late.
"""
function build_portfolio_case(name::AbstractString;
                              dir::AbstractString,
                              kappa = nothing,
                              verify_all::Bool = true,
                              optimizer = nothing,
                              regions = nothing,
                              quiet::Bool = false)
    src = acquire_pglib_case(name)
    sha = network_digest(src.network)
    quiet || @printf("  %s: network sha256 %s\n", name, first(sha, 16))

    # The regions may be supplied by a caller that has already computed them.
    # This is a pure cache, never a second source of truth: the balanced
    # assignment is an integer program that takes 38 minutes on `case2000_goc`,
    # and a reader who regenerates a case should pay for it ONCE rather than
    # once for the digest check and again for the build.
    regions = regions === nothing ? portfolio_regions(src.network) : regions
    buses, placement = portfolio_placement(src.network, sha)
    final_seed = portfolio_seed_for("protocol/final", sha)
    screen_seed = portfolio_seed_for("protocol/screening", sha)

    calibration = if kappa === nothing
        quiet || println("  calibrating the ACP headroom")
        calibrate_kappa(name, src.network, regions.regions;
                        seed = final_seed, optimizer = optimizer, log = !quiet)
    else
        (ok = true, kappa_max = Float64(kappa) / PORTFOLIO_KAPPA_MARGIN,
         kappa_case = Float64(kappa), evaluations = 0,
         trace = Dict{String,Any}[], reason = "taken from the portfolio manifest")
    end
    calibration.ok ||
        return (ok = false, case = nothing, record = nothing, regions = regions,
                placement = placement, calibration = calibration,
                verification = nothing)

    # The frozen level, and the retreat if the full grid rejects it. `retreats`
    # counts the grid steps taken; a case that exhausts them fails the gate and
    # is recorded, never quietly frozen at a level the verification rejected.
    κmax = calibration.kappa_max
    κ = calibration.kappa_case
    retreats = 0
    attempts = Dict{String,Any}[]
    verification = (ok = true, checked = 0, failures = Dict{String,Any}[])
    if verify_all
        while true
            verification = verify_kappa(name, src.network, regions.regions, κ;
                                        seed = final_seed, optimizer = optimizer,
                                        log = !quiet)
            push!(attempts, Dict{String,Any}("kappa_max" => κmax, "kappa_case" => κ,
                                             "retreat" => retreats,
                                             "checked" => verification.checked,
                                             "failures" => verification.failures))
            verification.ok && break
            retreats += 1
            quiet || @printf("    verification rejected κ_case=%.6f on %d of %d combinations; retreating one grid step\n",
                             κ, length(verification.failures), verification.checked)
            retreats <= PORTFOLIO_KAPPA_RETREATS || break
            κmax -= PORTFOLIO_KAPPA_STEP
            κ = PORTFOLIO_KAPPA_MARGIN * κmax
            κmax >= PORTFOLIO_KAPPA_LO ||
                return (ok = false, case = nothing, record = nothing, regions = regions,
                        placement = placement,
                        calibration = (ok = false, kappa_max = NaN, kappa_case = NaN,
                                       evaluations = calibration.evaluations,
                                       trace = calibration.trace,
                                       reason = "retreating below $PORTFOLIO_KAPPA_LO"),
                        verification = verification)
        end
    end
    verification.ok ||
        return (ok = false, case = nothing, record = nothing, regions = regions,
                placement = placement,
                calibration = (ok = false, kappa_max = NaN, kappa_case = NaN,
                               evaluations = calibration.evaluations,
                               trace = calibration.trace,
                               reason = "no level within $PORTFOLIO_KAPPA_RETREATS grid " *
                                        "steps of κ_max cleared all 24×6 stage/atom " *
                                        "combinations; the last rejected " *
                                        "$(length(verification.failures)) of " *
                                        "$(verification.checked)"),
                verification = verification)

    load_at = nominal_load_at_bus(src.network)
    peak = κ * maximum(PORTFOLIO_PROFILE) * sum(values(load_at))
    fleet, ratings = portfolio_fleet(src.network, buses, peak)
    support = portfolio_support(src.network, regions.regions, κ; seed = final_seed)

    case = build_case(src; dir = dir, batteries = fleet, support = support,
                      placement = Dict{String,Any}("buses" => placement,
                                                   "capacity" => ratings,
                                                   "regions" => Dict{String,Any}(
                                                       "count" => length(regions.regions),
                                                       "corridors" => regions.corridors,
                                                       "sizes" => regions.sizes,
                                                       "demand_pu" => regions.demand_pu,
                                                       "digest" => regions.digest),
                                                   # κ_case ONLY. The largest
                                                   # admissible level is a
                                                   # byproduct of the search and
                                                   # belongs to the portfolio
                                                   # manifest; putting it in a
                                                   # HASHED case artifact would
                                                   # mean a reader who
                                                   # regenerates the case from
                                                   # the frozen κ_case has to
                                                   # reproduce κ_max = κ_case/0.95
                                                   # to the last bit, and float
                                                   # division does not promise
                                                   # that.
                                                   "kappa" => Dict{String,Any}(
                                                       "kappa_case" => κ,
                                                       "margin" => PORTFOLIO_KAPPA_MARGIN)),
                      protocol_stages = PORTFOLIO_HORIZON,
                      protocol_scenarios = PORTFOLIO_FINAL_SCENARIOS,
                      screening_seed = screen_seed,
                      screening_scenarios = PORTFOLIO_SCREENING_SCENARIOS,
                      quiet = true)

    record = Dict{String,Any}(
        "case" => String(name),
        "network_sha256" => sha,
        "counts" => Dict{String,Any}(
            "bus" => length(src.network["bus"]),
            "gen" => length(src.network["gen"]),
            "branch" => length(src.network["branch"]),
            "load" => length(src.network["load"]),
            "eligible_bus" => length(portfolio_eligible_buses(src.network)),
            "battery" => length(fleet)),
        "kappa_max" => κmax,
        "kappa_case" => κ,
        "kappa_evaluations" => calibration.evaluations,
        "kappa_retreats" => retreats,
        "peak_demand_pu" => peak,
        "nominal_demand_pu" => sum(values(load_at)),
        "placement" => Dict{String,Any}(
            "algorithm" => placement["algorithm"],
            "count" => placement["count"],
            "eligible" => length(placement["eligible"]),
            "buses" => buses,
            "digest" => placement["digest"]),
        "battery" => Dict{String,Any}(
            "power_budget_pu" => ratings["power_budget_pu"],
            "power_min_pu" => minimum(values(ratings["power_pu"])),
            "power_max_pu" => maximum(values(ratings["power_pu"])),
            "energy_budget_puh" => ratings["power_budget_pu"] * PORTFOLIO_DURATION_HOURS,
            "digest" => ratings["digest"]),
        "regions" => Dict{String,Any}(
            "count" => length(regions.regions),
            "corridors" => regions.corridors,
            "sizes" => regions.sizes,
            "demand_pu" => regions.demand_pu,
            "demand_share" => regions.demand_share,
            "effective_regions" => regions.effective_regions,
            # The tradeoff, recorded next to the thing it was traded for.
            "dispersion" => regions.dispersion,
            "unconstrained_dispersion" => regions.unconstrained_dispersion,
            "unconstrained_share" => regions.unconstrained_share,
            "single_bus_exception" => regions.exception,
            "share_cap" => regions.cap,
            "sweeps" => regions.sweeps,
            "digest" => regions.digest),
        "support" => Dict{String,Any}("sha256" => support_digest(support)),
        "protocols" => Dict{String,Any}(
            "final" => Dict{String,Any}(
                "seed" => final_seed,
                "num_stages" => PORTFOLIO_HORIZON,
                "num_scenarios" => PORTFOLIO_FINAL_SCENARIOS,
                "sha256" => case.manifest["protocol"]["sha256"]),
            "screening" => Dict{String,Any}(
                "seed" => screen_seed,
                "num_stages" => PORTFOLIO_HORIZON,
                "num_scenarios" => PORTFOLIO_SCREENING_SCENARIOS,
                "excludes" => "final",
                "sha256" => case.manifest["screening"]["sha256"])),
        "artifacts" => Dict{String,Any}(case.manifest["artifacts"]),
        "verification" => Dict{String,Any}(
            "stage_atom_checked" => verification.checked,
            "stage_atom_failures" => length(verification.failures),
            "retreats" => retreats),
    )
    return (ok = true, case = case, record = record, regions = regions,
            placement = placement, calibration = calibration,
            verification = verification, attempts = attempts)
end

# ─────────────────────────────────────────────────────────────────────────────
# G2 — strict-stage validation of a frozen case
#
# The headroom calibration answers a question about the NETWORK. This answers the
# question about the CASE: with the fleet in place, is the strict formulation
# well posed at every stage — is the state admissible, is the reachable interval
# nonempty, is holding the state feasible, and does an ordinary strict solve
# complete without reaching for recourse.
# ─────────────────────────────────────────────────────────────────────────────

"""
    hold_trajectory(case) -> (states, ok, worst_slack)

Follow the "do nothing" policy — every battery targets the energy it came in
with — and report whether that target is reachable at every stage.

# Returns
- `states::Vector{Dict{Int,Float64}}`: the incoming state at each stage; with a
  hold policy they are all the initial state.
- `ok::Bool`: every reachable interval was nonempty and contained the hold target.
- `worst_slack::Float64`: the smallest distance from a hold target to the nearer
  end of its reachable interval, over all stages and batteries. Negative means
  the hold target was outside.

# Notes
Holding is not free: self-discharge means a battery must charge
``(1-\\alpha)e`` every stage just to stand still, so a hold target is only
reachable when the charging rating covers that. With
``\\alpha = `` `PORTFOLIO_SELF_DISCHARGE` and a duration of
`PORTFOLIO_DURATION_HOURS` hours it does, by a wide margin — but the margin is
measured here rather than argued, because it is a joint property of three
constants that a later revision could change one of.
"""
function hold_trajectory(case::BatteryCase)
    Δt = stage_hours(case)
    e = Dict(b.index => b.energy_initial for b in case.batteries)
    states = Dict{Int,Float64}[]
    ok = true
    worst = Inf
    for _ in 1:case.demand.horizon
        push!(states, copy(e))
        for b in case.batteries
            lo, hi = reachable_interval(b, e[b.index], Δt)
            hi >= lo || (ok = false)
            worst = min(worst, e[b.index] - lo, hi - e[b.index])
            (lo <= e[b.index] <= hi) || (ok = false)
        end
    end
    return states, ok, worst
end

"""
    validate_portfolio_case(case; stages=nothing, atoms=nothing, optimizer=nothing,
                            quiet=false) -> NamedTuple

The strict-stage validation every frozen case must pass.

# Keywords
- `stages`, `atoms`: which combinations to solve. `nothing` means all of them.

# Returns
A named tuple with `ok`, the state and reachability verdicts, the number of
strict solves attempted and completed, the worst residual and recourse over the
completed ones, and the list of failures.

# Notes
Six properties, each fail-closed:

1. every initial energy lies inside its own bounds;
2. every reachable interval is nonempty at every stage of the hold trajectory;
3. the idle/hold target lies inside that interval;
4. every requested strict-stage ACP solve completes;
5. the independently recomputed physical residual is at most
   `PORTFOLIO_RESIDUAL_TOL`;
6. no solve uses a meaningful nodal deficit or surplus.

Properties 5 and 6 are recomputed from the returned solution rather than read
from the solver, because a solver that reports success is asserting its own
convergence criterion and not the network's equations.
"""
function validate_portfolio_case(case::BatteryCase;
                                 stages = nothing, atoms = nothing,
                                 optimizer = nothing, quiet::Bool = false)
    opt = optimizer === nothing ? acp_optimizer() : optimizer
    Δt = stage_hours(case)
    failures = Dict{String,Any}[]

    states_ok = all(b.energy_min <= b.energy_initial <= b.energy_max for b in case.batteries)
    states_ok || push!(failures, Dict{String,Any}("kind" => "initial state out of bounds"))
    _, reach_ok, slack = hold_trajectory(case)
    reach_ok || push!(failures, Dict{String,Any}("kind" => "hold target unreachable",
                                                 "slack" => slack))

    ts = stages === nothing ? (1:case.demand.horizon) : stages
    hold = Dict(b.index => b.energy_initial for b in case.batteries)
    attempted, completed = 0, 0
    worst_res, worst_rec = 0.0, 0.0
    for t in ts
        as = atoms === nothing ? (1:num_atoms(case.demand, t)) : atoms
        for a in as
            attempted += 1
            sol = solve_strict_stage(case, PowerModels.ACPPowerModel;
                                     stage = t, atom = a, energy_in = hold,
                                     target = hold, optimizer = opt)
            if !sol.solved
                push!(failures, Dict{String,Any}("kind" => "strict solve incomplete",
                                                 "stage" => t, "atom" => a,
                                                 "status" => string(sol.status)))
                continue
            end
            completed += 1
            r = physical_residual(physical_residuals(case.network, case.batteries, Δt, sol))
            rec = worst_recourse(sol)
            worst_res = max(worst_res, r)
            worst_rec = max(worst_rec, rec)
            r <= PORTFOLIO_RESIDUAL_TOL ||
                push!(failures, Dict{String,Any}("kind" => "residual above tolerance",
                                                 "stage" => t, "atom" => a, "residual" => r))
            rec <= PORTFOLIO_RECOURSE_TOL ||
                push!(failures, Dict{String,Any}("kind" => "recourse used",
                                                 "stage" => t, "atom" => a, "recourse" => rec))
        end
    end
    quiet || @printf("    strict stages: %d/%d complete, worst residual %.2e, worst recourse %.2e\n",
                     completed, attempted, worst_res, worst_rec)
    return (ok = isempty(failures), states_ok = states_ok, reachable_ok = reach_ok,
            hold_slack = slack, attempted = attempted, completed = completed,
            worst_residual = worst_res, worst_recourse = worst_rec, failures = failures)
end

"""
    aggressive_charge_probe(case; stage=peak_stage(), atom=1, optimizer=nothing)
        -> NamedTuple

The deliberately aggressive diagnostic: every battery targets the TOP of its
reachable interval at one stage.

# Returns
`(solved, recourse, residual, rejected)`; `rejected` is `true` when the solve
used more recourse than `PORTFOLIO_RECOURSE_TOL`.

# Notes
This is a DIAGNOSTIC, not a gate. Charging the whole fleet at full rate on one
stage is an admissible target — it is inside the reachable interval by
construction — that the NETWORK may not be able to serve, and when it cannot, the
right outcome is that the admissibility check rejects the solution. A case is
never altered because this probe draws recourse; a probe that never drew any
would mean the fleet was too small to matter.
"""
function aggressive_charge_probe(case::BatteryCase; stage::Integer = peak_stage(),
                                 atom::Integer = 1, optimizer = nothing)
    opt = optimizer === nothing ? acp_optimizer() : optimizer
    Δt = stage_hours(case)
    e_in = Dict(b.index => b.energy_initial for b in case.batteries)
    target = Dict{Int,Float64}()
    for b in case.batteries
        _, hi = reachable_interval(b, e_in[b.index], Δt)
        target[b.index] = hi
    end
    sol = solve_strict_stage(case, PowerModels.ACPPowerModel;
                             stage = stage, atom = atom, energy_in = e_in,
                             target = target, optimizer = opt)
    sol.solved || return (solved = false, recourse = NaN, residual = NaN, rejected = true)
    rec = worst_recourse(sol)
    r = physical_residual(physical_residuals(case.network, case.batteries, Δt, sol))
    return (solved = true, recourse = rec, residual = r,
            rejected = rec > PORTFOLIO_RECOURSE_TOL)
end

# ─────────────────────────────────────────────────────────────────────────────
# H — the portfolio manifest
# ─────────────────────────────────────────────────────────────────────────────

"""
    portfolio_manifest_path(dir=@__DIR__) -> String

Path of the byte-identical portfolio manifest inside an example directory.
"""
portfolio_manifest_path(dir::AbstractString = @__DIR__) =
    joinpath(dir, "battery_portfolio.json")

"""
    portfolio_digest(manifest) -> String

SHA-256 of the canonical JSON of a manifest with its own `digest` field removed.

# Notes
A manifest that hashed itself including the hash could not be checked, so the
field is excluded and nothing else is. Any edit to any other field — a `κ`, one
bus of one placement, a protocol seed — changes the digest, which is what makes
[`verify_portfolio_manifest`](@ref) a tamper check and not a formatting check.
"""
function portfolio_digest(manifest::AbstractDict)
    body = Dict{String,Any}(k => v for (k, v) in manifest if k != "digest")
    return bytes2hex(sha256(canonical_json(plain(body))))
end

"""
    write_portfolio_manifest(path, cases; replacements=[]) -> Dict

Write the panel manifest as canonical JSON and return it.

# Arguments
- `cases`: the per-case records returned by [`build_portfolio_case`](@ref), in
  panel order.
- `replacements`: one record per primary case that was replaced, each carrying
  the case dropped, the reserve case promoted and the reason.

# Notes
The manifest is deliberately SMALL: it records what cannot be recomputed (the
calibrated levels, the panel order and the replacements) plus the digest of
everything that can, so a reader regenerates the artifacts and checks them
against the record rather than downloading them. Region MEMBERSHIP is a
recomputable quantity and only its digest, sizes and demand shares are stored;
the selected storage buses are stored in full because they are the panel's most
consequential single choice and a reader should be able to read them without
running anything.

Nothing here is a timestamp, a hostname or a path, so two machines that freeze
the same panel write the same bytes.
"""
function write_portfolio_manifest(path::AbstractString, cases::AbstractVector;
                                  replacements::AbstractVector = Any[],
                                  unfilled::AbstractVector = Any[],
                                  min_cases::Integer = PORTFOLIO_MIN_CASES)
    length(cases) >= min_cases ||
        error("the panel freezes at least $min_cases cases, got $(length(cases))")
    manifest = Dict{String,Any}(
        "schema" => PORTFOLIO_SCHEMA,
        "seed" => PORTFOLIO_SEED,
        "horizon" => PORTFOLIO_HORIZON,
        "stage_hours" => PORTFOLIO_STAGE_HOURS,
        "profile" => copy(PORTFOLIO_PROFILE),
        "panel" => Dict{String,Any}(
            "primary" => copy(PORTFOLIO_PRIMARY),
            "reserve" => copy(PORTFOLIO_RESERVE),
            "accepted" => [String(c["case"]) for c in cases],
            "replacements" => collect(replacements),
            # Preregistered slots that no case could fill, because the primary
            # failed the base-ACP data gate and the reserve list ran out. They
            # are recorded rather than backfilled: a case chosen outside the
            # preregistered order to make the count come out right is exactly
            # the selection freedom the preregistration removes.
            "unfilled" => collect(unfilled),
            "replacement_rule" => "a primary case is replaced only when it is unavailable " *
                                  "from the pinned PGLib version or when its unmodified base " *
                                  "ACP fails at demand level $PORTFOLIO_KAPPA_LO; a SOC, DC " *
                                  "or method failure is never a reason to replace a case"),
        "regions" => Dict{String,Any}(
            "count" => PORTFOLIO_REGIONS,
            "high" => PORTFOLIO_REGION_HIGH,
            "low" => PORTFOLIO_REGION_LOW,
            "max_corridors" => PORTFOLIO_MAX_CORRIDORS,
            "algorithm" => "demand-BALANCED assignment over unit-norm PTDF sensitivity " *
                           "signatures on the highest-reach rated corridors: farthest-point " *
                           "initialization, then Lloyd sweeps whose assignment step is a " *
                           "HiGHS integer program minimizing demand-weighted signature " *
                           "distance subject to per-region demand-share bounds; regions " *
                           "relabelled by descending demand",
            "share_min" => PORTFOLIO_REGION_SHARE_MIN,
            "share_max" => PORTFOLIO_REGION_SHARE_MAX,
            "min_effective_regions" => PORTFOLIO_MIN_EFFECTIVE_REGIONS,
            "balance_iterations" => PORTFOLIO_BALANCE_ITERATIONS,
            "modes" => [PORTFOLIO_REGION_HIGH, PORTFOLIO_REGION_LOW],
            "probabilities" => fill(1 / PORTFOLIO_REGIONS, PORTFOLIO_REGIONS),
            "support_mean" => (PORTFOLIO_REGION_HIGH +
                               (PORTFOLIO_REGIONS - 1) * PORTFOLIO_REGION_LOW) / PORTFOLIO_REGIONS),
        "battery" => Dict{String,Any}(
            "system_power_share" => PORTFOLIO_POWER_SHARE,
            "weight_cap" => PORTFOLIO_WEIGHT_CAP,
            "duration_hours" => PORTFOLIO_DURATION_HOURS,
            "reserve_fraction" => PORTFOLIO_RESERVE_FRACTION,
            "initial_fraction" => PORTFOLIO_INITIAL_FRACTION,
            "charge_efficiency" => PORTFOLIO_CHARGE_EFFICIENCY,
            "discharge_efficiency" => PORTFOLIO_DISCHARGE_EFFICIENCY,
            "self_discharge" => PORTFOLIO_SELF_DISCHARGE,
            "throughput_cost" => PORTFOLIO_THROUGHPUT_COST),
        "calibration" => Dict{String,Any}(
            "bracket" => [PORTFOLIO_KAPPA_LO, PORTFOLIO_KAPPA_HI],
            "tolerance" => PORTFOLIO_KAPPA_TOL,
            "margin" => PORTFOLIO_KAPPA_MARGIN,
            "residual_tolerance" => PORTFOLIO_RESIDUAL_TOL,
            "recourse_tolerance" => PORTFOLIO_RECOURSE_TOL,
            "step" => PORTFOLIO_KAPPA_STEP,
            "gate_stages" => gate_stages(),
            "formulation" => "PowerModels.ACPPowerModel, unmodified network, no batteries"),
        "protocols" => Dict{String,Any}(
            "screening_scenarios" => PORTFOLIO_SCREENING_SCENARIOS,
            "final_scenarios" => PORTFOLIO_FINAL_SCENARIOS,
            "num_stages" => PORTFOLIO_HORIZON,
            "algorithm" => "stage-major StableRNG draw from the frozen support; the " *
                           "screening protocol is drawn from an independent seed and " *
                           "repaired against the final protocol's columns, so the two " *
                           "panels are disjoint by construction",
            "seed_rule" => "SHA256(\"$PORTFOLIO_SCHEMA\\nprotocol/<panel>\\n$PORTFOLIO_SEED\\n" *
                           "<network sha256>\\nseed\\n\") reduced into 1:2^31-1"),
        "versions" => portfolio_versions(),
        "command" => PORTFOLIO_COMMAND,
        "cases" => collect(cases),
    )
    manifest["digest"] = portfolio_digest(manifest)
    write_canonical_json(path, manifest)
    return manifest
end

"""
    portfolio_versions() -> Dict{String,Any}

The package identities the panel's bytes came from.

# Notes
Every version is recorded as a STRING. `_package_version` returns a
`VersionNumber`, which prints one way and serializes another; the manifest is
hashed through `canonical_json`, so anything it carries has to be a JSON scalar
and not a Julia type that happens to have a `show` method.
"""
function portfolio_versions()
    out = Dict{String,Any}("julia" => string(VERSION))
    for p in ("PGLib", "PowerModels", "SDDP", "JuMP", "Ipopt", "Clarabel", "StableRNGs")
        v = _package_version(p)
        out[p] = v === nothing ? "" : string(v)
    end
    return out
end

"""
    read_portfolio_manifest(path=portfolio_manifest_path()) -> Dict

Read a portfolio manifest and check its schema and its self-digest.
"""
function read_portfolio_manifest(path::AbstractString = portfolio_manifest_path())
    isfile(path) || error("no portfolio manifest at $path")
    m = JSON.parsefile(path)
    m["schema"] == PORTFOLIO_SCHEMA ||
        error("unexpected portfolio schema $(m["schema"]); expected $PORTFOLIO_SCHEMA")
    got = portfolio_digest(m)
    got == m["digest"] ||
        error("portfolio manifest digest $got does not match the recorded $(m["digest"])")
    return m
end

"""
    verify_portfolio_manifest(path=portfolio_manifest_path()) -> Dict

Check a portfolio manifest against everything it can be checked against WITHOUT
solving anything: its self-digest, the preregistered constants it was frozen
under, the panel size, and the internal consistency of each case record.

# Notes
This is the cheap check a reader runs first. It does not acquire a PGLib case and
does not touch a solver, so it cannot confirm that the recorded digests are the
digests of the artifacts the constants produce — that is what
[`materialize_portfolio_case`](@ref) does, one case at a time.
"""
function verify_portfolio_manifest(path::AbstractString = portfolio_manifest_path();
                                   min_cases::Integer = PORTFOLIO_MIN_CASES)
    m = read_portfolio_manifest(path)
    m["seed"] == PORTFOLIO_SEED || error("manifest seed $(m["seed"]) is not $PORTFOLIO_SEED")
    m["horizon"] == PORTFOLIO_HORIZON || error("manifest horizon is not $PORTFOLIO_HORIZON")
    Float64.(m["profile"]) == PORTFOLIO_PROFILE ||
        error("manifest profile differs from the preregistered common profile")
    m["regions"]["count"] == PORTFOLIO_REGIONS || error("manifest region count is not $PORTFOLIO_REGIONS")
    isapprox(Float64(m["regions"]["support_mean"]), 1.0; atol = 1e-15) ||
        error("the recorded atom mean is $(m["regions"]["support_mean"]), not 1")
    length(m["cases"]) >= min_cases ||
        error("the panel has $(length(m["cases"])) cases, fewer than $min_cases")
    for c in m["cases"]
        c["counts"]["battery"] == length(c["placement"]["buses"]) ||
            error("$(c["case"]): battery count disagrees with the placement")
        allunique(c["placement"]["buses"]) ||
            error("$(c["case"]): the placement repeats a bus")
        issorted(c["placement"]["buses"]) ||
            error("$(c["case"]): the placement is not sorted")
        expected = min(c["counts"]["eligible_bus"],
                       clamp(round(Int, 0.20 * c["counts"]["bus"]), 24, 240))
        c["counts"]["battery"] == expected ||
            error("$(c["case"]): $(c["counts"]["battery"]) batteries where the rule gives $expected")
        isapprox(Float64(c["kappa_case"]),
                 PORTFOLIO_KAPPA_MARGIN * Float64(c["kappa_max"]); rtol = 1e-12) ||
            error("$(c["case"]): κ_case is not $(PORTFOLIO_KAPPA_MARGIN) × κ_max")
        PORTFOLIO_KAPPA_LO <= Float64(c["kappa_max"]) <= PORTFOLIO_KAPPA_HI ||
            error("$(c["case"]): κ_max is outside the preregistered bracket")
        sum(Int.(c["regions"]["sizes"])) > 0 || error("$(c["case"]): empty regions")
        all(>(0), Int.(c["regions"]["sizes"])) ||
            error("$(c["case"]): a region contains no bus")
        isapprox(sum(Float64.(c["regions"]["demand_share"])), 1.0; atol = 1e-9) ||
            error("$(c["case"]): region demand shares do not sum to 1")
        # The balance gate, re-checked from the recorded shares themselves.
        share = Float64.(c["regions"]["demand_share"])
        exception = Bool(get(c["regions"], "single_bus_exception", false))
        all(s -> s >= PORTFOLIO_REGION_SHARE_MIN - 1e-9, share) ||
            error("$(c["case"]): a region share is below $PORTFOLIO_REGION_SHARE_MIN")
        over = count(s -> s > PORTFOLIO_REGION_SHARE_MAX + 1e-9, share)
        over == 0 || (exception && over == 1) ||
            error("$(c["case"]): $over regions exceed $PORTFOLIO_REGION_SHARE_MAX " *
                  "with single_bus_exception = $exception")
        n_eff = 1 / sum(abs2, share)
        isapprox(n_eff, Float64(c["regions"]["effective_regions"]); rtol = 1e-9) ||
            error("$(c["case"]): the recorded effective region count does not match its shares")
        n_eff >= PORTFOLIO_MIN_EFFECTIVE_REGIONS - 1e-9 || exception ||
            error("$(c["case"]): effective region count $n_eff is below " *
                  "$PORTFOLIO_MIN_EFFECTIVE_REGIONS")
        c["protocols"]["screening"]["excludes"] == "final" ||
            error("$(c["case"]): the screening protocol does not exclude the final one")
        c["protocols"]["final"]["sha256"] == c["protocols"]["screening"]["sha256"] &&
            error("$(c["case"]): the two protocols are identical")
    end
    return m
end

"""
    materialize_portfolio_case(name; dir, manifest=nothing, verify_all=false,
                               optimizer=nothing, quiet=false) -> BatteryCase

Regenerate one frozen case into `dir` and check it against the manifest.

# Notes
This is the reader-facing reproduction path and it is fail-closed at four points:
the acquired network must hash to the recorded digest, the recomputed regions and
placement must hash to the recorded digests, the frozen support must hash to the
recorded digest, and every written artifact must hash to what the manifest
records. `κ_case` is READ from the manifest rather than recalibrated, because a
reader should not have to spend hundreds of ACP solves to obtain a case — the
calibration is checked instead by `verify_all`, which is off by default and
re-runs the full `24 × 6` gate when it is on.
"""
function materialize_portfolio_case(name::AbstractString;
                                    dir::AbstractString,
                                    manifest = nothing,
                                    verify_all::Bool = false,
                                    optimizer = nothing,
                                    quiet::Bool = false)
    m = manifest === nothing ? read_portfolio_manifest() : manifest
    idx = findfirst(c -> String(c["case"]) == String(name), m["cases"])
    idx === nothing && error("$name is not in the frozen panel")
    rec = m["cases"][idx]

    src = acquire_pglib_case(name)
    sha = network_digest(src.network)
    sha == rec["network_sha256"] ||
        error("$name: the acquired network hashes to $sha, the manifest records $(rec["network_sha256"])")

    regions = portfolio_regions(src.network)
    regions.digest == rec["regions"]["digest"] ||
        error("$name: regenerated region digest $(regions.digest) does not match the manifest")
    buses, _ = portfolio_placement(src.network, sha)
    buses == Int.(rec["placement"]["buses"]) ||
        error("$name: the regenerated placement differs from the manifest")

    built = build_portfolio_case(name; dir = dir, kappa = Float64(rec["kappa_case"]),
                                 verify_all = verify_all, optimizer = optimizer,
                                 regions = regions, quiet = quiet)
    built.ok || error("$name: reconstruction failed")
    for (f, want) in rec["artifacts"]
        got = built.record["artifacts"][f]
        got == want ||
            error("$name: regenerated $f hashes to $got, the manifest records $want")
    end
    built.record["support"]["sha256"] == rec["support"]["sha256"] ||
        error("$name: the regenerated support digest does not match the manifest")
    quiet || @printf("  %s regenerated into %s and matched every recorded digest\n", name, dir)
    return built.case
end

"""
    freeze_portfolio(; dir, manifest_paths, quiet=false) -> Dict

Build the whole panel and write the manifest.

# Keywords
- `dir`: root the per-case artifact directories are written under.
- `manifest_paths`: every location the byte-identical manifest is written to —
  in practice both public example directories.

# Notes
Cases are attempted in the preregistered order. A case that fails the base-ACP
data gate is recorded as a replacement and the next RESERVE case is promoted, in
reserve order; a case that fails for any other reason stops the freeze, because
"this case broke and we moved on" is exactly the selection freedom the
preregistration exists to remove.
"""
function freeze_portfolio(; dir::AbstractString,
                          manifest_paths::AbstractVector{<:AbstractString},
                          quiet::Bool = false)
    records = Dict{String,Any}[]
    replacements = Dict{String,Any}[]
    reserve = copy(PORTFOLIO_RESERVE)
    for name in PORTFOLIO_PRIMARY
        candidate = name
        while true
            quiet || println("\n", candidate)
            built = try
                build_portfolio_case(candidate; dir = joinpath(dir, candidate), quiet = quiet)
            catch e
                e isa ErrorException && occursin("unavailable", string(e)) ?
                    (ok = false, record = nothing,
                     calibration = (reason = "unavailable from the pinned PGLib version",)) :
                    rethrow()
            end
            if built.ok
                push!(records, built.record)
                break
            end
            push!(replacements, Dict{String,Any}(
                "dropped" => candidate, "reason" => built.calibration.reason,
                "promoted" => isempty(reserve) ? "" : first(reserve)))
            isempty(reserve) && error("the reserve panel is exhausted; $candidate cannot be replaced")
            candidate = popfirst!(reserve)
        end
    end
    manifest = nothing
    for p in manifest_paths
        manifest = write_portfolio_manifest(p, records; replacements = replacements)
    end
    hashes = unique([bytes2hex(sha256(read(p))) for p in manifest_paths])
    length(hashes) == 1 ||
        error("the portfolio manifest was not written byte-identically to every location")
    return manifest
end

# ─────────────────────────────────────────────────────────────────────────────
# I — command line
# ─────────────────────────────────────────────────────────────────────────────

"""
    portfolio_main(args) -> Int

The public command. Returns a process exit code.
"""
function portfolio_main(args::AbstractVector{<:AbstractString})
    opt(flag, default = nothing) = begin
        i = findfirst(==(flag), args)
        i === nothing || i == length(args) ? default : args[i + 1]
    end
    has(flag) = flag in args
    # `--manifest` exists so a reader can point the command at a manifest that is
    # not the one shipped beside this file — the panel's own freeze does exactly
    # that, and so does the regression suite.
    mpath = opt("--manifest", portfolio_manifest_path())

    if has("--list")
        m = read_portfolio_manifest(mpath)
        @printf("%-28s %7s %8s %8s %10s  %s\n",
                "case", "bus", "battery", "κ_case", "regions", "network sha256")
        for c in m["cases"]
            @printf("%-28s %7d %8d %8.4f %10s  %s\n",
                    c["case"], c["counts"]["bus"], c["counts"]["battery"],
                    c["kappa_case"], join(c["regions"]["sizes"], "/"),
                    first(String(c["network_sha256"]), 16))
        end
        return 0
    end

    if has("--verify")
        # `--min-cases` exists for the same reason as `--manifest`: a manifest
        # that is not the frozen panel — a single-case one written by the
        # regression suite — is still a manifest and must be checkable.
        m = verify_portfolio_manifest(mpath;
                                      min_cases = parse(Int, opt("--min-cases",
                                                                 string(PORTFOLIO_MIN_CASES))))
        @printf("portfolio manifest verified: %d cases, digest %s\n",
                length(m["cases"]), first(String(m["digest"]), 16))
        return 0
    end

    if has("--freeze")
        out = opt("--out", joinpath(@__DIR__, "case"))
        exa = opt("--mirror", nothing)
        paths = [portfolio_manifest_path()]
        exa === nothing || push!(paths, portfolio_manifest_path(exa))
        m = freeze_portfolio(; dir = out, manifest_paths = paths)
        @printf("\nfrozen %d cases, manifest digest %s\n",
                length(m["cases"]), first(String(m["digest"]), 16))
        return 0
    end

    name = opt("--case", nothing)
    name === nothing && (println("usage: ", PORTFOLIO_COMMAND); return 2)
    out = opt("--out", joinpath(@__DIR__, "case"))
    case = materialize_portfolio_case(name; dir = joinpath(out, name),
                                      manifest = read_portfolio_manifest(mpath),
                                      verify_all = has("--verify-all"))
    print(describe(case))
    return 0
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && exit(portfolio_main(ARGS))
