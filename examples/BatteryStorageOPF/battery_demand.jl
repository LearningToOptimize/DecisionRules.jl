# battery_demand.jl
#
# The AUTHORING side of the demand process: a composable sampler abstraction,
# the deterministic temporal profile, and the operation that freezes either of
# them into the finite support both methods train from.
#
# THE TWO OBJECTS, AND WHY THEY ARE NOT THE SAME OBJECT.
#
#   authoring sampler   an arbitrary joint law over per-load multipliers. May be
#                       continuous, may be a user callable, may be built by
#                       composing several pieces. Lives here, never in a case.
#   frozen support      a finite, stage-major list of joint multiplier vectors
#                       with explicit probabilities. Lives in `demand.json`, is
#                       hashed, and is mirrored byte-identically into both
#                       engines (see `battery_case.jl`).
#
# SDDP enumerates a finite support in its backward pass; TS-DDR samples atom
# indices from a finite support in its trajectories. If each were allowed to
# discretize a continuous authoring law on its own, the two would train on two
# different stochastic programs while every report still said "the same demand
# process". `freeze_demand_support` is therefore not a convenience — it is the
# boundary that makes the comparison well posed, and it is the ONLY supported
# path from a sampler to a trainable case.
#
# This file is authoring-only and is NOT mirrored into the Exa engine: that
# engine reads frozen bytes and never touches a sampler.

using Distributions
using StableRNGs
using Statistics

@isdefined(BatterySpec) || include(joinpath(@__DIR__, "battery_case.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# Sampler metadata
# ─────────────────────────────────────────────────────────────────────────────

"""
    demand_meta(network) -> NamedTuple

The case description every sampler and every user callable receives.

# Returns
A `NamedTuple` with

- `load_ids::Vector{Int}` — LOAD identifiers, sorted ascending. This vector fixes
  the order of every multiplier vector in this file and in the frozen support.
- `load_bus::Vector{Int}` — the bus of each load, in `load_ids` order.
- `nominal_pd::Vector{Float64}`, `nominal_qd::Vector{Float64}` — the original
  PGLib load values ``p^{d,0}_i`` and ``q^{d,0}_i`` (pu).
- `status::Vector{Int}` — the in-service flag of each load.
- `num_loads::Int`, `baseMVA::Float64`.

# Notes
Sorting the identifiers once, here, is what guarantees that "component `j`"
means the same load to a sampler, to the frozen support and to both engines.
Nothing downstream is permitted to re-sort or re-index; a sampler that wants a
per-bus view builds it from `load_bus`.
"""
function demand_meta(network::AbstractDict)
    ids = sort!([Int(l["index"]) for (_, l) in network["load"]])
    by_id = Dict{Int,Any}(Int(l["index"]) => l for (_, l) in network["load"])
    return (load_ids = ids,
            load_bus = [Int(by_id[i]["load_bus"]) for i in ids],
            nominal_pd = [Float64(by_id[i]["pd"]) for i in ids],
            nominal_qd = [Float64(by_id[i]["qd"]) for i in ids],
            status = [Int(get(by_id[i], "status", 1)) for i in ids],
            num_loads = length(ids),
            baseMVA = Float64(network["baseMVA"]))
end

# ─────────────────────────────────────────────────────────────────────────────
# The sampler interface
# ─────────────────────────────────────────────────────────────────────────────

"""
    DemandSampler

An authoring law for the uncertain demand multiplier.

# Interface
A concrete sampler implements

- `sample_multiplier(s, rng, t, meta) -> Vector{Float64}` — ONE draw of the JOINT
  multiplier vector ``m_{\\cdot,t}``, of length `meta.num_loads`, in
  `meta.load_ids` order;
- `finite_support(s, t, meta) -> Union{Nothing,Tuple{Matrix{Float64},Vector{Float64}}}`
  — the sampler's EXACT finite support at stage `t`, or `nothing` when the law is
  continuous or otherwise not finitely supported;
- `describe_sampler(s) -> Dict{String,Any}` — a serializable description.

# Notes
The fundamental output is a JOINT VECTOR, never a scalar and never an implicit
collection of independent draws. A system-wide scalar law and a set of
independent per-load laws are both expressed as samplers that happen to produce
a particular kind of vector — [`SystemMultiplier`](@ref) and
[`IndependentMultiplier`](@ref) — so correlated regional structure
([`GroupMultiplier`](@ref)) and arbitrary user laws
([`CallableMultiplier`](@ref)) are first-class rather than special cases.

Samplers draw ONLY through the `rng` they are handed. Nothing here touches the
global random state, which is what makes a seeded freeze bit-reproducible in a
process that has done other random work.
"""
abstract type DemandSampler end

"Exact finite support of a sampler at stage `t`, or `nothing` when it has none."
finite_support(::DemandSampler, ::Integer, ::NamedTuple) = nothing

"""
    DeterministicMultiplier(value)

A degenerate sampler: the same multiplier vector at every stage, with
probability 1.

# Arguments
- `value`: a `Real` applied to every load, a `Vector{<:Real}` in `load_ids`
  order, or a `Dict{Int,<:Real}` keyed by LOAD identifier (missing loads take
  `1.0`).

# Notes
This is what makes "run the study with no demand uncertainty" expressible in the
same API rather than as a special code path, and it is the identity element of
[`ProductMultiplier`](@ref).
"""
struct DeterministicMultiplier <: DemandSampler
    value::Any
end

sample_multiplier(s::DeterministicMultiplier, ::Any, ::Integer, meta::NamedTuple) =
    _expand_per_load(s.value, meta)

finite_support(s::DeterministicMultiplier, ::Integer, meta::NamedTuple) =
    (reshape(_expand_per_load(s.value, meta), meta.num_loads, 1), [1.0])

describe_sampler(s::DeterministicMultiplier) = Dict{String,Any}(
    "sampler" => "deterministic",
    "value" => s.value isa Real ? Float64(s.value) :
               s.value isa AbstractDict ? Dict{String,Any}(string(k) => Float64(v) for (k, v) in s.value) :
               Float64.(collect(s.value)),
)

"""
    SystemMultiplier(dist)

One scalar draw per stage, applied to EVERY load.

# Arguments
- `dist`: a `Distributions.Distribution` used at every stage, or a
  `Vector{<:Distribution}` indexed by stage (stage-dependent law), or a
  `Dict{Int,<:Distribution}` keyed by stage.

# Notes
The system-wide case: demand moves up and down together. This is maximal
correlation across loads and is the sampler under which storage value is purely
temporal, with no locational component from the uncertainty itself.
"""
struct SystemMultiplier <: DemandSampler
    dist::Any
end

function sample_multiplier(s::SystemMultiplier, rng, t::Integer, meta::NamedTuple)
    ξ = rand(rng, _stage_dist(s.dist, t))
    return fill(Float64(ξ), meta.num_loads)
end

function finite_support(s::SystemMultiplier, t::Integer, meta::NamedTuple)
    d = _stage_dist(s.dist, t)
    d isa DiscreteNonParametric || return nothing
    vals = support(d)
    probs = Distributions.probs(d)
    A = Matrix{Float64}(undef, meta.num_loads, length(vals))
    for (k, v) in enumerate(vals)
        A[:, k] .= Float64(v)
    end
    return A, Float64.(collect(probs))
end

describe_sampler(s::SystemMultiplier) = Dict{String,Any}(
    "sampler" => "system",
    "distribution" => _describe_dist(s.dist),
)

"""
    IndependentMultiplier(dist)

Independent draws, one per load.

# Arguments
- `dist`: a `Distribution` used for every load, or a `Dict{Int,<:Distribution}`
  keyed by LOAD identifier (missing loads are deterministic 1).

# Notes
A convenience, not the general interface: independence is a particular joint law,
and it is the one under which aggregate demand concentrates as the case grows,
so on a large system it produces LESS system-level uncertainty than a
system-wide sampler with the same marginal. Choose it when locational demand
diversity is the object of study, not as a default.
"""
struct IndependentMultiplier <: DemandSampler
    dist::Any
end

function sample_multiplier(s::IndependentMultiplier, rng, ::Integer, meta::NamedTuple)
    out = Vector{Float64}(undef, meta.num_loads)
    for (j, id) in enumerate(meta.load_ids)
        d = s.dist isa AbstractDict ? get(s.dist, id, nothing) : s.dist
        out[j] = d === nothing ? 1.0 : Float64(rand(rng, d))
    end
    return out
end

describe_sampler(s::IndependentMultiplier) = Dict{String,Any}(
    "sampler" => "independent",
    "distribution" => _describe_dist(s.dist),
)

"""
    GroupMultiplier(groups, dists; by=:load)

One draw per GROUP, applied jointly to every load of that group.

# Arguments
- `groups::Vector{<:AbstractVector{Int}}`: the groups, given as LOAD identifiers
  (`by = :load`) or BUS identifiers (`by = :bus`).
- `dists`: one `Distribution` per group, or a single `Distribution` used by every
  group.

# Notes
The regional sampler, and the smallest construction that produces a genuinely
CORRELATED joint vector: loads inside a region move together, regions move
independently of one another. It is what makes a demand process able to stress
one part of a network while leaving another slack — the structure that gives a
battery a locational reason to exist.

Loads in no group take multiplier 1. Groups must be disjoint: overlapping groups
would make a load's multiplier depend on group order, which is exactly the kind
of silent ordering dependence this file exists to prevent.
"""
struct GroupMultiplier <: DemandSampler
    groups::Vector{Vector{Int}}
    dists::Any
    by::Symbol
end

function GroupMultiplier(groups, dists; by::Symbol = :load)
    by in (:load, :bus) || throw(ArgumentError("by must be :load or :bus, got :$by"))
    g = [sort!(collect(Int.(x))) for x in groups]
    seen = Set{Int}()
    for grp in g, id in grp
        id in seen && throw(ArgumentError("group member $id appears in more than one group"))
        push!(seen, id)
    end
    return GroupMultiplier(g, dists, by)
end

function sample_multiplier(s::GroupMultiplier, rng, ::Integer, meta::NamedTuple)
    out = fill(1.0, meta.num_loads)
    key = s.by === :load ? meta.load_ids : meta.load_bus
    for (gi, grp) in enumerate(s.groups)
        d = s.dists isa AbstractVector ? s.dists[gi] : s.dists
        ξ = Float64(rand(rng, d))
        members = Set(grp)
        for j in 1:meta.num_loads
            key[j] in members && (out[j] = ξ)
        end
    end
    return out
end

function finite_support(s::GroupMultiplier, ::Integer, meta::NamedTuple)
    # A product of finitely supported group laws has a finite support: the
    # Cartesian product of the groups' atoms with the product probabilities.
    dists = [s.dists isa AbstractVector ? s.dists[gi] : s.dists
             for gi in eachindex(s.groups)]
    all(d -> d isa DiscreteNonParametric, dists) || return nothing
    key = s.by === :load ? meta.load_ids : meta.load_bus
    members = [Set(grp) for grp in s.groups]
    vals = [Float64.(collect(support(d))) for d in dists]
    ps = [Float64.(collect(Distributions.probs(d))) for d in dists]
    combos = Iterators.product(map(v -> 1:length(v), vals)...)
    cols = Vector{Vector{Float64}}()
    probs = Float64[]
    for c in combos
        m = fill(1.0, meta.num_loads)
        p = 1.0
        for (gi, ki) in enumerate(c)
            p *= ps[gi][ki]
            for j in 1:meta.num_loads
                key[j] in members[gi] && (m[j] = vals[gi][ki])
            end
        end
        push!(cols, m)
        push!(probs, p)
    end
    A = Matrix{Float64}(undef, meta.num_loads, length(cols))
    for (k, col) in enumerate(cols)
        A[:, k] .= col
    end
    return A, probs
end

describe_sampler(s::GroupMultiplier) = Dict{String,Any}(
    "sampler" => "group",
    "by" => String(s.by),
    "groups" => [copy(g) for g in s.groups],
    "distribution" => _describe_dist(s.dists),
)

"""
    JointRegionMultiplier(groups, modes, probabilities; by=:load)

Regions that move JOINTLY: a finite set of joint outcomes over regions, each
with its own probability.

# Arguments
- `groups`: `Vector{Vector{Int}}`, the regions, as load ids (`by=:load`) or bus
  ids (`by=:bus`). Must be disjoint; a load in no region takes multiplier 1.
- `modes`: `(num_regions, K)` matrix, or a `Vector` of `K` length-`num_regions`
  vectors. Column `k` is one joint outcome: what EVERY region does together.
- `probabilities`: length `K`, positive, summing to 1.

# Notes
[`GroupMultiplier`](@ref) draws each region independently, so its joint law is
forced to be a product and its regional correlation is always zero. That is the
wrong shape for a locational storage study: the interesting demand processes are
the ones where regions are ANTI-correlated in some outcomes (one region is
stressed while another is slack — the case for putting storage in a specific
place) and correlated in others (system-wide stress, which no amount of
locational cleverness can hedge).

Enumerating joint outcomes rather than per-region laws expresses both, exactly,
with no copula and no approximation. It is also CHEAPER: a product of `R`
regions with two atoms each has `2^R` atoms, while the joint form spends atoms
only on the outcomes that carry meaning. Since the number of atoms per stage
multiplies the cost of every SDDP backward pass, that is the difference between
a converged baseline and an unaffordable one.

Combined with [`StageMultiplier`](@ref) — a different `JointRegionMultiplier`
per stage — the mean, the tail and the inter-region correlation all become
functions of time, which is what forces a policy to hold a genuinely
multi-dimensional strategy rather than one storage schedule replicated
everywhere.
"""
struct JointRegionMultiplier <: DemandSampler
    groups::Vector{Vector{Int}}
    modes::Matrix{Float64}
    probabilities::Vector{Float64}
    by::Symbol
end

function JointRegionMultiplier(groups, modes, probabilities; by::Symbol = :load)
    by in (:load, :bus) || throw(ArgumentError("by must be :load or :bus, got :$by"))
    g = [sort!(collect(Int.(x))) for x in groups]
    seen = Set{Int}()
    for grp in g, id in grp
        id in seen && throw(ArgumentError("region member $id appears in more than one region"))
        push!(seen, id)
    end
    M = modes isa AbstractMatrix ? Float64.(Matrix(modes)) :
        reduce(hcat, [Float64.(collect(m)) for m in modes])
    size(M, 1) == length(g) ||
        throw(ArgumentError("modes have $(size(M, 1)) rows for $(length(g)) regions"))
    p = Float64.(collect(probabilities))
    size(M, 2) == length(p) ||
        throw(ArgumentError("$(size(M, 2)) modes but $(length(p)) probabilities"))
    isapprox(sum(p), 1.0; atol = 1e-12) ||
        throw(ArgumentError("mode probabilities sum to $(sum(p)), not 1"))
    all(>(0), p) || throw(ArgumentError("mode probabilities must be positive"))
    all(>(0), M) || throw(ArgumentError("mode multipliers must be positive"))
    return JointRegionMultiplier(g, M, p, by)
end

"The `(num_loads, K)` multiplier matrix this sampler's modes induce."
function _region_atoms(s::JointRegionMultiplier, meta::NamedTuple)
    key = s.by === :load ? meta.load_ids : meta.load_bus
    members = [Set(grp) for grp in s.groups]
    A = fill(1.0, meta.num_loads, size(s.modes, 2))
    for k in 1:size(s.modes, 2), (gi, mem) in enumerate(members)
        for j in 1:meta.num_loads
            key[j] in mem && (A[j, k] = s.modes[gi, k])
        end
    end
    return A
end

finite_support(s::JointRegionMultiplier, ::Integer, meta::NamedTuple) =
    (_region_atoms(s, meta), copy(s.probabilities))

sample_multiplier(s::JointRegionMultiplier, rng, ::Integer, meta::NamedTuple) =
    _region_atoms(s, meta)[:, _sample_index(rng, s.probabilities)]

describe_sampler(s::JointRegionMultiplier) = Dict{String,Any}(
    "sampler" => "joint_region",
    "by" => String(s.by),
    "groups" => [copy(g) for g in s.groups],
    "modes" => [Float64.(s.modes[:, k]) for k in 1:size(s.modes, 2)],
    "probabilities" => copy(s.probabilities),
)

"""
    CallableMultiplier(f; name="callable", support=nothing)

An arbitrary user law: `f(rng, t, meta) -> Vector{Float64}`.

# Arguments
- `f`: the callable. It receives the RNG (and must draw from nothing else), the
  ABSOLUTE stage index, and the [`demand_meta`](@ref) named tuple.

# Keywords
- `name::AbstractString`: what the manifest records in place of a rule it cannot
  serialize.
- `support`: an optional `(t, meta) -> (atoms, probs)` callable declaring an
  exact finite support, when the user law happens to have one.

# Notes
This is the general interface, and every other sampler in this file is a
convenience over it. The returned vector is validated on every draw — length,
finiteness, nonnegativity — because a user callable is the one place a silently
wrong dimension can enter.
"""
struct CallableMultiplier <: DemandSampler
    f::Any
    name::String
    support::Any
end

CallableMultiplier(f; name::AbstractString = "callable", support = nothing) =
    CallableMultiplier(f, String(name), support)

function sample_multiplier(s::CallableMultiplier, rng, t::Integer, meta::NamedTuple)
    v = Float64.(collect(s.f(rng, t, meta)))
    length(v) == meta.num_loads ||
        error("callable sampler \"$(s.name)\" returned $(length(v)) multipliers at stage $t, expected $(meta.num_loads)")
    return v
end

finite_support(s::CallableMultiplier, t::Integer, meta::NamedTuple) =
    s.support === nothing ? nothing : s.support(t, meta)

describe_sampler(s::CallableMultiplier) = Dict{String,Any}(
    "sampler" => "callable", "name" => s.name,
)

"""
    FiniteMultiplier(atoms, probabilities)

An explicitly enumerated finite law, the same at every stage.

# Arguments
- `atoms`: `Matrix{Float64}` of size `(num_loads, K)` whose columns are the joint
  multiplier vectors, or a `Vector{<:Real}` of `K` system-wide scalars.
- `probabilities`: `Vector{Float64}` of length `K`, summing to 1.

# Notes
Freezing this sampler PRESERVES its support and probabilities exactly: no
resampling, no reweighting, no merging beyond exact duplicates. That is the
contract for a user who has already decided what the scenarios are.
"""
struct FiniteMultiplier <: DemandSampler
    atoms::Any
    probabilities::Vector{Float64}
end

function FiniteMultiplier(atoms, probabilities)
    p = Float64.(collect(probabilities))
    isapprox(sum(p), 1.0; atol = 1e-12) ||
        throw(ArgumentError("finite-support probabilities sum to $(sum(p)), not 1"))
    all(>(0), p) || throw(ArgumentError("finite-support probabilities must be positive"))
    return FiniteMultiplier(atoms, p)
end

function finite_support(s::FiniteMultiplier, ::Integer, meta::NamedTuple)
    A = if s.atoms isa AbstractMatrix
        Float64.(Matrix(s.atoms))
    else
        # A vector of scalars is a system-wide support: broadcast each atom
        # across every load.
        v = Float64.(collect(s.atoms))
        [v[k] for _ in 1:meta.num_loads, k in 1:length(v)]
    end
    size(A, 1) == meta.num_loads ||
        error("FiniteMultiplier atoms have $(size(A, 1)) rows, expected $(meta.num_loads)")
    size(A, 2) == length(s.probabilities) ||
        error("FiniteMultiplier has $(size(A, 2)) atoms but $(length(s.probabilities)) probabilities")
    return A, copy(s.probabilities)
end

function sample_multiplier(s::FiniteMultiplier, rng, t::Integer, meta::NamedTuple)
    A, p = finite_support(s, t, meta)
    return A[:, _sample_index(rng, p)]
end

describe_sampler(s::FiniteMultiplier) = Dict{String,Any}(
    "sampler" => "finite",
    "num_atoms" => length(s.probabilities),
    "probabilities" => copy(s.probabilities),
    "atoms" => s.atoms isa AbstractMatrix ?
               [Float64.(s.atoms[:, k]) for k in 1:size(s.atoms, 2)] :
               Float64.(collect(s.atoms)),
)

"""
    StageMultiplier(by_stage; default=nothing)

A stage-dependent law: `by_stage[t]` (or `by_stage` keyed by stage) is the
sampler used at stage `t`.

# Arguments
- `by_stage`: `Vector{<:DemandSampler}` indexed by stage, or
  `Dict{Int,<:DemandSampler}` keyed by stage.

# Keywords
- `default`: the sampler used at stages `by_stage` does not cover. `nothing`
  makes an uncovered stage an error.

# Notes
This is how "quiet early stages, uncertain stressed late stages" is expressed —
the structure the study needs in order for stored energy to have a hedging value
rather than only an arbitrage value. Each stage keeps its own exact support when
its sampler has one, so a stage-dependent finite support survives freezing
unchanged.
"""
struct StageMultiplier <: DemandSampler
    by_stage::Any
    default::Any
end

StageMultiplier(by_stage; default = nothing) = StageMultiplier(by_stage, default)

function _stage_sampler(s::StageMultiplier, t::Integer)
    inner = s.by_stage isa AbstractDict ? get(s.by_stage, Int(t), nothing) :
            (1 <= t <= length(s.by_stage) ? s.by_stage[t] : nothing)
    inner === nothing && (inner = s.default)
    inner === nothing && error("StageMultiplier has no sampler for stage $t and no default")
    return inner
end

sample_multiplier(s::StageMultiplier, rng, t::Integer, meta::NamedTuple) =
    sample_multiplier(_stage_sampler(s, t), rng, t, meta)

finite_support(s::StageMultiplier, t::Integer, meta::NamedTuple) =
    finite_support(_stage_sampler(s, t), t, meta)

function describe_sampler(s::StageMultiplier)
    entries = if s.by_stage isa AbstractDict
        Dict{String,Any}(string(k) => describe_sampler(v) for (k, v) in s.by_stage)
    else
        Dict{String,Any}(string(t) => describe_sampler(s.by_stage[t])
                         for t in eachindex(s.by_stage))
    end
    return Dict{String,Any}("sampler" => "stage", "stages" => entries,
                            "default" => s.default === nothing ? nothing :
                                         describe_sampler(s.default))
end

"""
    ProductMultiplier(components...)

The composition operator: the element-wise PRODUCT of several samplers' joint
vectors.

# Notes
Composition is multiplicative because the multiplier is multiplicative: a
system-wide factor times a regional factor is a demand that is `(system ×
region)` times nominal. This is what lets a case be authored as "a common
economy-wide level, plus a regional weather effect, plus an idiosyncratic
per-load term" without any of the three needing to know about the others.

Components are drawn from the SAME rng in order, so the composite is
reproducible from one seed. When EVERY component has an exact finite support the
product's support is their Cartesian product with product probabilities;
otherwise the product is treated as continuous and freezing discretizes it.
"""
struct ProductMultiplier <: DemandSampler
    components::Vector{DemandSampler}
end

ProductMultiplier(components::DemandSampler...) = ProductMultiplier(collect(components))

function sample_multiplier(s::ProductMultiplier, rng, t::Integer, meta::NamedTuple)
    out = fill(1.0, meta.num_loads)
    for c in s.components
        out .*= sample_multiplier(c, rng, t, meta)
    end
    return out
end

function finite_support(s::ProductMultiplier, t::Integer, meta::NamedTuple)
    parts = [finite_support(c, t, meta) for c in s.components]
    any(isnothing, parts) && return nothing
    A = parts[1][1]
    p = copy(parts[1][2])
    for q in parts[2:end]
        B, pb = q
        A2 = Matrix{Float64}(undef, meta.num_loads, size(A, 2) * size(B, 2))
        p2 = Vector{Float64}(undef, size(A, 2) * size(B, 2))
        col = 0
        for ka in 1:size(A, 2), kb in 1:size(B, 2)
            col += 1
            @views A2[:, col] .= A[:, ka] .* B[:, kb]
            p2[col] = p[ka] * pb[kb]
        end
        A = A2
        p = p2
    end
    return A, p
end

describe_sampler(s::ProductMultiplier) = Dict{String,Any}(
    "sampler" => "product",
    "components" => [describe_sampler(c) for c in s.components],
)

# ── Small shared helpers ─────────────────────────────────────────────────────

"Expand a scalar / vector / load-keyed dictionary into a per-load vector."
function _expand_per_load(value, meta::NamedTuple)
    if value isa Real
        return fill(Float64(value), meta.num_loads)
    elseif value isa AbstractDict
        return [Float64(get(value, id, 1.0)) for id in meta.load_ids]
    else
        v = Float64.(collect(value))
        length(v) == meta.num_loads ||
            error("per-load value has $(length(v)) entries, expected $(meta.num_loads)")
        return v
    end
end

"Select the distribution governing stage `t` from a scalar / vector / dictionary."
function _stage_dist(dist, t::Integer)
    dist isa AbstractDict && return dist[Int(t)]
    dist isa AbstractVector && return dist[t]
    return dist
end

"Draw an index from an explicit probability vector using only `rand(rng)`."
function _sample_index(rng, p::AbstractVector{Float64})
    u = rand(rng)
    acc = 0.0
    for i in eachindex(p)
        acc += p[i]
        u <= acc && return i
    end
    return length(p)
end

"A serializable description of a distribution, a vector of them or a dictionary."
function _describe_dist(d)
    d isa AbstractDict && return Dict{String,Any}(string(k) => _describe_dist(v) for (k, v) in d)
    d isa AbstractVector && return [_describe_dist(x) for x in d]
    return string(d)
end

# ─────────────────────────────────────────────────────────────────────────────
# Deterministic temporal profile
# ─────────────────────────────────────────────────────────────────────────────

"""
    profile_matrix(profile, meta, horizon) -> Matrix{Float64}

Materialize the deterministic temporal profile ``h_{i,t}`` as a
`(num_loads × horizon)` matrix.

# Arguments
- `profile`: one of
  - a `Real` — flat, the same at every load and stage;
  - a `Vector{<:Real}` of length `horizon` — one value per stage, shared by every
    load;
  - a `Vector{<:Real}` of a SHORTER length `P` — a cyclic profile of period `P`,
    expanded as `profile[mod1(t, P)]`;
  - a `Matrix{<:Real}` of size `(num_loads, horizon)` — the fully general case;
  - a callable `(load_id, t) -> Real`.

# Notes
The profile is materialized ONCE, here, and frozen with the case. A cyclic
profile is expanded rather than stored as a period, because a horizon change
must not be able to silently reinterpret which stage is the peak.
"""
function profile_matrix(profile, meta::NamedTuple, horizon::Integer)
    n = meta.num_loads
    H = Int(horizon)
    out = Matrix{Float64}(undef, n, H)
    if profile isa Real
        fill!(out, Float64(profile))
    elseif profile isa AbstractMatrix
        size(profile) == (n, H) ||
            error("profile matrix must be $(n)×$H, got $(size(profile))")
        out .= Float64.(profile)
    elseif profile isa AbstractVector
        v = Float64.(collect(profile))
        P = length(v)
        P >= 1 || error("profile vector must be non-empty")
        for t in 1:H
            out[:, t] .= v[mod1(t, P)]
        end
    elseif profile isa Function
        for t in 1:H, j in 1:n
            out[j, t] = Float64(profile(meta.load_ids[j], t))
        end
    else
        error("unsupported profile of type $(typeof(profile))")
    end
    all(isfinite, out) || error("profile has a non-finite entry")
    all(>=(0), out) || error("profile has a negative entry")
    return out
end

"""
    diurnal_profile(horizon; amplitude=0.12, peak_hour=19, period=24) -> Vector{Float64}

A smooth single-peak daily profile of mean 1,
``h_t = 1 + a\\cos\\!\\left(2\\pi (t - t_{peak})/P\\right)``.

# Notes
Returned as a length-`horizon` vector rather than one period, so the caller can
see exactly which stage is the peak. Mean 1 means the profile describes the
SHAPE of demand and the sampler describes its LEVEL; keeping the two
separable is what lets a stressed late window be authored by changing the
sampler alone.
"""
function diurnal_profile(horizon::Integer; amplitude::Real = 0.12,
                         peak_hour::Integer = 19, period::Integer = 24)
    return [1.0 + amplitude * cos(2π * (t - peak_hour) / period) for t in 1:horizon]
end

# ─────────────────────────────────────────────────────────────────────────────
# Sampling and validation
# ─────────────────────────────────────────────────────────────────────────────

"""
    sample_multiplier_path(sampler, meta, horizon; seed) -> Matrix{Float64}

Draw one COMPLETE multiplier path: a `(num_loads × horizon)` matrix whose column
`t` is ``m_{\\cdot,t}``.

# Notes
Uses a `StableRNG(seed)` and nothing else, so the same seed reproduces the same
path on any platform and any Julia version. Every drawn vector is validated
before it is returned.
"""
function sample_multiplier_path(sampler::DemandSampler, meta::NamedTuple, horizon::Integer;
                                seed::Integer)
    rng = StableRNG(seed)
    out = Matrix{Float64}(undef, meta.num_loads, Int(horizon))
    for t in 1:Int(horizon)
        v = sample_multiplier(sampler, rng, t, meta)
        _check_multiplier(v, meta, t)
        out[:, t] .= v
    end
    return out
end

"""
    materialize_demand_path(network, profile, multipliers) -> (pd, qd)

Turn a multiplier path into the realized per-LOAD demand it describes.

# Arguments
- `profile::AbstractMatrix`, `multipliers::AbstractMatrix`: both
  `(num_loads × horizon)`.

# Returns
- `pd`, `qd`: `(num_loads × horizon)` matrices in pu.

# Notes
The same total multiplier scales the active and the reactive value of each load,
so `qd[j,t]/pd[j,t]` equals the load's nominal ratio at every stage of every
path. That invariant is asserted by [`validate_sampler`](@ref); it is the
formal content of "the uncertainty moves how much power is consumed, never what
kind".
"""
function materialize_demand_path(meta::NamedTuple, profile::AbstractMatrix,
                                 multipliers::AbstractMatrix)
    size(profile) == size(multipliers) ||
        error("profile $(size(profile)) and multipliers $(size(multipliers)) disagree")
    total = profile .* multipliers
    return meta.nominal_pd .* total, meta.nominal_qd .* total
end

"Fail closed on a multiplier vector a sampler produced."
function _check_multiplier(v::AbstractVector, meta::NamedTuple, t::Integer)
    length(v) == meta.num_loads ||
        error("sampler returned $(length(v)) multipliers at stage $t, expected $(meta.num_loads)")
    all(isfinite, v) || error("sampler returned a non-finite multiplier at stage $t")
    all(>=(0), v) || error("sampler returned a negative multiplier at stage $t")
    return nothing
end

"""
    validate_sampler(sampler, network, horizon; seed=1, draws=32) -> NamedTuple

Exercise a sampler and check every property the study relies on.

# Checks
1. **dimension and order** — every draw has one entry per load, in `load_ids`
   order;
2. **finiteness and nonnegativity** — no `NaN`, no `Inf`, no negative demand;
3. **exact seeded reproduction** — two runs from the same seed produce
   bit-identical paths;
4. **no hidden global RNG state** — perturbing the GLOBAL random stream between
   two seeded runs does not change the result;
5. **power-factor preservation** — the realized `qd/pd` ratio of every load
   equals its nominal ratio at every stage;
6. **finite support consistency** — where the sampler declares one, its
   dimensions and probabilities are valid;
7. **load order preservation** — the metadata's identifiers are sorted and
   unique, and no draw reorders them.

# Returns
A `NamedTuple` of the observed multiplier range and, when declared, the per-stage
support sizes. Failures are errors, not warnings.
"""
function validate_sampler(sampler::DemandSampler, network::AbstractDict, horizon::Integer;
                          seed::Integer = 1, draws::Integer = 32)
    meta = demand_meta(network)
    issorted(meta.load_ids) && allunique(meta.load_ids) ||
        error("demand_meta produced load identifiers that are not sorted and unique")

    a = sample_multiplier_path(sampler, meta, horizon; seed = seed)
    # Disturb the global stream: a sampler that reaches for it will now diverge.
    rand(1000)
    b = sample_multiplier_path(sampler, meta, horizon; seed = seed)
    a == b || error("sampler is not reproducible from its seed, or reaches for the global RNG")

    lo, hi = Inf, -Inf
    for d in 1:Int(draws)
        p = sample_multiplier_path(sampler, meta, horizon; seed = seed + d)
        lo = min(lo, minimum(p))
        hi = max(hi, maximum(p))
        pd, qd = materialize_demand_path(meta, profile_matrix(1.0, meta, horizon), p)
        for j in 1:meta.num_loads
            meta.nominal_pd[j] == 0 && continue
            nominal_ratio = meta.nominal_qd[j] / meta.nominal_pd[j]
            for t in 1:Int(horizon)
                pd[j, t] == 0 && continue
                isapprox(qd[j, t] / pd[j, t], nominal_ratio; rtol = 1e-12) ||
                    error("load $(meta.load_ids[j]) lost its power factor at stage $t")
            end
        end
    end

    sizes = Int[]
    for t in 1:Int(horizon)
        fs = finite_support(sampler, t, meta)
        fs === nothing && continue
        A, p = fs
        size(A, 1) == meta.num_loads ||
            error("declared support at stage $t has $(size(A, 1)) rows, expected $(meta.num_loads)")
        size(A, 2) == length(p) ||
            error("declared support at stage $t has $(size(A, 2)) atoms but $(length(p)) probabilities")
        isapprox(sum(p), 1.0; atol = 1e-12) ||
            error("declared support at stage $t has probabilities summing to $(sum(p))")
        all(>(0), p) || error("declared support at stage $t has a non-positive probability")
        push!(sizes, size(A, 2))
    end

    return (multiplier_min = lo, multiplier_max = hi,
            support_sizes = isempty(sizes) ? nothing : sizes,
            num_loads = meta.num_loads)
end

# ─────────────────────────────────────────────────────────────────────────────
# Freezing
# ─────────────────────────────────────────────────────────────────────────────

"""
    freeze_demand_support(sampler, network, horizon; seed, atoms_per_stage,
                          method=:auto, profile=1.0, protocol_seed=seed,
                          stage_hours=1.0, merge_duplicates=true)
        -> DemandSupport

Turn an authoring sampler into the FROZEN finite support both methods train from.

# Arguments
- `sampler::DemandSampler`: the authoring law.
- `network::AbstractDict`: the parsed PGLib network (supplies the load order).
- `horizon::Integer`: number of stages to freeze.

# Keywords
- `seed::Integer`: seed of the `StableRNG` used by an empirical discretization.
- `atoms_per_stage::Integer`: ``K_t`` for an empirical discretization. Ignored by
  an exact one.
- `method::Symbol`: `:auto` (exact where the sampler declares a support,
  empirical elsewhere), `:exact` (require a declared support at every stage) or
  `:empirical` (discretize even a declared support — for a deliberate
  sub-sampling of a large exact support).
- `profile`: anything [`profile_matrix`](@ref) accepts.
- `protocol_seed::Integer`: seed of the evaluation protocol.
- `stage_hours::Real`: ``\\Delta t``.
- `profile_period::Integer`: the cycle length the deterministic profile repeats
  on, recorded for [`profile_period`](@ref). It is a POLICY FEATURE only and
  enters no stage problem; the profile itself is materialized stage by stage and
  is not reconstructed from it.
- `merge_duplicates::Bool`: merge EXACTLY equal atoms within a stage, summing
  their probabilities.

# Returns
- A validated [`DemandSupport`](@ref), ready to be written into a case.

# Notes
**Exact preservation.** For a sampler that declares a finite support the atoms
and probabilities are carried through unchanged — no reweighting and no
resampling — because a user who enumerated their scenarios has already decided
what the stochastic program is.

**Empirical discretization.** For a continuous or general sampler the default is
the most transparent estimator there is: draw `atoms_per_stage` independent joint
vectors per stage from `StableRNG(seed)` and weight them equally. It converges to
the authoring law, it is reproducible from `(seed, atoms_per_stage)` alone, and
it makes no claim about matching moments. A user who wants a different
discretizer declares one by giving their sampler a `support` callable; there is
no hidden quadrature rule.

**Duplicate merging** is by EXACT equality of the multiplier vector, so it only
ever collapses atoms that are the same point of the support — typically a
discrete sampler drawn empirically. Two atoms that differ in the last bit are
two atoms.

The `source` record written into the support carries the sampler description,
the seed, the atom count, the method and the profile summary, so a frozen support
can always be traced back to the law it approximates.
"""
function freeze_demand_support(sampler::DemandSampler, network::AbstractDict,
                               horizon::Integer;
                               seed::Integer,
                               atoms_per_stage::Integer = 1,
                               method::Symbol = :auto,
                               profile = 1.0,
                               protocol_seed::Integer = seed,
                               stage_hours::Real = 1.0,
                               profile_period::Integer = 24,
                               merge_duplicates::Bool = true)
    method in (:auto, :exact, :empirical) ||
        throw(ArgumentError("method must be :auto, :exact or :empirical, got :$method"))
    meta = demand_meta(network)
    H = Int(horizon)
    H >= 1 || throw(ArgumentError("horizon must be at least 1"))
    prof = profile_matrix(profile, meta, H)

    rng = StableRNG(seed)
    atoms = Vector{Matrix{Float64}}(undef, H)
    probs = Vector{Vector{Float64}}(undef, H)
    exact_stages = Int[]

    for t in 1:H
        declared = method === :empirical ? nothing : finite_support(sampler, t, meta)
        if declared === nothing
            method === :exact &&
                error("method = :exact but the sampler declares no finite support at stage $t")
            atoms_per_stage >= 1 ||
                throw(ArgumentError("atoms_per_stage must be at least 1 for an empirical freeze"))
            A = Matrix{Float64}(undef, meta.num_loads, Int(atoms_per_stage))
            for k in 1:Int(atoms_per_stage)
                v = sample_multiplier(sampler, rng, t, meta)
                _check_multiplier(v, meta, t)
                A[:, k] .= v
            end
            p = fill(1.0 / Int(atoms_per_stage), Int(atoms_per_stage))
            atoms[t], probs[t] = A, p
        else
            A, p = declared
            A = Float64.(Matrix(A))
            p = Float64.(collect(p))
            size(A, 1) == meta.num_loads ||
                error("declared support at stage $t has $(size(A, 1)) rows, expected $(meta.num_loads)")
            size(A, 2) == length(p) ||
                error("declared support at stage $t has $(size(A, 2)) atoms but $(length(p)) probabilities")
            for k in 1:size(A, 2)
                _check_multiplier(view(A, :, k), meta, t)
            end
            atoms[t], probs[t] = A, p
            push!(exact_stages, t)
        end
        if merge_duplicates
            atoms[t], probs[t] = _merge_duplicate_atoms(atoms[t], probs[t])
        end
        # Renormalize only against accumulated floating-point error, never to
        # repair a support that does not sum to 1 in the first place.
        s = sum(probs[t])
        isapprox(s, 1.0; atol = 1e-9) ||
            error("stage $t probabilities sum to $s, not 1")
        probs[t] ./= s
    end

    source = Dict{String,Any}(
        "sampler" => describe_sampler(sampler),
        "method" => String(method),
        "seed" => Int(seed),
        "atoms_per_stage" => Int(atoms_per_stage),
        "exact_stages" => exact_stages,
        "merge_duplicates" => merge_duplicates,
        "profile_period" => Int(profile_period),
        "profile_min" => minimum(prof),
        "profile_max" => maximum(prof),
        "horizon" => H,
    )

    support = DemandSupport(Float64(stage_hours), H, copy(meta.load_ids), prof,
                            atoms, probs, Int(protocol_seed), source)
    validate_support(support)
    return support
end

"""
    _merge_duplicate_atoms(A, p) -> (A', p')

Collapse columns of `A` that are EXACTLY equal, summing their probabilities.

# Notes
First occurrence wins the position, so the order of the surviving atoms is the
order they were generated in — which keeps an empirically frozen support's bytes
a pure function of the seed.
"""
function _merge_duplicate_atoms(A::Matrix{Float64}, p::Vector{Float64})
    seen = Dict{Vector{Float64},Int}()
    cols = Vector{Vector{Float64}}()
    out = Float64[]
    for k in 1:size(A, 2)
        col = A[:, k]
        j = get(seen, col, 0)
        if j == 0
            push!(cols, col)
            push!(out, p[k])
            seen[col] = length(cols)
        else
            out[j] += p[k]
        end
    end
    B = Matrix{Float64}(undef, size(A, 1), length(cols))
    for (k, col) in enumerate(cols)
        B[:, k] .= col
    end
    return B, out
end

# ─────────────────────────────────────────────────────────────────────────────
# Serialization of an authoring sampler
# ─────────────────────────────────────────────────────────────────────────────

"""
    sampler_to_dict(sampler) -> Dict{String,Any}

Serialize a sampler to a plain dictionary (the same description the frozen
support records).
"""
sampler_to_dict(s::DemandSampler) = describe_sampler(s)

"""
    sampler_from_dict(d) -> DemandSampler

Reconstruct a sampler from [`sampler_to_dict`](@ref).

# Notes
Every built-in sampler round-trips. A [`CallableMultiplier`](@ref) cannot: a
Julia closure has no serialization, and inventing one that reconstructs "some
callable with the right name" would be worse than failing, because the
reconstructed object would silently be a different law. The frozen SUPPORT is
what carries a callable-authored case forward, which is exactly why the support
rather than the sampler is the thing that is frozen.
"""
function sampler_from_dict(d::AbstractDict)
    kind = String(d["sampler"])
    if kind == "deterministic"
        v = d["value"]
        return DeterministicMultiplier(v isa AbstractDict ?
                                       Dict(parse(Int, k) => Float64(x) for (k, x) in v) :
                                       v isa AbstractVector ? Float64.(v) : Float64(v))
    elseif kind == "finite"
        atoms = d["atoms"]
        A = atoms isa AbstractVector && !isempty(atoms) && atoms[1] isa AbstractVector ?
            reduce(hcat, [Float64.(a) for a in atoms]) : Float64.(collect(atoms))
        return FiniteMultiplier(A, Float64.(d["probabilities"]))
    elseif kind == "joint_region"
        # Round-trips exactly: unlike the group/system samplers this one carries
        # plain numbers, not a Distributions.jl object.
        return JointRegionMultiplier([Int.(g) for g in d["groups"]],
                                     [Float64.(m) for m in d["modes"]],
                                     Float64.(d["probabilities"]);
                                     by = Symbol(d["by"]))
    elseif kind == "product"
        return ProductMultiplier([sampler_from_dict(c) for c in d["components"]])
    elseif kind == "stage"
        stages = Dict{Int,DemandSampler}(parse(Int, k) => sampler_from_dict(v)
                                         for (k, v) in d["stages"])
        default = d["default"] === nothing ? nothing : sampler_from_dict(d["default"])
        return StageMultiplier(stages; default = default)
    elseif kind in ("system", "independent", "group")
        error("sampler_from_dict: \"$kind\" carries a Distributions.jl object, which is " *
              "described but not serialized; rebuild it in code, or use the frozen support")
    elseif kind == "callable"
        error("sampler_from_dict: a callable sampler cannot be reconstructed; " *
              "the frozen support is what carries such a case forward")
    else
        error("sampler_from_dict: unknown sampler kind \"$kind\"")
    end
end
