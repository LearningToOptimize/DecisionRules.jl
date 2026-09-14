# battery_case.jl
#
# Frozen case contract for the multistage battery-storage AC-OPF study.
#
# This file is the SINGLE source of truth for what "the case" is, and it is
# shipped BYTE-IDENTICALLY in both public engines (DecisionRules.jl, the
# JuMP/PowerModels/SDDP engine, and DecisionRulesExa.jl, the ExaModels/GPU
# engine). Neither engine may re-derive a battery parameter, a demand
# realization, or a scenario index on its own: both read them from here, so a
# disagreement between the engines can never be a disagreement about the case.
#
# It therefore depends only on JSON, SHA, StableRNGs and the Julia standard
# library. In particular it does NOT depend on PowerModels, PGLib or
# Distributions: those are needed to BUILD the frozen artifacts (see
# `battery_demand.jl` and `build_battery_case.jl`, which
# exist only in the JuMP engine), never to READ them.
#
# Artifacts of one case live in one directory:
#
#   <case_dir>/network.json        the parsed PGLib network, per-unit, verbatim
#   <case_dir>/batteries.json      battery placement and parameters
#   <case_dir>/demand.json         the FROZEN finite demand support
#   <case_dir>/case_manifest.json  units, counts, stage duration and SHA-256s
#
# Every number that both engines must agree on is in one of those four files.
#
# THE DEMAND CONTRACT, stated once.
# The authoring sampler (a `Distribution`, a callable, a regional group model —
# see `battery_demand.jl`) is NOT part of the case. What is frozen, hashed and
# mirrored is its FINITE SUPPORT: for every stage t, a list of joint multiplier
# vectors over the case's loads together with their probabilities. Both SDDP and
# TS-DDR train from those bytes and neither is permitted to resample or
# rediscretize the authoring sampler. That is what makes "the two methods faced
# the same stochastic program" a checkable statement rather than an intention.

using JSON
using SHA
using StableRNGs
using Printf

# ─────────────────────────────────────────────────────────────────────────────
# Schema tags
#
# Every artifact carries a schema string. A loader that meets an unknown schema
# FAILS rather than guessing, because a silently-shifted field is exactly the
# class of defect that makes two engines solve two different problems.
#
# The demand artifact is at schema 2: schema 1 carried a single scalar
# multiplier per stage, which cannot express a joint per-load realization.
# ─────────────────────────────────────────────────────────────────────────────

const BATTERY_NETWORK_SCHEMA  = "battery_storage_opf/network/1"
const BATTERY_BATTERY_SCHEMA  = "battery_storage_opf/batteries/2"
const BATTERY_DEMAND_SCHEMA   = "battery_storage_opf/demand/2"
const BATTERY_MANIFEST_SCHEMA = "battery_storage_opf/manifest/3"

"""
    STAGE_AVAILABILITY_KEY

Name of the optional generator field that carries a per-stage availability
schedule: `gen[STAGE_AVAILABILITY_KEY][t]` is a nonnegative multiplier applied to
that generator's active and reactive limits at stage `t` by
[`apply_stage_availability!`](@ref).

# Notes
The schedule lives INSIDE the network table rather than beside it in an artifact
of its own, for one reason: the frozen case hashes `network.json`, so a schedule
carried there is covered by the case digest, travels with the case to every
consumer, and cannot drift out of step with the network it describes. A separate
artifact would have needed its own hash, its own read-back check and its own
statement of which generator each row refers to.

The network schema is unchanged because the field is OPTIONAL and additive: a
generator that does not carry it is available in every stage, so every case built
before this convention existed still means exactly what it meant then.
"""
const STAGE_AVAILABILITY_KEY = "stage_availability"

# ─────────────────────────────────────────────────────────────────────────────
# Canonical JSON
#
# `JSON.print` iterates a `Dict` in hash order, so writing the same object twice
# from two processes can produce two different byte strings and destroy the
# point of hashing an artifact. The emitter below sorts object keys and prints
# every scalar through a round-tripping representation, which makes the bytes a
# pure function of the value.
# ─────────────────────────────────────────────────────────────────────────────

"""
    canonical_json(value) -> String

Serialize `value` to JSON whose bytes depend only on the value, not on
dictionary iteration order or on floating-point printing defaults.

# Arguments
- `value`: any nesting of `AbstractDict{<:AbstractString}`, `AbstractVector`,
  `AbstractString`, `Bool`, `Integer`, `AbstractFloat` and `nothing`.

# Returns
- A `String` holding the canonical JSON text (2-space indentation, object keys
  sorted lexicographically by `isless` on the key strings).

# Notes
Floats are printed with `Base.print`, which emits the shortest decimal literal
that round-trips through `parse(Float64, ·)`. Reading the emitted text back
with `JSON.parsefile` therefore reproduces the original `Float64` bit pattern,
which is what lets the two engines hash and compare the same artifact.

Non-finite floats are rejected: JSON has no representation for them, and a
silently emitted `NaN` token would be unparseable by a conforming reader.
"""
function canonical_json(value)
    io = IOBuffer()
    _canonical_json!(io, value, 0)
    return String(take!(io))
end

# Recursive canonical writer. `depth` is the current indentation level; the
# emitter never depends on the container's iteration order.
function _canonical_json!(io::IO, value, depth::Int)
    pad     = "  "^depth
    pad_in  = "  "^(depth + 1)
    if value === nothing
        print(io, "null")
    elseif value isa Bool
        # Checked before Integer: `Bool <: Integer` in Julia.
        print(io, value ? "true" : "false")
    elseif value isa Integer
        print(io, string(value))
    elseif value isa AbstractFloat
        isfinite(value) || error("canonical_json: non-finite float $value has no JSON representation")
        # Shortest round-tripping decimal; `1.0` stays `1.0` (never `1`), which
        # keeps the emitted type distinguishable from an integer on re-read.
        print(io, string(Float64(value)))
    elseif value isa AbstractString
        _canonical_json_string!(io, value)
    elseif value isa AbstractDict
        isempty(value) && return print(io, "{}")
        # Stringify the keys once, then sort: sorting is what makes the bytes
        # order-independent, and going through the pair list avoids re-indexing
        # the source dictionary with a converted key.
        pairs = sort!([(string(k), v) for (k, v) in value]; by = first)
        allunique(first.(pairs)) ||
            error("canonical_json: dictionary has keys that collide once stringified")
        print(io, "{\n")
        for (i, (k, v)) in enumerate(pairs)
            print(io, pad_in)
            _canonical_json_string!(io, k)
            print(io, ": ")
            _canonical_json!(io, v, depth + 1)
            print(io, i == length(pairs) ? "\n" : ",\n")
        end
        print(io, pad, "}")
    elseif value isa AbstractVector
        isempty(value) && return print(io, "[]")
        print(io, "[\n")
        for (i, v) in enumerate(value)
            print(io, pad_in)
            _canonical_json!(io, v, depth + 1)
            print(io, i == length(value) ? "\n" : ",\n")
        end
        print(io, pad, "]")
    else
        error("canonical_json: unsupported value of type $(typeof(value))")
    end
    return nothing
end

# Minimal RFC 8259 string escaping.
function _canonical_json_string!(io::IO, s::AbstractString)
    print(io, '"')
    for c in s
        if c == '"'
            print(io, "\\\"")
        elseif c == '\\'
            print(io, "\\\\")
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\r'
            print(io, "\\r")
        elseif c == '\t'
            print(io, "\\t")
        elseif c < ' '
            print(io, "\\u", lpad(string(UInt16(c); base = 16), 4, '0'))
        else
            print(io, c)
        end
    end
    print(io, '"')
    return nothing
end

"""
    write_canonical_json(path, value) -> String

Write `canonical_json(value)` to `path` and return the SHA-256 of the bytes
actually written.

# Arguments
- `path::AbstractString`: destination file.
- `value`: object accepted by [`canonical_json`](@ref).

# Returns
- Lowercase hexadecimal SHA-256 digest of the file contents.
"""
function write_canonical_json(path::AbstractString, value)
    text = canonical_json(value)
    mkpath(dirname(path))
    write(path, text)
    return bytes2hex(sha256(text))
end

"""
    sha256_file(path) -> String

Lowercase hexadecimal SHA-256 digest of the bytes of `path`.
"""
sha256_file(path::AbstractString) = bytes2hex(sha256(read(path)))

"""
    plain(value)

Recursively rebuild a parsed-JSON tree out of plain `Dict{String,Any}`,
`Vector{Any}` and scalars.

# Notes
JSON parsers return their own container types (`JSON.Object`, lazily-typed
arrays). Those satisfy the `AbstractDict`/`AbstractVector` interfaces but not the
CONCRETE types that downstream modelling packages assume when they build typed
lookup tables from a network dictionary, which surfaces as a `convert` error
deep inside a library rather than as a data problem. Normalizing once, at the
boundary where the case is read, keeps that class of failure out of every
consumer.
"""
function plain(value)
    if value isa AbstractDict
        return Dict{String,Any}(string(k) => plain(v) for (k, v) in value)
    elseif value isa AbstractString
        return String(value)
    elseif value isa AbstractVector
        return Any[plain(v) for v in value]
    else
        return value
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Battery parameters
# ─────────────────────────────────────────────────────────────────────────────

"""
    BatterySpec

Parameters of one battery, in the per-unit system of the host network.

# Fields
- `index::Int`: battery identifier. Identifiers are arbitrary positive integers
  and need not be consecutive; every engine keys on this value.
- `bus::Int`: identifier of the bus the battery injects into. Again an
  arbitrary network identifier, never a positional index.
- `energy_min::Float64`, `energy_max::Float64`: energy bounds ``\\underline e_b``
  and ``\\overline e_b`` in per-unit-hours (pu·h), i.e. per-unit power sustained
  for one hour.
- `energy_initial::Float64`: ``e_{b,0}``, the energy carried into stage 1 (pu·h).
- `charge_max::Float64`, `discharge_max::Float64`: ``\\overline p^{ch}_b`` and
  ``\\overline p^{dis}_b`` in per-unit power (pu).
- `charge_efficiency::Float64`, `discharge_efficiency::Float64`:
  ``\\eta^{ch}_b, \\eta^{dis}_b \\in (0,1]``.
- `self_discharge::Float64`: ``\\alpha_b \\in (0,1]``, the fraction of stored
  energy retained across one stage.
- `throughput_cost::Float64`: ``c^{deg}_b``, the degradation price charged on
  ``\\Delta t\\,(p^{ch}+p^{dis})``, in objective units per pu·h.

# Notes
The state transition these fields parameterize is

```math
e_{b,t} = \\alpha_b e_{b,t-1}
        + \\eta^{ch}_b \\Delta t\\, p^{ch}_{b,t}
        - \\frac{\\Delta t}{\\eta^{dis}_b} p^{dis}_{b,t},
```

with ``e_{b,t}`` the END-of-stage energy. That convention is fixed here and is
never re-stated with a different meaning anywhere in either engine.
"""
struct BatterySpec
    index::Int
    bus::Int
    energy_min::Float64
    energy_max::Float64
    energy_initial::Float64
    charge_max::Float64
    discharge_max::Float64
    charge_efficiency::Float64
    discharge_efficiency::Float64
    self_discharge::Float64
    throughput_cost::Float64
end

"""
    reachable_interval(b::BatterySpec, e_prev, Δt) -> (lower, upper)

One-stage battery-dynamic reachable interval for the outgoing energy.

# Arguments
- `b::BatterySpec`: battery parameters.
- `e_prev::Real`: incoming energy ``e_{b,t-1}`` (pu·h).
- `Δt::Real`: stage duration in hours.

# Returns
- `(lower, upper)`: the closed interval

```math
\\underline r = \\max\\{\\underline e_b,\\;
    \\alpha_b e_{t-1} - \\tfrac{\\Delta t}{\\eta^{dis}_b}\\overline p^{dis}_b\\},
\\qquad
\\overline r = \\min\\{\\overline e_b,\\;
    \\alpha_b e_{t-1} + \\eta^{ch}_b \\Delta t\\, \\overline p^{ch}_b\\}.
```

# Notes
Every value in `[lower, upper]` is attained by an admissible
``(p^{ch}, p^{dis})`` pair, because the transition is affine and monotone in
each control and the controls' own boxes are intervals containing 0. This is
the map the strict policy squashes its normalized output into; note that BOTH
endpoints depend on `e_prev` with slope ``\\alpha_b`` wherever the energy bound
is not the binding term, which is why a policy that differentiates through this
map must not treat the endpoints as constants.

The returned interval is nonempty whenever ``\\underline e_b \\le \\alpha_b
e_{t-1} + \\eta^{ch}\\Delta t \\overline p^{ch}`` and ``\\alpha_b e_{t-1} -
\\Delta t \\overline p^{dis}/\\eta^{dis} \\le \\overline e_b``; with
``\\alpha_b = 1`` and ``e_{t-1} \\in [\\underline e_b, \\overline e_b]`` both
hold, so the interval is nonempty by induction along any trajectory the policy
itself generates.
"""
function reachable_interval(b::BatterySpec, e_prev::Real, Δt::Real)
    decayed = b.self_discharge * e_prev
    lower = max(b.energy_min, decayed - (Δt / b.discharge_efficiency) * b.discharge_max)
    upper = min(b.energy_max, decayed + b.charge_efficiency * Δt * b.charge_max)
    return lower, upper
end

"""
    dispatch_for_target(b::BatterySpec, e_prev, e_target, Δt) -> (p_ch, p_dis)

The charge/discharge pair that realizes `e_target` from `e_prev` in one stage.

# Arguments
- `b::BatterySpec`, `e_prev::Real`, `e_target::Real`, `Δt::Real`.

# Returns
- `(p_ch, p_dis)`: nonnegative powers (pu) satisfying the state transition
  exactly, with at most one of them nonzero.

# Notes
Writing ``\\delta := e_{target} - \\alpha_b e_{prev}``, the transition
``\\delta = \\eta^{ch}\\Delta t\\,p^{ch} - (\\Delta t/\\eta^{dis})p^{dis}``
is solved by

```math
p^{ch} = \\frac{\\max(\\delta, 0)}{\\eta^{ch}\\Delta t},
\\qquad
p^{dis} = \\frac{\\eta^{dis}\\max(-\\delta, 0)}{\\Delta t}.
```

This is the unique solution with `p_ch * p_dis == 0`; it is admissible exactly
when `e_target` lies in [`reachable_interval`](@ref). It is used to CERTIFY
recourse (given a reachable target, exhibit the controls that hit it) and never
to replace an optimizer's choice inside a stage problem.
"""
function dispatch_for_target(b::BatterySpec, e_prev::Real, e_target::Real, Δt::Real)
    δ = e_target - b.self_discharge * e_prev
    p_ch  = max(δ, 0.0) / (b.charge_efficiency * Δt)
    p_dis = b.discharge_efficiency * max(-δ, 0.0) / Δt
    return p_ch, p_dis
end

"""
    battery_injection(b::BatterySpec, p_ch, p_dis) -> Float64

Active power injected into the network, ``p^{bat} = p^{dis} - p^{ch}`` (pu).

# Notes
The battery operates at unity power factor: ``q^{bat} \\equiv 0``. Discharging
is a positive injection; charging is a negative one.
"""
battery_injection(::BatterySpec, p_ch::Real, p_dis::Real) = float(p_dis - p_ch)

"""
    RecourseCosts

Prices of the two-sided physical active-power recourse, in objective units per
pu per stage.

# Fields
- `deficit::Float64`: ``C^{def}``, price of the nonnegative uncapped injection
  ``d_{i,t}`` (unserved load, or the active power a charging target needs and
  the grid cannot deliver).
- `surplus::Float64`: ``C^{sur}``, price of the nonnegative uncapped sink
  ``s_{i,t}`` (active power a discharging target produces and the grid cannot
  absorb).

# Notes
Both prices must sit far above the most expensive generator so that recourse is
never an economic substitute for dispatch; both must be IDENTICAL in the
PowerModels ACP model, the PowerModels SOC-WR model, the Exa ACP model, and in
both SDDP passes. They travel in the frozen case artifact for exactly that
reason: neither engine gets to choose them.

`d` and `s` are physical operating recourse, not target slack. They appear only
in the nodal ACTIVE balance; they appear in no battery state equation and in no
strict target equality, and there is no target-slack variable anywhere in the
supported formulation.
"""
struct RecourseCosts
    deficit::Float64
    surplus::Float64
end

# ─────────────────────────────────────────────────────────────────────────────
# The frozen finite demand support
#
# Demand is the study's ONLY uncertainty. For the original PGLib load values
# p^{d,0}_i and q^{d,0}_i, the realized demand of load i at stage t under atom k
# is
#
#     p^d_{i,t,k} = h_{i,t} m^{(k)}_{i,t} p^{d,0}_i
#     q^d_{i,t,k} = h_{i,t} m^{(k)}_{i,t} q^{d,0}_i
#
# with h a DETERMINISTIC temporal profile and m the uncertain multiplier. The
# SAME multiplier scales the active and the reactive demand, so every
# realization has the case's own power factor: the uncertainty moves how much
# power is consumed, never what kind.
#
# The multiplier is a JOINT VECTOR over loads, not a scalar and not a collection
# of independent draws — an independent-per-load sampler is one way to produce
# such a vector, never the representation itself.
# ─────────────────────────────────────────────────────────────────────────────

"""
    DemandSupport

The frozen, stage-major finite support of the demand process.

# Fields
- `stage_hours::Float64`: ``\\Delta t``, the duration of one stage in hours.
- `horizon::Int`: number of stages the support is frozen for. Stage indices
  outside `1:horizon` are an error, never silently wrapped.
- `load_ids::Vector{Int}`: LOAD identifiers, sorted ascending. This vector fixes
  the order of every multiplier and profile row and is the only definition of
  "component `j`" in this artifact.
- `profile::Matrix{Float64}`: ``h_{i,t}``, size `(length(load_ids), horizon)`,
  the deterministic temporal profile.
- `atoms::Vector{Matrix{Float64}}`: `atoms[t]` has size
  `(length(load_ids), K_t)`; column `k` is the joint multiplier vector
  ``m^{(k)}_{\\cdot,t}``.
- `probabilities::Vector{Vector{Float64}}`: `probabilities[t]` has length
  `K_t` and sums to 1.
- `protocol_seed::Int`: seed of the `StableRNG` that generates evaluation
  scenario index matrices.
- `source::Dict{String,Any}`: a description of the AUTHORING sampler and of the
  discretization that produced these atoms. Documentation, not data: nothing
  reads it to build a model. It exists so a frozen support can be traced back to
  the sampler it came from.

# Notes
The support is stage-dependent by construction (`K_t` may differ across stages)
and stagewise independent: an atom index at stage `t` carries no information
about stage `t+1`. That is what both SDDP's backward enumeration and TS-DDR's
trajectory sampling assume, and it is enforced by the representation rather than
by a comment.
"""
struct DemandSupport
    stage_hours::Float64
    horizon::Int
    load_ids::Vector{Int}
    profile::Matrix{Float64}
    atoms::Vector{Matrix{Float64}}
    probabilities::Vector{Vector{Float64}}
    protocol_seed::Int
    source::Dict{String,Any}
end

"Number of loads the support is defined over."
num_loads(s::DemandSupport) = length(s.load_ids)

"Number of stages the support is frozen for."
horizon(s::DemandSupport) = s.horizon

"""
    profile_period(s::DemandSupport) -> Int

The cycle length the deterministic profile repeats on, in stages.

# Notes
Recorded by the freezing operation and used for exactly one purpose: as the
period of the ``(\\sin, \\cos)`` clock feature a policy is given, so that the
position in the daily cycle is encoded without the discontinuity a raw stage
index would introduce at midnight. It is DETERMINISTIC information — knowing the
clock is not knowing the future demand — and it never enters a stage problem.

Falls back to the frozen horizon when a support was written without one, which
degrades the feature to "position in the horizon" rather than silently claiming
a 24-stage cycle a case may not have.
"""
profile_period(s::DemandSupport) = Int(get(s.source, "profile_period", s.horizon))

"""
    num_atoms(s::DemandSupport, t) -> Int

Size ``K_t`` of the finite support at stage `t`.
"""
function num_atoms(s::DemandSupport, t::Integer)
    _check_stage(s, t)
    return length(s.probabilities[t])
end

"""
    atom_probabilities(s::DemandSupport, t) -> Vector{Float64}

The probabilities ``(p_{t,1},\\ldots,p_{t,K_t})`` of stage `t`'s atoms.
"""
function atom_probabilities(s::DemandSupport, t::Integer)
    _check_stage(s, t)
    return s.probabilities[t]
end

"""
    demand_multipliers(s::DemandSupport, t, atom) -> Vector{Float64}

The TOTAL per-load multiplier ``h_{i,t}\\,m^{(atom)}_{i,t}`` at stage `t`, in
`load_ids` order.

# Notes
This is the only place the deterministic profile and the uncertain multiplier are
combined, so the two can never be applied twice or in the wrong order anywhere
downstream.
"""
function demand_multipliers(s::DemandSupport, t::Integer, atom::Integer)
    _check_stage(s, t)
    K = num_atoms(s, t)
    1 <= atom <= K ||
        throw(ArgumentError("atom index $atom outside 1:$K at stage $t"))
    return @views s.profile[:, t] .* s.atoms[t][:, atom]
end

# Fail closed on a stage index outside the frozen window. Silently wrapping (as
# a cyclic profile would) is how a horizon change becomes an undetected change
# of problem.
function _check_stage(s::DemandSupport, t::Integer)
    1 <= t <= s.horizon ||
        throw(ArgumentError("stage $t outside the frozen horizon 1:$(s.horizon)"))
    return nothing
end

"""
    support_digest(s::DemandSupport) -> String

SHA-256 of the frozen support, in a fixed textual encoding.

# Notes
This digest is what makes "SDDP and TS-DDR consumed the same demand support" a
verifiable claim: both engines recompute it from the bytes they loaded and it is
recorded in the manifest. It covers the stage duration, the horizon, the load
order, the profile, every atom and every probability — that is, everything a
stage problem's demand depends on — and deliberately NOT the `source`
description, which is prose about how the atoms were authored and must not be
able to change the identity of a support.
"""
function support_digest(s::DemandSupport)
    io = IOBuffer()
    println(io, "battery_storage_opf/support/2")
    println(io, s.stage_hours, " ", s.horizon, " ", num_loads(s), " ", s.protocol_seed)
    println(io, join(s.load_ids, ","))
    for t in 1:s.horizon
        println(io, "t", t, " ", num_atoms(s, t))
        println(io, join((string(x) for x in @views s.profile[:, t]), ","))
        for k in 1:num_atoms(s, t)
            println(io, string(s.probabilities[t][k]), " ",
                    join((string(x) for x in @views s.atoms[t][:, k]), ","))
        end
    end
    return bytes2hex(sha256(take!(io)))
end

"""
    scenario_index_matrix(s::DemandSupport, num_stages, num_scenarios;
                          seed=nothing, exclude=nothing) -> Matrix{Int}

A paired evaluation protocol, reproduced by construction rather than stored.

# Arguments
- `s::DemandSupport`: supplies the default seed and the per-stage support sizes.
- `num_stages::Integer`, `num_scenarios::Integer`: shape of the protocol.

# Keywords
- `seed`: `nothing` for the support's own `protocol_seed` — which is the FINAL
  protocol's seed — or another integer for an INDEPENDENT protocol. A study needs
  at least two: a small screening protocol it may look at while choosing a case
  and selecting checkpoints, and a final one no policy was ever selected on.
  Taking the screening set as a prefix of the final one would make the final
  protocol not fresh, which is the whole property it exists to have.
- `exclude`: an iterable of length-`num_stages` integer columns this protocol may
  not contain. Passing the FINAL protocol's columns here is what makes a
  screening protocol disjoint from it BY CONSTRUCTION rather than by the
  probabilistic argument that a collision is unlikely — see the notes.

# Returns
- `Matrix{Int}` of size `(num_stages, num_scenarios)`; entry `[t, s]` is the
  atom index realized at stage `t` of paired column `s`.

# Notes
Drawn from `StableRNG(protocol_seed)`, whose stream is fixed across Julia
versions and platforms, so both engines regenerate the identical matrix and only
its SHA-256 needs to be recorded in the manifest. Scenario columns are global
and immutable: column `s` means the same demand path to every policy and to
every shard of an evaluation.

Draw ORDER is stage-major (all scenarios of stage 1, then all of stage 2, …).
Because each stage's support size ``K_t`` may differ, a scenario-major order
would make the stream position depend on the horizon; stage-major keeps a
protocol of `num_scenarios` columns a prefix of a protocol of more columns only
within a stage, which is the property shard boundaries rely on.

The exclusion is applied as a REPAIR after that stage-major draw, never as a
per-draw filter, precisely so the stage-major property survives it: the matrix is
drawn exactly as it would have been without `exclude`, then any column that is
banned or that repeats an earlier column of this same matrix is redrawn — in
ascending column order, from the continuation of the same stream, retrying until
the column is admissible. On a support with more paths than columns nothing is
ever redrawn and the matrix is bit-identical to the unexcluded one; the repair
exists so that "screening and final share no scenario" is a structural fact on
a small support too, where a collision is not merely unlikely but certain.
"""
function scenario_index_matrix(s::DemandSupport, num_stages::Integer, num_scenarios::Integer;
                               seed = nothing, exclude = nothing)
    num_stages >= 1 || throw(ArgumentError("num_stages must be positive"))
    num_scenarios >= 1 || throw(ArgumentError("num_scenarios must be positive"))
    num_stages <= s.horizon ||
        throw(ArgumentError("protocol asks for $num_stages stages but the support is frozen for $(s.horizon)"))
    rng = StableRNG(seed === nothing ? s.protocol_seed : Int(seed))
    m = Matrix{Int}(undef, num_stages, num_scenarios)
    for t in 1:num_stages
        K = num_atoms(s, t)
        for c in 1:num_scenarios
            m[t, c] = rand(rng, 1:K)
        end
    end
    exclude === nothing && return m

    # The banned set: the columns the caller forbids, plus — as they are
    # accepted — the columns of this protocol itself, so a repaired protocol
    # never contains the same scenario twice either.
    banned = Set{Vector{Int}}()
    for col in exclude
        v = Int.(collect(col))
        length(v) == num_stages ||
            throw(ArgumentError("excluded column has $(length(v)) stages, expected $num_stages"))
        push!(banned, v)
    end
    # The support has ∏_t K_t distinct paths; asking for more admissible columns
    # than exist is a specification error, not something to discover by looping.
    capacity = prod(BigInt(num_atoms(s, t)) for t in 1:num_stages)
    capacity >= length(banned) + num_scenarios ||
        throw(ArgumentError("the support has $capacity distinct $num_stages-stage paths, " *
                            "which cannot supply $num_scenarios columns disjoint from " *
                            "$(length(banned)) excluded ones"))
    for c in 1:num_scenarios
        col = Int[m[t, c] for t in 1:num_stages]
        while col in banned
            for t in 1:num_stages
                col[t] = rand(rng, 1:num_atoms(s, t))
            end
        end
        for t in 1:num_stages
            m[t, c] = col[t]
        end
        push!(banned, col)
    end
    return m
end

"""
    protocol_columns(m::AbstractMatrix{<:Integer}) -> Vector{Vector{Int}}

The columns of a protocol index matrix, in the form
[`scenario_index_matrix`](@ref) accepts as `exclude`.
"""
protocol_columns(m::AbstractMatrix{<:Integer}) =
    [Int[m[t, c] for t in 1:size(m, 1)] for c in 1:size(m, 2)]

"""
    protocol_digest(s::DemandSupport, num_stages, num_scenarios;
                    seed=nothing, exclude=nothing) -> String

SHA-256 of the protocol index matrix, in a fixed textual encoding.

# Notes
The digest, not the matrix, is what the manifest stores. Both engines recompute
the matrix from the seed and must obtain this digest; a mismatch means the two
engines are not evaluating the same scenarios and no comparison between them is
meaningful.

The digest is of the MATRIX, so it says nothing about how the matrix was
repaired: two calls that produce the same columns hash the same whether or not an
exclusion set was in force. What records the exclusion is the manifest field that
names it, which is also what a reader needs in order to regenerate the matrix.
"""
function protocol_digest(s::DemandSupport, num_stages::Integer, num_scenarios::Integer;
                         seed = nothing, exclude = nothing)
    m = scenario_index_matrix(s, num_stages, num_scenarios; seed = seed, exclude = exclude)
    io = IOBuffer()
    println(io, "battery_storage_opf/protocol/2")
    println(io, num_stages, " ", num_scenarios, " ",
            seed === nothing ? s.protocol_seed : Int(seed))
    println(io, join((num_atoms(s, t) for t in 1:num_stages), ","))
    for t in 1:num_stages
        println(io, join(view(m, t, :), ","))
    end
    return bytes2hex(sha256(take!(io)))
end

"""
    validate_support(s::DemandSupport) -> Nothing

Fail closed on every property the rest of the study assumes of a frozen support.

# Notes
Each check corresponds to a way two engines could end up solving different
problems while both reporting success: a probability vector that does not
normalize silently reweights an SDDP backward pass; a negative or non-finite
multiplier produces a load the network was never meant to serve; a load order
that is not sorted-unique makes "component `j`" mean two different things in two
engines.
"""
function validate_support(s::DemandSupport)
    s.stage_hours > 0 || error("stage_hours must be positive, got $(s.stage_hours)")
    s.horizon >= 1 || error("horizon must be at least 1, got $(s.horizon)")
    n = num_loads(s)
    n >= 1 || error("a demand support must cover at least one load")
    issorted(s.load_ids) && allunique(s.load_ids) ||
        error("load_ids must be sorted and unique; got $(s.load_ids)")
    size(s.profile) == (n, s.horizon) ||
        error("profile must be $(n)×$(s.horizon), got $(size(s.profile))")
    all(isfinite, s.profile) || error("profile has a non-finite entry")
    all(>=(0), s.profile) || error("profile has a negative entry")
    length(s.atoms) == s.horizon ||
        error("support has $(length(s.atoms)) stages of atoms but horizon $(s.horizon)")
    length(s.probabilities) == s.horizon ||
        error("support has $(length(s.probabilities)) stages of probabilities but horizon $(s.horizon)")
    for t in 1:s.horizon
        A = s.atoms[t]
        p = s.probabilities[t]
        size(A, 1) == n ||
            error("stage $t atoms have $(size(A, 1)) rows but the support covers $n loads")
        size(A, 2) == length(p) ||
            error("stage $t has $(size(A, 2)) atoms but $(length(p)) probabilities")
        length(p) >= 1 || error("stage $t has an empty support")
        all(isfinite, A) || error("stage $t has a non-finite multiplier")
        all(>=(0), A) || error("stage $t has a negative multiplier")
        all(>(0), p) || error("stage $t has a non-positive probability")
        isapprox(sum(p), 1.0; atol = 1e-12) ||
            error("stage $t probabilities sum to $(sum(p)), not 1")
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Battery placement
#
# Placement is authoring-side, but it lives here because the manifest must be
# able to record exactly how a fleet was chosen, and because the eligibility and
# validity rules are properties of the case contract rather than of a script.
# ─────────────────────────────────────────────────────────────────────────────

"""
    PlacementStrategy

How the buses hosting batteries are chosen. Concrete strategies:
[`ExplicitPlacement`](@ref), [`SampledPlacement`](@ref),
[`CallablePlacement`](@ref).

# Notes
Every strategy returns bus IDENTIFIERS, never positions, and every strategy is
reproducible from the data recorded in the manifest alone.
"""
abstract type PlacementStrategy end

"""
    ExplicitPlacement(buses)

Place batteries at the given bus identifiers, in ascending order.

# Notes
`count` is not consulted: the list IS the fleet size. Duplicates are rejected
rather than deduplicated, because a repeated identifier is far more likely to be
a typo than a request for two batteries at one bus (which is expressed by giving
two `BatterySpec`s at the same bus instead).
"""
struct ExplicitPlacement <: PlacementStrategy
    buses::Vector{Int}
end

ExplicitPlacement(buses) = ExplicitPlacement(sort!(collect(Int.(buses))))

"""
    SampledPlacement(count; seed, weight=nothing)

Draw `count` distinct eligible buses without replacement.

# Fields
- `count::Int`: fleet size.
- `seed::Int`: seed of the `StableRNG` driving the draw.
- `weight`: `nothing` for a uniform draw, or a callable `bus -> Float64`
  returning a nonnegative sampling weight (e.g. nominal demand at the bus).

# Notes
Without replacement means a bus drawn once is removed from the pool, so a
weighted draw is a successive-sampling scheme rather than `count` independent
draws. The candidate pool is SORTED before the first draw, which is what makes
the result independent of dictionary iteration order.
"""
struct SampledPlacement <: PlacementStrategy
    count::Int
    seed::Int
    weight::Any
end

SampledPlacement(count::Integer; seed::Integer, weight = nothing) =
    SampledPlacement(Int(count), Int(seed), weight)

"""
    CallablePlacement(f; name="callable")

Place batteries at `f(candidates, meta)`, where `candidates` is the sorted vector
of eligible bus identifiers and `meta` is the placement metadata named tuple.

# Notes
The escape hatch for a placement rule the study does not anticipate — a
graph-theoretic centrality, an optimization, a hand-drawn map. Whatever it
returns is validated exactly as any other strategy's output, and `name` is what
the manifest records in place of a rule it cannot serialize.
"""
struct CallablePlacement <: PlacementStrategy
    f::Any
    name::String
end

CallablePlacement(f; name::AbstractString = "callable") = CallablePlacement(f, String(name))

"""
    load_buses(network) -> Vector{Int}

Sorted identifiers of buses hosting at least one in-service load.

# Notes
The default eligible set. A battery at a bus that neither consumes nor generates
is a pure network-support device, which is a different study; restricting to load
buses keeps a randomly placed fleet physically interpretable on any PGLib case.
"""
function load_buses(network::AbstractDict)
    out = Set{Int}()
    for (_, load) in network["load"]
        Int(get(load, "status", 1)) == 0 && continue
        push!(out, Int(load["load_bus"]))
    end
    return sort!(collect(out))
end

"""
    nominal_load_at_bus(network) -> Dict{Int,Float64}

Nominal in-service active demand aggregated per bus identifier (pu).
"""
function nominal_load_at_bus(network::AbstractDict)
    out = Dict{Int,Float64}()
    for (_, load) in network["load"]
        Int(get(load, "status", 1)) == 0 && continue
        bus = Int(load["load_bus"])
        out[bus] = get(out, bus, 0.0) + Float64(load["pd"])
    end
    return out
end

"""
    eligible_buses(network; eligible=nothing) -> Vector{Int}

The sorted candidate set a sampling placement draws from.

# Keywords
- `eligible`: `nothing` for the default (in-service load buses), an iterable of
  bus identifiers, or a predicate `bus_dict -> Bool` applied to each bus entry of
  the network.

# Notes
Whatever the source, the returned buses are checked to exist, to be in service
and to be connected — a bus with no incident in-service branch cannot host a
battery that participates in the study, and PGLib cases do contain isolated
buses.
"""
function eligible_buses(network::AbstractDict; eligible = nothing)
    ids = Set(Int(b["index"]) for (_, b) in network["bus"])
    candidates = if eligible === nothing
        load_buses(network)
    elseif eligible isa Function
        sort!([Int(b["index"]) for (_, b) in network["bus"] if eligible(b)])
    else
        sort!(collect(Int.(eligible)))
    end
    allunique(candidates) || error("eligible bus set contains duplicates")
    in_service = Set(Int(b["index"]) for (_, b) in network["bus"]
                     if Int(get(b, "bus_type", 1)) != 4)
    connected = Set{Int}()
    for (_, br) in network["branch"]
        Int(get(br, "br_status", 1)) == 0 && continue
        push!(connected, Int(br["f_bus"]))
        push!(connected, Int(br["t_bus"]))
    end
    for b in candidates
        b in ids || error("bus $b is not in the network")
        b in in_service || error("bus $b is out of service (bus_type 4)")
        b in connected || error("bus $b has no in-service branch and is disconnected")
    end
    isempty(candidates) && error("no eligible bus remains after filtering")
    return candidates
end

"""
    select_battery_buses(network, strategy; eligible=nothing) -> (buses, record)

Apply a [`PlacementStrategy`](@ref) and return both the chosen buses and the
manifest record describing how they were chosen.

# Returns
- `buses::Vector{Int}`: sorted, distinct, validated bus identifiers.
- `record::Dict{String,Any}`: strategy name, seed, eligible set, weights and the
  selection, in a form the manifest can serialize verbatim.

# Notes
The eligible set is recorded in FULL, not summarized. "Three buses were drawn
from the load buses" is not reproducible if a later revision of the case adds a
load; the actual pool that was drawn from is.
"""
function select_battery_buses(network::AbstractDict, strategy::PlacementStrategy;
                              eligible = nothing)
    candidates = eligible_buses(network; eligible = eligible)
    record = Dict{String,Any}("eligible" => candidates)

    buses = if strategy isa ExplicitPlacement
        allunique(strategy.buses) || error("explicit battery buses must be distinct")
        for b in strategy.buses
            b in candidates ||
                error("explicit battery bus $b is not in the eligible set")
        end
        record["strategy"] = "explicit"
        copy(strategy.buses)

    elseif strategy isa SampledPlacement
        strategy.count >= 1 || throw(ArgumentError("count must be at least 1"))
        strategy.count <= length(candidates) ||
            throw(ArgumentError("cannot place $(strategy.count) batteries on $(length(candidates)) eligible buses"))
        rng = StableRNG(strategy.seed)
        weights = strategy.weight === nothing ?
                  fill(1.0, length(candidates)) :
                  [Float64(strategy.weight(b)) for b in candidates]
        all(isfinite, weights) || error("placement weights must be finite")
        all(>=(0), weights) || error("placement weights must be nonnegative")
        record["strategy"] = strategy.weight === nothing ? "uniform" : "weighted"
        record["seed"] = strategy.seed
        record["weights"] = weights
        sort!(_sample_without_replacement(rng, candidates, weights, strategy.count))

    elseif strategy isa CallablePlacement
        chosen = sort!(collect(Int.(strategy.f(candidates, (network = network,
                                                           candidates = candidates)))))
        allunique(chosen) || error("callable placement returned duplicate buses")
        for b in chosen
            b in candidates || error("callable placement returned ineligible bus $b")
        end
        record["strategy"] = "callable:" * strategy.name
        chosen

    else
        error("unsupported placement strategy $(typeof(strategy))")
    end

    isempty(buses) && error("placement selected no bus")
    record["selected"] = buses
    return buses, record
end

"""
    _sample_without_replacement(rng, items, weights, count) -> Vector

Successive weighted sampling without replacement.

# Notes
At each of `count` rounds the remaining items are sampled with probability
proportional to their weight and the chosen item is removed. With all weights
equal this reduces to a uniform draw without replacement. The implementation
consumes the stream through `rand(rng)` only, so the result depends on the seed
and not on any `Random` API whose behaviour is free to change between Julia
versions.
"""
function _sample_without_replacement(rng, items::AbstractVector, weights::AbstractVector,
                                     count::Integer)
    pool = collect(items)
    w = collect(Float64.(weights))
    out = eltype(items)[]
    for _ in 1:count
        total = sum(w)
        total > 0 || error("placement weights of the remaining pool sum to zero")
        u = rand(rng) * total
        acc = 0.0
        j = length(w)
        for i in eachindex(w)
            acc += w[i]
            if u <= acc
                j = i
                break
            end
        end
        push!(out, pool[j])
        deleteat!(pool, j)
        deleteat!(w, j)
    end
    return out
end

"""
    battery_fleet(network, buses; power, energy_hours, charge_efficiency,
                  discharge_efficiency, self_discharge, throughput_cost,
                  initial_fraction) -> (Vector{BatterySpec}, record)

Give the selected buses their ratings.

# Arguments
- `buses::AbstractVector{Int}`: the output of [`select_battery_buses`](@ref).

# Keywords
- `power`: the power rating rule. Either a `Real` in pu applied to every
  battery, a `Dict{Int,<:Real}` keyed by bus, or a callable `bus -> Real`. A
  callable closing over a `Distribution` and an RNG is how a SAMPLED capacity is
  expressed without this file depending on Distributions.jl.
- `energy_hours`: energy rating as hours at full discharge power; same three
  forms as `power`.
- `charge_efficiency`, `discharge_efficiency`, `self_discharge`,
  `throughput_cost`, `initial_fraction`: same three forms; scalars in practice.
- `reserve_fraction`: the OPERATING BAND. `energy_min = reserve_fraction *
  energy_max` and `energy_max` is unchanged, so a nonzero value keeps the battery
  off the exact bottom of its box. Physically it is the reserve a real battery is
  not allowed to discharge below; numerically it matters more than it sounds,
  because at an exact box corner the one-stage reachable interval collapses
  against a bound, the transition equality and the energy bound become parallel,
  and the resulting near-degenerate face is what defeats a conic interior-point
  method on the SOC-WR relaxation.

# Returns
- The fleet sorted by battery index, and the manifest record of the capacity rule.

# Notes
Batteries are indexed `1:n` in the order of the (sorted) bus identifiers. Battery
INDEX is an identity, not a position — every engine keys on it — but assigning
them consecutively at construction keeps the frozen artifact readable.

Every parameter is validated here rather than at read time as well, so a case
that cannot be built is rejected where the rule that produced it is still in
scope.
"""
function battery_fleet(network::AbstractDict, buses::AbstractVector{<:Integer};
                       power,
                       energy_hours,
                       charge_efficiency = 0.95,
                       discharge_efficiency = 0.95,
                       self_discharge = 1.0,
                       throughput_cost = 0.0,
                       initial_fraction = 0.5,
                       reserve_fraction = 0.0)
    resolve(rule, bus) = rule isa Function ? Float64(rule(bus)) :
                         rule isa AbstractDict ? Float64(rule[bus]) : Float64(rule)

    specs = BatterySpec[]
    record = Dict{String,Any}("power_pu" => Dict{String,Any}(),
                              "energy_hours" => Dict{String,Any}())
    for (i, bus) in enumerate(buses)
        p = resolve(power, bus)
        h = resolve(energy_hours, bus)
        ηc = resolve(charge_efficiency, bus)
        ηd = resolve(discharge_efficiency, bus)
        α  = resolve(self_discharge, bus)
        c  = resolve(throughput_cost, bus)
        f0 = resolve(initial_fraction, bus)
        rf = resolve(reserve_fraction, bus)

        p > 0 || error("battery at bus $bus: power rating must be positive, got $p")
        h > 0 || error("battery at bus $bus: energy duration must be positive, got $h")
        0 < ηc <= 1 || error("battery at bus $bus: charge_efficiency out of (0,1]")
        0 < ηd <= 1 || error("battery at bus $bus: discharge_efficiency out of (0,1]")
        0 < α <= 1 || error("battery at bus $bus: self_discharge out of (0,1]")
        c >= 0 || error("battery at bus $bus: throughput_cost must be nonnegative")
        0 <= f0 <= 1 || error("battery at bus $bus: initial_fraction out of [0,1]")
        0 <= rf < 1 || error("battery at bus $bus: reserve_fraction out of [0,1)")
        rf <= f0 || error("battery at bus $bus: initial_fraction $f0 is below the reserve $rf")

        e_max = h * p
        push!(specs, BatterySpec(i, Int(bus), rf * e_max, e_max, f0 * e_max, p, p,
                                 ηc, ηd, α, c))
        record["power_pu"][string(bus)] = p
        record["reserve_fraction"] = rf
        record["energy_hours"][string(bus)] = h
    end
    return specs, record
end

# ─────────────────────────────────────────────────────────────────────────────
# Case container and I/O
# ─────────────────────────────────────────────────────────────────────────────

"""
    BatteryCase

Everything both engines need in order to build the same stage problem.

# Fields
- `dir::String`: directory the artifacts were read from.
- `name::String`: PGLib case name, e.g. `"pglib_opf_case14_ieee"`.
- `network::Dict{String,Any}`: the parsed PGLib network, per-unit, verbatim.
- `batteries::Vector{BatterySpec}`: sorted by battery index.
- `recourse::RecourseCosts`: prices of the two-sided nodal active recourse.
- `demand::DemandSupport`: the frozen finite demand support.
- `manifest::Dict{String,Any}`: the manifest as read from disk.
"""
struct BatteryCase
    dir::String
    name::String
    network::Dict{String,Any}
    batteries::Vector{BatterySpec}
    recourse::RecourseCosts
    demand::DemandSupport
    manifest::Dict{String,Any}
end

"Stage duration ``\\Delta t`` in hours."
stage_hours(c::BatteryCase) = c.demand.stage_hours

"""
    nominal_load_demand(case) -> (pd::Vector{Float64}, qd::Vector{Float64})

Nominal active and reactive demand of every load in `case.demand.load_ids`
order (pu).

# Notes
The support's load order — not the network dictionary's iteration order — is what
indexes every multiplier vector, so it is what indexes the nominal values too.
"""
function nominal_load_demand(case::BatteryCase)
    by_id = Dict{Int,Any}(Int(l["index"]) => l for (_, l) in case.network["load"])
    pd = Vector{Float64}(undef, num_loads(case.demand))
    qd = Vector{Float64}(undef, num_loads(case.demand))
    for (j, id) in enumerate(case.demand.load_ids)
        load = by_id[id]
        pd[j] = Float64(load["pd"])
        qd[j] = Float64(load["qd"])
    end
    return pd, qd
end

"""
    nominal_bus_demand(case) -> (pd::Dict{Int,Float64}, qd::Dict{Int,Float64})

Nominal active and reactive demand aggregated per BUS identifier (pu).

# Notes
A bus may host several loads; the network's nodal balance constrains only their
sum, so both engines aggregate to the bus before anything else happens. Buses
with no load appear with an explicit `0.0` so downstream code can index every
bus without a `get` default and its attendant typo risk.
"""
function nominal_bus_demand(case::BatteryCase)
    pd = Dict{Int,Float64}(Int(b["index"]) => 0.0 for (_, b) in case.network["bus"])
    qd = Dict{Int,Float64}(Int(b["index"]) => 0.0 for (_, b) in case.network["bus"])
    for (_, load) in case.network["load"]
        Int(get(load, "status", 1)) == 0 && continue
        bus = Int(load["load_bus"])
        pd[bus] += Float64(load["pd"])
        qd[bus] += Float64(load["qd"])
    end
    return pd, qd
end

"""
    realized_bus_demand(case, t, atom) -> (pd::Dict{Int,Float64}, qd::Dict{Int,Float64})

Per-bus demand realized at stage `t` under atom index `atom` (pu).

# Notes
Each load is scaled by its own total multiplier
``h_{i,t} m^{(atom)}_{i,t}`` and the scaled loads are then aggregated to their
bus. The same multiplier scales active and reactive demand, so the power factor
of every individual load is preserved exactly — which is a stronger statement
than preserving the aggregate power factor at the bus, and is the one the study
claims.

Loads that are out of service contribute nothing, and buses with no load appear
with `0.0`, so the returned dictionaries cover every bus of the network.
"""
function realized_bus_demand(case::BatteryCase, t::Integer, atom::Integer)
    pd = Dict{Int,Float64}(Int(b["index"]) => 0.0 for (_, b) in case.network["bus"])
    qd = Dict{Int,Float64}(Int(b["index"]) => 0.0 for (_, b) in case.network["bus"])
    mult = demand_multipliers(case.demand, t, atom)
    by_id = Dict{Int,Any}(Int(l["index"]) => l for (_, l) in case.network["load"])
    for (j, id) in enumerate(case.demand.load_ids)
        load = by_id[id]
        Int(get(load, "status", 1)) == 0 && continue
        bus = Int(load["load_bus"])
        pd[bus] += Float64(load["pd"]) * mult[j]
        qd[bus] += Float64(load["qd"]) * mult[j]
    end
    return pd, qd
end

"""
    demand_path(case, atoms) -> Vector{Tuple{Dict{Int,Float64},Dict{Int,Float64}}}

Materialize a COMPLETE demand path: the per-bus `(pd, qd)` of every stage of the
atom-index vector `atoms`.

# Notes
A "demand path" is the object a deterministic-equivalent solve and a
perfect-foresight panel consume; giving it a name here keeps every caller from
re-deriving the stage-to-atom mapping and getting the stage offset wrong.
"""
function demand_path(case::BatteryCase, atoms::AbstractVector{<:Integer})
    return [realized_bus_demand(case, t, atoms[t]) for t in eachindex(atoms)]
end

"""
    write_battery_case(dir; name, network, batteries, recourse, demand,
                       source_version, placement, protocol_stages,
                       protocol_scenarios) -> Dict{String,Any}

Write the four frozen artifacts and return the manifest that was written.

# Notes
The manifest is written LAST and records the SHA-256 of the three artifacts as
they landed on disk, so a manifest can never describe bytes that were never
written. Every value the two engines must agree on — stage duration, unit
conventions, component counts, the support digest, the protocol digest — is
recorded here rather than recomputed independently on each side.

Stage duration is recorded once, in hours, and read back by
[`read_battery_case`](@ref) with a fail-closed check. This is the battery
analogue of the hydro `stage_hours`, whose omission once rescaled a whole
study's dynamics by a factor of 168.
"""
function write_battery_case(dir::AbstractString;
                            name::AbstractString,
                            network::AbstractDict,
                            batteries::AbstractVector{BatterySpec},
                            recourse::RecourseCosts,
                            demand::DemandSupport,
                            source_version::AbstractString,
                            placement::AbstractDict = Dict{String,Any}(),
                            protocol_stages::Integer,
                            protocol_scenarios::Integer,
                            screening_seed::Union{Nothing,Integer} = nothing,
                            screening_scenarios::Integer = 0)
    validate_support(demand)
    mkpath(dir)

    network_obj = Dict{String,Any}(
        "schema" => BATTERY_NETWORK_SCHEMA,
        "name" => name,
        "source_version" => source_version,
        "data" => network,
    )
    network_sha = write_canonical_json(joinpath(dir, "network.json"), network_obj)

    batteries_obj = Dict{String,Any}(
        "schema" => BATTERY_BATTERY_SCHEMA,
        "case" => name,
        # The two-sided nodal active recourse is part of the same extension
        # layer as the batteries: it is what makes a strict, dynamically
        # reachable target admissible under the true network. Freezing its
        # prices here is what guarantees both engines and both SDDP passes
        # charge for it identically.
        "recourse" => Dict{String,Any}(
            "deficit_cost" => recourse.deficit,
            "surplus_cost" => recourse.surplus,
        ),
        # How the fleet was chosen, in enough detail to redraw it.
        "placement" => Dict{String,Any}(placement),
        "batteries" => [Dict{String,Any}(
            "index" => b.index,
            "bus" => b.bus,
            "energy_min" => b.energy_min,
            "energy_max" => b.energy_max,
            "energy_initial" => b.energy_initial,
            "charge_max" => b.charge_max,
            "discharge_max" => b.discharge_max,
            "charge_efficiency" => b.charge_efficiency,
            "discharge_efficiency" => b.discharge_efficiency,
            "self_discharge" => b.self_discharge,
            "throughput_cost" => b.throughput_cost,
        ) for b in batteries],
    )
    batteries_sha = write_canonical_json(joinpath(dir, "batteries.json"), batteries_obj)

    demand_obj = Dict{String,Any}(
        "schema" => BATTERY_DEMAND_SCHEMA,
        "case" => name,
        "stage_hours" => demand.stage_hours,
        "horizon" => demand.horizon,
        "load_ids" => demand.load_ids,
        # Stage-major, load-minor: `profile[t][j]` is load `load_ids[j]` at stage
        # `t`. Writing it stage-major matches how a stage problem reads it.
        "profile" => [Float64[demand.profile[j, t] for j in 1:num_loads(demand)]
                      for t in 1:demand.horizon],
        "atoms" => [[Float64[demand.atoms[t][j, k] for j in 1:num_loads(demand)]
                     for k in 1:num_atoms(demand, t)] for t in 1:demand.horizon],
        "probabilities" => [copy(demand.probabilities[t]) for t in 1:demand.horizon],
        "protocol_seed" => demand.protocol_seed,
        "source" => Dict{String,Any}(demand.source),
    )
    demand_sha = write_canonical_json(joinpath(dir, "demand.json"), demand_obj)

    manifest = Dict{String,Any}(
        "schema" => BATTERY_MANIFEST_SCHEMA,
        "case" => name,
        "source" => Dict{String,Any}("package" => "PGLib.jl", "version" => source_version),
        "stage_hours" => demand.stage_hours,
        "units" => Dict{String,Any}(
            "power" => "per-unit on network baseMVA",
            "energy" => "per-unit-hours (pu power sustained for one hour)",
            "time" => "hours",
            # Every number this study builds, solves and reports is in this one
            # unit. No stage objective is rescaled on its way into a solver and
            # none is converted on its way out: the model a solver sees carries
            # the physical stage cost, so a reported cost, a cut and a multiplier
            # are all comparable without any conversion step.
            "cost" => "objective units of the PGLib case per hour",
        ),
        "counts" => Dict{String,Any}(
            "bus" => length(network["bus"]),
            "gen" => length(network["gen"]),
            "branch" => length(network["branch"]),
            "load" => length(network["load"]),
            "shunt" => length(get(network, "shunt", Dict())),
            "battery" => length(batteries),
            "horizon" => demand.horizon,
            "demand_atom_min" => minimum(num_atoms(demand, t) for t in 1:demand.horizon),
            "demand_atom_max" => maximum(num_atoms(demand, t) for t in 1:demand.horizon),
        ),
        "baseMVA" => Float64(network["baseMVA"]),
        "recourse" => Dict{String,Any}(
            "deficit_cost" => recourse.deficit,
            "surplus_cost" => recourse.surplus,
        ),
        "support" => Dict{String,Any}(
            "sha256" => support_digest(demand),
            "source" => Dict{String,Any}(demand.source),
        ),
        # The FINAL paired protocol. Generated from the support's own seed and
        # hashed here; a phase that must not evaluate on it can still record what
        # it will be. It is generated FIRST and depends on nothing else, which is
        # what keeps it independent of every screening decision.
        "protocol" => Dict{String,Any}(
            "seed" => demand.protocol_seed,
            "num_stages" => protocol_stages,
            "num_scenarios" => protocol_scenarios,
            "sha256" => protocol_digest(demand, protocol_stages, protocol_scenarios),
        ),
        # The SCREENING protocol, drawn from an INDEPENDENT seed and repaired
        # against the final protocol's columns, so the two panels share no
        # scenario BY CONSTRUCTION. Everything a case-selection or
        # checkpoint-selection decision may look at comes from here, which is what
        # leaves the final protocol fresh.
        "screening" => screening_seed === nothing ? nothing : Dict{String,Any}(
            "seed" => Int(screening_seed),
            "num_stages" => protocol_stages,
            "num_scenarios" => Int(screening_scenarios),
            "excludes" => "protocol",
            "sha256" => protocol_digest(demand, protocol_stages, Int(screening_scenarios);
                                        seed = Int(screening_seed),
                                        exclude = protocol_columns(
                                            scenario_index_matrix(demand, protocol_stages,
                                                                  protocol_scenarios))),
        ),
        "artifacts" => Dict{String,Any}(
            "network.json" => network_sha,
            "batteries.json" => batteries_sha,
            "demand.json" => demand_sha,
        ),
    )
    write_canonical_json(joinpath(dir, "case_manifest.json"), manifest)
    return manifest
end

"""
    read_battery_case(dir; verify=true) -> BatteryCase

Read a frozen case from `dir`.

# Keywords
- `verify::Bool`: when `true` (the default) every artifact hash, schema tag,
  stage duration, support digest and protocol digest recorded in the manifest is
  re-checked against the bytes on disk before anything is returned.

# Notes
Verification is on by default and failures are ERRORS, never warnings: an
engine that proceeds on a case it could not verify is producing numbers that
cannot be compared with the other engine's.
"""
function read_battery_case(dir::AbstractString; verify::Bool = true)
    manifest_path = joinpath(dir, "case_manifest.json")
    isfile(manifest_path) || error("no case_manifest.json in $dir")
    manifest = JSON.parsefile(manifest_path)
    manifest["schema"] == BATTERY_MANIFEST_SCHEMA ||
        error("unexpected manifest schema $(manifest["schema"]); expected $BATTERY_MANIFEST_SCHEMA")

    if verify
        for (file, want) in manifest["artifacts"]
            path = joinpath(dir, file)
            isfile(path) || error("case artifact $file missing from $dir")
            got = sha256_file(path)
            got == want || error("case artifact $file has SHA-256 $got but the manifest records $want")
        end
    end

    network_obj = JSON.parsefile(joinpath(dir, "network.json"))
    network_obj["schema"] == BATTERY_NETWORK_SCHEMA ||
        error("unexpected network schema $(network_obj["schema"])")
    network = plain(network_obj["data"])::Dict{String,Any}

    batteries_obj = JSON.parsefile(joinpath(dir, "batteries.json"))
    batteries_obj["schema"] == BATTERY_BATTERY_SCHEMA ||
        error("unexpected batteries schema $(batteries_obj["schema"])")
    batteries = BatterySpec[
        BatterySpec(Int(b["index"]), Int(b["bus"]),
                    Float64(b["energy_min"]), Float64(b["energy_max"]),
                    Float64(b["energy_initial"]),
                    Float64(b["charge_max"]), Float64(b["discharge_max"]),
                    Float64(b["charge_efficiency"]), Float64(b["discharge_efficiency"]),
                    Float64(b["self_discharge"]), Float64(b["throughput_cost"]))
        for b in batteries_obj["batteries"]]
    sort!(batteries; by = b -> b.index)
    recourse = RecourseCosts(Float64(batteries_obj["recourse"]["deficit_cost"]),
                             Float64(batteries_obj["recourse"]["surplus_cost"]))

    demand_obj = JSON.parsefile(joinpath(dir, "demand.json"))
    demand_obj["schema"] == BATTERY_DEMAND_SCHEMA ||
        error("unexpected demand schema $(demand_obj["schema"])")
    load_ids = Int.(demand_obj["load_ids"])
    T = Int(demand_obj["horizon"])
    n = length(load_ids)
    profile = Matrix{Float64}(undef, n, T)
    for t in 1:T
        col = Float64.(demand_obj["profile"][t])
        length(col) == n ||
            error("demand.json profile row $t has $(length(col)) entries, expected $n")
        profile[:, t] .= col
    end
    atoms = Vector{Matrix{Float64}}(undef, T)
    probs = Vector{Vector{Float64}}(undef, T)
    for t in 1:T
        raw = demand_obj["atoms"][t]
        K = length(raw)
        A = Matrix{Float64}(undef, n, K)
        for k in 1:K
            col = Float64.(raw[k])
            length(col) == n ||
                error("demand.json stage $t atom $k has $(length(col)) entries, expected $n")
            A[:, k] .= col
        end
        atoms[t] = A
        probs[t] = Float64.(demand_obj["probabilities"][t])
    end
    demand = DemandSupport(Float64(demand_obj["stage_hours"]), T, load_ids,
                           profile, atoms, probs,
                           Int(demand_obj["protocol_seed"]),
                           plain(get(demand_obj, "source", Dict{String,Any}())))

    # ── Fail-closed contract checks ─────────────────────────────────────────
    # Each of these has a documented failure mode behind it; none is cosmetic.
    validate_support(demand)
    demand.stage_hours == Float64(manifest["stage_hours"]) ||
        error("demand.json stage_hours $(demand.stage_hours) disagrees with manifest $(manifest["stage_hours"])")
    network_load_ids = sort!([Int(l["index"]) for (_, l) in network["load"]])
    demand.load_ids == network_load_ids ||
        error("demand support covers loads $(demand.load_ids) but the network has $(network_load_ids)")
    bus_ids = Set(Int(b["index"]) for (_, b) in network["bus"])
    for b in batteries
        b.bus in bus_ids || error("battery $(b.index) sits at bus $(b.bus), which is not in the network")
        0 < b.charge_efficiency <= 1 || error("battery $(b.index): charge_efficiency out of (0,1]")
        0 < b.discharge_efficiency <= 1 || error("battery $(b.index): discharge_efficiency out of (0,1]")
        0 < b.self_discharge <= 1 || error("battery $(b.index): self_discharge out of (0,1]")
        b.energy_min <= b.energy_initial <= b.energy_max ||
            error("battery $(b.index): initial energy $(b.energy_initial) outside [$(b.energy_min), $(b.energy_max)]")
        b.charge_max >= 0 && b.discharge_max >= 0 ||
            error("battery $(b.index): negative power rating")
        b.throughput_cost >= 0 || error("battery $(b.index): negative throughput cost")
    end
    allunique(b.index for b in batteries) || error("battery indices are not unique")
    recourse.deficit > 0 && recourse.surplus > 0 ||
        error("recourse prices must be strictly positive; got $(recourse)")
    recourse.deficit == Float64(manifest["recourse"]["deficit_cost"]) &&
        recourse.surplus == Float64(manifest["recourse"]["surplus_cost"]) ||
        error("batteries.json recourse prices disagree with the manifest")
    if verify
        want_support = manifest["support"]["sha256"]
        got_support = support_digest(demand)
        got_support == want_support ||
            error("regenerated support digest $got_support does not match the manifest's $want_support")
        want = manifest["protocol"]["sha256"]
        got = protocol_digest(demand, Int(manifest["protocol"]["num_stages"]),
                              Int(manifest["protocol"]["num_scenarios"]))
        got == want ||
            error("regenerated protocol digest $got does not match the manifest's $want")
        scr = get(manifest, "screening", nothing)
        if scr !== nothing
            wants = scr["sha256"]
            # The screening protocol is regenerated exactly as it was written:
            # from its own seed, excluding the final protocol's columns when the
            # record says it does. Regenerating it without the exclusion would
            # silently pass on every case where no repair was needed and fail
            # only on the small supports where the property actually bites.
            ex = get(scr, "excludes", nothing) == "protocol" ?
                 protocol_columns(scenario_index_matrix(demand,
                                                        Int(manifest["protocol"]["num_stages"]),
                                                        Int(manifest["protocol"]["num_scenarios"]))) :
                 nothing
            gots = protocol_digest(demand, Int(scr["num_stages"]),
                                   Int(scr["num_scenarios"]); seed = Int(scr["seed"]),
                                   exclude = ex)
            gots == wants ||
                error("regenerated screening digest $gots does not match the manifest's $wants")
        end
    end

    return BatteryCase(String(dir), String(manifest["case"]), network, batteries,
                       recourse, demand, manifest)
end

"""
    _sampler_kind(d) -> String

A one-line name for a recorded authoring sampler.

# Notes
The full description is a nested dictionary that can run to thousands of
characters on a stage-dependent regional sampler. It stays in the artifact, where
it belongs; a case summary that scrolled it off the screen would be worse than
useless.
"""
function _sampler_kind(d)
    d isa AbstractDict || return "(unrecorded)"
    kind = String(get(d, "sampler", "?"))
    kind == "product" && return "product(" *
        join([_sampler_kind(c) for c in get(d, "components", [])], " × ") * ")"
    if kind == "stage"
        inner = sort!(unique([_sampler_kind(v) for (_, v) in get(d, "stages", Dict())]))
        return "stage[" * join(inner, "|") * "]"
    end
    kind == "group" && return "group(" * string(length(get(d, "groups", []))) * " regions)"
    return kind
end

"""
    describe(case::BatteryCase) -> String

One-screen human summary of a frozen case: counts, stage duration, battery
ratings and the demand support.
"""
function describe(case::BatteryCase)
    s = case.demand
    io = IOBuffer()
    println(io, "battery case: ", case.name, "  (", case.dir, ")")
    @printf(io, "  buses %d  gens %d  branches %d  loads %d  baseMVA %.1f\n",
            length(case.network["bus"]), length(case.network["gen"]),
            length(case.network["branch"]), length(case.network["load"]),
            Float64(case.network["baseMVA"]))
    @printf(io, "  stage duration %.4f h   horizon %d   loads in support %d\n",
            s.stage_hours, s.horizon, num_loads(s))
    ks = [num_atoms(s, t) for t in 1:s.horizon]
    @printf(io, "  atoms per stage: min %d  max %d   support sha %s\n",
            minimum(ks), maximum(ks), support_digest(s)[1:16])
    @printf(io, "  profile range over stages: [%.4f, %.4f]\n",
            minimum(s.profile), maximum(s.profile))
    @printf(io, "  authoring sampler: %s   (freeze %s, seed %s)\n",
            _sampler_kind(get(s.source, "sampler", nothing)),
            get(s.source, "method", "?"), string(get(s.source, "seed", "?")))
    @printf(io, "  recourse prices: deficit %.1f  surplus %.1f (per pu per stage)\n",
            case.recourse.deficit, case.recourse.surplus)
    for b in case.batteries
        @printf(io, "  battery %d @ bus %-4d e∈[%.4f, %.4f] e0=%.4f  pch≤%.4f pdis≤%.4f  η=(%.3f,%.3f) α=%.4f c_deg=%.4f\n",
                b.index, b.bus, b.energy_min, b.energy_max, b.energy_initial,
                b.charge_max, b.discharge_max,
                b.charge_efficiency, b.discharge_efficiency,
                b.self_discharge, b.throughput_cost)
    end
    return String(take!(io))
end

"""
    evaluation_protocol(case) -> (matrix, kind)

The protocol a policy may be SELECTED on, and which one it is.

# Returns
- `matrix::Matrix{Int}`: the `(stages × scenarios)` atom-index matrix.
- `kind::Symbol`: `:screening` when the case declares a screening protocol,
  `:sole` when it declares only one protocol and therefore has no final/screening
  split at all.

# Notes
**Selection may never touch the final protocol.** A case of the study's panel
declares two: a large final one, drawn from the support's own `protocol_seed`
and evaluated ONCE after every selection is made, and a small screening one from
an independent seed, repaired against the final one so the two share no scenario
by construction. Regenerating the screening protocol therefore has to regenerate
the final one's COLUMNS as the exclusion set — which is index arithmetic on the
frozen support, exactly what `read_battery_case` already does on every load, and
not an evaluation of anything.

An earlier revision of this file read `manifest["protocol"]` here. That is the
FINAL protocol, so checkpoint selection was scoring policies on the very panel
that exists to be fresh. The defect was silent — the columns solve, the costs are
finite and the numbers look like a panel — which is why the protocol's kind is
returned beside the matrix, recorded in the checkpoint and printed by the
trainer, rather than left as something a reader has to re-derive.

`:sole` is reachable only on a case built without a screening protocol at all —
the small correctness fixture this package's regression suite runs on. Every
panel case has the split, so a study run cannot land there. The digest is
re-verified against the manifest either way.
"""
function evaluation_protocol(case::BatteryCase)
    scr = get(case.manifest, "screening", nothing)
    if scr === nothing
        stages = Int(case.manifest["protocol"]["num_stages"])
        scen = Int(case.manifest["protocol"]["num_scenarios"])
        m = scenario_index_matrix(case.demand, stages, scen)
        protocol_digest(case.demand, stages, scen) == case.manifest["protocol"]["sha256"] ||
            error("regenerated protocol digest does not match the manifest's")
        return m, :sole
    end
    ex = get(scr, "excludes", nothing) == "protocol" ?
         protocol_columns(scenario_index_matrix(case.demand,
                                                Int(case.manifest["protocol"]["num_stages"]),
                                                Int(case.manifest["protocol"]["num_scenarios"]))) :
         nothing
    stages, scen, seed = Int(scr["num_stages"]), Int(scr["num_scenarios"]), Int(scr["seed"])
    m = scenario_index_matrix(case.demand, stages, scen; seed = seed, exclude = ex)
    protocol_digest(case.demand, stages, scen; seed = seed, exclude = ex) == scr["sha256"] ||
        error("regenerated screening digest does not match the manifest's")
    return m, :screening
end
