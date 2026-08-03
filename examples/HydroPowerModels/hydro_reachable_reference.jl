# Engine-independent reference for the hydro reachable-target map.
#
# This file contains NO package code. It defines a small synthetic cascade and
# recomputes the reachable bounds, the cascade clamp and the emitted target from
# the equations in the paper, in plain Julia. It is included by BOTH
# `DecisionRules.jl` and `DecisionRulesExa.jl` test files, so asserting each
# engine against it establishes CPU parity between the two implementations
# without either package having to load the other.
#
# `cascade_upper_reference` is deliberately the ORIGINAL mutating loop
# (`fill(Inf, n)` then `upper[d] = min(upper[d], ...)`). Reverse-mode AD cannot
# traverse that loop, which is why both engines were rewritten to a non-mutating
# reduction — but the rewrite must not change the VALUE, and this is what pins
# it down.

module HydroReachableReference

using Test

# ── Synthetic case ────────────────────────────────────────────────────────────
# Three reservoirs, two cascade links, chosen to exercise every branch:
#
#   1 --turbine only--> 2 --turbine + spill--> 3
#
# Reservoir 1 has no incoming link (constant `Inf` branch of the clamp);
# link 1->2 is turbine-capped (the `min` branch); link 2->3 is not (the plain
# `max(0, R)` branch). Capacities are large enough that the reachable upper
# bound is OFF its `max_vol` ceiling at the operating points below, which is the
# regime in which the upper bound depends on the previous state.
const NHYD = 3
const K = 0.6048                       # 0.0036 * 168, the frozen weekly value
const MIN_VOL = [0.0, 0.0, 0.0]
const MAX_VOL = [10.0, 6.0, 4.0]
const MIN_TURN = [0.0, 0.0, 0.0]
const MAX_TURN = [2.0, 3.0, 1.5]

# `upstream_turn[r]` / `upstream_spill[r]`: (upstream position, its max_turn).
const UPSTREAM_TURN = [Tuple{Int,Float64}[], [(1, 2.0)], [(2, 3.0)]]
const UPSTREAM_SPILL = [Tuple{Int,Float64}[], Tuple{Int,Float64}[], [(2, 3.0)]]

"""
    hydro_meta() -> NamedTuple

Metadata in the shape both engines' `hydro_reachable_policy` constructors
expect.
"""
hydro_meta() = (
    nHyd = NHYD,
    min_vol = MIN_VOL,
    max_vol = MAX_VOL,
    min_turn = MIN_TURN,
    max_turn = MAX_TURN,
    K = K,
    upstream_turn = UPSTREAM_TURN,
    upstream_spill = UPSTREAM_SPILL,
)

"""
    upstream_max() -> Vector{Float64}

`upstream_max[r] = sum over upstream u of K * max_turn_u`, the maximum
same-stage cascade contribution reservoir `r` can receive.
"""
upstream_max() = [sum(K * m for (_, m) in list; init = 0.0) for list in UPSTREAM_TURN]

"""
    reachable_bounds(x_prev, inflow; spill_max = nothing) -> (lower, upper)

One-stage reachable reservoir interval, from first principles.

Upper: release as little as possible (`min_turn`, no spill) and receive as much
upstream water as physically possible, capped at `max_vol`.
Lower: release as much as possible (`max_turn` plus `spill_max`), floored at
`min_vol`. With `spill_max === nothing` spillage is unlimited, so the lower
bound is `min_vol`.

# Arguments
- `x_prev::AbstractVector`: reservoir volumes entering the stage.
- `inflow::AbstractVector`: per-reservoir stage inflow.

# Keywords
- `spill_max`: per-reservoir spill cap, or `nothing` for unlimited spillage.

# Returns
- `(lower, upper)`: two length-`NHYD` vectors with `lower <= upper` everywhere.
"""
function reachable_bounds(x_prev, inflow; spill_max = nothing)
    upper_raw = x_prev .+ K .* inflow .- K .* MIN_TURN .+ upstream_max()
    upper = min.(MAX_VOL, upper_raw)
    lower = if spill_max === nothing
        MIN_VOL
    else
        max.(MIN_VOL, x_prev .+ K .* inflow .- K .* MAX_TURN .- spill_max)
    end
    return lower, max.(upper, lower)
end

"""
    cascade_links() -> Vector{NamedTuple}

The cascade link list both engines derive from `UPSTREAM_TURN`/`UPSTREAM_SPILL`:
one link per turbine connection, marked `turn_only` when the same pair has no
spill connection, and carrying the turbine-only contribution cap `K * max_turn`.
"""
function cascade_links()
    spill_dests = Dict{Int,Set{Int}}()
    for (r, list) in enumerate(UPSTREAM_SPILL), (u, _) in list
        push!(get!(spill_dests, u, Set{Int}()), r)
    end
    links = NamedTuple{(:downstream, :upstream, :turn_only, :K_max_turn),
                       Tuple{Int,Int,Bool,Float64}}[]
    for (r, list) in enumerate(UPSTREAM_TURN), (u, u_max_turn) in list
        has_spill = haskey(spill_dests, u) && r in spill_dests[u]
        push!(links, (downstream = r, upstream = u, turn_only = !has_spill,
                      K_max_turn = K * u_max_turn))
    end
    for (r, list) in enumerate(UPSTREAM_SPILL), (u, _) in list
        any(l -> l.downstream == r && l.upstream == u, links) && continue
        push!(links, (downstream = r, upstream = u, turn_only = false,
                      K_max_turn = K * MAX_TURN[u]))
    end
    return links
end

"""
    cascade_upper_reference(target, inflow, x_prev) -> Vector{Float64}

The ORIGINAL mutating cascade-clamp loop, kept verbatim as the value oracle.

Both engines replaced it with a non-mutating reduction so that reverse-mode AD
can traverse it. That rewrite must be value-preserving, and this function is
what proves it.

# Returns
- `Vector{Float64}`: per-reservoir upper bound; `Inf` where no link arrives.
"""
function cascade_upper_reference(target, inflow, x_prev)
    upper = fill(Inf, NHYD)
    for link in cascade_links()
        u, d = link.upstream, link.downstream
        release = K * inflow[u] + x_prev[u] - target[u]
        contribution = link.turn_only ?
            min(link.K_max_turn, max(0.0, release)) : max(0.0, release)
        true_upper = x_prev[d] + K * inflow[d] - K * MIN_TURN[d] + contribution
        upper[d] = min(upper[d], min(MAX_VOL[d], true_upper))
    end
    return upper
end

"""
    emitted_target(x_prev, inflow, y; spill_max = nothing) -> Vector{Float64}

The full reference target map: scale the normalized head output `y` onto the
reachable interval, then apply the cascade clamp.

# Arguments
- `x_prev`, `inflow`: stage state and inflow.
- `y::AbstractVector`: normalized head output, one entry per reservoir in
  `[0, 1]`.
"""
function emitted_target(x_prev, inflow, y; spill_max = nothing)
    lower, upper = reachable_bounds(x_prev, inflow; spill_max = spill_max)
    raw = lower .+ (upper .- lower) .* y
    return min.(raw, cascade_upper_reference(raw, inflow, x_prev))
end

# ── Fixed operating points ────────────────────────────────────────────────────
# `OPERATING_POINTS` keeps the upper bound strictly below `max_vol`, so the
# reachable interval is genuinely state-dependent everywhere and the derivative
# through it is nonzero. `CLIPPED_POINT` pushes reservoir 1 onto its `max_vol`
# ceiling, where the correct derivative through the upper bound is zero.
const OPERATING_POINTS = [
    (x_prev = [1.0, 0.5, 0.25], inflow = [0.8, 1.2, 0.4]),
    (x_prev = [3.0, 2.0, 1.0], inflow = [2.5, 0.3, 1.7]),
    (x_prev = [0.0, 0.0, 0.0], inflow = [4.0, 3.0, 2.0]),
]
const CLIPPED_POINT = (x_prev = [9.9, 0.5, 0.25], inflow = [5.0, 1.2, 0.4])
const Y_POINTS = [
    [0.25, 0.60, 0.85],
    [0.05, 0.50, 0.95],
    [0.999, 0.001, 0.500],
]

# Both engines store the policy's hydro metadata in Float32 (the precision the
# GPU training path runs in), while this reference computes in exact Float64.
# Values therefore agree to Float32 resolution, not to Float64 resolution, and
# every value comparison against the reference uses this tolerance. It is ~8x
# Float32 eps and ~14 orders of magnitude tighter than anything physically
# meaningful in the case.
const PARITY_RTOL = 1e-6

"""
    kink_margin(x_prev, inflow, y; spill_max = nothing) -> Float64

Distance from the operating point to the NEAREST non-differentiable kink of the
target map.

The map is piecewise affine: `min`/`max` against `max_vol`, `min_vol`, zero
release, the turbine-only contribution cap, and the cascade clamp itself. Each
introduces a kink where a finite difference straddles two different linear
pieces and therefore disagrees with the (correct) one-sided derivative.

Finite-difference tests must run at points whose margin comfortably exceeds the
difference step; this function is what makes that a checked property rather than
an assumption.

# Returns
- `Float64`: the smallest absolute gap to any switching surface.
"""
function kink_margin(x_prev, inflow, y; spill_max = nothing)
    up_max = upstream_max()
    upper_raw = x_prev .+ K .* inflow .- K .* MIN_TURN .+ up_max
    margins = Float64[]
    append!(margins, abs.(MAX_VOL .- upper_raw))               # upper vs max_vol
    lower, upper = reachable_bounds(x_prev, inflow; spill_max = spill_max)
    append!(margins, abs.(upper .- lower))                      # upper vs lower
    if spill_max !== nothing
        lower_raw = x_prev .+ K .* inflow .- K .* MAX_TURN .- spill_max
        append!(margins, abs.(MIN_VOL .- lower_raw))            # lower vs min_vol
    end
    raw = lower .+ (upper .- lower) .* y
    for link in cascade_links()
        u, d = link.upstream, link.downstream
        release = K * inflow[u] + x_prev[u] - raw[u]
        push!(margins, abs(release))                            # release vs zero
        link.turn_only && push!(margins, abs(link.K_max_turn - max(0.0, release)))
        contribution = link.turn_only ?
            min(link.K_max_turn, max(0.0, release)) : max(0.0, release)
        link_upper = x_prev[d] + K * inflow[d] - K * MIN_TURN[d] + contribution
        push!(margins, abs(MAX_VOL[d] - link_upper))            # link vs max_vol
        push!(margins, abs(raw[d] - min(MAX_VOL[d], link_upper)))  # clamp binding
    end
    return minimum(margins)
end

# Operating points chosen to be FAR from every kink above, so a central finite
# difference measures the derivative of one linear piece. `check_smoothness`
# below asserts the margin rather than trusting the choice.
#
#   S1  every release strictly positive, the turbine cap slack, and the cascade
#       clamp strictly BINDING on reservoir 3 — the regime in which the clamp's
#       derivative with respect to the upstream target is -1.
#   S2  the 1->2 link strictly AT its turbine cap and no clamp binding — the
#       regime in which that derivative is correctly zero.
#   S3  an interior point with no branch near a switch.
const SMOOTH_POINTS = [
    (x_prev = [1.0, 0.5, 0.25], inflow = [0.8, 1.2, 0.4], y = [0.35, 0.10, 0.90]),
    (x_prev = [3.0, 2.0, 1.00], inflow = [2.5, 0.3, 1.7], y = [0.35, 0.10, 0.20]),
    (x_prev = [0.5, 1.0, 0.50], inflow = [1.0, 0.5, 1.0], y = [0.50, 0.40, 0.60]),
]
# A central difference with h = 1e-6 needs a margin far larger than h; 1e-2 is
# four orders above it and still easy to satisfy.
const MIN_KINK_MARGIN = 1e-2

"""
    central_difference(f, x, i; h) -> Float64

Central finite difference of scalar `f` in coordinate `i` of vector `x`.

Used instead of a finite-difference package so the regression tests depend only
on what the example environments already carry.
"""
function central_difference(f, x, i; h = 1e-6)
    xp = copy(x); xp[i] += h
    xm = copy(x); xm[i] -= h
    return (f(xp) - f(xm)) / (2h)
end

"""
    central_gradient(f, x; h) -> Vector{Float64}

Coordinate-wise central-difference gradient of scalar `f` at `x`.
"""
central_gradient(f, x; h = 1e-6) =
    [central_difference(f, x, i; h = h) for i in eachindex(x)]

"""
    check_invariants(target, lower, upper) -> Nothing

Assert the two guarantees the policy exists to provide: the emitted target lies
inside the one-stage reachable interval, and inside the physical volume box.

The cascade clamp can only LOWER a target, so it cannot break the upper
invariant; it can push a target below `lower` only in the documented physically
infeasible case, which these operating points do not produce.
"""
function check_invariants(target, lower, upper)
    @test all(target .>= lower .- 1e-9)
    @test all(target .<= upper .+ 1e-9)
    @test all(target .>= MIN_VOL .- 1e-9)
    @test all(target .<= MAX_VOL .+ 1e-9)
    return nothing
end

end # module
