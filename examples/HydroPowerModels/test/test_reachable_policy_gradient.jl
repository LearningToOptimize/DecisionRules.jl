# Regression tests for the DIFFERENTIABLE reachable-target policy.
#
# Every test here fails under the previous implementation, in which
# `_hydro_reachable_bounds` and `_cascade_upper_bounds` carried
# `ChainRulesCore.@non_differentiable`:
#
#   * the reachable interval's dependence on the previous reservoir state was
#     dropped, so the adjoint recursion was truncated at EVERY stage — measured
#     at the production operating point as cosine 0.673 / norm ratio 0.059
#     against the true gradient;
#   * `_cascade_upper_bounds` was a mutating `fill`/`upper[d] = ...` loop that
#     reverse-mode AD cannot traverse at all, so removing the annotation without
#     rewriting it would have raised "Mutating arrays is not supported".
#
# The tests use a small synthetic three-reservoir cascade defined in
# `hydro_reachable_reference.jl`, not the 126-stage production problem: the
# defect is per-stage and per-coordinate, so it is fully exposed by one stage.
#
# Usage:
#   julia --project=examples/HydroPowerModels examples/HydroPowerModels/test_reachable_policy_gradient.jl

using Test
using Random
using LinearAlgebra
using Flux
using Zygote
using ChainRulesCore
using DecisionRules

const SCRIPT_DIR = dirname(dirname(@__FILE__))   # examples/HydroPowerModels
include(joinpath(SCRIPT_DIR, "hydro_reachable_policy.jl"))
include(joinpath(SCRIPT_DIR, "hydro_reachable_reference.jl"))
const REF = HydroReachableReference

"""
    build_policy(; spill_max = nothing, seed = 20260802) -> HydroReachablePolicy

A small reachable policy over the synthetic reference cascade.

Deliberately tiny (a 4-unit LSTM and a linear head): these tests exercise the
PHYSICS path, and network size only slows them down.
"""
function build_policy(; spill_max = nothing, seed = 20260802)
    Random.seed!(seed)
    return hydro_reachable_policy(
        REF.hydro_meta(), [4];
        spill_max = spill_max,
    )
end

"""
    target_map(policy, inflow, x_prev, y; freeze_bounds) -> Vector

The policy's target map with the head output `y` held FIXED, so the only path
from `x_prev` to the target is the physics one under test.

# Keywords
- `freeze_bounds::Bool`: when `true`, `x_prev` is passed to the bounds and the
  cascade clamp through `ChainRulesCore.ignore_derivatives`, reproducing exactly
  what the removed `@non_differentiable` annotations did. The FORWARD value is
  identical either way; only the pullback differs.
"""
function target_map(policy, inflow, x_prev, y; freeze_bounds::Bool)
    xb = freeze_bounds ? ChainRulesCore.ignore_derivatives(x_prev) : x_prev
    lower, upper = _hydro_reachable_bounds(policy, inflow, xb)
    raw = lower .+ (upper .- lower) .* y
    isempty(policy.cascade) && return raw
    return min.(raw, _cascade_upper_bounds(policy, raw, inflow, xb))
end

"""
    gradient_or_zeros(f, x) -> Vector

`Zygote.gradient(f, x)`, with a `nothing` result materialized as zeros.

Zygote returns `nothing` — not a zero vector — when NO differentiable path
reaches `x`. That is precisely the outcome the removed annotations produced, so
the distinction must be handled rather than crashed on.
"""
function gradient_or_zeros(f, x)
    g = only(Zygote.gradient(f, x))
    return g === nothing ? zeros(eltype(x), length(x)) : collect(g)
end

@testset "reachable policy — differentiable bounds" begin

    # ── 1. Forward parity on representative cascade metadata ─────────────────
    # The non-mutating rewrite must reproduce the original mutating loop's
    # VALUES, including the `Inf` column for a reservoir with no incoming link
    # and the turbine-capped `min` branch. `REF.PARITY_RTOL` is the Float32
    # resolution at which the policy stores its hydro metadata.
    @testset "forward parity against the original mutating loop" begin
        policy = build_policy()
        @test length(policy.cascade) == length(REF.cascade_links())
        for point in REF.OPERATING_POINTS, y in REF.Y_POINTS
            x_prev, inflow = point.x_prev, point.inflow

            lower, upper = _hydro_reachable_bounds(policy, inflow, x_prev)
            ref_lower, ref_upper = REF.reachable_bounds(x_prev, inflow)
            @test lower ≈ ref_lower rtol = REF.PARITY_RTOL
            @test upper ≈ ref_upper rtol = REF.PARITY_RTOL

            raw = lower .+ (upper .- lower) .* y
            cascade_upper = _cascade_upper_bounds(policy, raw, inflow, x_prev)
            @test cascade_upper ≈ REF.cascade_upper_reference(raw, inflow, x_prev) rtol = REF.PARITY_RTOL
            # Reservoir 1 receives no link: its bound must be exactly `Inf`, so
            # that `min(raw, Inf)` leaves it untouched.
            @test isinf(cascade_upper[1])

            emitted = min.(raw, cascade_upper)
            @test emitted ≈ REF.emitted_target(x_prev, inflow, y) rtol = REF.PARITY_RTOL
            REF.check_invariants(emitted, lower, upper)
        end
    end

    # A limited-spill policy exercises the state-dependent LOWER bound too.
    @testset "forward parity with a finite spill cap" begin
        spill_max = [0.4, 0.3, 0.2]
        policy = build_policy(; spill_max = spill_max)
        for point in REF.OPERATING_POINTS
            lower, upper = _hydro_reachable_bounds(policy, point.inflow, point.x_prev)
            ref_lower, ref_upper =
                REF.reachable_bounds(point.x_prev, point.inflow; spill_max = spill_max)
            @test lower ≈ ref_lower rtol = REF.PARITY_RTOL
            @test upper ≈ ref_upper rtol = REF.PARITY_RTOL
        end
    end

    # ── 2. Finite differences through a state-dependent upper bound ───────────
    # The upper bound is `min(max_vol, x_prev + K w - K min_turn + upstream)`.
    # At these operating points it is OFF the ceiling, so d(upper)/d(x_prev) = I.
    # Under the removed annotation the AD gradient here was exactly zero.
    @testset "finite differences — upper bound" begin
        policy = build_policy()
        weights = [0.7, -1.3, 0.4]
        for point in REF.OPERATING_POINTS
            inflow = point.inflow
            f = x -> sum(weights .* _hydro_reachable_bounds(policy, inflow, x)[2])
            ad = gradient_or_zeros(f, point.x_prev)
            @test ad ≈ REF.central_gradient(f, point.x_prev) atol = 1e-6
            # Analytically: unclipped upper => identity Jacobian => grad = weights.
            @test ad ≈ weights atol = 1e-9
        end
    end

    # Where the ceiling BINDS the derivative through the bound is genuinely
    # zero. That is the correct derivative of the executed map, not a dropped
    # term, and it must survive the repair.
    @testset "finite differences — clipped upper bound" begin
        policy = build_policy()
        point = REF.CLIPPED_POINT
        _, upper = _hydro_reachable_bounds(policy, point.inflow, point.x_prev)
        @test upper[1] == REF.MAX_VOL[1]                   # reservoir 1 is clipped
        f = x -> _hydro_reachable_bounds(policy, point.inflow, x)[2][1]
        ad = gradient_or_zeros(f, point.x_prev)
        @test ad[1] == 0.0
        @test ad ≈ REF.central_gradient(f, point.x_prev) atol = 1e-6
    end

    # ── 3. Nonzero derivative through the previous reservoir state ────────────
    # This is the whole defect: with the head output held constant, the OLD
    # implementation produced NO gradient path from the previous state at all,
    # while the true derivative is diag(y) on unclipped coordinates.
    # `freeze_bounds = true` reproduces the old behaviour exactly, so the test
    # asserts both the correct value and the fact that it differs from what the
    # old code returned.
    #
    # Finite differences are compared only at points certified away from every
    # kink of the piecewise-affine map — a central difference straddling a kink
    # measures neither one-sided derivative.
    @testset "derivative through the previous reservoir state" begin
        policy = build_policy()
        multipliers = [1.1, -0.6, 0.9]
        for point in REF.SMOOTH_POINTS
            inflow, x_prev, y = point.inflow, point.x_prev, point.y
            @test REF.kink_margin(x_prev, inflow, y) > REF.MIN_KINK_MARGIN

            live = x -> sum(multipliers .* target_map(policy, inflow, x, y; freeze_bounds = false))
            frozen = x -> sum(multipliers .* target_map(policy, inflow, x, y; freeze_bounds = true))

            # Forward values are IDENTICAL; only the pullback differs.
            @test live(x_prev) == frozen(x_prev)

            g_live = gradient_or_zeros(live, x_prev)
            g_frozen = gradient_or_zeros(frozen, x_prev)

            @test g_live ≈ REF.central_gradient(live, x_prev) atol = 1e-5
            @test norm(g_live) > 1e-3                       # load-bearing, not noise
            @test all(iszero, g_frozen)                     # what the old code gave
            @test !isapprox(g_live, g_frozen; atol = 1e-6)  # the repair is not inert
        end
    end

    # ── 4. Gradient through a BINDING cascade clamp ──────────────────────────
    # At SMOOTH_POINTS[1] the 2->3 clamp binds AND the implied upstream release
    # is strictly positive, so the downstream target inherits
    # d/d(upstream target) = -1 through `release = K w + x - target`. Reservoir 1
    # is not upstream of 3 and must carry no gradient.
    @testset "gradient through a binding cascade clamp" begin
        policy = build_policy()
        point = REF.SMOOTH_POINTS[1]
        inflow, x_prev, y = point.inflow, point.x_prev, point.y

        lower, upper = _hydro_reachable_bounds(policy, inflow, x_prev)
        raw = lower .+ (upper .- lower) .* y
        cascade_upper = _cascade_upper_bounds(policy, raw, inflow, x_prev)
        @test cascade_upper[3] < raw[3]              # the 2->3 clamp binds
        @test raw[2] < cascade_upper[2]              # the 1->2 clamp does not

        f = t -> _cascade_upper_bounds(policy, t, inflow, x_prev)[3]
        ad = gradient_or_zeros(f, raw)
        @test ad[2] ≈ -1.0 atol = 1e-9
        @test ad[1] == 0.0                           # not an upstream of 3
        @test ad ≈ REF.central_gradient(f, raw) atol = 1e-6

        # At SMOOTH_POINTS[2] the 1->2 link sits strictly AT its turbine cap, so
        # the same derivative is correctly zero — the `min` branch, not a
        # dropped term.
        capped = REF.SMOOTH_POINTS[2]
        lower2, upper2 = _hydro_reachable_bounds(policy, capped.inflow, capped.x_prev)
        raw2 = lower2 .+ (upper2 .- lower2) .* capped.y
        g = t -> _cascade_upper_bounds(policy, t, capped.inflow, capped.x_prev)[2]
        ad2 = gradient_or_zeros(g, raw2)
        @test ad2[1] == 0.0
        @test ad2 ≈ REF.central_gradient(g, raw2) atol = 1e-6
    end

    # ── 5. Reachability and box invariants over random draws ─────────────────
    @testset "reachability and box invariants" begin
        policy = build_policy()
        rng = MersenneTwister(4242)
        for _ in 1:200
            x_prev = REF.MIN_VOL .+ rand(rng, REF.NHYD) .* (REF.MAX_VOL .- REF.MIN_VOL)
            inflow = 5 .* rand(rng, REF.NHYD)
            y = rand(rng, REF.NHYD)
            lower, upper = _hydro_reachable_bounds(policy, inflow, x_prev)
            @test all(lower .<= upper)
            REF.check_invariants(
                target_map(policy, inflow, x_prev, y; freeze_bounds = false),
                lower, upper,
            )
        end
    end

    # ── 6. The real forward pass still runs, and its gradient is finite ──────
    # Exercises the policy end to end (encoder, head, bounds, cascade) in the
    # Float32 precision production uses. The old MAIN implementation could not
    # reach this point once the annotations were removed: the mutating cascade
    # loop raised "Mutating arrays is not supported".
    @testset "end-to-end policy gradient" begin
        policy = build_policy()
        Flux.reset!(policy)
        input = Float32[0.8, 1.2, 0.4, 1.0, 0.5, 0.25]   # [inflow; x_prev]
        multipliers = Float32[1.1, -0.6, 0.9]

        emitted = policy(input)
        @test length(emitted) == REF.NHYD
        @test all(isfinite, emitted)

        Flux.reset!(policy)
        grads = Zygote.gradient(policy) do m
            sum(multipliers .* m(input))
        end
        parameter_grads = collect(Iterators.filter(
            !isnothing,
            (g for g in Flux.trainables(only(grads))),
        ))
        @test !isempty(parameter_grads)
        @test all(g -> all(isfinite, g), parameter_grads)
        @test any(g -> any(!iszero, g), parameter_grads)
    end
end
