"""
True optimal expected cost for the SULS problem via backward dynamic programming.

The DP value V_1(I_0) is the exact expected optimal cost for the 12-stage
stochastic uncapacitated lot-sizing problem with d ~ U[10,30].

Run:
  julia --project=examples/inventory_control examples/inventory_control/solve_optimal_dp.jl
"""

using CSV, DataFrames, Statistics, Random, Printf

# ── Problem parameters ────────────────────────────────────────────────────────
const T      = 12
const K      = 30.0
const c      = 2.0
const h      = 1.0
const p      = 10.0
const Q_MAX  = 80.0
const I_0    = 20.0
const D_MIN  = 10.0
const D_MAX  = 30.0

# ── DP discretisation ─────────────────────────────────────────────────────────
# State grid: wide enough that the optimal policy never hits the boundary.
const S_LO   = -100.0
const S_HI   = 160.0
const DS     = 0.5
s_grid = collect(S_LO:DS:S_HI)
N_S    = length(s_grid)

# Gauss-Legendre quadrature (or simple midpoint) over d ∈ [D_MIN, D_MAX]
const N_QUAD = 201
quad_d  = collect(range(D_MIN, D_MAX; length=N_QUAD))
quad_w  = fill(1.0 / N_QUAD, N_QUAD)

# ── Analytical expected holding + backlog cost for order-up-to level y ────────
function expected_stage_cost(y::Float64)
    D_range = D_MAX - D_MIN
    if y <= D_MIN
        return p * ((D_MIN + D_MAX) / 2 - y)
    elseif y >= D_MAX
        return h * (y - (D_MIN + D_MAX) / 2)
    else
        hold = h * (y - D_MIN)^2 / (2 * D_range)
        back = p * (D_MAX - y)^2 / (2 * D_range)
        return hold + back
    end
end

# ── Linear interpolation into value function ──────────────────────────────────
@inline function interp_V(V::Vector{Float64}, x::Float64)
    if x <= S_LO
        # Extrapolate: approximate with large holding/backlog cost beyond boundary
        return V[1] + (x - S_LO) * (V[2] - V[1]) / DS
    elseif x >= S_HI
        return V[end] + (x - S_HI) * (V[end] - V[end-1]) / DS
    end
    idx = floor(Int, (x - S_LO) / DS) + 1
    idx = clamp(idx, 1, N_S - 1)
    t   = (x - s_grid[idx]) / DS
    return V[idx] * (1 - t) + V[idx + 1] * t
end

# ── Expected future value E[V_{t+1}(y - d)] over d ~ U[D_MIN, D_MAX] ─────────
function expected_future(V_next::Vector{Float64}, y::Float64)
    acc = 0.0
    @inbounds for k in 1:N_QUAD
        acc += interp_V(V_next, y - quad_d[k])
    end
    return acc / N_QUAD
end

# ── Backward DP ───────────────────────────────────────────────────────────────
function run_dp()
    V_next   = zeros(N_S)
    policy_S = fill(NaN, T, N_S)

    println("Running backward DP (T=$T, grid size=$N_S, quadrature=$N_QUAD points)...")

    for t in T:-1:1
        V_curr = Vector{Float64}(undef, N_S)

        # Precompute E[V_next(y - d)] on a fine y-grid then interpolate cheaply
        Y_LO_PRE = S_LO
        Y_HI_PRE = S_HI + Q_MAX
        N_Y_PRE  = round(Int, (Y_HI_PRE - Y_LO_PRE) / DS) + 1
        y_pregrid  = collect(range(Y_LO_PRE, Y_HI_PRE; length=N_Y_PRE))
        EV_pregrid = Vector{Float64}(undef, N_Y_PRE)
        @inbounds for j in 1:N_Y_PRE
            EV_pregrid[j] = expected_future(V_next, y_pregrid[j])
        end

        ddy_pre = (Y_HI_PRE - Y_LO_PRE) / (N_Y_PRE - 1)
        function ev_at(y::Float64)
            y <= Y_LO_PRE && return EV_pregrid[1]
            y >= Y_HI_PRE && return EV_pregrid[end]
            idx  = floor(Int, (y - Y_LO_PRE) / ddy_pre) + 1
            idx  = clamp(idx, 1, N_Y_PRE - 1)
            frac = (y - y_pregrid[idx]) / ddy_pre
            return EV_pregrid[idx] * (1 - frac) + EV_pregrid[idx + 1] * frac
        end

        for i in 1:N_S
            s = s_grid[i]

            # Option A: don't order
            cost_no_order = expected_stage_cost(s) + ev_at(s)

            # Option B: order to y ∈ (s, s + Q_MAX] — fine grid search
            best_order = Inf
            best_y     = s
            y_lo = s + 1e-6
            y_hi = s + Q_MAX
            n_y  = max(2, round(Int, (y_hi - y_lo) / 0.25) + 1)
            @inbounds for y in range(y_lo, y_hi; length=n_y)
                cost = K + c * (y - s) + expected_stage_cost(y) + ev_at(y)
                if cost < best_order
                    best_order = cost
                    best_y     = y
                end
            end

            if cost_no_order <= best_order
                V_curr[i]      = cost_no_order
                policy_S[t, i] = NaN
            else
                V_curr[i]      = best_order
                policy_S[t, i] = best_y
            end
        end

        V_next = V_curr
        @printf("  stage %2d: V(I_0=%.1f) = %.4f\n", t, I_0, interp_V(V_curr, I_0))
    end

    return V_next, policy_S
end

V_final, policy_S = run_dp()
optimal_expected_cost = interp_V(V_final, I_0)
println("\nDP optimal expected cost (from I_0=$I_0): $(@sprintf("%.4f", optimal_expected_cost))")

# ── Simulate optimal policy out-of-sample ─────────────────────────────────────
N_SIM = 2000
rng   = MersenneTwister(7777)

sim_costs = Vector{Float64}(undef, N_SIM)

for sim in 1:N_SIM
    s = I_0
    total_cost = 0.0
    for t in 1:T
        d = D_MIN + (D_MAX - D_MIN) * rand(rng)

        # Look up optimal policy
        if s <= S_LO
            idx = 1
        elseif s >= S_HI
            idx = N_S
        else
            idx = floor(Int, (s - S_LO) / DS) + 1
            idx = clamp(idx, 1, N_S)
        end

        S_opt = policy_S[t, idx]
        if isnan(S_opt)
            # Don't order
            z = 0; q = 0.0
        else
            # Interpolate order-up-to level from neighbours
            if idx < N_S && !isnan(policy_S[t, idx+1])
                frac = (s - s_grid[idx]) / DS
                y = S_opt * (1 - frac) + policy_S[t, idx+1] * frac
            else
                y = S_opt
            end
            y = clamp(y, s, s + Q_MAX)
            q = y - s
            if q < 1e-6
                z = 0; q = 0.0
            else
                z = 1
            end
        end

        s_next = s + q - d
        inv_h  = max(s_next, 0.0)
        back   = max(-s_next, 0.0)
        total_cost += K * z + c * q + h * inv_h + p * back
        s = s_next
    end
    sim_costs[sim] = total_cost
end

μ = mean(sim_costs)
σ = std(sim_costs)
println("Simulation ($N_SIM scenarios): mean = $(@sprintf("%.1f", μ)) ± $(@sprintf("%.1f", σ))")
println("DP exact value:                $(@sprintf("%.1f", optimal_expected_cost))")

# ── Save ──────────────────────────────────────────────────────────────────────
result_dir = joinpath(@__DIR__, "results")
mkpath(result_dir)

CSV.write(
    joinpath(result_dir, "optimal_costs.csv"),
    DataFrame(operational_cost = sim_costs),
)
open(joinpath(result_dir, "optimal_dp_value.txt"), "w") do io
    println(io, optimal_expected_cost)
end
println("Saved results/optimal_costs.csv and results/optimal_dp_value.txt")
