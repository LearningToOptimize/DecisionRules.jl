"""
Specialized dynamic-programming benchmark for a simplified marginal seasonal
model of the ex-ante lot-sizing problem.

The true evaluation process has latent persistent regimes and autocorrelated
shocks. This DP knows the seasonal marginal bands but not that latent process,
so it is a strong structure-aware benchmark, not an exact optimum.
"""

using CSV, DataFrames, Statistics, Random, Printf

include(joinpath(@__DIR__, "build_inventory_problem.jl"))

const S_LO = -260.0
const S_HI = 280.0
const DS = 1.0
const N_QUAD = 121
const N_SIM = 3000

s_grid = collect(S_LO:DS:S_HI)
const N_S = length(s_grid)

@inline function interp_grid(V::Vector{Float64}, x::Float64)
    if x <= S_LO
        return V[1] + (x - S_LO) * (V[2] - V[1]) / DS
    elseif x >= S_HI
        return V[end] + (x - S_HI) * (V[end] - V[end-1]) / DS
    end
    idx = floor(Int, (x - S_LO) / DS) + 1
    idx = clamp(idx, 1, N_S - 1)
    frac = (x - s_grid[idx]) / DS
    return V[idx] * (1 - frac) + V[idx + 1] * frac
end

function expected_one_period_cost(y::Float64, lo::Float64, hi::Float64)
    width = hi - lo
    if y <= lo
        return INVENTORY_P * ((lo + hi) / 2 - y)
    elseif y >= hi
        return INVENTORY_H * (y - (lo + hi) / 2)
    else
        hold = INVENTORY_H * (y - lo)^2 / (2 * width)
        back = INVENTORY_P * (hi - y)^2 / (2 * width)
        return hold + back
    end
end

function expected_future(V_next::Vector{Float64}, y::Float64, lo::Float64, hi::Float64)
    acc = 0.0
    for d in range(lo, hi; length=N_QUAD)
        acc += interp_grid(V_next, y - d)
    end
    return acc / N_QUAD
end

function run_dp()
    V_next = zeros(N_S)
    policy_y = fill(NaN, INVENTORY_T, N_S)

    println("Running backward DP (T=$INVENTORY_T, grid size=$N_S, quadrature=$N_QUAD)...")

    for t in INVENTORY_T:-1:1
        lo, hi = D_LO[t], D_HI[t]
        V_curr = Vector{Float64}(undef, N_S)

        y_lo_pre = S_LO
        y_hi_pre = S_HI + INVENTORY_Q_MAX
        y_pregrid = collect(y_lo_pre:DS:y_hi_pre)
        ev_pre = [expected_future(V_next, y, lo, hi) for y in y_pregrid]

        function ev_at(y::Float64)
            if y <= y_lo_pre
                return ev_pre[1]
            elseif y >= y_hi_pre
                return ev_pre[end]
            end
            idx = floor(Int, (y - y_lo_pre) / DS) + 1
            idx = clamp(idx, 1, length(y_pregrid) - 1)
            frac = (y - y_pregrid[idx]) / DS
            return ev_pre[idx] * (1 - frac) + ev_pre[idx + 1] * frac
        end

        for (i, s) in enumerate(s_grid)
            y_no = s
            cost_no = expected_one_period_cost(y_no, lo, hi) + ev_at(y_no)

            best_order = Inf
            best_y = s
            for y in s:1.0:(s + INVENTORY_Q_MAX)
                q = max(y - s, 0.0)
                setup = q <= 1e-8 ? 0.0 : INVENTORY_K
                cost = setup + INVENTORY_C * q +
                       expected_one_period_cost(y, lo, hi) + ev_at(y)
                if cost < best_order
                    best_order = cost
                    best_y = y
                end
            end

            if cost_no <= best_order
                V_curr[i] = cost_no
                policy_y[t, i] = NaN
            else
                V_curr[i] = best_order
                policy_y[t, i] = best_y
            end
        end

        V_next = V_curr
        @printf("  stage %2d: V(I_0=%.1f) = %.4f\n", t, INVENTORY_I0, interp_grid(V_curr, INVENTORY_I0))
    end

    return V_next, policy_y
end

dp_start = time()
V_final, policy_y = run_dp()
dp_seconds = time() - dp_start
optimal_expected_cost = interp_grid(V_final, INVENTORY_I0)
println("\nMarginal DP expected cost (from I_0=$INVENTORY_I0): $(@sprintf("%.4f", optimal_expected_cost))")

rng = MersenneTwister(7777)
sim_costs = Vector{Float64}(undef, N_SIM)

dp_eval_start = time()
for sim in 1:N_SIM
    s = INVENTORY_I0
    total_cost = 0.0
    demand_path = sample_inventory_demand_path(rng)
    for t in 1:INVENTORY_T
        idx = clamp(floor(Int, (s - S_LO) / DS) + 1, 1, N_S)
        y = policy_y[t, idx]
        if isnan(y) || y <= s + 1e-8
            z = 0.0
            q = 0.0
            y = s
        else
            y = min(y, s + INVENTORY_Q_MAX)
            z = 1.0
            q = y - s
        end
        d = demand_path[t]
        s_next = y - d
        total_cost += INVENTORY_K * z + INVENTORY_C * q +
                      INVENTORY_H * max(s_next, 0.0) +
                      INVENTORY_P * max(-s_next, 0.0)
        s = s_next
    end
    sim_costs[sim] = total_cost
end
dp_eval_seconds = time() - dp_eval_start

μ = mean(sim_costs)
σ = std(sim_costs)
println("True-process simulation ($N_SIM scenarios): mean = $(@sprintf("%.1f", μ)) +/- $(@sprintf("%.1f", σ))")
println("Marginal DP model value:       $(@sprintf("%.1f", optimal_expected_cost))")

result_dir = joinpath(@__DIR__, "results")
mkpath(result_dir)
CSV.write(joinpath(result_dir, "optimal_costs.csv"), DataFrame(operational_cost=sim_costs))
CSV.write(
    joinpath(result_dir, "optimal_timing.csv"),
    DataFrame(
        method=["Marginal DP policy"],
        fit_seconds=[dp_seconds],
        inference_seconds=[dp_eval_seconds],
        n_eval=[N_SIM],
        inference_ms_per_scenario=[1000 * dp_eval_seconds / N_SIM],
    ),
)
open(joinpath(result_dir, "optimal_dp_value.txt"), "w") do io
    println(io, optimal_expected_cost)
end
println("Saved results/optimal_costs.csv and results/optimal_dp_value.txt")
