"""
Evaluate baselines for the stochastic lot-sizing problem.

Policies evaluated:
1. Base-stock (S* tuned by grid search)
2. Random (untrained TS-DDR network)

Results saved to results/ alongside trained-policy and SDDP results from
train_dr_inventory.jl, solve_sddp.jl, so compare_results.jl can load everything.
"""

using DecisionRules
using JuMP, HiGHS
using Flux
using CSV, DataFrames
using Random, Statistics

include(joinpath(@__DIR__, "build_inventory_problem.jl"))

# ── Problem parameters (must match train_dr_inventory.jl) ────────────────────
const T_STAGES    = 12
const K_FIXED     = 30.0
const C_UNIT      = 2.0
const H_HOLD      = 1.0
const P_BACK      = 10.0
const Q_MAX_VAL   = 80.0
const I_INIT      = 20.0
const D_MIN_VAL   = 10.0
const D_MAX_VAL   = 30.0
const PENALTY_VAL = 50.0   # must match train_dr_inventory.jl

# ── Build subproblems ─────────────────────────────────────────────────────────
println("Building stage-wise subproblems...")
subproblems, spi, spo, unc_eval, init_state = build_inventory_subproblems(;
    T=T_STAGES, K=K_FIXED, c=C_UNIT, h=H_HOLD, p=P_BACK,
    Q_max=Q_MAX_VAL, I_0=I_INIT, d_min=D_MIN_VAL, d_max=D_MAX_VAL,
    num_scenarios=200, penalty=PENALTY_VAL, seed=99,
)

# ── Helpers ───────────────────────────────────────────────────────────────────
function evaluate_policy_stagewise(
    policy,
    subproblems,
    spi,
    spo,
    unc_eval,
    init_state;
    n_test = 200,
    seed   = 555,
)
    Random.seed!(seed)
    traj_inv = Matrix{Float64}(undef, n_test, T_STAGES + 1)
    op_costs = Vector{Float64}(undef, n_test)

    for s in 1:n_test
        unc_sample = sample(unc_eval)
        state      = Float64.(init_state)
        traj_inv[s, 1] = state[1]
        op_costs[s] = 0.0

        for t in 1:T_STAGES
            d_val  = unc_sample[t][1][2]
            target = Float64(policy(Float32[d_val, state[1]])[1])

            set_parameter_value(spi[t][1], state[1])
            set_parameter_value(unc_sample[t][1][1], d_val)
            set_parameter_value(spo[t][1][1], target)
            optimize!(subproblems[t])

            z_val = value(subproblems[t][:z])
            q_val = value(subproblems[t][:q])
            s_val = value(subproblems[t][:s_out])
            op_costs[s] += K_FIXED * z_val + C_UNIT * q_val +
                           H_HOLD * max(s_val, 0.0) + P_BACK * max(-s_val, 0.0)
            traj_inv[s, t+1] = s_val
            state = [s_val]
        end
    end
    return op_costs, traj_inv
end

# ── Base-stock: grid search for optimal S* ────────────────────────────────────
# Newsvendor critical ratio p/(p+h) = 10/11 ≈ 0.909; for U[10,30] that's ≈28.
# With fixed ordering cost K=30 we expect S* somewhat above 28.
println("\nGrid-searching optimal S* for base-stock policy...")
best_S  = 25.0
best_mu = Inf
for S in 20.0:2.0:60.0
    pol = base_stock_policy(S)
    costs, _ = evaluate_policy_stagewise(pol, subproblems, spi, spo, unc_eval, init_state;
                                         n_test=200, seed=777)
    mu = mean(costs)
    println("  S* = $S  →  mean cost = $(round(mu, digits=1))")
    if mu < best_mu
        best_mu = mu
        best_S  = S
    end
end
println("Best S* = $best_S  (mean cost = $(round(best_mu, digits=1)))")

# Fine search around the coarse best
fine_best_S  = best_S
fine_best_mu = best_mu
for S in (best_S - 4.0):1.0:(best_S + 4.0)
    S <= 0 && continue
    pol = base_stock_policy(S)
    costs, _ = evaluate_policy_stagewise(pol, subproblems, spi, spo, unc_eval, init_state;
                                         n_test=200, seed=777)
    mu = mean(costs)
    if mu < fine_best_mu
        fine_best_mu = mu
        fine_best_S  = S
    end
end
S_STAR = fine_best_S
println("Fine-tuned S* = $S_STAR  (mean cost = $(round(fine_best_mu, digits=1)))")

# ── Evaluate base-stock at optimal S* ─────────────────────────────────────────
base_stock = base_stock_policy(S_STAR)
println("\nEvaluating base-stock (S* = $S_STAR) on 200 test scenarios...")
bs_costs, bs_traj = evaluate_policy_stagewise(
    base_stock, subproblems, spi, spo, unc_eval, init_state; n_test=200, seed=555,
)
println("  Mean: $(round(mean(bs_costs), digits=1)) ± $(round(std(bs_costs), digits=1))")

# ── Random (untrained) policy ─────────────────────────────────────────────────
Random.seed!(2024)
random_policy = Chain(Dense(2, 16, relu), Dense(16, 8, relu), Dense(8, 1))

println("\nEvaluating random (untrained) policy...")
rand_costs, _ = evaluate_policy_stagewise(
    random_policy, subproblems, spi, spo, unc_eval, init_state; n_test=200, seed=555,
)
println("  Mean: $(round(mean(rand_costs), digits=1)) ± $(round(std(rand_costs), digits=1))")

# ── Save ──────────────────────────────────────────────────────────────────────
result_dir = joinpath(@__DIR__, "results")
mkpath(result_dir)

n_test = length(bs_costs)
df_bs_traj = DataFrame(bs_traj, [Symbol("t$i") for i in 0:T_STAGES])
df_bs_traj[!, :scenario] = 1:n_test

CSV.write(joinpath(result_dir, "basestock_trajectories.csv"), df_bs_traj)
CSV.write(joinpath(result_dir, "basestock_costs.csv"),
    DataFrame(scenario=1:n_test, operational_cost=bs_costs))
CSV.write(joinpath(result_dir, "random_costs.csv"),
    DataFrame(scenario=1:n_test, operational_cost=rand_costs))

open(joinpath(result_dir, "basestock_S_star.txt"), "w") do io
    println(io, S_STAR)
end

println("\nSaved to results/")
