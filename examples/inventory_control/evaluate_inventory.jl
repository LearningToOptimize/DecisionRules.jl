"""
Evaluate non-neural baselines for the inventory control problem.

Evaluates base-stock and random policies for both relaxed and integer cases.
"""

using DecisionRules
using JuMP
using Flux
using CSV, DataFrames
using Random, Statistics

include(joinpath(@__DIR__, "build_inventory_problem.jl"))

const N_EVAL = 300

# ═══════════════════════════════════════════════════════════════════════════════
# Base-stock evaluation (analytical — no subproblem needed)
# ═══════════════════════════════════════════════════════════════════════════════
function evaluate_basestock_direct(S_star; n_test=N_EVAL, seed=555,
    keep_traj=false, integer=true)
    Random.seed!(seed)
    traj_inv = Matrix{Float64}(undef, n_test, INVENTORY_T + 1)
    op_costs = Vector{Float64}(undef, n_test)

    for s in 1:n_test
        inventory = INVENTORY_I0
        traj_inv[s, 1] = inventory
        op_costs[s] = 0.0
        demands = sample_inventory_demand_path()

        for t in 1:INVENTORY_T
            q = inventory < S_star ? min(S_star - inventory, INVENTORY_Q_MAX) : 0.0
            z = q > 1e-8 ? 1.0 : 0.0
            s_out = inventory + q - demands[t]
            if integer
                op_costs[s] += INVENTORY_K * z + INVENTORY_C * q +
                               INVENTORY_H * max(s_out, 0.0) +
                               INVENTORY_P * max(-s_out, 0.0)
            else
                op_costs[s] += INVENTORY_C * q +
                               INVENTORY_H * max(s_out, 0.0) +
                               INVENTORY_P * max(-s_out, 0.0)
            end
            traj_inv[s, t+1] = s_out
            inventory = s_out
        end
    end
    return keep_traj ? (op_costs, traj_inv) : (op_costs, nothing)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Random policy evaluation (uses subproblems)
# ═══════════════════════════════════════════════════════════════════════════════
function evaluate_policy_stagewise(policy, subproblems, spi, spo, unc_eval, init_state;
    n_test=N_EVAL, seed=555, keep_traj=false, integer=true)
    Random.seed!(seed)
    traj_inv = Matrix{Float64}(undef, n_test, INVENTORY_T + 1)
    op_costs = Vector{Float64}(undef, n_test)

    for s in 1:n_test
        unc_sample = sample(unc_eval)
        state = Float64.(init_state)
        traj_inv[s, 1] = state[1]
        op_costs[s] = 0.0

        for t in 1:INVENTORY_T
            d_val = unc_sample[t][1][2]
            target = Float64.(policy(Float32[d_val, state...]))

            for i in eachindex(spi[t])
                set_parameter_value(spi[t][i], state[i])
            end
            for (param, value) in unc_sample[t]
                set_parameter_value(param, value)
            end
            for i in eachindex(spo[t])
                set_parameter_value(spo[t][i][1], target[i])
            end
            optimize!(subproblems[t])

            q_val = value(subproblems[t][:q])
            s_val = value(subproblems[t][:s_out])

            if integer
                z_val = round(value(subproblems[t][:z]))
                op_costs[s] += INVENTORY_K * z_val + INVENTORY_C * q_val +
                               INVENTORY_H * max(s_val, 0.0) +
                               INVENTORY_P * max(-s_val, 0.0)
            else
                op_costs[s] += INVENTORY_C * q_val +
                               INVENTORY_H * max(s_val, 0.0) +
                               INVENTORY_P * max(-s_val, 0.0)
            end
            traj_inv[s, t+1] = s_val
            state = [s_val, d_val, state[2]]
        end
    end
    return keep_traj ? (op_costs, traj_inv) : (op_costs, nothing)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Run baselines for a given formulation
# ═══════════════════════════════════════════════════════════════════════════════
function evaluate_baselines(; tag::String, integer::Bool)
    println("=" ^ 60)
    println("Evaluating baselines — $(tag) (integer=$integer)")
    println("=" ^ 60)

    println("Building subproblems (integer=$integer)...")
    subproblems, spi, spo, unc_eval, init_state = build_inventory_subproblems(;
        num_scenarios=N_EVAL, penalty=INVENTORY_PENALTY, seed=99, integer=integer)

    # --- Base-stock grid search ---
    println("Grid-searching base-stock level...")
    tune_start = time()
    best_S = 80.0
    best_mu = Inf
    for S in 30.0:10.0:280.0
        costs, _ = evaluate_basestock_direct(S; n_test=600, seed=777, integer=integer)
        mu = mean(costs)
        if mu < best_mu
            best_mu = mu
            best_S = S
        end
    end
    for S in (best_S - 12.0):2.0:(best_S + 12.0)
        S <= 0 && continue
        costs, _ = evaluate_basestock_direct(S; n_test=600, seed=777, integer=integer)
        mu = mean(costs)
        if mu < best_mu
            best_mu = mu
            best_S = S
        end
    end
    tune_seconds = time() - tune_start
    println("  S* = $(best_S) (mean cost = $(round(best_mu, digits=1)))")

    println("Evaluating base-stock (S*=$best_S) on $N_EVAL scenarios...")
    bs_eval_start = time()
    bs_costs, bs_traj = evaluate_basestock_direct(best_S; n_test=N_EVAL, seed=555,
        keep_traj=true, integer=integer)
    bs_eval_seconds = time() - bs_eval_start
    println("  Mean: $(round(mean(bs_costs), digits=1)) ± $(round(std(bs_costs), digits=1))")

    # --- Random policy ---
    println("Evaluating random untrained policy...")
    random_policy = build_exante_policy(; seed=7)
    # Warmup JIT
    evaluate_policy_stagewise(random_policy, subproblems, spi, spo, unc_eval, init_state;
        n_test=1, seed=0, integer=integer)
    random_eval_start = time()
    rand_costs, _ = evaluate_policy_stagewise(random_policy, subproblems, spi, spo,
        unc_eval, init_state; n_test=N_EVAL, seed=555, integer=integer)
    random_eval_seconds = time() - random_eval_start
    println("  Mean: $(round(mean(rand_costs), digits=1)) ± $(round(std(rand_costs), digits=1))")

    # --- Save results ---
    result_dir = joinpath(@__DIR__, "results")
    mkpath(result_dir)

    df_bs_traj = DataFrame(bs_traj, [Symbol("t$i") for i in 0:INVENTORY_T])
    df_bs_traj[!, :scenario] = 1:N_EVAL
    CSV.write(joinpath(result_dir, "$(tag)_basestock_trajectories.csv"), df_bs_traj)
    CSV.write(joinpath(result_dir, "$(tag)_basestock_costs.csv"),
        DataFrame(scenario=1:N_EVAL, operational_cost=bs_costs))
    CSV.write(joinpath(result_dir, "$(tag)_random_costs.csv"),
        DataFrame(scenario=1:N_EVAL, operational_cost=rand_costs))
    CSV.write(joinpath(result_dir, "$(tag)_baseline_timing.csv"),
        DataFrame(
            method=["Base-stock", "Random (untrained)"],
            fit_seconds=[tune_seconds, 0.0],
            eval_seconds=[bs_eval_seconds / N_EVAL, random_eval_seconds / (N_EVAL * INVENTORY_T)],
            n_eval=[N_EVAL, N_EVAL]))
    open(joinpath(result_dir, "$(tag)_basestock_S_star.txt"), "w") do io
        println(io, best_S)
    end
    return bs_costs, rand_costs, best_S
end

# ═══════════════════════════════════════════════════════════════════════════════
# Run both
# ═══════════════════════════════════════════════════════════════════════════════
evaluate_baselines(tag="relaxed", integer=false)
println()
evaluate_baselines(tag="integer", integer=true)
println("\nAll baseline results saved to results/")
