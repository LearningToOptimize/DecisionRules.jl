"""
Solve the seasonal ex-ante lot-sizing problem with SDDP.jl.

The SDDP model uses 2T stages:
  odd stages  = order before demand is observed
  even stages = demand realization and inventory/backlog cost
"""

using SDDP
using JuMP
using HiGHS
using CSV, DataFrames
using Statistics
using Random

include(joinpath(@__DIR__, "build_inventory_problem.jl"))

const N_TRAIN = 40
const N_SIM = 300
const ITERATION_LIMIT = 350

rng_train = MersenneTwister(42)
training_paths = [sample_inventory_demand_path(rng_train) for _ in 1:N_TRAIN]
demand_scenarios = [[path[t] for path in training_paths] for t in 1:INVENTORY_T]

model = SDDP.LinearPolicyGraph(
    stages=2 * INVENTORY_T,
    sense=:Min,
    lower_bound=0.0,
    optimizer=HiGHS.Optimizer,
) do sp, stage
    set_silent(sp)
    @variable(sp, s, SDDP.State, initial_value=INVENTORY_I0)

    if isodd(stage)
        @variable(sp, 0 <= q <= INVENTORY_Q_MAX)
        @variable(sp, 0 <= z <= 1)
        @constraint(sp, q <= INVENTORY_Q_MAX * z)
        @constraint(sp, s.out == s.in + q)
        @stageobjective(sp, INVENTORY_K * z + INVENTORY_C * q)
    else
        t = stage ÷ 2
        @variable(sp, d_par)
        @variable(sp, inv_hold >= 0)
        @variable(sp, back >= 0)
        SDDP.parameterize(sp, demand_scenarios[t]) do d_val
            JuMP.fix(d_par, d_val; force=true)
        end
        @constraint(sp, s.out == s.in - d_par)
        @constraint(sp, inv_hold - back == s.out)
        @stageobjective(sp, INVENTORY_H * inv_hold + INVENTORY_P * back)
    end
end

println("Training 24-stage SDDP relaxation...")
sddp_train_start = time()
SDDP.train(
    model;
    duality_handler=SDDP.ContinuousConicDuality(),
    iteration_limit=ITERATION_LIMIT,
    stopping_rules=[SDDP.BoundStalling(80, 1e-3)],
    print_level=1,
)
sddp_train_seconds = time() - sddp_train_start

lower_bound = SDDP.calculate_bound(model)
println("\nSDDP LP lower bound (expected cost): $lower_bound")

function training_log_dataframe(model)
    log = model.most_recent_training_results.log
    rows = DataFrame(iteration=Int[], bound=Float64[], simulation_value=Float64[], time=Float64[])
    for row in log
        iter = hasproperty(row, :iteration) ? row.iteration : row[:iteration]
        bound = hasproperty(row, :bound) ? row.bound : row[:bound]
        sim = try
            hasproperty(row, :simulation_value) ? row.simulation_value : row[:simulation_value]
        catch
            NaN
        end
        tm = try
            hasproperty(row, :time) ? row.time : row[:time]
        catch
            NaN
        end
        push!(rows, (iteration=iter, bound=bound, simulation_value=sim, time=tm))
    end
    return rows
end

function rollout_sddp(model, n_sim)
    costs = Vector{Float64}(undef, n_sim)
    traj_inv = Matrix{Float64}(undef, n_sim, INVENTORY_T + 1)

    for sim in 1:n_sim
        state = INVENTORY_I0
        total_cost = 0.0
        traj_inv[sim, 1] = state
        demand_path = sample_inventory_demand_path()

        for t in 1:INVENTORY_T
            order_sp = model.nodes[2t - 1].subproblem
            JuMP.fix(order_sp[:s].in, state; force=true)
            optimize!(order_sp)

            q = clamp(value(order_sp[:q]), 0.0, INVENTORY_Q_MAX)
            z = q <= 1e-7 ? 0.0 : 1.0
            s_mid = state + q

            d = demand_path[t]
            s_out = s_mid - d

            total_cost += INVENTORY_K * z + INVENTORY_C * q +
                          INVENTORY_H * max(s_out, 0.0) +
                          INVENTORY_P * max(-s_out, 0.0)
            state = s_out
            traj_inv[sim, t+1] = state
        end
        costs[sim] = total_cost
    end
    return costs, traj_inv
end

println("\nInteger rollout on $N_SIM fresh seasonal scenarios...")
Random.seed!(555)
sddp_eval_start = time()
sddp_costs, sddp_traj = rollout_sddp(model, N_SIM)
sddp_eval_seconds = time() - sddp_eval_start

μ = mean(sddp_costs)
σ = std(sddp_costs)
println("SDDP policy — mean cost: $(round(μ, digits=1)) +/- $(round(σ, digits=1))")
println("SDDP LP lower bound:    $(round(lower_bound, digits=1))")
println("LP gap to rollout:      $(round(100 * (μ - lower_bound) / μ, digits=1))%")

result_dir = joinpath(@__DIR__, "results")
mkpath(result_dir)
CSV.write(joinpath(result_dir, "sddp_costs.csv"), DataFrame(operational_cost=sddp_costs))
CSV.write(
    joinpath(result_dir, "sddp_trajectories.csv"),
    DataFrame(sddp_traj, [Symbol("t$i") for i in 0:INVENTORY_T]),
)
CSV.write(joinpath(result_dir, "sddp_training_log.csv"), training_log_dataframe(model))
CSV.write(
    joinpath(result_dir, "sddp_timing.csv"),
    DataFrame(
        method=["SDDP.jl integer rollout"],
        fit_seconds=[0.0],
        eval_seconds=[sddp_train_seconds],
        n_eval=[N_SIM],
    ),
)
open(joinpath(result_dir, "sddp_bound.txt"), "w") do io
    println(io, lower_bound)
end
println("\nSaved SDDP results to $(relpath(result_dir, @__DIR__))")
