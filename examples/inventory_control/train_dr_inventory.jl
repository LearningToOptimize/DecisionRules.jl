"""
Train a TS-DDR policy for the seasonal ex-ante inventory-control problem.
"""

using DecisionRules
using JuMP
using Flux
using JLD2
using CSV, DataFrames
using Random, Statistics

include(joinpath(@__DIR__, "build_inventory_problem.jl"))

const N_TRAIN_SCENARIOS = 50
const N_TEST = 300
const NUM_BATCHES = 260
const TRAIN_PER_BATCH = 5

example_dir = @__DIR__
model_dir = joinpath(example_dir, "models")
result_dir = joinpath(example_dir, "results")
mkpath(model_dir)
mkpath(result_dir)

model_path = joinpath(model_dir, "inventory_policy.jld2")
curve_path = joinpath(result_dir, "training_curve.csv")

println("Building deterministic equivalent for TS-DDR training...")
det_eq, spi_train, spo_train, unc_train, init_state = build_inventory_det_equivalent(;
    num_scenarios=N_TRAIN_SCENARIOS,
    penalty=INVENTORY_PENALTY,
    seed=42,
)
println("Building stage-wise ex-ante subproblems for evaluation...")
eval_subproblems, spi_eval, spo_eval, unc_eval, _ = build_inventory_subproblems(;
    num_scenarios=N_TEST,
    penalty=INVENTORY_PENALTY,
    seed=99,
)

policy = build_exante_policy(; seed=2024)

Random.seed!(111)
pre_costs = [
    let unc = sample(unc_train)
        simulate_multistage(
            det_eq,
            spi_train,
            spo_train,
            unc,
            simulate_states(init_state, unc, policy);
            integer_strategy=FixedDiscreteIntegerStrategy(),
        )
    end for _ in 1:12
]
pre_mean = mean(pre_costs)
println("Pre-training mean cost (with target penalty): $(round(pre_mean, digits=2))")

save_best = SaveBest(pre_mean, model_path)
training_log = DataFrame(batch=Int[], loss=Float64[])

println("\nTraining TS-DDR (FixedDiscreteIntegerStrategy, ex-ante policy input)...")
Random.seed!(2024)
train_start = time()
train_multistage(
    policy,
    init_state,
    det_eq,
    spi_train,
    spo_train,
    unc_train;
    num_batches=NUM_BATCHES,
    num_train_per_batch=TRAIN_PER_BATCH,
    optimizer=Flux.Adam(0.0015),
    integer_strategy=FixedDiscreteIntegerStrategy(),
    penalty_schedule=[(1, 50, 0.4), (51, NUM_BATCHES, 1.0)],
    record=(sample_log, iter, model) -> begin
        loss = isempty(sample_log.objectives_no_deficit) ? NaN : mean(sample_log.objectives_no_deficit)
        push!(training_log, (batch=iter, loss=loss))
        if mod(iter, 20) == 0 || iter == 1
            println("  Batch $(lpad(iter, 3)) / $NUM_BATCHES  loss = $(round(loss, digits=2))")
        end
        save_best(iter, model, loss)
        return false
    end,
)
train_seconds = time() - train_start

CSV.write(curve_path, training_log)
println("Training curve saved -> $(relpath(curve_path, example_dir))")
println("Best model saved     -> $(relpath(model_path, example_dir))")

model_state = JLD2.load(model_path, "model_state")
Flux.loadmodel!(policy, model_state)

function rollout_policy(policy, subproblems, spi, spo, unc_eval, init_state; n_test=N_TEST, seed=555)
    Random.seed!(seed)
    traj_inv = Matrix{Float64}(undef, n_test, INVENTORY_T + 1)
    traj_z = Matrix{Float64}(undef, n_test, INVENTORY_T)
    traj_q = Matrix{Float64}(undef, n_test, INVENTORY_T)
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

            z_val = round(value(subproblems[t][:z]))
            q_val = value(subproblems[t][:q])
            s_val = value(subproblems[t][:s_out])
            op_costs[s] += INVENTORY_K * z_val + INVENTORY_C * q_val +
                           INVENTORY_H * max(s_val, 0.0) +
                           INVENTORY_P * max(-s_val, 0.0)
            traj_z[s, t] = z_val
            traj_q[s, t] = q_val
            traj_inv[s, t+1] = s_val
            state = [s_val, d_val, state[2]]
        end
    end
    return op_costs, traj_inv, traj_z, traj_q
end

println("\nEvaluating trained policy on $N_TEST test scenarios...")
eval_start = time()
op_costs, traj_inv, traj_z, traj_q = rollout_policy(
    policy, eval_subproblems, spi_eval, spo_eval, unc_eval, init_state;
    n_test=N_TEST,
    seed=555,
)
eval_seconds = time() - eval_start

df_inv = DataFrame(traj_inv, [Symbol("t$i") for i in 0:INVENTORY_T])
df_inv[!, :scenario] = 1:N_TEST
CSV.write(joinpath(result_dir, "dr_trajectories.csv"), df_inv)

df_orders = DataFrame(
    scenario=repeat(1:N_TEST, inner=INVENTORY_T),
    period=repeat(1:INVENTORY_T, outer=N_TEST),
    z=vec(traj_z'),
    q=vec(traj_q'),
)
CSV.write(joinpath(result_dir, "dr_orders.csv"), df_orders)
CSV.write(joinpath(result_dir, "dr_costs.csv"), DataFrame(scenario=1:N_TEST, operational_cost=op_costs))
CSV.write(
    joinpath(result_dir, "dr_timing.csv"),
    DataFrame(
        method=["TS-DDR (trained)"],
        fit_seconds=[train_seconds],
        inference_seconds=[eval_seconds],
        n_eval=[N_TEST],
        inference_ms_per_scenario=[1000 * eval_seconds / N_TEST],
    ),
)

println("Mean operational cost (trained): $(round(mean(op_costs), digits=2))")
println("Std  operational cost (trained): $(round(std(op_costs), digits=2))")
println("Training time (s): $(round(train_seconds, digits=2))")
println("Inference time (s): $(round(eval_seconds, digits=2))")
println("Trajectories saved -> results/dr_trajectories.csv")
println("Costs saved        -> results/dr_costs.csv")
