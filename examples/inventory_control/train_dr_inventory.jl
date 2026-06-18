"""
Train TS-DDR policies for the inventory control problem.

Trains two policies:
1. Relaxed (continuous LP subproblems, standard LP duals)
2. Integer (MIP subproblems, FixedDiscreteIntegerStrategy) — uses more
   batches and lower learning rate for stable convergence.
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

example_dir = @__DIR__
model_dir = joinpath(example_dir, "models")
result_dir = joinpath(example_dir, "results")
mkpath(model_dir)
mkpath(result_dir)

# ═══════════════════════════════════════════════════════════════════════════════
# Rollout helper
# ═══════════════════════════════════════════════════════════════════════════════
function rollout_policy(policy, subproblems, spi, spo, unc_eval, init_state;
    n_test=N_TEST, seed=555, integer=true)
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

            q_val = value(subproblems[t][:q])
            s_val = value(subproblems[t][:s_out])

            if integer
                z_val = round(value(subproblems[t][:z]))
                op_costs[s] += INVENTORY_K * z_val + INVENTORY_C * q_val +
                               INVENTORY_H * max(s_val, 0.0) +
                               INVENTORY_P * max(-s_val, 0.0)
                traj_z[s, t] = z_val
            else
                op_costs[s] += INVENTORY_C * q_val +
                               INVENTORY_H * max(s_val, 0.0) +
                               INVENTORY_P * max(-s_val, 0.0)
                traj_z[s, t] = q_val > 1e-7 ? 1.0 : 0.0
            end
            traj_q[s, t] = q_val
            traj_inv[s, t+1] = s_val
            state = [s_val, d_val, state[2]]
        end
    end
    return op_costs, traj_inv, traj_z, traj_q
end

# ═══════════════════════════════════════════════════════════════════════════════
# Train + evaluate helper
# ═══════════════════════════════════════════════════════════════════════════════
function train_and_evaluate(;
    tag::String,
    integer::Bool,
    num_batches::Int=400,
    train_per_batch::Int=5,
    lr::Float64=0.0015,
    warmup_batches::Int=80,
    int_strategy_override::Union{Nothing, AbstractIntegerStrategy}=nothing,
)
    println("=" ^ 60)
    println("Training TS-DDR — $(tag) (integer=$integer)")
    println("=" ^ 60)

    model_path = joinpath(model_dir, "$(tag)_policy.jld2")
    curve_path = joinpath(result_dir, "$(tag)_training_curve.csv")

    println("Building deterministic equivalent...")
    det_eq, spi_train, spo_train, unc_train, init_state = build_inventory_det_equivalent(;
        num_scenarios=N_TRAIN_SCENARIOS, penalty=INVENTORY_PENALTY, seed=42, integer=integer)

    println("Building stage-wise subproblems...")
    eval_subproblems, spi_eval, spo_eval, unc_eval, _ = build_inventory_subproblems(;
        num_scenarios=N_TEST, penalty=INVENTORY_PENALTY, seed=99, integer=integer)

    policy = build_exante_policy(; seed=2024)

    int_strategy = if int_strategy_override !== nothing
        int_strategy_override
    elseif integer
        FixedDiscreteIntegerStrategy()
    else
        NoIntegerStrategy()
    end

    Random.seed!(111)
    pre_costs = [
        let unc = sample(unc_train)
            simulate_multistage(
                det_eq, spi_train, spo_train, unc,
                simulate_states(init_state, unc, policy);
                integer_strategy=int_strategy,
            )
        end for _ in 1:12
    ]
    pre_mean = mean(pre_costs)
    println("Pre-training mean cost: $(round(pre_mean, digits=2))")

    save_best = SaveBest(pre_mean, model_path)
    training_log = DataFrame(batch=Int[], loss=Float64[])

    println("Training ($num_batches batches × $train_per_batch scenarios, lr=$lr)...")
    Random.seed!(2024)
    train_start = time()
    train_multistage(
        policy, init_state, det_eq, spi_train, spo_train, unc_train;
        num_batches=num_batches,
        num_train_per_batch=train_per_batch,
        optimizer=Flux.Adam(lr),
        integer_strategy=int_strategy,
        penalty_schedule=[(1, warmup_batches, 0.4), (warmup_batches+1, num_batches, 1.0)],
        record=(sample_log, iter, model) -> begin
            loss = isempty(sample_log.objectives_no_deficit) ? NaN : mean(sample_log.objectives_no_deficit)
            push!(training_log, (batch=iter, loss=loss))
            if mod(iter, 20) == 0 || iter == 1
                println("  Batch $(lpad(iter, 3)) / $num_batches  loss = $(round(loss, digits=2))")
            end
            save_best(iter, model, loss)
            return false
        end,
    )
    train_seconds = time() - train_start
    CSV.write(curve_path, training_log)
    println("Training time: $(round(train_seconds, digits=1))s")

    model_state = JLD2.load(model_path, "model_state")
    Flux.loadmodel!(policy, model_state)

    println("Evaluating on $N_TEST test scenarios...")
    eval_start = time()
    op_costs, traj_inv, traj_z, traj_q = rollout_policy(
        policy, eval_subproblems, spi_eval, spo_eval, unc_eval, init_state;
        n_test=N_TEST, seed=555, integer=integer)
    eval_seconds = time() - eval_start

    df_inv = DataFrame(traj_inv, [Symbol("t$i") for i in 0:INVENTORY_T])
    df_inv[!, :scenario] = 1:N_TEST
    CSV.write(joinpath(result_dir, "$(tag)_dr_trajectories.csv"), df_inv)
    CSV.write(joinpath(result_dir, "$(tag)_dr_costs.csv"),
        DataFrame(scenario=1:N_TEST, operational_cost=op_costs))
    method_name = if int_strategy isa ContinuousRelaxationIntegerStrategy
        "TS-DDR (ContRelax)"
    elseif int_strategy isa FixedDiscreteIntegerStrategy
        "TS-DDR (FixedDiscrete)"
    else
        "TS-DDR (trained)"
    end
    CSV.write(joinpath(result_dir, "$(tag)_dr_timing.csv"),
        DataFrame(method=[method_name],
                  fit_seconds=[train_seconds],
                  eval_seconds=[eval_seconds / (N_TEST * INVENTORY_T)],
                  n_eval=[N_TEST]))

    μ = mean(op_costs)
    σ = std(op_costs)
    println("$(tag) TS-DDR — mean: $(round(μ, digits=1)) ± $(round(σ, digits=1))")
    println("  Eval/stage: $(round(eval_seconds/(N_TEST*INVENTORY_T), digits=4))s")
    return op_costs
end

# ═══════════════════════════════════════════════════════════════════════════════
# Run both
# ═══════════════════════════════════════════════════════════════════════════════
train_and_evaluate(tag="relaxed", integer=false,
    num_batches=400, train_per_batch=5, lr=0.0015, warmup_batches=80)
println()
train_and_evaluate(tag="integer", integer=true,
    num_batches=800, train_per_batch=10, lr=0.0008, warmup_batches=120)
println()
train_and_evaluate(tag="integer_cr", integer=true,
    num_batches=800, train_per_batch=10, lr=0.0008, warmup_batches=120,
    int_strategy_override=ContinuousRelaxationIntegerStrategy())
println("\nAll TS-DDR results saved to $(relpath(result_dir, example_dir))")
