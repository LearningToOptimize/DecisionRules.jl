"""
Train a TS-DDR policy for the stochastic lot-sizing problem using
`FixedDiscreteIntegerStrategy`.

Saves:
  models/inventory_policy.jld2   — best trained model weights
  results/training_curve.csv     — per-batch loss
  results/dr_trajectories.csv    — net inventory trajectories on test scenarios
  results/dr_costs.csv           — per-scenario operational costs
"""

using DecisionRules
using JuMP, HiGHS
using Flux
using JLD2
using CSV, DataFrames
using Random, Statistics

include(joinpath(@__DIR__, "build_inventory_problem.jl"))

# ── Problem parameters ────────────────────────────────────────────────────

const T_STAGES    = 12
const K_FIXED     = 30.0
const C_UNIT      = 2.0
const H_HOLD      = 1.0
const P_BACK      = 10.0
const Q_MAX_VAL   = 80.0
const I_INIT      = 20.0
const D_MIN_VAL   = 10.0
const D_MAX_VAL   = 30.0
const PENALTY_VAL = 50.0   # low: keeps operational-cost gradient from being swamped

# ── Build models ─────────────────────────────────────────────────────────

println("Building deterministic equivalent (T=$(T_STAGES) stages)...")
det_eq, spi, spo, unc_samples, init_state = build_inventory_det_equivalent(;
    T=T_STAGES, K=K_FIXED, c=C_UNIT, h=H_HOLD, p=P_BACK,
    Q_max=Q_MAX_VAL, I_0=I_INIT, d_min=D_MIN_VAL, d_max=D_MAX_VAL,
    num_scenarios=30, penalty=PENALTY_VAL, seed=42,
)

println("Building stage-wise subproblems (for evaluation)...")
subproblems, spi_eval, spo_eval, unc_eval, _ = build_inventory_subproblems(;
    T=T_STAGES, K=K_FIXED, c=C_UNIT, h=H_HOLD, p=P_BACK,
    Q_max=Q_MAX_VAL, I_0=I_INIT, d_min=D_MIN_VAL, d_max=D_MAX_VAL,
    num_scenarios=100, penalty=PENALTY_VAL, seed=99,
)

# ── Policy ───────────────────────────────────────────────────────────────
# Input: [demand_t, net_inventory_{t-1}]  →  target net inventory ŝ_t

Random.seed!(2024)
policy = Chain(
    Dense(2, 16, relu),
    Dense(16, 8, relu),
    Dense(8, 1),
)

# ── Directories ──────────────────────────────────────────────────────────

example_dir = @__DIR__
model_dir   = joinpath(example_dir, "models")
result_dir  = joinpath(example_dir, "results")
mkpath(model_dir)
mkpath(result_dir)

model_path = joinpath(model_dir, "inventory_policy.jld2")
curve_path = joinpath(result_dir, "training_curve.csv")

# ── Pre-training baseline ────────────────────────────────────────────────

Random.seed!(111)
pre_costs = [
    simulate_multistage(
        det_eq, spi, spo,
        init_state,
        sample(unc_samples),
        policy;
        integer_strategy = FixedDiscreteIntegerStrategy(),
    )
    for _ in 1:10
]
pre_mean = mean(pre_costs)
println("Pre-training mean cost (det-eq, w/ deficit): $(round(pre_mean, digits=2))")

save_best = SaveBest(pre_mean, model_path)
training_log = DataFrame(batch=Int[], loss=Float64[])

# ── Train ────────────────────────────────────────────────────────────────

println("\nTraining (FixedDiscreteIntegerStrategy, det-eq)...")
Random.seed!(2024)
train_multistage(
    policy,
    init_state,
    det_eq,
    spi,
    spo,
    unc_samples;
    num_batches         = 500,
    num_train_per_batch = 8,
    optimizer           = Flux.Adam(0.001),
    integer_strategy    = FixedDiscreteIntegerStrategy(),
    penalty_schedule    = [(1, 50, 0.5), (51, 500, 1.0)],  # mild warm-up, then 1×
    record              = (sample_log, iter, model) -> begin
        loss = isempty(sample_log.objectives_no_deficit) ? NaN :
               mean(sample_log.objectives_no_deficit)
        push!(training_log, (batch=iter, loss=loss))
        if mod(iter, 50) == 0 || iter == 1
            println("  Batch $(lpad(iter,3)) / 500  loss = $(round(loss, digits=2))")
        end
        save_best(iter, model, loss)
        return false
    end,
)

CSV.write(curve_path, training_log)
println("Training curve saved → $(relpath(curve_path, example_dir))")
println("Best model saved     → $(relpath(model_path, example_dir))")

# ── Load best weights ────────────────────────────────────────────────────

model_state = JLD2.load(model_path, "model_state")
Flux.loadmodel!(policy, model_state)

# ── Evaluate trained policy on test scenarios ────────────────────────────

println("\nEvaluating trained policy on 100 test scenarios...")
Random.seed!(555)

n_test   = 100
traj_inv = Matrix{Float64}(undef, n_test, T_STAGES + 1)  # net inventory
traj_z   = Matrix{Float64}(undef, n_test, T_STAGES)       # order indicator
traj_q   = Matrix{Float64}(undef, n_test, T_STAGES)       # order quantity
op_costs = Vector{Float64}(undef, n_test)                  # operational cost

for s in 1:n_test
    unc_sample = sample(unc_eval)
    state      = Float64.(init_state)
    traj_inv[s, 1] = state[1]
    op_costs[s] = 0.0

    for t in 1:T_STAGES
        d_val  = unc_sample[t][1][2]
        target = Float64(policy(Float32[d_val, state[1]])[1])

        # Set parameters, then solve MIP for evaluation.
        set_parameter_value(spi_eval[t][1], state[1])
        set_parameter_value(unc_sample[t][1][1], d_val)
        set_parameter_value(spo_eval[t][1][1], target)
        optimize!(subproblems[t])

        z_val = value(subproblems[t][:z])
        q_val = value(subproblems[t][:q])
        s_val = value(subproblems[t][:s_out])
        # Operational cost: K·z + c·q + h·I_t + p·B_t (no deficit penalty)
        op_costs[s] += K_FIXED * z_val + C_UNIT * q_val +
                       H_HOLD * max(s_val, 0.0) + P_BACK * max(-s_val, 0.0)

        traj_z[s, t]     = z_val
        traj_q[s, t]     = q_val
        traj_inv[s, t+1] = s_val
        state = [s_val]
    end
end

# Save trajectories
df_inv = DataFrame(traj_inv, [Symbol("t$i") for i in 0:T_STAGES])
df_inv[!, :scenario] = 1:n_test
CSV.write(joinpath(result_dir, "dr_trajectories.csv"), df_inv)

df_costs = DataFrame(scenario=1:n_test, operational_cost=op_costs)
CSV.write(joinpath(result_dir, "dr_costs.csv"), df_costs)

println("Mean operational cost (trained):    $(round(mean(op_costs), digits=2))")
println("Std  operational cost (trained):    $(round(std(op_costs),  digits=2))")
println("Trajectories saved → results/dr_trajectories.csv")
println("Costs saved        → results/dr_costs.csv")
