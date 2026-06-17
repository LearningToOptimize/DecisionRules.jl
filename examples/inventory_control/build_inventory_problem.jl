"""
Stochastic uncapacitated lot-sizing (SULS) problem.

Reference: arXiv:1701.04102.

At each stage t the decision-maker chooses whether to place an order (binary
z_t ∈ {0,1}) and how much to order (q_t ≥ 0, q_t ≤ Q_max · z_t).
Demand d_t is random and revealed at the start of each period.

Net inventory:     s_t = s_{t-1} + q_t - d_t
Inventory on hand: I_t = max(s_t, 0)
Backlog:           B_t = max(-s_t, 0)

Stage-t cost: K·z_t + c·q_t + h·I_t + p·B_t
  K  = fixed ordering cost
  c  = unit ordering cost
  h  = holding cost per unit per period
  p  = backlog penalty per unit per period

The TS-DDR policy predicts a target net inventory ŝ_t at each stage.
A deficit penalty |s_t - ŝ_t| is added to encourage tracking.

Training uses `FixedDiscreteIntegerStrategy`:
  1. Solve the MIP to get incumbent binary decisions z*_t.
  2. Fix z_t = z*_t and relax integrality.
  3. Re-solve the resulting LP.
  4. Read LP duals for the target parameters ŝ_t as gradient signal.
"""

using JuMP
import MathOptInterface as MOI
using HiGHS
using DecisionRules
using Random

# ─────────────────────────────────────────────────────────────────────────────
# Stage-wise subproblems
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_inventory_subproblems(; T, K, c, h, p, Q_max, I_0, d_min, d_max,
                                  num_scenarios, penalty, seed)

Return T per-stage JuMP models for the stochastic lot-sizing problem.
Each model contains a binary ordering variable z and is compatible with
`FixedDiscreteIntegerStrategy` for stage-wise rollout.

Returns:
  `subproblems, state_params_in, state_params_out, uncertainty_samples, [I_0]`
"""
function build_inventory_subproblems(;
    T            = 12,
    K            = 30.0,
    c            = 2.0,
    h            = 1.0,
    p            = 10.0,
    Q_max        = 80.0,
    I_0          = 20.0,
    d_min        = 10.0,
    d_max        = 30.0,
    num_scenarios = 30,
    penalty      = 500.0,
    seed         = 42,
)
    rng = MersenneTwister(seed)
    demand_scenarios = [d_min .+ (d_max - d_min) .* rand(rng, num_scenarios) for _ in 1:T]

    subproblems      = Vector{JuMP.Model}(undef, T)
    state_params_in  = Vector{Vector{Any}}(undef, T)
    state_params_out = Vector{Vector{Tuple{Any, VariableRef}}}(undef, T)
    uncertainty_samples = Vector{Vector{Tuple{VariableRef, Vector{Float64}}}}(undef, T)

    for t in 1:T
        m = Model(optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false))
        set_silent(m)

        @variable(m, z, Bin)               # order indicator
        @variable(m, 0 <= q <= Q_max)      # order quantity
        @variable(m, inv_hold >= 0)        # on-hand inventory
        @variable(m, back >= 0)            # backlog
        @variable(m, s_out)               # net inventory (realized state)

        @variable(m, s_in in MOI.Parameter(I_0))
        @variable(m, demand in MOI.Parameter((d_min + d_max) / 2))
        @variable(m, s_target in MOI.Parameter(I_0))

        @constraint(m, q <= Q_max * z)               # order only if z = 1
        @constraint(m, s_out == s_in + q - demand)   # balance
        @constraint(m, inv_hold - back == s_out)      # on-hand / backlog split

        # Set base objective BEFORE create_deficit! so set_objective_coefficient works correctly.
        @objective(m, Min, K * z + c * q + h * inv_hold + p * back)
        _, _def = create_deficit!(m, 1; penalty_l1=penalty)
        @constraint(m, _def[1] == s_out - s_target)

        subproblems[t]         = m
        state_params_in[t]     = Any[s_in]
        state_params_out[t]    = [(s_target, s_out)]
        uncertainty_samples[t] = [(demand, demand_scenarios[t])]
    end

    return subproblems, state_params_in, state_params_out, uncertainty_samples, [I_0]
end

# ─────────────────────────────────────────────────────────────────────────────
# Deterministic equivalent (full T-stage model)
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_inventory_det_equivalent(; T, K, c, h, p, Q_max, I_0, d_min, d_max,
                                     num_scenarios, penalty, seed)

Build the full T-stage deterministic-equivalent model for the stochastic
lot-sizing problem.  All T binary ordering decisions z_t are in a single
JuMP model.  Demand and policy targets are MOI parameters.

`state_params_in[1]`  = the initial-inventory parameter (set to initial_state[1]).
`state_params_in[t]`  for t > 1 = s_net[t-1] (a variable; not re-set each call).
`state_params_out[t]` = `(s_target[t], s_net[t])`.

This model is intended for training with `train_multistage` + `FixedDiscreteIntegerStrategy`.

Returns:
  `det_eq, state_params_in, state_params_out, uncertainty_samples, [I_0]`
"""
function build_inventory_det_equivalent(;
    T            = 12,
    K            = 30.0,
    c            = 2.0,
    h            = 1.0,
    p            = 10.0,
    Q_max        = 80.0,
    I_0          = 20.0,
    d_min        = 10.0,
    d_max        = 30.0,
    num_scenarios = 30,
    penalty      = 500.0,
    seed         = 42,
)
    rng = MersenneTwister(seed)
    demand_scenarios = [d_min .+ (d_max - d_min) .* rand(rng, num_scenarios) for _ in 1:T]

    m = Model(optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false))
    set_silent(m)

    @variable(m, z[1:T], Bin)
    @variable(m, 0 <= q[1:T] <= Q_max)
    @variable(m, inv_hold[1:T] >= 0)
    @variable(m, back[1:T] >= 0)
    @variable(m, s_net[1:T])              # net inventory per stage

    # Parameters
    @variable(m, s_init in MOI.Parameter(I_0))
    @variable(m, demand[t=1:T] in MOI.Parameter((d_min + d_max) / 2))
    @variable(m, s_target[t=1:T] in MOI.Parameter(I_0))

    # Ordering constraint
    @constraint(m, [t=1:T], q[t] <= Q_max * z[t])

    # Inventory balance across stages
    @constraint(m, s_net[1] == s_init + q[1] - demand[1])
    @constraint(m, [t=2:T], s_net[t] == s_net[t-1] + q[t] - demand[t])

    # On-hand / backlog split
    @constraint(m, [t=1:T], inv_hold[t] - back[t] == s_net[t])

    # Target-tracking deficit — one per stage, created in bulk to avoid name
    # conflicts that occur when create_deficit! is called in a loop on the same model.
    # base_name="norm_deficit" gives names norm_deficit[1], ..., norm_deficit[T]
    # so get_objective_no_target_deficit (which searches for "norm_deficit") works.
    @variable(m, norm_deficit_arr[1:T] >= 0.0, base_name="norm_deficit")
    @variable(m, _deficit_arr[1:T])

    @constraint(m, [t=1:T], _deficit_arr[t] == s_net[t] - s_target[t])
    @constraint(m, [t=1:T], [norm_deficit_arr[t]; _deficit_arr[t:t]] in MOI.NormOneCone(2))

    @objective(
        m, Min,
        sum(K * z[t] + c * q[t] + h * inv_hold[t] + p * back[t] for t in 1:T)
        + penalty * sum(norm_deficit_arr),
    )

    # ── State-param structure ────────────────────────────────────────────────
    state_params_in  = Vector{Vector{Any}}(undef, T)
    state_params_out = Vector{Vector{Tuple{Any, VariableRef}}}(undef, T)
    uncertainty_params = Vector{Vector{Tuple{VariableRef, Vector{Float64}}}}(undef, T)

    state_params_in[1] = Any[s_init]
    for t in 2:T
        # s_net[t-1] is a JuMP variable; _set_multistage_parameters! only calls
        # set_parameter_value for t = 1 (initial state), so this is safe.
        state_params_in[t] = Any[s_net[t-1]]
    end
    for t in 1:T
        state_params_out[t]    = [(s_target[t], s_net[t])]
        uncertainty_params[t]  = [(demand[t], demand_scenarios[t])]
    end

    return m, state_params_in, state_params_out, uncertainty_params, [I_0]
end

# ─────────────────────────────────────────────────────────────────────────────
# Baseline: simple base-stock policy
# ─────────────────────────────────────────────────────────────────────────────

"""
    base_stock_policy(S_star) -> policy

Return a base-stock policy that orders up to `S_star` units of net inventory.
  - If s_{t-1} < S_star: order q = min(S_star - s_{t-1}, Q_max) and set z = 1.
  - Otherwise: do nothing (z = 0, q = 0).

The policy is returned as a function `(input_vec) -> [S_star]` so it can be
used directly with `simulate_multistage`.  (The target is always `S_star`
regardless of input; the subproblem optimizes q subject to z.)
"""
function base_stock_policy(S_star::Float64)
    return (_) -> Float32[S_star]
end
