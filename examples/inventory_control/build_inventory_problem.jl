"""
Latent ex-ante stochastic lot-sizing problem.

Demand has a latent seasonal phase, persistent high/low regimes, and
autocorrelated shocks — none of which are directly observed. The policy
sees only inventory and realized demand history.

Two formulations controlled by the `integer` keyword:
- Relaxed (integer=false): q ∈ [0, Q_max], cost = c·q + h·hold + p·back
- Integer (integer=true):  z ∈ {0,1}, q ≤ Q_max·z, cost = K·z + c·q + h·hold + p·back
"""

using JuMP
import MathOptInterface as MOI
using HiGHS
using DecisionRules
using Flux
using Functors
using Random

# ---------------------------------------------------------------------------
# Problem constants
# ---------------------------------------------------------------------------
const INVENTORY_T = 12
const INVENTORY_K = 500.0
const INVENTORY_C = 2.0
const INVENTORY_H = 1.0
const INVENTORY_P = 25.0
const INVENTORY_Q_MAX = 350.0
const INVENTORY_I0 = 30.0
const INVENTORY_LAST_DEMAND0 = 0.0
const INVENTORY_PREV_DEMAND0 = 0.0
const INVENTORY_PENALTY = 75.0

# Wider seasonal bands (≈1.3× original)
const D_LO = Float64[18, 23, 34, 49, 70, 86, 75, 57, 40, 30, 22, 17]
const D_HI = Float64[44, 55, 81, 112, 153, 185, 164, 122, 88, 65, 49, 40]

# ---------------------------------------------------------------------------
# Latent demand process (stronger structure than original)
# ---------------------------------------------------------------------------
function sample_inventory_demand_path(rng::AbstractRNG=Random.default_rng())
    phase_shift = rand(rng, 0:(INVENTORY_T - 1))
    regime = rand(rng, (-1.0, 0.0, 1.0))
    shock = 0.0
    path = Vector{Float64}(undef, INVENTORY_T)
    for t in 1:INVENTORY_T
        seasonal_t = mod1(t + phase_shift, INVENTORY_T)
        if rand(rng) < 0.04
            regime = rand(rng, (-1.0, 0.0, 1.0))
        end
        shock = 0.92 * shock + 0.35 * randn(rng)
        center = (D_LO[seasonal_t] + D_HI[seasonal_t]) / 2
        half_width = (D_HI[seasonal_t] - D_LO[seasonal_t]) / 2
        demand = center + half_width * (0.85 * regime + 0.42 * shock + 0.12 * randn(rng))
        path[t] = clamp(demand, 5.0, D_HI[seasonal_t] + 0.55 * half_width)
    end
    return path
end

# ---------------------------------------------------------------------------
# DecisionRules sampler
# ---------------------------------------------------------------------------
struct InventoryProcessSampler
    params::Vector{VariableRef}
end

function DecisionRules.sample(sampler::InventoryProcessSampler)
    demand_path = sample_inventory_demand_path()
    return [[(sampler.params[t], demand_path[t])] for t in 1:INVENTORY_T]
end

# ---------------------------------------------------------------------------
# Stage-wise subproblems
# ---------------------------------------------------------------------------
function build_inventory_subproblems(;
    T = INVENTORY_T,
    K = INVENTORY_K,
    c = INVENTORY_C,
    h = INVENTORY_H,
    p = INVENTORY_P,
    Q_max = INVENTORY_Q_MAX,
    I_0 = INVENTORY_I0,
    num_scenarios = 100,
    penalty = INVENTORY_PENALTY,
    seed = 42,
    integer = true,
)
    Random.seed!(seed)
    subproblems = Vector{JuMP.Model}(undef, T)
    state_params_in = Vector{Vector{Any}}(undef, T)
    state_params_out = Vector{Vector{Tuple{Any,VariableRef}}}(undef, T)
    uncertainty_params = Vector{VariableRef}(undef, T)

    for t in 1:T
        m = Model(optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false))
        set_silent(m)

        @variable(m, 0 <= q <= Q_max)
        @variable(m, s_mid)
        @variable(m, s_out)
        @variable(m, inv_hold >= 0)
        @variable(m, back >= 0)
        @variable(m, last_demand_out)
        @variable(m, prev_demand_out)

        @variable(m, s_in in MOI.Parameter(I_0))
        @variable(m, last_demand_in in MOI.Parameter(INVENTORY_LAST_DEMAND0))
        @variable(m, prev_demand_in in MOI.Parameter(INVENTORY_PREV_DEMAND0))
        @variable(m, demand in MOI.Parameter((D_LO[t] + D_HI[t]) / 2))
        @variable(m, s_target in MOI.Parameter(I_0))
        @variable(m, last_demand_target in MOI.Parameter(INVENTORY_LAST_DEMAND0))
        @variable(m, prev_demand_target in MOI.Parameter(INVENTORY_PREV_DEMAND0))

        if integer
            @variable(m, z, Bin)
            @constraint(m, q <= Q_max * z)
            @objective(m, Min, K * z + c * q + h * inv_hold + p * back)
        else
            @objective(m, Min, c * q + h * inv_hold + p * back)
        end

        @constraint(m, s_mid == s_in + q)
        @constraint(m, s_out == s_mid - demand)
        @constraint(m, last_demand_out == demand)
        @constraint(m, prev_demand_out == last_demand_in)
        @constraint(m, inv_hold - back == s_out)

        _, deficit = create_deficit!(m, 1; penalty_l1=penalty)
        @constraint(m, deficit[1] == s_mid - s_target)

        subproblems[t] = m
        state_params_in[t] = Any[s_in, last_demand_in, prev_demand_in]
        state_params_out[t] = [
            (s_target, s_out),
            (last_demand_target, last_demand_out),
            (prev_demand_target, prev_demand_out),
        ]
        uncertainty_params[t] = demand
    end

    return subproblems, state_params_in, state_params_out,
           InventoryProcessSampler(uncertainty_params),
           [I_0, INVENTORY_LAST_DEMAND0, INVENTORY_PREV_DEMAND0]
end

# ---------------------------------------------------------------------------
# Deterministic equivalent (full-horizon)
# ---------------------------------------------------------------------------
function build_inventory_det_equivalent(;
    T = INVENTORY_T,
    K = INVENTORY_K,
    c = INVENTORY_C,
    h = INVENTORY_H,
    p = INVENTORY_P,
    Q_max = INVENTORY_Q_MAX,
    I_0 = INVENTORY_I0,
    num_scenarios = 100,
    penalty = INVENTORY_PENALTY,
    seed = 42,
    integer = true,
)
    Random.seed!(seed)
    m = Model(optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false))
    set_silent(m)

    @variable(m, 0 <= q[1:T] <= Q_max)
    @variable(m, s_mid[1:T])
    @variable(m, s_out[1:T])
    @variable(m, last_demand_out[1:T])
    @variable(m, prev_demand_out[1:T])
    @variable(m, inv_hold[1:T] >= 0)
    @variable(m, back[1:T] >= 0)

    @variable(m, s_init in MOI.Parameter(I_0))
    @variable(m, last_demand_init in MOI.Parameter(INVENTORY_LAST_DEMAND0))
    @variable(m, prev_demand_init in MOI.Parameter(INVENTORY_PREV_DEMAND0))
    @variable(m, demand[t=1:T] in MOI.Parameter((D_LO[t] + D_HI[t]) / 2))
    @variable(m, s_target[t=1:T] in MOI.Parameter(I_0))
    @variable(m, last_demand_target[t=1:T] in MOI.Parameter(INVENTORY_LAST_DEMAND0))
    @variable(m, prev_demand_target[t=1:T] in MOI.Parameter(INVENTORY_PREV_DEMAND0))

    if integer
        @variable(m, z[1:T], Bin)
        @constraint(m, [t=1:T], q[t] <= Q_max * z[t])
    end

    @constraint(m, s_mid[1] == s_init + q[1])
    @constraint(m, [t=2:T], s_mid[t] == s_out[t-1] + q[t])
    @constraint(m, [t=1:T], s_out[t] == s_mid[t] - demand[t])
    @constraint(m, [t=1:T], last_demand_out[t] == demand[t])
    @constraint(m, prev_demand_out[1] == prev_demand_init)
    @constraint(m, [t=2:T], prev_demand_out[t] == last_demand_out[t-1])
    @constraint(m, [t=1:T], inv_hold[t] - back[t] == s_out[t])

    @variable(m, norm_deficit_arr[1:T] >= 0.0)
    @variable(m, deficit_arr[1:T])
    @constraint(m, [t=1:T], deficit_arr[t] == s_mid[t] - s_target[t])
    @constraint(m, [t=1:T], [norm_deficit_arr[t]; deficit_arr[t:t]] in MOI.NormOneCone(2))

    if integer
        @objective(m, Min,
            sum(K * z[t] + c * q[t] + h * inv_hold[t] + p * back[t] for t in 1:T) +
            penalty * sum(norm_deficit_arr))
    else
        @objective(m, Min,
            sum(c * q[t] + h * inv_hold[t] + p * back[t] for t in 1:T) +
            penalty * sum(norm_deficit_arr))
    end

    state_params_in = Vector{Vector{Any}}(undef, T)
    state_params_out = Vector{Vector{Tuple{Any,VariableRef}}}(undef, T)
    uncertainty_params = Vector{VariableRef}(undef, T)

    state_params_in[1] = Any[s_init, last_demand_init, prev_demand_init]
    for t in 2:T
        state_params_in[t] = Any[s_out[t-1], last_demand_out[t-1], prev_demand_out[t-1]]
    end
    for t in 1:T
        state_params_out[t] = [
            (s_target[t], s_out[t]),
            (last_demand_target[t], last_demand_out[t]),
            (prev_demand_target[t], prev_demand_out[t]),
        ]
        uncertainty_params[t] = demand[t]
    end

    return m, state_params_in, state_params_out,
           InventoryProcessSampler(uncertainty_params),
           [I_0, INVENTORY_LAST_DEMAND0, INVENTORY_PREV_DEMAND0]
end

# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
function base_stock_policy(S_star::Float64)
    return x -> Float32[S_star, x[1], x[2]]
end

struct ExAnteInventoryPolicy{N}
    net::N
end

Functors.@functor ExAnteInventoryPolicy (net,)

function (policy::ExAnteInventoryPolicy)(x)
    current_demand = Float32(x[1])
    inventory = Float32(x[2])
    last_demand = Float32(x[3])
    prev_demand = Float32(x[4])
    order_features = Float32[inventory / 100, last_demand / 100, prev_demand / 100]
    target = 500f0 .* Flux.sigmoid.(policy.net(order_features))
    return Float32[target[1], current_demand, last_demand]
end

function build_exante_policy(; seed=2024)
    Random.seed!(seed)
    net = Chain(
        Dense(3, 32, relu),
        Dense(32, 24, relu),
        Dense(24, 1),
    )
    return ExAnteInventoryPolicy(net)
end
