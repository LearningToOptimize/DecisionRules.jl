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

"""
    sample_inventory_demand_path(rng = Random.default_rng()) -> Vector{Float64}

Draw one realization of the latent demand process over `T` periods.

The process has three hidden components:
- a seasonal phase shift (uniform in 0:T-1);
- a persistent regime ∈ {-1, 0, 1} with 4% switching probability;
- an AR(1) shock with coefficient 0.92.

None of these are observed by the policy — only realized demand values.

# Arguments
- `rng::AbstractRNG`: random number generator.

# Examples
```julia
path = sample_inventory_demand_path()
```
"""
function sample_inventory_demand_path(rng::AbstractRNG=Random.default_rng())
    # Draw a random seasonal phase offset so demand peaks at different months.
    phase_shift = rand(rng, 0:(INVENTORY_T - 1))

    # Draw the initial demand regime (low / neutral / high).
    regime = rand(rng, (-1.0, 0.0, 1.0))

    # Initialize the AR(1) shock ε₀ = 0.
    shock = 0.0

    # Pre-allocate the demand path vector.
    path = Vector{Float64}(undef, INVENTORY_T)

    for t in 1:INVENTORY_T
        # Map period t to the shifted seasonal index κ_t = 1 + ((t+φ-1) mod T).
        seasonal_t = mod1(t + phase_shift, INVENTORY_T)

        # With 4% probability, jump to a new demand regime.
        if rand(rng) < 0.04
            regime = rand(rng, (-1.0, 0.0, 1.0))
        end

        # Update the AR(1) shock: ε_t = 0.92 ε_{t-1} + 0.35 η_t.
        shock = 0.92 * shock + 0.35 * randn(rng)

        # Compute the seasonal center μ_κ = (D_LO[κ] + D_HI[κ]) / 2.
        center = (D_LO[seasonal_t] + D_HI[seasonal_t]) / 2

        # Compute the seasonal half-width w_κ = (D_HI[κ] - D_LO[κ]) / 2.
        half_width = (D_HI[seasonal_t] - D_LO[seasonal_t]) / 2

        # Demand: d_t = μ_κ + w_κ · (0.85·regime + 0.42·shock + 0.12·noise).
        demand = center + half_width * (0.85 * regime + 0.42 * shock + 0.12 * randn(rng))

        # Clip demand to [5, D_HI[κ] + 0.55·w_κ] to prevent negative or extreme values.
        path[t] = clamp(demand, 5.0, D_HI[seasonal_t] + 0.55 * half_width)
    end

    return path
end

# ---------------------------------------------------------------------------
# DecisionRules sampler
# ---------------------------------------------------------------------------

"""
    InventoryProcessSampler

Wraps JuMP demand parameters so `DecisionRules.sample` returns uncertainty
realizations in the format expected by `train_multistage` and
`simulate_multistage`.

# Fields
- `params::Vector{VariableRef}`: one demand parameter per stage.
"""
struct InventoryProcessSampler
    params::Vector{VariableRef}
end

"""
    DecisionRules.sample(sampler::InventoryProcessSampler)

Draw one demand path and return it as a vector of `[(param, value)]` pairs.
"""
function DecisionRules.sample(sampler::InventoryProcessSampler)
    # Draw a fresh demand path from the latent process.
    demand_path = sample_inventory_demand_path()

    # Pair each stage's JuMP parameter with the sampled demand value.
    return [[(sampler.params[t], demand_path[t])] for t in 1:INVENTORY_T]
end

# ---------------------------------------------------------------------------
# Stage-wise subproblems
# ---------------------------------------------------------------------------

"""
    build_inventory_subproblems(; kwargs...) -> (subproblems, state_in, state_out, sampler, x0)

Build `T` independent JuMP stage models for stage-wise rollout evaluation.

Each model has demand as a parameter, input state `(inventory, d_{t-1},
d_{t-2})`, and a target constraint on mid-period inventory `s_mid` using
`create_deficit!`.

Returns the five-tuple expected by `simulate_multistage` and
`train_multistage`.

# Keyword Arguments
- `T`, `K`, `c`, `h`, `p`, `Q_max`: problem parameters.
- `I_0`: initial inventory.
- `num_scenarios`: number of uncertainty samples per SGD batch.
- `penalty`: target-deficit penalty λ (ignored when `strict = true`).
- `seed`: RNG seed for demand sampling.
- `integer`: whether to include binary setup variable z.
- `strict`: replace the L1 target penalty with the hard equality
  `s_mid == s_target` (no deficit variable, no penalty hyperparameter).
  Requires a policy whose targets are always one-stage reachable, i.e.
  `s_target ∈ [s_in, s_in + Q_max]` — see [`InventoryReachablePolicy`](@ref).
  The demand pass-through targets stay unconstrained in both modes: their
  reachable sets are singletons (the observed demands), so constraining them
  adds nothing and predicting them is not a decision.
"""
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
    strict = false,
)
    # Fix the random seed so demand samples are reproducible.
    Random.seed!(seed)

    # Pre-allocate one JuMP model per stage.
    subproblems = Vector{JuMP.Model}(undef, T)

    # Each stage has 3 input-state parameters: (inventory, d_{t-1}, d_{t-2}).
    state_params_in = Vector{Vector{Any}}(undef, T)

    # Each stage has 3 output pairs: (target_param, realized_variable).
    state_params_out = Vector{Vector{Tuple{Any,VariableRef}}}(undef, T)

    # One demand parameter per stage.
    uncertainty_params = Vector{VariableRef}(undef, T)

    for t in 1:T
        # Create a HiGHS LP/MIP model for this stage.
        m = Model(optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false))
        set_silent(m)

        # --- Decision variables ---
        # q: order quantity, bounded by capacity Q_max.
        @variable(m, 0 <= q <= Q_max)
        # s_mid: mid-period inventory after order arrives but before demand.
        @variable(m, s_mid)
        # s_out: end-of-period inventory after demand realizes.
        @variable(m, s_out)
        # inv_hold: positive part of s_out (holding cost component).
        @variable(m, inv_hold >= 0)
        # back: negative part of s_out (backlog cost component).
        @variable(m, back >= 0)
        # Demand pass-through to the next stage state.
        @variable(m, last_demand_out)
        @variable(m, prev_demand_out)

        # --- Parametric inputs (set before each solve) ---
        # s_in: incoming inventory from the previous stage.
        @variable(m, s_in in MOI.Parameter(I_0))
        # last_demand_in: demand observed one period ago (part of state).
        @variable(m, last_demand_in in MOI.Parameter(INVENTORY_LAST_DEMAND0))
        # prev_demand_in: demand observed two periods ago (part of state).
        @variable(m, prev_demand_in in MOI.Parameter(INVENTORY_PREV_DEMAND0))
        # demand: current-period demand realization (uncertainty).
        @variable(m, demand in MOI.Parameter((D_LO[t] + D_HI[t]) / 2))
        # s_target: target mid-period inventory from the policy.
        @variable(m, s_target in MOI.Parameter(I_0))
        # Target pass-throughs for demand state entries.
        @variable(m, last_demand_target in MOI.Parameter(INVENTORY_LAST_DEMAND0))
        @variable(m, prev_demand_target in MOI.Parameter(INVENTORY_PREV_DEMAND0))

        if integer
            # z ∈ {0,1}: binary setup decision.
            @variable(m, z, Bin)
            # If z = 0, no order is allowed: q ≤ Q_max · z.
            @constraint(m, q <= Q_max * z)
            # Objective: K·z + c·q + h·hold + p·backlog.
            @objective(m, Min, K * z + c * q + h * inv_hold + p * back)
        else
            # Relaxed objective: no setup cost or binary variable.
            @objective(m, Min, c * q + h * inv_hold + p * back)
        end

        # s_mid = s_in + q: order arrives before demand.
        @constraint(m, s_mid == s_in + q)
        # s_out = s_mid - demand: demand subtracts from inventory.
        @constraint(m, s_out == s_mid - demand)
        # Pass current demand to next stage as "last demand".
        @constraint(m, last_demand_out == demand)
        # Pass previous "last demand" to next stage as "prev demand".
        @constraint(m, prev_demand_out == last_demand_in)
        # Split end-of-period inventory into holding and backlog parts.
        @constraint(m, inv_hold - back == s_out)

        if strict
            # Strict mode: hard equality, no slack, no penalty term. The dual
            # of this constraint is the pure shadow price ∂q_t/∂ŝ_target.
            # Feasible iff s_target ∈ [s_in, s_in + Q_max] (exactly the
            # one-stage reachable set of s_mid = s_in + q with q ∈ [0, Q_max];
            # s_out and the hold/backlog split are free, so no other
            # constraint restricts s_mid).
            @constraint(m, s_mid == s_target)
        else
            # L1 target-deficit penalty: λ · |s_mid - ŝ_target|.
            _, deficit = create_deficit!(m, 1; penalty_l1=penalty)
            @constraint(m, deficit[1] == s_mid - s_target)
        end

        # Store the model and parameter mappings.
        subproblems[t] = m
        state_params_in[t] = Any[s_in, last_demand_in, prev_demand_in]
        state_params_out[t] = [
            (s_target, s_out),
            (last_demand_target, last_demand_out),
            (prev_demand_target, prev_demand_out),
        ]
        uncertainty_params[t] = demand
    end

    # Return the five-tuple: (models, state_in, state_out, sampler, x0).
    return subproblems, state_params_in, state_params_out,
           InventoryProcessSampler(uncertainty_params),
           [I_0, INVENTORY_LAST_DEMAND0, INVENTORY_PREV_DEMAND0]
end

# ---------------------------------------------------------------------------
# Deterministic equivalent (full-horizon)
# ---------------------------------------------------------------------------

"""
    build_inventory_det_equivalent(; kwargs...) -> (model, state_in, state_out, sampler, x0)

Build a single JuMP model coupling all `T` stages for direct transcription
training.

The deterministic equivalent jointly optimizes over the full horizon. Target
constraints appear as NormOneCone (L1) penalties so the training gradient
captures inter-stage cost coupling that stage-wise rollouts miss.

The penalty term is

```math
\\lambda \\sum_{t=1}^{T} |s_t^{mid} - \\hat{s}_t| .
```

# Keyword Arguments
- `T`, `K`, `c`, `h`, `p`, `Q_max`: problem parameters.
- `I_0`: initial inventory.
- `num_scenarios`: number of uncertainty samples per SGD batch.
- `penalty`: target-deficit penalty ``\\lambda`` (ignored when `strict = true`).
- `seed`: RNG seed for demand sampling.
- `integer`: whether to include binary setup variable z.
- `strict`: replace the L1 penalty with hard equalities
  ``s^{mid}_t = \\hat{s}_t`` and drop the penalty term from the objective.
  **Feasibility caveat**: unlike the hydro case, the target (order-up-to
  position ``s^{mid}``) and the carried state (post-demand inventory
  ``s^{out} = s^{mid} - d``) are *different quantities*, so the standard
  target-feedback DE recursion hands the policy ``\\hat{s}^{mid}_{t-1}``
  where a reachable policy expects ``s^{out}_{t-1}`` — the induction
  guarantee does NOT transfer naively. Either train stage-wise (what the
  `strict` variants do; see `build_training_problem`) or reconstruct the
  realized inventory inside the recursion as
  ``s^{out}_{t-1} = \\hat{s}_{t-1} - d_{t-1}`` (target minus the demand
  pass-through) before computing reachable bounds.

# Examples
```julia
model, spi, spo, sampler, x0 = build_inventory_det_equivalent(;
    num_scenarios = 50,
    integer = true,
)
```
"""
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
    strict = false,
)
    # Fix the random seed so demand samples are reproducible.
    Random.seed!(seed)

    # One monolithic model for the entire T-period horizon.
    m = Model(optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false))
    set_silent(m)

    # --- Decision variables (one per stage, indexed 1:T) ---
    # q[t]: order quantity in period t.
    @variable(m, 0 <= q[1:T] <= Q_max)
    # s_mid[t]: mid-period inventory after order arrives.
    @variable(m, s_mid[1:T])
    # s_out[t]: end-of-period inventory after demand.
    @variable(m, s_out[1:T])
    # Demand pass-through state variables.
    @variable(m, last_demand_out[1:T])
    @variable(m, prev_demand_out[1:T])
    # Holding and backlog split of s_out.
    @variable(m, inv_hold[1:T] >= 0)
    @variable(m, back[1:T] >= 0)

    # --- Parametric inputs (set before each DE solve) ---
    # Initial state at t = 0.
    @variable(m, s_init in MOI.Parameter(I_0))
    @variable(m, last_demand_init in MOI.Parameter(INVENTORY_LAST_DEMAND0))
    @variable(m, prev_demand_init in MOI.Parameter(INVENTORY_PREV_DEMAND0))
    # Demand realizations for each period (set per scenario).
    @variable(m, demand[t=1:T] in MOI.Parameter((D_LO[t] + D_HI[t]) / 2))
    # Target state from the policy (set per scenario).
    @variable(m, s_target[t=1:T] in MOI.Parameter(I_0))
    @variable(m, last_demand_target[t=1:T] in MOI.Parameter(INVENTORY_LAST_DEMAND0))
    @variable(m, prev_demand_target[t=1:T] in MOI.Parameter(INVENTORY_PREV_DEMAND0))

    if integer
        # z[t] ∈ {0,1}: binary setup decision.
        @variable(m, z[1:T], Bin)
        # q[t] ≤ Q_max · z[t]: no order if setup is off.
        @constraint(m, [t=1:T], q[t] <= Q_max * z[t])
    end

    # --- Dynamics ---
    # First stage links to the initial inventory parameter.
    @constraint(m, s_mid[1] == s_init + q[1])
    # Subsequent stages chain from the previous end-of-period inventory.
    @constraint(m, [t=2:T], s_mid[t] == s_out[t-1] + q[t])
    # Demand subtracts from mid-period inventory.
    @constraint(m, [t=1:T], s_out[t] == s_mid[t] - demand[t])
    # Pass demand through to state for the next stage.
    @constraint(m, [t=1:T], last_demand_out[t] == demand[t])
    @constraint(m, prev_demand_out[1] == prev_demand_init)
    @constraint(m, [t=2:T], prev_demand_out[t] == last_demand_out[t-1])
    # Split end-of-period inventory into holding and backlog.
    @constraint(m, [t=1:T], inv_hold[t] - back[t] == s_out[t])

    if strict
        # --- Strict targets: hard equalities, no deficit machinery ---
        # Feasible for any target path with ŝ_t ∈ [s_{t-1}, s_{t-1} + Q_max]
        # generated by a reachable policy: by induction, the equality pins
        # s_mid[t] (hence s_out[t] = ŝ_t − d_t) to the value the policy
        # planned from, so every later target stays reachable.
        @constraint(m, [t=1:T], s_mid[t] == s_target[t])

        # --- Objective: pure operational cost ---
        if integer
            @objective(m, Min,
                sum(K * z[t] + c * q[t] + h * inv_hold[t] + p * back[t] for t in 1:T))
        else
            @objective(m, Min,
                sum(c * q[t] + h * inv_hold[t] + p * back[t] for t in 1:T))
        end
    else
        # --- Target-deficit penalty via NormOneCone ---
        # norm_deficit_arr[t] ≥ |s_mid[t] - s_target[t]| (L1 norm).
        @variable(m, norm_deficit_arr[1:T] >= 0.0)
        @variable(m, deficit_arr[1:T])
        @constraint(m, [t=1:T], deficit_arr[t] == s_mid[t] - s_target[t])
        @constraint(m, [t=1:T], [norm_deficit_arr[t]; deficit_arr[t:t]] in MOI.NormOneCone(2))

        # --- Objective: operational cost + target penalty ---
        if integer
            @objective(m, Min,
                sum(K * z[t] + c * q[t] + h * inv_hold[t] + p * back[t] for t in 1:T) +
                penalty * sum(norm_deficit_arr))
        else
            @objective(m, Min,
                sum(c * q[t] + h * inv_hold[t] + p * back[t] for t in 1:T) +
                penalty * sum(norm_deficit_arr))
        end
    end

    # --- Build parameter mappings for DecisionRules interface ---
    state_params_in = Vector{Vector{Any}}(undef, T)
    state_params_out = Vector{Vector{Tuple{Any,VariableRef}}}(undef, T)
    uncertainty_params = Vector{VariableRef}(undef, T)

    # Stage 1 reads from the initial-state parameters.
    state_params_in[1] = Any[s_init, last_demand_init, prev_demand_init]

    # Stages 2..T read from the previous stage's realized output variables.
    for t in 2:T
        state_params_in[t] = Any[s_out[t-1], last_demand_out[t-1], prev_demand_out[t-1]]
    end

    # Each stage maps (target_parameter → realized_variable) for gradient reading.
    for t in 1:T
        state_params_out[t] = [
            (s_target[t], s_out[t]),
            (last_demand_target[t], last_demand_out[t]),
            (prev_demand_target[t], prev_demand_out[t]),
        ]
        uncertainty_params[t] = demand[t]
    end

    # Return the five-tuple: (model, state_in, state_out, sampler, x0).
    return m, state_params_in, state_params_out,
           InventoryProcessSampler(uncertainty_params),
           [I_0, INVENTORY_LAST_DEMAND0, INVENTORY_PREV_DEMAND0]
end

# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

"""
    base_stock_policy(S_star) -> Function

Return a constant-target base-stock policy.

The target is the mid-period inventory level ``s^{mid} = S^*``; pass-through
state entries are current and lagged demand.

# Arguments
- `S_star::Float64`: order-up-to level.

# Examples
```julia
policy = base_stock_policy(160.0)
target = policy(Float32[d_t, inventory, d_{t-1}, d_{t-2}])
```
"""
function base_stock_policy(S_star::Float64)
    # Return a closure: target is always S*, pass through d_t and d_{t-1}.
    return x -> Float32[S_star, x[1], x[2]]
end

"""
    ExAnteInventoryPolicy{N}

Feedforward ex-ante inventory policy.

Input: `[d_t, inventory, d_{t-1}, d_{t-2}]`. The policy ignores the current
demand `d_t` to respect the ex-ante information pattern. Features passed to
the network are `[inventory/100, d_{t-1}/100, d_{t-2}/100]`.

Output: `[500 σ(net(features)), d_t, d_{t-1}]` — a target for mid-period
inventory, plus pass-through state entries.

# Fields
- `net::N`: Flux `Chain` mapping ℝ³ → ℝ¹.
"""
struct ExAnteInventoryPolicy{N}
    net::N
end

Functors.@functor ExAnteInventoryPolicy (net,)

Flux.reset!(::ExAnteInventoryPolicy) = nothing

function (policy::ExAnteInventoryPolicy)(x)
    # Unpack the input vector: [d_t, inventory, d_{t-1}, d_{t-2}].
    current_demand = Float32(x[1])
    inventory = Float32(x[2])
    last_demand = Float32(x[3])
    prev_demand = Float32(x[4])

    # Scale features to ≈ [0, 1] range for stable neural network training.
    order_features = Float32[inventory / 100, last_demand / 100, prev_demand / 100]

    # Map through the network and squash to [0, 500] via sigmoid.
    target = 500f0 .* Flux.sigmoid.(policy.net(order_features))

    # Return [target_s_mid, d_t, d_{t-1}] — target inventory + pass-through state.
    return Float32[target[1], current_demand, last_demand]
end

"""
    build_exante_policy(; seed = 2024) -> ExAnteInventoryPolicy

Construct the default feedforward ex-ante policy.

Architecture: Dense(3 → 32, relu) → Dense(32 → 24, relu) → Dense(24 → 1).

# Keyword Arguments
- `seed::Int`: random seed for weight initialization.

# Examples
```julia
policy = build_exante_policy(; seed = 2024)
```
"""
function build_exante_policy(; seed=2024)
    # Fix the random seed for reproducible weight initialization.
    Random.seed!(seed)

    # Three-layer feedforward: 3 inputs → 32 hidden → 24 hidden → 1 output.
    net = Chain(
        Dense(3, 32, relu),
        Dense(32, 24, relu),
        Dense(24, 1),
    )
    return ExAnteInventoryPolicy(net)
end

# ---------------------------------------------------------------------------
# LSTM ex-ante policy (temporal demand encoding, strictly ex-ante)
# ---------------------------------------------------------------------------

"""
    LSTMExAntePolicy{E,C,S}

Recurrent ex-ante inventory policy with temporal demand encoding.

An `LSTMCell` encoder processes the *lagged* demand ``d_{t-1}`` at each
stage, building a hidden representation of the demand history. The combiner
maps the LSTM output concatenated with `[inventory, d_{t-2}]` to a single
target value.

The policy is strictly ex-ante: it never sees the current-period demand
``d_t``. Temporal information comes from the LSTM state accumulated over
previous stages.

Output parameterization is affine: `raw * 200 + 150`, centered on typical
mid-period inventory and free from sigmoid saturation.

# Fields
- `encoder::E`: `Flux.LSTMCell` processing one demand value per stage.
- `combiner::C`: `Dense` layer mapping encoded + state features to target.
- `state::S`: current LSTM hidden state (reset between scenarios).

# Examples
```julia
policy = build_lstm_exante_policy(; seed = 2024, hidden = 16)
Flux.reset!(policy)
target = policy(Float32[d_t, inventory, d_{t-1}, d_{t-2}])
```
"""
mutable struct LSTMExAntePolicy{E,C,S}
    encoder::E
    combiner::C
    state::S
end

Functors.@functor LSTMExAntePolicy (encoder, combiner)

function (policy::LSTMExAntePolicy)(x)
    # Extract features from input: [d_t, inventory, d_{t-1}, d_{t-2}].
    # Only d_{t-1} (lagged) feeds the LSTM — d_t is NOT used (ex-ante).
    last_demand = Float32(x[3])
    inventory = Float32(x[2])
    prev_demand = Float32(x[4])

    # Match the element type of the LSTM state (Float32 during training).
    T = eltype(first(policy.state))

    # Feed the normalized lagged demand through the LSTM cell.
    # The cell returns the encoded output and the updated hidden state.
    encoded, new_state = policy.encoder(T[last_demand / 100], policy.state)

    # Update the hidden state for the next stage call within this scenario.
    policy.state = new_state

    # Concatenate LSTM output with current inventory and prev demand.
    combined = vcat(encoded, T[inventory / 100, prev_demand / 100])

    # Map combined features to a single scalar through the Dense combiner.
    raw = policy.combiner(combined)

    # Affine output: target = raw × 200 + 150 (centered, no saturation).
    target_s_mid = raw[1] * 200f0 + 150f0

    # Return [target_s_mid, d_t, d_{t-1}] — target inventory + pass-through.
    return Float32[target_s_mid, x[1], last_demand]
end

"""
    Flux.reset!(policy::LSTMExAntePolicy) -> Nothing

Reset the LSTM hidden state to its initial value.

Must be called before each scenario rollout so hidden state from previous
scenarios does not leak.
"""
function Flux.reset!(policy::LSTMExAntePolicy)
    # Restore the LSTM hidden state to its fresh initial values.
    policy.state = Flux.initialstates(policy.encoder)
    return nothing
end

"""
    build_lstm_exante_policy(; seed = 2024, hidden = 16) -> LSTMExAntePolicy

Construct an LSTM ex-ante policy.

Architecture: LSTMCell(1 → hidden) encoder, Dense(hidden + 2 → 1) combiner.

# Keyword Arguments
- `seed::Int`: random seed for weight initialization.
- `hidden::Int`: LSTM hidden dimension.

# Examples
```julia
policy = build_lstm_exante_policy(; seed = 2024, hidden = 16)
```
"""
function build_lstm_exante_policy(; seed=2024, hidden=16)
    # Fix the random seed for reproducible weight initialization.
    Random.seed!(seed)

    # LSTM cell: 1 input (normalized lagged demand) → hidden state.
    encoder = Flux.LSTMCell(1 => hidden)

    # Dense combiner: [LSTM output; inventory; prev_demand] → 1 target.
    combiner = Dense(hidden + 2, 1)

    # Initialize the LSTM hidden state to its default zeros.
    state = Flux.initialstates(encoder)

    return LSTMExAntePolicy(encoder, combiner, state)
end

# ---------------------------------------------------------------------------
# Reachable ex-ante policy (strict mode: always-feasible targets)
# ---------------------------------------------------------------------------

@doc raw"""
    InventoryReachablePolicy{E,C,S}

Recurrent ex-ante policy whose targets are always one-stage reachable,
enabling **strict mode** (hard target equalities, no penalty hyperparameter).

## Exact reachable set

The only decision behind the target is the order quantity
``q \in [0, Q_{max}]`` with ``s^{mid} = s + q``, where ``s`` is the realized
incoming inventory. ``s^{mid}`` appears in no other restrictive constraint
(``s^{out}`` is free and the hold/backlog split is feasible for any sign), so
the one-stage reachable set of the target is **exactly**

```math
\hat{s} \in [\, s,\; s + Q_{max} \,],
```

with no conservatism and no cross-component coupling. The binary setup
variable does not shrink this set (``z = 1`` admits every
``q \in [0, Q_{max}]``); it only reshapes the cost over it.

## Boundary attainment (why `hardsigmoid`, not `sigmoid`)

With a fixed ordering cost ``K > 0``, the endpoint ``q = 0`` ("do not
order") is an economically essential decision, not a measure-zero boundary.
A strict sigmoid keeps ``q = Q_{max}\,\sigma(z) > 0`` for every finite
``z``, which under strict equalities forces ``z_{setup} = 1`` and pays
``K`` every period — silently deleting the no-order action from the policy
class. The output therefore uses `hardsigmoid`, which attains 0 and 1
exactly on finite inputs:

```math
\hat{s} = s + Q_{max} \cdot \operatorname{hard\sigma}(z), \qquad
\operatorname{hard\sigma}(z) = \min(\max(z/6 + 1/2,\, 0),\, 1).
```

(Contrast with the hydro reachable policy, where the interval endpoints are
economically irrelevant and a strict sigmoid is harmless.)

## Pass-through components

The second and third state components (``d_{t-1}``, ``d_{t-2}`` histories)
are exogenous information states: their one-stage "reachable sets" are
singletons — the realized demand values. The policy passes them through
deterministically (no trainable parameters); the stage models leave their
target parameters unconstrained in both penalty and strict modes.

## Information pattern and gradients

Strictly **ex-ante**: the LSTM encoder consumes only the *lagged* demand
``d_{t-1}``; current demand ``d_t`` is never used for the order decision
(it appears in the output only as state pass-through plumbing). The bound
map ``\hat{s} = s + Q_{max} y`` is affine and exact, so — unlike the hydro
policy's physics bounds — it is left differentiable: the additive ``s``
term carries an exact gradient path through the realized state.

# Fields
- `encoder::E`: `Flux.LSTMCell` over the normalized lagged demand.
- `combiner::C`: `Dense` head mapping `[encoded; s/100; d_{t-2}/100]` to the
  pre-activation order fraction.
- `state::S`: LSTM hidden state, threaded across stages; reset per scenario.
- `Q_max::Float32`: order capacity (the constant reachable-interval width).

# Examples
```julia
policy = build_reachable_inventory_policy(; seed = 2024)
Flux.reset!(policy)
target = policy(Float32[d_t, inventory, d_lag1, d_lag2])
@assert inventory <= target[1] <= inventory + INVENTORY_Q_MAX
```
"""
mutable struct InventoryReachablePolicy{E,C,S}
    encoder::E
    combiner::C
    state::S
    Q_max::Float32
end

Functors.@functor InventoryReachablePolicy (encoder, combiner)

function (policy::InventoryReachablePolicy)(x)
    # Extract features from input: [d_t, inventory, d_{t-1}, d_{t-2}].
    # Only d_{t-1} (lagged) feeds the LSTM — d_t is NOT used (ex-ante).
    inventory = Float32(x[2])
    last_demand = Float32(x[3])
    prev_demand = Float32(x[4])

    # Match the element type of the LSTM state (Float32 during training).
    T = eltype(first(policy.state))

    # Feed the normalized lagged demand through the LSTM cell and thread
    # the hidden state to the next stage call within this scenario.
    encoded, new_state = policy.encoder(T[last_demand / 100], policy.state)
    policy.state = new_state

    # Concatenate LSTM output with current inventory and prev demand.
    combined = vcat(encoded, T[inventory / 100, prev_demand / 100])

    # Map to the order fraction y ∈ [0, 1]; hardsigmoid attains both
    # endpoints exactly (q = 0 and q = Q_max are real decisions).
    y = Flux.hardsigmoid(policy.combiner(combined)[1])

    # Affine map onto the exact reachable interval [s, s + Q_max], computed
    # in Float64 FROM THE RAW INPUT STATE: the implied order quantity is then
    # q = target − s = Q_max·y ≥ 0 bitwise. Using the Float32-cast state here
    # would make the strict equality infeasible whenever y saturates at 0 and
    # Float32(s) < s (the solver would need q ≈ −1e−5·s < 0) — a failure mode
    # the integer variant hits constantly because the fixed cost K actively
    # pushes the policy to the q = 0 boundary.
    target_s_mid = Float64(x[2]) + Float64(policy.Q_max) * Float64(y)

    # Return [target_s_mid, d_t, d_{t-1}] — target + state pass-throughs.
    return [target_s_mid, Float64(x[1]), Float64(last_demand)]
end

"""
    Flux.reset!(policy::InventoryReachablePolicy) -> Nothing

Reset the LSTM hidden state to its initial value. Must be called at every
scenario boundary so demand-history memory does not leak across rollouts.
"""
function Flux.reset!(policy::InventoryReachablePolicy)
    # Restore the LSTM hidden state to its fresh initial values.
    policy.state = Flux.initialstates(policy.encoder)
    return nothing
end

"""
    build_reachable_inventory_policy(; seed = 2024, hidden = 16, Q_max = INVENTORY_Q_MAX)
        -> InventoryReachablePolicy

Construct the reachable ex-ante policy used by the strict variants.

Architecture matches [`build_lstm_exante_policy`](@ref) exactly —
LSTMCell(1 → hidden) encoder, Dense(hidden + 2 → 1) head — so
penalty-vs-strict comparisons isolate the output parameterization, not
model capacity.

# Keyword Arguments
- `seed::Int`: random seed for weight initialization.
- `hidden::Int`: LSTM hidden dimension.
- `Q_max`: order capacity (reachable-interval width).

# Examples
```julia
policy = build_reachable_inventory_policy(; seed = 2024, hidden = 16)
```
"""
function build_reachable_inventory_policy(; seed=2024, hidden=16, Q_max=INVENTORY_Q_MAX)
    # Fix the random seed for reproducible weight initialization.
    Random.seed!(seed)

    # LSTM cell: 1 input (normalized lagged demand) → hidden state.
    encoder = Flux.LSTMCell(1 => hidden)

    # Dense head: [LSTM output; inventory; prev_demand] → order fraction.
    combiner = Dense(hidden + 2, 1)

    # Initialize the LSTM hidden state to its default zeros.
    state = Flux.initialstates(encoder)

    return InventoryReachablePolicy(encoder, combiner, state, Float32(Q_max))
end
