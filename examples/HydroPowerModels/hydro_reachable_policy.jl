# Hydro Reachable Policy — feasibility-guaranteeing target policy for strict subproblems
#
# This file defines HydroReachablePolicy, a policy architecture that guarantees
# one-stage reachability for hydro reservoir targets. It wraps an LSTM encoder +
# feed-forward combiner (same architecture family as StateConditionedPolicy) but
# bounds the output to the one-stage reachable set via sigmoid activation.
#
# Depends on: DecisionRules (for _step_encoder, _init_recurrent_state, _state_eltype)
# Must be included AFTER `using DecisionRules`.

using Functors
using ChainRulesCore

"""
    HydroReachablePolicy{E,C,S,V,SM}

A policy that guarantees one-stage reachability for hydro reservoir targets.

The policy architecture mirrors [`StateConditionedPolicy`](@ref): an LSTM
encoder processes only the uncertainty (inflow) sequence, then a feed-forward
combiner maps `[encoded_inflow; current_reservoir_state]` to normalized targets
in `[0, 1]` via sigmoid activation. These normalized targets are scaled to the
one-stage **reachable set** `[lower, upper]` for each reservoir:

```math
target_r = lower_r + (upper_r - lower_r) \\cdot \\sigma(z_r)
```

where `lower_r` and `upper_r` are the minimum and maximum reservoir volumes achievable
in one stage from the current state `x_r` given inflow `w_r`, turbine bounds
`[min\\_turn_r, max\\_turn_r]`, and upstream cascade inflows.

# Reachable bounds (per reservoir r)

The upper reachable bound assumes minimum outflow (no turbine, no spill) plus maximum
upstream inflow from cascade connections:

```math
upper_r = \\min(max\\_vol_r,\\;  x_r + K \\cdot w_r - K \\cdot min\\_turn_r + upstream\\_max_r)
```

The lower reachable bound assumes maximum outflow (full turbine + max spill):

```math
lower_r = \\max(min\\_vol_r,\\;  x_r + K \\cdot w_r - K \\cdot max\\_turn_r - spill\\_max_r)
```

When `spill_max === nothing` (unlimited spillage), `lower_r = min_vol_r` since the
reservoir can always be emptied to its physical minimum.

The bounds are marked `@non_differentiable` — gradients flow only through the sigmoid
path `σ(z_r)`, not through the bounds themselves. This matches the reference
implementation in DecisionRulesExa.jl.

# Strict-mode guarantee

If `x₀` is a feasible initial reservoir state and each policy call returns
``\hat{x}_t ∈ R(x_{t-1}, w_t)``, then the strict equality
``x_t = \hat{x}_t`` is feasible for every stage solved in sequence. The proof is
by induction: stage 1 is feasible because ``\hat{x}_1`` is reachable from
``x_0``; if stage ``t`` is feasible and realizes ``x_t = \hat{x}_t``, then the
policy computes ``\hat{x}_{t+1}`` from a feasible previous state, so stage
``t+1`` is feasible.

# Fields
- `encoder::E`:          Recurrent cell or Chain of cells (processes inflow only)
- `combiner::C`:         Feed-forward head combining encoder output with previous state
- `state::S`:            Recurrent state (threaded across stages)
- `n_uncertainty::Int`:  Number of inflow dimensions (= nHyd)
- `n_state::Int`:        Number of state dimensions (= nHyd)
- `min_vol::V`:          Per-unit minimum reservoir volume
- `max_vol::V`:          Per-unit maximum reservoir volume
- `min_turn::V`:         Per-unit minimum turbine outflow
- `max_turn::V`:         Per-unit maximum turbine outflow
- `upstream_max::V`:     Pre-computed maximum upstream inflow contribution per unit
- `spill_max::SM`:       Per-unit max spillage (`nothing` = unlimited)
- `K::Float64`:          Stage duration (hours), used in water balance

See also: [`hydro_reachable_policy`](@ref), [`StateConditionedPolicy`](@ref)
"""
mutable struct HydroReachablePolicy{E,C,S,V,SM}
    encoder::E           # Recurrent encoder (LSTM/GRU chain) processing inflow
    combiner::C          # Feed-forward [encoder_out; state] => normalized target
    state::S             # Carried recurrent state, threaded across stages
    n_uncertainty::Int   # Number of uncertainty (inflow) dimensions
    n_state::Int         # Number of state (reservoir) dimensions
    min_vol::V           # Per-unit minimum reservoir volume [nHyd]
    max_vol::V           # Per-unit maximum reservoir volume [nHyd]
    min_turn::V          # Per-unit minimum turbine outflow [nHyd]
    max_turn::V          # Per-unit maximum turbine outflow [nHyd]
    upstream_max::V      # Pre-computed K × Σ(upstream max_turn) per unit [nHyd]
    spill_max::SM        # Per-unit max spill, or nothing for unlimited
    K::Float64           # Stage duration in hours
end

# Only encoder and combiner are trainable. Bounds, state, dimensions are frozen.
Functors.@functor HydroReachablePolicy (encoder, combiner)

"""
    _hydro_reachable_bounds(policy::HydroReachablePolicy, inflow, x_prev)

Compute the one-stage reachable reservoir bounds given current state `x_prev`
and per-unit inflow `inflow`. Returns `(lower, upper)` vectors.

The upper bound is the maximum volume achievable in one stage: current volume
plus inflow minus minimum turbine outflow plus maximum upstream cascade inflow,
clamped to `max_vol`. The lower bound is the minimum volume achievable: current
volume plus inflow minus maximum turbine outflow minus maximum spill, clamped
to `min_vol`.

Marked `@non_differentiable` — gradients do not flow through the bounds. The gradient
path is solely through the sigmoid-scaled output `σ(z_r)`.

# Arguments
- `policy::HydroReachablePolicy`: policy containing hydro bounds and parameters
- `inflow`: per-unit inflow vector for this stage
- `x_prev`: current reservoir volumes (state from previous stage)

# Returns
- `(lower, upper)`: tuple of vectors, each of length `n_state`
"""
function _hydro_reachable_bounds(policy::HydroReachablePolicy, inflow, x_prev)
    # Cast all bound vectors to match the element type of x_prev for type stability
    T = eltype(x_prev)
    K        = T(policy.K)
    min_vol  = T.(policy.min_vol)
    max_vol  = T.(policy.max_vol)
    min_turn = T.(policy.min_turn)
    max_turn = T.(policy.max_turn)
    upstream = T.(policy.upstream_max)

    # Upper reachable: minimum outflow (min turbine, no spill) + max upstream
    upper_raw = x_prev .+ K .* inflow .- K .* min_turn .+ upstream
    # Clamp to physical maximum volume
    upper = min.(max_vol, upper_raw)

    # Lower reachable: maximum outflow (max turbine + max spill)
    lower = if policy.spill_max === nothing
        # Unlimited spill → can always dump down to min_vol
        min_vol
    else
        spill_max = T.(policy.spill_max)
        # Volume after maximum discharge and maximum spill
        lower_raw = x_prev .+ K .* inflow .- K .* max_turn .- spill_max
        # Clamp to physical minimum volume
        max.(min_vol, lower_raw)
    end

    # Ensure lower ≤ upper (numerical safety for edge cases like CHJ with max_vol=0)
    upper = max.(upper, lower)

    return lower, upper
end
# Gradients do not flow through the bounds — only through σ(z)
ChainRulesCore.@non_differentiable _hydro_reachable_bounds(::Any, ::Any, ::Any)

"""
    (m::HydroReachablePolicy)(x)

Forward pass: given input `x = [inflow₁..nHyd; x_prev₁..nHyd]`, produce
one-stage reachable reservoir targets.

1. Split input into inflow and previous state
2. Encode inflow through recurrent encoder (LSTM), carrying state across stages
3. Combine encoder output with previous state via a sigmoid head → y_norm ∈ [0,1]
4. Compute reachable bounds [lower, upper] from physics (no gradient)
5. Scale: target = lower + (upper - lower) × y_norm

# Arguments
- `x`: concatenated input vector `[inflow..., previous_state...]`

# Returns
- `Vector`: target reservoir volumes, guaranteed within one-stage reachable set
"""
function (m::HydroReachablePolicy)(x)
    # Split input: first n_uncertainty elements are inflow, rest is previous state
    inflow = x[1:m.n_uncertainty]
    x_prev = x[m.n_uncertainty+1:end]

    # Encode inflow through the recurrent encoder, carrying state across calls.
    # Cast to encoder precision for type stability (avoids Zygote codegen bugs).
    T = DecisionRules._state_eltype(m.state)
    encoded, new_state = DecisionRules._step_encoder(m.encoder, T.(inflow), m.state)
    # Thread recurrent state to the next call
    m.state = new_state

    # Raw output from combiner (sigmoid activation → values in [0, 1])
    y_norm = m.combiner(vcat(encoded, x_prev))

    # Compute reachable bounds from current state and inflow (no gradient)
    lower, upper = _hydro_reachable_bounds(m, inflow, x_prev)

    # Scale normalized output to the reachable interval [lower, upper]
    return lower .+ (upper .- lower) .* y_norm
end

"""
    Flux.reset!(m::HydroReachablePolicy)

Reset the encoder's recurrent state to `Flux.initialstates`, e.g. before starting
a new rollout. The hydro bounds (min_vol, max_vol, etc.) are unchanged.
"""
function Flux.reset!(m::HydroReachablePolicy)
    # Reinitialize recurrent state from the encoder's initial states
    m.state = DecisionRules._init_recurrent_state(m.encoder)
    return nothing
end

"""
    hydro_reachable_policy(hydro_meta, layers; encoder_type=Flux.LSTM,
                           spill_max=nothing, combiner_layers=Int[])

Create a [`HydroReachablePolicy`](@ref) from hydro metadata (as returned by
the 7th return value of `build_hydropowermodels`).

The architecture mirrors [`state_conditioned_policy`](@ref): an LSTM encoder
processes inflows, then a feed-forward combiner produces normalized targets in
`[0, 1]`. These are scaled to the one-stage reachable interval. The combiner
uses sigmoid activation — this is mandatory and cannot be overridden. Set
`combiner_layers` to add hidden layers to the nonrecurrent state-conditioned
target map. This is the preferred way to depart from linear decision rules
without adding recurrence over the reservoir-state input.

# Arguments
- `hydro_meta::NamedTuple`: hydro system metadata with fields `nHyd`, `min_vol`,
  `max_vol`, `min_turn`, `max_turn`, `K`, `upstream_turn`
- `layers::Vector{Int}`: hidden layer sizes for the LSTM encoder
  (e.g. `[128, 128]` for a 2-layer LSTM)
- `encoder_type`: recurrent layer/cell type (default: `Flux.LSTM`). Must support
  `Flux.initialstates` and the stateful `(x, state) -> (output, new_state)` call
- `spill_max`: per-unit maximum spillage vector (`nothing` = unlimited spillage,
  meaning `lower = min_vol` always)
- `combiner_layers::Vector{Int}`: hidden widths for the nonrecurrent target head

# Returns
- `HydroReachablePolicy`: ready-to-train policy with sigmoid-bounded outputs

# Examples
```julia
subproblems, _, _, _, _, _, hydro_meta = build_hydropowermodels(
    case_dir, formulation_file; strict=true, optimizer=diff_opt
)
policy = hydro_reachable_policy(hydro_meta, [128, 128])
policy_with_deep_state_head = hydro_reachable_policy(
    hydro_meta,
    [128, 128];
    combiner_layers=[256, 256],
)
```

See also: [`HydroReachablePolicy`](@ref), [`state_conditioned_policy`](@ref),
[`load_policy_weights!`](@ref)
"""
function hydro_reachable_policy(
    hydro_meta::NamedTuple,
    layers::Vector{Int};
    encoder_type=Flux.LSTM,
    spill_max=nothing,
    combiner_layers=Int[],
)
    nHyd = hydro_meta.nHyd
    # Validate layer sizes
    isempty(layers) && throw(ArgumentError("layers must be non-empty"))

    # Build encoder: stack of recurrent cells processing inflow (nHyd-dimensional)
    if length(layers) == 1
        # Single-layer encoder
        encoder = DecisionRules._as_cell(encoder_type(nHyd => layers[1]))
    else
        # Multi-layer encoder: chain of recurrent cells
        encoder_layers = [DecisionRules._as_cell(encoder_type(nHyd => layers[1]))]
        for i in 1:(length(layers) - 1)
            push!(
                encoder_layers,
                DecisionRules._as_cell(encoder_type(layers[i] => layers[i + 1])),
            )
        end
        encoder = Chain(encoder_layers...)
    end

    # Build combiner — sigmoid activation is MANDATORY for reachable bounds guarantee
    # The sigmoid output ∈ [0, 1] is scaled to [lower, upper] in the forward pass
    combiner = DecisionRules.dense_policy_head(
        layers[end] + nHyd,
        nHyd,
        collect(Int, combiner_layers);
        activation=sigmoid,
    )

    # Pre-compute maximum upstream inflow contribution per unit:
    # upstream_max[r] = Σ_{u ∈ upstream(r)} K × max_turn_u
    K = hydro_meta.K
    upstream_max = zeros(Float32, nHyd)
    for (r, upstream_list) in enumerate(hydro_meta.upstream_turn)
        for (u_pos, u_max_turn) in upstream_list
            # Each upstream unit's turbine outflow contributes at most K × max_turn
            upstream_max[r] += Float32(K * u_max_turn)
        end
    end

    # Validate spill_max dimensions if provided
    if spill_max !== nothing && length(spill_max) != nHyd
        throw(ArgumentError("spill_max length must be nHyd=$nHyd; got $(length(spill_max))"))
    end

    return HydroReachablePolicy(
        encoder,
        combiner,
        DecisionRules._init_recurrent_state(encoder),   # initial recurrent state
        nHyd,                              # n_uncertainty = nHyd (one inflow per unit)
        nHyd,                              # n_state = nHyd (one reservoir per unit)
        Float32.(hydro_meta.min_vol),      # per-unit min volume
        Float32.(hydro_meta.max_vol),      # per-unit max volume
        Float32.(hydro_meta.min_turn),     # per-unit min turbine outflow
        Float32.(hydro_meta.max_turn),     # per-unit max turbine outflow
        upstream_max,                      # pre-computed upstream contribution
        spill_max === nothing ? nothing : Float32.(collect(spill_max)),  # spill bounds
        K,                                 # stage duration in hours
    )
end

"""
    load_policy_weights!(policy::HydroReachablePolicy, state)

Load encoder/combiner weights from a saved model state (e.g., from a
[`StateConditionedPolicy`](@ref) checkpoint). Hydro bounds are preserved.

This enables warmstarting: train a `StateConditionedPolicy` with non-strict
subproblems, then load its encoder/combiner weights into a `HydroReachablePolicy`
for strict fine-tuning.

# Arguments
- `policy::HydroReachablePolicy`: target policy (bounds are preserved)
- `state`: saved model state (from `Flux.state(model)` or JLD2 checkpoint)

# Returns
- `policy`: the modified policy (mutated in place)

See also: [`hydro_reachable_policy`](@ref)
"""
function load_policy_weights!(policy::HydroReachablePolicy, state)
    # Load only the encoder and combiner weights, keeping hydro bounds unchanged
    Flux.loadmodel!(policy.encoder, state.encoder)
    Flux.loadmodel!(policy.combiner, state.combiner)
    return policy
end

"""
    load_hydro_reachable_policy(checkpoint_path, hydro_meta, layers;
                                encoder_type=Flux.LSTM, spill_max=nothing,
                                combiner_layers=Int[])

Load a [`HydroReachablePolicy`](@ref) from a JLD2 checkpoint, reconstructing
the hydro bounds from `hydro_meta` (since JLD2 may not preserve exact types).

# Arguments
- `checkpoint_path::String`: path to JLD2 file with `"model_state"` key
- `hydro_meta::NamedTuple`: hydro metadata from `build_hydropowermodels`
- `layers::Vector{Int}`: encoder hidden layer sizes (must match checkpoint)
- `encoder_type`: recurrent layer type (default: `Flux.LSTM`)
- `spill_max`: per-unit max spillage, or `nothing` for unlimited
- `combiner_layers::Vector{Int}`: hidden widths for the nonrecurrent target head

# Returns
- `HydroReachablePolicy`: policy with loaded weights and fresh hydro bounds

See also: [`hydro_reachable_policy`](@ref), [`load_policy_weights!`](@ref)
"""
function load_hydro_reachable_policy(
    checkpoint_path::String,
    hydro_meta::NamedTuple,
    layers::Vector{Int};
    encoder_type=Flux.LSTM,
    spill_max=nothing,
    combiner_layers=Int[],
)
    # Build fresh policy with correct bounds from hydro_meta
    policy = hydro_reachable_policy(hydro_meta, layers; encoder_type=encoder_type,
                                    spill_max=spill_max,
                                    combiner_layers=combiner_layers)
    # Load saved weights into the fresh policy
    model_state = JLD2.load(checkpoint_path, "model_state")
    load_policy_weights!(policy, model_state)
    return policy
end
