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
    stretchedsigmoid(x) -> y ∈ [0, 1 - 1e-3]

Mature strict-target activation. It reaches the lower endpoint at finite
pre-activation while retaining a safe interior margin at the upper endpoint,
which avoids forcing minimum turbine release and zero spill simultaneously in
an interior-point solve.
"""
function stretchedsigmoid(x::Real)
    T = float(typeof(x))
    return clamp((sigmoid(x) - T(0.03)) / T(0.94), zero(T), one(T) - T(1e-3))
end

function hardsigmoidsafe(x::Real)
    T = float(typeof(x))
    return min(Flux.hardsigmoid(x), one(T) - T(1e-3))
end

# ── Cascade link: upstream→downstream water-balance coupling ──────────────────

struct CascadeLink
    downstream::Int      # array position of downstream unit
    upstream::Int        # array position of upstream unit
    turn_only::Bool      # true if only turbine outflow (not spill) reaches downstream
    K_max_turn::Float32  # K × max_turn of the upstream unit
end

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

The bounds are DIFFERENTIABLE: gradient flows both through the sigmoid path
`σ(z_r)` and through the endpoints' affine dependence on `x_prev`, which is what
carries the adjoint recursion across stages.

# Cascade-aware target clamping

For downstream units receiving water from upstream cascade connections, the initial
upper bound uses `upstream_max_r = Σ K × max_turn_u`, which can overestimate the
actual upstream contribution when the upstream unit stores water (target increases).

After computing initial targets for all units, a **cascade clamping** step adjusts
downstream targets using the actual upstream release implied by the upstream target:

```math
R_u = K \\cdot w_u + x_u - \\hat{x}_u
```

- **Turn + spill connection**: max upstream contribution = ``\\max(0, R_u)``
- **Turn-only connection**: max contribution = ``\\min(K \\cdot max\\_turn_u, \\max(0, R_u))``

The downstream target is clamped: ``\\hat{x}_r ← \\min(\\hat{x}_r, true\\_upper_r)``.
This clamping is DIFFERENTIABLE — when the clamp binds, the downstream target
inherits the upstream target's influence through ``R_u``.

# Strict-mode guarantee

If `x₀` is a feasible initial reservoir state and each policy call returns
``\\hat{x}_t ∈ R(x_{t-1}, w_t)`` (including cascade-consistent bounds), then the
strict equality ``x_t = \\hat{x}_t`` is feasible for every stage solved in sequence.
The proof is by induction: stage 1 is feasible because ``\\hat{x}_1`` is reachable from
``x_0``; if stage ``t`` is feasible and realizes ``x_t = \\hat{x}_t``, then the
policy computes ``\\hat{x}_{t+1}`` from a feasible previous state with cascade-clamped
bounds, so stage ``t+1`` is feasible.

# Fields
- `encoder::E`:          Recurrent cell or Chain of cells (processes inflow only)
- `combiner::C`:         Feed-forward head combining encoder output with previous state
- `state::S`:            Recurrent state (threaded across stages)
- `n_context::Int`:      Number of context dimensions prepended before inflow
- `n_uncertainty::Int`:  Number of inflow dimensions (= nHyd)
- `n_state::Int`:        Number of state dimensions (= nHyd)
- `min_vol::V`:          Per-unit minimum reservoir volume
- `max_vol::V`:          Per-unit maximum reservoir volume
- `min_turn::V`:         Per-unit minimum turbine outflow
- `max_turn::V`:         Per-unit maximum turbine outflow
- `upstream_max::V`:     Pre-computed maximum upstream inflow contribution per unit
- `spill_max::SM`:       Per-unit max spillage (`nothing` = unlimited)
- `K::Float64`:          Water-balance conversion factor from flow to volume

See also: [`hydro_reachable_policy`](@ref), [`StateConditionedPolicy`](@ref)
"""
mutable struct HydroReachablePolicy{E,C,S,V,SM}
    encoder::E           # Recurrent encoder (LSTM/GRU chain) processing inflow
    combiner::C          # Feed-forward [encoder_out; state] => normalized target
    state::S             # Carried recurrent state, threaded across stages
    n_context::Int       # Number of context dimensions prepended before inflow
    n_uncertainty::Int   # Number of uncertainty (inflow) dimensions
    n_state::Int         # Number of state (reservoir) dimensions
    min_vol::V           # Per-unit minimum reservoir volume [nHyd]
    max_vol::V           # Per-unit maximum reservoir volume [nHyd]
    min_turn::V          # Per-unit minimum turbine outflow [nHyd]
    max_turn::V          # Per-unit maximum turbine outflow [nHyd]
    upstream_max::V      # Pre-computed K × Σ(upstream max_turn) per unit [nHyd]
    spill_max::SM        # Per-unit max spill, or nothing for unlimited
    K::Float64           # Water-balance conversion factor from flow to volume
    cascade::Vector{CascadeLink}  # Upstream→downstream connections for target clamping
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

# Differentiability

This function is DIFFERENTIABLE in `x_prev`, and that term is load-bearing. The
emitted target is ``\\hat{x}_t = l_t + (u_t - l_t) \\odot y_t`` with both endpoints
affine in the previous state, so

```math
\\frac{\\partial \\hat{x}_t}{\\partial x_{t-1}} = \\operatorname{diag}(y_t)
```

wherever the upper bound is off its `max_vol` ceiling (plus
``\\operatorname{diag}(1 - y_t)`` on coordinates whose lower bound is the
spill-limited `lower_raw`). Since the TS-DDR actor loss ``\\langle \\lambda,
\\hat{x}(\\theta) \\rangle`` feeds ``\\hat{x}_{t-1}`` back as the next stage's state
input, suppressing this term truncates the adjoint recursion at EVERY stage, not
only where a constraint binds, and the error compounds with the horizon.

A `ChainRulesCore.@non_differentiable` declaration used to sit here. Measured on
the shared production operating point (Bolivia, ``T = 126``, `min_turn` ≡ 0) it
dropped the term on 34.6% of (stage, reservoir) pairs and left the applied update
at cosine 0.673 / norm ratio 0.059 against the true gradient — ~48° off-direction
at 6% magnitude — while the differentiable form matches finite differences to six
digits. It is therefore removed here and in DecisionRulesExa.jl's copy.

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

"""
    _cascade_upper_bounds(policy, target, inflow, x_prev)

Compute the true reachable upper bound for downstream units given the actual
upstream targets. Returns a vector of upper bounds (Inf for units with no
upstream connections).

# Documented assumptions
- **Single-level cascades**: the implied upstream release
  ``R_u = K w_u + x_u - \\hat{x}_u`` omits the upstream unit's own incoming
  cascade contribution, which is conservative (underestimates the release)
  for multi-level chains.
- **Gradient through binding clamps**: this function is DIFFERENTIABLE, so a
  binding clamp propagates ``\\partial / \\partial \\hat{x}_u = -1`` from the
  implied release into the downstream target (turbine-only links additionally
  pass through `min`, whose pullback selects the active branch). Units with no
  incoming link take the constant `Inf` branch and carry no gradient, which is
  exact because `min(raw_target, Inf) == raw_target`.
"""
function _cascade_upper_bounds(policy::HydroReachablePolicy, target, inflow, x_prev)
    cascade = policy.cascade
    T = eltype(target)
    K = T(policy.K)
    n = length(target)
    isempty(cascade) && return fill(T(Inf), n)

    # Constant link metadata, gathered once. `_cascade_link_meta` is
    # `@non_differentiable` so this gather never enters the pullback.
    up, dn, turn_only, k_max_turn = _cascade_link_meta(policy, T)
    min_turn = T.(policy.min_turn)
    max_vol = T.(policy.max_vol)

    # Release implied by asking each upstream reservoir to end at its target.
    release = K .* inflow[up] .+ x_prev[up] .- target[up]
    positive_release = max.(zero(T), release)
    # Turbine-only links cannot pass more than K·max_turn downstream.
    max_contrib = ifelse.(turn_only, min.(k_max_turn, positive_release), positive_release)

    # One upper bound per LINK, expressed for its downstream reservoir.
    link_upper = min.(
        max_vol[dn],
        x_prev[dn] .+ K .* inflow[dn] .- K .* min_turn[dn] .+ max_contrib,
    )

    # Reduce link-wise bounds to one bound per reservoir WITHOUT mutation (the
    # previous `upper[d] = min(...)` loop is unreachable for reverse-mode AD):
    # build a links × reservoirs matrix that holds `link_upper` in the column of
    # the link's downstream reservoir and `Inf` elsewhere, then take a column
    # minimum. Reservoirs with no incoming link get an all-`Inf` column and are
    # left unchanged by the caller's `min.(raw_target, cascade_upper)`.
    link_by_reservoir = ifelse.(
        reshape(dn, :, 1) .== reshape(1:n, 1, :),
        reshape(link_upper, :, 1),
        T(Inf),
    )
    return vec(minimum(link_by_reservoir; dims = 1))
end

"""
    _cascade_link_meta(policy, T) -> (upstream, downstream, turn_only, k_max_turn)

Flatten `policy.cascade` into per-link index and constant vectors.

# Arguments
- `policy::HydroReachablePolicy`: policy carrying the cascade link list.
- `T::Type`: element type to which the float constants are converted.

# Returns
- `(upstream, downstream, turn_only, k_max_turn)`: four length-`nlinks`
  vectors — upstream/downstream reservoir positions, a turbine-only flag, and
  the turbine-only contribution cap ``K \\cdot max\\_turn_u``.

# Notes
Frozen topology metadata, constant in every differentiated quantity, so it is
declared `@non_differentiable` and never appears in the pullback of
[`_cascade_upper_bounds`](@ref).
"""
function _cascade_link_meta(policy::HydroReachablePolicy, ::Type{T}) where {T}
    return (
        Int[c.upstream for c in policy.cascade],
        Int[c.downstream for c in policy.cascade],
        Bool[c.turn_only for c in policy.cascade],
        T[c.K_max_turn for c in policy.cascade],
    )
end
ChainRulesCore.@non_differentiable _cascade_link_meta(::Any, ::Any)

"""
    (m::HydroReachablePolicy)(x)

Forward pass: given input `x = [context; inflow₁..nHyd; x_prev₁..nHyd]`, produce
one-stage reachable reservoir targets.

1. Split input into context, inflow, and previous state
2. Encode `[context; inflow]` through recurrent encoder, carrying state across stages
3. Combine encoder output with previous state via a sigmoid head → y_norm ∈ [0,1]
4. Compute reachable bounds [lower, upper] from physics (differentiable in `x_prev`)
5. Scale: target = lower + (upper - lower) × y_norm
6. Clamp downstream targets to cascade-aware upper bounds (differentiable)

# Documented assumptions
- **Single-level cascades**: the cascade clamp uses the implied upstream release
  ``R_u = K w_u + x_u - \\hat{x}_u``, which omits the upstream unit's own
  incoming cascade contribution — conservative for multi-level chains.
- **Gradient through binding clamps**: reachable bounds and cascade clamps are
  DIFFERENTIABLE; the reachable interval's dependence on `x_prev` and a binding
  clamp's dependence on the upstream target both carry gradient.
- **Physically-infeasible edge case**: if the cascade upper bound falls below
  the reachable lower bound, the clamped target may fall below `lower`. No
  policy-level remedy exists in that case — the underlying problem is
  infeasible.

# Arguments
- `x`: concatenated input vector `[context..., inflow..., previous_state...]`

# Returns
- `Vector`: target reservoir volumes, guaranteed within one-stage reachable set
"""
function (m::HydroReachablePolicy)(x)
    # Split input: optional context first, then true inflow, then previous state.
    # Physics bounds must use only the true inflow slice.
    c_end = m.n_context
    w_start = c_end + 1
    w_end = c_end + m.n_uncertainty
    context = c_end == 0 ? x[1:0] : x[1:c_end]
    inflow = x[w_start:w_end]
    x_prev = x[w_end+1:end]
    encoder_input = c_end == 0 ? inflow : vcat(context, inflow)

    # Encode inflow through the recurrent encoder, carrying state across calls.
    # Cast to encoder precision for type stability (avoids Zygote codegen bugs).
    T = DecisionRules._state_eltype(m.state)
    encoded, new_state = DecisionRules._step_encoder(m.encoder, T.(encoder_input), m.state)
    # Thread recurrent state to the next call
    m.state = new_state

    # Raw output from combiner (sigmoid activation → values in [0, 1])
    y_norm = m.combiner(vcat(encoded, x_prev))

    # Reachable interval from the current state and inflow. Both endpoints are
    # affine in `x_prev`, and that dependence CARRIES GRADIENT — it is the term
    # that propagates the adjoint from stage t back to stage t-1.
    lower, upper = _hydro_reachable_bounds(m, inflow, x_prev)

    # Scale normalized output to the reachable interval [lower, upper]
    raw_target = lower .+ (upper .- lower) .* y_norm

    # Clamp downstream targets to cascade-aware reachable bounds
    if !isempty(m.cascade)
        cascade_upper = _cascade_upper_bounds(m, raw_target, inflow, x_prev)
        return min.(raw_target, cascade_upper)
    end
    return raw_target
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
    activation=stretchedsigmoid,
    spill_max=nothing,
    combiner_layers=Int[],
    n_context::Int=0,
)
    nHyd = hydro_meta.nHyd
    # Validate layer sizes
    isempty(layers) && throw(ArgumentError("layers must be non-empty"))
    n_context >= 0 || throw(ArgumentError("n_context must be nonnegative"))

    # Build encoder: stack of recurrent cells processing [context; inflow].
    encoder_input_dim = nHyd + n_context
    if length(layers) == 1
        # Single-layer encoder
        encoder = DecisionRules._as_cell(encoder_type(encoder_input_dim => layers[1]))
    else
        # Multi-layer encoder: chain of recurrent cells
        encoder_layers = [DecisionRules._as_cell(encoder_type(encoder_input_dim => layers[1]))]
        for i in 1:(length(layers) - 1)
            push!(
                encoder_layers,
                DecisionRules._as_cell(encoder_type(layers[i] => layers[i + 1])),
            )
        end
        encoder = Chain(encoder_layers...)
    end

    activation in (sigmoid, stretchedsigmoid, hardsigmoidsafe) ||
        throw(ArgumentError("activation must map into [0, 1]"))
    # The bounded output is scaled to [lower, upper] in the forward pass.
    combiner = DecisionRules.dense_policy_head(
        layers[end] + nHyd,
        nHyd,
        collect(Int, combiner_layers);
        activation=activation,
    )

    # Pre-compute maximum upstream inflow contribution per unit:
    # upstream_max[r] = Σ_{u ∈ upstream(r)} K × max_turn_u
    K = hydro_meta.K
    upstream_max = zeros(Float32, nHyd)
    for (r, upstream_list) in enumerate(hydro_meta.upstream_turn)
        for (u_pos, u_max_turn) in upstream_list
            upstream_max[r] += Float32(K * u_max_turn)
        end
    end

    # Build cascade connections for target clamping
    spill_dests = Dict{Int,Set{Int}}()
    for (r, upstream_list) in enumerate(hydro_meta.upstream_spill)
        for (u_pos, _) in upstream_list
            push!(get!(spill_dests, u_pos, Set{Int}()), r)
        end
    end
    cascade = CascadeLink[]
    for (r, upstream_list) in enumerate(hydro_meta.upstream_turn)
        for (u_pos, u_max_turn) in upstream_list
            has_spill = haskey(spill_dests, u_pos) && r in spill_dests[u_pos]
            push!(cascade, CascadeLink(r, u_pos, !has_spill, Float32(K * u_max_turn)))
        end
    end
    for (r, upstream_list) in enumerate(hydro_meta.upstream_spill)
        for (u_pos, _) in upstream_list
            already = any(c -> c.downstream == r && c.upstream == u_pos, cascade)
            already || push!(cascade, CascadeLink(r, u_pos, false, Float32(K * hydro_meta.max_turn[u_pos])))
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
        n_context,                         # context dimensions prepended before inflow
        nHyd,                              # n_uncertainty = nHyd (one inflow per unit)
        nHyd,                              # n_state = nHyd (one reservoir per unit)
        Float32.(hydro_meta.min_vol),      # per-unit min volume
        Float32.(hydro_meta.max_vol),      # per-unit max volume
        Float32.(hydro_meta.min_turn),     # per-unit min turbine outflow
        Float32.(hydro_meta.max_turn),     # per-unit max turbine outflow
        upstream_max,                      # pre-computed upstream contribution
        spill_max === nothing ? nothing : Float32.(collect(spill_max)),  # spill bounds
        K,                                 # water-balance conversion factor
        cascade,                           # cascade connections for target clamping
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
# Strip a `Recurrence`/`Flux.LSTM` wrapper from a saved encoder-state tree so it
# loads into DecisionRules.jl's BARE `LSTMCell` encoder (built via `_as_cell`,
# which rolls out with `_step_encoder`). Checkpoints trained by the ExaModels
# strict trainer save each encoder layer as `Flux.LSTM` -> layer state
# `(cell = (Wi, Wh, bias),)`; DR.jl's encoder layers are the inner cells, state
# `(Wi, Wh, bias)`. This is the state-tree mirror of `_as_cell`: unwrap the
# `cell` field of every layer so `Flux.loadmodel!` sees matching structures. The
# weights are identical; only the wrapper level differs.
_unwrap_encoder_cells(enc_state) =
    hasproperty(enc_state, :layers) ?
        (; layers = map(L -> (L isa NamedTuple && hasproperty(L, :cell)) ? L.cell : L,
                        enc_state.layers)) :
        enc_state

function load_policy_weights!(policy::HydroReachablePolicy, state)
    # Load only the encoder and combiner weights, keeping hydro bounds unchanged.
    # First try the structures as-saved; if the encoder is `Flux.LSTM`-wrapped
    # (ExaModels-trained checkpoint) fall back to unwrapping the cell wrapper.
    try
        Flux.loadmodel!(policy.encoder, state.encoder)
        Flux.loadmodel!(policy.combiner, state.combiner)
    catch err
        try
            Flux.loadmodel!(policy.encoder, _unwrap_encoder_cells(state.encoder))
            Flux.loadmodel!(policy.combiner, state.combiner)
        catch _
            throw(ArgumentError(
                "Could not load HydroReachablePolicy weights. The checkpoint architecture " *
                "must match encoder/head widths and n_context=$(policy.n_context); " *
                "contextual policies require newly trained checkpoints. Original error: $err",
            ))
        end
    end
    return policy
end

function load_policy_weights!(policy::DecisionRules.ContextualPolicy, state)
    inner_state = hasproperty(state, :policy) ? getproperty(state, :policy) : state
    load_policy_weights!(policy.policy, inner_state)
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
    n_context::Int=0,
)
    # Build fresh policy with correct bounds from hydro_meta
    policy = hydro_reachable_policy(hydro_meta, layers; encoder_type=encoder_type,
                                    spill_max=spill_max,
                                    combiner_layers=combiner_layers,
                                    n_context=n_context)
    # Load saved weights into the fresh policy
    model_state = JLD2.load(checkpoint_path, "model_state")
    load_policy_weights!(policy, model_state)
    return policy
end
