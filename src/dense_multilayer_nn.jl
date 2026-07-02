using Functors
using ChainRulesCore

"""
    dense_multilayer_nn(num_inputs, num_outputs, layers; activation=Flux.relu, dense=Dense)

Create a multi-layer neural network with the specified architecture.

Given hidden layer widths ``(h_1, \\dots, h_L)`` and activation ``\\sigma``,
the resulting `Chain` computes

```math
f(x) = W_{L+1} \\, \\sigma\\bigl(W_L \\cdots \\sigma(W_1 x + b_1) \\cdots + b_L\\bigr) + b_{L+1},
```

where ``W_k \\in \\mathbb{R}^{h_k \\times h_{k-1}}``, ``h_0 = \\text{num\\_inputs}``,
and the final layer ``W_{L+1} \\in \\mathbb{R}^{\\text{num\\_outputs} \\times h_L}``
has no activation. When `layers` is empty a single linear layer is returned.

If `dense` is a recurrent type (`LSTM`, `GRU`, `RNN`), pair notation
`in => out` is used instead of `(in, out, σ)`, and the last layer omits the
activation so it can act as a linear projection.

# Arguments
- `num_inputs::Int`: number of input features ``h_0``.
- `num_outputs::Int`: number of output features.
- `layers::Vector{Int}`: hidden layer widths ``(h_1, \\dots, h_L)``.
- `activation`: element-wise activation ``\\sigma`` (default: `Flux.relu`).
- `dense`: layer constructor (`Dense`, `LSTM`, `GRU`, `RNN`).

# Returns
- `Chain` (or a single layer when `layers` is empty).

# Examples
```julia
mlp = dense_multilayer_nn(10, 3, [64, 32])       # 10 → 64 → 32 → 3
rnn = dense_multilayer_nn(5, 2, [16]; dense=LSTM) # 5 → 16 → 2 (LSTM)
```
"""
function dense_multilayer_nn(
    num_inputs::Int,
    num_outputs::Int,
    layers::Vector{Int};
    activation=Flux.relu,
    dense=Dense,
)
    # Recurrent layers use pair notation (in => out) and ignore activation.
    is_recurrent = dense in (LSTM, GRU, RNN)

    # Helper that builds one hidden layer with the appropriate constructor form.
    _make_layer(in_dim, out_dim) =
        is_recurrent ? dense(in_dim => out_dim) : dense(in_dim, out_dim, activation)

    # No hidden layers: return a single linear projection.
    if length(layers) == 0
        return if is_recurrent
            dense(num_inputs => num_outputs)
        else
            dense(num_inputs, num_outputs, activation)
        end
    end

    # Interior hidden layers: h_2 → h_3 → … → h_{L-1}.
    midlayers = [_make_layer(layers[i], layers[i + 1]) for i in 1:(length(layers) - 1)]

    # First hidden layer maps from the input dimension.
    first_layer = _make_layer(num_inputs, layers[1])

    # Final layer omits activation so the output is an unrestricted linear map.
    last_layer =
        is_recurrent ? dense(layers[end] => num_outputs) : dense(layers[end], num_outputs)

    return Chain(first_layer, midlayers..., last_layer)
end

"""
    policy_input_dim(num_uncertainties, num_states)

Compute the input dimension for a policy network.

Policy networks receive ``[w_t;\\; x_{t-1}]`` as input, so the required
dimension is

```math
d_{\\text{in}} = \\dim(w_t) + \\dim(x_{t-1}).
```

This format is consistent between subproblem and deterministic-equivalent
formulations, enabling warm-starting policies trained with det_eq for use
with subproblems.

# Arguments
- `num_uncertainties::Int`: dimensionality of the stage uncertainty ``w_t``.
- `num_states::Int`: dimensionality of the state ``x_{t-1}``.

# Returns
- `Int`: total input dimension ``d_{\\text{in}}``.

# Examples
```julia
d = policy_input_dim(5, 3)  # 8
```
"""
function policy_input_dim(num_uncertainties::Int, num_states::Int)
    # Input is the concatenation [w_t; x_{t-1}].
    return num_uncertainties + num_states
end

"""
    policy_input_dim(uncertainty_samples, initial_state)

Compute the input dimension for a policy network from problem data.

Infers ``\\dim(w_t)`` from the first uncertainty sample and ``\\dim(x)``
from the initial state vector, then delegates to
[`policy_input_dim(::Int, ::Int)`](@ref).

# Arguments
- `uncertainty_samples::Vector`: uncertainty samples; `length(uncertainty_samples[1])`
  gives ``\\dim(w_t)``.
- `initial_state::Vector`: initial state vector; `length(initial_state)` gives
  ``\\dim(x)``.

# Returns
- `Int`: total input dimension ``d_{\\text{in}}``.

# Examples
```julia
d = policy_input_dim(uncertainty_samples, x0)
```
"""
function policy_input_dim(uncertainty_samples::Vector, initial_state::Vector)
    # Infer dimensions from the first sample and the initial state.
    num_uncertainties = length(uncertainty_samples[1])
    num_states = length(initial_state)
    return policy_input_dim(num_uncertainties, num_states)
end

"""
    StateConditionedPolicy

A policy architecture that separates temporal encoding from state conditioning.

The encoder is a recurrent cell (`LSTMCell`/`GRUCell`/`RNNCell`, or a `Chain`
of cells) that processes only the uncertainty sequence to capture temporal
dependencies. The combiner is a `Dense` layer that merges the encoder output
with the previous state to predict the next state.

Given uncertainty ``w_t`` and previous state ``x_{t-1}``, the forward pass is

```math
h_t, s_t = \\text{LSTM}(w_t, s_{t-1}) \\\\
\\hat{x}_t = f_{\\text{combine}}([h_t;\\; x_{t-1}])
```

where ``s_t`` is the hidden recurrent state carried across stages and
``f_{\\text{combine}}`` is a `Dense` layer with activation ``\\sigma``.

Flux's recurrent cells are stateless (Flux >= 0.16): each call returns
`(output, new_state)` instead of mutating an internal `Recur`.
`StateConditionedPolicy` therefore carries the encoder's recurrent state
itself in `state`, threading it through one call per stage. Call
`Flux.reset!` to clear it (back to `Flux.initialstates`) at the start of
a rollout.

# Fields
- `encoder::E`: recurrent cell or `Chain` of cells encoding uncertainty.
- `combiner::C`: `Dense` layer mapping ``[h_t;\\; x_{t-1}]`` to ``\\hat{x}_t``.
- `state::S`: current recurrent state ``s_t``, carried across calls.
- `n_uncertainty::Int`: dimensionality of ``w_t``.
- `n_state::Int`: dimensionality of ``x_{t-1}``.

Input format: `[uncertainty..., previous_state...]`.
"""
mutable struct StateConditionedPolicy{E,C,S}
    encoder::E          # Recurrent cell, or Chain of cells, that processes uncertainty only
    combiner::C         # Dense that combines encoder output with previous state
    state::S            # Encoder recurrent state, carried across calls
    n_uncertainty::Int  # Number of uncertainty dimensions
    n_state::Int        # Number of state dimensions
end

# Use Functors.@functor with explicit trainable fields (excludes state, n_uncertainty, n_state)
# We use Functors.@functor instead of Flux.@layer to avoid MutableTangent issues with LSTM
Functors.@functor StateConditionedPolicy (encoder, combiner)

"""
    materialize_tangent(x::Number) -> Number

Return numeric tangents unchanged (leaf values are already plain scalars).
"""
materialize_tangent(x::Number) = x

"""
    materialize_tangent(x::AbstractArray) -> AbstractArray

Return array tangents unchanged (arrays are already `Flux.update!`-compatible).
"""
materialize_tangent(x::AbstractArray) = x

"""
    materialize_tangent(::Nothing) -> Nothing

Map `nothing` tangents (unused parameters) to `nothing`.
"""
materialize_tangent(::Nothing) = nothing

"""
    materialize_tangent(::ChainRulesCore.NoTangent) -> Nothing

Map `NoTangent` (structural zeros for non-differentiable fields) to `nothing`.
"""
materialize_tangent(::ChainRulesCore.NoTangent) = nothing

"""
    materialize_tangent(::ChainRulesCore.ZeroTangent) -> Nothing

Map `ZeroTangent` (additive identity in the tangent space) to `nothing`.
"""
materialize_tangent(::ChainRulesCore.ZeroTangent) = nothing

"""
    materialize_tangent(t::ChainRulesCore.MutableTangent) -> NamedTuple

Unwrap a `MutableTangent` (produced by Zygote for mutable structs such as
`Flux.Recur`) by extracting its backing `NamedTuple` and recursing.
"""
function materialize_tangent(t::ChainRulesCore.MutableTangent)
    # MutableTangent stores values in RefValues; extract them via backing.
    backing = ChainRulesCore.backing(t)
    return materialize_tangent(backing)
end

"""
    materialize_tangent(t::ChainRulesCore.Tangent) -> NamedTuple

Unwrap an immutable `Tangent` by extracting its backing and recursing.
"""
function materialize_tangent(t::ChainRulesCore.Tangent)
    # Tangent backing is already a NamedTuple; recurse to handle nested types.
    backing = ChainRulesCore.backing(t)
    return materialize_tangent(backing)
end

"""
    materialize_tangent(nt::NamedTuple{K}) -> NamedTuple{K}

Recurse through every field of a `NamedTuple`, materializing each value.
"""
function materialize_tangent(nt::NamedTuple{K}) where {K}
    # Apply materialize_tangent element-wise and reconstruct the same keys.
    vals = map(materialize_tangent, values(nt))
    return NamedTuple{K}(vals)
end

"""
    materialize_tangent(t::Tuple) -> Tuple

Recurse through every element of a `Tuple`, materializing each value.
"""
function materialize_tangent(t::Tuple)
    return map(materialize_tangent, t)
end

"""
    materialize_tangent(ref::Base.RefValue)

Dereference a `RefValue` wrapper (used inside `MutableTangent` fields)
and recurse on the contained value.
"""
function materialize_tangent(ref::Base.RefValue)
    # RefValue wraps a single value; extract it before recursing.
    return materialize_tangent(ref[])
end

"""
    _as_cell(layer)

Return the underlying recurrent cell of `layer`. `Flux.LSTM`/`GRU`/`RNN` wrap a cell
(`LSTMCell`/`GRUCell`/`RNNCell`) in a `.cell` field; if `layer` has no such field it is
already a cell and is returned unchanged.
"""
# Unwrap .cell field if present (LSTM → LSTMCell); return unchanged otherwise.
_as_cell(layer) = hasfield(typeof(layer), :cell) ? layer.cell : layer

"""
    _init_recurrent_state(encoder)

Return the initial recurrent state for `encoder`: `Flux.initialstates(encoder)` for a
single cell, or a tuple of per-layer initial states for a `Chain` of cells.
"""
# Single cell: return (h0, c0) or equivalent from Flux.initialstates.
_init_recurrent_state(cell) = Flux.initialstates(cell)
# Chain of cells: one initial state per layer, returned as a tuple.
_init_recurrent_state(chain::Chain) = map(_init_recurrent_state, chain.layers)

"""
    _step_encoder(encoder, x, state) -> (output, new_state)

Advance `encoder` by one step on input `x` from recurrent `state`, returning the
output and the updated state. For a `Chain` of cells, each layer's output feeds the
next and each layer's state is threaded independently.
"""
# Single cell: one stateful call returns (output, new_state).
_step_encoder(cell, x, state) = cell(x, state)
function _step_encoder(chain::Chain, x, states::Tuple)
    # Delegate to the recursive tuple-based implementation.
    return _step_encoder_layers(chain.layers, x, states)
end

"""
    _step_encoder_layers(layers, x, states) -> (output, new_states)

Recursively advance a tuple of recurrent layers by one time step.

Each layer receives the output of the previous layer as input and its own
independent recurrent state. The base case (`layers == ()`) returns the
input unchanged with an empty state tuple.

# Arguments
- `layers::Tuple`: remaining recurrent cells to evaluate.
- `x`: current input (or output of the prior layer).
- `states::Tuple`: per-layer recurrent states, same length as `layers`.

# Returns
- `output`: output of the last layer in `layers`.
- `new_states::Tuple`: updated recurrent states, one per layer.
"""
_step_encoder_layers(::Tuple{}, x, ::Tuple{}) = x, ()
function _step_encoder_layers(layers::Tuple, x, states::Tuple)
    # Advance the first layer with its own recurrent state.
    out, new_state = _step_encoder(first(layers), x, first(states))

    # Recurse on remaining layers, feeding this layer's output as input.
    rest_out, rest_states = _step_encoder_layers(Base.tail(layers), out, Base.tail(states))

    # Reassemble the full state tuple: this layer's state followed by the rest.
    return rest_out, (new_state, rest_states...)
end

"""
    _state_eltype(state) -> Type

Return the scalar element type of a recurrent state.

For nested tuple states (e.g. LSTM's `(h, c)` or a `Chain`'s tuple of
per-layer states) this recurses into the first element until it reaches an
`AbstractVector`, then returns `eltype(v)`. The result is used to cast
inputs to the encoder's precision before each step.

# Arguments
- `state::Tuple`: nested recurrent state.
- `v::AbstractVector`: leaf state vector.

# Returns
- `Type`: the scalar element type (e.g. `Float32`).
"""
_state_eltype(state::Tuple) = _state_eltype(first(state))
_state_eltype(v::AbstractVector) = eltype(v)

"""
    (m::StateConditionedPolicy)(x) -> AbstractVector

Execute the forward pass of the state-conditioned policy.

The input `x` is split into the uncertainty portion ``w_t`` and the previous
state ``x_{t-1}``. The forward pass computes

```math
h_t, s_t = \\text{encoder}(w_t, s_{t-1}) \\\\
\\hat{x}_t = f_{\\text{combine}}([h_t;\\; x_{t-1}])
```

where ``s_t`` is the updated recurrent state (stored in `m.state` for the
next call) and ``f_{\\text{combine}}`` is a `Dense` layer.

# Arguments
- `x::AbstractVector`: concatenated input `[w_t..., x_{t-1}...]` of length
  `m.n_uncertainty + m.n_state`.

# Returns
- `AbstractVector`: predicted next state ``\\hat{x}_t``.
"""
function (m::StateConditionedPolicy)(x)
    # Split the concatenated input into uncertainty w_t and previous state x_{t-1}.
    uncertainty = x[1:m.n_uncertainty]
    prev_state = x[(m.n_uncertainty + 1):end]

    # Determine the encoder's scalar precision from its current recurrent state.
    # Casting the input avoids a Zygote codegen bug with nested-tuple convert
    # when an upstream solver feeds Float64 into a Float32 LSTM.
    T = _state_eltype(m.state)

    # Advance the recurrent encoder by one step: h_t, s_t = encoder(w_t, s_{t-1}).
    encoded, new_state = _step_encoder(m.encoder, T.(uncertainty), m.state)

    # Persist the new recurrent state so the next call starts from s_t.
    m.state = new_state

    # Concatenate encoder output h_t with previous state x_{t-1}.
    combined = vcat(encoded, prev_state)

    # Map the combined vector through the Dense combiner to produce x_hat_t.
    return m.combiner(combined)
end

"""
    Flux.reset!(m::StateConditionedPolicy)

Reset the encoder's recurrent state to `Flux.initialstates`, e.g. before starting a
new rollout.
"""
function Flux.reset!(m::StateConditionedPolicy)
    # Reinitialize s_0 to the cell's default (zeros for LSTM h/c).
    m.state = _init_recurrent_state(m.encoder)
    return nothing
end

"""
    state_conditioned_policy(n_uncertainty, n_state, n_output, layers;
                             activation=Flux.relu, encoder_type=Flux.LSTM)

Create a [`StateConditionedPolicy`](@ref) with the specified architecture.

The resulting policy computes

```math
h_t, s_t = \\text{encoder}(w_t, s_{t-1}) \\\\
\\hat{x}_t = \\sigma\\bigl(W [h_t;\\; x_{t-1}] + b\\bigr)
```

where the encoder is a stack of recurrent cells of width
``(l_1, \\dots, l_L)`` and the combiner has input dimension
``l_L + n_{\\text{state}}``.

# Arguments
- `n_uncertainty::Int`: dimensionality of the uncertainty input ``w_t``.
- `n_state::Int`: dimensionality of the state ``x_{t-1}`` (both input and output).
- `n_output::Int`: output dimension (typically equal to `n_state`).
- `layers::Vector{Int}`: hidden layer widths ``(l_1, \\dots, l_L)`` for the encoder.
- `activation`: activation ``\\sigma`` for the combiner `Dense` layer
  (default: `Flux.relu`).
- `encoder_type`: recurrent layer/cell constructor (`LSTM`, `GRU`, `RNN`, or
  their `*Cell` variants; default: `Flux.LSTM`). Must support
  `Flux.initialstates` and the stateful `(x, state) -> (output, new_state)`
  interface (Flux >= 0.16).

# Returns
- `StateConditionedPolicy`: ready-to-use policy with initialized recurrent state.

# Examples
```julia
policy = state_conditioned_policy(5, 3, 3, [16, 16])
x = randn(Float32, 8)  # [uncertainty(5); state(3)]
y = policy(x)           # predicted next state (length 3)
```
"""
function state_conditioned_policy(
    n_uncertainty::Int,
    n_state::Int,
    n_output::Int,
    layers::Vector{Int};
    activation=Flux.relu,
    encoder_type=Flux.LSTM,
)
    # Build encoder: a stack of recurrent cells that process only the uncertainty
    # input w_t. The number of hidden layers determines the encoder topology.
    if length(layers) == 0
        # No hidden layers: single cell maps uncertainty directly to n_state.
        encoder = _as_cell(encoder_type(n_uncertainty => n_state))
        encoder_output_dim = n_state
    elseif length(layers) == 1
        # One hidden layer: single cell with the specified width.
        encoder = _as_cell(encoder_type(n_uncertainty => layers[1]))
        encoder_output_dim = layers[1]
    else
        # Multiple hidden layers: chain of cells, first maps from n_uncertainty.
        encoder_layers = [_as_cell(encoder_type(n_uncertainty => layers[1]))]
        for i in 1:(length(layers) - 1)
            # Each subsequent cell maps from the previous layer's width.
            push!(encoder_layers, _as_cell(encoder_type(layers[i] => layers[i + 1])))
        end
        encoder = Chain(encoder_layers...)
        encoder_output_dim = layers[end]
    end

    # Build combiner: Dense([h_t; x_{t-1}]) → x_hat_t with activation σ.
    combiner = Dense(encoder_output_dim + n_state => n_output, activation)

    # Initialize the recurrent state to Flux.initialstates for the chosen cell(s).
    return StateConditionedPolicy(
        encoder, combiner, _init_recurrent_state(encoder), n_uncertainty, n_state
    )
end
