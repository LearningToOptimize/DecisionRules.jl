using Functors
using ChainRulesCore

"""
    dense_multilayer_nn(num_inputs, num_outputs, layers; activation=Flux.relu, dense=Dense)

Create a multi-layer neural network with the specified architecture.

# Arguments
- `num_inputs::Int`: Number of input features
- `num_outputs::Int`: Number of output features  
- `layers::Vector{Int}`: Hidden layer sizes
- `activation`: Activation function (default: Flux.relu)
- `dense`: Layer type (Dense, LSTM, etc.)
"""
function dense_multilayer_nn(
    num_inputs::Int,
    num_outputs::Int,
    layers::Vector{Int};
    activation = Flux.relu,
    dense = Dense,
)
    is_recurrent = dense in (LSTM, GRU, RNN)
    _make_layer(in_dim, out_dim) = is_recurrent ? dense(in_dim => out_dim) : dense(in_dim, out_dim, activation)
    if length(layers) == 0
        return is_recurrent ? dense(num_inputs => num_outputs) : dense(num_inputs, num_outputs, activation)
    end
    midlayers = [_make_layer(layers[i], layers[i+1]) for i in 1:(length(layers)-1)]
    first_layer = _make_layer(num_inputs, layers[1])
    last_layer = is_recurrent ? dense(layers[end] => num_outputs) : dense(layers[end], num_outputs)
    return Chain(first_layer, midlayers..., last_layer)
end

"""
    policy_input_dim(num_uncertainties, num_states)

Compute the input dimension for a policy network.

Policy networks receive `[uncertainty..., previous_state...]` as input,
so the input dimension is `num_uncertainties + num_states`.

This format is consistent between subproblems and deterministic equivalent
formulations, enabling warmstarting policies trained with det_eq for use
with subproblems.

# Arguments
- `num_uncertainties::Int`: Number of uncertainty parameters per stage
- `num_states::Int`: Number of state variables

# Returns
- `Int`: Total input dimension for the policy network
"""
function policy_input_dim(num_uncertainties::Int, num_states::Int)
    return num_uncertainties + num_states
end

"""
    policy_input_dim(uncertainty_samples, initial_state)

Compute the input dimension for a policy network from problem data.

# Arguments
- `uncertainty_samples`: Uncertainty samples from problem construction
- `initial_state`: Initial state vector

# Returns
- `Int`: Total input dimension for the policy network
"""
function policy_input_dim(uncertainty_samples::Vector, initial_state::Vector)
    num_uncertainties = length(uncertainty_samples[1])
    num_states = length(initial_state)
    return policy_input_dim(num_uncertainties, num_states)
end

"""
    StateConditionedPolicy

A policy architecture that separates temporal encoding from state conditioning:
- `encoder`: a recurrent cell (`LSTMCell`/`GRUCell`/`RNNCell`, or a `Chain` of them)
  that encodes only the uncertainty sequence (temporal dependencies)
- `combiner`: a `Dense` layer that combines the encoder output with the previous
  state to produce the next state

Flux's recurrent cells are stateless (Flux >= 0.16): each call returns
`(output, new_state)` instead of mutating an internal `Recur`. `StateConditionedPolicy`
therefore carries the encoder's recurrent state itself in `state`, threading it through
one call per stage. Call `Flux.reset!` to clear it (back to `Flux.initialstates`) at the
start of a rollout.

Input format: [uncertainty..., previous_state...]
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
    materialize_tangent(x)

Recursively convert ChainRulesCore tangent types (MutableTangent, Tangent) 
to plain NamedTuples/Arrays that Flux.update! can handle.

This is needed because Zygote produces MutableTangent for mutable structs (like Flux.Recur),
but Flux.update!/Optimisers.jl expects plain NamedTuples.
"""
materialize_tangent(x::Number) = x
materialize_tangent(x::AbstractArray) = x
materialize_tangent(::Nothing) = nothing
materialize_tangent(::ChainRulesCore.NoTangent) = nothing
materialize_tangent(::ChainRulesCore.ZeroTangent) = nothing

function materialize_tangent(t::ChainRulesCore.MutableTangent)
    # MutableTangent stores values in RefValues, extract them
    backing = ChainRulesCore.backing(t)
    return materialize_tangent(backing)
end

function materialize_tangent(t::ChainRulesCore.Tangent)
    backing = ChainRulesCore.backing(t)
    return materialize_tangent(backing)
end

function materialize_tangent(nt::NamedTuple{K}) where {K}
    vals = map(materialize_tangent, values(nt))
    return NamedTuple{K}(vals)
end

function materialize_tangent(t::Tuple)
    return map(materialize_tangent, t)
end

function materialize_tangent(ref::Base.RefValue)
    return materialize_tangent(ref[])
end

"""
    _as_cell(layer)

Return the underlying recurrent cell of `layer`. `Flux.LSTM`/`GRU`/`RNN` wrap a cell
(`LSTMCell`/`GRUCell`/`RNNCell`) in a `.cell` field; if `layer` has no such field it is
already a cell and is returned unchanged.
"""
_as_cell(layer) = hasfield(typeof(layer), :cell) ? layer.cell : layer

"""
    _init_recurrent_state(encoder)

Return the initial recurrent state for `encoder`: `Flux.initialstates(encoder)` for a
single cell, or a tuple of per-layer initial states for a `Chain` of cells.
"""
_init_recurrent_state(cell) = Flux.initialstates(cell)
_init_recurrent_state(chain::Chain) = map(_init_recurrent_state, chain.layers)

"""
    _step_encoder(encoder, x, state) -> (output, new_state)

Advance `encoder` by one step on input `x` from recurrent `state`, returning the
output and the updated state. For a `Chain` of cells, each layer's output feeds the
next and each layer's state is threaded independently.
"""
_step_encoder(cell, x, state) = cell(x, state)
_step_encoder(chain::Chain, x, states::Tuple) =
    _step_encoder_layers(chain.layers, x, states)

_step_encoder_layers(::Tuple{}, x, ::Tuple{}) = x, ()
function _step_encoder_layers(layers::Tuple, x, states::Tuple)
    out, new_state = _step_encoder(first(layers), x, first(states))
    rest_out, rest_states = _step_encoder_layers(Base.tail(layers), out, Base.tail(states))
    return rest_out, (new_state, rest_states...)
end

_state_eltype(state::Tuple) = _state_eltype(first(state))
_state_eltype(v::AbstractVector) = eltype(v)

function (m::StateConditionedPolicy)(x)
    # Split input into uncertainty and previous state
    uncertainty = x[1:m.n_uncertainty]
    prev_state = x[(m.n_uncertainty+1):end]

    # Encode uncertainty through the recurrent encoder, carrying state across calls.
    # Cast to encoder precision to keep the recurrent state type stable
    # (avoids a Zygote codegen bug with nested-tuple convert when solver
    # feeds Float64 into a Float32 LSTM).
    T = _state_eltype(m.state)
    encoded, new_state = _step_encoder(m.encoder, T.(uncertainty), m.state)
    m.state = new_state

    # Combine encoded uncertainty with previous state
    combined = vcat(encoded, prev_state)

    # Output next state prediction
    return m.combiner(combined)
end

"""
    Flux.reset!(m::StateConditionedPolicy)

Reset the encoder's recurrent state to `Flux.initialstates`, e.g. before starting a
new rollout.
"""
function Flux.reset!(m::StateConditionedPolicy)
    m.state = _init_recurrent_state(m.encoder)
    return nothing
end

"""
    state_conditioned_policy(n_uncertainty, n_state, n_output, layers;
                             activation=Flux.relu, encoder_type=Flux.LSTM)

Create a StateConditionedPolicy with the specified architecture.

# Arguments
- `n_uncertainty::Int`: Number of uncertainty input dimensions
- `n_state::Int`: Number of state dimensions (both input and output)
- `n_output::Int`: Number of output dimensions (typically same as n_state)
- `layers::Vector{Int}`: Hidden layer sizes for the encoder
- `activation`: Activation function for dense layers (default: relu)
- `encoder_type`: Recurrent layer/cell type (`LSTM`, `GRU`, `RNN`, or their `*Cell`
  variants; default: `Flux.LSTM`). Must support `Flux.initialstates` and the stateful
  `(x, state) -> (output, new_state)` call (Flux >= 0.16).

# Architecture
- Encoder: encoder_type(n_uncertainty => layers[1]) -> ... -> layers[end]
- Combiner: Dense(layers[end] + n_state => n_output)
"""
function state_conditioned_policy(
    n_uncertainty::Int,
    n_state::Int,
    n_output::Int,
    layers::Vector{Int};
    activation = Flux.relu,
    encoder_type = Flux.LSTM,
)
    # Build encoder (stack of recurrent cells that process uncertainty)
    if length(layers) == 0
        encoder = _as_cell(encoder_type(n_uncertainty => n_state))
        encoder_output_dim = n_state
    elseif length(layers) == 1
        encoder = _as_cell(encoder_type(n_uncertainty => layers[1]))
        encoder_output_dim = layers[1]
    else
        encoder_layers = [_as_cell(encoder_type(n_uncertainty => layers[1]))]
        for i = 1:(length(layers)-1)
            push!(encoder_layers, _as_cell(encoder_type(layers[i] => layers[i+1])))
        end
        encoder = Chain(encoder_layers...)
        encoder_output_dim = layers[end]
    end

    # Build combiner (Dense that combines encoder output with previous state)
    # Input: [encoded_uncertainty, previous_state]
    # Output: next_state
    combiner = Dense(encoder_output_dim + n_state => n_output, activation)

    return StateConditionedPolicy(
        encoder,
        combiner,
        _init_recurrent_state(encoder),
        n_uncertainty,
        n_state,
    )
end
