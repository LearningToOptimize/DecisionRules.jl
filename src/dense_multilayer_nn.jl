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
function dense_multilayer_nn(num_inputs::Int, num_outputs::Int, layers::Vector{Int}; activation=Flux.relu, dense=Dense)
    if length(layers) == 0
        if dense == LSTM
            return dense(num_inputs, num_outputs)
        end
        return dense(num_inputs, num_outputs, activation)
    end
    midlayers = []
    for i in 1:length(layers) - 1
        if dense == LSTM
            push!(midlayers, dense(layers[i], layers[i + 1]))
        else
            push!(midlayers, dense(layers[i], layers[i + 1], activation))
        end
    end
    first_layer = if dense == LSTM
        dense(num_inputs, layers[1])
    else
        dense(num_inputs, layers[1], activation)
    end
    model = Chain(first_layer, midlayers..., dense(layers[end], num_outputs))
    return model
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
- LSTM/RNN: Encodes only the uncertainty sequence (temporal dependencies)
- Dense: Combines LSTM output with previous state to produce next state

This design allows the LSTM to work independently of the state recurrence,
making it more memory-efficient and compatible with the original training loop.

Input format: [uncertainty..., previous_state...]
"""
struct StateConditionedPolicy{E, C}
    encoder::E          # LSTM/RNN that processes uncertainty only
    combiner::C         # Dense that combines encoder output with previous state
    n_uncertainty::Int  # Number of uncertainty dimensions
    n_state::Int        # Number of state dimensions
end

# Use Functors.@functor with explicit trainable fields (excludes n_uncertainty, n_state)
# We use Functors.@functor instead of Flux.@layer to avoid MutableTangent issues with LSTM
Functors.@functor StateConditionedPolicy (encoder, combiner,)

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

function (m::StateConditionedPolicy)(x)
    # Split input into uncertainty and previous state
    uncertainty = x[1:m.n_uncertainty]
    prev_state = x[m.n_uncertainty+1:end]
    
    # Encode uncertainty through LSTM (temporal encoding)
    encoded = m.encoder(uncertainty)
    
    # Combine encoded uncertainty with previous state
    combined = vcat(encoded, prev_state)
    
    # Output next state prediction
    return m.combiner(combined)
end

# Reset hidden state of the encoder (for LSTM/RNN)
Flux.reset!(m::StateConditionedPolicy) = Flux.reset!(m.encoder)

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
- `encoder_type`: Type of encoder (LSTM, RNN, GRU) (default: LSTM)

# Architecture
- Encoder: encoder_type(n_uncertainty => layers[1]) -> ... -> layers[end]
- Combiner: Dense(layers[end] + n_state => n_output)
"""
function state_conditioned_policy(n_uncertainty::Int, n_state::Int, n_output::Int, layers::Vector{Int}; 
                                  activation=Flux.relu, encoder_type=Flux.LSTM)
    # Build encoder (LSTM stack that processes uncertainty)
    if length(layers) == 0
        encoder = encoder_type(n_uncertainty => n_state)
        encoder_output_dim = n_state
    elseif length(layers) == 1
        encoder = encoder_type(n_uncertainty => layers[1])
        encoder_output_dim = layers[1]
    else
        encoder_layers = []
        push!(encoder_layers, encoder_type(n_uncertainty => layers[1]))
        for i in 1:length(layers)-1
            push!(encoder_layers, encoder_type(layers[i] => layers[i+1]))
        end
        encoder = Chain(encoder_layers...)
        encoder_output_dim = layers[end]
    end
    
    # Build combiner (Dense that combines encoder output with previous state)
    # Input: [encoded_uncertainty, previous_state]
    # Output: next_state
    combiner = Dense(encoder_output_dim + n_state => n_output, activation)
    
    return StateConditionedPolicy(encoder, combiner, n_uncertainty, n_state)
end