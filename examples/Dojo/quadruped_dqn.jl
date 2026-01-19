# Quadruped DQN Reinforcement Learning (Crux.jl)
# Adapted from Dojo.jl cartpole_dqn.jl style, but trained with Crux DQN via POMDPs.jl

# ### Setup
using Crux
using POMDPs
using Random
using Flux
using Flux: glorot_uniform
using StableRNGs: StableRNG
using POMDPs
using POMDPTools: Deterministic
using CUDA
device(x) = CUDA.functional() ? gpu(x) : cpu(x)

# Load the shared environment definition
include("quadruped_env.jl")

# -----------------------------
# ### POMDPs MDP wrapper for Crux
# -----------------------------
mutable struct QuadrupedMDP{T} <: MDP{Vector{Float32}, Int}
    env::QuadrupedEnv{T}
    S
    A::Vector{Int}
    γ::Float32
end

function QuadrupedMDP(env::QuadrupedEnv{T}; γ=0.99f0) where {T}
    ns = length(state(env))
    S  = Crux.ContinuousSpace((ns,))          # Crux space (state dim)
    A  = collect(action_space(env))           # 1:9
    QuadrupedMDP{T}(env, S, A, γ)
end

POMDPs.discount(m::QuadrupedMDP) = m.γ
POMDPs.actions(m::QuadrupedMDP) = m.A
POMDPs.isterminal(m::QuadrupedMDP, s) = is_terminated(m.env)

# function POMDPs.initialstate(m::QuadrupedMDP, rng::AbstractRNG)
#     reset!(m.env)
#     s = copy(state(m.env))
#     # Ensure we return the full state vector
#     if length(s) != 36
#         println("WARNING: initialstate returned wrong dimension: $(length(s))")
#         println("  state value: $s")
#     end
#     @assert length(s) == 36 "initialstate returned wrong dimension: $(length(s))"
#     return Float32.(s)
# end

POMDPs.initialstate(m::QuadrupedMDP, rng::AbstractRNG) = rand(rng, POMDPs.initialstate(m))

function POMDPs.initialstate(m::QuadrupedMDP)
    reset!(m.env)
    s0 = Float32.(copy(state(m.env)))
    @assert length(s0) == 36
    return Deterministic(s0)
end

# Define state_space for Crux (needed for proper sampling)
function Crux.state_space(m::QuadrupedMDP; μ=0f0, σ=1f0)
    return m.S
end

function POMDPs.gen(m::QuadrupedMDP, s, a::Int, rng::AbstractRNG)
    # NOTE: The environment maintains its own state trajectory
    # The 's' parameter is used by Crux for tracking but we use m.env.state

    # Check if terminal from previous step - if so, reset
    if is_terminated(m.env)
        reset!(m.env)
    end

    # Take action in the environment
    m.env(a)

    # Get next state and reward
    sp = copy(state(m.env))
    if length(sp) != 36
        println("ERROR in gen: state has wrong dimension: $(length(sp))")
        if s isa AbstractVector
            println("  s input length: $(length(s))")
        else
            println("  s input is not a vector: $(typeof(s))")
        end
        println("  sp value: $sp")
        println("  m.env.state: $(m.env.state)")
    end
    @assert length(sp) == 36 "gen returned state with wrong dimension: $(length(sp))"

    r = Float32(reward(m.env))
    terminal = is_terminated(m.env)

    # Return NamedTuple expected by POMDPs/Crux
    return (sp=Float32.(sp), r=r, terminal=terminal)
end

# -----------------------------
# ### DQN Training (Crux)
# -----------------------------
println("Creating Quadruped environment...")
env = QuadrupedEnv(; record=false, rng=MersenneTwister(123))
mdp = QuadrupedMDP(env; γ=0.99f0)

S  = state_space(mdp)
as = actions(mdp)

println("State dimension: ", dim(S))
println("Action count: ", length(as))

seed = 123
rng = StableRNG(seed)

Q() = DiscreteNetwork(
    Chain(
        Dense(dim(S)..., 128, relu; init=glorot_uniform(rng)),
        Dense(128, 128, relu; init=glorot_uniform(rng)),
        Dense(128, length(as); init=glorot_uniform(rng)),
    ),# |> cpu,
    as,
)

# NOTE: N counts *decisions* (your env holds each decision for steps_per_action sim steps)
N_train = 50_000
max_decisions_per_ep = cld(env.params.max_steps, 100)  # ~ 2000/100 = 20

solver = DQN(;
    S = S,
    π = Q(),
    N = N_train,
    ΔN = 4,
    max_steps = max_decisions_per_ep,
    buffer_size = 10_000,
    buffer_init = 500,
    c_opt = (; optimizer = Flux.Adam(1f-3), batch_size = 32),
)

println("\nStarting Crux DQN training...")
policy = solve(solver, mdp)
println("Training complete.")

# -----------------------------
# ### Test and save the actions of the trained policy
# -----------------------------
println("\n=== Testing trained policy ===")
test_env = QuadrupedEnv(; record=true, rng=MersenneTwister(456))
test_mdp = QuadrupedMDP(test_env; γ=0.99f0)

s = initialstate(test_mdp, MersenneTwister(0))
R = 0.0
actions_taken = Int[]
while !isterminal(test_mdp, s)
    a_dist = action(policy, s)           # POMDPs.action(policy, state) returns distribution/vector
    # Extract the best action (argmax) from the distribution
    a = a_dist isa AbstractVector ? argmax(a_dist) : a_dist
    push!(actions_taken, a)
    out = gen(test_mdp, s, a, MersenneTwister(0))
    s = out.sp
    R += out.r
end

println("Test return: ", R)
println("Final x position: ", test_env.state[1])

# Save actions and environment state
using JLD2
jldsave("quadruped_actions.jld2"; actions=actions_taken, test_return=R, final_x=test_env.state[1])
println("Saved actions to quadruped_actions.jld2")