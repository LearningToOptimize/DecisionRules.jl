# Quadruped DQN Reinforcement Learning (Crux.jl)
# Adapted from Dojo.jl cartpole_dqn.jl style, but trained with Crux DQN via POMDPs.jl

# ### Setup
using Crux
using POMDPs
using POMDPTools: Deterministic
using Random
using Flux
# using Optimisers
using Flux: glorot_uniform
using StableRNGs: StableRNG
using Dojo
using DojoEnvironments
using LinearAlgebra

# -----------------------------
# ### Custom Quadruped Environment (stateful, Dojo-backed)
# -----------------------------
struct QuadrupedEnvParams{T}
    max_steps::Int
    dt::T
    x_target::T
    path_waypoints::Vector{Vector{Float64}}
    max_y_deviation::T
    gravity::T
end

function QuadrupedEnvParams{T}(;
    max_steps=2000,
    dt=0.001,
    x_target=1.5,
    path_waypoints=[
        [0.0, 0.0],
        [0.3, 0.2],
        [0.6, 0.3],
        [0.9, 0.2],
        [1.2, 0.0],
        [1.5, -0.2]
    ],
    max_y_deviation=0.3,
    gravity=-9.81
) where {T}
    QuadrupedEnvParams{T}(max_steps, dt, x_target, path_waypoints, max_y_deviation, gravity)
end

mutable struct QuadrupedEnv{T}
    dojo_env::DojoEnvironments.Environment
    record::Bool
    params::QuadrupedEnvParams{T}
    state::Vector{T}
    action::Vector{T}     # gait params [freq, amp1, offset1, amp2, offset2]
    done::Bool
    t::Int
    last_x::T
    rng::AbstractRNG
end

function QuadrupedEnv(;
    T=Float64,
    record=false,
    rng=Random.GLOBAL_RNG,
    kwargs...
)
    params = QuadrupedEnvParams{T}(; kwargs...)

    dojo_env = get_environment(:quadruped_sampling;
        horizon=params.max_steps,
        timestep=params.dt,
        gravity=params.gravity,
        contact_body=false
    )

    env = QuadrupedEnv{T}(
        dojo_env,
        record,
        params,
        zeros(T, 36),
        zeros(T, 5),
        false,
        0,
        zero(T),
        rng
    )

    reset!(env)
    return env
end

# --- API helpers (keep these simple and explicit) ---
state(env::QuadrupedEnv) = env.state
action_space(env::QuadrupedEnv) = Base.OneTo(9)
is_terminated(env::QuadrupedEnv) = env.done

function reset!(env::QuadrupedEnv{T}) where {T}
    DojoEnvironments.initialize!(env.dojo_env, :quadruped;
        body_position=[0.0, 0.0, 0.0],
        hip_angle=0.0,
        thigh_angle=1.0,
        calf_angle=-1.5
    )

    env.state = DojoEnvironments.get_state(env.dojo_env)
    env.t = 0
    env.done = false
    env.last_x = env.state[1]

    # random initial gait params (not that important, but keeps things non-degenerate)
    env.action = T.(rand(env.rng, 5) .* [0.2, 0.3, 2.0, 0.3, 2.0])

    return nothing
end

# Reward: progress toward target + staying on path - penalties
# NOTE: this updates env.last_x as a side-effect so progress works across decisions.
function reward(env::QuadrupedEnv{T}) where {T}
    x_pos = env.state[1]
    y_pos = env.state[2]
    z_pos = env.state[3]

    if env.done
        if z_pos < 0
            return T(-100.0)
        end
        if x_pos >= env.params.x_target
            return T(100.0)
        end
        return T(-50.0)  # timeout / failure
    end

    # Forward progress since last decision
    progress_reward = (x_pos - env.last_x) * T(10.0)
    env.last_x = x_pos

    # Path following (interpolate target y)
    target_y = T(0.0)
    if x_pos <= env.params.path_waypoints[1][1]
        target_y = T(env.params.path_waypoints[1][2])
    elseif x_pos >= env.params.path_waypoints[end][1]
        target_y = T(env.params.path_waypoints[end][2])
    else
        for j in 1:(length(env.params.path_waypoints)-1)
            if env.params.path_waypoints[j][1] <= x_pos <= env.params.path_waypoints[j+1][1]
                x1, y1 = env.params.path_waypoints[j]
                x2, y2 = env.params.path_waypoints[j+1]
                τ = (x_pos - x1) / (x2 - x1)
                target_y = T(y1 + τ * (y2 - y1))
                break
            end
        end
    end

    y_deviation = abs(y_pos - target_y)
    path_reward = -y_deviation * T(5.0)

    alive_bonus = T(0.1)

    return progress_reward + path_reward + alive_bonus
end

# -----------------------------
# ### Discrete gait library + controller
# -----------------------------
function discrete_to_gait(action::Int)
    gait_library = [
        [0.1, 0.0, 1.0, 0.0, -1.5],
        [0.15, 0.1, 1.0, 0.1, -1.5],
        [0.2, 0.15, 1.0, 0.15, -1.5],
        [0.25, 0.2, 0.9, 0.2, -1.4],
        [0.2, 0.15, 1.1, 0.15, -1.6],
        [0.2, 0.15, 0.9, 0.15, -1.4],
        [0.18, 0.12, 1.05, 0.12, -1.55],
        [0.22, 0.18, 0.95, 0.18, -1.45],
        [0.12, 0.05, 1.0, 0.05, -1.5],
    ]
    return gait_library[action]
end

function gait_controller(state, gait_params, k)
    freq, amp1, offset1, amp2, offset2 = gait_params

    legmovement(k, a, b, c, phase) = a * cos(k * b * 0.01 * 2π + phase) + c

    angle21 = legmovement(k, amp1, freq, offset1, 0)
    angle22 = legmovement(k, amp1, freq, offset1, π)
    angle31 = legmovement(k, amp2, freq, offset2, -π/2)
    angle32 = legmovement(k, amp2, freq, offset2,  π/2)

    Kp = [100.0, 80.0, 60.0]
    Kd = [5.0, 4.0, 3.0]

    u = zeros(12)
    for i in 1:4
        θ1  = state[12 + (i-1)*6 + 1]
        dθ1 = state[12 + (i-1)*6 + 2]
        θ2  = state[12 + (i-1)*6 + 3]
        dθ2 = state[12 + (i-1)*6 + 4]
        θ3  = state[12 + (i-1)*6 + 5]
        dθ3 = state[12 + (i-1)*6 + 6]

        if i == 1 || i == 4
            u[(i-1)*3 + 1] = Kp[1] * (0 - θ1)     + Kd[1] * (0 - dθ1)
            u[(i-1)*3 + 2] = Kp[2] * (angle21-θ2) + Kd[2] * (0 - dθ2)
            u[(i-1)*3 + 3] = Kp[3] * (angle31-θ3) + Kd[3] * (0 - dθ3)
        else
            u[(i-1)*3 + 1] = Kp[1] * (0 - θ1)     + Kd[1] * (0 - dθ1)
            u[(i-1)*3 + 2] = Kp[2] * (angle22-θ2) + Kd[2] * (0 - dθ2)
            u[(i-1)*3 + 3] = Kp[3] * (angle32-θ3) + Kd[3] * (0 - dθ3)
        end
    end

    return u
end

# Apply one discrete action for many sim steps (action-hold)
function (env::QuadrupedEnv)(a::Int)
    @assert a in action_space(env)

    env.action = discrete_to_gait(a)
    steps_per_action = 100

    for _ in 1:steps_per_action
        env.t += 1

        u = gait_controller(env.state, env.action, env.t)

        DojoEnvironments.step!(env.dojo_env, env.state, u; k=env.t, record=env.record)
        env.state = DojoEnvironments.get_state(env.dojo_env)
        
        # Safety check: ensure state is still the right size
        if length(env.state) != 36
            println("ERROR: env.state corrupted! Length = $(length(env.state))")
            println("  env.state = $(env.state)")
            error("State dimension mismatch after step!")
        end

        x_pos = env.state[1]
        z_pos = env.state[3]

        env.done = (
            z_pos < 0 ||
            !all(isfinite.(env.state)) ||
            abs(x_pos) > 100.0 ||
            x_pos >= env.params.x_target ||
            env.t >= env.params.max_steps
        )

        env.done && break
    end

    return nothing
end

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
    ) |> cpu,
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
# ### Test the trained policy
# -----------------------------
println("\n=== Testing trained policy ===")
test_env = QuadrupedEnv(; record=true, rng=MersenneTwister(456))
test_mdp = QuadrupedMDP(test_env; γ=0.99f0)

s = initialstate(test_mdp, MersenneTwister(0))
R = 0.0
while !isterminal(test_mdp, s)
    a = action(policy, s)                # POMDPs.action(policy, state)
    out = gen(test_mdp, s, a, MersenneTwister(0))
    s = out.sp
    R += out.r
end

println("Test return: ", R)
println("Final x position: ", test_env.state[1])

# Visualize
println("\nVisualizing test trajectory...")
vis = DojoEnvironments.visualize(test_env.dojo_env)
DojoEnvironments.render(vis)