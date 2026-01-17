# Quadruped Environment Definition
# Shared between quadruped_dqn.jl (training) and visualize_actions.jl (visualization)

using Dojo
using DojoEnvironments
using Random

# --- Environment Parameters ---
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

# --- Environment State ---
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

# --- API helpers ---
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

# --- Reward Function ---
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

# --- Discrete Gait Library ---
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

# --- Gait Controller ---
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

# --- Environment Step (apply one discrete action for many sim steps) ---
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
