#!/usr/bin/env julia
using Dojo
using DojoEnvironments
using LinearAlgebra

# ============================
# User config
# ============================
# 1 = forward, 2 = backward, 3 = turn right, 4 = turn left, 5 = stop/stand
sequence = [3,3,3,3,3,3] # fill(1, 20) #[1,1,1,1,3,1,1,4,1,1,5,1,1]

cmd_duration = 0.6
dt = 0.001

# Learned params from your Dojo demo run:
# [freq, thigh_amp, thigh_offset, calf_amp, calf_offset]
p_best = [0.26525665033578044,
          0.04331492786316298,
          0.8633149317316002,
         -0.3418982967378247,
         -1.3331979248705708]

Kp = [100.0, 80.0, 60.0]
Kd = [  5.0,  4.0,  3.0]

TURN_SCALE_INNER = 0.85
TURN_SCALE_OUTER = 1.15
HIP_STEER_BIAS   = 0.06
SMOOTH_TAU       = 0.08

U_MAX = 250.0
FAILSAFE_STAND = true

# ============================
# Environment
# ============================
N_per_cmd = Int(round(cmd_duration / dt))
N_total = N_per_cmd * length(sequence)

env = get_environment(
    :quadruped_sampling;
    horizon=N_total,
    timestep=dt,
    joint_limits=Dict(),
    gravity=-9.81,
    contact_body=false
)

# ============================
# Helpers
# ============================
legmovement(k, a, b, c, offset) = a * cos(k*b*0.01*2*pi + offset) + c

# If turning goes the wrong way, swap these sets.
const LEFT_LEGS  = (2, 4)
const RIGHT_LEGS = (1, 3)

function cmd_to_gait(cmd::Int)
    if cmd == 1
        return (1.0, 1.0, 0.0, 0.0, 0.0, true)          # forward
    elseif cmd == 2
        return (1.0, 1.0, 0.0, 0.0, pi,  true)          # backward
    elseif cmd == 3
        return (TURN_SCALE_OUTER, TURN_SCALE_INNER, +HIP_STEER_BIAS, -HIP_STEER_BIAS, 0.0, true)
    elseif cmd == 4
        return (TURN_SCALE_INNER, TURN_SCALE_OUTER, -HIP_STEER_BIAS, +HIP_STEER_BIAS, 0.0, true)
    else
        return (1.0, 1.0, 0.0, 0.0, 0.0, false)         # stop
    end
end

function reset_state!(env, p)
    _, _, Cth, _, Ccf = p
    initialize!(env, :quadruped; body_position=[0.0;0.0;-0.43], hip_angle=0.0, thigh_angle=Cth, calf_angle=Ccf)

    calf_state = get_body(env.mechanism, :FR_calf).state
    position = get_sdf(get_contact(env.mechanism, :FR_calf_contact),
                       Dojo.current_position(calf_state),
                       Dojo.current_orientation(calf_state))

    initialize!(env, :quadruped; body_position=[0.0;0.0;-position-0.43], hip_angle=0.0, thigh_angle=Cth, calf_angle=Ccf)
end

function compute_u12(x, k, p, scaleL, scaleR, hipBL, hipBR, phaseS, osc_on, Kp, Kd)
    freq, Ath, Cth, Acf, Ccf = p

    a_th_L = Ath * scaleL
    a_th_R = Ath * scaleR
    a_cf_L = Acf * scaleL
    a_cf_R = Acf * scaleR

    thigh_A_L = legmovement(k, a_th_L, freq, Cth, 0.0 + phaseS)
    thigh_B_L = legmovement(k, a_th_L, freq, Cth, pi  + phaseS)
    thigh_A_R = legmovement(k, a_th_R, freq, Cth, 0.0 + phaseS)
    thigh_B_R = legmovement(k, a_th_R, freq, Cth, pi  + phaseS)

    calf_A_L  = legmovement(k, a_cf_L, freq, Ccf, -pi/2 + phaseS)
    calf_B_L  = legmovement(k, a_cf_L, freq, Ccf,  pi/2 + phaseS)
    calf_A_R  = legmovement(k, a_cf_R, freq, Ccf, -pi/2 + phaseS)
    calf_B_R  = legmovement(k, a_cf_R, freq, Ccf,  pi/2 + phaseS)

    u = zeros(12)

    for i in 1:4
        θ1  = x[12+(i-1)*6+1]; dθ1 = x[12+(i-1)*6+2]
        θ2  = x[12+(i-1)*6+3]; dθ2 = x[12+(i-1)*6+4]
        θ3  = x[12+(i-1)*6+5]; dθ3 = x[12+(i-1)*6+6]

        is_left = (i in LEFT_LEGS)
        hip_bias = is_left ? hipBL : hipBR

        if !osc_on
            θ2ref = Cth
            θ3ref = Ccf
        else
            if i == 1 || i == 4
                θ2ref = is_left ? thigh_A_L : thigh_A_R
                θ3ref = is_left ? calf_A_L  : calf_A_R
            else
                θ2ref = is_left ? thigh_B_L : thigh_B_R
                θ3ref = is_left ? calf_B_L  : calf_B_R
            end
        end

        u[(i-1)*3 + 1] = Kp[1]*((hip_bias) - θ1) + Kd[1]*(0.0 - dθ1)
        u[(i-1)*3 + 2] = Kp[2]*(θ2ref - θ2)      + Kd[2]*(0.0 - dθ2)
        u[(i-1)*3 + 3] = Kp[3]*(θ3ref - θ3)      + Kd[3]*(0.0 - dθ3)
    end

    return clamp.(u, -U_MAX, U_MAX)
end

# ============================
# Controller state (fixes scoping)
# ============================
mutable struct CtrlState
    scaleL::Float64
    scaleR::Float64
    hipBL::Float64
    hipBR::Float64
    phaseS::Float64
    osc_on::Bool
end

state = CtrlState(1.0, 1.0, 0.0, 0.0, 0.0, true)
α = dt / (SMOOTH_TAU + dt)

# Outputs you want
actions12 = zeros(12, N_total)
actions18 = zeros(18, N_total)

# ============================
# One-pass closed-loop sim + record
# ============================
reset_state!(env, p_best)

function env_controller!(environment, k)
    x = get_state(environment)

    unstable = (x[3] < 0) || (!all(isfinite.(x))) || (abs(x[1]) > 1000)

    cmd = sequence[clamp(Int(fld(k-1, N_per_cmd)) + 1, 1, length(sequence))]
    t_scaleL, t_scaleR, t_hipBL, t_hipBR, t_phaseS, t_osc_on = cmd_to_gait(cmd)

    # smooth transitions
    state.scaleL = (1-α)*state.scaleL + α*t_scaleL
    state.scaleR = (1-α)*state.scaleR + α*t_scaleR
    state.hipBL  = (1-α)*state.hipBL  + α*t_hipBL
    state.hipBR  = (1-α)*state.hipBR  + α*t_hipBR
    state.phaseS = (1-α)*state.phaseS + α*t_phaseS
    state.osc_on = t_osc_on

    if unstable && FAILSAFE_STAND
        u12 = compute_u12(x, k, p_best, 1.0, 1.0, 0.0, 0.0, 0.0, false, Kp, Kd)
    else
        u12 = compute_u12(x, k, p_best, state.scaleL, state.scaleR, state.hipBL, state.hipBR, state.phaseS, state.osc_on, Kp, Kd)
    end

    if 1 <= k <= N_total
        actions12[:, k] .= u12
        actions18[7:18, k] .= u12
    end

    set_input!(environment, u12)
    return nothing
end

simulate!(env, env_controller!; record=true)
vis = visualize(env)
render(vis)

println("Done.")
println("actions12 size = ", size(actions12))
println("actions18 size = ", size(actions18))