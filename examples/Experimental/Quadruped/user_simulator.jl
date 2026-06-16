#!/usr/bin/env julia
using Dojo
using DojoEnvironments
using LinearAlgebra

# ============================
# User config
# ============================
# 1 = forward, 2 = backward, 3 = turn right, 4 = turn left, 5 = stop/stand
# sequence = [1,1,1,1,3,3,3,3,3,2,2,2,2,2,4,4,4,4,5,5,5,5,5,1,1,1,1] # falls
# sequence = [1,1,1,1,5,5,5,3,3,3,5,5,5,2,2,2,5,5,5,4,4,4,5,5,5] # stable
sequence = [1,1,1,1,3,3,3,3,3,3,3,5,5,5,2,2,2,5,5,5,4,4,4,4,4,4,4,4,5,5,5]

cmd_duration = 0.6
dt = 0.001

# Forward gait params (your Dojo demo result)
p_fwd = [0.26525665033578044,
         0.04331492786316298,
         0.8633149317316002,
        -0.3418982967378247,
        -1.3331979248705708]

# Backward gait params (learned)
p_back = [0.21293431713739921,
         -0.26125882252618804,
          0.7896398347924382,
         -0.15418034163543437,
         -1.395866081515512]

# PD gains (same style as Dojo demo)
Kp = [100.0, 80.0, 60.0]   # hip, thigh, calf
Kd = [  5.0,  4.0,  3.0]

# Turning knobs (keep as what worked)
TURN_SCALE_INNER = 0.85
TURN_SCALE_OUTER = 1.15
HIP_STEER_BIAS   = 0.06
SMOOTH_TAU       = 0.08

# Safety
U_MAX = 250.0
FAILSAFE_STAND = true

# IMPORTANT: use YOUR corrected mapping
const LEFT_LEGS  = (2, 4)
const RIGHT_LEGS = (1, 3)

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

function reset_state!(env, p)
    _, _, Cth, _, Ccf = p
    initialize!(env, :quadruped; body_position=[0.0;0.0;-0.43], hip_angle=0.0, thigh_angle=Cth, calf_angle=Ccf)
    calf_state = get_body(env.mechanism, :FR_calf).state
    position = get_sdf(get_contact(env.mechanism, :FR_calf_contact),
                       Dojo.current_position(calf_state),
                       Dojo.current_orientation(calf_state))
    initialize!(env, :quadruped; body_position=[0.0;0.0;-position-0.43], hip_angle=0.0, thigh_angle=Cth, calf_angle=Ccf)
end

# cmd → (which gait params, scaleL, scaleR, hipBL, hipBR, osc_on)
function cmd_to_intent(cmd::Int)
    if cmd == 1
        return (:fwd,  1.0, 1.0, 0.0, 0.0, true)
    elseif cmd == 2
        return (:back, 1.0, 1.0, 0.0, 0.0, true)
    elseif cmd == 3
        return (:fwd,  TURN_SCALE_OUTER, TURN_SCALE_INNER, +HIP_STEER_BIAS, -HIP_STEER_BIAS, true)
    elseif cmd == 4
        return (:fwd,  TURN_SCALE_INNER, TURN_SCALE_OUTER, -HIP_STEER_BIAS, +HIP_STEER_BIAS, true)
    else
        return (:fwd,  1.0, 1.0, 0.0, 0.0, false)
    end
end

function compute_u12(x, k, p, scaleL, scaleR, hipBL, hipBR, osc_on)
    freq, Ath, Cth, Acf, Ccf = p

    a_th_L = Ath * scaleL
    a_th_R = Ath * scaleR
    a_cf_L = Acf * scaleL
    a_cf_R = Acf * scaleR

    thigh_A_L = legmovement(k, a_th_L, freq, Cth, 0.0)
    thigh_B_L = legmovement(k, a_th_L, freq, Cth, pi)
    thigh_A_R = legmovement(k, a_th_R, freq, Cth, 0.0)
    thigh_B_R = legmovement(k, a_th_R, freq, Cth, pi)

    calf_A_L  = legmovement(k, a_cf_L, freq, Ccf, -pi/2)
    calf_B_L  = legmovement(k, a_cf_L, freq, Ccf,  pi/2)
    calf_A_R  = legmovement(k, a_cf_R, freq, Ccf, -pi/2)
    calf_B_R  = legmovement(k, a_cf_R, freq, Ccf,  pi/2)

    u = zeros(12)
    for i in 1:4
        θ1  = x[12+(i-1)*6+1]; dθ1 = x[12+(i-1)*6+2]
        θ2  = x[12+(i-1)*6+3]; dθ2 = x[12+(i-1)*6+4]
        θ3  = x[12+(i-1)*6+5]; dθ3 = x[12+(i-1)*6+6]

        is_left = (i in LEFT_LEGS)
        hip_bias = is_left ? hipBL : hipBR

        if !osc_on
            θ2ref = Cth; θ3ref = Ccf
        else
            useA = (i == 1 || i == 4)  # demo grouping
            if useA
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

mutable struct CtrlState
    scaleL::Float64
    scaleR::Float64
    hipBL::Float64
    hipBR::Float64
    osc_on::Bool
end

# ============================
# Run closed-loop + record actions
# ============================
# Reset using forward pose (works fine; controller will switch p as needed)
reset_state!(env, p_fwd)

actions12 = zeros(12, N_total)
actions18 = zeros(18, N_total)

st = CtrlState(1.0, 1.0, 0.0, 0.0, true)
α = dt / (SMOOTH_TAU + dt)

function env_controller!(environment, k)
    x = get_state(environment)
    unstable = (x[3] < 0) || (!all(isfinite.(x))) || (abs(x[1]) > 1000)

    seg = clamp(Int(fld(k-1, N_per_cmd)) + 1, 1, length(sequence))
    cmd = sequence[seg]
    gait, t_scaleL, t_scaleR, t_hipBL, t_hipBR, t_osc_on = cmd_to_intent(cmd)

    st.scaleL = (1-α)*st.scaleL + α*t_scaleL
    st.scaleR = (1-α)*st.scaleR + α*t_scaleR
    st.hipBL  = (1-α)*st.hipBL  + α*t_hipBL
    st.hipBR  = (1-α)*st.hipBR  + α*t_hipBR
    st.osc_on = t_osc_on

    p = (gait == :back) ? p_back : p_fwd

    u12 = if unstable && FAILSAFE_STAND
        compute_u12(x, k, p_fwd, 1.0, 1.0, 0.0, 0.0, false)
    else
        compute_u12(x, k, p, st.scaleL, st.scaleR, st.hipBL, st.hipBR, st.osc_on)
    end

    actions12[:, k] .= u12
    actions18[7:18, k] .= u12

    set_input!(environment, u12)
    return nothing
end

simulate!(env, env_controller!; record=true)
vis = visualize(env)
render(vis)

println("Done.")
println("actions12 size = ", size(actions12))
println("actions18 size = ", size(actions18))