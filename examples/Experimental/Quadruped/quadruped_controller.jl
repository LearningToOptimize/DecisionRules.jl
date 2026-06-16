# ============================================================================
# Shared quadruped controller module
# ============================================================================
# Contains the gait controller parameters and logic shared between
# dataset generation and trajectory visualization/validation.
#
# Usage:
#   include("quadruped_controller.jl")
# ============================================================================

using Dojo
using DojoEnvironments
using LinearAlgebra

# ============================
# Controller parameters
# ============================
const QC_dt = 0.001
const QC_cmd_duration = 1.6

# Forward gait params
const QC_p_fwd = [0.26525665033578044,
                  0.04331492786316298,
                  0.8633149317316002,
                 -0.3418982967378247,
                 -1.3331979248705708]

# Backward gait params
const QC_p_back = [0.21293431713739921,
                  -0.26125882252618804,
                   0.7896398347924382,
                  -0.15418034163543437,
                  -1.395866081515512]

# PD gains
const QC_Kp = [100.0, 80.0, 60.0]
const QC_Kd = [  5.0,  4.0,  3.0]

# Turning knobs
const QC_TURN_SCALE_INNER = 0.85
const QC_TURN_SCALE_OUTER = 1.15
const QC_HIP_STEER_BIAS   = 0.06
const QC_SMOOTH_TAU       = 0.08

# Safety
const QC_U_MAX = 250.0
const QC_FAILSAFE_STAND = true

# Leg mapping
const QC_LEFT_LEGS  = (2, 4)
const QC_RIGHT_LEGS = (1, 3)

# Command encoding
const QC_CMD_FORWARD = 1
const QC_CMD_BACKWARD = 2
const QC_CMD_TURN_RIGHT = 3
const QC_CMD_TURN_LEFT = 4
const QC_CMD_STAND = 5

# ============================
# Controller state
# ============================
mutable struct QuadrupedCtrlState
    scaleL::Float64
    scaleR::Float64
    hipBL::Float64
    hipBR::Float64
    osc_on::Bool
end

QuadrupedCtrlState() = QuadrupedCtrlState(1.0, 1.0, 0.0, 0.0, true)

# ============================
# Helper functions
# ============================
function qc_legmovement(k, a, b, c, offset)
    return a * cos(k * b * 0.01 * 2 * pi + offset) + c
end

"""
Reset quadruped to initial standing position based on gait parameters.
"""
function qc_reset_state!(env, p=QC_p_fwd)
    _, _, Cth, _, Ccf = p
    initialize!(env, :quadruped; body_position=[0.0; 0.0; -0.43], hip_angle=0.0, thigh_angle=Cth, calf_angle=Ccf)
    calf_state = get_body(env.mechanism, :FR_calf).state
    position = get_sdf(get_contact(env.mechanism, :FR_calf_contact),
                       Dojo.current_position(calf_state),
                       Dojo.current_orientation(calf_state))
    initialize!(env, :quadruped; body_position=[0.0; 0.0; -position - 0.43], hip_angle=0.0, thigh_angle=Cth, calf_angle=Ccf)
end

"""
Convert command integer to controller intent.
Returns (gait, scaleL, scaleR, hipBL, hipBR, osc_on)
"""
function qc_cmd_to_intent(cmd::Int)
    if cmd == QC_CMD_FORWARD
        return (:fwd, 1.0, 1.0, 0.0, 0.0, true)
    elseif cmd == QC_CMD_BACKWARD
        return (:back, 1.0, 1.0, 0.0, 0.0, true)
    elseif cmd == QC_CMD_TURN_RIGHT
        return (:fwd, QC_TURN_SCALE_OUTER, QC_TURN_SCALE_INNER, +QC_HIP_STEER_BIAS, -QC_HIP_STEER_BIAS, true)
    elseif cmd == QC_CMD_TURN_LEFT
        return (:fwd, QC_TURN_SCALE_INNER, QC_TURN_SCALE_OUTER, -QC_HIP_STEER_BIAS, +QC_HIP_STEER_BIAS, true)
    else  # STAND or unknown
        return (:fwd, 1.0, 1.0, 0.0, 0.0, false)
    end
end

"""
Compute 12-dim joint torques from state and gait parameters.
"""
function qc_compute_u12(x, k, p, scaleL, scaleR, hipBL, hipBR, osc_on)
    freq, Ath, Cth, Acf, Ccf = p

    a_th_L = Ath * scaleL
    a_th_R = Ath * scaleR
    a_cf_L = Acf * scaleL
    a_cf_R = Acf * scaleR

    thigh_A_L = qc_legmovement(k, a_th_L, freq, Cth, 0.0)
    thigh_B_L = qc_legmovement(k, a_th_L, freq, Cth, pi)
    thigh_A_R = qc_legmovement(k, a_th_R, freq, Cth, 0.0)
    thigh_B_R = qc_legmovement(k, a_th_R, freq, Cth, pi)

    calf_A_L = qc_legmovement(k, a_cf_L, freq, Ccf, -pi / 2)
    calf_B_L = qc_legmovement(k, a_cf_L, freq, Ccf, pi / 2)
    calf_A_R = qc_legmovement(k, a_cf_R, freq, Ccf, -pi / 2)
    calf_B_R = qc_legmovement(k, a_cf_R, freq, Ccf, pi / 2)

    u = zeros(12)
    for i in 1:4
        θ1 = x[12+(i-1)*6+1]; dθ1 = x[12+(i-1)*6+2]
        θ2 = x[12+(i-1)*6+3]; dθ2 = x[12+(i-1)*6+4]
        θ3 = x[12+(i-1)*6+5]; dθ3 = x[12+(i-1)*6+6]

        is_left = (i in QC_LEFT_LEGS)
        hip_bias = is_left ? hipBL : hipBR

        if !osc_on
            θ2ref = Cth
            θ3ref = Ccf
        else
            useA = (i == 1 || i == 4)
            if useA
                θ2ref = is_left ? thigh_A_L : thigh_A_R
                θ3ref = is_left ? calf_A_L : calf_A_R
            else
                θ2ref = is_left ? thigh_B_L : thigh_B_R
                θ3ref = is_left ? calf_B_L : calf_B_R
            end
        end

        u[(i-1)*3+1] = QC_Kp[1] * ((hip_bias) - θ1) + QC_Kd[1] * (0.0 - dθ1)
        u[(i-1)*3+2] = QC_Kp[2] * (θ2ref - θ2) + QC_Kd[2] * (0.0 - dθ2)
        u[(i-1)*3+3] = QC_Kp[3] * (θ3ref - θ3) + QC_Kd[3] * (0.0 - dθ3)
    end

    return clamp.(u, -QC_U_MAX, QC_U_MAX)
end

"""
Create a feedback controller for a given command sequence.

Returns a tuple (controller!, ctrl_state, states_storage, actions_storage) where:
- controller!: function(env, k) for use with simulate!
- ctrl_state: QuadrupedCtrlState that can be inspected
- states_storage: pre-allocated matrix for states
- actions_storage: pre-allocated matrix for actions
"""
function qc_create_controller(sequence::Vector{Int}, N_total::Int; dt::Float64=QC_dt)
    N_per_cmd = Int(round(QC_cmd_duration / dt))
    
    # Pre-allocate storage (will be filled during simulation)
    # We need to determine state_dim at runtime, so we start with nothing
    states_storage = Ref{Matrix{Float64}}(zeros(0, 0))
    actions_storage = zeros(12, N_total)
    
    # Controller state
    ctrl_state = QuadrupedCtrlState()
    α = dt / (QC_SMOOTH_TAU + dt)
    
    function controller!(environment, k)
        x = get_state(environment)
        
        # Initialize states storage on first call if needed
        if size(states_storage[], 1) == 0
            state_dim = length(x)
            states_storage[] = zeros(state_dim, N_total)
        end
        
        # Store state
        states_storage[][:, k] .= x
        
        # Check for instability
        unstable = (x[3] < 0) || (!all(isfinite.(x))) || (abs(x[1]) > 1000)
        
        # Get current command
        seg = clamp(Int(fld(k - 1, N_per_cmd)) + 1, 1, length(sequence))
        cmd = sequence[seg]
        gait, t_scaleL, t_scaleR, t_hipBL, t_hipBR, t_osc_on = qc_cmd_to_intent(cmd)
        
        # Smooth transition
        ctrl_state.scaleL = (1 - α) * ctrl_state.scaleL + α * t_scaleL
        ctrl_state.scaleR = (1 - α) * ctrl_state.scaleR + α * t_scaleR
        ctrl_state.hipBL = (1 - α) * ctrl_state.hipBL + α * t_hipBL
        ctrl_state.hipBR = (1 - α) * ctrl_state.hipBR + α * t_hipBR
        ctrl_state.osc_on = t_osc_on
        
        # Select gait parameters
        p = (gait == :back) ? QC_p_back : QC_p_fwd
        
        # Compute control
        u12 = if unstable && QC_FAILSAFE_STAND
            qc_compute_u12(x, k, QC_p_fwd, 1.0, 1.0, 0.0, 0.0, false)
        else
            qc_compute_u12(x, k, p, ctrl_state.scaleL, ctrl_state.scaleR, ctrl_state.hipBL, ctrl_state.hipBR, ctrl_state.osc_on)
        end
        
        # Store action
        actions_storage[:, k] .= u12
        
        # Apply control
        set_input!(environment, u12)
        return nothing
    end
    
    return (controller!, ctrl_state, states_storage, actions_storage)
end

"""
Create a quadruped environment with standard parameters.
"""
function qc_create_environment(N_total::Int; dt::Float64=QC_dt)
    return get_environment(
        :quadruped_sampling;
        horizon=N_total,
        timestep=dt,
        joint_limits=Dict(),
        gravity=-9.81,
        contact_body=false
    )
end

"""
Simulate a trajectory with the feedback controller.

Returns (states, actions, is_stable, reason)
"""
function qc_simulate_trajectory(sequence::Vector{Int}; dt::Float64=QC_dt, record::Bool=false)
    N_per_cmd = Int(round(QC_cmd_duration / dt))
    N_total = N_per_cmd * length(sequence)
    
    # Create environment
    env = qc_create_environment(N_total; dt=dt)
    
    # Reset to initial standing position
    qc_reset_state!(env, QC_p_fwd)
    
    # Create controller
    controller!, ctrl_state, states_ref, actions = qc_create_controller(sequence, N_total; dt=dt)
    
    # Simulate
    try
        simulate!(env, controller!; record=record)
    catch e
        states = states_ref[]
        if size(states, 1) == 0
            # Controller never ran - create dummy states
            states = zeros(36, N_total)
        end
        return (states, actions, false, "simulation_error: $e", env)
    end
    
    states = states_ref[]
    
    # Check stability
    is_stable, reason = qc_check_stability(states)
    
    return (states, actions, is_stable, reason, env)
end

"""
Check if a trajectory is stable based on state history.
Returns (is_stable, reason)
"""
function qc_check_stability(states::Matrix{Float64})
    T = size(states, 2)
    
    # Check 1: z position (height) should stay above ground
    z_positions = states[3, :]
    if any(z_positions .< -0.1)
        return (false, "fell_through_ground")
    end
    
    # Check 2: Check for NaN or Inf values
    if !all(isfinite.(states))
        return (false, "numerical_instability")
    end
    
    # Check 3: Position shouldn't explode
    positions = states[1:3, :]
    max_pos = maximum(abs.(positions))
    if max_pos > 100.0
        return (false, "position_exploded")
    end
    
    # Check 4: Velocity check
    velocities = states[7:9, :]
    if any(abs.(velocities) .> 50.0)
        return (false, "velocity_exploded")
    end
    
    # Check 5: Check if robot tips over
    z_std = std(z_positions)
    z_mean = mean(z_positions)
    if z_mean < -0.6 || z_std > 0.3
        return (false, "tipped_over")
    end
    
    return (true, "stable")
end

# Export key functions/types
# (In Julia, you'd typically use a module for this, but include works too)
