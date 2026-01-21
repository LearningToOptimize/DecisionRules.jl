#!/usr/bin/env julia
using Dojo
using DojoEnvironments
using LinearAlgebra
using Random
using HDF5
using Statistics

# ============================
# Parse command line arguments
# ============================
if length(ARGS) < 2
    println("Usage: julia generate_dataset.jl <num_trajectories> <output_dir> [random_seed]")
    println("Example: julia generate_dataset.jl 5 ./data 42")
    exit(1)
end

const NUM_TRAJECTORIES = parse(Int, ARGS[1])
const OUTPUT_DIR = ARGS[2]
const RANDOM_SEED = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 42

println("Generating $NUM_TRAJECTORIES trajectories")
println("Output directory: $OUTPUT_DIR")
println("Random seed: $RANDOM_SEED")

Random.seed!(RANDOM_SEED)

# Create output directory
mkpath(OUTPUT_DIR)

# ============================
# Controller parameters (from user_simulator.jl)
# ============================
const dt = 0.001
const cmd_duration = 1.6

# Forward gait params
const p_fwd = [0.26525665033578044,
               0.04331492786316298,
               0.8633149317316002,
              -0.3418982967378247,
              -1.3331979248705708]

# Backward gait params
const p_back = [0.21293431713739921,
               -0.26125882252618804,
                0.7896398347924382,
               -0.15418034163543437,
               -1.395866081515512]

# PD gains
const Kp = [100.0, 80.0, 60.0]
const Kd = [  5.0,  4.0,  3.0]

# Turning knobs
const TURN_SCALE_INNER = 0.85
const TURN_SCALE_OUTER = 1.15
const HIP_STEER_BIAS   = 0.06
const SMOOTH_TAU       = 0.08

# Safety
const U_MAX = 250.0
const FAILSAFE_STAND = true

# Leg mapping
const LEFT_LEGS  = (2, 4)
const RIGHT_LEGS = (1, 3)

# ============================
# Command generation parameters
# ============================
const MIN_COMMANDS = 5
const MAX_COMMANDS = 15
const COMMAND_POOL = [1, 2, 3, 4]  # forward, backward, turn right, turn left
const STAND_CMD = 5

# ============================
# Stability checking
# ============================
"""
Check if a trajectory is stable based on state history.
Returns (is_stable, reason)
"""
function check_stability(states::Matrix{Float64})
    # states: (state_dim, T)
    T = size(states, 2)
    
    # Check 1: z position (height) should stay above ground
    z_positions = states[3, :]
    if any(z_positions .< -0.1)  # robot fell through ground
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
    
    # Check 4: Orientation shouldn't flip too much (check roll/pitch)
    # states[4:6] are quaternion components (assuming [qw, qx, qy, qz] ordering)
    # For a rough check, we can look at body rotation rates or orientation
    velocities = states[7:9, :]  # linear velocities
    if any(abs.(velocities) .> 50.0)
        return (false, "velocity_exploded")
    end
    
    # Check 5: Check if robot tips over - use z position consistency
    z_std = std(z_positions)
    z_mean = mean(z_positions)
    if z_mean < -0.6 || z_std > 0.3  # fell or very unstable
        return (false, "tipped_over")
    end
    
    return (true, "stable")
end

# ============================
# Controller helpers (from user_simulator.jl)
# ============================
function legmovement(k, a, b, c, offset)
    return a * cos(k*b*0.01*2*pi + offset) + c
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
            useA = (i == 1 || i == 4)
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
# Command sequence generation
# ============================
"""
Generate a random command sequence with standing breaks for stability.
Returns vector of command integers.
"""
function generate_command_sequence()
    num_cmds = rand(MIN_COMMANDS:MAX_COMMANDS)
    sequence = Int[]
    
    for i in 1:num_cmds
        # Add a movement command
        push!(sequence, rand(COMMAND_POOL))
        # Add standing break (except after last command)
        if i < num_cmds
            push!(sequence, STAND_CMD)
        end
    end
    
    return sequence
end

# ============================
# Trajectory simulation
# ============================
"""
Simulate one trajectory and return states, actions, and stability info.
"""
function simulate_trajectory(traj_id::Int, sequence::Vector{Int})
    N_per_cmd = Int(round(cmd_duration / dt))
    N_total = N_per_cmd * length(sequence)
    
    println("  Trajectory $traj_id: $(length(sequence)) commands, $N_total timesteps")
    println("  Command sequence: $sequence")
    
    # Create environment
    env = get_environment(
        :quadruped_sampling;
        horizon=N_total,
        timestep=dt,
        joint_limits=Dict(),
        gravity=-9.81,
        contact_body=false
    )
    
    # Reset
    reset_state!(env, p_fwd)
    
    # Storage
    state_dim = length(get_state(env))
    states = zeros(state_dim, N_total)
    actions = zeros(12, N_total)
    
    # Controller state
    st = CtrlState(1.0, 1.0, 0.0, 0.0, true)
    α = dt / (SMOOTH_TAU + dt)
    
    # Controller closure
    function env_controller!(environment, k)
        x = get_state(environment)
        states[:, k] .= x
        
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
        
        actions[:, k] .= u12
        set_input!(environment, u12)
        return nothing
    end
    
    # Simulate
    try
        simulate!(env, env_controller!; record=false)
    catch e
        println("  ERROR: Simulation failed - $e")
        return (states, actions, false, "simulation_error")
    end
    
    # Check stability
    is_stable, reason = check_stability(states)
    println("  Stability: $is_stable ($reason)")
    
    return (states, actions, is_stable, reason)
end

# ============================
# Main data generation loop
# ============================
function generate_dataset()
    println("\n" * "="^60)
    println("Starting dataset generation")
    println("="^60 * "\n")
    
    stable_count = 0
    rejected_count = 0
    attempts = 0
    
    while stable_count < NUM_TRAJECTORIES
        attempts += 1
        traj_id = stable_count + 1
        
        println("\nGenerating trajectory $traj_id/$NUM_TRAJECTORIES (attempt $attempts)...")
        
        # Generate command sequence
        sequence = generate_command_sequence()
        
        # Simulate
        states, actions, is_stable, reason = simulate_trajectory(traj_id, sequence)
        
        # CHECK STABILITY FIRST - only save if stable
        if !is_stable
            rejected_count += 1
            println("  ✗ Rejected: $reason")
            continue
        end
        
        # Save trajectory (only stable ones)
        output_file = joinpath(OUTPUT_DIR, "trajectory_$(lpad(stable_count + 1, 5, '0')).h5")
        
        try
            h5open(output_file, "w") do file
                # Create datasets with compression
                file["states"] = states
                file["actions"] = actions
                file["command_sequence"] = sequence
                
                # Store metadata as attributes
                attrs(file)["trajectory_id"] = stable_count + 1
                attrs(file)["is_stable"] = is_stable
                attrs(file)["stability_reason"] = reason
                attrs(file)["dt"] = dt
                attrs(file)["cmd_duration"] = cmd_duration
                attrs(file)["num_timesteps"] = size(states, 2)
                attrs(file)["state_dim"] = size(states, 1)
                
                # Enable compression on datasets
                file["states"]["compress"] = 4  # gzip level 4
                file["actions"]["compress"] = 4
            end
            
            stable_count += 1
            file_size_kb = filesize(output_file) / 1024
            println("  ✓ Saved stable trajectory to $output_file ($(round(file_size_kb, digits=1)) KB)")
        catch e
            println("  ERROR: Failed to save trajectory - $e")
        end
    end
    
    println("\n" * "="^60)
    println("Dataset generation complete!")
    println("="^60)
    println("Total stable trajectories saved: $stable_count")
    println("Total rejected (unstable): $rejected_count")
    println("Total attempts: $attempts")
    println("Acceptance rate: $(round(100*stable_count/attempts, digits=1))%")
    println("Output directory: $OUTPUT_DIR")
    println("="^60 * "\n")
    
    # Save summary
    summary_file = joinpath(OUTPUT_DIR, "summary.h5")
    h5open(summary_file, "w") do file
        attrs(file)["num_trajectories"] = stable_count
        attrs(file)["stable_count"] = stable_count
        attrs(file)["rejected_count"] = rejected_count
        attrs(file)["total_attempts"] = attempts
        attrs(file)["acceptance_rate"] = stable_count / attempts
        attrs(file)["random_seed"] = RANDOM_SEED
        attrs(file)["dt"] = dt
        attrs(file)["cmd_duration"] = cmd_duration
    end
    println("Summary saved to $summary_file\n")
end

# Run!
generate_dataset()
