#!/usr/bin/env julia
# ==============================================================================
# Heuristic open-loop gait generator for Dojo quadruped
#
# - Input: a scripted sequence like [1,2,1,1,...]
# - Output: `actions` (nu x N) containing all control inputs
# - Then simulates and renders using Dojo's visualize()/render() pipeline
#
# Requires:
#   ] add Dojo DojoEnvironments
#
# Notes:
#   - This is a *simple* torque-impulse gait heuristic (open-loop CPG-ish).
#   - It does NOT try to be dynamically consistent; it’s meant as a warm-start
#     seed generator / data generator.
# ==============================================================================

using Dojo
using DojoEnvironments
using LinearAlgebra
using Random

# ----------------------------
# Command encoding
# ----------------------------
const CMD_FWD        = 1
const CMD_BWD        = 2
const CMD_RIGHT      = 3
const CMD_LEFT       = 4
const CMD_TURN_RIGHT = 5
const CMD_TURN_LEFT  = 6
const CMD_STOP       = 7

"""
Map an integer command to a (vx, vy, yawrate) "intent".
Units are arbitrary (they just scale the gait).
"""
function cmd_to_intent(cmd::Int)
    # Forward is +vx, right is +vy, turn right is -yawrate (RH yaw)
    if cmd == CMD_FWD
        return ( 1.0,  0.0,  0.0)
    elseif cmd == CMD_BWD
        return (-1.0,  0.0,  0.0)
    elseif cmd == CMD_RIGHT
        return ( 0.0,  1.0,  0.0)
    elseif cmd == CMD_LEFT
        return ( 0.0, -1.0,  0.0)
    elseif cmd == CMD_TURN_RIGHT
        return ( 0.3,  0.0, -1.0)
    elseif cmd == CMD_TURN_LEFT
        return ( 0.3,  0.0,  1.0)
    elseif cmd == CMD_STOP
        return ( 0.0,  0.0,  0.0)
    else
        # unknown -> stop
        return (0.0, 0.0, 0.0)
    end
end

# ----------------------------
# Quadruped mechanism selection
# ----------------------------
"""
Try a few common DojoEnvironments names for quadrupeds.
You can hard-set `mech_name` below if you already know it.
"""
function get_quadruped_mechanism(; timestep::Float64)
    candidates = (
        :quadruped,
        :anymal,
        :go1,
        :aliengo,
        :spot,
        :mini_cheetah,
        :quadruped_min,
    )
    last_err = nothing
    for sym in candidates
        try
            mech = get_mechanism(sym; timestep=timestep)
            return mech, sym
        catch err
            last_err = err
        end
    end
    error("Could not load a quadruped mechanism from DojoEnvironments. Last error:\n$last_err")
end

# ----------------------------
# Heuristic gait (open-loop torque impulses)
# ----------------------------
"""
Trot-like phase offsets for 4 legs:
  - FR & RL in phase
  - FL & RR in opposite phase
Leg order assumed [FL, FR, RL, RR] for *our internal generator*.

We will map to joint inputs in blocks of 3 per leg if possible:
  [hip_ab/ad, hip_pitch, knee] per leg  => 12 inputs total
If the mechanism has a different input_dimension, we:
  - fill the first min(nu,12) with our pattern
  - leave the rest as 0.0
"""
function gait_u12(t::Float64, intent, params)
    vx, vy, wz = intent
    # Normalize intent a bit
    vmag = clamp(sqrt(vx^2 + vy^2), 0.0, 1.5)
    vx_n = clamp(vx, -1.5, 1.5)
    vy_n = clamp(vy, -1.5, 1.5)
    wz_n = clamp(wz, -2.0, 2.0)

    # Base oscillator
    ω = 2π * params.gait_hz
    ϕ = ω * t

    # Trot phasing (FL, FR, RL, RR)
    #   FR & RL: phase 0
    #   FL & RR: phase π
    leg_phase = (π, 0.0, 0.0, π)

    # Amplitudes (impulse-torque-ish)
    Ahip   = params.Ahip_base   + params.Ahip_vel * abs(vx_n) + params.Ahip_lat * abs(vy_n)
    Aknee  = params.Aknee_base  + params.Aknee_vel * vmag
    Aabd   = params.Aabd_base   + params.Aabd_lat * abs(vy_n) + params.Aabd_yaw * abs(wz_n)

    # Bias terms to steer/turn (small offsets)
    hip_bias_pitch = params.hip_bias_pitch * vx_n
    hip_bias_abd   = params.hip_bias_abd   * vy_n
    yaw_bias_abd   = params.yaw_bias_abd   * wz_n

    # Build u (12)
    u = zeros(12)

    # Signs to make lateral & yaw do something different per leg
    # (very heuristic): left legs vs right legs
    # leg index: 1 FL, 2 FR, 3 RL, 4 RR
    side_sign = ( +1.0, -1.0, +1.0, -1.0 )  # left:+, right:-
    front_sign = ( +1.0, +1.0, -1.0, -1.0 ) # front:+, rear:-

    for i in 1:4
        ϕi = ϕ + leg_phase[i]

        # A simple stance/swing waveform
        s = sin(ϕi)
        c = cos(ϕi)

        # Hip pitch: drive swing forward/back; add bias for forward/back commands
        τ_hip_pitch =  Ahip * s + hip_bias_pitch

        # Knee: bend more during "swing" (use phase shift)
        τ_knee = Aknee * sin(ϕi + π/2)

        # Hip ab/ad: lateral + yaw coupling
        #  - lateral: push legs outward/inward depending on side
        #  - yaw: front vs rear get opposite sign to create turning moment
        τ_abd = Aabd * c * side_sign[i] + hip_bias_abd * side_sign[i] + yaw_bias_abd * front_sign[i] * side_sign[i]

        # Map into u: [abd, pitch, knee] for each leg block
        base = 3*(i-1)
        u[base + 1] = τ_abd
        u[base + 2] = τ_hip_pitch
        u[base + 3] = τ_knee
    end

    return u
end

# ----------------------------
# Main script parameters
# ----------------------------
struct GaitParams
    gait_hz::Float64

    Ahip_base::Float64
    Ahip_vel::Float64
    Ahip_lat::Float64

    Aknee_base::Float64
    Aknee_vel::Float64

    Aabd_base::Float64
    Aabd_lat::Float64
    Aabd_yaw::Float64

    hip_bias_pitch::Float64
    hip_bias_abd::Float64
    yaw_bias_abd::Float64
end

# ----------------------------
# User-config section
# ----------------------------
dt = 0.02                         # simulation timestep
cmd_duration = 1.0                # seconds per discrete command
sequence = [1, 1, 1, 5, 1, 3, 1]   # <--- EDIT ME (1=fwd, 2=bwd, 3=right, 4=left, 5=turnR, 6=turnL, 7=stop)

# Gait tuning (start conservative; increase if it barely moves)
params = GaitParams(
    2.0,     # gait_hz
    0.25,    # Ahip_base
    0.35,    # Ahip_vel
    0.20,    # Ahip_lat
    0.35,    # Aknee_base
    0.45,    # Aknee_vel
    0.12,    # Aabd_base
    0.18,    # Aabd_lat
    0.10,    # Aabd_yaw
    0.05,    # hip_bias_pitch
    0.03,    # hip_bias_abd
    0.03,    # yaw_bias_abd
)

# ----------------------------
# Build mechanism + initialize
# ----------------------------
mechanism, mech_name = get_quadruped_mechanism(; timestep=dt)

# Try to initialize with a standard initializer if one exists for that mechanism name.
# (Dojo's docs show initialize!(mechanism, :name; kwargs...) usage.)  [oai_citation:0‡dojo-sim.github.io](https://dojo-sim.github.io/Dojo.jl/dev/creating_simulation/define_controller.html)
try
    initialize!(mechanism, mech_name)
catch
    # Fallback: at least zero things out
    try
        zero_coordinates!(mechanism)
        zero_velocities!(mechanism)
    catch
        # If even that fails, we'll proceed (simulate! may still run).
    end
end

# ----------------------------
# Precompute full action trajectory
# ----------------------------
total_time = cmd_duration * length(sequence)
N = Int(round(total_time / dt))
nu = input_dimension(mechanism)   # number of inputs for the mechanism  [oai_citation:1‡dojo-sim.github.io](https://dojo-sim.github.io/Dojo.jl/stable/api.html)

actions = zeros(nu, N)            # <-- this is the variable you asked for

for k in 1:N
    t = (k-1) * dt
    seg = clamp(Int(floor(t / cmd_duration)) + 1, 1, length(sequence))
    intent = cmd_to_intent(sequence[seg])

    u12 = gait_u12(t, intent, params)

    # Fit to mechanism input dimension
    m = min(nu, 12)
    actions[1:m, k] .= u12[1:m]
    # remaining entries already 0
end

# ----------------------------
# Simulate using the precomputed actions
# ----------------------------
function controller!(mech, k)
    # Dojo controllers receive (mechanism, k).  [oai_citation:2‡dojo-sim.github.io](https://dojo-sim.github.io/Dojo.jl/dev/creating_simulation/define_controller.html)
    # Clamp k in case simulate! calls beyond our precomputed length.
    kk = clamp(k, 1, size(actions, 2))
    set_input!(mech, actions[:, kk])  # set input for each joint in mechanism  [oai_citation:3‡dojo-sim.github.io](https://dojo-sim.github.io/Dojo.jl/stable/api.html)
    return nothing
end

storage = simulate!(mechanism, total_time, controller!; record=true)  #  [oai_citation:4‡dojo-sim.github.io](https://dojo-sim.github.io/Dojo.jl/dev/creating_simulation/define_controller.html)

# ----------------------------
# Visualize / render
# ----------------------------
vis = visualize(mechanism, storage)  #  [oai_citation:5‡dojo-sim.github.io](https://dojo-sim.github.io/Dojo.jl/dev/creating_simulation/define_controller.html)
render(vis)                          #  [oai_citation:6‡dojo-sim.github.io](https://dojo-sim.github.io/Dojo.jl/dev/creating_simulation/define_controller.html)

# At this point:
#   - `actions` is populated (nu x N)
#   - `storage` contains the trajectory
#   - a viewer should open / render depending on your Dojo backend
println("Done. actions size = ", size(actions), ", mech_name = ", mech_name, ", nu = ", nu)