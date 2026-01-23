#!/usr/bin/env julia
# ==========================================================================================
# quadruped_param_policy_trajopt.jl
#
# Optimize a low-dimensional gait policy (sinusoid parameters) to track a reference
# trajectory, using differentiable dynamics linearization and sensitivity propagation.
# Warmstarts from the controller parameters in quadruped_controller.jl.
#
# Usage:
#   julia --project quadruped_param_policy_trajopt.jl <trajectory_file.h5> [--horizon N]
#
# Example:
#   julia --project quadruped_param_policy_trajopt.jl ./data/trajectory_00001.h5 --horizon 101
# ==========================================================================================

using JuMP, Ipopt, HSL_jll
using LinearAlgebra
using ForwardDiff
using HDF5

using Dojo
using DojoEnvironments

include("quadruped_controller.jl")

# ------------------------------------------------------------------------------------------
# Command line parsing
# ------------------------------------------------------------------------------------------
function parse_args()
    if length(ARGS) < 1
        println("Usage: julia quadruped_param_policy_trajopt.jl <trajectory_file.h5> [options]")
        println("Options:")
        println("  --horizon N     Number of knot points (default: auto from trajectory)")
        println("  --dt T          Time step (default: 0.02)")
        println("  --start_idx K   Start index in trajectory (default: 1)")
        exit(1)
    end

    traj_file = ARGS[1]
    horizon = nothing
    dt = 0.02
    start_idx = 1

    i = 2
    while i <= length(ARGS)
        if ARGS[i] == "--horizon"
            horizon = parse(Int, ARGS[i + 1]); i += 2
        elseif ARGS[i] == "--dt"
            dt = parse(Float64, ARGS[i + 1]); i += 2
        elseif ARGS[i] == "--start_idx"
            start_idx = parse(Int, ARGS[i + 1]); i += 2
        else
            println("Unknown argument: $(ARGS[i])")
            i += 1
        end
    end

    return traj_file, horizon, dt, start_idx
end

# ------------------------------------------------------------------------------------------
# Indices (consistent with warmstart script)
# ------------------------------------------------------------------------------------------
const BASE_X_IDX = 1
const BASE_Y_IDX = 2
const BASE_Z_IDX = 3
const BASE_QW_IDX = 4
const BASE_QX_IDX = 5
const BASE_QY_IDX = 6
const BASE_QZ_IDX = 7

const BASE_VX_IDX = 19
const BASE_VY_IDX = 20
const BASE_VZ_IDX = 21
const BASE_WX_IDX = 22
const BASE_WY_IDX = 23
const BASE_WZ_IDX = 24

const JOINT_POS_START = 8
const JOINT_VEL_START = 25

# ------------------------------------------------------------------------------------------
# Objective weights (match warmstart script)
# ------------------------------------------------------------------------------------------
const w_track_xy = 10.0
const w_track_z  = 20.0
const w_terminal = 50.0
const w_orient   = 15.0
const w_vel_xy   = 2.0
const w_vel_z    = 5.0
const w_ang_vel  = 3.0
const w_u        = 1e-3
const w_du       = 1e-2
const w_u_ref    = 1e-2

# ------------------------------------------------------------------------------------------
# Dojo solver options
# ------------------------------------------------------------------------------------------
const DOJO_OPTS_STEP = Dojo.SolverOptions{Float64}(;
    rtol = 1e-4, btol = 1e-3, max_iter = 20, max_ls = 8, verbose = false
)
const DOJO_OPTS_GRAD = Dojo.SolverOptions{Float64}(;
    rtol = 1e-3, btol = 1e-2, max_iter = 10, max_ls = 6, verbose = false
)

# ------------------------------------------------------------------------------------------
# Load trajectory and references
# ------------------------------------------------------------------------------------------
function load_reference(traj_file::String, target_dt::Float64, start_idx::Int, horizon::Union{Int, Nothing}, mech_input_dim::Int)
    println("Loading trajectory from: $traj_file")
    states = h5read(traj_file, "states")
    actions = h5read(traj_file, "actions")

    attrs = h5readattr(traj_file, "/")
    traj_dt = attrs["dt"]

    println("  Trajectory dt: $traj_dt, Target dt: $target_dt")
    println("  Original states shape: $(size(states))")
    println("  Original actions shape: $(size(actions))")

    subsample_rate = max(1, round(Int, target_dt / traj_dt))
    println("  Subsample rate: $subsample_rate")

    states_sub = states[:, start_idx:subsample_rate:end]
    actions_sub = actions[:, start_idx:subsample_rate:end]

    available_N = size(states_sub, 2)
    if isnothing(horizon)
        N = min(available_N, 101)
    else
        N = min(horizon, available_N)
    end

    println("  Using horizon N = $N")

    X_ref = [Vector{Float64}(states_sub[:, k]) for k in 1:N]

    function pad_action(u_small::Vector{Float64}, target_dim::Int)
        if length(u_small) == target_dim
            return u_small
        elseif length(u_small) < target_dim
            return vcat(zeros(target_dim - length(u_small)), u_small)
        else
            error("Action dimension $(length(u_small)) > mechanism input dimension $target_dim")
        end
    end

    U_ref = [pad_action(Vector{Float64}(actions_sub[:, min(k, size(actions_sub, 2))]), mech_input_dim) for k in 1:(N-1)]

    # Optional command sequence
    cmd_seq = Int[]
    try
        cmd_seq = Vector{Int}(h5read(traj_file, "command_sequence"))
        println("  Found command sequence (len=$(length(cmd_seq)))")
    catch
        println("  No command sequence found; defaulting to forward")
    end

    return N, X_ref, U_ref, cmd_seq
end

# ------------------------------------------------------------------------------------------
# Policy parameterization
# θ = [p_fwd(5); p_back(5); turn_inner; turn_outer; hip_bias]
# ------------------------------------------------------------------------------------------
const PARAM_DIM = 13

@inline function split_params(θ::AbstractVector)
    p_fwd = view(θ, 1:5)
    p_back = view(θ, 6:10)
    turn_inner = θ[11]
    turn_outer = θ[12]
    hip_bias = θ[13]
    return p_fwd, p_back, turn_inner, turn_outer, hip_bias
end

function cmd_to_intent_param(cmd::Int, turn_inner::T, turn_outer::T, hip_bias::T) where {T<:Real}
    oneT = one(T)
    zeroT = zero(T)
    if cmd == QC_CMD_FORWARD
        return (:fwd, oneT, oneT, zeroT, zeroT, true)
    elseif cmd == QC_CMD_BACKWARD
        return (:back, oneT, oneT, zeroT, zeroT, true)
    elseif cmd == QC_CMD_TURN_RIGHT
        return (:fwd, turn_outer, turn_inner, +hip_bias, -hip_bias, true)
    elseif cmd == QC_CMD_TURN_LEFT
        return (:fwd, turn_inner, turn_outer, -hip_bias, +hip_bias, true)
    else
        return (:fwd, oneT, oneT, zeroT, zeroT, false)
    end
end

function policy_u12!(u12::AbstractVector{T}, x::AbstractVector, k::Int, θ::AbstractVector{T}, cmd::Int) where {T<:Real}
    p_fwd, p_back, turn_inner, turn_outer, hip_bias = split_params(θ)
    gait, scaleL, scaleR, hipBL, hipBR, osc_on = cmd_to_intent_param(cmd, turn_inner, turn_outer, hip_bias)
    p = (gait == :back) ? p_back : p_fwd
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

    fill!(u12, zero(T))
    for leg in 1:4
        joint0 = (leg - 1) * 3
        pos0 = JOINT_POS_START + joint0
        vel0 = JOINT_VEL_START + joint0

        θ1 = x[pos0]; dθ1 = x[vel0]
        θ2 = x[pos0 + 1]; dθ2 = x[vel0 + 1]
        θ3 = x[pos0 + 2]; dθ3 = x[vel0 + 2]

        is_left = (leg in QC_LEFT_LEGS)
        hipB = is_left ? hipBL : hipBR

        if !osc_on
            θ2ref = Cth
            θ3ref = Ccf
        else
            useA = (leg == 1 || leg == 4)
            if useA
                θ2ref = is_left ? thigh_A_L : thigh_A_R
                θ3ref = is_left ? calf_A_L : calf_A_R
            else
                θ2ref = is_left ? thigh_B_L : thigh_B_R
                θ3ref = is_left ? calf_B_L : calf_B_R
            end
        end

        u12[joint0 + 1] = QC_Kp[1] * (hipB - θ1) + QC_Kd[1] * (zero(T) - dθ1)
        u12[joint0 + 2] = QC_Kp[2] * (θ2ref - θ2) + QC_Kd[2] * (zero(T) - dθ2)
        u12[joint0 + 3] = QC_Kp[3] * (θ3ref - θ3) + QC_Kd[3] * (zero(T) - dθ3)
    end

    return u12
end

function policy_u18!(u18::AbstractVector{T}, x::AbstractVector, k::Int, θ::AbstractVector{T}, cmd::Int) where {T<:Real}
    u12 = view(u18, 7:18)
    policy_u12!(u12, x, k, θ, cmd)
    fill!(view(u18, 1:6), zero(T))
    return u18
end

function du_dx_matrix(n::Int)
    du_dx = zeros(12, n)
    for leg in 1:4
        joint0 = (leg - 1) * 3
        pos0 = JOINT_POS_START + joint0
        vel0 = JOINT_VEL_START + joint0

        du_dx[joint0 + 1, pos0] = -QC_Kp[1]
        du_dx[joint0 + 1, vel0] = -QC_Kd[1]
        du_dx[joint0 + 2, pos0 + 1] = -QC_Kp[2]
        du_dx[joint0 + 2, vel0 + 1] = -QC_Kd[2]
        du_dx[joint0 + 3, pos0 + 2] = -QC_Kp[3]
        du_dx[joint0 + 3, vel0 + 2] = -QC_Kd[3]
    end
    return du_dx
end

# ------------------------------------------------------------------------------------------
# Evaluator (single shooting, policy parameters)
# ------------------------------------------------------------------------------------------
mutable struct PolicyEvaluator
    mech
    x0::Vector{Float64}
    N::Int
    n::Int
    m::Int
    p::Int

    X::Matrix{Float64}
    U::Matrix{Float64}
    A::Array{Float64,3}
    B::Array{Float64,3}

    x_ref::Vector{Float64}
    y_ref::Vector{Float64}
    z_ref::Vector{Float64}
    vx_ref::Vector{Float64}
    vy_ref::Vector{Float64}
    vz_ref::Vector{Float64}
    qw_ref::Vector{Float64}
    qx_ref::Vector{Float64}
    qy_ref::Vector{Float64}
    qz_ref::Vector{Float64}
    u_ref::Matrix{Float64}

    w_track::Vector{Float64}
    cmd_k::Vector{Int}

    du_dx12::Matrix{Float64}

    xk::Vector{Float64}
    u18::Vector{Float64}

    last_theta_cost::Vector{Float64}
    have_cost::Bool
    last_theta_grad::Vector{Float64}
    have_grad::Bool
end

function PolicyEvaluator(mech, x0::Vector{Float64}, N::Int, X_ref::Vector{Vector{Float64}}, U_ref::Vector{Vector{Float64}}, cmd_seq::Vector{Int}, dt::Float64)
    n = length(x0)
    m = Dojo.input_dimension(mech)

    x_ref = [X_ref[k][BASE_X_IDX] for k in 1:N]
    y_ref = [X_ref[k][BASE_Y_IDX] for k in 1:N]
    z_ref = [X_ref[k][BASE_Z_IDX] for k in 1:N]
    vx_ref = [X_ref[k][BASE_VX_IDX] for k in 1:N]
    vy_ref = [X_ref[k][BASE_VY_IDX] for k in 1:N]
    vz_ref = [X_ref[k][BASE_VZ_IDX] for k in 1:N]
    qw_ref = [X_ref[k][BASE_QW_IDX] for k in 1:N]
    qx_ref = [X_ref[k][BASE_QX_IDX] for k in 1:N]
    qy_ref = [X_ref[k][BASE_QY_IDX] for k in 1:N]
    qz_ref = [X_ref[k][BASE_QZ_IDX] for k in 1:N]

    u_ref = zeros(m, N-1)
    for k in 1:(N-1)
        u_ref[:, k] .= U_ref[k]
    end

    Kdrop = 3
    Kramp = 8
    w_track = zeros(Float64, N)
    for k in 1:N
        if k <= Kdrop
            w_track[k] = 0.0
        elseif k < Kdrop + Kramp
            w_track[k] = (k - Kdrop) / Kramp
        else
            w_track[k] = 1.0
        end
    end

    cmd_k = fill(QC_CMD_FORWARD, N-1)
    if !isempty(cmd_seq)
        N_per_cmd = Int(round(QC_cmd_duration / dt))
        for k in 1:(N-1)
            seg = clamp(Int(fld(k - 1, N_per_cmd)) + 1, 1, length(cmd_seq))
            cmd_k[k] = cmd_seq[seg]
        end
    end

    return PolicyEvaluator(
        deepcopy(mech), copy(x0), N, n, m, PARAM_DIM,
        zeros(n, N), zeros(m, N-1),
        zeros(n, n, N-1), zeros(n, m, N-1),
        x_ref, y_ref, z_ref, vx_ref, vy_ref, vz_ref,
        qw_ref, qx_ref, qy_ref, qz_ref,
        u_ref, w_track, cmd_k,
        du_dx_matrix(n),
        zeros(n), zeros(m),
        Float64[], false,
        Float64[], false
    )
end

function ensure_cost_rollout!(E::PolicyEvaluator, θ::Vector{Float64})
    if E.have_cost && E.last_theta_cost == θ
        return
    end
    E.last_theta_cost = copy(θ)
    E.have_cost = true
    E.have_grad = false

    n, m, N = E.n, E.m, E.N
    mech = E.mech

    @inbounds for i in 1:n
        E.X[i, 1] = E.x0[i]
    end

    for k in 1:(N-1)
        @inbounds for i in 1:n
            E.xk[i] = E.X[i, k]
        end
        policy_u18!(E.u18, E.xk, k, θ, E.cmd_k[k])
        E.U[:, k] .= E.u18

        Dojo.set_minimal_state!(mech, E.xk)
        Dojo.set_input!(mech, E.u18)
        Dojo.step_minimal_coordinates!(mech, E.xk, E.u18; opts=DOJO_OPTS_STEP)

        xn = Dojo.get_minimal_state(mech)
        @inbounds for i in 1:n
            E.X[i, k+1] = xn[i]
        end
    end
end

function ensure_grad_data!(E::PolicyEvaluator, θ::Vector{Float64})
    ensure_cost_rollout!(E, θ)
    if E.have_grad && E.last_theta_grad == θ
        return
    end
    E.last_theta_grad = copy(θ)
    E.have_grad = true

    n, m, N = E.n, E.m, E.N
    mech = E.mech

    @inbounds for k in 1:(N-1)
        @inbounds for i in 1:n
            E.xk[i] = E.X[i, k]
        end
        policy_u18!(E.u18, E.xk, k, θ, E.cmd_k[k])
        E.U[:, k] .= E.u18

        Dojo.set_minimal_state!(mech, E.xk)
        Dojo.set_input!(mech, E.u18)

        A2, B2 = Dojo.get_minimal_gradients!(mech, E.xk, E.u18; opts=DOJO_OPTS_GRAD)
        @inbounds for i in 1:n, j in 1:n
            E.A[i, j, k] = A2[i, j]
        end
        @inbounds for i in 1:n, j in 1:m
            E.B[i, j, k] = B2[i, j]
        end

        xn = Dojo.get_minimal_state(mech)
        @inbounds for i in 1:n
            E.X[i, k+1] = xn[i]
        end
    end
end

function cost_from_rollout(E::PolicyEvaluator)
    N = E.N
    cost = 0.0

    for k in 1:(N-1)
        xk = view(E.X, :, k)
        uk = view(E.U, :, k)
        wtk = E.w_track[k]

        dx = xk[BASE_X_IDX] - E.x_ref[k]
        dy = xk[BASE_Y_IDX] - E.y_ref[k]
        dz = xk[BASE_Z_IDX] - E.z_ref[k]

        cost += wtk * w_track_xy * (dx^2 + dy^2)
        cost += wtk * w_track_z * (dz^2)

        dq = (xk[BASE_QX_IDX] - E.qx_ref[k])^2 +
             (xk[BASE_QY_IDX] - E.qy_ref[k])^2 +
             (xk[BASE_QZ_IDX] - E.qz_ref[k])^2 +
             (xk[BASE_QW_IDX] - E.qw_ref[k])^2
        cost += wtk * w_orient * dq

        dvx = xk[BASE_VX_IDX] - E.vx_ref[k]
        dvy = xk[BASE_VY_IDX] - E.vy_ref[k]
        dvz = xk[BASE_VZ_IDX] - E.vz_ref[k]
        cost += wtk * (w_vel_xy * (dvx^2 + dvy^2) + w_vel_z * dvz^2)

        cost += wtk * w_ang_vel * (xk[BASE_WX_IDX]^2 + xk[BASE_WY_IDX]^2 + xk[BASE_WZ_IDX]^2)

        cost += w_u * sum(uk .^ 2)
        cost += w_u_ref * sum((uk .- view(E.u_ref, :, k)).^2)
        if k >= 2
            cost += w_du * sum((uk .- view(E.U, :, k-1)).^2)
        end
    end

    xN = view(E.X, :, N)
    dxN = xN[BASE_X_IDX] - E.x_ref[N]
    dyN = xN[BASE_Y_IDX] - E.y_ref[N]
    dzN = xN[BASE_Z_IDX] - E.z_ref[N]

    dqn = (xN[BASE_QX_IDX] - E.qx_ref[N])^2 +
          (xN[BASE_QY_IDX] - E.qy_ref[N])^2 +
          (xN[BASE_QZ_IDX] - E.qz_ref[N])^2 +
          (xN[BASE_QW_IDX] - E.qw_ref[N])^2

    cost += E.w_track[N] * w_terminal * (dxN^2 + dyN^2 + dzN^2 + 0.5 * dqn)
    return cost
end

function grad_from_rollout!(E::PolicyEvaluator, g_out::AbstractVector{T}, θ::AbstractVector{T}) where {T<:Real}
    fill!(g_out, zero(T))

    n, m, N, p = E.n, E.m, E.N, E.p

    S = zeros(n, p)
    S_hist = [zeros(n, p) for _ in 1:N]
    dU_dθ = [zeros(m, p) for _ in 1:(N-1)]

    du_dx12 = E.du_dx12
    du_dx18 = zeros(m, n)
    du_dx18[7:18, :] .= du_dx12

    for k in 1:(N-1)
        S_hist[k] .= S
        xk = view(E.X, :, k)

        # du/dθ via ForwardDiff on policy
        u_func = θloc -> begin
            u12 = zeros(eltype(θloc), 12)
            policy_u12!(u12, xk, k, θloc, E.cmd_k[k])
            return copy(u12)
        end
        du12_dθ = ForwardDiff.jacobian(u_func, θ)
        du18_dθ = zeros(m, p)
        du18_dθ[7:18, :] .= du12_dθ
        dU_dθ[k] .= du18_dθ

        # Propagate sensitivity: S_{k+1} = A*S + B*(du_dθ + du_dx*S)
        tmp = du18_dθ + du_dx18 * S
        S = E.A[:, :, k] * S + E.B[:, :, k] * tmp
    end
    S_hist[N] .= S

    for k in 1:(N-1)
        xk = view(E.X, :, k)
        uk = view(E.U, :, k)
        wtk = E.w_track[k]

        dldx = zeros(T, n)
        dldu = zeros(T, m)

        dldx[BASE_X_IDX] += 2 * wtk * w_track_xy * (xk[BASE_X_IDX] - E.x_ref[k])
        dldx[BASE_Y_IDX] += 2 * wtk * w_track_xy * (xk[BASE_Y_IDX] - E.y_ref[k])
        dldx[BASE_Z_IDX] += 2 * wtk * w_track_z * (xk[BASE_Z_IDX] - E.z_ref[k])

        dldx[BASE_QX_IDX] += 2 * wtk * w_orient * (xk[BASE_QX_IDX] - E.qx_ref[k])
        dldx[BASE_QY_IDX] += 2 * wtk * w_orient * (xk[BASE_QY_IDX] - E.qy_ref[k])
        dldx[BASE_QZ_IDX] += 2 * wtk * w_orient * (xk[BASE_QZ_IDX] - E.qz_ref[k])
        dldx[BASE_QW_IDX] += 2 * wtk * w_orient * (xk[BASE_QW_IDX] - E.qw_ref[k])

        dldx[BASE_VX_IDX] += 2 * wtk * w_vel_xy * (xk[BASE_VX_IDX] - E.vx_ref[k])
        dldx[BASE_VY_IDX] += 2 * wtk * w_vel_xy * (xk[BASE_VY_IDX] - E.vy_ref[k])
        dldx[BASE_VZ_IDX] += 2 * wtk * w_vel_z * (xk[BASE_VZ_IDX] - E.vz_ref[k])

        dldx[BASE_WX_IDX] += 2 * wtk * w_ang_vel * xk[BASE_WX_IDX]
        dldx[BASE_WY_IDX] += 2 * wtk * w_ang_vel * xk[BASE_WY_IDX]
        dldx[BASE_WZ_IDX] += 2 * wtk * w_ang_vel * xk[BASE_WZ_IDX]

        dldu .+= 2 * w_u .* uk
        dldu .+= 2 * w_u_ref .* (uk .- view(E.u_ref, :, k))
        if k >= 2
            dldu .+= 2 * w_du .* (uk .- view(E.U, :, k-1))
        end
        if k < (N-1)
            dldu .+= 2 * w_du .* (uk .- view(E.U, :, k+1))
        end

        g_out .+= (S_hist[k]' * dldx)
        g_out .+= (dU_dθ[k]' * dldu)
    end

    xN = view(E.X, :, N)
    dldxN = zeros(T, n)
    dldxN[BASE_X_IDX] = 2 * E.w_track[N] * w_terminal * (xN[BASE_X_IDX] - E.x_ref[N])
    dldxN[BASE_Y_IDX] = 2 * E.w_track[N] * w_terminal * (xN[BASE_Y_IDX] - E.y_ref[N])
    dldxN[BASE_Z_IDX] = 2 * E.w_track[N] * w_terminal * (xN[BASE_Z_IDX] - E.z_ref[N])

    dldxN[BASE_QX_IDX] = E.w_track[N] * w_terminal * (xN[BASE_QX_IDX] - E.qx_ref[N])
    dldxN[BASE_QY_IDX] = E.w_track[N] * w_terminal * (xN[BASE_QY_IDX] - E.qy_ref[N])
    dldxN[BASE_QZ_IDX] = E.w_track[N] * w_terminal * (xN[BASE_QZ_IDX] - E.qz_ref[N])
    dldxN[BASE_QW_IDX] = E.w_track[N] * w_terminal * (xN[BASE_QW_IDX] - E.qw_ref[N])

    g_out .+= (S_hist[N]' * dldxN)
    return
end

# ------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------
traj_file, horizon_arg, dt, start_idx = parse_args()

mech = DojoEnvironments.get_mechanism(:quadruped)

N, X_ref, U_ref, cmd_seq = load_reference(traj_file, dt, start_idx, horizon_arg, Dojo.input_dimension(mech))

# Match the reference initial state
x0 = copy(X_ref[1])
Dojo.set_minimal_state!(mech, x0)

EVAL = PolicyEvaluator(mech, x0, N, X_ref, U_ref, cmd_seq, dt)

function param_cost(args...)
    θ = collect(args)
    ensure_cost_rollout!(EVAL, θ)
    return cost_from_rollout(EVAL)
end

function param_cost_grad(g_out::AbstractVector{Float64}, args...)
    θ = collect(args)
    ensure_grad_data!(EVAL, θ)
    grad_from_rollout!(EVAL, g_out, θ)
    return
end

model = Model(optimizer_with_attributes(Ipopt.Optimizer,
    "linear_solver" => "ma97",
    "hessian_approximation" => "limited-memory",
    "print_level" => 5,
    "mu_strategy" => "adaptive",
))

@variable(model, θ[1:PARAM_DIM])

# Parameter bounds (conservative)
set_lower_bound(θ[1], 0.01); set_upper_bound(θ[1], 2.0)
set_lower_bound(θ[2], 0.0);  set_upper_bound(θ[2], 2.0)
set_lower_bound(θ[3], -2.0); set_upper_bound(θ[3], 2.0)
set_lower_bound(θ[4], 0.0);  set_upper_bound(θ[4], 2.0)
set_lower_bound(θ[5], -2.5); set_upper_bound(θ[5], 0.0)

set_lower_bound(θ[6], 0.01); set_upper_bound(θ[6], 2.0)
set_lower_bound(θ[7], 0.0);  set_upper_bound(θ[7], 2.0)
set_lower_bound(θ[8], -2.0); set_upper_bound(θ[8], 2.0)
set_lower_bound(θ[9], 0.0);  set_upper_bound(θ[9], 2.0)
set_lower_bound(θ[10], -2.5); set_upper_bound(θ[10], 0.0)

set_lower_bound(θ[11], 0.5); set_upper_bound(θ[11], 1.5)
set_lower_bound(θ[12], 0.5); set_upper_bound(θ[12], 1.5)
set_lower_bound(θ[13], -0.3); set_upper_bound(θ[13], 0.3)

JuMP.register(model, :param_cost, PARAM_DIM, param_cost, param_cost_grad)
@NLobjective(model, Min, param_cost(θ...))

# Warmstart from controller params
θ0 = vcat(QC_p_fwd, QC_p_back, QC_TURN_SCALE_INNER, QC_TURN_SCALE_OUTER, QC_HIP_STEER_BIAS)
for i in 1:PARAM_DIM
    set_start_value(θ[i], θ0[i])
end

println("\nParam policy optimization")
println("N=$N, dt=$dt, params=$(PARAM_DIM)")
optimize!(model)

println("\ntermination_status = ", termination_status(model))
println("objective_value    = ", try objective_value(model) catch; NaN end)

θ_opt = value.(θ)
println("\nOptimized params:")
println("  p_fwd  = ", θ_opt[1:5])
println("  p_back = ", θ_opt[6:10])
println("  turn_inner, turn_outer, hip_bias = ", θ_opt[11:13])

# Playback with optimized parameters
println("\nSimulating optimized policy...")
Dojo.set_minimal_state!(mech, x0)

function controller_policy!(mechanism, k)
    x = Dojo.get_minimal_state(mechanism)
    u = zeros(Dojo.input_dimension(mechanism))
    cmd = (k <= length(EVAL.cmd_k)) ? EVAL.cmd_k[k] : QC_CMD_FORWARD
    policy_u18!(u, x, k, θ_opt, cmd)
    Dojo.set_input!(mechanism, u)
    return
end

Tsim = dt * (N - 1)
storage = Dojo.simulate!(mech, Tsim, controller_policy!; record=true, opts=DOJO_OPTS_STEP)
vis = Dojo.visualize(mech, storage)
Dojo.render(vis)
