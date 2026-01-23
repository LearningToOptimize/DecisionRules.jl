#!/usr/bin/env julia
# ==========================================================================================
# quadruped_trajopt_warmstart.jl
#
# Direct transcription trajectory optimization for Dojo quadruped with warmstart
# from a pre-recorded training trajectory. This script loads a trajectory from the 
# training dataset and uses it to warmstart the optimization.
#
# KEY IMPROVEMENTS OVER quadruped_simple_simulation.jl:
# ------------------------------------------------------------------------------------------
# 1. WARMSTART: Uses actual trajectory from dataset instead of zero-torque rollout
# 
# 2. IMPROVED OBJECTIVE FUNCTION:
#    - Orientation tracking: Penalizes deviation from upright quaternion (balancing)
#    - Velocity tracking: Encourages smooth, consistent forward velocity
#    - Angular velocity penalty: Discourages excessive body rotation rates
#    - Foot clearance proxy: Penalizes low body height (encourages proper stepping)
#    - Smoother control regularization: Uses L2 + L1-like (Huber) control penalties
#
# 3. REFERENCE PATH: Extracted directly from the training trajectory
#
# Usage:
#   julia --project quadruped_trajopt_warmstart.jl <trajectory_file.h5> [--horizon N]
#
# Example:
#   julia --project quadruped_trajopt_warmstart.jl ./data/trajectory_00001.h5 --horizon 51
#
# ==========================================================================================

using JuMP, Ipopt, HSL_jll
import MathOptInterface
const MOI = MathOptInterface

using LinearAlgebra
using SparseArrays
using ForwardDiff
using HDF5

using Dojo
using DojoEnvironments

# ------------------------------------------------------------------------------------------
# Command line parsing
# ------------------------------------------------------------------------------------------
function parse_warmstart_args()
    if length(ARGS) < 1
        println("Usage: julia quadruped_trajopt_warmstart.jl <trajectory_file.h5> [options]")
        println("Options:")
        println("  --horizon N     Number of knot points (default: auto from trajectory)")
        println("  --dt T          Time step (default: 0.02)")
        println("  --start_idx K   Start index in trajectory (default: 1)")
        println("  --stages S      Number of staged solves (default: 2)")
        println("  --lambda_track_init L   Initial tracking multiplier (default: 10.0)")
        println("  --lambda_u_ref_init L   Initial control-ref multiplier (default: 10.0)")
        println("  --trust_pos_eps E       Trust-region epsilon for base position (default: 0.0=off)")
        println("  --trust_orient_eps E    Trust-region epsilon for base orientation (default: 0.0=off)")
        println("  --trust_vel_eps E       Trust-region epsilon for base velocities (default: 0.0=off)")
        exit(1)
    end
    
    traj_file = ARGS[1]
    horizon = nothing  # auto-detect from trajectory
    dt = 0.02
    start_idx = 1
    stages = 2
    lambda_track_init = 10.0
    lambda_u_ref_init = 10.0
    trust_pos_eps = 0.0
    trust_orient_eps = 0.0
    trust_vel_eps = 0.0
    
    i = 2
    while i <= length(ARGS)
        if ARGS[i] == "--horizon"
            horizon = parse(Int, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--dt"
            dt = parse(Float64, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--start_idx"
            start_idx = parse(Int, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--stages"
            stages = parse(Int, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--lambda_track_init"
            lambda_track_init = parse(Float64, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--lambda_u_ref_init"
            lambda_u_ref_init = parse(Float64, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--trust_pos_eps"
            trust_pos_eps = parse(Float64, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--trust_orient_eps"
            trust_orient_eps = parse(Float64, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--trust_vel_eps"
            trust_vel_eps = parse(Float64, ARGS[i+1])
            i += 2
        else
            println("Unknown argument: $(ARGS[i])")
            i += 1
        end
    end
    
    return traj_file, horizon, dt, start_idx, stages, lambda_track_init, lambda_u_ref_init, trust_pos_eps, trust_orient_eps, trust_vel_eps
end

# ------------------------------------------------------------------------------------------
# Knobs (most can be overridden by args)
# ------------------------------------------------------------------------------------------
# State indices for the quadruped (36D minimal state):
# Positions: [x, y, z, qw, qx, qy, qz, joints...] (first 18)
# Velocities: [vx, vy, vz, wx, wy, wz, joint_vels...] (next 18)
const BASE_X_IDX = 1
const BASE_Y_IDX = 2
const BASE_Z_IDX = 3
const BASE_QW_IDX = 4   # quaternion w (scalar)
const BASE_QX_IDX = 5   # quaternion x
const BASE_QY_IDX = 6   # quaternion y
const BASE_QZ_IDX = 7   # quaternion z

const BASE_VX_IDX = 19  # linear velocity x
const BASE_VY_IDX = 20  # linear velocity y
const BASE_VZ_IDX = 21  # linear velocity z
const BASE_WX_IDX = 22  # angular velocity x
const BASE_WY_IDX = 23  # angular velocity y
const BASE_WZ_IDX = 24  # angular velocity z

const JOINT_POS_START = 8   # joint positions start at index 8
const JOINT_VEL_START = 25  # joint velocities start at index 25

const u_max_abs = 40.0

# ------------------------------------------------------------------------------------------
# OBJECTIVE FUNCTION WEIGHTS (tunable)
# ------------------------------------------------------------------------------------------
# NOTE: Weights are scaled down to help Ipopt convergence. The objective value ~1e4 was
# causing poor conditioning. These weights are relative to each other.

# Path tracking (lower weights initially - let dynamics drive the motion)
const w_track_xy = 10.0        # XY position tracking
const w_track_z  = 20.0        # Height tracking (important for balance)
const w_terminal = 50.0        # Terminal cost multiplier

# Orientation tracking (keep robot upright)
const w_orient = 15.0          # Penalize deviation from upright orientation

# Velocity tracking (smooth locomotion) - keep low, let dynamics guide
const w_vel_xy = 2.0           # Track desired XY velocity  
const w_vel_z = 5.0            # Penalize vertical velocity (bouncing)

# Angular velocity penalty (stability)
const w_ang_vel = 3.0          # Penalize excessive body rotation rates

# Joint regularization (prefer reference joint configuration)
const w_joint_pos = 0.5        # Track reference joint positions
const w_joint_vel = 0.2        # Penalize joint velocities

# Control costs (important for smoothness)
const w_u = 1e-3               # Control effort
const w_du = 1e-2              # Control smoothness
const w_u_ref = 1e-2           # Track reference control - important for staying near feasible

# Dojo solver options
const DOJO_OPTS = Dojo.SolverOptions{Float64}(;
    rtol = 1e-4,
    btol = 1e-3,
    max_iter = 20,
    max_ls = 8,
    verbose = false
)

# ------------------------------------------------------------------------------------------
# Dojo wrappers (same as original)
# ------------------------------------------------------------------------------------------
function dojo_step!(xnext::AbstractVector, mech, x::AbstractVector, u::AbstractVector; opts=DOJO_OPTS)
    Dojo.set_minimal_state!(mech, x)
    Dojo.set_input!(mech, u)
    Dojo.step_minimal_coordinates!(mech, x, u; opts=opts)
    xn = Dojo.get_minimal_state(mech)
    copyto!(xnext, xn)
    return nothing
end

function dojo_linearize!(A::AbstractMatrix, B::AbstractMatrix, mech, x::AbstractVector, u::AbstractVector; opts=DOJO_OPTS)
    Dojo.set_minimal_state!(mech, x)
    Dojo.set_input!(mech, u)
    A2, B2 = Dojo.get_minimal_gradients!(mech, x, u; opts=opts)
    copyto!(A, A2)
    copyto!(B, B2)
    return nothing
end

# ------------------------------------------------------------------------------------------
# Mechanism + initial state helpers
# ------------------------------------------------------------------------------------------
function build_quadruped()
    mech = DojoEnvironments.get_mechanism(:quadruped)
    x0 = Vector(Dojo.get_minimal_state(mech))
    m  = Dojo.input_dimension(mech)
    u0 = zeros(m)
    return mech, x0, u0
end

# ------------------------------------------------------------------------------------------
# Load and process trajectory from HDF5 file
# ------------------------------------------------------------------------------------------
function load_trajectory_for_warmstart(traj_file::String, target_dt::Float64, start_idx::Int, horizon::Union{Int, Nothing}, mech_input_dim::Int)
    println("Loading trajectory from: $traj_file")
    
    states = h5read(traj_file, "states")
    actions = h5read(traj_file, "actions")
    
    attrs = h5readattr(traj_file, "/")
    traj_dt = attrs["dt"]
    
    dataset_action_dim = size(actions, 1)
    println("  Trajectory dt: $traj_dt, Target dt: $target_dt")
    println("  Original states shape: $(size(states))")
    println("  Original actions shape: $(size(actions))")
    println("  Dataset action dim: $dataset_action_dim, Mechanism input dim: $mech_input_dim")
    
    # Subsample if needed to match target_dt
    subsample_rate = max(1, round(Int, target_dt / traj_dt))
    println("  Subsample rate: $subsample_rate")
    
    # Apply subsampling
    states_sub = states[:, start_idx:subsample_rate:end]
    actions_sub = actions[:, start_idx:subsample_rate:end]
    
    println("  Subsampled states shape: $(size(states_sub))")
    
    # Determine horizon
    available_N = size(states_sub, 2)
    if isnothing(horizon)
        N = min(available_N, 101)  # Default max horizon
    else
        N = min(horizon, available_N)
    end
    
    println("  Using horizon N = $N")
    
    # Extract X and U for warmstart
    # Note: Dataset has 12D actions (joint torques only), but Dojo's minimal coordinates
    # API expects 18D inputs (6 for floating base + 12 joints). The first 6 are ignored
    # (floating base is unactuated), so we pad with zeros.
    X_ws = [Vector{Float64}(states_sub[:, k]) for k in 1:N]
    
    # Pad actions if needed (12D -> 18D)
    function pad_action(u_small::Vector{Float64}, target_dim::Int)
        if length(u_small) == target_dim
            return u_small
        elseif length(u_small) < target_dim
            # Pad with zeros at the front (floating base DOFs are unactuated)
            return vcat(zeros(target_dim - length(u_small)), u_small)
        else
            error("Action dimension $(length(u_small)) > mechanism input dimension $target_dim")
        end
    end
    
    U_ws = [pad_action(Vector{Float64}(actions_sub[:, min(k, size(actions_sub, 2))]), mech_input_dim) for k in 1:(N-1)]
    
    # Extract reference path (positions)
    x_path = [states_sub[BASE_X_IDX, k] for k in 1:N]
    y_path = [states_sub[BASE_Y_IDX, k] for k in 1:N]
    z_path = [states_sub[BASE_Z_IDX, k] for k in 1:N]
    
    # Extract reference velocities
    vx_path = [states_sub[BASE_VX_IDX, k] for k in 1:N]
    vy_path = [states_sub[BASE_VY_IDX, k] for k in 1:N]
    vz_path = [states_sub[BASE_VZ_IDX, k] for k in 1:N]
    
    # Extract reference orientations (quaternion)
    qw_path = [states_sub[BASE_QW_IDX, k] for k in 1:N]
    qx_path = [states_sub[BASE_QX_IDX, k] for k in 1:N]
    qy_path = [states_sub[BASE_QY_IDX, k] for k in 1:N]
    qz_path = [states_sub[BASE_QZ_IDX, k] for k in 1:N]
    
    return N, X_ws, U_ws, x_path, y_path, z_path, vx_path, vy_path, vz_path, qw_path, qx_path, qy_path, qz_path
end

# ------------------------------------------------------------------------------------------
# Indexing for z = [vec(X); vec(U)]
# ------------------------------------------------------------------------------------------
idx_x(n::Int, k::Int, i::Int) = (k - 1) * n + i
idx_u(n::Int, N::Int, m::Int, k::Int, j::Int) = n * N + (k - 1) * m + j

# ------------------------------------------------------------------------------------------
# Sparsity structures (same as original)
# ------------------------------------------------------------------------------------------
function build_jacobian_structure(n::Int, m::Int, N::Int)
    S = Vector{Tuple{Int,Int}}()
    sizehint!(S, (N-1) * (n*n + n*m + n))
    for k in 1:(N-1)
        row0 = (k - 1) * n
        for i in 1:n, j in 1:n
            push!(S, (row0 + i, idx_x(n, k, j)))
        end
        for i in 1:n, j in 1:m
            push!(S, (row0 + i, idx_u(n, N, m, k, j)))
        end
        for i in 1:n
            push!(S, (row0 + i, idx_x(n, k + 1, i)))
        end
    end
    return S
end

function build_hessian_structure(n::Int, m::Int, N::Int)
    S = Vector{Tuple{Int,Int}}()
    block = n + m
    sizehint!(S, (N-1) * (block * (block + 1) ÷ 2))
    for k in 1:(N-1)
        x_inds = (idx_x(n, k, 1)):(idx_x(n, k, n))
        u_inds = (idx_u(n, N, m, k, 1)):(idx_u(n, N, m, k, m))
        inds = vcat(collect(x_inds), collect(u_inds))
        for a in 1:length(inds)
            ia = inds[a]
            for b in 1:a
                ib = inds[b]
                if ia >= ib
                    push!(S, (ia, ib))
                else
                    push!(S, (ib, ia))
                end
            end
        end
    end
    return S
end

# ------------------------------------------------------------------------------------------
# Per-step cache
# ------------------------------------------------------------------------------------------
mutable struct StepCache
    x::Vector{Float64}
    u::Vector{Float64}
    fx::Vector{Float64}
    A::Matrix{Float64}
    B::Matrix{Float64}
    have_fx::Bool
    have_AB::Bool
end

function StepCache(n::Int, m::Int)
    StepCache(zeros(n), zeros(m), zeros(n), zeros(n,n), zeros(n,m), false, false)
end

# ------------------------------------------------------------------------------------------
# Vector oracle object (same as original)
# ------------------------------------------------------------------------------------------
mutable struct QuadrupedDynamicsOracle
    mechs::Vector
    caches::Vector{StepCache}
    N::Int
    dt::Float64
    n::Int
    m::Int
    xk::Vector{Float64}
    uk::Vector{Float64}
    xnext::Vector{Float64}
    A::Matrix{Float64}
    B::Matrix{Float64}
    xu::Vector{Float64}
    H::Matrix{Float64}
    jac_struct::Vector{Tuple{Int,Int}}
    hess_struct::Vector{Tuple{Int,Int}}
end

function QuadrupedDynamicsOracle(mech, x0::Vector{Float64}, u0::Vector{Float64}; N::Int, dt::Float64)
    n = length(x0); m = length(u0)
    mechs  = [deepcopy(mech) for _ in 1:(N-1)]
    caches = [StepCache(n, m) for _ in 1:(N-1)]
    xk = zeros(n); uk = zeros(m); xnext = zeros(n)
    A  = zeros(n,n); B = zeros(n,m)
    xu = zeros(n+m); H = zeros(n+m, n+m)
    jac_struct  = build_jacobian_structure(n, m, N)
    hess_struct = build_hessian_structure(n, m, N)
    return QuadrupedDynamicsOracle(mechs, caches, N, dt, n, m, xk, uk, xnext, A, B, xu, H, jac_struct, hess_struct)
end

function ensure_fx_and_AB!(O::QuadrupedDynamicsOracle, k::Int, x::Vector{Float64}, u::Vector{Float64})
    c = O.caches[k]
    if (c.have_fx && c.have_AB && x == c.x && u == c.u)
        return
    end
    copyto!(c.x, x); copyto!(c.u, u)

    mech = O.mechs[k]
    Dojo.set_minimal_state!(mech, c.x)
    Dojo.set_input!(mech, c.u)

    # Get linearization (A, B) around this step and compute next-state fx consistently
    A2, B2 = Dojo.get_minimal_gradients!(mech, c.x, c.u; opts=DOJO_OPTS)
    copyto!(c.A, A2); copyto!(c.B, B2)

    # Compute next state using the same mapping
    dojo_step!(c.fx, mech, c.x, c.u; opts=DOJO_OPTS)

    c.have_AB = true
    c.have_fx = true
end

function ensure_fx!(O::QuadrupedDynamicsOracle, k::Int, x::Vector{Float64}, u::Vector{Float64})
    c = O.caches[k]
    if c.have_fx && (x == c.x) && (u == c.u)
        return
    end
    copyto!(c.x, x); copyto!(c.u, u)
    dojo_step!(c.fx, O.mechs[k], c.x, c.u; opts=DOJO_OPTS)
    c.have_fx = true
    return
end

function eval_f!(O::QuadrupedDynamicsOracle, ret::AbstractVector, z::AbstractVector)
    n, m, N = O.n, O.m, O.N
    @inbounds for k in 1:(N-1)
        x = O.xk; u = O.uk
        for i in 1:n; x[i] = z[idx_x(n,k,i)]; end
        for j in 1:m; u[j] = z[idx_u(n,N,m,k,j)]; end

        ensure_fx!(O, k, x, u)
        row0 = (k-1)*n
        c = O.caches[k]
        for i in 1:n
            ret[row0 + i] = z[idx_x(n,k+1,i)] - c.fx[i]
        end
    end
    return nothing
end

function eval_jacobian!(O::QuadrupedDynamicsOracle, ret::AbstractVector, z::AbstractVector)
    n, m, N = O.n, O.m, O.N
    idx = 0
    @inbounds for k in 1:(N-1)
        x = O.xk; u = O.uk
        for i in 1:n; x[i] = z[idx_x(n,k,i)]; end
        for j in 1:m; u[j] = z[idx_u(n,N,m,k,j)]; end

        ensure_fx_and_AB!(O, k, x, u)
        c = O.caches[k]

        for i in 1:n, j in 1:n
            idx += 1
            ret[idx] = -c.A[i,j]
        end
        for i in 1:n, j in 1:m
            idx += 1
            ret[idx] = -c.B[i,j]
        end
        for i in 1:n
            idx += 1
            ret[idx] = 1.0
        end
    end
    return nothing
end

function eval_hessian_lagrangian!(O::QuadrupedDynamicsOracle, ret::AbstractVector, z::AbstractVector, μ::AbstractVector)
    n, m, N = O.n, O.m, O.N
    idx = 0
    @inbounds for k in 1:(N-1)
        μk = view(μ, (k-1)*n + 1 : k*n)

        for i in 1:n; O.xu[i] = z[idx_x(n,k,i)]; end
        for j in 1:m; O.xu[n+j] = z[idx_u(n,N,m,k,j)]; end

        mech = O.mechs[k]
        φ = function(xu_vec)
            xk = view(xu_vec, 1:n)
            uk = view(xu_vec, n+1:n+m)
            tmp = similar(xk)
            dojo_step!(tmp, mech, Vector(xk), Vector(uk); opts=DOJO_OPTS)
            return -dot(μk, tmp)
        end

        ForwardDiff.hessian!(O.H, φ, O.xu)

        for a in 1:(n+m)
            for b in 1:a
                idx += 1
                ret[idx] = O.H[a,b]
            end
        end
    end
    return nothing
end

# ==========================================================================================
# Re-rollout warmstart to ensure dynamic feasibility
# ==========================================================================================
"""
Re-simulate the trajectory using the warmstart controls to get dynamically consistent states.
This is crucial because the subsampled trajectory from the dataset may not satisfy the
discrete dynamics at the optimization timestep.
"""
function rerollout_warmstart!(X_ws::Vector{Vector{Float64}}, U_ws::Vector{Vector{Float64}}, mech)
    N = length(X_ws)
    x0 = X_ws[1]
    
    # Create a fresh mechanism copy for rollout
    tmp_mech = deepcopy(mech)
    Dojo.set_minimal_state!(tmp_mech, x0)
    
    xnext = similar(x0)
    for k in 1:(N-1)
        # Simulate one step with the warmstart control
        dojo_step!(xnext, tmp_mech, X_ws[k], U_ws[k]; opts=DOJO_OPTS)
        # Update the next state in warmstart
        X_ws[k+1] .= xnext
    end
    
    return nothing
end

# ==========================================================================================
# MAIN SCRIPT
# ==========================================================================================

traj_file, horizon_arg, dt, start_idx, stages, lambda_track_init, lambda_u_ref_init, trust_pos_eps, trust_orient_eps, trust_vel_eps = parse_warmstart_args()

# Build mechanism
mech, x0_default, u0 = build_quadruped()
n, m = length(x0_default), length(u0)

# Load trajectory and extract warmstart + reference
N, X_ws, U_ws, x_path, y_path, z_path, vx_path, vy_path, vz_path, qw_path, qx_path, qy_path, qz_path = 
    load_trajectory_for_warmstart(traj_file, dt, start_idx, horizon_arg, m)

# Use initial state from trajectory
x0 = X_ws[1]

# CRITICAL: Re-rollout the warmstart to make it dynamically feasible
# The subsampled trajectory from the dataset does not satisfy x_{k+1} = f(x_k, u_k)
# at the optimization timestep, causing huge initial constraint violations.
println("Re-rolling out warmstart for dynamic feasibility...")
rerollout_warmstart!(X_ws, U_ws, mech)

# IMPORTANT: Update reference paths to match the re-rolled trajectory!
# Otherwise the objective tries to track the old (infeasible) path.
println("Updating reference paths to match re-rolled trajectory...")
for k in 1:N
    x_path[k] = X_ws[k][BASE_X_IDX]
    y_path[k] = X_ws[k][BASE_Y_IDX]
    z_path[k] = X_ws[k][BASE_Z_IDX]
    qw_path[k] = X_ws[k][BASE_QW_IDX]
    qx_path[k] = X_ws[k][BASE_QX_IDX]
    qy_path[k] = X_ws[k][BASE_QY_IDX]
    qz_path[k] = X_ws[k][BASE_QZ_IDX]
    vx_path[k] = X_ws[k][BASE_VX_IDX]
    vy_path[k] = X_ws[k][BASE_VY_IDX]
    vz_path[k] = X_ws[k][BASE_VZ_IDX]
end

println("\n" * "="^60)
println("Trajectory Optimization Setup")
println("="^60)
println("Dojo dims: n=$n, m=$m")
println("Horizon: N=$N, dt=$dt, total_time=$(dt*(N-1))s")
println("Initial position: ($(round(x0[BASE_X_IDX], digits=3)), $(round(x0[BASE_Y_IDX], digits=3)), $(round(x0[BASE_Z_IDX], digits=3)))")
println("Target position:  ($(round(x_path[N], digits=3)), $(round(y_path[N], digits=3)), $(round(z_path[N], digits=3)))")
println("="^60 * "\n")
println("Staged solves: $stages, lambda_track_init=$lambda_track_init, lambda_u_ref_init=$lambda_u_ref_init")
println("Trust-region eps: pos=$trust_pos_eps, orient=$trust_orient_eps, vel=$trust_vel_eps")

# Build oracle
oracle = QuadrupedDynamicsOracle(mech, x0, u0; N=N, dt=dt)
d = n*N + m*(N-1)
p = n*(N-1)

# Build JuMP model
# Ipopt options tuned for trajectory optimization with physics constraints:
# - mu_strategy adaptive: better for nonconvex problems
# - warm_start_init_point: use our dynamically-feasible warmstart
# - acceptable tolerances: allow early termination if making good progress
model = Model(optimizer_with_attributes(Ipopt.Optimizer, 
    "linear_solver" => "ma97",
    "hessian_approximation" => "limited-memory",
    "print_user_options" => "yes",
    "max_iter" => 1000,
    # Barrier parameter strategy
    "mu_strategy" => "adaptive",
    "mu_oracle" => "quality-function",
    # Warm start
    "warm_start_init_point" => "yes",
    "warm_start_bound_push" => 1e-6,
    "warm_start_mult_bound_push" => 1e-6,
    # Acceptable tolerances (allow early termination)
    "acceptable_tol" => 1e-4,
    "acceptable_iter" => 10,
    "acceptable_constr_viol_tol" => 1e-4,
    # NLP scaling
    "nlp_scaling_method" => "gradient-based",
    "obj_scaling_factor" => 1e-2,  # Scale down objective (it's large)
    # Derivative checking (disable in production)
    "check_derivatives_for_naninf" => "yes",
))

@variable(model, X[1:n, 1:N])
@variable(model, U[1:m, 1:(N-1)])

# Initial state constraint
@constraint(model, X[:,1] .== x0)

# Control bounds
@constraint(model, box_con[i in 1:m, k in 1:(N-1)], -u_max_abs <= U[i,k] <= u_max_abs)

# ==========================================================================================
# IMPROVED OBJECTIVE FUNCTION
# ==========================================================================================
# 
# The original objective only tracked XY position and height with simple quadratic costs.
# This improved version adds several terms for better balance and motion quality:
#
# 1. POSITION TRACKING (same as before, but now uses reference from trajectory):
#    - Penalizes deviation from reference XY path
#    - Penalizes deviation from reference height
#
# 2. ORIENTATION TRACKING (NEW):
#    - Quaternion error: penalizes deviation from upright orientation
#    - Uses geodesic-like error: 1 - qw² (minimum when qw=±1, i.e., upright)
#    - This keeps the robot balanced and prevents tilting/rolling
#
# 3. VELOCITY TRACKING (NEW):
#    - Penalizes deviation from reference XY velocity (smooth forward motion)
#    - Penalizes vertical velocity (reduces bouncing)
#
# 4. ANGULAR VELOCITY PENALTY (NEW):
#    - Penalizes body angular velocity (wx, wy, wz)
#    - Prevents excessive spinning/wobbling
#
# 5. CONTROL REGULARIZATION:
#    - L2 control effort (same as before)
#    - Control smoothness (Δu penalty, increased weight)
#    - Reference tracking (NEW): penalizes deviation from reference controls
#
# ==========================================================================================

# Tracking weight ramp (ignore first few steps to allow settling)
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

# Convert reference arrays to vectors for JuMP (avoid scalar indexing issues)
x_ref = Vector{Float64}(x_path)
y_ref = Vector{Float64}(y_path)
z_ref = Vector{Float64}(z_path)
vx_ref = Vector{Float64}(vx_path)
vy_ref = Vector{Float64}(vy_path)
vz_ref = Vector{Float64}(vz_path)
qw_ref = Vector{Float64}(qw_path)
qx_ref = Vector{Float64}(qx_path)
qy_ref = Vector{Float64}(qy_path)
qz_ref = Vector{Float64}(qz_path)

# Optional trust-region constraints around rerolled reference (now that refs exist)
trust_pos_con_x = Vector{JuMP.ConstraintRef}(undef, N)
trust_pos_con_y = Vector{JuMP.ConstraintRef}(undef, N)
trust_pos_con_z = Vector{JuMP.ConstraintRef}(undef, N)
trust_orient_con_qx = Vector{JuMP.ConstraintRef}(undef, N)
trust_orient_con_qy = Vector{JuMP.ConstraintRef}(undef, N)
trust_orient_con_qz = Vector{JuMP.ConstraintRef}(undef, N)
trust_orient_con_qw = Vector{JuMP.ConstraintRef}(undef, N)
trust_vel_con_vx = Vector{JuMP.ConstraintRef}(undef, N)
trust_vel_con_vy = Vector{JuMP.ConstraintRef}(undef, N)
trust_vel_con_vz = Vector{JuMP.ConstraintRef}(undef, N)

if trust_pos_eps > 0.0
    for k in 1:N
        trust_pos_con_x[k] = @constraint(model, -trust_pos_eps <= X[BASE_X_IDX, k] - x_ref[k] <= trust_pos_eps)
        trust_pos_con_y[k] = @constraint(model, -trust_pos_eps <= X[BASE_Y_IDX, k] - y_ref[k] <= trust_pos_eps)
        trust_pos_con_z[k] = @constraint(model, -trust_pos_eps <= X[BASE_Z_IDX, k] - z_ref[k] <= trust_pos_eps)
    end
end
if trust_orient_eps > 0.0
    for k in 1:N
        trust_orient_con_qx[k] = @constraint(model, -trust_orient_eps <= X[BASE_QX_IDX, k] - qx_ref[k] <= trust_orient_eps)
        trust_orient_con_qy[k] = @constraint(model, -trust_orient_eps <= X[BASE_QY_IDX, k] - qy_ref[k] <= trust_orient_eps)
        trust_orient_con_qz[k] = @constraint(model, -trust_orient_eps <= X[BASE_QZ_IDX, k] - qz_ref[k] <= trust_orient_eps)
        trust_orient_con_qw[k] = @constraint(model, -trust_orient_eps <= X[BASE_QW_IDX, k] - qw_ref[k] <= trust_orient_eps)
    end
end
if trust_vel_eps > 0.0
    for k in 1:N
        trust_vel_con_vx[k] = @constraint(model, -trust_vel_eps <= X[BASE_VX_IDX, k] - vx_ref[k] <= trust_vel_eps)
        trust_vel_con_vy[k] = @constraint(model, -trust_vel_eps <= X[BASE_VY_IDX, k] - vy_ref[k] <= trust_vel_eps)
        trust_vel_con_vz[k] = @constraint(model, -trust_vel_eps <= X[BASE_VZ_IDX, k] - vz_ref[k] <= trust_vel_eps)
    end
end

# Stage costs
@expression(model, stage_pos_xy[k=1:N-1],
    w_track[k] * w_track_xy * (
        (X[BASE_X_IDX, k] - x_ref[k])^2 + 
        (X[BASE_Y_IDX, k] - y_ref[k])^2
    )
)

@expression(model, stage_pos_z[k=1:N-1],
    w_track[k] * w_track_z * (X[BASE_Z_IDX, k] - z_ref[k])^2
)

# Orientation tracking: penalize deviation from reference quaternion
# Using simplified error: sum of squared differences for qx, qy, qz (should be near 0 for upright)
# Plus (1 - qw²) which is minimum when qw = ±1
@expression(model, stage_orient[k=1:N-1],
    w_track[k] * w_orient * (
        (X[BASE_QX_IDX, k] - qx_ref[k])^2 +
        (X[BASE_QY_IDX, k] - qy_ref[k])^2 +
        (X[BASE_QZ_IDX, k] - qz_ref[k])^2 +
        (X[BASE_QW_IDX, k] - qw_ref[k])^2
    )
)

# Velocity tracking
@expression(model, stage_vel[k=1:N-1],
    w_track[k] * (
        w_vel_xy * ((X[BASE_VX_IDX, k] - vx_ref[k])^2 + (X[BASE_VY_IDX, k] - vy_ref[k])^2) +
        w_vel_z * X[BASE_VZ_IDX, k]^2  # Penalize vertical velocity
    )
)

# Angular velocity penalty
@expression(model, stage_ang_vel[k=1:N-1],
    w_track[k] * w_ang_vel * (
        X[BASE_WX_IDX, k]^2 + X[BASE_WY_IDX, k]^2 + X[BASE_WZ_IDX, k]^2
    )
)

# Control effort
@expression(model, stage_u[k=1:N-1], w_u * sum(U[:, k].^2))

# Control smoothness
@expression(model, stage_du[k=2:N-1], w_du * sum((U[:, k] - U[:, k-1]).^2))

# Reference control tracking (use warmstart controls as soft reference)
U_ref = [Vector{Float64}(U_ws[k]) for k in 1:(N-1)]
@expression(model, stage_u_ref[k=1:N-1], 
    w_u_ref * sum((U[i, k] - U_ref[k][i])^2 for i in 1:m)
)

# Terminal cost (higher weight on final state)
@expression(model, terminal_cost,
    w_track[N] * w_terminal * (
        (X[BASE_X_IDX, N] - x_ref[N])^2 +
        (X[BASE_Y_IDX, N] - y_ref[N])^2 +
        (X[BASE_Z_IDX, N] - z_ref[N])^2 +
        0.5 * (  # Also penalize terminal orientation
            (X[BASE_QX_IDX, N] - qx_ref[N])^2 +
            (X[BASE_QY_IDX, N] - qy_ref[N])^2 +
            (X[BASE_QZ_IDX, N] - qz_ref[N])^2 +
            (X[BASE_QW_IDX, N] - qw_ref[N])^2
        )
    )
)

# Objective with stageable multipliers
@variable(model, lambda_track >= 0)
@variable(model, lambda_u_ref >= 0)

@expression(model, tracking_terms,
    sum(stage_pos_xy) + sum(stage_pos_z) +
    sum(stage_orient) + sum(stage_vel) + sum(stage_ang_vel) +
    terminal_cost
)

@objective(model, Min,
    lambda_track * tracking_terms +
    sum(stage_u) + sum(stage_du) +
    lambda_u_ref * sum(stage_u_ref)
)

# ==========================================================================================
# WARMSTART from trajectory
# ==========================================================================================
println("Setting warmstart from trajectory...")
for k in 1:N
    set_start_value.(X[:, k], X_ws[k])
end
for k in 1:(N-1)
    set_start_value.(U[:, k], U_ws[k])
end

# Dynamics constraint via VectorNonlinearOracle
zvars = vcat(vec(X), vec(U))
@assert length(zvars) == d

oracle_set = MOI.VectorNonlinearOracle(;
    dimension = d,
    l = zeros(p),
    u = zeros(p),
    eval_f = (ret, z) -> eval_f!(oracle, ret, z),
    jacobian_structure = oracle.jac_struct,
    eval_jacobian = (ret, z) -> eval_jacobian!(oracle, ret, z),
    hessian_lagrangian_structure = oracle.hess_struct,
    eval_hessian_lagrangian = (ret, z, μ) -> eval_hessian_lagrangian!(oracle, ret, z, μ),
)

@constraint(model, dyn, zvars in oracle_set)

# ==========================================================================================
# SOLVE
# ==========================================================================================
println("\nSolving with staged trust-region...")

# Stage 1: strong tracking and reference controls
fix(lambda_track, lambda_track_init; force=true)
fix(lambda_u_ref, lambda_u_ref_init; force=true)
optimize!(model)

println("Stage 1 done. Status=$(termination_status(model)), Obj=$(try objective_value(model) catch; NaN end)")

# Prepare warmstart for next stage
for k in 1:N
    set_start_value.(X[:, k], value.(X[:, k]))
end
for k in 1:(N-1)
    set_start_value.(U[:, k], value.(U[:, k]))
end

# Optionally relax trust-region bounds for stage 2
if trust_pos_eps > 0.0
    for k in 1:N
        JuMP.set_lower_bound(trust_pos_con_x[k], -Inf)
        JuMP.set_upper_bound(trust_pos_con_x[k], Inf)
        JuMP.set_lower_bound(trust_pos_con_y[k], -Inf)
        JuMP.set_upper_bound(trust_pos_con_y[k], Inf)
        JuMP.set_lower_bound(trust_pos_con_z[k], -Inf)
        JuMP.set_upper_bound(trust_pos_con_z[k], Inf)
    end
end
if trust_orient_eps > 0.0
    for k in 1:N
        JuMP.set_lower_bound(trust_orient_con_qx[k], -Inf)
        JuMP.set_upper_bound(trust_orient_con_qx[k], Inf)
        JuMP.set_lower_bound(trust_orient_con_qy[k], -Inf)
        JuMP.set_upper_bound(trust_orient_con_qy[k], Inf)
        JuMP.set_lower_bound(trust_orient_con_qz[k], -Inf)
        JuMP.set_upper_bound(trust_orient_con_qz[k], Inf)
        JuMP.set_lower_bound(trust_orient_con_qw[k], -Inf)
        JuMP.set_upper_bound(trust_orient_con_qw[k], Inf)
    end
end
if trust_vel_eps > 0.0
    for k in 1:N
        JuMP.set_lower_bound(trust_vel_con_vx[k], -Inf)
        JuMP.set_upper_bound(trust_vel_con_vx[k], Inf)
        JuMP.set_lower_bound(trust_vel_con_vy[k], -Inf)
        JuMP.set_upper_bound(trust_vel_con_vy[k], Inf)
        JuMP.set_lower_bound(trust_vel_con_vz[k], -Inf)
        JuMP.set_upper_bound(trust_vel_con_vz[k], Inf)
    end
end

# Stage 2+: relax multipliers to 1.0 and resolve
for s in 2:stages
    fix(lambda_track, 1.0; force=true)
    fix(lambda_u_ref, 1.0; force=true)
    optimize!(model)
    println("Stage $(s) done. Status=$(termination_status(model)), Obj=$(try objective_value(model) catch; NaN end)")
    # Refresh warmstart for potential next stage
    for k in 1:N
        set_start_value.(X[:, k], value.(X[:, k]))
    end
    for k in 1:(N-1)
        set_start_value.(U[:, k], value.(U[:, k]))
    end
end

println("\n" * "="^60)
println("Optimization Results")
println("="^60)
println("Termination status: $(termination_status(model))")
println("Objective value: $(try objective_value(model) catch; NaN end)")

Xopt = value.(X)
Uopt = value.(U)

println("\nInitial position: ($(round(Xopt[BASE_X_IDX, 1], digits=4)), $(round(Xopt[BASE_Y_IDX, 1], digits=4)), $(round(Xopt[BASE_Z_IDX, 1], digits=4)))")
println("Final position:   ($(round(Xopt[BASE_X_IDX, N], digits=4)), $(round(Xopt[BASE_Y_IDX, N], digits=4)), $(round(Xopt[BASE_Z_IDX, N], digits=4)))")
println("Reference final:  ($(round(x_ref[N], digits=4)), $(round(y_ref[N], digits=4)), $(round(z_ref[N], digits=4)))")

# Position error
pos_error = sqrt((Xopt[BASE_X_IDX, N] - x_ref[N])^2 + (Xopt[BASE_Y_IDX, N] - y_ref[N])^2 + (Xopt[BASE_Z_IDX, N] - z_ref[N])^2)
println("Final position error: $(round(pos_error, digits=4)) m")

println("="^60 * "\n")

# ==========================================================================================
# Visualization
# ==========================================================================================
println("Setting up visualization...")

# Re-initialize mechanism
Dojo.set_minimal_state!(mech, x0)

function controller_replay!(mechanism, k)
    if 1 <= k <= size(Uopt, 2)
        Dojo.set_input!(mechanism, Uopt[:, k])
    else
        Dojo.set_input!(mechanism, zeros(size(Uopt, 1)))
    end
    return
end

Tsim = dt * (N - 1)
storage = Dojo.simulate!(mech, Tsim, controller_replay!; record=true, opts=DOJO_OPTS)

vis = Dojo.visualize(mech, storage)
Dojo.render(vis)

# Print trajectory summary
println("\nBase XY trace:")
for k in 1:5:N
    xk = storage.state[k]
    println("k=$k  x=$(round(xk[BASE_X_IDX], digits=3))  y=$(round(xk[BASE_Y_IDX], digits=3))  z=$(round(xk[BASE_Z_IDX], digits=3))")
end
