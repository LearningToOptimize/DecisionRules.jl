using JuMP
import Ipopt, HSL_jll
using DiffOpt
using LinearAlgebra
using SparseArrays
using ForwardDiff
import MathOptInterface
const MOI = MathOptInterface

using Dojo
using DojoEnvironments

# Base coordinate indices (minimal coordinates as exposed by Dojo)
const BASE_X_IDX = 1
const BASE_Y_IDX = 2
const BASE_Z_IDX = 3

# Dojo solver options
const DOJO_OPTS = Dojo.SolverOptions{Float64}(;
    rtol = 1e-4,
    btol = 1e-3,
    max_iter = 20,
    max_ls = 8,
    verbose = false
)

"""
    interpolate_path(path_waypoints, N)

Given waypoints as `[[x1,y1], [x2,y2], ...]`, produce `N` targets
(spread uniformly in x) with linear interpolation on each segment.
Returns a Vector of tuples `(x_path, y_path)` of length `N`.
"""
function interpolate_path(path_waypoints::Vector{<:AbstractVector}, N::Int)
    xs = getindex.(path_waypoints, 1)
    ys = getindex.(path_waypoints, 2)
    x_grid = range(first(xs), last(xs); length=N)

    targets = Vector{Tuple{Float64,Float64}}(undef, N)
    for (i, xq) in enumerate(x_grid)
        if xq <= xs[1]
            targets[i] = (xs[1], ys[1])
            continue
        elseif xq >= xs[end]
            targets[i] = (xs[end], ys[end])
            continue
        end
        # Find segment [xs[j], xs[j+1]] containing xq
        seg = findfirst(j -> xs[j] <= xq <= xs[j+1], 1:length(xs)-1)
        @assert seg !== nothing "Path waypoints must be ordered in x"
        x1, y1 = xs[seg], ys[seg]
        x2, y2 = xs[seg+1], ys[seg+1]
        t = (xq - x1) / (x2 - x1)
        targets[i] = (xq, y1 + t * (y2 - y1))
    end
    return targets
end

"""
    dojo_step!(xnext, mech, x, u; opts)
Compute xnext = f(x,u) in minimal coordinates using Dojo.
"""
function dojo_step!(xnext::AbstractVector, mech, x::AbstractVector, u::AbstractVector; opts=DOJO_OPTS)
    Dojo.set_minimal_state!(mech, x)
    Dojo.set_input!(mech, u)
    Dojo.step_minimal_coordinates!(mech, x, u; opts=opts)
    xn = Dojo.get_minimal_state(mech)
    copyto!(xnext, xn)
    return nothing
end

"""
    dojo_linearize!(A, B, mech, x, u; opts)
Fill A = ∂f/∂x, B = ∂f/∂u at (x,u) in minimal coordinates using Dojo.
"""
function dojo_linearize!(A::AbstractMatrix, B::AbstractMatrix, mech, x::AbstractVector, u::AbstractVector; opts=DOJO_OPTS)
    Dojo.set_minimal_state!(mech, x)
    Dojo.set_input!(mech, u)
    A2, B2 = Dojo.get_minimal_gradients!(mech, x, u; opts=opts)
    copyto!(A, A2)
    copyto!(B, B2)
    return nothing
end

"""
    StageOracle
Mutable struct to hold Dojo mechanism and cached gradients for a single stage.
"""
mutable struct StageOracle
    mech::Any
    x_prev_val::Vector{Float64}
    u_val::Vector{Float64}
    xnext::Vector{Float64}
    A::Matrix{Float64}
    B::Matrix{Float64}
    have_fx::Bool
    have_AB::Bool
end

"""
    build_quadruped_subproblems(; N=21, dt=0.02, path_waypoints=..., u_max_abs=40.0,
                                  w_path=50.0, w_u=1e-2, w_z=10.0, penalty_norm=1e6)

Create per-period subproblems for the quadruped using VectorNonlinearOracle for full Dojo dynamics.
Each subproblem:
- has parameters for incoming state `x_prev` (from rollout)
- has parameters for desired state `x_target` (from the neural network)
- uses a one-step dynamics constraint via VectorNonlinearOracle
- minimizes path tracking + network target + control effort
Returns `(subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples)`.
"""
function build_quadruped_subproblems(; N::Int=21, dt::Float64=0.02,
    path_waypoints::Vector{<:AbstractVector} = [
        [0.0, 0.0],
        [0.3, 0.2],
        [0.6, 0.3],
        [0.9, 0.2],
        [1.2, 0.0],
        [1.5, -0.2]
    ],
    u_max_abs::Float64 = 40.0,
    w_path::Float64 = 50.0,
    w_u::Float64 = 1e-2,
    w_z::Float64 = 10.0,
    penalty_norm::Float64 = 1e6,
    solver = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0,
        "linear_solver" => "ma97",
        "hessian_approximation" => "limited-memory",
        "mu_target" => 1e-8,
    )
)
    mech = DojoEnvironments.get_mechanism(:quadruped)
    x0 = Vector(Dojo.get_minimal_state(mech))
    m = Dojo.input_dimension(mech)
    n = length(x0)

    # Pre-compute path targets over the horizon
    path_targets = interpolate_path(path_waypoints, N)
    z_nom = x0[BASE_Z_IDX]  # encourage staying near start height

    subproblems = Vector{JuMP.Model}(undef, N - 1)
    state_params_in = Vector{Vector{Any}}(undef, N - 1)
    state_params_out = Vector{Vector{Tuple{Any, VariableRef}}}(undef, N - 1)
    uncertainty_samples = [Vector{Tuple{VariableRef, Vector{Float64}}}() for _ in 1:(N - 1)]

    for k in 1:(N - 1)
        model = DiffOpt.diff_model(solver)

        @variable(model, x[1:n])
        @variable(model, -u_max_abs <= u[1:m] <= u_max_abs)
        @variable(model, x_prev[1:n] in MOI.Parameter.(0.0))
        @variable(model, x_target[1:n] in MOI.Parameter.(0.0))
        @variable(model, norm_deficit >= 0)
        @variable(model, stage_cost >= 0)

        # ===== Dojo VectorNonlinearOracle for dynamics =====
        # Dynamics defect: xnext - f(x_prev, u) = 0
        # We pack xnext, u into a combined variable and use VectorNonlinearOracle
        
        mech_stage = deepcopy(mech)  # separate mechanism copy for this stage
        oracle = StageOracle(mech_stage, zeros(n), zeros(m), zeros(n), zeros(n,n), zeros(n,m), false, false)
        
        function eval_f_stage(ret::AbstractVector, xnext_u::AbstractVector)
            copyto!(oracle.xnext, view(xnext_u, 1:n))
            copyto!(oracle.u_val, view(xnext_u, n+1:n+m))
            
            # Compute f(x_prev, u)
            tmp = similar(oracle.xnext)
            dojo_step!(tmp, oracle.mech, oracle.x_prev_val, oracle.u_val; opts=DOJO_OPTS)
            
            # Defect: xnext - f(x_prev, u)
            for i in 1:n
                ret[i] = oracle.xnext[i] - tmp[i]
            end
            return nothing
        end
        
        function eval_jac_stage(ret::AbstractVector, xnext_u::AbstractVector)
            copyto!(oracle.xnext, view(xnext_u, 1:n))
            copyto!(oracle.u_val, view(xnext_u, n+1:n+m))
            
            # Jacobian: [I, -B] where B = ∂f/∂u at (x_prev, u)
            dojo_linearize!(oracle.A, oracle.B, oracle.mech, oracle.x_prev_val, oracle.u_val; opts=DOJO_OPTS)
            
            # ∂defect/∂xnext = I (identity)
            idx = 0
            for i in 1:n
                idx += 1
                ret[idx] = 1.0
            end
            # ∂defect/∂u = -B
            for i in 1:n, j in 1:m
                idx += 1
                ret[idx] = -oracle.B[i, j]
            end
            return nothing
        end
        
        function eval_hess_lag_stage(ret::AbstractVector, xnext_u::AbstractVector, μ::AbstractVector)
            # Hessian of -μᵀ * defect(xnext, u)
            # Since defect is linear in xnext, Hessian only depends on u part
            copyto!(oracle.xnext, view(xnext_u, 1:n))
            copyto!(oracle.u_val, view(xnext_u, n+1:n+m))
            
            # Use ForwardDiff for the u-part of the Hessian
            φ_u = function(u_vec)
                tmp = similar(oracle.xnext)
                dojo_step!(tmp, oracle.mech, oracle.x_prev_val, u_vec; opts=DOJO_OPTS)
                return -dot(μ[1:n], tmp)  # -μᵀ * f(x_prev, u)
            end
            
            # Hessian w.r.t [xnext; u] is block: [[0, 0], [0, H_uu]]
            H_uu = zeros(m, m)
            ForwardDiff.hessian!(H_uu, φ_u, oracle.u_val)
            
            # Fill sparse lower triangle for [xnext; u]
            idx = 0
            for i in 1:(n+m)
                for j in 1:i
                    idx += 1
                    if i <= n
                        ret[idx] = 0.0  # xnext block is zero
                    else
                        # u block: H_uu
                        ret[idx] = H_uu[i - n, j <= n ? 1 : j - n]
                    end
                end
            end
            return nothing
        end
        
        # Jacobian structure: [I, -B] is dense (n × (n+m))
        jac_struct = [(i, j) for i in 1:n for j in 1:(n+m)]
        
        # Hessian structure: lower triangle, block zero on xnext, H_uu on u
        hess_struct = [(i, j) for i in 1:(n+m) for j in 1:i if i > n]
        
        oracle_set = MOI.VectorNonlinearOracle(;
            dimension = n + m,
            l = zeros(n),
            u = zeros(n),
            eval_f = (ret, z) -> eval_f_stage(ret, z),
            jacobian_structure = jac_struct,
            eval_jacobian = (ret, z) -> eval_jac_stage(ret, z),
            hessian_lagrangian_structure = hess_struct,
            eval_hessian_lagrangian = (ret, z, μ) -> eval_hess_lag_stage(ret, z, μ),
        )
        
        combined_vars = vcat(x, u)
        @constraint(model, dyn, combined_vars in oracle_set)
        
        # ===== Update x_prev parameter before each solve (user will set it) =====
        # This will be done via set_parameter_value in the training loop
        
        # ===== Objectives =====
        # L1 slack on target tracking
        @constraint(model, norm_deficit >= sum(abs(x[i] - x_target[i]) for i in 1:n))

        x_path, y_path = path_targets[k + 1]

        @constraint(model, stage_cost >= w_path * ((x[BASE_X_IDX] - x_path)^2 + (x[BASE_Y_IDX] - y_path)^2) +
            w_z * (x[BASE_Z_IDX] - z_nom)^2 +
            w_u * sum(u[j]^2 for j in 1:m)
        )

        @objective(model, Min, stage_cost + penalty_norm * norm_deficit)

        state_params_in[k] = x_prev
        state_params_out[k] = [(x_target[i], x[i]) for i in 1:n]
        subproblems[k] = model
    end

    initial_state = x0
    return subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples
end
