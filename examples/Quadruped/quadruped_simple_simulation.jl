#!/usr/bin/env julia
# ==========================================================================================
# quadruped_trajopt_vector_oracle_dojo_stable.jl
#
# Direct transcription trajectory optimization for Dojo quadruped using:
#   - JuMP + Ipopt
#   - MOI.VectorNonlinearOracle (vector-valued dynamics defects)
#   - Dojo step_minimal_coordinates! for dynamics
#   - Dojo get_minimal_gradients! returning (A,B) for Jacobians
#
# This matches the "Atlas JuMP TO" style: X,U decision variables + dynamics constraints.
#
# Dependencies:
#   ] add JuMP Ipopt MathOptInterface ForwardDiff LinearAlgebra SparseArrays
#   ] add Dojo DojoEnvironments
# ==========================================================================================

using JuMP, Ipopt, HSL_jll
import MathOptInterface
const MOI = MathOptInterface

using LinearAlgebra
using SparseArrays
using ForwardDiff

using Dojo
using DojoEnvironments

# ------------------------------------------------------------------------------------------
# Knobs
# ------------------------------------------------------------------------------------------
const N  = 21  # number of knot points (N-1 steps); adjust as needed for your task/horizon
const dt = 0.02

# # Start with LBFGS; you can enable exact Hessians later (expensive).
# const USE_EXACT_HESSIAN = false

# Indices into minimal state for base position (often 1:3, but model-dependent).
const BASE_X_IDX = 1
const BASE_Y_IDX = 2
const BASE_Z_IDX = 3

const x_target = 0.50
const u_max_abs = 40.0

const w_track_xy = 50.0
const w_track_z  = 100.0
const w_terminal = 200.0
const w_u        = 1e-2
const w_du       = 1e-2

# Dojo solver options for each step/linearization (tune as needed)
const DOJO_OPTS = Dojo.SolverOptions{Float64}(;
    rtol = 1e-4,
    btol = 1e-3,
    max_iter = 20,
    max_ls = 8,
    verbose = false
)

# ------------------------------------------------------------------------------------------
# Dojo wrappers that match the stable API
# ------------------------------------------------------------------------------------------

"""
    dojo_step!(xnext, mech, x, u; opts)

Compute xnext = f(x,u) in minimal coordinates using Dojo.
Uses step_minimal_coordinates!(mechanism, x, u; opts) then reads get_minimal_state(mech).
"""
function dojo_step!(xnext::AbstractVector, mech, x::AbstractVector, u::AbstractVector; opts=DOJO_OPTS)
    # Ensure mechanism starts from (x,u) deterministically.
    Dojo.set_minimal_state!(mech, x)
    Dojo.set_input!(mech, u)

    # Step with minimal coordinates (documented API)  [oai_citation:2‡dojo-sim.github.io](https://dojo-sim.github.io/Dojo.jl/stable/api.html)
    Dojo.step_minimal_coordinates!(mech, x, u; opts=opts)

    # Read the next minimal state
    xn = Dojo.get_minimal_state(mech)
    copyto!(xnext, xn)
    return nothing
end

"""
    dojo_linearize!(A, B, mech, x, u; opts)

Fill A = ∂f/∂x, B = ∂f/∂u at (x,u) in minimal coordinates.

Dojo stable API:
    A, B = get_minimal_gradients!(mechanism, y, u; opts)
This call *requires* simulating a step internally (per docs).  [oai_citation:3‡dojo-sim.github.io](https://dojo-sim.github.io/Dojo.jl/stable/api.html)
"""
function dojo_linearize!(A::AbstractMatrix, B::AbstractMatrix, mech, x::AbstractVector, u::AbstractVector; opts=DOJO_OPTS)
    # Reset state so the gradients are evaluated at the right point
    Dojo.set_minimal_state!(mech, x)
    Dojo.set_input!(mech, u)

    A2, B2 = Dojo.get_minimal_gradients!(mech, x, u; opts=opts)  #  [oai_citation:4‡dojo-sim.github.io](https://dojo-sim.github.io/Dojo.jl/stable/api.html)
    copyto!(A, A2)
    copyto!(B, B2)
    return nothing
end

# ------------------------------------------------------------------------------------------
# Mechanism + initial state helpers
# ------------------------------------------------------------------------------------------

function build_quadruped()
    # Prefer the standard environment mechanism getter
    mech = DojoEnvironments.get_mechanism(:quadruped)

    x0 = Vector(Dojo.get_minimal_state(mech))
    m  = Dojo.input_dimension(mech)
    u0 = zeros(m)

    return mech, x0, u0
end

# ------------------------------------------------------------------------------------------
# Indexing for z = [vec(X); vec(U)]
# ------------------------------------------------------------------------------------------
idx_x(n::Int, k::Int, i::Int) = (k - 1) * n + i
idx_u(n::Int, N::Int, m::Int, k::Int, j::Int) = n * N + (k - 1) * m + j

# ------------------------------------------------------------------------------------------
# Sparsity structures
# ------------------------------------------------------------------------------------------
function build_jacobian_structure(n::Int, m::Int, N::Int)
    # defect_k = x_{k+1} - f(x_k, u_k)
    # ∂defect/∂x_k dense (-A), ∂/∂u_k dense (-B), ∂/∂x_{k+1} identity
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
    # Block-diagonal per (x_k,u_k) for μᵀ defects (since x_{k+1} is linear)
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
# Per-step cache so eval_f and eval_jacobian don’t re-simulate redundantly
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

@inline function cache_matches(c::StepCache, x::Vector{Float64}, u::Vector{Float64})
    return c.have_fx && c.have_AB && (x == c.x) && (u == c.u)
end

@inline function cache_matches_fx(c::StepCache, x::Vector{Float64}, u::Vector{Float64})
    return c.have_fx && (x == c.x) && (u == c.u)
end

@inline function cache_matches_AB(c::StepCache, x::Vector{Float64}, u::Vector{Float64})
    return c.have_AB && (x == c.x) && (u == c.u)
end

# ------------------------------------------------------------------------------------------
# Vector oracle object
# ------------------------------------------------------------------------------------------
mutable struct QuadrupedDynamicsOracle
    mechs::Vector           # one mechanism copy per step to avoid interference
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

    # gradients call (Dojo simulates internally)
    A2, B2 = Dojo.get_minimal_gradients!(mech, c.x, c.u; opts=DOJO_OPTS)
    copyto!(c.A, A2); copyto!(c.B, B2)

    # grab the resulting next state from mechanism
    xn = Dojo.get_minimal_state(mech)
    copyto!(c.fx, xn)

    c.have_AB = true
    c.have_fx = true
end

# function ensure_fx!(O::QuadrupedDynamicsOracle, k::Int, x::Vector{Float64}, u::Vector{Float64})
#     c = O.caches[k]
#     if cache_matches_fx(c, x, u)
#         return
#     end
#     copyto!(c.x, x); copyto!(c.u, u)
#     dojo_step!(c.fx, O.mechs[k], c.x, c.u; opts=DOJO_OPTS)
#     c.have_fx = true
#     # fx does not imply AB is valid
#     return
# end

# function ensure_AB!(O::QuadrupedDynamicsOracle, k::Int, x::Vector{Float64}, u::Vector{Float64})
#     c = O.caches[k]
#     if cache_matches_AB(c, x, u)
#         return
#     end
#     copyto!(c.x, x); copyto!(c.u, u)
#     dojo_linearize!(c.A, c.B, O.mechs[k], c.x, c.u; opts=DOJO_OPTS)
#     c.have_AB = true
#     return
# end

function eval_f!(O::QuadrupedDynamicsOracle, ret::AbstractVector, z::AbstractVector)
    n, m, N = O.n, O.m, O.N
    @inbounds for k in 1:(N-1)
        # load x_k and u_k
        x = O.xk; u = O.uk
        for i in 1:n; x[i] = z[idx_x(n,k,i)]; end
        for j in 1:m; u[j] = z[idx_u(n,N,m,k,j)]; end

        ensure_fx!(O, k, x, u)
        # ensure_fx_and_AB!(O, k, x, u)
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
        # load x_k and u_k
        x = O.xk; u = O.uk
        for i in 1:n; x[i] = z[idx_x(n,k,i)]; end
        for j in 1:m; u[j] = z[idx_u(n,N,m,k,j)]; end

        # ensure_AB!
        ensure_fx_and_AB!(O, k, x, u)
        c = O.caches[k]

        # Must match build_jacobian_structure order
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
    # Exact Hessian is optional; when Ipopt uses LBFGS it typically won’t call this heavily.
    # This computes block Hessians of -μ_kᵀ f_step(x_k,u_k) via ForwardDiff.
    n, m, N = O.n, O.m, O.N
    idx = 0
    @inbounds for k in 1:(N-1)
        μk = view(μ, (k-1)*n + 1 : k*n)

        # pack xu = [x_k; u_k]
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

        # Fill dense lower triangle (matches build_hessian_structure)
        for a in 1:(n+m)
            for b in 1:a
                idx += 1
                ret[idx] = O.H[a,b]
            end
        end
    end
    return nothing
end

# ------------------------------------------------------------------------------------------
# Build + solve
# ------------------------------------------------------------------------------------------

mech, x0, u0 = build_quadruped()
n, m = length(x0), length(u0)

println("Dojo dims: n=$n, m=$m | N=$N, dt=$dt, horizon=$(dt*(N-1))s")
println("Dojo API sanity:")
println("  has get_minimal_gradients! = ", isdefined(Dojo, :get_minimal_gradients!))
println("  has step_minimal_coordinates! = ", isdefined(Dojo, :step_minimal_coordinates!))

oracle = QuadrupedDynamicsOracle(mech, x0, u0; N=N, dt=dt)
d = n*N + m*(N-1)         # z dimension
p = n*(N-1)               # defects dimension

model = Model(optimizer_with_attributes(Ipopt.Optimizer, 
        # "print_level" => 0,
        "linear_solver" => "ma97",
        "hessian_approximation" => "limited-memory",
        # "max_iter" => 20,
        # "mu_target" => 1e-8,
        "print_user_options" => "yes",
))

@variable(model, X[1:n, 1:N])
@variable(model, U[1:m, 1:(N-1)])

@constraint(model, X[:,1] .== x0)
@constraint(model, box_con[i in 1:m, k in 1:(N-1)], -u_max_abs <= U[i,k] <= u_max_abs)

# Reference “path”: straight-line base x
x_path = range(x0[BASE_X_IDX], x0[BASE_X_IDX] + x_target, length=N)
y_path = fill(x0[BASE_Y_IDX], N)
z_nom  = x0[BASE_Z_IDX] * 0.8

Kdrop = 5    # ignore tracking for first 5 steps
Kramp = 10   # ramp up over next 10 steps

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

# Use w_track[k] to gate the tracking cost
@expression(model, stage_track[k=1:N-1],
    w_track[k] * (
        w_track_xy * ((X[BASE_X_IDX,k] - x_path[k])^2 + (X[BASE_Y_IDX,k] - y_path[k])^2) +
        w_track_z  * (X[BASE_Z_IDX,k] - z_nom)^2
    )
)

@expression(model, stage_u[k=1:N-1], w_u * sum(U[:,k].^2))
@expression(model, stage_du[k=2:N-1], w_du * sum((U[:,k] - U[:,k-1]).^2))
@expression(model, terminal_cost,
    w_track[N] * w_terminal * (
        (X[BASE_X_IDX,N] - x_path[N])^2 +
        (X[BASE_Y_IDX,N] - y_path[N])^2 +
        (X[BASE_Z_IDX,N] - z_nom)^2
    )
)
@objective(model, Min, sum(stage_track) + sum(stage_u) + sum(stage_du) + terminal_cost)

# Rollout a consistent initial trajectory under some U guess
U_guess = [zeros(m) for _ in 1:(N-1)]
X_guess = Vector{Vector{Float64}}(undef, N)
X_guess[1] = copy(x0)

tmp_mech = deepcopy(mech)
for k in 1:(N-1)
    xnext = similar(x0)
    dojo_step!(xnext, tmp_mech, X_guess[k], U_guess[k]; opts=DOJO_OPTS)
    X_guess[k+1] = xnext
end

for k in 1:N
    set_start_value.(X[:,k], X_guess[k])
end
for k in 1:(N-1)
    set_start_value.(U[:,k], 0.0)
end

# VectorNonlinearOracle constraint: zvars in oracle_set
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

println("Solving...")
optimize!(model)

println("\ntermination_status = ", termination_status(model))
println("objective_value    = ", try objective_value(model) catch; NaN end)

Xopt = value.(X)
Uopt = value.(U)

println("\nFinal base pos (using BASE_*_IDX):")
println("  x = ", Xopt[BASE_X_IDX, N])
println("  y = ", Xopt[BASE_Y_IDX, N])
println("  z = ", Xopt[BASE_Z_IDX, N])

# ==========================================================================================
# Playback + visualization (append after solve)
# ==========================================================================================

# If your script returned only Xopt/Uopt, make sure you still have `mech` in scope.
# If not, reconstruct it the same way you did earlier:
# mech, x0, _ = build_quadruped()

# Re-initialize to the same initial state used in optimization
Dojo.set_minimal_state!(mech, x0)

# Build a replay controller matching Dojo's simulate!(mechanism, T, controller!; record=true)
# The callback signature in Dojo examples is controller!(mechanism, k) where k is the step index.
function controller_replay!(mechanism, k)
    if 1 <= k <= size(Uopt, 2)
        Dojo.set_input!(mechanism, Uopt[:, k])
    else
        Dojo.set_input!(mechanism, zeros(size(Uopt, 1)))
    end
    return
end

# Simulate for the same horizon (N-1 steps of dt)
Tsim = dt * (N - 1)

# Record a trajectory for visualization.
# If your Dojo version supports passing opts, keep it; otherwise remove opts=...
storage = Dojo.simulate!(mech, Tsim, controller_replay!; record=true, opts=DOJO_OPTS)

# Visualize + render (MeshCat-based). This is exactly the Dojo example pattern.
vis = Dojo.visualize(mech, storage)
Dojo.render(vis)

# ------------------------------------------------------------------------------------------
# Optional: quick sanity printout of base XY over time (using your assumed indices)
# ------------------------------------------------------------------------------------------
println("\nBase XY trace (from recorded states, using BASE_X_IDX/BASE_Y_IDX):")
for k in 1:N
    xk = storage.state[k]  # some Dojo versions store as storage.state / storage.x / etc.
    # If the next line errors, print `fieldnames(typeof(storage))` and adjust.
    println("k=$k  x=$(xk[BASE_X_IDX])  y=$(xk[BASE_Y_IDX])")
end