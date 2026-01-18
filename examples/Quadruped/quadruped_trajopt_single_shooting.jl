#!/usr/bin/env julia
# quadruped_trajopt_single_shooting_corrected.jl
#
# Single-shooting trajectory optimization:
#   - Decision vars: U[:,1:N-1] only
#   - Full rollout from x0 inside the objective
#   - Exact gradient via adjoint using Dojo get_minimal_gradients!
#
# Dependencies:
#   ] add JuMP Ipopt HSL_jll Dojo DojoEnvironments LinearAlgebra

using JuMP, Ipopt, HSL_jll
using LinearAlgebra
using Dojo
using DojoEnvironments

# -----------------------------
# Knobs
# -----------------------------
const N  = 21
const dt = 0.02

const BASE_X_IDX = 1
const BASE_Y_IDX = 2
const BASE_Z_IDX = 3

const x_target  = 0.50
const u_max_abs = 40.0

const w_track_xy = 50.0
const w_track_z  = 100.0
const w_terminal = 200.0
const w_u        = 1e-2
const w_du       = 1e-2

const DOJO_OPTS = Dojo.SolverOptions{Float64}(;
    rtol = 1e-4,
    btol = 1e-3,
    max_iter = 20,
    max_ls = 8,
    verbose = false
)

# -----------------------------
# Build mechanism + initial state
# -----------------------------
function build_quadruped()
    mech = DojoEnvironments.get_mechanism(:quadruped)
    x0 = Vector(Dojo.get_minimal_state(mech))
    m  = Dojo.input_dimension(mech)
    return mech, x0, m
end

# -----------------------------
# Single-shooting evaluator with caching
# -----------------------------
mutable struct ShootingEvaluator
    mech
    x0::Vector{Float64}
    N::Int
    n::Int
    m::Int

    # tracking path/weights
    x_path::Vector{Float64}
    y_path::Vector{Float64}
    z_nom::Float64
    w_track::Vector{Float64}

    # cache key (exact match)
    last_u::Vector{Float64}
    have_last::Bool

    # rollout storage
    X::Matrix{Float64}          # n × N
    A::Array{Float64,3}         # n × n × (N-1)
    B::Array{Float64,3}         # n × m × (N-1)

    # scratch buffers (IMPORTANT: real Vectors, not views)
    xk::Vector{Float64}
    uk::Vector{Float64}
    dldx::Vector{Float64}
    λnext::Vector{Float64}
end

function ShootingEvaluator(mech, x0::Vector{Float64}, m::Int; N::Int)
    n = length(x0)

    # Reference path: straight-line in base x
    x_path = collect(range(x0[BASE_X_IDX], x0[BASE_X_IDX] + x_target, length=N))
    y_path = fill(x0[BASE_Y_IDX], N)
    z_nom  = x0[BASE_Z_IDX] * 0.8

    # Gate tracking
    Kdrop, Kramp = 5, 10
    w_track = zeros(Float64, N)
    for k in 1:N
        w_track[k] = k <= Kdrop ? 0.0 : (k < Kdrop + Kramp ? (k - Kdrop) / Kramp : 1.0)
    end

    return ShootingEvaluator(
        deepcopy(mech),
        copy(x0),
        N, n, m,
        x_path, y_path, z_nom, w_track,
        Float64[], false,
        zeros(n, N),
        zeros(n, n, N-1),
        zeros(n, m, N-1),
        zeros(n),
        zeros(m),
        zeros(n),
        zeros(n),
    )
end

@inline function u_matches(E::ShootingEvaluator, u::Vector{Float64})
    E.have_last && length(E.last_u) == length(u) &&
        all(@inbounds(E.last_u[i] == u[i]) for i in eachindex(u))
end

# Rollout from x0 and store X, A, B using get_minimal_gradients!
function ensure_rollout!(E::ShootingEvaluator, u::Vector{Float64})
    if u_matches(E, u)
        return
    end
    E.last_u = copy(u)
    E.have_last = true

    n, m, N = E.n, E.m, E.N
    mech = E.mech

    # X[:,1] = x0
    @inbounds for i in 1:n
        E.X[i,1] = E.x0[i]
    end

    # Forward rollout
    @inbounds for k in 1:(N-1)
        off = (k-1) * m
        for j in 1:m
            E.uk[j] = u[off + j]
        end

        # IMPORTANT: xk must be a real Vector{Float64} (not a view/SubArray)
        for i in 1:n
            E.xk[i] = E.X[i,k]
        end

        Dojo.set_minimal_state!(mech, E.xk)
        Dojo.set_input!(mech, E.uk)

        # Dojo returns Jacobians; also performs internal step needed for gradients
        A2, B2 = Dojo.get_minimal_gradients!(mech, E.xk, E.uk; opts=DOJO_OPTS)

        for i in 1:n, j in 1:n
            E.A[i,j,k] = A2[i,j]
        end
        for i in 1:n, j in 1:m
            E.B[i,j,k] = B2[i,j]
        end

        xn = Dojo.get_minimal_state(mech)
        for i in 1:n
            E.X[i,k+1] = xn[i]
        end
    end
end

# Objective value from rollout + u
function cost_from_rollout(E::ShootingEvaluator, u::Vector{Float64})
    n, m, N = E.n, E.m, E.N
    J = 0.0

    # stage costs
    @inbounds for k in 1:(N-1)
        wx = E.w_track[k]
        if wx != 0.0
            dx = E.X[BASE_X_IDX,k] - E.x_path[k]
            dy = E.X[BASE_Y_IDX,k] - E.y_path[k]
            dz = E.X[BASE_Z_IDX,k] - E.z_nom
            J += wx * (w_track_xy * (dx*dx + dy*dy) + w_track_z * (dz*dz))
        end

        off = (k-1)*m
        s = 0.0
        for j in 1:m
            uj = u[off + j]
            s += uj*uj
        end
        J += w_u * s
    end

    # du smoothness
    @inbounds for k in 2:(N-1)
        off  = (k-1)*m
        offp = (k-2)*m
        s = 0.0
        for j in 1:m
            d = u[off + j] - u[offp + j]
            s += d*d
        end
        J += w_du * s
    end

    # terminal
    @inbounds begin
        wx = E.w_track[N]
        dx = E.X[BASE_X_IDX,N] - E.x_path[N]
        dy = E.X[BASE_Y_IDX,N] - E.y_path[N]
        dz = E.X[BASE_Z_IDX,N] - E.z_nom
        J += wx * w_terminal * (dx*dx + dy*dy + dz*dz)
    end

    return J
end

# Add ∂ℓ/∂x_k (only base xyz terms here) into dldx buffer
function add_dldx!(E::ShootingEvaluator, k::Int; terminal::Bool)
    fill!(E.dldx, 0.0)
    wx = E.w_track[k]
    if wx == 0.0
        return
    end

    dx = E.X[BASE_X_IDX,k] - E.x_path[k]
    dy = E.X[BASE_Y_IDX,k] - E.y_path[k]
    dz = E.X[BASE_Z_IDX,k] - E.z_nom

    if terminal
        E.dldx[BASE_X_IDX] += 2.0 * wx * w_terminal * dx
        E.dldx[BASE_Y_IDX] += 2.0 * wx * w_terminal * dy
        E.dldx[BASE_Z_IDX] += 2.0 * wx * w_terminal * dz
    else
        E.dldx[BASE_X_IDX] += 2.0 * wx * w_track_xy * dx
        E.dldx[BASE_Y_IDX] += 2.0 * wx * w_track_xy * dy
        E.dldx[BASE_Z_IDX] += 2.0 * wx * w_track_z  * dz
    end
end

# Exact gradient via adjoint
function grad_from_rollout!(E::ShootingEvaluator, g::Vector{Float64}, u::Vector{Float64})
    fill!(g, 0.0)
    n, m, N = E.n, E.m, E.N

    # terminal costate λ_N = ∂ℓ_N/∂x_N
    add_dldx!(E, N; terminal=true)
    @inbounds for i in 1:n
        E.λnext[i] = E.dldx[i]
    end

    # backward pass
    @inbounds for k in (N-1):-1:1
        off = (k-1)*m

        # ∂/∂u_k of w_u ||u_k||^2
        for j in 1:m
            g[off + j] += 2.0 * w_u * u[off + j]
        end

        # du terms: contribute to both neighbors
        if k >= 2
            offp = (k-2)*m
            for j in 1:m
                g[off + j] += 2.0 * w_du * (u[off + j] - u[offp + j])
            end
        end
        if k <= N-2
            offn = k*m
            for j in 1:m
                g[off + j] += 2.0 * w_du * (u[off + j] - u[offn + j])
            end
        end

        # + B_k' * λ_{k+1}
        for j in 1:m
            s = 0.0
            for i in 1:n
                s += E.B[i,j,k] * E.λnext[i]
            end
            g[off + j] += s
        end

        # λ_k = ∂ℓ_k/∂x_k + A_k' * λ_{k+1}
        add_dldx!(E, k; terminal=false)
        λk = similar(E.λnext)
        for i in 1:n
            s = 0.0
            for r in 1:n
                s += E.A[r,i,k] * E.λnext[r]
            end
            λk[i] = E.dldx[i] + s
        end
        E.λnext .= λk
    end
end

# -----------------------------
# Hook into JuMP via register()
# -----------------------------
const mech, x0, m = build_quadruped()
const EVAL = ShootingEvaluator(mech, x0, m; N=N)
const dU = m*(N-1)

function traj_cost(args...)
    u = collect(Float64, args)
    ensure_rollout!(EVAL, u)
    return cost_from_rollout(EVAL, u)
end

function traj_cost_grad(g_out::Vector{Float64}, args...)
    u = collect(Float64, args)
    ensure_rollout!(EVAL, u)
    grad_from_rollout!(EVAL, g_out, u)
    return
end

# -----------------------------
# Optimize
# -----------------------------
model = Model(optimizer_with_attributes(Ipopt.Optimizer,
    "linear_solver" => "ma97",
    "hessian_approximation" => "limited-memory",
    "print_user_options" => "yes",
))

@variable(model, U[1:m, 1:(N-1)])
@constraint(model, [i in 1:m, k in 1:(N-1)], -u_max_abs <= U[i,k] <= u_max_abs)

# Register custom nonlinear objective: takes dU scalar inputs
JuMP.register(model, :traj_cost, dU, traj_cost, traj_cost_grad)

uvars = vec(U)
@NLobjective(model, Min, traj_cost(uvars...))

# start from zeros
for k in 1:(N-1)
    set_start_value.(U[:,k], 0.0)
end

println("Solving (single shooting)...")
optimize!(model)

println("\ntermination_status = ", termination_status(model))
println("objective_value    = ", try objective_value(model) catch; NaN end)

Uopt = value.(U)

# -----------------------------
# Playback
# -----------------------------
Dojo.set_minimal_state!(mech, x0)
function controller_replay!(mechanism, k)
    if 1 <= k <= size(Uopt, 2)
        Dojo.set_input!(mechanism, Uopt[:, k])
    else
        Dojo.set_input!(mechanism, zeros(size(Uopt, 1)))
    end
end

Tsim = dt * (N - 1)
storage = Dojo.simulate!(mech, Tsim, controller_replay!; record=true, opts=DOJO_OPTS)
vis = Dojo.visualize(mech, storage)
Dojo.render(vis)

println("\nFinal base pos from playback (assuming BASE_*_IDX):")
xN = storage.state[end]
println("  x = ", xN[BASE_X_IDX])
println("  y = ", xN[BASE_Y_IDX])
println("  z = ", xN[BASE_Z_IDX])