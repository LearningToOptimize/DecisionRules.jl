#!/usr/bin/env julia
# quadruped_trajopt_single_shooting_profiled_fixedvec.jl

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

const DOJO_OPTS_STEP = Dojo.SolverOptions{Float64}(;
    rtol = 1e-4,
    btol = 1e-3,
    max_iter = 20,
    max_ls = 8,
    verbose = false
)

const DOJO_OPTS_GRAD = Dojo.SolverOptions{Float64}(;
    rtol = 1e-3,
    btol = 1e-2,
    max_iter = 10,
    max_ls = 6,
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
# Evaluator
# -----------------------------
mutable struct ShootingEvaluator
    mech
    x0::Vector{Float64}
    N::Int
    n::Int
    m::Int
    dU::Int

    x_path::Vector{Float64}
    y_path::Vector{Float64}
    z_nom::Float64
    w_track::Vector{Float64}

    last_u_cost::Vector{Float64}
    have_cost::Bool
    last_u_grad::Vector{Float64}
    have_grad::Bool

    X::Matrix{Float64}
    A::Array{Float64,3}
    B::Array{Float64,3}

    xk::Vector{Float64}
    uk::Vector{Float64}
    ubuf::Vector{Float64}

    dldx::Vector{Float64}
    λnext::Vector{Float64}
    λk::Vector{Float64}

    # profiling
    cost_calls::Int
    grad_calls::Int
    t_cost_rollout_ns::Int
    t_grad_AB_ns::Int
    t_adjoint_ns::Int
end

function ShootingEvaluator(mech, x0::Vector{Float64}, m::Int; N::Int)
    n = length(x0)
    dU = m*(N-1)

    x_path = collect(range(x0[BASE_X_IDX], x0[BASE_X_IDX] + x_target, length=N))
    y_path = fill(x0[BASE_Y_IDX], N)
    z_nom  = x0[BASE_Z_IDX] * 0.8

    Kdrop, Kramp = 5, 10
    w_track = zeros(Float64, N)
    for k in 1:N
        w_track[k] = k <= Kdrop ? 0.0 : (k < Kdrop + Kramp ? (k - Kdrop)/Kramp : 1.0)
    end

    return ShootingEvaluator(
        deepcopy(mech), copy(x0), N, n, m, dU,
        x_path, y_path, z_nom, w_track,
        Float64[], false,
        Float64[], false,
        zeros(n, N),
        zeros(n, n, N-1),
        zeros(n, m, N-1),
        zeros(n),
        zeros(m),
        zeros(dU),
        zeros(n),
        zeros(n),
        zeros(n),
        0, 0, 0, 0, 0
    )
end

@inline function same_u(a::Vector{Float64}, b::Vector{Float64})
    length(a) == length(b) && all(@inbounds(a[i] == b[i]) for i in eachindex(a))
end

@inline function load_u!(E::ShootingEvaluator, args)
    @inbounds for i in 1:E.dU
        E.ubuf[i] = args[i]
    end
    return E.ubuf
end

function ensure_cost_rollout!(E::ShootingEvaluator, u::Vector{Float64})
    if E.have_cost && same_u(E.last_u_cost, u)
        return
    end
    E.last_u_cost = copy(u)
    E.have_cost = true
    E.have_grad = false

    t0 = time_ns()

    n, m, N = E.n, E.m, E.N
    mech = E.mech

    @inbounds for i in 1:n
        E.X[i,1] = E.x0[i]
    end

    @inbounds for k in 1:(N-1)
        off = (k-1)*m
        for j in 1:m
            E.uk[j] = u[off+j]
        end
        for i in 1:n
            E.xk[i] = E.X[i,k]
        end

        Dojo.set_minimal_state!(mech, E.xk)
        Dojo.set_input!(mech, E.uk)

        Dojo.step_minimal_coordinates!(mech, E.xk, E.uk; opts=DOJO_OPTS_STEP)

        xn = Dojo.get_minimal_state(mech)
        for i in 1:n
            E.X[i,k+1] = xn[i]
        end
    end

    E.t_cost_rollout_ns += (time_ns() - t0)
end

function ensure_grad_data!(E::ShootingEvaluator, u::Vector{Float64})
    ensure_cost_rollout!(E, u)
    if E.have_grad && same_u(E.last_u_grad, u)
        return
    end
    E.last_u_grad = copy(u)
    E.have_grad = true

    t0 = time_ns()

    n, m, N = E.n, E.m, E.N
    mech = E.mech

    @inbounds for k in 1:(N-1)
        off = (k-1)*m
        for j in 1:m
            E.uk[j] = u[off+j]
        end
        for i in 1:n
            E.xk[i] = E.X[i,k]
        end

        Dojo.set_minimal_state!(mech, E.xk)
        Dojo.set_input!(mech, E.uk)

        A2, B2 = Dojo.get_minimal_gradients!(mech, E.xk, E.uk; opts=DOJO_OPTS_GRAD)

        for i in 1:n, j in 1:n
            E.A[i,j,k] = A2[i,j]
        end
        for i in 1:n, j in 1:m
            E.B[i,j,k] = B2[i,j]
        end
    end

    E.t_grad_AB_ns += (time_ns() - t0)
end

function cost_from_X(E::ShootingEvaluator, u::Vector{Float64})
    n, m, N = E.n, E.m, E.N
    J = 0.0

    @inbounds for k in 1:(N-1)
        wx = E.w_track[k]
        if wx != 0.0
            dx = E.X[BASE_X_IDX,k] - E.x_path[k]
            dy = E.X[BASE_Y_IDX,k] - E.y_path[k]
            dz = E.X[BASE_Z_IDX,k] - E.z_nom
            J += wx * (w_track_xy*(dx*dx + dy*dy) + w_track_z*(dz*dz))
        end

        off = (k-1)*m
        s = 0.0
        for j in 1:m
            uj = u[off+j]
            s += uj*uj
        end
        J += w_u * s
    end

    @inbounds for k in 2:(N-1)
        off  = (k-1)*m
        offp = (k-2)*m
        s = 0.0
        for j in 1:m
            d = u[off+j] - u[offp+j]
            s += d*d
        end
        J += w_du * s
    end

    @inbounds begin
        wx = E.w_track[N]
        dx = E.X[BASE_X_IDX,N] - E.x_path[N]
        dy = E.X[BASE_Y_IDX,N] - E.y_path[N]
        dz = E.X[BASE_Z_IDX,N] - E.z_nom
        J += wx * w_terminal * (dx*dx + dy*dy + dz*dz)
    end

    return J
end

function add_dldx!(E::ShootingEvaluator, k::Int; terminal::Bool)
    fill!(E.dldx, 0.0)
    wx = E.w_track[k]
    wx == 0.0 && return

    dx = E.X[BASE_X_IDX,k] - E.x_path[k]
    dy = E.X[BASE_Y_IDX,k] - E.y_path[k]
    dz = E.X[BASE_Z_IDX,k] - E.z_nom

    if terminal
        E.dldx[BASE_X_IDX] += 2.0*wx*w_terminal*dx
        E.dldx[BASE_Y_IDX] += 2.0*wx*w_terminal*dy
        E.dldx[BASE_Z_IDX] += 2.0*wx*w_terminal*dz
    else
        E.dldx[BASE_X_IDX] += 2.0*wx*w_track_xy*dx
        E.dldx[BASE_Y_IDX] += 2.0*wx*w_track_xy*dy
        E.dldx[BASE_Z_IDX] += 2.0*wx*w_track_z *dz
    end
end

function grad_from_data!(E::ShootingEvaluator, g::AbstractVector{Float64}, u::Vector{Float64})
    t0 = time_ns()

    fill!(g, 0.0)
    n, m, N = E.n, E.m, E.N

    add_dldx!(E, N; terminal=true)
    @inbounds for i in 1:n
        E.λnext[i] = E.dldx[i]
    end

    @inbounds for k in (N-1):-1:1
        off = (k-1)*m

        for j in 1:m
            g[off+j] += 2.0*w_u*u[off+j]
        end
        if k >= 2
            offp = (k-2)*m
            for j in 1:m
                g[off+j] += 2.0*w_du*(u[off+j] - u[offp+j])
            end
        end
        if k <= N-2
            offn = k*m
            for j in 1:m
                g[off+j] += 2.0*w_du*(u[off+j] - u[offn+j])
            end
        end

        for j in 1:m
            s = 0.0
            for i in 1:n
                s += E.B[i,j,k] * E.λnext[i]
            end
            g[off+j] += s
        end

        add_dldx!(E, k; terminal=false)
        for i in 1:n
            s = 0.0
            for r in 1:n
                s += E.A[r,i,k] * E.λnext[r]
            end
            E.λk[i] = E.dldx[i] + s
        end
        E.λnext .= E.λk
    end

    E.t_adjoint_ns += (time_ns() - t0)
end

# -----------------------------
# Global evaluator for callbacks
# -----------------------------
const mech, x0, m = build_quadruped()
const EVAL = ShootingEvaluator(mech, x0, m; N=N)

function maybe_print_stats!(E::ShootingEvaluator)
    if (E.cost_calls % 50 == 0) || (E.grad_calls % 50 == 0)
        println("\n--- callback stats ---")
        println("cost_calls = ", E.cost_calls, " | grad_calls = ", E.grad_calls)
        println("t_cost_rollout_s = ", E.t_cost_rollout_ns / 1e9)
        println("t_grad_AB_s      = ", E.t_grad_AB_ns / 1e9)
        println("t_adjoint_s      = ", E.t_adjoint_ns / 1e9)
        println("----------------------\n")
    end
end

function traj_cost(args...)
    EVAL.cost_calls += 1
    u = load_u!(EVAL, args)
    ensure_cost_rollout!(EVAL, u)
    maybe_print_stats!(EVAL)
    return cost_from_X(EVAL, u)
end

function traj_cost_grad(g_out::AbstractVector{Float64}, args...)
    EVAL.grad_calls += 1
    u = load_u!(EVAL, args)
    ensure_grad_data!(EVAL, u)
    grad_from_data!(EVAL, g_out, u)
    maybe_print_stats!(EVAL)
    return
end

# -----------------------------
# Optimize
# -----------------------------
model = Model(optimizer_with_attributes(Ipopt.Optimizer,
    "linear_solver" => "ma97",
    "hessian_approximation" => "limited-memory",
    "print_level" => 5,
    "print_timing_statistics" => "yes",
    "mu_strategy" => "adaptive",
))

@variable(model, U[1:m, 1:(N-1)])
@constraint(model, [i in 1:m, k in 1:(N-1)], -u_max_abs <= U[i,k] <= u_max_abs)

JuMP.register(model, :traj_cost, EVAL.dU, traj_cost, traj_cost_grad)

# IMPORTANT: create the flat vector outside NL macros
uvars = vec(U)
@NLobjective(model, Min, traj_cost(uvars...))

for k in 1:(N-1)
    set_start_value.(U[:,k], 0.0)
end

println("Dojo dims: n=$(length(x0)), m=$m | dU=$(EVAL.dU) | N=$N")
println("Solving (single shooting, profiled)...")
optimize!(model)

println("\ntermination_status = ", termination_status(model))
println("objective_value    = ", try objective_value(model) catch; NaN end)

Uopt = value.(U)

# Playback
Dojo.set_minimal_state!(mech, x0)
function controller_replay!(mechanism, k)
    if 1 <= k <= size(Uopt, 2)
        Dojo.set_input!(mechanism, Uopt[:, k])
    else
        Dojo.set_input!(mechanism, zeros(size(Uopt, 1)))
    end
end

Tsim = dt * (N - 1)
storage = Dojo.simulate!(mech, Tsim, controller_replay!; record=true, opts=DOJO_OPTS_STEP)
vis = Dojo.visualize(mech, storage)
Dojo.render(vis)