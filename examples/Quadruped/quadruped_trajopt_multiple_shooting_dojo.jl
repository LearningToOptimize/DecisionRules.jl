#!/usr/bin/env julia
# ==========================================================================================
# quadruped_trajopt_multiple_shooting_dojo.jl
#
# Multiple shooting trajectory optimization for Dojo quadruped:
#   - Decision variables:
#       Xb[:, s]   boundary states, s = 1..(M+1)
#       U[:, k]    controls,        k = 1..(N-1) where N-1 = M*L
#   - Dynamics constraints (vector oracle), one constraint per segment:
#       Xb[:, s+1] - rollout_L_steps(Xb[:, s], U[:, k0:k1]) == 0
#   - eval_f uses step_minimal_coordinates! (cheap)
#   - eval_jacobian uses get_minimal_gradients! and propagates segment sensitivities (Aseg, Bseg)
#   - Objective (simple): track only at boundary times + control effort + smoothness + terminal.
#
# Notes:
#   * This avoids using "infeasible intermediate X_k variables" (there are none),
#     but the boundary states Xb are still decision variables.
#   * Choose smaller L (more segments) if you need more “state flexibility”.
#
# Dependencies:
#   ] add JuMP Ipopt HSL_jll MathOptInterface LinearAlgebra Dojo DojoEnvironments
# ==========================================================================================

using JuMP, Ipopt, HSL_jll
import MathOptInterface
const MOI = MathOptInterface

using LinearAlgebra
using Dojo
using DojoEnvironments

# -----------------------------
# Knobs
# -----------------------------
const dt = 0.02

# Total knots N = (N-1)+1, where N-1 is total steps.
# Multiple shooting parameters:
const L = 5             # steps per segment
const M = 4             # number of segments
const N = M*L + 1       # total knot points (for reference path / playback)

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

# Solver options: keep gradients a bit looser if needed
const DOJO_OPTS_STEP = Dojo.SolverOptions{Float64}(;
    rtol = 1e-4, btol = 1e-3, max_iter = 20, max_ls = 8, verbose = false
)
const DOJO_OPTS_GRAD = Dojo.SolverOptions{Float64}(;
    rtol = 1e-3, btol = 1e-2, max_iter = 10, max_ls = 6, verbose = false
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
# Indexing into z = [vec(Xb); vec(U)]
# -----------------------------
idx_xb(n::Int, s::Int, i::Int) = (s - 1) * n + i
idx_u(n::Int, Nb::Int, m::Int, k::Int, j::Int) = n * Nb + (k - 1) * m + j
# where Nb = M+1 boundaries, k = 1..(N-1)

# Map segment s (1..M) to global control step range k0..k1 (inclusive)
@inline seg_k0(s::Int) = (s - 1) * L + 1
@inline seg_k1(s::Int) = s * L

# -----------------------------
# Jacobian sparsity for segment constraints
# -----------------------------
function build_jacobian_structure(n::Int, m::Int, M::Int, L::Int)
    # For segment s: defect_s = Xb_{s+1} - Fseg(Xb_s, U_{k0:k1})
    # Nonzeros:
    #   ∂/∂Xb_s      : dense (n×n)
    #   ∂/∂U steps   : dense (n×(m*L))
    #   ∂/∂Xb_{s+1}  : identity (n)
    Nb = M + 1
    S = Vector{Tuple{Int,Int}}()
    sizehint!(S, M * (n*n + n*m*L + n))

    for s in 1:M
        row0 = (s - 1) * n
        # Xb_s block
        for i in 1:n, j in 1:n
            push!(S, (row0 + i, idx_xb(n, s, j)))
        end
        # U blocks for this segment
        k0 = seg_k0(s)
        for t in 0:(L-1)
            k = k0 + t
            for i in 1:n, j in 1:m
                push!(S, (row0 + i, idx_u(n, Nb, m, k, j)))
            end
        end
        # Xb_{s+1} identity
        for i in 1:n
            push!(S, (row0 + i, idx_xb(n, s+1, i)))
        end
    end
    return S
end

# -----------------------------
# Segment cache
# -----------------------------
mutable struct SegmentCache
    x0::Vector{Float64}       # boundary start
    useg::Vector{Float64}     # packed u for segment (m*L)
    xend::Vector{Float64}     # end state
    Aseg::Matrix{Float64}     # ∂xend/∂xstart (n×n)
    Bseg::Matrix{Float64}     # ∂xend/∂useg  (n×(m*L))
    have_fx::Bool
    have_sens::Bool
end

function SegmentCache(n::Int, m::Int, L::Int)
    SegmentCache(zeros(n), zeros(m*L), zeros(n), zeros(n,n), zeros(n, m*L), false, false)
end

@inline function same_vec(a::Vector{Float64}, b::Vector{Float64})
    length(a) == length(b) && all(@inbounds(a[i] == b[i]) for i in eachindex(a))
end

@inline function cache_match_fx(c::SegmentCache, x0::Vector{Float64}, useg::Vector{Float64})
    c.have_fx && same_vec(c.x0, x0) && same_vec(c.useg, useg)
end

@inline function cache_match_sens(c::SegmentCache, x0::Vector{Float64}, useg::Vector{Float64})
    c.have_sens && same_vec(c.x0, x0) && same_vec(c.useg, useg)
end

# -----------------------------
# Oracle object
# -----------------------------
mutable struct MultipleShootingOracle
    mech                    # one mechanism instance (we reset state each step)
    n::Int
    m::Int
    M::Int
    L::Int
    Nb::Int

    # scratch buffers (must be Vector, not view/SubArray)
    xk::Vector{Float64}
    uk::Vector{Float64}

    # caches per segment
    caches::Vector{SegmentCache}

    jac_struct::Vector{Tuple{Int,Int}}
end

function MultipleShootingOracle(mech, n::Int, m::Int; M::Int, L::Int)
    Nb = M + 1
    caches = [SegmentCache(n, m, L) for _ in 1:M]
    jac_struct = build_jacobian_structure(n, m, M, L)
    return MultipleShootingOracle(deepcopy(mech), n, m, M, L, Nb, zeros(n), zeros(m), caches, jac_struct)
end

# Pack segment controls from z into a (m*L) vector
function load_useg!(O::MultipleShootingOracle, useg::Vector{Float64}, z::AbstractVector, s::Int)
    k0 = seg_k0(s)
    m, L, n, Nb = O.m, O.L, O.n, O.Nb
    idx = 0
    @inbounds for t in 0:(L-1)
        k = k0 + t
        for j in 1:m
            idx += 1
            useg[idx] = z[idx_u(n, Nb, m, k, j)]
        end
    end
    return useg
end

# Ensure segment end state (cheap) using step_minimal_coordinates!
function ensure_segment_fx!(O::MultipleShootingOracle, s::Int, x0::Vector{Float64}, useg::Vector{Float64})
    c = O.caches[s]
    if cache_match_fx(c, x0, useg)
        return
    end

    copyto!(c.x0, x0)
    copyto!(c.useg, useg)

    mech = O.mech
    n, m, L = O.n, O.m, O.L

    # start from boundary state
    copyto!(O.xk, x0)

    @inbounds for t in 1:L
        # uk = useg block
        off = (t-1)*m
        for j in 1:m
            O.uk[j] = useg[off + j]
        end

        Dojo.set_minimal_state!(mech, O.xk)
        Dojo.set_input!(mech, O.uk)

        Dojo.step_minimal_coordinates!(mech, O.xk, O.uk; opts=DOJO_OPTS_STEP)

        xn = Dojo.get_minimal_state(mech)
        for i in 1:n
            O.xk[i] = xn[i]
        end
    end

    copyto!(c.xend, O.xk)
    c.have_fx = true
    c.have_sens = false
end

# Ensure segment sensitivities using get_minimal_gradients! and forward sensitivity propagation
function ensure_segment_sens!(O::MultipleShootingOracle, s::Int, x0::Vector{Float64}, useg::Vector{Float64})
    c = O.caches[s]
    if cache_match_sens(c, x0, useg)
        return
    end

    # We will recompute from scratch for clarity/stability
    copyto!(c.x0, x0)
    copyto!(c.useg, useg)

    mech = O.mech
    n, m, L = O.n, O.m, O.L

    # Phi: ∂x/∂x0, initialize I
    fill!(c.Aseg, 0.0)
    @inbounds for i in 1:n
        c.Aseg[i,i] = 1.0
    end

    # S: ∂x/∂useg, initialize 0 (n × (m*L))
    fill!(c.Bseg, 0.0)

    # Current state
    copyto!(O.xk, x0)

    # Scratch for A,B each step
    Astep = zeros(n,n)
    Bstep = zeros(n,m)

    @inbounds for t in 1:L
        off = (t-1)*m
        for j in 1:m
            O.uk[j] = useg[off + j]
        end

        Dojo.set_minimal_state!(mech, O.xk)
        Dojo.set_input!(mech, O.uk)

        A2, B2 = Dojo.get_minimal_gradients!(mech, O.xk, O.uk; opts=DOJO_OPTS_GRAD)

        # copy to dense scratch (avoid alias surprises)
        for i in 1:n, j in 1:n
            Astep[i,j] = A2[i,j]
        end
        for i in 1:n, j in 1:m
            Bstep[i,j] = B2[i,j]
        end

        # x_{t+1}
        xn = Dojo.get_minimal_state(mech)
        for i in 1:n
            O.xk[i] = xn[i]
        end

        # Update sensitivities:
        #   Phi <- Astep * Phi
        #   S   <- Astep * S; and add Bstep into the block for this t
        Phi_new = similar(c.Aseg)
        for i in 1:n, j in 1:n
            ssum = 0.0
            for k in 1:n
                ssum += Astep[i,k] * c.Aseg[k,j]
            end
            Phi_new[i,j] = ssum
        end
        c.Aseg .= Phi_new

        # S_new = Astep*S
        S_new = similar(c.Bseg)
        for i in 1:n, j in 1:(m*L)
            ssum = 0.0
            for k in 1:n
                ssum += Astep[i,k] * c.Bseg[k,j]
            end
            S_new[i,j] = ssum
        end

        # Add Bstep into columns (off+1):(off+m)
        for i in 1:n, j in 1:m
            S_new[i, off + j] += Bstep[i,j]
        end
        c.Bseg .= S_new
    end

    copyto!(c.xend, O.xk)
    c.have_fx = true
    c.have_sens = true
end

# -----------------------------
# Oracle callbacks
# -----------------------------
function eval_f!(O::MultipleShootingOracle, ret::AbstractVector, z::AbstractVector)
    n, m, M, L, Nb = O.n, O.m, O.M, O.L, O.Nb
    useg = Vector{Float64}(undef, m*L)

    @inbounds for s in 1:M
        # boundary start state x0
        x0 = O.xk
        for i in 1:n
            x0[i] = z[idx_xb(n, s, i)]
        end
        load_useg!(O, useg, z, s)

        ensure_segment_fx!(O, s, x0, useg)
        c = O.caches[s]

        row0 = (s-1)*n
        for i in 1:n
            ret[row0 + i] = z[idx_xb(n, s+1, i)] - c.xend[i]
        end
    end
    return nothing
end

function eval_jacobian!(O::MultipleShootingOracle, ret::AbstractVector, z::AbstractVector)
    n, m, M, L, Nb = O.n, O.m, O.M, O.L, O.Nb
    useg = Vector{Float64}(undef, m*L)

    idx = 0
    @inbounds for s in 1:M
        # x0
        x0 = O.xk
        for i in 1:n
            x0[i] = z[idx_xb(n, s, i)]
        end
        load_useg!(O, useg, z, s)

        ensure_segment_sens!(O, s, x0, useg)
        c = O.caches[s]

        # Must match build_jacobian_structure order:
        # 1) -Aseg (dense n×n) wrt Xb_s
        for i in 1:n, j in 1:n
            idx += 1
            ret[idx] = -c.Aseg[i,j]
        end
        # 2) -Bseg (dense n×(m*L)) wrt U segment
        for i in 1:n, j in 1:(m*L)
            idx += 1
            ret[idx] = -c.Bseg[i,j]
        end
        # 3) +I wrt Xb_{s+1}
        for i in 1:n
            idx += 1
            ret[idx] = 1.0
        end
    end
    return nothing
end

# -----------------------------
# Build + solve
# -----------------------------
mech, x0, m = build_quadruped()
n = length(x0)

@assert N == M*L + 1
Nb = M + 1
d  = n*Nb + m*(N-1)     # z dimension
p  = n*M                # constraint dimension

println("Dojo dims: n=$n, m=$m | N=$N (steps=$(N-1)), segments M=$M, L=$L, boundaries Nb=$Nb")
println("Horizon = ", dt*(N-1), " s")

oracle = MultipleShootingOracle(mech, n, m; M=M, L=L)

model = Model(optimizer_with_attributes(Ipopt.Optimizer,
    "linear_solver" => "ma97",
    "hessian_approximation" => "limited-memory",
    "print_level" => 5,
))

@variable(model, Xb[1:n, 1:Nb])          # boundary states
@variable(model, U[1:m, 1:(N-1)])        # controls

@constraint(model, Xb[:,1] .== x0)
@constraint(model, [i in 1:m, k in 1:(N-1)], -u_max_abs <= U[i,k] <= u_max_abs)

# --- Objective: track base only at boundary indices (coarse, but cheap/robust)
# Boundary s corresponds to knot index k = 1 + (s-1)*L
b2k(s) = 1 + (s-1)*L

x_path = collect(range(x0[BASE_X_IDX], x0[BASE_X_IDX] + x_target, length=N))
y_path = fill(x0[BASE_Y_IDX], N)
z_nom  = x0[BASE_Z_IDX] * 0.8

Kdrop, Kramp = 1, max(1, Nb-1)   # gate over boundaries (not knots)
w_track_b = zeros(Float64, Nb)
for s in 1:Nb
    if s <= Kdrop
        w_track_b[s] = 0.0
    elseif s < Kdrop + Kramp
        w_track_b[s] = (s - Kdrop) / Kramp
    else
        w_track_b[s] = 1.0
    end
end

@expression(model, boundary_track[s=1:Nb-1],
    w_track_b[s] * (
        w_track_xy * ((Xb[BASE_X_IDX,s] - x_path[b2k(s)])^2 + (Xb[BASE_Y_IDX,s] - y_path[b2k(s)])^2) +
        w_track_z  * (Xb[BASE_Z_IDX,s] - z_nom)^2
    )
)

@expression(model, stage_u[k=1:N-1], w_u * sum(U[:,k].^2))
@expression(model, stage_du[k=2:N-1], w_du * sum((U[:,k] - U[:,k-1]).^2))

@expression(model, terminal_cost,
    w_track_b[Nb] * w_terminal * (
        (Xb[BASE_X_IDX,Nb] - x_path[N])^2 +
        (Xb[BASE_Y_IDX,Nb] - y_path[N])^2 +
        (Xb[BASE_Z_IDX,Nb] - z_nom)^2
    )
)

@objective(model, Min, sum(boundary_track) + sum(stage_u) + sum(stage_du) + terminal_cost)

# --- Initial guess: boundary rollout under zero controls
tmp_mech = deepcopy(mech)
Xb_guess = Vector{Vector{Float64}}(undef, Nb)
Xb_guess[1] = copy(x0)

for s in 1:M
    x = copy(Xb_guess[s])
    for t in 1:L
        u = zeros(m)
        Dojo.set_minimal_state!(tmp_mech, x)
        Dojo.set_input!(tmp_mech, u)
        Dojo.step_minimal_coordinates!(tmp_mech, x, u; opts=DOJO_OPTS_STEP)
        x = Vector(Dojo.get_minimal_state(tmp_mech))
    end
    Xb_guess[s+1] = x
end

for s in 1:Nb
    set_start_value.(Xb[:,s], Xb_guess[s])
end
for k in 1:(N-1)
    set_start_value.(U[:,k], 0.0)
end

# --- VectorNonlinearOracle constraint: z = [vec(Xb); vec(U)]
zvars = vcat(vec(Xb), vec(U))
@assert length(zvars) == d

oracle_set = MOI.VectorNonlinearOracle(;
    dimension = d,
    l = zeros(p),
    u = zeros(p),
    eval_f = (ret, z) -> eval_f!(oracle, ret, z),
    jacobian_structure = oracle.jac_struct,
    eval_jacobian = (ret, z) -> eval_jacobian!(oracle, ret, z),
)

@constraint(model, dyn, zvars in oracle_set)

println("Solving (multiple shooting)...")
optimize!(model)

println("\ntermination_status = ", termination_status(model))
println("objective_value    = ", try objective_value(model) catch; NaN end)

Xb_opt = value.(Xb)
Uopt   = value.(U)

println("\nFinal boundary base pos:")
println("  x = ", Xb_opt[BASE_X_IDX, Nb])
println("  y = ", Xb_opt[BASE_Y_IDX, Nb])
println("  z = ", Xb_opt[BASE_Z_IDX, Nb])

# -----------------------------
# Playback full trajectory with Uopt
# -----------------------------
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
storage = Dojo.simulate!(mech, Tsim, controller_replay!; record=true, opts=DOJO_OPTS_STEP)
vis = Dojo.visualize(mech, storage)
Dojo.render(vis)