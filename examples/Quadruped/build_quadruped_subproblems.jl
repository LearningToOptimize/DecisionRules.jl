using JuMP
import Ipopt, HSL_jll
using DiffOpt
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface

using Dojo
using DojoEnvironments

# Base coordinate indices (minimal coordinates as exposed by Dojo)
const BASE_X_IDX = 1
const BASE_Y_IDX = 2
const BASE_Z_IDX = 3

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
    build_quadruped_subproblems(; N=21, dt=0.02, path_waypoints=..., u_max_abs=40.0,
                                  w_path=50.0, w_nn=10.0, w_u=1e-2, w_z=10.0)

Create per-period subproblems for the quadruped using a linearized dynamics
surrogate `x⁺ = x + dt * B*u` with `B` taken at the initial state. Each subproblem:
- has parameters for incoming state `x_prev` (from rollout)
- has parameters for desired state `x_target` (from the neural network)
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
)
    mech = DojoEnvironments.get_mechanism(:quadruped)
    x0 = Vector(Dojo.get_minimal_state(mech))
    m = Dojo.input_dimension(mech)
    n = length(x0)

    # Single-point linearization used for all stages (lightweight surrogate)
    _, B0 = Dojo.get_minimal_gradients!(mech, x0, zeros(m))

    # Pre-compute path targets over the horizon
    path_targets = interpolate_path(path_waypoints, N)
    z_nom = x0[BASE_Z_IDX]  # encourage staying near start height

    subproblems = Vector{JuMP.Model}(undef, N - 1)
    state_params_in = Vector{Vector{Any}}(undef, N - 1)
    state_params_out = Vector{Vector{Tuple{Any, VariableRef}}}(undef, N - 1)
    uncertainty_samples = [Vector{Tuple{VariableRef, Vector{Float64}}}() for _ in 1:(N - 1)]

    for k in 1:(N - 1)
        model = DiffOpt.diff_model(optimizer_with_attributes(Ipopt.Optimizer,
            "print_level" => 0,
            "hsllib" => HSL_jll.libhsl_path,
            "linear_solver" => "ma27",
        ))

        @variable(model, x[1:n])
        @variable(model, -u_max_abs <= u[1:m] <= u_max_abs)
        @variable(model, x_prev[1:n] in MOI.Parameter.(0.0))
        @variable(model, x_target[1:n] in MOI.Parameter.(0.0))
        @variable(model, norm_deficit >= 0)
        @variable(model, stage_cost >= 0)

        # Linearized step: x⁺ ≈ x_prev + dt * B0 * u
        @constraint(model, [i in 1:n], x[i] == x_prev[i] + dt * sum(B0[i, j] * u[j] for j in 1:m))

        # L1 slack on target tracking, similar to rocket example
        @constraint(model, norm_deficit >= sum(abs(x[i] - x_target[i]) for i in 1:n))

        x_path, y_path = path_targets[k + 1]  # target for the next knot (consistent with step k)

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
