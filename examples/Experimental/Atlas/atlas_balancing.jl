using Revise
using JLD2
using SparseArrays
using MathOptInterface.Nonlinear.SymbolicAD
import MathOptInterface as MOI
using JuMP, Ipopt, HSL_jll
using LinearAlgebra

# include(joinpath(@__DIR__, "../mpc_utils.jl"))
include(joinpath(@__DIR__, "atlas_utils.jl"))
include(joinpath(@__DIR__, "atlas_visualization.jl"))

# Setup model and visualizer
atlas = Atlas();
vis = Visualizer();
mvis = init_visualizer(atlas, vis)

# Load balanced reference
@load joinpath(@__DIR__, "atlas_ref.jld2") x_ref u_ref;
visualize!(atlas, mvis, x_ref)

# Calculate discrete dynamics for a balanced position
# h = 0.01;
# Ad = FD.jacobian(x->rk4(atlas, x, u_ref, h), x_ref);
# Bd = FD.jacobian(u->rk4(atlas, x_ref, u, h), u_ref);

h = 0.01
@time rk4(atlas, x_ref, u_ref, h)

# # Simulate
N = 300;
X = [zeros(atlas.nx) for _ = 1:N];
U = zeros(atlas.nu)
X[1] = deepcopy(x_ref);
X[1][atlas.nq + 5] = 1.3; # Perturb i.c.

# Run simulation
@time for k = 1:N - 1
    # Integrate
    global X[k + 1] = rk4(atlas, X[k], u_ref, h)
end

animate!(atlas, mvis, X, Δt=h);


# """
#     memoize_dynamics_and_jacobian(foo, n_in, n_out)

# Returns an array of length `n_out`, where each element is a tuple 
# `(scalar_fun, grad_fun!)`.

# - `scalar_fun(x...)` returns the i-th component of `foo(x...)` in normal calls (Float64). 
# - In Dual/AD mode, it returns the i-th component but triggers a single Jacobian evaluation and caches it.

# - `grad_fun!(g, x...)` writes into `g` the partial derivatives of that i-th component w.r.t. x.

# All `n_out` scalar functions share the same cached value of the *entire* vector `foo(x...)`
# and the *entire* Jacobian for the last input `x...`.
# """
# function memoize_dynamics_and_jacobian(foo::Function, n_in::Int, n_out::Int)
#     # Cache for normal (Float64) evaluation
#     last_x_f   = nothing
#     last_f     = nothing  # Vector{Float64} from foo(x)

#     # Cache for AD (Dual) evaluation
#     last_x_J   = nothing
#     last_f_J   = nothing  # Vector{Dual} from foo(x)
#     last_J     = nothing  # Matrix of partials (size n_out x n_in)

#     # Optional: pre-make a JacobianConfig if you like:
#     # x0 = zeros(n_in)
#     # jac_cfg = ForwardDiff.JacobianConfig(foo, x0)

#     # This local function returns the i-th scalar output for the given x...
#     # In normal Float64 calls, it returns last_f[i].
#     # In AD calls (Dual etc.), it ensures last_J is computed and returns last_f_J[i].
#     function f_i(i, x::Vararg{T}) where {T<:Real}
#         if T == Float64
#             # Normal evaluation mode
#             if x !== last_x_f
#                 last_x_f = x
#                 last_f   = foo(x...)
#             end
#             return last_f[i]
#         else
#             # Dual/AD evaluation mode
#             if x !== last_x_J
#                 last_x_J = x
#                 # Evaluate all outputs
#                 local x_vec = collect(x)
#                 last_f_J    = foo(x_vec...)

#                 # Evaluate the entire Jacobian in one pass
#                 # last_J = ForwardDiff.jacobian(z -> foo(z...), x_vec, jac_cfg)
#                 last_J = ForwardDiff.jacobian(z -> foo(z...), x_vec)
#             end
#             # Return the i-th output (primal part).
#             return last_f_J[i]
#         end
#     end

#     # Now build an array of (scalar_fun, gradient_callback!) for each output dimension
#     result = Vector{Tuple{Function, Function}}(undef, n_out)
#     for i in 1:n_out
#         # 1) The scalar function to pass to JuMP
#         scalar_fun = (args...) -> f_i(i, args...)

#         # 2) The gradient callback that JuMP/Ipopt calls: we fill `g` with partial derivatives
#         # grad_fun! = (g, args...) -> begin
#         #     # Make sure last_J is up to date for this x...
#         #     _ = f_i(i, args...)  # triggers AD if needed
#         #     # Now fill g with row i of last_J
#         #     @inbounds for col in 1:length(args)
#         #         g[col] = last_J[i, col]
#         #     end
#         #     return
#         # end
#         grad_fun! = (g, args...) -> begin
#         if last_J === nothing
#             # We haven't computed the Jacobian yet, or we are in reverse mode.
#             # Manually call ForwardDiff now with `args` to get it:
#             xvec = collect(args)
#             # Evaluate the function for the primal:
#             last_f_J = foo(xvec...)  # store if needed
#             last_J   = ForwardDiff.jacobian(z -> foo(z...), xvec)
#         end
#         # Now fill g with row i of last_J
#         @inbounds for col in 1:length(args)
#             g[col] = last_J[i, col]
#         end
#         return
#     end

#         result[i] = (scalar_fun, grad_fun!)
#     end
#     return result
# end


function atlas_dynamics(xu::T...) where {T<:Real}
    h = 0.01
    x = collect(xu[1:atlas.nx])
    u = collect(xu[atlas.nx+1:end])
    #return rk4(atlas, x, u, h)
    return explicit_euler(atlas, x, u, h)
end

# Lets define a jump model that solves the MPC problem
# min ∑_t ||x_t - x_ref||_2^2
# s.t. x_{t+1} = rk4(x_t, u_t) : this needs to be a jump operator for which we will need to define a method to calculate the jacobian and hessian
#      u_min <= u_t <= u_max

function build_and_solve_mpc(atlas_obj::Atlas, x_ref::Vector{Float64}, X_start::Vector{Float64}, N::Int; optimizer=optimizer_with_attributes(Ipopt.Optimizer, 
        # "print_level" => 0,
        "hsllib" => HSL_jll.libhsl_path,
        "linear_solver" => "MA27",
        "max_iter" => 20,
    )
)
    model = Model()
    set_optimizer(model, optimizer)

    # For demonstration, let's define:
    #   x[t=1:N, 1:atlas.nx]
    #   u[t=1:N-1, 1:atlas.nu]
    @variable(model, x[1:N, 1:atlas_obj.nx])
    @variable(model, -atlas_obj.torque_limits[i] <= u[t=1:N-1,i=1:atlas.nu] <= atlas.torque_limits[i])
    # ^ adapt the indexing if torque_limits is an array of length nu, etc.

    # Objective: sum of squared error from x_ref
    @objective(model, Min, sum( (x[t,j] - x_ref[j])^2 for t in 2:N for j in 1:atlas_obj.nx ) )

    # Build the memoized function array for the multi-output dynamics
    # so we can call the i-th dimension of next state individually
    #dyn_ops = memoize_dynamics_and_jacobian(atlas_dynamics,
    #                                        atlas_obj.nx + atlas_obj.nu,
    #                                        atlas_obj.nx)

    VNO_dim = 2*atlas_obj.nx + atlas_obj.nu
    # The input variable to this function:
    # [ x[t+1,:], x[t,:], u[t,:] ]
    jacobian_structure = Tuple{Int,Int}[]
    append!(jacobian_structure, map(i -> (i,i), 1:atlas_obj.nx))
    for i in 1:atlas_obj.nx, j in 1:(atlas_obj.nx + atlas_obj.nu)
        push!(jacobian_structure, (i, j + atlas_obj.nx))
    end
    hessian_lagrangian_structure = [
        (i, j)
        for i in atlas_obj.nx+1:VNO_dim
        for j in atlas_obj.nx+1:VNO_dim
    ]
    VNO = MOI.VectorNonlinearOracle(;
        dimension = VNO_dim,
        l = zeros(atlas_obj.nx),
        u = zeros(atlas_obj.nx),
        eval_f = (ret, x) -> begin
            ret[1:atlas_obj.nx] .= x[1:atlas_obj.nx] - atlas_dynamics(x[atlas_obj.nx + 1:VNO_dim]...)
            return
        end,
        jacobian_structure,
        eval_jacobian = (ret, x) -> begin
            dyn_jac = ForwardDiff.jacobian(x -> atlas_dynamics(x...), x[atlas_obj.nx + 1:VNO_dim])
            jnnz = length(jacobian_structure)
            ret[1:atlas_obj.nx] .= ones(atlas_obj.nx)
            nx = atlas_obj.nx; nu = atlas_obj.nu
            ret[atlas_obj.nx+1:jnnz] .= - reshape(dyn_jac', nx * (nx + nu))
            return
        end,
        hessian_lagrangian_structure,
        eval_hessian_lagrangian = (ret, x, λ) -> begin
            hess = ForwardDiff.hessian(x -> dot(λ, atlas_dynamics(x...)), x[atlas_obj.nx + 1:VNO_dim])
            hnnz = length(hessian_lagrangian_structure)
            ret[1:hnnz] .= - reshape(hess, hnnz)
            return
        end
    )
    for t in 1:(N-1)
        vars = vcat(x[t+1,:], x[t, :], u[t, :])
        JuMP.@constraint(model, vars in VNO)
    end

    # Add constraints: x[t+1] = f(x[t], u[t]) for t=1 to N-1
    #for t in 1:(N-1), i in 1:atlas_obj.nx
    #    scalar_fun, grad_fun! = dyn_ops[i]
    #    op = add_nonlinear_operator(
    #        model,
    #        atlas_obj.nx + atlas_obj.nu,  # number of inputs
    #        scalar_fun,                   # f_i
    #        grad_fun!,                    # gradient callback
    #        name = Symbol("dyn_$(t)_$(i)")
    #    )
    #    # Bind the operator to the constraint x[t+1, i] == f_i( [x[t,:]; u[t,:]]... )
    #    @constraint(model, x[t+1,i] == op( [x[t, :]; u[t, :]]... ))
    #end

    # Possibly set an initial condition constraint: x[1, :] = x_ref
    @constraint(model, [j in 1:atlas_obj.nx], x[1,j] == X_start[j])

    # Solve
    optimize!(model)
    return model, value.(x), value.(u)
end

N = 10
X_start = deepcopy(x_ref)
X_start[atlas.nq + 5] = 1.3
model, X_solve, U_solve = build_and_solve_mpc(atlas, x_ref, X_start, N; optimizer=optimizer_with_attributes(Ipopt.Optimizer, 
        # "print_level" => 0,
        "linear_solver" => "ma97",
        #"hessian_approximation" => "limited-memory",
        #"max_iter" => 20,
        "mu_target" => 1e-8,
        "print_user_options" => "yes",
    )
)
termination_status(model)

# Now we can simulate the system with the controls U_solve
X = [zeros(atlas.nx) for _ = 1:N];
X[1] = deepcopy(x_ref);
X[1][atlas.nq + 5] = 1.3; # Perturb i.c.

for k = 1:N - 1
    X[k + 1] = rk4(atlas, X[k], U_solve[k, :], h)
end

animate!(atlas, mvis, X, Δt=h);

error = sum( norm(X_solve[t,:] .- X[t]) for t in 1:N )

# # Set up cost matrices (hand-tuned)
# Q = spdiagm([1e3*ones(12); repeat([1e1; 1e1; 1e3], 3); 1e1*ones(8); 1e2*ones(12); repeat([1; 1; 1e2], 3); 1*ones(8)]);
# R = spdiagm(1e-3*ones(atlas.nu));

# # Calculate infinite-horizon LQR cost-to-go and gain matrices
# K, Qf = ihlqr(Ad, Bd, Q, R, Q, max_iters = 1000);

# # Define additional constraints for the QP (just torques for Atlas)
# horizon = 2;
# A_torque = kron(I(horizon), [I(atlas.nu) zeros(atlas.nu, atlas.nx)]);
# l_torque = repeat(-atlas.torque_limits - u_ref, horizon);
# u_torque = repeat(atlas.torque_limits - u_ref, horizon);

# # Setup QP
# H, g, A, l, u, g_x0, lu_x0 = gen_condensed_mpc_qp(Ad, Bd, Q, R, Qf, horizon, A_torque, l_torque, u_torque, K);

# # Setup solver
# m = ReLUQP.setup(H, g, A, l, u, verbose = false, eps_primal=1e-2, eps_dual=1e-2, max_iters=10, iters_btw_checks=1);

# # Simulate
# N = 300;
# X = [zeros(atlas.nx) for _ = 1:N];
# U = [zeros(atlas.nu) for _ = 1:N];
# X[1] = deepcopy(x_ref);
# X[1][atlas.nq + 5] = 1.3; # Perturb i.c.

# # Warmstart solver
# Δx = X[1] - x_ref;
# ReLUQP.update!(m, g = g + g_x0*Δx, l = l + lu_x0*Δx, u = u + lu_x0*Δx);
# m.opts.max_iters = 4000;
# m.opts.check_convergence = false;
# ReLUQP.solve(m);
# m.opts.max_iters = 10;

# Run simulation
# for k = 1:N - 1
    # Get error
    # global Δx = X[k] - x_ref

    # Update solver
    # ReLUQP.update!(m, g = g + g_x0*Δx, l = l + lu_x0*Δx, u = u + lu_x0*Δx)

    # Solve and get controls
    # results = ReLUQP.solve(m)
    # global U[k] = results.x[1:atlas.nu] - K*Δx

    # Integrate
    # global X[k + 1] = rk4(atlas, X[k], clamp.(u_ref + U[k], -atlas.torque_limits, atlas.torque_limits), h)
# end

#X = [value.(x[t,:]) for t=1:N]
#X = [X_solve[t, :] for t=1:N]
#
#animate!(atlas, mvis, X, Δt=h);
#readline()
#
#
################
#include(joinpath(@__DIR__, "rigidbodyutils.jl"))
#include(joinpath(@__DIR__, "atlas_utils.jl"))
#
#model = Model()
#optimizer=optimizer_with_attributes(Ipopt.Optimizer, 
#    # "print_level" => 0,
#    "hsllib" => HSL_jll.libhsl_path,
#    "linear_solver" => "MA27"
#)
#set_optimizer(model, optimizer)
#
#@variable(model, x[1:N, 1:atlas.nx])
#@variable(model, -atlas.torque_limits[i] <= u[t=1:N-1,i=1:atlas.nu] <= atlas.torque_limits[i])
#
#dyn_result = dynamics(atlas, convert.(NonlinearExpr, x[1, :]), convert.(NonlinearExpr,u[1, :]))
#
#function atlas_dynamics(x, u)
#    h = 0.01
#    return rk4(atlas, x, u, h)
#end
#
##function build_and_solve_mpc(atlas_obj::Atlas, x_ref::Vector{Float64}, X_start::Vector{Float64}, N::Int; optimizer=optimizer_with_attributes(Ipopt.Optimizer, 
##        # "print_level" => 0,
##        "hsllib" => HSL_jll.libhsl_path,
##        "linear_solver" => "MA27"
##    )
##)
##    model = Model()
##    set_optimizer(model, optimizer)
##
##    # For demonstration, let's define:
##    #   x[t=1:N, 1:atlas.nx]
##    #   u[t=1:N-1, 1:atlas.nu]
##    @variable(model, x[1:N, 1:atlas_obj.nx])
##    @variable(model, -atlas_obj.torque_limits[i] <= u[t=1:N-1,i=1:atlas.nu] <= atlas_obj.torque_limits[i])
##    # ^ adapt the indexing if torque_limits is an array of length nu, etc.
##
##    # Objective: sum of squared error from x_ref
##    @objective(model, Min, sum( (x[t,j] - x_ref[j])^2 for t in 2:N for j in 1:atlas_obj.nx ) )
##
##    for t in 1:(N-1)
##        # Bindconstraint x[t+1, i] == f_i( [x[t,:]; u[t,:]]... )
##        @constraint(model, x[t+1, :] == atlas_dynamics( convert.(NonlinearExpr, x[t, :]), convert.(NonlinearExpr,u[t, :])))
##    end
##
##    # Possibly set an initial condition constraint: x[1, :] = x_ref
##    @constraint(model, [j in 1:atlas_obj.nx], x[1,j] == X_start[j])
##
##    # Solve
##    optimize!(model)
##    return model, value.(x), value.(u)
##end
#
#N = 2
#X_start = deepcopy(x_ref)
#X_start[atlas.nq + 5] = 1.3
#model, X_solve, U_solve = build_and_solve_mpc(atlas, x_ref, X_start, N; optimizer=optimizer_with_attributes(Ipopt.Optimizer, 
#        # "print_level" => 0,
#        "linear_solver" => "ma97",
#        "hessian_approximation" => "limited-memory",
#        "max_iter" => 300,
#        "mu_target" => 1e-8,
#        "print_info_string" => "yes",
#    )
#)
#termination_status(model)
#
## Now we can simulate the system with the controls U_solve
#X = [zeros(atlas.nx) for _ = 1:N];
#X[1] = deepcopy(x_ref);
#X[1][atlas.nq + 5] = 1.3; # Perturb i.c.
#
#for k = 1:N - 1
#    X[k + 1] = rk4(atlas, X[k], U_solve[k, :], h)
#end
#
#animate!(atlas, mvis, X, Δt=h);
#
#error = sum( norm(X_solve[t,:] .- X[t]) for t in 1:N )
#
#
#function foo!(q::AbstractVector{T}) where {T}
#    q .= zero(T)
#end
