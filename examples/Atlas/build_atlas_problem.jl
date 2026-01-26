# Build Atlas Balancing Problem for DecisionRules.jl
# 
# This script creates a stochastic multi-stage optimization problem for Atlas robot balancing,
# similar to the Goddard Rocket Control Problem example. Random perturbations are added
# at each stage to simulate external disturbances (e.g., pushes, wind, ground irregularities).
#
# Based on the working atlas_balancing.jl MPC implementation.

using JuMP
using DiffOpt
import Ipopt, HSL_jll
import MathOptInterface as MOI
using LinearAlgebra
using ForwardDiff
using Random
using JLD2

include(joinpath(@__DIR__, "atlas_utils.jl"))

# Dynamics function matching atlas_balancing.jl
function atlas_dynamics(atlas::Atlas, h::Float64, xu::T...) where {T<:Real}
    x = collect(xu[1:atlas.nx])
    u = collect(xu[atlas.nx+1:end])
    return explicit_euler(atlas, x, u, h)
end

"""
    build_atlas_subproblems(; kwargs...)

Build a multi-stage stochastic optimization problem for Atlas robot balancing.

The problem minimizes deviation from a reference state while subject to:
- Discrete-time dynamics with stochastic perturbations
- Torque limits on all joints

# Penalty arguments
- `penalty`: Legacy argument for L1 norm penalty (backwards compatible)
- `penalty_l1`: Penalty for L1 norm (NormOneCone). If provided, creates L1 deviation constraint.
- `penalty_l2`: Penalty for L2 norm (SecondOrderCone). If provided, creates L2 deviation constraint.
- If both `penalty_l1` and `penalty_l2` are provided, both norms are used with separate penalties.

Returns the components needed for DecisionRules.jl training:
- subproblems: Vector of JuMP models for each stage
- state_params_in: Input state parameters for each stage
- state_params_out: Output state parameters (targets) for each stage  
- initial_state: Initial state vector
- uncertainty_samples: Random perturbation samples for each stage
"""
function build_atlas_subproblems(;
    atlas::Atlas = Atlas(),
    x_ref::Union{Nothing, Vector{Float64}} = nothing,
    u_ref::Union{Nothing, Vector{Float64}} = nothing,
    h::Float64 = 0.01,
    N::Int = 100,
    perturbation_scale::Float64 = 0.1,
    perturbation_indices::Union{Nothing, Vector{Int}} = nothing,
    num_scenarios::Int = 10,
    penalty::Float64 = 1e3,
    penalty_l1::Union{Nothing, Float64} = nothing,
    penalty_l2::Union{Nothing, Float64} = nothing,
    optimizer = nothing,
)
    # Handle penalty arguments: if penalty_l1/penalty_l2 not specified, use legacy penalty for L1
    if isnothing(penalty_l1) && isnothing(penalty_l2)
        penalty_l1 = penalty
    end
    
    # Load reference state if not provided
    if isnothing(x_ref) || isnothing(u_ref)
        @load joinpath(@__DIR__, "atlas_ref.jld2") x_ref u_ref
    end
    
    # Default optimizer
    if isnothing(optimizer)
        optimizer = () -> DiffOpt.diff_optimizer(optimizer_with_attributes(Ipopt.Optimizer, 
            "print_level" => 0,
            "hsllib" => HSL_jll.libhsl_path,
            "linear_solver" => "ma27"
        ))
    end
    
    # Default perturbation indices: perturb velocity states
    if isnothing(perturbation_indices)
        perturbation_indices = [atlas.nq + 5]  # Single velocity perturbation like original
    end
    
    nx = atlas.nx
    nu = atlas.nu
    n_perturb = length(perturbation_indices)
    
    # Initialize outputs
    subproblems = Vector{JuMP.Model}(undef, N-1)
    state_params_in = Vector{Vector{Any}}(undef, N-1)
    state_params_out = Vector{Vector{Tuple{Any, VariableRef}}}(undef, N-1)
    uncertainty_samples = Vector{Vector{Tuple{VariableRef, Vector{Float64}}}}(undef, N-1)
    
    # Store references to state variables for analysis
    X_vars = Vector{Vector{VariableRef}}(undef, N-1)
    U_vars = Vector{Vector{VariableRef}}(undef, N-1)
    
    # Build VectorNonlinearOracle structure (same for all subproblems)
    # Following atlas_balancing.jl structure exactly
    VNO_dim = 2*nx + nu
    
    # Jacobian structure from atlas_balancing.jl:
    # Input variables: [ x[t+1,:], x[t,:], u[t,:] ]
    jacobian_structure = Tuple{Int,Int}[]
    append!(jacobian_structure, map(i -> (i, i), 1:nx))  # Identity for x_next
    for i in 1:nx, j in 1:(nx + nu)
        push!(jacobian_structure, (i, j + nx))
    end
    
    # Hessian structure
    hessian_lagrangian_structure = [
        (i, j)
        for i in nx+1:VNO_dim
        for j in nx+1:VNO_dim
    ]
    
    # Create closure for dynamics with captured atlas and h
    local_atlas = atlas
    local_h = h
    local_nx = nx
    local_nu = nu
    
    # VectorNonlinearOracle following atlas_balancing.jl exactly
    VNO = MOI.VectorNonlinearOracle(;
        dimension = VNO_dim,
        l = zeros(nx),
        u = zeros(nx),
        eval_f = (ret, z) -> begin
            # z = [x_next; x_prev; u]
            # Constraint: x_next - f(x_prev, u) = 0
            ret[1:local_nx] .= z[1:local_nx] - atlas_dynamics(local_atlas, local_h, z[local_nx+1:VNO_dim]...)
            return
        end,
        jacobian_structure = jacobian_structure,
        eval_jacobian = (ret, z) -> begin
            dyn_jac = ForwardDiff.jacobian(
                xu -> atlas_dynamics(local_atlas, local_h, xu...), 
                z[local_nx+1:VNO_dim]
            )
            jnnz = length(jacobian_structure)
            ret[1:local_nx] .= ones(local_nx)  # Identity for x_next
            ret[local_nx+1:jnnz] .= -reshape(dyn_jac', local_nx * (local_nx + local_nu))
            return
        end,
        hessian_lagrangian_structure = hessian_lagrangian_structure,
        eval_hessian_lagrangian = (ret, z, λ) -> begin
            hess = ForwardDiff.hessian(
                xu -> dot(λ, atlas_dynamics(local_atlas, local_h, xu...)), 
                z[local_nx+1:VNO_dim]
            )
            hnnz = length(hessian_lagrangian_structure)
            ret[1:hnnz] .= -reshape(hess, hnnz)
            return
        end
    )
    
    for t in 1:N-1
        # Create subproblem model
        subproblems[t] = Model(optimizer)
        model = subproblems[t]
        
        # State variables for this stage (after dynamics)
        @variable(model, x[i=1:nx], start = x_ref[i])
        
        # Control variables
        @variable(model, -atlas.torque_limits[i] <= u[i=1:nu] <= atlas.torque_limits[i], start = u_ref[i])
        
        # Target state parameters (what the policy should output)
        @variable(model, target_x[i=1:nx] ∈ MOI.Parameter(x_ref[i]))
        
        # Perturbation parameters (stochastic disturbances applied to previous state)
        @variable(model, w[i=1:n_perturb] ∈ MOI.Parameter(0.0))
        
        # Previous state parameters (input from previous stage)
        @variable(model, x_prev[i=1:nx] ∈ MOI.Parameter(x_ref[i]))
        
        # Perturbed previous state (x_prev with perturbations added)
        @variable(model, x_prev_perturbed[i=1:nx])
        
        # Deviation penalty variable (single variable for logging compatibility)
        @variable(model, norm_deficit >= 0)
        
        # Perturbation constraints: x_prev_perturbed = x_prev + w (on perturbed indices)
        for i in 1:nx
            perturb_idx = findfirst(==(i), perturbation_indices)
            if !isnothing(perturb_idx)
                @constraint(model, x_prev_perturbed[i] == x_prev[i] + w[perturb_idx])
            else
                @constraint(model, x_prev_perturbed[i] == x_prev[i])
            end
        end
        
        # Objective: minimize state deviation from reference + deviation from target
        # Create norm constraints based on penalty arguments
        use_l1 = !isnothing(penalty_l1)
        use_l2 = !isnothing(penalty_l2)
        
        if use_l1 && use_l2
            # Both L1 and L2 squared norms
            @variable(model, norm_l1 >= 0)
            @variable(model, norm_l2_sq >= 0)  # L2 squared (sum of squares)
            @constraint(model, [norm_l1; target_x .- x] in MOI.NormOneCone(nx + 1))
            @constraint(model, norm_l2_sq >= sum((target_x[i] - x[i])^2 for i in 1:nx))
            @constraint(model, norm_deficit >= penalty_l1 * norm_l1 + penalty_l2 * norm_l2_sq)
            deficit_coef = 1.0
        elseif use_l1
            # L1 norm only
            @constraint(model, [norm_deficit; target_x .- x] in MOI.NormOneCone(nx + 1))
            deficit_coef = penalty_l1
        elseif use_l2
            # L2 squared norm only (sum of squares)
            @constraint(model, norm_deficit >= sum((target_x[i] - x[i])^2 for i in 1:nx))
            deficit_coef = penalty_l2
        else
            error("At least one of penalty_l1 or penalty_l2 must be specified")
        end

        @variable(model, obj_cost >= 0)
        @constraint(model, obj_cost >= sum((x[i] - x_ref[i])^2 for i in 1:nx))
        
        @objective(model, Min, 
            obj_cost + 
            deficit_coef * norm_deficit
        )
        
        # Dynamics constraint using VectorNonlinearOracle
        # Variables: [x_next; x_prev_perturbed; u]
        vars = vcat(x, x_prev_perturbed, u)
        @constraint(model, vars in VNO)
        
        # Store variable references
        X_vars[t] = x
        U_vars[t] = u
        
        # Setup state parameters for DecisionRules.jl
        state_params_in[t] = collect(x_prev)
        state_params_out[t] = [(target_x[i], x[i]) for i in 1:nx]
        
        # Generate uncertainty samples (random perturbations)
        uncertainty_samples[t] = [(w[i], perturbation_scale * randn(num_scenarios)) for i in 1:n_perturb]
    end
    
    initial_state = copy(x_ref)
    
    return subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples, 
           X_vars, U_vars, x_ref, u_ref, atlas
end


"""
    build_atlas_deterministic_equivalent(; kwargs...)

Build a deterministic equivalent formulation for the Atlas balancing problem.
This creates a single large optimization problem instead of decomposed subproblems.

# Penalty arguments
- `penalty`: Legacy argument for L1 norm penalty (backwards compatible)
- `penalty_l1`: Penalty for L1 norm (NormOneCone). If provided, creates L1 deviation constraint.
- `penalty_l2`: Penalty for L2 norm (SecondOrderCone). If provided, creates L2 deviation constraint.
- If both `penalty_l1` and `penalty_l2` are provided, both norms are used with separate penalties.
"""
function build_atlas_deterministic_equivalent(;
    atlas::Atlas = Atlas(),
    x_ref::Union{Nothing, Vector{Float64}} = nothing,
    u_ref::Union{Nothing, Vector{Float64}} = nothing,
    h::Float64 = 0.01,
    N::Int = 100,
    perturbation_scale::Float64 = 0.1,
    perturbation_indices::Union{Nothing, Vector{Int}} = nothing,
    num_scenarios::Int = 10,
    penalty::Float64 = 1e3,
    penalty_l1::Union{Nothing, Float64} = nothing,
    penalty_l2::Union{Nothing, Float64} = nothing,
)
    # Handle penalty arguments: if penalty_l1/penalty_l2 not specified, use legacy penalty for L1
    if isnothing(penalty_l1) && isnothing(penalty_l2)
        penalty_l1 = penalty
    end
    
    # Load reference state if not provided
    if isnothing(x_ref) || isnothing(u_ref)
        @load joinpath(@__DIR__, "atlas_ref.jld2") x_ref u_ref
    end
    
    # Default perturbation indices
    if isnothing(perturbation_indices)
        perturbation_indices = [atlas.nq + 5]
    end
    
    nx = atlas.nx
    nu = atlas.nu
    n_perturb = length(perturbation_indices)
    
    # Create model
    det_equivalent = DiffOpt.diff_model(optimizer_with_attributes(Ipopt.Optimizer, 
        "print_level" => 0,
        "hsllib" => HSL_jll.libhsl_path,
        "linear_solver" => "ma27"
    ))
    
    # State and control variables for all stages
    @variable(det_equivalent, X[t=1:N, i=1:nx], start = x_ref[i])
    @variable(det_equivalent, -atlas.torque_limits[i] <= U[t=1:N-1, i=1:nu] <= atlas.torque_limits[i], start = u_ref[i])
    
    # Perturbed state variables
    @variable(det_equivalent, X_perturbed[t=1:N-1, i=1:nx], start = x_ref[i])
    
    # Target parameters (policy outputs)
    @variable(det_equivalent, target[t=1:N-1, i=1:nx] ∈ MOI.Parameter(x_ref[i]))
    
    # Perturbation parameters
    @variable(det_equivalent, w[t=1:N-1, i=1:n_perturb] ∈ MOI.Parameter(0.0))
    
    # Deviation variable
    @variable(det_equivalent, norm_deficit >= 0)
    
    # Fix initial condition
    for i in 1:nx
        fix(X[1, i], x_ref[i]; force=true)
    end
    
    # Perturbation constraints
    for t in 1:N-1
        for i in 1:nx
            perturb_idx = findfirst(==(i), perturbation_indices)
            if !isnothing(perturb_idx)
                @constraint(det_equivalent, X_perturbed[t, i] == X[t, i] + w[t, perturb_idx])
            else
                @constraint(det_equivalent, X_perturbed[t, i] == X[t, i])
            end
        end
    end
    
    # Objective
    @variable(det_equivalent, cost >= 0)
    
    # Create norm constraints based on penalty arguments
    use_l1 = !isnothing(penalty_l1)
    use_l2 = !isnothing(penalty_l2)
    deviation_expr = vec([target[t,i] - X[t+1,i] for t in 1:N-1, i in 1:nx])
    deviation_dim = (N-1) * nx
    
    if use_l1 && use_l2
        # Both L1 and L2 squared norms
        @variable(det_equivalent, norm_l1 >= 0)
        @variable(det_equivalent, norm_l2_sq >= 0)  # L2 squared (sum of squares)
        @constraint(det_equivalent, [norm_l1; deviation_expr] in MOI.NormOneCone(1 + deviation_dim))
        @constraint(det_equivalent, norm_l2_sq >= sum(deviation_expr[i]^2 for i in 1:deviation_dim))
        @constraint(det_equivalent, norm_deficit >= penalty_l1 * norm_l1 + penalty_l2 * norm_l2_sq)
        deficit_coef = 1.0
    elseif use_l1
        # L1 norm only
        @constraint(det_equivalent, [norm_deficit; deviation_expr] in MOI.NormOneCone(1 + deviation_dim))
        deficit_coef = penalty_l1
    elseif use_l2
        # L2 squared norm only (sum of squares)
        @constraint(det_equivalent, norm_deficit >= sum(deviation_expr[i]^2 for i in 1:deviation_dim))
        deficit_coef = penalty_l2
    else
        error("At least one of penalty_l1 or penalty_l2 must be specified")
    end
    
    @constraint(det_equivalent, cost >= sum((X[t,i] - x_ref[i])^2 for t in 2:N, i in 1:nx))
    @objective(det_equivalent, Min, 
        cost +
        deficit_coef * norm_deficit
    )
    
    # Build VectorNonlinearOracle (same as subproblems)
    VNO_dim = 2*nx + nu
    
    jacobian_structure = Tuple{Int,Int}[]
    append!(jacobian_structure, map(i -> (i, i), 1:nx))
    for i in 1:nx, j in 1:(nx + nu)
        push!(jacobian_structure, (i, j + nx))
    end
    
    hessian_lagrangian_structure = [
        (i, j)
        for i in nx+1:VNO_dim
        for j in nx+1:VNO_dim
    ]
    
    local_atlas = atlas
    local_h = h
    local_nx = nx
    local_nu = nu
    
    VNO = MOI.VectorNonlinearOracle(;
        dimension = VNO_dim,
        l = zeros(nx),
        u = zeros(nx),
        eval_f = (ret, z) -> begin
            ret[1:local_nx] .= z[1:local_nx] - atlas_dynamics(local_atlas, local_h, z[local_nx+1:VNO_dim]...)
            return
        end,
        jacobian_structure = jacobian_structure,
        eval_jacobian = (ret, z) -> begin
            dyn_jac = ForwardDiff.jacobian(
                xu -> atlas_dynamics(local_atlas, local_h, xu...), 
                z[local_nx+1:VNO_dim]
            )
            jnnz = length(jacobian_structure)
            ret[1:local_nx] .= ones(local_nx)
            ret[local_nx+1:jnnz] .= -reshape(dyn_jac', local_nx * (local_nx + local_nu))
            return
        end,
        hessian_lagrangian_structure = hessian_lagrangian_structure,
        eval_hessian_lagrangian = (ret, z, λ) -> begin
            hess = ForwardDiff.hessian(
                xu -> dot(λ, atlas_dynamics(local_atlas, local_h, xu...)), 
                z[local_nx+1:VNO_dim]
            )
            hnnz = length(hessian_lagrangian_structure)
            ret[1:hnnz] .= -reshape(hess, hnnz)
            return
        end
    )
    
    # Dynamics constraints
    for t in 1:N-1
        vars = vcat(X[t+1, :], X_perturbed[t, :], U[t, :])
        @constraint(det_equivalent, vars in VNO)
    end
    
    # Generate uncertainty samples
    uncertainty_samples = Vector{Vector{Tuple{VariableRef, Vector{Float64}}}}(undef, N-1)
    for t in 1:N-1
        uncertainty_samples[t] = [(w[t, i], perturbation_scale * randn(num_scenarios)) for i in 1:n_perturb]
    end
    
    # State parameters for DecisionRules.jl
    state_params_in = [collect(X[t, :]) for t in 1:N-1]
    state_params_out = [[(target[t,i], X[t+1,i]) for i in 1:nx] for t in 1:N-1]
    
    initial_state = copy(x_ref)
    
    return det_equivalent, state_params_in, state_params_out, initial_state, uncertainty_samples,
           X, U, x_ref, u_ref, atlas
end
