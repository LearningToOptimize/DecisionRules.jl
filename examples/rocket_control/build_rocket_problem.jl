using Flux
using DecisionRules
using Random
using Statistics

using JuMP
using Ipopt: Ipopt
using DiffOpt

# Dimensionless parameters from COPS3 (https://www.mcs.anl.gov/~more/cops/cops3.pdf).
function build_rocket_problem(;
    h_0=1,                      # Initial height
    v_0=0,                      # Initial velocity
    m_0=1.0,                    # Initial mass
    m_T=0.6,                    # Final mass
    g_0=1,                      # Gravity at the surface
    h_c=500,                    # Used for drag
    c=0.5 * sqrt(g_0 * h_0),    # Thrust-to-fuel mass
    D_c=0.5 * 620 * m_0 / g_0,  # Drag scaling
    u_t_max=3.5 * g_0 * m_0,    # Maximum thrust
    T=1_000,                    # Number of time steps
    Δt=0.2 / T,                 # Time per discretized step
    penalty=10,                 # Penalty for violating target (legacy, used for L1 if penalty_l1/penalty_l2 not specified)
    penalty_l1=nothing,         # Penalty for L1 norm (NormOneCone)
    penalty_l2=nothing,         # Penalty for L2 norm (SecondOrderCone)
    num_scenarios=10,           # Number of samples
)
    if isnothing(penalty_l1) && isnothing(penalty_l2)
        penalty_l1 = penalty
    end

    det_equivalent = DiffOpt.diff_model(
        optimizer_with_attributes(
            Ipopt.Optimizer,
            "print_level" => 0,
            "linear_solver" => "mumps",
        ),
    )

    # State and control variables
    @variable(det_equivalent, x_v[1:T], start = v_0)
    @variable(det_equivalent, x_h[1:T] >= 0, start = h_0)
    @variable(det_equivalent, x_m[1:T] >= m_T, start = m_0)
    @variable(det_equivalent, 0 <= u_t[1:(T - 1)] <= u_t_max, start = 0)

    # Physical-state target parameters (NN predicts these)
    @variable(det_equivalent, target_v[1:(T - 1)] ∈ MOI.Parameter.(0.0))
    @variable(det_equivalent, target_h[1:(T - 1)] ∈ MOI.Parameter.(0.0))
    @variable(det_equivalent, target_m[1:(T - 1)] ∈ MOI.Parameter.(0.0))

    @variable(det_equivalent, w[1:(T - 1)] ∈ MOI.Parameter.(1.0))

    # Initial-state parameters (stage 1 input to the NN)
    @variable(det_equivalent, x_v_init ∈ MOI.Parameter(v_0))
    @variable(det_equivalent, x_h_init ∈ MOI.Parameter(h_0))
    @variable(det_equivalent, x_m_init ∈ MOI.Parameter(m_0))

    # Boundary conditions
    fix(x_v[1], v_0; force=true)
    fix(x_h[1], h_0; force=true)
    fix(x_m[1], m_0; force=true)

    # Per-stage target-deviation penalty
    use_l1 = !isnothing(penalty_l1)
    use_l2 = !isnothing(penalty_l2)
    @variable(det_equivalent, norm_deficit[1:(T - 1)] >= 0)

    for t in 1:(T - 1)
        deviation_expr = [
            target_v[t] - x_v[t + 1],
            target_h[t] - x_h[t + 1],
            target_m[t] - x_m[t + 1],
        ]
        if use_l1 && use_l2
            @variable(det_equivalent, _norm_l1 >= 0)
            @variable(det_equivalent, _norm_l2_sq >= 0)
            @constraint(det_equivalent, [_norm_l1; deviation_expr] in MOI.NormOneCone(4))
            @constraint(det_equivalent, _norm_l2_sq >= sum(d^2 for d in deviation_expr))
            @constraint(
                det_equivalent,
                norm_deficit[t] >= penalty_l1 * _norm_l1 + penalty_l2 * _norm_l2_sq,
            )
        elseif use_l1
            @constraint(
                det_equivalent,
                [norm_deficit[t]; deviation_expr] in MOI.NormOneCone(4),
            )
        elseif use_l2
            @constraint(
                det_equivalent,
                norm_deficit[t] >= sum(d^2 for d in deviation_expr),
            )
        else
            error("At least one of penalty_l1 or penalty_l2 must be specified")
        end
    end

    deficit_coef = if use_l1 && use_l2
        1.0
    elseif use_l1
        penalty_l1
    else
        penalty_l2
    end

    @objective(
        det_equivalent,
        Min,
        -x_h[T] + deficit_coef * sum(norm_deficit),
    )

    # Aerodynamic drag and gravity
    D(x_h, x_v) = D_c * x_v^2 * exp(-h_c * (x_h - h_0) / h_0)
    g(x_h) = g_0 * (h_0 / x_h)^2

    # Forward Euler discretization
    ddt(x::Vector, t::Int) = (x[t] - x[t - 1]) / Δt

    @constraint(det_equivalent, [t in 2:T], ddt(x_h, t) == x_v[t - 1])
    @constraint(
        det_equivalent,
        [t in 2:T],
        ddt(x_v, t) ==
            (u_t[t - 1] - D(x_h[t - 1], x_v[t - 1])) / x_m[t - 1] - g(x_h[t - 1]) - w[t - 1],
    )
    @constraint(det_equivalent, [t in 2:T], ddt(x_m, t) == -u_t[t - 1] / c)

    # Stochastic wind disturbance samples
    uncertainty_samples = Vector{Vector{Tuple{VariableRef,Vector{Float64}}}}(undef, T - 1)
    for t in 1:(T - 1)
        uncertainty_samples[t] = [(w[t], randn(num_scenarios))]
    end

    # Stage 1 uses parameter proxies; stages 2+ use optimization variables
    state_params_in = Vector{Vector{VariableRef}}(undef, T - 1)
    state_params_in[1] = VariableRef[x_v_init, x_h_init, x_m_init]
    for t in 2:(T - 1)
        state_params_in[t] = VariableRef[x_v[t], x_h[t], x_m[t]]
    end

    return det_equivalent,
    state_params_in,
    [
        [
            (target_v[t], x_v[t + 1]),
            (target_h[t], x_h[t + 1]),
            (target_m[t], x_m[t + 1]),
        ] for t in 1:(T - 1)
    ],
    Float64[v_0, h_0, m_0],
    uncertainty_samples,
    x_v,
    x_h,
    x_m,
    u_t_max
end

function build_rocket_subproblems(;
    h_0=1,                      # Initial height
    v_0=0,                      # Initial velocity
    m_0=1.0,                    # Initial mass
    m_T=0.6,                    # Final mass
    g_0=1,                      # Gravity at the surface
    h_c=500,                    # Used for drag
    c=0.5 * sqrt(g_0 * h_0),    # Thrust-to-fuel mass
    D_c=0.5 * 620 * m_0 / g_0,  # Drag scaling
    u_t_max=3.5 * g_0 * m_0,    # Maximum thrust
    T=1_000,                    # Number of time steps
    Δt=0.2 / T,                 # Time per discretized step
    penalty=10,                 # Penalty for violating target (legacy, used for L1 if penalty_l1/penalty_l2 not specified)
    penalty_l1=nothing,         # Penalty for L1 norm (NormOneCone)
    penalty_l2=nothing,         # Penalty for L2 norm (SecondOrderCone)
    final_u_state=0.0,          # Final state of the control
    num_scenarios=10,           # Number of samples
)
    if isnothing(penalty_l1) && isnothing(penalty_l2)
        penalty_l1 = penalty
    end

    subproblems = Vector{JuMP.Model}(undef, T-1)
    state_params_in = Vector{Vector{Any}}(undef, T-1)
    state_params_out = Vector{Vector{Tuple{Any,VariableRef}}}(undef, T-1)
    uncertainty_samples = Vector{Vector{Tuple{VariableRef,Vector{Float64}}}}(undef, T-1)
    heights = Vector{VariableRef}(undef, T-1)
    masses = Vector{VariableRef}(undef, T-1)
    velocities = Vector{VariableRef}(undef, T-1)
    thrusts = Vector{VariableRef}(undef, T-1)
    initial_state = [v_0, h_0, m_0]

    for t in 1:(T - 1)
        subproblems[t] = DiffOpt.diff_model(
            optimizer_with_attributes(
                Ipopt.Optimizer,
                "print_level" => 0,
                "linear_solver" => "mumps",
            ),
        )
        @variable(subproblems[t], x_v, start = v_0)
        @variable(subproblems[t], x_h >= 0, start = h_0)
        @variable(subproblems[t], x_m >= m_T, start = m_0)
        @variable(subproblems[t], 0 <= u_t <= u_t_max, start = 0)
        @variable(subproblems[t], target_v ∈ MOI.Parameter(0.0))
        @variable(subproblems[t], target_h ∈ MOI.Parameter(0.0))
        @variable(subproblems[t], target_m ∈ MOI.Parameter(0.0))
        @variable(subproblems[t], w ∈ MOI.Parameter(1.0))
        @variable(subproblems[t], norm_deficit >= 0)

        # Previous-stage state (passed as parameters for stage-wise decomposition)
        @variable(subproblems[t], x_v_prev ∈ MOI.Parameter(v_0))
        @variable(subproblems[t], x_h_prev ∈ MOI.Parameter(h_0))
        @variable(subproblems[t], x_m_prev ∈ MOI.Parameter(m_0))

        # Create norm constraints based on penalty arguments
        use_l1 = !isnothing(penalty_l1)
        use_l2 = !isnothing(penalty_l2)
        deviation_expr = [target_v - x_v, target_h - x_h, target_m - x_m]

        if use_l1 && use_l2
            # Both L1 and L2 squared norms
            @variable(subproblems[t], norm_l1 >= 0)
            @variable(subproblems[t], norm_l2_sq >= 0)  # L2 squared (sum of squares)
            @constraint(subproblems[t], [norm_l1; deviation_expr] in MOI.NormOneCone(4))
            @constraint(subproblems[t], norm_l2_sq >= sum(deviation_expr[i]^2 for i in 1:3))
            @constraint(
                subproblems[t],
                norm_deficit >= penalty_l1 * norm_l1 + penalty_l2 * norm_l2_sq
            )
            deficit_coef = 1.0
        elseif use_l1
            # L1 norm only
            @constraint(
                subproblems[t], [norm_deficit; deviation_expr] in MOI.NormOneCone(4)
            )
            deficit_coef = penalty_l1
        elseif use_l2
            # L2 squared norm only (sum of squares)
            @constraint(
                subproblems[t], norm_deficit >= sum(deviation_expr[i]^2 for i in 1:3)
            )
            deficit_coef = penalty_l2
        else
            error("At least one of penalty_l1 or penalty_l2 must be specified")
        end

        @objective(subproblems[t], Min, -x_h + deficit_coef * norm_deficit)

        D(x_h, x_v) = D_c * x_v^2 * exp(-h_c * (x_h - h_0) / h_0)
        g(x_h) = g_0 * (h_0 / x_h)^2
        ddt(x_curr, x_prev) = (x_curr - x_prev) / Δt
        @constraint(subproblems[t], ddt(x_h, x_h_prev) == x_v_prev)
        @constraint(
            subproblems[t],
            ddt(x_v, x_v_prev) ==
                (u_t - D(x_h_prev, x_v_prev)) / x_m_prev - g(x_h_prev) - w,
        )
        @constraint(subproblems[t], ddt(x_m, x_m_prev) == -u_t / c)

        uncertainty_samples[t] = [(w, randn(num_scenarios))]

        state_params_in[t] = [x_v_prev, x_h_prev, x_m_prev]
        state_params_out[t] = [(target_v, x_v), (target_h, x_h), (target_m, x_m)]
        heights[t] = x_h
        masses[t] = x_m
        velocities[t] = x_v
        thrusts[t] = u_t
    end

    return subproblems,
    state_params_in,
    state_params_out,
    initial_state,
    uncertainty_samples,
    velocities,
    heights,
    masses,
    u_t_max
end

function build_rocket_mpc(;
    h_0=1,                      # Initial height
    v_0=0,                      # Initial velocity
    m_0=1.0,                    # Initial mass
    m_T=0.6,                    # Final mass
    g_0=1,                      # Gravity at the surface
    h_c=500,                    # Used for drag
    c=0.5 * sqrt(g_0 * h_0),    # Thrust-to-fuel mass
    D_c=0.5 * 620 * m_0 / g_0,  # Drag scaling
    u_t_max=3.5 * g_0 * m_0,    # Maximum thrust
    T=1_000,                    # Number of time steps
    Δt=0.2 / T,                 # Time per discretized step
    w=[0.0; randn(T-2)],        # Wind
)
    model = Model()
    set_optimizer(
        model,
        optimizer_with_attributes(
            Ipopt.Optimizer,
            "print_level" => 0,
            "linear_solver" => "mumps",
        ),
    )

    @variable(model, x_v[1:T], start = v_0)
    @variable(model, x_h[1:T] >= 0, start = h_0)
    @variable(model, x_m[1:T] >= m_T, start = m_0)
    @variable(model, 0 <= u_t[1:T] <= u_t_max, start = 0)

    fix(x_v[1], v_0; force=true)
    fix(x_h[1], h_0; force=true)
    fix(x_m[1], m_0; force=true)
    fix(u_t[T], 0.0; force=true)

    @objective(model, Max, x_h[T])

    D(x_h, x_v) = D_c * x_v^2 * exp(-h_c * (x_h - h_0) / h_0)
    g(x_h) = g_0 * (h_0 / x_h)^2

    ddt(x::Vector, t::Int) = (x[t] - x[t - 1]) / Δt
    @constraint(model, [t in 2:T], ddt(x_h, t) == x_v[t - 1])
    @constraint(
        model,
        [t in 2:T],
        ddt(x_v, t) ==
            (u_t[t - 1] - D(x_h[t - 1], x_v[t - 1])) / x_m[t - 1] - g(x_h[t - 1]) - w[t - 1],
    )
    @constraint(model, [t in 2:T], ddt(x_m, t) == -u_t[t - 1] / c)

    optimize!(model)
    @assert is_solved_and_feasible(model) "Model solve failed. Termaition status: $(termination_status(model))"

    return value(x_h[2]), value(x_m[2]), value(x_v[2]), value(u_t[1])
end

function run_rolling_mpc_time(;
    h_0=1,                      # Initial height
    v_0=0,                      # Initial velocity
    m_0=1.0,                    # Initial mass
    m_T=0.6,                    # Final mass
    g_0=1,                      # Gravity at the surface
    h_c=500,                    # Used for drag
    c=0.5 * sqrt(g_0 * h_0),    # Thrust-to-fuel mass
    D_c=0.5 * 620 * m_0 / g_0,  # Drag scaling
    u_t_max=3.5 * g_0 * m_0,    # Maximum thrust
    T=1_000,                    # Number of time steps
    Δt=0.2 / T,                 # Time per discretized step
    w=randn(T-1),               # Actual Wind
)
    x_h = zeros(T)
    x_m = zeros(T)
    x_v = zeros(T)
    u_t = zeros(T)

    x_h[1], x_m[1], x_v[1] = h_0, m_0, v_0

    for t in 1:(T - 1)
        x_h[t + 1], x_m[t + 1], x_v[t + 1], u_t[t] = build_rocket_mpc(;
            h_0=x_h[t],
            v_0=x_v[t],
            m_0=x_m[t],
            m_T=m_T,
            g_0=g_0,
            h_c=h_c,
            c=c,
            D_c=D_c,
            u_t_max=u_t_max,
            T=T-t+1,
            Δt=Δt,
            w=[w[t]; zeros(T-t-1)],
        )
        @show t, x_h[t + 1], x_m[t + 1], x_v[t + 1], u_t[t], w[t]
    end

    u_t[T] = 0.0

    return x_h, x_m, x_v, u_t
end
