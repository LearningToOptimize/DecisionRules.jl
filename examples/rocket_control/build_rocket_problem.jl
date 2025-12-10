using Flux
using DecisionRules
using Random
using Statistics

using JuMP
import Ipopt, HSL_jll
using DiffOpt
# import Plots


# All parameters in this model have been normalized to be dimensionless, and
# they are taken from [COPS3](https://www.mcs.anl.gov/~more/cops/cops3.pdf).
function build_rocket_problem(;
    h_0 = 1,                      # Initial height
    v_0 = 0,                      # Initial velocity
    m_0 = 1.0,                    # Initial mass
    m_T = 0.6,                    # Final mass
    g_0 = 1,                      # Gravity at the surface
    h_c = 500,                    # Used for drag
    c = 0.5 * sqrt(g_0 * h_0),    # Thrust-to-fuel mass
    D_c = 0.5 * 620 * m_0 / g_0,  # Drag scaling
    u_t_max = 3.5 * g_0 * m_0,    # Maximum thrust
    T = 1_000,                    # Number of time steps
    Δt = 0.2 / T,                 # Time per discretized step
    penalty = 10,                 # Penalty for violating target
    final_u_state = 0.0,          # Final state of the control
    num_scenarios = 10,           # Number of samples
)
    # ## JuMP formulation

    # First, we create a model and choose an optimizer. Since this is a nonlinear
    # program, we need to use a nonlinear solver like Ipopt. We cannot use a linear
    # solver like HiGHS.

    det_equivalent = DiffOpt.diff_model(optimizer_with_attributes(Ipopt.Optimizer, 
        "print_level" => 0,
        "hsllib" => HSL_jll.libhsl_path,
        "linear_solver" => "ma27"
    ))

    # Next, we create our state and control variables, which are each indexed by
    # `t`. It is good practice for nonlinear programs to always provide a starting
    # solution for each variable.

    @variable(det_equivalent, x_v[1:T], start = v_0)           # Velocity
    @variable(det_equivalent, x_h[1:T] >= 0, start = h_0)           # Height
    @variable(det_equivalent, x_m[1:T] >= m_T, start = m_0)         # Mass
    @variable(det_equivalent, 0 <= u_t[1:T-1] <= u_t_max, start = 0); # Thrust
    @variable(det_equivalent, target[1:T-1] ∈ MOI.Parameter.(0.0)); # Target thrust
    @variable(det_equivalent, w[1:T-1] ∈ MOI.Parameter.(1.0));      # Wind
    @variable(det_equivalent, norm_deficit >= 0);                   # Wind
    @variable(det_equivalent, u_T ∈ MOI.Parameter(final_u_state));    # Final thrust target

    # We implement boundary conditions by fixing variables to values.

    fix(x_v[1], v_0; force = true)
    fix(x_h[1], h_0; force = true)
    fix(x_m[1], m_0; force = true)
    # fix(u_T, final_u_state; force = true)

    # The objective is to maximize altitude at end of time of flight.
    @constraint(det_equivalent, [norm_deficit; (target.-u_t[1:T-1])] in MOI.NormOneCone(T))

    @objective(det_equivalent, Min, -x_h[T] + penalty * norm_deficit)


    # Forces are defined as functions:

    D(x_h, x_v) = D_c * x_v^2 * exp(-h_c * (x_h - h_0) / h_0)
    g(x_h) = g_0 * (h_0 / x_h)^2

    # The dynamical equations are implemented as constraints.

    ddt(x::Vector, t::Int) = (x[t] - x[t-1]) / Δt
    @constraint(det_equivalent, [t in 2:T], ddt(x_h, t) == x_v[t-1])
    @constraint(
        det_equivalent,
        [t in 2:T],
        ddt(x_v, t) == (u_t[t-1] - D(x_h[t-1], x_v[t-1])) / x_m[t-1] - g(x_h[t-1]) - w[t-1],
    )
    @constraint(det_equivalent, [t in 2:T], ddt(x_m, t) == -u_t[t-1] / c)

    # uncertainty
    uncertainty_samples = Vector{Vector{Tuple{VariableRef, Vector{Float64}}}}(undef, T-1)
    # uncertainty_samples = Vector{Dict{Any, Vector{Float64}}}(undef, T-1)
    for t in 1:T-1
        # uncertainty_dict = Dict{Any, Vector{Float64}}()
        # uncertainty_dict[w[t]] = randn(num_scenarios)
        # uncertainty_samples[t] = uncertainty_dict
        uncertainty_samples[t] = [(w[t], randn(num_scenarios))]
    end

    return det_equivalent, vcat([VariableRef[u_T]], [VariableRef[i] for i in u_t[1:T-2]]), [[(target[t], u_t[t])] for t in 1:T-1], [final_u_state], uncertainty_samples, x_v, x_h, x_m, u_t_max
end


function build_rocket_mpc(;
    h_0 = 1,                      # Initial height
    v_0 = 0,                      # Initial velocity
    m_0 = 1.0,                    # Initial mass
    m_T = 0.6,                    # Final mass
    g_0 = 1,                      # Gravity at the surface
    h_c = 500,                    # Used for drag
    c = 0.5 * sqrt(g_0 * h_0),    # Thrust-to-fuel mass
    D_c = 0.5 * 620 * m_0 / g_0,  # Drag scaling
    u_t_max = 3.5 * g_0 * m_0,    # Maximum thrust
    T = 1_000,                    # Number of time steps
    Δt = 0.2 / T,                 # Time per discretized step
    w = [0.0; randn(T-2)],        # Wind
)

    # ## JuMP formulation

    # First, we create a model and choose an optimizer. Since this is a nonlinear
    # program, we need to use a nonlinear solver like Ipopt. We cannot use a linear
    # solver like HiGHS.

    model = Model()
    set_optimizer(model, optimizer_with_attributes(Ipopt.Optimizer, 
        "print_level" => 0,
        "hsllib" => HSL_jll.libhsl_path,
        "linear_solver" => "ma27"
    ))

    # Next, we create our state and control variables, which are each indexed by
    # `t`. It is good practice for nonlinear programs to always provide a starting
    # solution for each variable.

    @variable(model, x_v[1:T], start = v_0)           # Velocity
    @variable(model, x_h[1:T] >= 0, start = h_0)           # Height
    @variable(model, x_m[1:T] >= m_T, start = m_0)         # Mass
    @variable(model, 0 <= u_t[1:T] <= u_t_max, start = 0); # Thrust

    # We implement boundary conditions by fixing variables to values.

    fix(x_v[1], v_0; force = true)
    fix(x_h[1], h_0; force = true)
    fix(x_m[1], m_0; force = true)
    fix(u_t[T], 0.0; force = true)

    # The objective is to maximize altitude at end of time of flight.

    @objective(model, Max, x_h[T])

    # Forces are defined as functions:

    D(x_h, x_v) = D_c * x_v^2 * exp(-h_c * (x_h - h_0) / h_0)
    g(x_h) = g_0 * (h_0 / x_h)^2

    # The dynamical equations are implemented as constraints.

    ddt(x::Vector, t::Int) = (x[t] - x[t-1]) / Δt
    @constraint(model, [t in 2:T], ddt(x_h, t) == x_v[t-1])
    @constraint(
        model,
        [t in 2:T],
        ddt(x_v, t) == (u_t[t-1] - D(x_h[t-1], x_v[t-1])) / x_m[t-1] - g(x_h[t-1]) - w[t-1],
    )
    @constraint(model, [t in 2:T], ddt(x_m, t) == -u_t[t-1] / c);

    # Now we optimize the model and check that we found a solution:

    optimize!(model)
    @assert is_solved_and_feasible(model) "Model solve failed. Termaition status: $(termination_status(model))"
    
    return value(x_h[2]), value(x_m[2]), value(x_v[2]), value(u_t[1])

end

function run_rolling_mpc_time(;
    h_0 = 1,                      # Initial height
    v_0 = 0,                      # Initial velocity
    m_0 = 1.0,                    # Initial mass
    m_T = 0.6,                    # Final mass
    g_0 = 1,                      # Gravity at the surface
    h_c = 500,                    # Used for drag
    c = 0.5 * sqrt(g_0 * h_0),    # Thrust-to-fuel mass
    D_c = 0.5 * 620 * m_0 / g_0,  # Drag scaling
    u_t_max = 3.5 * g_0 * m_0,    # Maximum thrust
    T = 1_000,                    # Number of time steps
    Δt = 0.2 / T,                 # Time per discretized step
    w = randn(T-1),               # Actual Wind
)

    x_h = zeros(T)
    x_m = zeros(T)
    x_v = zeros(T)
    u_t = zeros(T)

    x_h[1], x_m[1], x_v[1] = h_0, m_0, v_0

    for t in 1:T-1
        x_h[t+1], x_m[t+1], x_v[t+1], u_t[t] = build_rocket_mpc(
            h_0 = x_h[t],
            v_0 = x_v[t],
            m_0 = x_m[t],
            m_T = m_T,
            g_0 = g_0,
            h_c = h_c,
            c = c,
            D_c = D_c,
            u_t_max = u_t_max,
            T = T-t+1,
            Δt = Δt,
            w = [w[t]; zeros(T-t-1)],
        )
        @show t, x_h[t+1], x_m[t+1], x_v[t+1], u_t[t], w[t]
    end

    u_t[T] = 0.0

    return x_h, x_m, x_v, u_t
end