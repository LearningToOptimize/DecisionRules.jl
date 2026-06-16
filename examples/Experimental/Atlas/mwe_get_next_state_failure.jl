using JuMP
using DiffOpt
import Ipopt, HSL_jll
import MathOptInterface as MOI
using Statistics
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "build_atlas_problem.jl"))

function sample_uncertainty(uncertainty_stage::Vector{Tuple{VariableRef, Vector{Float64}}})
    return [(p, vals[1]) for (p, vals) in uncertainty_stage]
end

function set_stage_data!(
    subproblem::JuMP.Model,
    state_param_in::Vector{Any},
    state_param_out::Vector{Tuple{Any, VariableRef}},
    uncertainty::Vector{Tuple{VariableRef, Float64}},
    state_in::Vector{Float64},
    state_out_target::Vector{Float64},
)
    for (i, state_var) in enumerate(state_param_in)
        set_parameter_value(state_var, state_in[i])
    end
    for (uncertainty_param, uncertainty_value) in uncertainty
        set_parameter_value(uncertainty_param, uncertainty_value)
    end
    for i in eachindex(state_param_out)
        set_parameter_value(state_param_out[i][1], state_out_target[i])
    end
    return
end

function solve_and_get_next_state!(
    subproblem::JuMP.Model,
    state_param_in::Vector{Any},
    state_param_out::Vector{Tuple{Any, VariableRef}},
    uncertainty::Vector{Tuple{VariableRef, Float64}},
    state_in::Vector{Float64},
    state_out_target::Vector{Float64},
)
    set_stage_data!(
        subproblem,
        state_param_in,
        state_param_out,
        uncertainty,
        state_in,
        state_out_target,
    )
    optimize!(subproblem)
    term = termination_status(subproblem)
    if !(term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED))
        error("Unexpected solver status: $term")
    end
    return [value(state_param_out[i][2]) for i in eachindex(state_param_out)]
end

function scalar_output(
    subproblem::JuMP.Model,
    state_param_in::Vector{Any},
    state_param_out::Vector{Tuple{Any, VariableRef}},
    uncertainty::Vector{Tuple{VariableRef, Float64}},
    state_in::Vector{Float64},
    state_out_target::Vector{Float64},
    Δy::Vector{Float64},
)
    y = solve_and_get_next_state!(
        subproblem,
        state_param_in,
        state_param_out,
        uncertainty,
        state_in,
        state_out_target,
    )
    return dot(Δy, y), y
end

function reverse_vjp(
    subproblem::JuMP.Model,
    state_param_in::Vector{Any},
    state_param_out::Vector{Tuple{Any, VariableRef}},
    uncertainty::Vector{Tuple{VariableRef, Float64}},
    state_in::Vector{Float64},
    state_out_target::Vector{Float64},
    Δy::Vector{Float64},
)
    ϕ, y = scalar_output(
        subproblem,
        state_param_in,
        state_param_out,
        uncertainty,
        state_in,
        state_out_target,
        Δy,
    )

    DiffOpt.empty_input_sensitivities!(subproblem)
    for i in eachindex(state_param_out)
        DiffOpt.set_reverse_variable(subproblem, state_param_out[i][2], Δy[i])
    end
    DiffOpt.reverse_differentiate!(subproblem)

    g_state_in = [DiffOpt.get_reverse_parameter(subproblem, p) for p in state_param_in]
    g_state_target = [DiffOpt.get_reverse_parameter(subproblem, p) for (p, _) in state_param_out]
    return ϕ, y, g_state_in, g_state_target
end

function fd_direction_state_in(
    subproblem::JuMP.Model,
    state_param_in::Vector{Any},
    state_param_out::Vector{Tuple{Any, VariableRef}},
    uncertainty::Vector{Tuple{VariableRef, Float64}},
    state_in::Vector{Float64},
    state_out_target::Vector{Float64},
    Δy::Vector{Float64},
    direction::Vector{Float64};
    eps::Float64 = 1e-5,
)
    state_plus = state_in .+ eps .* direction
    state_minus = state_in .- eps .* direction
    ϕ_plus, _ = scalar_output(
        subproblem,
        state_param_in,
        state_param_out,
        uncertainty,
        state_plus,
        state_out_target,
        Δy,
    )
    ϕ_minus, _ = scalar_output(
        subproblem,
        state_param_in,
        state_param_out,
        uncertainty,
        state_minus,
        state_out_target,
        Δy,
    )
    return (ϕ_plus - ϕ_minus) / (2eps)
end

function fd_direction_state_target(
    subproblem::JuMP.Model,
    state_param_in::Vector{Any},
    state_param_out::Vector{Tuple{Any, VariableRef}},
    uncertainty::Vector{Tuple{VariableRef, Float64}},
    state_in::Vector{Float64},
    state_out_target::Vector{Float64},
    Δy::Vector{Float64},
    direction::Vector{Float64};
    eps::Float64 = 1e-5,
)
    target_plus = state_out_target .+ eps .* direction
    target_minus = state_out_target .- eps .* direction
    ϕ_plus, _ = scalar_output(
        subproblem,
        state_param_in,
        state_param_out,
        uncertainty,
        state_in,
        target_plus,
        Δy,
    )
    ϕ_minus, _ = scalar_output(
        subproblem,
        state_param_in,
        state_param_out,
        uncertainty,
        state_in,
        target_minus,
        Δy,
    )
    return (ϕ_plus - ϕ_minus) / (2eps)
end

function fd_component_state_in(
    subproblem::JuMP.Model,
    state_param_in::Vector{Any},
    state_param_out::Vector{Tuple{Any, VariableRef}},
    uncertainty::Vector{Tuple{VariableRef, Float64}},
    state_in::Vector{Float64},
    state_out_target::Vector{Float64},
    Δy::Vector{Float64},
    idx::Int;
    eps::Float64 = 1e-5,
)
    e = zeros(length(state_in))
    e[idx] = 1.0
    return fd_direction_state_in(
        subproblem,
        state_param_in,
        state_param_out,
        uncertainty,
        state_in,
        state_out_target,
        Δy,
        e;
        eps = eps,
    )
end

function fd_component_state_target(
    subproblem::JuMP.Model,
    state_param_in::Vector{Any},
    state_param_out::Vector{Tuple{Any, VariableRef}},
    uncertainty::Vector{Tuple{VariableRef, Float64}},
    state_in::Vector{Float64},
    state_out_target::Vector{Float64},
    Δy::Vector{Float64},
    idx::Int;
    eps::Float64 = 1e-5,
)
    e = zeros(length(state_out_target))
    e[idx] = 1.0
    return fd_direction_state_target(
        subproblem,
        state_param_in,
        state_param_out,
        uncertainty,
        state_in,
        state_out_target,
        Δy,
        e;
        eps = eps,
    )
end

function spread_indices(n::Int)
    return unique(sort([1, cld(n, 4), cld(n, 2), cld(3n, 4), n]))
end

function report_subset_quality(label::String, g_rev::Vector{Float64}, g_fd::Vector{Float64}, idx::Vector{Int})
    abs_err = abs.(g_rev .- g_fd)
    rel_err = abs_err ./ max.(abs.(g_fd), 1e-8)
    i_max = argmax(abs_err)
    println("$label coordinate-check quality:")
    println("  checked_indices = $idx")
    println("  max_abs_error   = $(maximum(abs_err))")
    println("  mean_abs_error  = $(mean(abs_err))")
    println("  max_rel_error   = $(maximum(rel_err))")
    println("  mean_rel_error  = $(mean(rel_err))")
    println("  worst_pair(rev,fd)= ($(g_rev[i_max]), $(g_fd[i_max])) at checked offset $i_max")
end

function main()
    println("Building Atlas with N=2 (one subproblem)...")
    quick_optimizer = () -> begin
        m = DiffOpt.diff_optimizer(
            optimizer_with_attributes(
                Ipopt.Optimizer,
                "print_level" => 0,
                "hsllib" => HSL_jll.libhsl_path,
                "linear_solver" => "ma27",
                "hessian_approximation" => "limited-memory",
                "max_iter" => 50,
                "tol" => 1e-3,
            ),
        )
        MOI.set(m, DiffOpt.ModelConstructor(), DiffOpt.NonLinearProgram.Model)
        return m
    end

    subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples,
    _, _, _, _, _ = build_atlas_subproblems(;
        N = 2,
        h = 0.01,
        perturbation_scale = 0.5,
        num_scenarios = 2,
        penalty = 10.0,
        perturbation_frequency = 51,
        optimizer = quick_optimizer,
    )

    subproblem = subproblems[1]
    state_param_in = state_params_in[1]
    state_param_out = state_params_out[1]
    uncertainty = sample_uncertainty(uncertainty_samples[1])
    state_in = copy(initial_state)
    state_out_target = copy(initial_state)

    println("1) Solving baseline one-subproblem forward pass...")
    ϕ0, y0 = scalar_output(
        subproblem,
        state_param_in,
        state_param_out,
        uncertainty,
        state_in,
        state_out_target,
        ones(length(state_param_out)),
    )
    println("   phi(ones' * y) baseline = $ϕ0")
    println("   output dimension        = $(length(y0))")

    # Use a sparse cotangent so the reverse pass corresponds to one output component.
    Δy = zeros(length(y0))
    Δy[1] = 1.0

    println("2) Reverse-mode VJP with DiffOpt...")
    _, _, g_rev_state_in, g_rev_state_target = reverse_vjp(
        subproblem,
        state_param_in,
        state_param_out,
        uncertainty,
        state_in,
        state_out_target,
        Δy,
    )
    println("   reverse pass done")

    println("3) Finite-difference gradient checks (directional + coordinate subset)...")
    eps = 1e-5

    Random.seed!(42)
    d_state_in = randn(length(g_rev_state_in))
    d_state_in ./= max(norm(d_state_in), 1e-16)
    d_state_target = randn(length(g_rev_state_target))
    d_state_target ./= max(norm(d_state_target), 1e-16)

    fd_dir_state_in = fd_direction_state_in(
        subproblem,
        state_param_in,
        state_param_out,
        uncertainty,
        state_in,
        state_out_target,
        Δy,
        d_state_in;
        eps = eps,
    )
    fd_dir_state_target = fd_direction_state_target(
        subproblem,
        state_param_in,
        state_param_out,
        uncertainty,
        state_in,
        state_out_target,
        Δy,
        d_state_target;
        eps = eps,
    )

    println("4) Gradient quality report")
    rev_dir_state_in = dot(g_rev_state_in, d_state_in)
    rev_dir_state_target = dot(g_rev_state_target, d_state_target)
    println("state_in directional check:")
    println("  reverse directional derivative = $rev_dir_state_in")
    println("  finite-diff directional deriv  = $fd_dir_state_in")
    println("  abs_error = $(abs(rev_dir_state_in - fd_dir_state_in))")
    println("  rel_error = $(abs(rev_dir_state_in - fd_dir_state_in) / max(abs(fd_dir_state_in), 1e-8))")
    println("state_out_target directional check:")
    println("  reverse directional derivative = $rev_dir_state_target")
    println("  finite-diff directional deriv  = $fd_dir_state_target")
    println("  abs_error = $(abs(rev_dir_state_target - fd_dir_state_target))")
    println("  rel_error = $(abs(rev_dir_state_target - fd_dir_state_target) / max(abs(fd_dir_state_target), 1e-8))")

    idx_in = spread_indices(length(g_rev_state_in))
    fd_subset_in = [
        fd_component_state_in(
            subproblem,
            state_param_in,
            state_param_out,
            uncertainty,
            state_in,
            state_out_target,
            Δy,
            i;
            eps = eps,
        )
        for i in idx_in
    ]
    rev_subset_in = [g_rev_state_in[i] for i in idx_in]
    report_subset_quality("state_in", rev_subset_in, fd_subset_in, idx_in)

    idx_target = spread_indices(length(g_rev_state_target))
    fd_subset_target = [
        fd_component_state_target(
            subproblem,
            state_param_in,
            state_param_out,
            uncertainty,
            state_in,
            state_out_target,
            Δy,
            i;
            eps = eps,
        )
        for i in idx_target
    ]
    rev_subset_target = [g_rev_state_target[i] for i in idx_target]
    report_subset_quality("state_out_target", rev_subset_target, fd_subset_target, idx_target)
end

main()
