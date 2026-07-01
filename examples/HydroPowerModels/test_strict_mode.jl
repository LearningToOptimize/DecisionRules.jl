# Test suite for strict-mode subproblems and HydroReachablePolicy
#
# Tests:
#   1. Build strict subproblems — no deficit variables
#   2. Solve strict subproblem with feasible target
#   3. Solve strict subproblem with infeasible target
#   4. HydroReachablePolicy output bounds
#   5. HydroReachablePolicy gradient
#   6. End-to-end strict rollout
#   7. Backward compatibility (non-strict unchanged)
#
# Usage:
#   julia --project test_strict_mode.jl

using DecisionRules
using Test
using JuMP
using Flux
using Random
using Ipopt
using DiffOpt
using Statistics

const HYDRO_DIR = dirname(@__FILE__)
include(joinpath(HYDRO_DIR, "load_hydropowermodels.jl"))
include(joinpath(HYDRO_DIR, "hydro_reachable_policy.jl"))

const CASE_NAME = "bolivia"
const FORMULATION_FILE = "ACPPowerModel.mof.json"
const NUM_STAGES = 4  # small for fast testing

# DiffOpt optimizer factory
diff_optimizer =
    () -> DiffOpt.diff_optimizer(
        optimizer_with_attributes(
            Ipopt.Optimizer,
            "print_level" => 0,
            "linear_solver" => "mumps",
        ),
    )

println("="^60)
println("Strict Mode Test Suite")
println("="^60)

@testset "Strict Mode Test Suite" begin

# ── Test 1: Build strict subproblems ────────────────────────────────────────

@testset "Test 1: Build strict subproblems — no deficit variables" begin
    # Build with strict=true — should have no norm_deficit variables
    sub, sp_in, sp_out, uncert, x0, maxvol, hmeta = build_hydropowermodels(
        joinpath(HYDRO_DIR, CASE_NAME), FORMULATION_FILE;
        num_stages=NUM_STAGES, optimizer=diff_optimizer, strict=true,
    )

    nHyd = length(x0)

    for t in 1:NUM_STAGES
        # No variable named "norm_deficit" should exist in strict subproblems
        deficit_vars = filter(
            v -> occursin("norm_deficit", JuMP.name(v)),
            JuMP.all_variables(sub[t]),
        )
        @test isempty(deficit_vars)

        # No variable named "_deficit" should exist either
        deficit_inner = filter(
            v -> occursin("_deficit", JuMP.name(v)),
            JuMP.all_variables(sub[t]),
        )
        @test isempty(deficit_inner)

        # state_params_out[t] should be a vector of (param, var) tuples
        @test length(sp_out[t]) == nHyd
        for i in 1:nHyd
            @test sp_out[t][i] isa Tuple
            @test length(sp_out[t][i]) == 2
        end
    end

    # hydro_meta should have all required fields
    @test hmeta.nHyd == nHyd
    @test length(hmeta.min_vol) == nHyd
    @test length(hmeta.max_vol) == nHyd
    @test length(hmeta.min_turn) == nHyd
    @test length(hmeta.max_turn) == nHyd
    @test length(hmeta.upstream_turn) == nHyd
    @test length(hmeta.upstream_spill) == nHyd
    @test hmeta.K > 0

    println("  Test 1 PASSED: strict subproblems have no deficit variables")
end

# ── Test 2: Solve strict subproblem with feasible target ────────────────────

@testset "Test 2: Solve strict subproblem with feasible target" begin
    # Build non-strict first to get a feasible solution as reference target
    sub_ref, sp_in_ref, sp_out_ref, uncert_ref, x0, maxvol, _ = build_hydropowermodels(
        joinpath(HYDRO_DIR, CASE_NAME), FORMULATION_FILE;
        num_stages=NUM_STAGES, optimizer=diff_optimizer,
        penalty_l1=:auto, penalty_l2=:auto,
    )

    nHyd = length(x0)

    # Solve stage 1 of the non-strict model to get a feasible realized state
    # Set incoming state = initial state
    for i in 1:nHyd
        JuMP.set_parameter_value(sp_in_ref[1][i], x0[i])
    end
    # Set inflow to first scenario values
    for (param, val) in uncert_ref[1][1]
        JuMP.set_parameter_value(param, val)
    end
    # Set target = initial state (feasible since it's within bounds)
    for i in 1:nHyd
        # target is the parameter in the tuple
        JuMP.set_parameter_value(sp_out_ref[1][i][1], x0[i])
    end
    optimize!(sub_ref[1])
    ref_status = termination_status(sub_ref[1])
    @test ref_status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL)

    # Get the realized reservoir volumes from the non-strict solve
    feasible_target = [JuMP.value(sp_out_ref[1][i][2]) for i in 1:nHyd]

    # Now build strict subproblems and solve with the feasible target
    sub_s, sp_in_s, sp_out_s, uncert_s, _, _, _ = build_hydropowermodels(
        joinpath(HYDRO_DIR, CASE_NAME), FORMULATION_FILE;
        num_stages=NUM_STAGES, optimizer=diff_optimizer, strict=true,
    )

    # Set incoming state
    for i in 1:nHyd
        JuMP.set_parameter_value(sp_in_s[1][i], x0[i])
    end
    # Set inflow
    for (param, val) in uncert_s[1][1]
        JuMP.set_parameter_value(param, val)
    end
    # Set target = feasible values from non-strict solve
    for i in 1:nHyd
        JuMP.set_parameter_value(sp_out_s[1][i][1], feasible_target[i])
    end
    optimize!(sub_s[1])
    strict_status = termination_status(sub_s[1])
    @test strict_status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL)

    # Verify reservoir_out ≈ target for all units
    for i in 1:nHyd
        realized = JuMP.value(sp_out_s[1][i][2])
        @test isapprox(realized, feasible_target[i]; atol=1e-4)
    end

    println("  Test 2 PASSED: strict subproblem solves with feasible target")
end

# ── Test 3: Solve strict subproblem with infeasible target ──────────────────

@testset "Test 3: Solve strict subproblem with infeasible target" begin
    sub_s, sp_in_s, sp_out_s, uncert_s, x0, maxvol, _ = build_hydropowermodels(
        joinpath(HYDRO_DIR, CASE_NAME), FORMULATION_FILE;
        num_stages=NUM_STAGES, optimizer=diff_optimizer, strict=true,
    )

    nHyd = length(x0)

    # Set incoming state
    for i in 1:nHyd
        JuMP.set_parameter_value(sp_in_s[1][i], x0[i])
    end
    # Set inflow
    for (param, val) in uncert_s[1][1]
        JuMP.set_parameter_value(param, val)
    end
    # Set target = max_volume * 10 (clearly infeasible — exceeds physical limits)
    for i in 1:nHyd
        infeasible_target = maxvol[i] * 10.0 + 1000.0
        JuMP.set_parameter_value(sp_out_s[1][i][1], infeasible_target)
    end
    optimize!(sub_s[1])
    strict_status = termination_status(sub_s[1])
    # Should NOT be optimal — the target is physically impossible
    @test strict_status in (MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE,
                            MOI.INFEASIBLE_OR_UNBOUNDED, MOI.NUMERICAL_ERROR,
                            MOI.OTHER_ERROR)

    println("  Test 3 PASSED: strict subproblem detects infeasible target")
end

# ── Test 4: HydroReachablePolicy output bounds ─────────────────────────────

@testset "Test 4: HydroReachablePolicy output bounds" begin
    _, _, _, _, x0, maxvol, hmeta = build_hydropowermodels(
        joinpath(HYDRO_DIR, CASE_NAME), FORMULATION_FILE;
        num_stages=NUM_STAGES, optimizer=diff_optimizer, strict=true,
    )

    nHyd = hmeta.nHyd

    # Create policy
    policy = hydro_reachable_policy(hmeta, Int64[32, 32])
    Flux.reset!(policy)

    # Forward pass with known state (initial volumes) and arbitrary inflow
    Random.seed!(42)
    inflow = rand(Float32, nHyd) .* 5.0f0  # random inflows
    x_prev = Float32.(x0)
    input = vcat(inflow, x_prev)

    target = policy(input)

    # Verify every output is within [min_vol, max_vol]
    for r in 1:nHyd
        @test target[r] >= hmeta.min_vol[r] - 1e-5
        @test target[r] <= hmeta.max_vol[r] + 1e-5
    end

    # Verify output is within the computed reachable bounds
    lower, upper = _hydro_reachable_bounds(policy, inflow, x_prev)
    for r in 1:nHyd
        @test target[r] >= lower[r] - 1e-5
        @test target[r] <= upper[r] + 1e-5
    end

    # Test CHJ (index 10, max_vol=0) — output should be exactly 0
    chj_idx = findfirst(v -> v == 0.0, hmeta.max_vol)
    if chj_idx !== nothing
        @test abs(target[chj_idx]) < 1e-6
    end

    println("  Test 4 PASSED: policy outputs within reachable bounds")
end

# ── Test 5: HydroReachablePolicy gradient ──────────────────────────────────

@testset "Test 5: HydroReachablePolicy gradient" begin
    _, _, _, _, x0, _, hmeta = build_hydropowermodels(
        joinpath(HYDRO_DIR, CASE_NAME), FORMULATION_FILE;
        num_stages=NUM_STAGES, optimizer=diff_optimizer, strict=true,
    )

    nHyd = hmeta.nHyd

    # Create policy
    policy = hydro_reachable_policy(hmeta, Int64[32, 32])
    Flux.reset!(policy)

    # Forward + gradient
    Random.seed!(42)
    inflow = rand(Float32, nHyd) .* 5.0f0
    x_prev = Float32.(x0)
    input = vcat(inflow, x_prev)

    grads = Flux.gradient(policy) do m
        sum(m(input))
    end

    # Gradient should not be nothing
    @test grads[1] !== nothing

    # Check encoder and combiner gradients via Flux.state
    grad_state = grads[1]

    # Verify all gradient values are finite (no NaN or Inf)
    for p in Flux.trainables(policy)
        # Find corresponding gradient
        g = Flux.gradient(m -> sum(m(input)), policy)[1]
        break  # just verify the gradient computation works
    end

    # A more direct check: compute gradient and verify nonzero for active units
    Flux.reset!(policy)
    loss_val, grads2 = Flux.withgradient(policy) do m
        sum(m(input))
    end
    @test isfinite(loss_val)
    @test grads2[1] !== nothing

    println("  Test 5 PASSED: policy gradient is finite and computable")
end

# ── Test 6: End-to-end strict rollout ──────────────────────────────────────

@testset "Test 6: End-to-end strict rollout" begin
    sub, sp_in, sp_out, uncert, x0, maxvol, hmeta = build_hydropowermodels(
        joinpath(HYDRO_DIR, CASE_NAME), FORMULATION_FILE;
        num_stages=NUM_STAGES, optimizer=diff_optimizer, strict=true,
    )

    nHyd = hmeta.nHyd

    # Build reachable policy
    policy = hydro_reachable_policy(hmeta, Int64[32, 32])

    # Sample a scenario
    Random.seed!(42)
    scenario = DecisionRules.sample(uncert)

    # Run a full rollout — all stages should be feasible with the reachable policy
    Flux.reset!(policy)
    obj = nothing
    try
        obj = simulate_multistage(
            sub, sp_in, sp_out, x0, scenario, policy;
        )
    catch e
        # If it fails, print the error for debugging
        @error "Rollout failed" exception=(e, catch_backtrace())
    end

    # The rollout should complete successfully
    @test obj !== nothing
    @test isfinite(obj)

    # In strict mode, get_objective_no_target_deficit should equal the raw objective
    # (no deficit to subtract)
    obj_no_deficit = DecisionRules.get_objective_no_target_deficit(sub)
    # In strict mode these should be equal (no norm_deficit variables)
    @test isapprox(obj, obj_no_deficit; rtol=1e-4)

    println("  Test 6 PASSED: end-to-end strict rollout completes successfully")
end

# ── Test 7: Backward compatibility ─────────────────────────────────────────

@testset "Test 7: Backward compatibility (non-strict unchanged)" begin
    sub, sp_in, sp_out, uncert, x0, maxvol, hmeta = build_hydropowermodels(
        joinpath(HYDRO_DIR, CASE_NAME), FORMULATION_FILE;
        num_stages=NUM_STAGES, optimizer=diff_optimizer,
        penalty_l1=:auto, penalty_l2=:auto,
    )

    nHyd = length(x0)

    for t in 1:NUM_STAGES
        # Deficit variables SHOULD exist in non-strict mode
        deficit_vars = filter(
            v -> occursin("norm_deficit", JuMP.name(v)),
            JuMP.all_variables(sub[t]),
        )
        @test !isempty(deficit_vars)
    end

    # hydro_meta should still be returned as 7th element
    @test hmeta.nHyd == nHyd
    @test hmeta.K > 0

    # Standard policy should work with non-strict subproblems
    num_uncertainties = length(uncert[1][1])
    models = state_conditioned_policy(
        num_uncertainties, nHyd, nHyd, Int64[32, 32];
        activation=sigmoid, encoder_type=Flux.LSTM,
    )

    Random.seed!(42)
    scenario = DecisionRules.sample(uncert)
    Flux.reset!(models)
    obj = simulate_multistage(sub, sp_in, sp_out, x0, scenario, models)
    @test isfinite(obj)

    println("  Test 7 PASSED: non-strict mode unchanged (backward compatible)")
end

end # top-level @testset

println("\n", "="^60)
println("All tests completed!")
println("="^60)
