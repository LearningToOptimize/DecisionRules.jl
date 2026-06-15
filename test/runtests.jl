using DecisionRules
using Test
using Ipopt
using MadNLP
using JuMP
import MathOptInterface as MOI
using Zygote
using Flux
using Random
using DiffOpt
using ChainRules: @ignore_derivatives
using ChainRulesCore

function build_subproblem(d; state_i_val=5.0, state_out_val=4.0, uncertainty_val=2.0, subproblem=JuMP.Model())
    # set_attributes(subproblem, "output_flag" => false)
    @variable(subproblem, x >= 0.0)
    @variable(subproblem, 0.0 <= y <= 8.0)
    @variable(subproblem, 0.0 <= state_out_var <= 8.0)
    @variable(subproblem, _deficit)
    @variable(subproblem, norm_deficit)
    @variable(subproblem, state_in in MOI.Parameter(state_i_val))
    @variable(subproblem, state_out in MOI.Parameter(state_out_val))
    @variable(subproblem, uncertainty in MOI.Parameter(uncertainty_val))
    @constraint(subproblem, state_out_var == state_in + uncertainty - x)
    @constraint(subproblem, x + y >= d)
    @constraint(subproblem, _deficit == state_out_var - state_out)
    @constraint(subproblem, [norm_deficit; _deficit] in MOI.NormOneCone(2))
    @objective(subproblem, Min, 30 * y + norm_deficit * 10^4)
    return subproblem, state_in, state_out, state_out_var, uncertainty
end

@testset "DecisionRules.jl" begin
    @testset "pdual at infeasibility" begin
        subproblem1, state_in_1, state_out_1, state_out_var_1, uncertainty_1 = build_subproblem(10; subproblem=DiffOpt.conic_diff_model(Ipopt.Optimizer), state_out_val=9.0)
        optimize!(subproblem1)
        @test DecisionRules.pdual(state_in_1) ≈ -1.0e4 rtol=1.0e-1
        @test DecisionRules.pdual(state_out_1) ≈ 1.0e4 rtol=1.0e-1
    end

    subproblem1, state_in_1, state_out_1, state_out_var_1, uncertainty_1 = build_subproblem(10; subproblem=DiffOpt.conic_diff_model(Ipopt.Optimizer))

    optimize!(subproblem1)

    @testset "pdual" begin
        @test DecisionRules.pdual(state_in_1) ≈ -30.0 rtol=1.0e-1
        @test DecisionRules.pdual(state_out_1) ≈ 30.0 rtol=1.0e-1
    end

    @testset "simulate_stage" begin
        inflow = 2.0
        state_param_in = Vector{Any}(undef, 1)
        state_param_out = Vector{Tuple{Any, VariableRef}}(undef, 1)
        state_param_in .= [state_in_1]
        state_param_out .= [(state_out_1, state_out_var_1)]
        uncertainty_sample = [(uncertainty_1, inflow)]
        state_in_val = [5.0]
        state_out_val = [4.0]
        # Test simulate_stage
        @test DecisionRules.simulate_stage(subproblem1, state_param_in, state_param_out, uncertainty_sample, state_in_val, state_out_val) ≈ 210 rtol=1.0e-1
        grad = Zygote.gradient(DecisionRules.simulate_stage, subproblem1, state_param_in, state_param_out, uncertainty_sample, state_in_val, state_out_val)
        @test grad[5][1] ≈ -30.0 rtol=1.0e-1
        @test grad[6][1] ≈ 30.0 rtol=1.0e-1
        # Test get next state
        jac = Zygote.jacobian(DecisionRules.get_next_state, subproblem1, state_param_in, state_param_out, state_in_val, state_out_val)
        @test jac[4][1] ≈ 0.0 atol=1.0e-4 # ∂next_state/∂state_in
        @test jac[5][1] ≈ 1.0 rtol=1.0e-1 # ∂next_state/∂state_out

        # Train flux DR
        subproblem = subproblem1
        Random.seed!(222)
        m = Chain(Dense(1, 10), Dense(10, 1))
        # 90 is what we expect after training, so we start above that for a random policy
        @test DecisionRules.simulate_stage(subproblem, state_param_in, state_param_out, uncertainty_sample, state_in_val, m([inflow])) > 90.0
        for _ in 1:100
            _inflow = rand(1.:5)
            uncertainty_samp = [(uncertainty_1, _inflow)]
            Flux.train!((m, inflow) -> DecisionRules.simulate_stage(subproblem, state_param_in, state_param_out, uncertainty_sample, state_in_val, m(inflow)), m, [[_inflow] for _ =1:10], Flux.Adam())
        end
        # since we trained towards 90, we should be close to it now
        @test DecisionRules.simulate_stage(subproblem, state_param_in, state_param_out, uncertainty_sample, state_in_val, m([inflow])) <= 92
    end

    @testset "simulate_multistage (per-stage)" begin
        subproblem1, state_in_1, state_out_1, state_out_var_1, uncertainty_1 = build_subproblem(10; subproblem=DiffOpt.conic_diff_model(Ipopt.Optimizer))
        subproblem2, state_in_2, state_out_2, state_out_var_2, uncertainty_2 = build_subproblem(10; state_i_val=1.0, state_out_val=9.0, uncertainty_val=2.0, subproblem=DiffOpt.conic_diff_model(Ipopt.Optimizer))

        subproblems = [subproblem1, subproblem2]
        state_params_in = Vector{Vector{Any}}(undef, 2)
        state_params_out = Vector{Vector{Tuple{Any, VariableRef}}}(undef, 2)
        state_params_in .= [[state_in_1], [state_in_2]]
        state_params_out .= [[(state_out_1, state_out_var_1)], [(state_out_2, state_out_var_2)]]
        uncertainty_samples = [[(uncertainty_1, [2.0])], [(uncertainty_2, [1.0])]]
        initial_state = [5.0]

        function simulate(initial_state, params)
            i = 0
            m = (ars...) -> begin i= i+1; return params[i] end
            DecisionRules.simulate_multistage(subproblems, state_params_in, state_params_out, initial_state, sample(uncertainty_samples), m)
        end
        grad = Zygote.gradient(simulate, initial_state, [[4.0], [3.]])
        @test grad[2][1][1] + grad[2][2][1] ≈ 30.0 atol=1.0e-2

        rand_policy = simulate([5.0], [[4.0], [3.]])

        @test rand_policy ≈ 450 rtol=1.0e-1

        @test simulate([9.0], [[7.], [4.000]]) ≈ 359 rtol=1.0e-1

        Random.seed!(222)
        # Policy input: [uncertainty, previous_state] = 1 + 1 = 2 dimensions
        m = Chain(Dense(2, 10), Dense(10, 1))
        obj_val_prev = DecisionRules.simulate_multistage(
            subproblems, state_params_in, state_params_out, 
            initial_state, sample(uncertainty_samples), 
            m
        )
        train_multistage(m, initial_state, subproblems, state_params_in, state_params_out, uncertainty_samples)
        obj_val = DecisionRules.simulate_multistage(
            subproblems, state_params_in, state_params_out, 
            initial_state, sample(uncertainty_samples), 
            m
        )

        @test obj_val < rand_policy
        @test obj_val < obj_val_prev
    end

    @testset "deterministic_equivalent" begin
        subproblem1, state_in_1, state_out_1, state_out_var_1, uncertainty_1 = build_subproblem(10)
        subproblem2, state_in_2, state_out_2, state_out_var_2, uncertainty_2 = build_subproblem(10; state_i_val=4.0, state_out_val=3.0, uncertainty_val=1.0)

        subproblems = [subproblem1, subproblem2]
        state_params_in = Vector{Vector{Any}}(undef, 2)
        state_params_out = Vector{Vector{Tuple{Any, VariableRef}}}(undef, 2)
        state_params_in .= [[state_in_1], [state_in_2]]
        state_params_out .= [[(state_out_1, state_out_var_1)], [(state_out_2, state_out_var_2)]]
        uncertainty_samples = [[(uncertainty_1, [2.0])], [(uncertainty_2, [1.0])]]
        initial_state = [5.0]

        det_equivalent, uncertainty_samples = DecisionRules.deterministic_equivalent!(
            DiffOpt.nonlinear_diff_model(Ipopt.Optimizer),
            subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples
        )

        obj_val = DecisionRules.simulate_multistage(det_equivalent, state_params_in, state_params_out, sample(uncertainty_samples), [[9.0], [7.], [4.000]])
        @test obj_val ≈ 359 rtol=1.0e-1
        grad = Zygote.gradient(DecisionRules.simulate_multistage, det_equivalent, state_params_in, state_params_out, sample(uncertainty_samples), [[9.0], [7.], [4.0]])
        @test grad[5][1][1] ≈ -30.0 rtol=1.0e-1
        @test grad[5][3][1] ≈ 30.0 rtol=1.0e-1

        Random.seed!(222)
        uncertainty_sample = sample(uncertainty_samples)
        
        # Policy input: [uncertainty, previous_state] = 1 + 1 = 2 dimensions
        m = Chain(Dense(2, 10), Dense(10, 1))
        obj_val_before = DecisionRules.simulate_multistage(
            det_equivalent, state_params_in, state_params_out, 
            initial_state, uncertainty_sample, 
            m
        )

        train_multistage(m, initial_state, det_equivalent, state_params_in, state_params_out, uncertainty_samples, num_batches=200)

        obj_val_after = DecisionRules.simulate_multistage(
            det_equivalent, state_params_in, state_params_out, 
            initial_state, uncertainty_sample, 
            m
        )

        @test obj_val_after < obj_val_before
    end

    @testset "compute_parameter_dual" begin
        # Test 1: Simple LP with parameter in constraint
        # min x  s.t. x >= p
        # At optimality x* = p, dual of constraint is 1
        # ∂obj/∂p = -1 * 1 = -1 (since p appears with coef 1 in RHS equivalent: x - p >= 0)
        # But in our formulation: x >= p => x - p >= 0, so coef of p is -1
        # dual contribution = -(-1) * 1 = 1
        model1 = Model(Ipopt.Optimizer)
        @variable(model1, x1 >= 0)
        @variable(model1, p1 in MOI.Parameter(2.0))
        @constraint(model1, con1, x1 - p1 >= 0)
        @objective(model1, Min, x1)
        optimize!(model1)
        @test compute_parameter_dual(model1, p1) ≈ 1.0 rtol=1.0e-2

        # Test 2: Parameter in objective
        # min x + 2*p  s.t. x >= 1
        # ∂obj/∂p = 2 (from objective directly, minimization)
        model2 = Model(Ipopt.Optimizer)
        @variable(model2, x2 >= 0)
        @variable(model2, p2 in MOI.Parameter(3.0))
        @constraint(model2, x2 >= 1)
        @objective(model2, Min, x2 + 2 * p2)
        optimize!(model2)
        @test compute_parameter_dual(model2, p2) ≈ 2.0 rtol=1.0e-2

        # Test 3: Parameter in both constraint and objective
        # min x + p  s.t. x >= 2*p
        # At optimality x* = 2p, constraint dual = 1
        # ∂obj/∂p = 1 (from obj) + (-(-2) * 1) = 1 + 2 = 3
        model3 = Model(Ipopt.Optimizer)
        @variable(model3, x3 >= 0)
        @variable(model3, p3 in MOI.Parameter(1.0))
        @constraint(model3, con3, x3 - 2 * p3 >= 0)
        @objective(model3, Min, x3 + p3)
        optimize!(model3)
        @test compute_parameter_dual(model3, p3) ≈ 3.0 rtol=1.0e-2

        # Test 4: Maximization problem
        # max -x + p  s.t. x >= 1
        # Equivalent to min x - p, so ∂obj/∂p = -(-1) = 1 for max
        model4 = Model(Ipopt.Optimizer)
        @variable(model4, x4 >= 0)
        @variable(model4, p4 in MOI.Parameter(1.0))
        @constraint(model4, x4 >= 1)
        @objective(model4, Max, -x4 + p4)
        optimize!(model4)
        @test compute_parameter_dual(model4, p4) ≈ -1.0 rtol=1.0e-2

        # Test 5: SOC constraint with parameter (similar to existing pdual tests)
        model5 = Model(Ipopt.Optimizer)
        @variable(model5, x5 >= 0.0)
        @variable(model5, 0.0 <= y5 <= 8.0)
        @variable(model5, 0.0 <= state_out_var5 <= 8.0)
        @variable(model5, _deficit5)
        @variable(model5, norm_deficit5)
        @variable(model5, state_in5 in MOI.Parameter(5.0))
        @variable(model5, state_out5 in MOI.Parameter(4.0))
        @variable(model5, uncertainty5 in MOI.Parameter(2.0))
        @constraint(model5, state_out_var5 == state_in5 + uncertainty5 - x5)
        @constraint(model5, x5 + y5 >= 10)
        @constraint(model5, _deficit5 == state_out_var5 - state_out5)
        @constraint(model5, [norm_deficit5; _deficit5] in MOI.NormOneCone(2))
        @objective(model5, Min, 30 * y5 + norm_deficit5 * 10^4)
        optimize!(model5)
        @test compute_parameter_dual(model5, state_in5) ≈ -30.0 rtol=1.0e-1
        @test compute_parameter_dual(model5, state_out5) ≈ 30.0 rtol=1.0e-1
    end

    @testset "create_deficit!" begin
        # Test 1: L1 norm only (default/legacy behavior)
        model1 = Model(Ipopt.Optimizer)
        @variable(model1, x1)
        @variable(model1, state[1:3])
        @variable(model1, target[1:3])
        @objective(model1, Min, x1)
        fix(x1, 1.0)
        norm_deficit1, _deficit1 = create_deficit!(model1, 3; penalty=100.0)
        for i in 1:3
            @constraint(model1, _deficit1[i] == state[i] - target[i])
        end
        # Set state and target values to create deficit
        for i in 1:3
            fix(state[i], 1.0 * i)
            fix(target[i], 0.5 * i)  # deficit = 0.5 * i for each
        end
        optimize!(model1)
        @test termination_status(model1) == MOI.LOCALLY_SOLVED
        @test value(norm_deficit1) ≈ 3.0 rtol=1.0e-2  # L1 norm = |0.5| + |1.0| + |1.5| = 3.0
        
        # Test 2: L2 squared norm only (sum of squares)
        model2 = Model(Ipopt.Optimizer)
        @variable(model2, x2)
        @variable(model2, state2[1:3])
        @variable(model2, target2[1:3])
        @objective(model2, Min, x2)
        fix(x2, 1.0)
        norm_deficit2, _deficit2 = create_deficit!(model2, 3; penalty_l2=100.0)
        for i in 1:3
            @constraint(model2, _deficit2[i] == state2[i] - target2[i])
        end
        for i in 1:3
            fix(state2[i], 1.0 * i)
            fix(target2[i], 0.5 * i)
        end
        optimize!(model2)
        @test termination_status(model2) == MOI.LOCALLY_SOLVED
        # L2 squared = 0.5^2 + 1.0^2 + 1.5^2 = 3.5
        @test value(norm_deficit2) ≈ 3.5 rtol=1.0e-2
        
        # Test 3: Both L1 and L2 squared norms
        model3 = Model(Ipopt.Optimizer)
        @variable(model3, x3)
        @variable(model3, state3[1:3])
        @variable(model3, target3[1:3])
        @objective(model3, Min, x3)
        fix(x3, 1.0)
        norm_deficit3, _deficit3 = create_deficit!(model3, 3; penalty_l1=100.0, penalty_l2=50.0)
        for i in 1:3
            @constraint(model3, _deficit3[i] == state3[i] - target3[i])
        end
        for i in 1:3
            fix(state3[i], 1.0 * i)
            fix(target3[i], 0.5 * i)
        end
        optimize!(model3)
        @test termination_status(model3) == MOI.LOCALLY_SOLVED
        # Combined: 100 * 3.0 + 50 * 3.5 = 300 + 175 = 475
        expected_combined = 100.0 * 3.0 + 50.0 * 3.5
        @test value(norm_deficit3) ≈ expected_combined rtol=1.0e-2
        
        # Test 4: Verify backward compatibility with legacy 'penalty' argument
        model4 = Model(Ipopt.Optimizer)
        @variable(model4, x4)
        @objective(model4, Min, x4)
        fix(x4, 1.0)
        norm_deficit4, _deficit4 = create_deficit!(model4, 2; penalty=500.0)
        @test length(_deficit4) == 2
        # Should create L1 norm constraint (backwards compatible)
        optimize!(model4)
        @test termination_status(model4) == MOI.LOCALLY_SOLVED
        
        # Test 5: Verify objective contribution with L1 norm
        model5 = Model(Ipopt.Optimizer)
        @variable(model5, y5 >= 0)
        @objective(model5, Min, 10 * y5)
        @constraint(model5, y5 >= 1)  # Forces y5 = 1
        norm_deficit5, _deficit5 = create_deficit!(model5, 2; penalty_l1=100.0)
        @constraint(model5, _deficit5[1] == 2.0)  # Fixed deficit
        @constraint(model5, _deficit5[2] == 3.0)  # Fixed deficit
        optimize!(model5)
        @test termination_status(model5) == MOI.LOCALLY_SOLVED
        @test value(norm_deficit5) ≈ 5.0 rtol=1.0e-2  # L1 = 2 + 3 = 5
        # Total objective = 10 * 1 + 100 * 5 = 510
        @test objective_value(model5) ≈ 510.0 rtol=1.0e-2
        
        # Test 6: Verify objective contribution with L2 squared norm
        model6 = Model(Ipopt.Optimizer)
        @variable(model6, y6 >= 0)
        @objective(model6, Min, 10 * y6)
        @constraint(model6, y6 >= 1)
        norm_deficit6, _deficit6 = create_deficit!(model6, 2; penalty_l2=100.0)
        @constraint(model6, _deficit6[1] == 3.0)
        @constraint(model6, _deficit6[2] == 4.0)
        optimize!(model6)
        @test termination_status(model6) == MOI.LOCALLY_SOLVED
        @test value(norm_deficit6) ≈ 25.0 rtol=1.0e-2  # L2 squared = 9 + 16 = 25
        # Total objective = 10 * 1 + 100 * 25 = 2510
        @test objective_value(model6) ≈ 2510.0 rtol=1.0e-2
    end

    @testset "penalty schedule (annealing)" begin
        @testset "default_annealed_schedule" begin
            @test default_annealed_schedule(24) == [(1, 2, 0.1), (3, 4, 1.0), (5, 8, 10.0), (9, 24, 30.0)]
            @test default_annealed_schedule(20) == [(1, 2, 0.1), (3, 4, 1.0), (5, 7, 10.0), (8, 20, 30.0)]
            @test default_annealed_schedule(4) == [(1, 1, 0.1), (2, 2, 1.0), (3, 3, 10.0), (4, 4, 30.0)]
            # Fewer batches than multipliers: keep the last multipliers, always end at 30x
            @test default_annealed_schedule(2) == [(1, 1, 10.0), (2, 2, 30.0)]
            @test default_annealed_schedule(1) == [(1, 1, 30.0)]
            @test_throws ArgumentError default_annealed_schedule(0)
            # Every phase covered, contiguous from batch 1
            for n in (4, 5, 6, 24, 100, 3000)
                sched = default_annealed_schedule(n)
                @test sched[1][1] == 1
                @test sched[end][2] == n
                @test all(sched[k+1][1] == sched[k][2] + 1 for k in 1:length(sched)-1)
            end
        end

        @testset "schedule resolution and validation" begin
            sched = default_annealed_schedule(4)
            @test DecisionRules._penalty_multiplier_for(sched, 1) == 0.1
            @test DecisionRules._penalty_multiplier_for(sched, 4) == 30.0
            @test DecisionRules._penalty_multiplier_for(sched, 99) == 30.0  # past-end hold
            @test DecisionRules._resolve_penalty_schedule(nothing, 10) === nothing
            @test DecisionRules._resolve_penalty_schedule(:default_annealed, 24) == default_annealed_schedule(24)
            @test_throws ArgumentError DecisionRules._resolve_penalty_schedule(:not_a_schedule, 10)
            @test_throws ArgumentError DecisionRules._resolve_penalty_schedule([], 10)                              # empty
            @test_throws ArgumentError DecisionRules._resolve_penalty_schedule([(2, 3, 1.0)], 10)                   # starts after 1
            @test_throws ArgumentError DecisionRules._resolve_penalty_schedule([(1, 2, 1.0), (4, 5, 2.0)], 10)      # gap
            @test_throws ArgumentError DecisionRules._resolve_penalty_schedule([(1, 3, 1.0), (3, 5, 2.0)], 10)      # overlap
            @test_throws ArgumentError DecisionRules._resolve_penalty_schedule([(1, 2, -1.0)], 10)                  # nonpositive
            @test_throws ArgumentError DecisionRules._resolve_penalty_schedule([(2, 1, 1.0)], 10)                   # lo > hi
        end

        @testset "deficit penalty scaling across create_deficit! modes" begin
            # L1-only: penalty is the objective coefficient of norm_deficit
            ml1 = Model()
            @variable(ml1, yl1 >= 0)
            @objective(ml1, Min, 10 * yl1)
            ndl1, _ = create_deficit!(ml1, 2; penalty_l1=100.0)
            bases = DecisionRules._deficit_penalty_bases(ml1)
            @test bases == Dict(ndl1 => 100.0)
            DecisionRules._apply_deficit_penalty_multiplier!(ml1, bases, 10.0)
            @test coefficient(objective_function(ml1), ndl1) ≈ 1000.0
            DecisionRules._apply_deficit_penalty_multiplier!(ml1, bases, 1.0)
            @test coefficient(objective_function(ml1), ndl1) ≈ 100.0

            # L2-only: same entry point
            ml2 = Model()
            @variable(ml2, yl2 >= 0)
            @objective(ml2, Min, 10 * yl2)
            ndl2, _ = create_deficit!(ml2, 2; penalty_l2=100.0)
            DecisionRules._apply_deficit_penalty_multiplier!(ml2, DecisionRules._deficit_penalty_bases(ml2), 0.5)
            @test coefficient(objective_function(ml2), ndl2) ≈ 50.0

            # Both-mode: penalties live in the linking constraint; the objective coefficient
            # of norm_deficit is 1.0 and scaling it scales L1 and L2 uniformly
            mb = Model()
            @variable(mb, yb >= 0)
            @objective(mb, Min, 10 * yb)
            ndb, _ = create_deficit!(mb, 2; penalty_l1=100.0, penalty_l2=50.0)
            basesb = DecisionRules._deficit_penalty_bases(mb)
            @test basesb == Dict(ndb => 1.0)
            DecisionRules._apply_deficit_penalty_multiplier!(mb, basesb, 30.0)
            @test coefficient(objective_function(mb), ndb) ≈ 30.0

            # No matching variable with a schedule active is an error, not a silent no-op
            mnone = Model()
            @variable(mnone, ynone >= 0)
            @objective(mnone, Min, ynone)
            @test_throws ArgumentError DecisionRules._check_deficit_penalty_bases(
                DecisionRules._deficit_penalty_bases(mnone))
        end

        @testset "train_multistage penalty_schedule end-to-end" begin
            # nothing (default) leaves coefficients untouched
            sp1, si1, so1, sov1, u1 = build_subproblem(10; subproblem=DiffOpt.conic_diff_model(Ipopt.Optimizer))
            sp2, si2, so2, sov2, u2 = build_subproblem(10; state_i_val=1.0, state_out_val=9.0, subproblem=DiffOpt.conic_diff_model(Ipopt.Optimizer))
            sps = [sp1, sp2]
            spi = Vector{Vector{Any}}(undef, 2); spi .= [[si1], [si2]]
            spo = Vector{Vector{Tuple{Any, VariableRef}}}(undef, 2)
            spo .= [[(so1, sov1)], [(so2, sov2)]]
            usamples = [[(u1, [2.0])], [(u2, [1.0])]]
            nd1 = variable_by_name(sp1, "norm_deficit")
            nd2 = variable_by_name(sp2, "norm_deficit")
            Random.seed!(222)
            m = Chain(Dense(2, 4), Dense(4, 1))
            train_multistage(m, [5.0], sps, spi, spo, usamples; num_batches=2, num_train_per_batch=1)
            @test coefficient(objective_function(sp1), nd1) ≈ 1.0e4
            @test coefficient(objective_function(sp2), nd2) ≈ 1.0e4

            # Explicit two-phase schedule crosses a boundary on DiffOpt models and the final
            # coefficients hold the last multiplier times the built base
            train_multistage(m, [5.0], sps, spi, spo, usamples;
                num_batches=4, num_train_per_batch=1,
                penalty_schedule=[(1, 2, 1.0), (3, 4, 3.0)])
            @test coefficient(objective_function(sp1), nd1) ≈ 3.0e4
            @test coefficient(objective_function(sp2), nd2) ≈ 3.0e4
        end

        @testset "deterministic-equivalent overload applies the schedule to the copies" begin
            sp1, si1, so1, sov1, u1 = build_subproblem(10)
            sp2, si2, so2, sov2, u2 = build_subproblem(10; state_i_val=4.0, state_out_val=3.0, uncertainty_val=1.0)
            sps = [sp1, sp2]
            spi = Vector{Vector{Any}}(undef, 2); spi .= [[si1], [si2]]
            spo = Vector{Vector{Tuple{Any, VariableRef}}}(undef, 2)
            spo .= [[(so1, sov1)], [(so2, sov2)]]
            usamples = [[(u1, [2.0])], [(u2, [1.0])]]
            det_equivalent, usamples = DecisionRules.deterministic_equivalent!(
                DiffOpt.nonlinear_diff_model(Ipopt.Optimizer),
                sps, spi, spo, [5.0], usamples)
            nd_copies = [v for v in all_variables(det_equivalent) if occursin("norm_deficit", JuMP.name(v))]
            @test length(nd_copies) == 2  # one renamed copy per stage
            @test all(coefficient(objective_function(det_equivalent), v) ≈ 1.0e4 for v in nd_copies)
            Random.seed!(222)
            m = Chain(Dense(2, 10), Dense(10, 1))
            train_multistage(m, [5.0], det_equivalent, spi, spo, usamples;
                num_batches=4, num_train_per_batch=1,
                penalty_schedule=[(1, 2, 1.0), (3, 4, 3.0)])
            @test all(coefficient(objective_function(det_equivalent), v) ≈ 3.0e4 for v in nd_copies)
        end

        @testset "train_multiple_shooting applies the schedule to the window models" begin
            sp, si, so, sov, u = build_subproblem(10)
            spi = Vector{Vector{Any}}(undef, 1); spi .= [[si]]
            spo = Vector{Vector{Tuple{Any, VariableRef}}}(undef, 1)
            spo .= [[(so, sov)]]
            usamples = [[(u, [2.0])]]
            windows = DecisionRules.setup_shooting_windows(
                [sp], spi, spo, [5.0], usamples;
                window_size=1,
                model_factory=() -> DiffOpt.conic_diff_model(Ipopt.Optimizer))
            wm = windows[1].model
            nd_window = [v for v in all_variables(wm) if occursin("norm_deficit", JuMP.name(v))]
            @test length(nd_window) == 1
            @test coefficient(objective_function(wm), nd_window[1]) ≈ 1.0e4
            model = Dense(2, 1; bias=false)
            model.weight .= 0.5f0
            DecisionRules.train_multiple_shooting(model, [5.0], windows, () -> usamples;
                num_batches=4, num_train_per_batch=1, optimizer=Flux.Descent(0.0),
                record_loss=(iter, m, loss, tag) -> false,
                penalty_schedule=[(1, 2, 1.0), (3, 4, 3.0)])
            @test coefficient(objective_function(wm), nd_window[1]) ≈ 3.0e4
        end
    end

    @testset "sample logger and per-batch record" begin
        function build_two_stage()
            sp1, si1, so1, sov1, u1 = build_subproblem(10; subproblem=DiffOpt.conic_diff_model(Ipopt.Optimizer))
            sp2, si2, so2, sov2, u2 = build_subproblem(10; state_i_val=1.0, state_out_val=9.0, subproblem=DiffOpt.conic_diff_model(Ipopt.Optimizer))
            sps = [sp1, sp2]
            spi = Vector{Vector{Any}}(undef, 2)
            spi .= [[si1], [si2]]
            spo = Vector{Vector{Tuple{Any, VariableRef}}}(undef, 2)
            spo .= [[(so1, sov1)], [(so2, sov2)]]
            usamples = [[(u1, [2.0])], [(u2, [1.0])]]
            return sps, spi, spo, usamples
        end

        @testset "default path caches both metrics per sample" begin
            sps, spi, spo, usamples = build_two_stage()
            Random.seed!(222)
            m = Chain(Dense(2, 4), Dense(4, 1))
            recorded = []
            sl = SampleLog()
            train_multistage(m, [5.0], sps, spi, spo, usamples;
                num_batches=2, num_train_per_batch=2, sample_log=sl,
                record=(sample_log, iter, model) -> begin
                    push!(recorded, (iter,
                        copy(sample_log.objectives),
                        copy(sample_log.objectives_no_deficit),
                        sum(objective_value, sps) == sample_log.objectives[end],
                        DecisionRules.get_objective_no_target_deficit(sps) == sample_log.objectives_no_deficit[end]))
                    return false
                end)
            @test length(recorded) == 2
            for (iter, objs, objs_nd, obj_matches, nd_matches) in recorded
                @test length(objs) == 2 && length(objs_nd) == 2   # cache cleared per batch
                @test all(isfinite, objs) && all(isfinite, objs_nd)
                @test all(objs_nd .<= objs .+ 1.0e-6)             # slack penalty is nonnegative
                @test obj_matches                                  # cache equals re-read of the live models
                @test nd_matches
            end
            @test default_record(sl, 1, m) == false
        end

        @testset "deterministic-equivalent overload caches per sample" begin
            sps, spi, spo, usamples = build_two_stage()
            det_equivalent, usamples_det = DecisionRules.deterministic_equivalent!(
                DiffOpt.nonlinear_diff_model(Ipopt.Optimizer),
                sps, spi, spo, [5.0], usamples)
            Random.seed!(222)
            m = Chain(Dense(2, 10), Dense(10, 1))
            recorded = []
            train_multistage(m, [5.0], det_equivalent, spi, spo, usamples_det;
                num_batches=2, num_train_per_batch=2,
                record=(sample_log, iter, model) -> begin
                    push!(recorded, (length(sample_log.objectives),
                        sample_log.objectives[end] == objective_value(det_equivalent),
                        sample_log.objectives_no_deficit[end] ==
                            DecisionRules.get_objective_no_target_deficit(det_equivalent)))
                    return false
                end)
            @test length(recorded) == 2
            for (n, obj_matches, nd_matches) in recorded
                @test n == 2          # cache cleared per batch
                @test obj_matches     # cache equals re-read of the live det-eq model
                @test nd_matches
            end
        end

        @testset "per-sample hook fires with the live models" begin
            sps, spi, spo, usamples = build_two_stage()
            Random.seed!(222)
            m = Chain(Dense(2, 4), Dense(4, 1))
            hook_calls = []
            sl = SampleLog(on_sample=(s, models, log) -> push!(hook_calls, (s, termination_status(models[1]))))
            train_multistage(m, [5.0], sps, spi, spo, usamples;
                num_batches=1, num_train_per_batch=3, sample_log=sl,
                record=(sample_log, iter, model) -> false)
            @test [c[1] for c in hook_calls] == [1, 2, 3]
            @test all(c[2] == MOI.LOCALLY_SOLVED for c in hook_calls)
        end

        @testset "record early-stop contract" begin
            sps, spi, spo, usamples = build_two_stage()
            Random.seed!(222)
            m = Chain(Dense(2, 4), Dense(4, 1))
            batches_seen = Ref(0)
            train_multistage(m, [5.0], sps, spi, spo, usamples;
                num_batches=5, num_train_per_batch=1,
                record=(sample_log, iter, model) -> begin
                    batches_seen[] += 1
                    return true
                end)
            @test batches_seen[] == 1
        end

        @testset "deprecated record_loss adapter" begin
            sps, spi, spo, usamples = build_two_stage()
            Random.seed!(222)
            m = Chain(Dense(2, 4), Dense(4, 1))
            calls = []
            train_multistage(m, [5.0], sps, spi, spo, usamples;
                num_batches=2, num_train_per_batch=1,
                record_loss=(iter, model, loss, tag) -> begin
                    push!(calls, tag)
                    return false
                end)
            @test calls == ["metrics/loss", "metrics/training_loss", "metrics/loss", "metrics/training_loss"]
            # a stop requested on the first call short-circuits the second, as historically
            calls2 = []
            train_multistage(m, [5.0], sps, spi, spo, usamples;
                num_batches=3, num_train_per_batch=1,
                record_loss=(iter, model, loss, tag) -> begin
                    push!(calls2, tag)
                    return true
                end)
            @test calls2 == ["metrics/loss"]
            @test_throws ArgumentError train_multistage(m, [5.0], sps, spi, spo, usamples;
                num_batches=1, num_train_per_batch=1,
                record=(sample_log, iter, model) -> false,
                record_loss=(iter, model, loss, tag) -> false)
        end

        @testset "RolloutEvaluation on a fixed scenario set" begin
            sps, spi, spo, usamples = build_two_stage()
            Random.seed!(222)
            scenarios = [DecisionRules.sample(usamples) for _ in 1:2]
            # A policy on the reachable frontier: y <= 8 with x + y >= 10 forces x >= 2,
            # so the largest reachable state_out is state_in + uncertainty - 2. The policy
            # input is [uncertainty..., previous_state...], hence the target below is
            # exactly reachable at every stage and the violation share is ~0 by
            # construction.
            reachable_policy = x -> [x[1] + x[2] - 2.0]
            rollout_eval = RolloutEvaluation(sps, spi, spo, [5.0], scenarios; stride=1)
            @test rollout_eval(1, reachable_policy) === nothing
            @test isfinite(rollout_eval.last_objective_no_deficit)
            @test abs(rollout_eval.last_violation_share) < 1.0e-4
            # stride: no evaluation on off-stride batches
            rollout_eval2 = RolloutEvaluation(sps, spi, spo, [5.0], scenarios; stride=2)
            rollout_eval2(1, reachable_policy)
            @test isnan(rollout_eval2.last_objective_no_deficit)
            @test_throws ArgumentError RolloutEvaluation(sps, spi, spo, [5.0], []; stride=1)
            @test isnan(DecisionRules._target_violation_share(0.0, 0.0))
            @test DecisionRules._target_violation_share(200.0, 150.0) ≈ 0.25
        end
    end

    @testset "StateConditionedPolicy" begin
        # Test construction
        n_uncertainty = 5
        n_state = 3
        n_output = 3
        layers = [8, 8]
        
        policy = state_conditioned_policy(n_uncertainty, n_state, n_output, layers; 
                                          activation=sigmoid, encoder_type=Flux.LSTM)
        
        @test policy.n_uncertainty == n_uncertainty
        @test policy.n_state == n_state
        
        # Test forward pass
        Flux.reset!(policy)
        input = rand(Float32, n_uncertainty + n_state)
        output = policy(input)
        @test length(output) == n_output
        
        # Test sequential calls (recurrent behavior)
        Flux.reset!(policy)
        prev_state = rand(Float32, n_state)
        for t in 1:5
            uncertainty = rand(Float32, n_uncertainty)
            input = vcat(uncertainty, prev_state)
            next_state = policy(input)
            @test length(next_state) == n_output
            prev_state = next_state
        end

        # Test the encoder's recurrent state is carried across calls: repeating the
        # same input twice in a row must give different outputs (state changed after
        # the first call), and Flux.reset! must restore the initial-state output.
        Flux.reset!(policy)
        repeated_input = rand(Float32, n_uncertainty + n_state)
        out1 = policy(repeated_input)
        out2 = policy(repeated_input)
        @test out1 != out2
        Flux.reset!(policy)
        @test policy(repeated_input) ≈ out1

        # Test gradient computation with Flux.gradient
        function test_loss(m, n_uncertainty, n_state)
            Flux.reset!(m)
            total = 0.0f0
            prev_state = rand(Float32, n_state)
            for t in 1:3
                uncertainty = rand(Float32, n_uncertainty)
                input = vcat(uncertainty, prev_state)
                next_state = m(input)
                total += sum(next_state)
                prev_state = next_state
            end
            return total
        end
        
        loss, grads = Flux.withgradient(policy) do m
            test_loss(m, n_uncertainty, n_state)
        end
        
        @test loss > 0
        @test grads[1] !== nothing
        @test grads[1].encoder !== nothing
        @test grads[1].combiner !== nothing
        
        # Test Flux.update! works (this was the original bug)
        opt_state = Flux.setup(Flux.Adam(0.01), policy)
        Flux.update!(opt_state, policy, grads[1])
        
        # Test single layer encoder
        policy_single = state_conditioned_policy(n_uncertainty, n_state, n_output, [8]; 
                                                  activation=relu, encoder_type=Flux.LSTM)
        Flux.reset!(policy_single)
        output_single = policy_single(rand(Float32, n_uncertainty + n_state))
        @test length(output_single) == n_output
        
        # Test empty layers (edge case)
        policy_empty = state_conditioned_policy(n_uncertainty, n_state, n_output, Int[];
                                                 activation=Base.identity, encoder_type=Flux.LSTM)
        Flux.reset!(policy_empty)
        output_empty = policy_empty(rand(Float32, n_uncertainty + n_state))
        @test length(output_empty) == n_output

        # Regression: Float64 input must not break Zygote gradient through LSTM
        # (solver feeds Float64 into a Float32 LSTM; without _state_eltype cast the
        # recurrent state drifts to Float64 and triggers a Zygote codegen bug)
        @testset "Float64 input regression" begin
            policy_f64 = state_conditioned_policy(n_uncertainty, n_state, n_output, [8];
                activation=sigmoid, encoder_type=Flux.LSTM)

            Flux.reset!(policy_f64)
            input_f64 = rand(Float64, n_uncertainty + n_state)
            out_f64 = policy_f64(input_f64)
            @test length(out_f64) == n_output
            @test eltype(out_f64) == Float32

            function f64_loss(m)
                Flux.reset!(m)
                total = 0.0
                prev = rand(Float64, n_state)
                for _ in 1:3
                    u = rand(Float64, n_uncertainty)
                    next = m(vcat(u, prev))
                    total += sum(next)
                    prev = Float64.(next)
                end
                return total
            end

            loss_val, grads_f64 = Flux.withgradient(policy_f64) do m
                f64_loss(m)
            end
            @test isfinite(loss_val)
            @test grads_f64[1] !== nothing
            @test grads_f64[1].encoder !== nothing

            opt_f64 = Flux.setup(Flux.Adam(0.01), policy_f64)
            Flux.update!(opt_f64, policy_f64, grads_f64[1])
        end
    end

    @testset "Multiple Shooting" begin
        # Test setup_shooting_windows
        @testset "setup_shooting_windows" begin
            Random.seed!(456)
            
            # Create 6 subproblems to test windowing with multiple windows
            num_stages = 6
            subproblems = Vector{JuMP.Model}(undef, num_stages)
            state_params_in = Vector{Vector{Any}}(undef, num_stages)
            state_params_out = Vector{Vector{Tuple{Any, VariableRef}}}(undef, num_stages)
            uncertainty_samples = Vector{Vector{Tuple{VariableRef, Vector{Float64}}}}(undef, num_stages)
            
            for t in 1:num_stages
                subproblems[t] = DiffOpt.diff_model(Ipopt.Optimizer)
                @variable(subproblems[t], x[1:3] >= 0)
                @variable(subproblems[t], state_in in MOI.Parameter(1.0))
                @variable(subproblems[t], uncertainty in MOI.Parameter(0.5))
                @variable(subproblems[t], state_out in MOI.Parameter(1.0))
                @variable(subproblems[t], state_out_var)
                @constraint(subproblems[t], sum(x) >= state_in + uncertainty)
                @constraint(subproblems[t], state_out_var == sum(x[1:2]))
                @constraint(subproblems[t], state_out_var >= state_out - 3.0)
                @constraint(subproblems[t], state_out_var <= state_out + 3.0)
                @objective(subproblems[t], Min, sum(x) + 5 * (state_out - state_out_var)^2)
                
                state_params_in[t] = [state_in]
                state_params_out[t] = [(state_out, state_out_var)]
                uncertainty_samples[t] = [(subproblems[t][:uncertainty], [0.3, 0.5, 0.7])]
            end
            
            initial_state = [2.0]
            window_size = 2
            
            # Test setup_shooting_windows
            windows = DecisionRules.setup_shooting_windows(
                subproblems,
                state_params_in,
                state_params_out,
                initial_state,
                uncertainty_samples;
                window_size=window_size,
                model_factory=() -> DiffOpt.nonlinear_diff_model(Ipopt.Optimizer)
            )
            
            @test length(windows) == 3  # 6 stages / 2 window_size = 3 windows
            
            # Verify each window
            for (w, window) in enumerate(windows)
                @test window.model !== nothing
                @test window.stage_range == ((w-1)*window_size + 1):min(w*window_size, num_stages)
                @test length(window.state_out_params) == length(window.stage_range)
                @test length(window.uncertainty_params) == length(window.stage_range)
                
                # Verify we can solve the window model
                optimize!(window.model)
                @test termination_status(window.model) in [MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED]
            end
            
            # Test with odd window size that doesn't divide evenly
            windows_odd = DecisionRules.setup_shooting_windows(
                subproblems,
                state_params_in,
                state_params_out,
                initial_state,
                uncertainty_samples;
                window_size=4,  # 6 stages / 4 = 2 windows, last window has 2 stages
                model_factory=() -> DiffOpt.nonlinear_diff_model(Ipopt.Optimizer)
            )
            
            @test length(windows_odd) == 2
            @test windows_odd[1].stage_range == 1:4
            @test windows_odd[2].stage_range == 5:6
        end
        
        # Test predict_window_targets
        @testset "predict_window_targets" begin
            # Simple decision rule that adds uncertainty to state
            simple_rule(x) = x[1:1] .+ x[2:2]  # state + uncertainty (uncertainty first, then state)
            
            initial_state = Float32[0.0]
            uncertainties = [Float32[1.0], Float32[2.0], Float32[3.0]]
            
            targets = DecisionRules.predict_window_targets(simple_rule, initial_state, uncertainties)
            
            # targets[1] = simple_rule([1, 0]) = [1]  (first uncertainty + initial state)
            # targets[2] = simple_rule([2, 1]) = [3]  (second uncertainty + targets[1])
            # targets[3] = simple_rule([3, 3]) = [6]  (third uncertainty + targets[2])
            @test length(targets) == 3  # One target per stage
            @test targets[1] ≈ [1.0f0]
            @test targets[2] ≈ [3.0f0]
            @test targets[3] ≈ [6.0f0]
        end
        
        # Test gradients flow through predict_window_targets
        @testset "predict_window_targets gradients" begin
            # Use a simple neural network
            m = Chain(Dense(2, 1, identity))
            
            initial_state = Float32[1.0]
            uncertainties = [Float32[0.5], Float32[0.3]]
            
            loss, grads = Flux.withgradient(m) do model
                targets = DecisionRules.predict_window_targets(model, initial_state, uncertainties)
                sum(sum.(targets))  # Simple loss to get gradients
            end
            
            @test loss != 0  # Should have some value
            @test grads[1] !== nothing
        end
        
        # Test solve_window
        @testset "solve_window" begin
            Random.seed!(789)
            
            # Create a simple 2-stage window model
            num_stages = 2
            subproblems = Vector{JuMP.Model}(undef, num_stages)
            state_params_in = Vector{Vector{Any}}(undef, num_stages)
            state_params_out = Vector{Vector{Tuple{Any, VariableRef}}}(undef, num_stages)
            uncertainty_samples = Vector{Vector{Tuple{VariableRef, Vector{Float64}}}}(undef, num_stages)
            
            for t in 1:num_stages
                subproblems[t] = DiffOpt.diff_model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
                @variable(subproblems[t], x[1:4] >= 0)
                @variable(subproblems[t], state_in in MOI.Parameter(1.0))
                @variable(subproblems[t], uncertainty in MOI.Parameter(0.5))
                @variable(subproblems[t], state_out in MOI.Parameter(1.0))
                @variable(subproblems[t], state_out_var)
                @constraint(subproblems[t], sum(x) >= state_in + uncertainty)
                @constraint(subproblems[t], state_out_var == sum(x[1:2]))
                @constraint(subproblems[t], state_out_var >= state_out - 5.0)
                @constraint(subproblems[t], state_out_var <= state_out + 5.0)
                @objective(subproblems[t], Min, sum(x) + 10 * (state_out - state_out_var)^2)
                
                state_params_in[t] = [state_in]
                state_params_out[t] = [(state_out, state_out_var)]
                uncertainty_samples[t] = [(subproblems[t][:uncertainty], [0.2, 0.4, 0.6])]
            end
            
            initial_state = [1.5]
            
            # Setup windows
            windows = DecisionRules.setup_shooting_windows(
                subproblems,
                state_params_in,
                state_params_out,
                initial_state,
                uncertainty_samples;
                window_size=2,
                model_factory=() -> DiffOpt.nonlinear_diff_model(Ipopt.Optimizer)
            )
            
            @test length(windows) == 1  # Only one window for 2 stages with window_size=2
            
            window = windows[1]
            
            # Set uncertainty values
            for t in 1:num_stages
                for param in window.uncertainty_params[t]
                    set_parameter_value(param, 0.4)
                end
            end
            
            s_in = Float32[1.5]
            targets = [Float32[1.2], Float32[1.0]]  # Two targets for 2 stages
            
            # Solve window
            obj = DecisionRules.solve_window(
                window.model,
                window.state_in_params,
                window.state_out_params,
                s_in,
                targets
            )

            s_out = DecisionRules.get_last_realized_state(
                window.model,
                window.state_in_params,
                window.state_out_params,
                s_in,
                targets
            )
            
            @test obj > 0
            @test length(s_out) == 1  # State dimension is 1
            @test !isnan(s_out[1])
        end

        @testset "setup_shooting_windows converts non-parameter state/target" begin
            num_stages = 1
            subproblems = Vector{JuMP.Model}(undef, num_stages)
            state_params_in = Vector{Vector{Any}}(undef, num_stages)
            state_params_out = Vector{Vector{Tuple{Any, VariableRef}}}(undef, num_stages)
            uncertainty_samples = Vector{Vector{Tuple{VariableRef, Vector{Float64}}}}(undef, num_stages)

            subproblems[1] = DiffOpt.diff_model(Ipopt.Optimizer)
            @variable(subproblems[1], x)
            @variable(subproblems[1], state_in)
            @variable(subproblems[1], uncertainty in MOI.Parameter(0.1))
            @variable(subproblems[1], state_out)
            @variable(subproblems[1], state_out_var)
            # Extra free variable so Ipopt has >=1 degree of freedom (n_vars > n_eq_constraints);
            # otherwise it throws TOO_FEW_DOF before attempting to solve.
            @variable(subproblems[1], _dof_slack)
            @constraint(subproblems[1], state_out_var == state_in + uncertainty)
            @constraint(subproblems[1], x == state_out_var)
            @objective(subproblems[1], Min, x)

            state_params_in[1] = [state_in]
            state_params_out[1] = [(state_out, state_out_var)]
            uncertainty_samples[1] = [(subproblems[1][:uncertainty], [0.1])]

            windows = DecisionRules.setup_shooting_windows(
                subproblems,
                state_params_in,
                state_params_out,
                [1.0],
                uncertainty_samples;
                window_size=1,
                model_factory=() -> DiffOpt.conic_diff_model(Ipopt.Optimizer)
            )

            window = windows[1]
            @test all(JuMP.is_parameter, window.state_in_params)
            @test JuMP.is_parameter(window.state_out_params[1][1][1])

            for param in window.uncertainty_params[1]
                set_parameter_value(param, 0.1)
            end
            target_val = Float32[1.0 + 0.1]
            obj = DecisionRules.solve_window(
                window.model,
                window.state_in_params,
                window.state_out_params,
                Float32[1.0],
                [target_val]
            )
            @test isfinite(obj)
        end
        
        # Test solve_window gradients
        @testset "solve_window gradients" begin
            Random.seed!(321)
            
            num_stages = 2
            subproblems = Vector{JuMP.Model}(undef, num_stages)
            state_params_in = Vector{Vector{Any}}(undef, num_stages)
            state_params_out = Vector{Vector{Tuple{Any, VariableRef}}}(undef, num_stages)
            uncertainty_samples = Vector{Vector{Tuple{VariableRef, Vector{Float64}}}}(undef, num_stages)
            
            for t in 1:num_stages
                subproblems[t] = DiffOpt.diff_model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
                @variable(subproblems[t], x[1:4] >= 0)
                @variable(subproblems[t], state_in in MOI.Parameter(1.0))
                @variable(subproblems[t], uncertainty in MOI.Parameter(0.5))
                @variable(subproblems[t], state_out in MOI.Parameter(1.0))
                @variable(subproblems[t], state_out_var)
                @constraint(subproblems[t], sum(x) >= state_in + uncertainty)
                @constraint(subproblems[t], state_out_var == sum(x[1:2]))
                @constraint(subproblems[t], state_out_var >= state_out - 5.0)
                @constraint(subproblems[t], state_out_var <= state_out + 5.0)
                @objective(subproblems[t], Min, sum(x) + 10 * (state_out - state_out_var)^2)
                
                state_params_in[t] = [state_in]
                state_params_out[t] = [(state_out, state_out_var)]
                uncertainty_samples[t] = [(subproblems[t][:uncertainty], [0.3, 0.5, 0.7])]
            end
            
            initial_state = [1.5]
            
            windows = DecisionRules.setup_shooting_windows(
                subproblems,
                state_params_in,
                state_params_out,
                initial_state,
                uncertainty_samples;
                window_size=2,
                model_factory=() -> DiffOpt.nonlinear_diff_model(Ipopt.Optimizer)
            )
            
            window = windows[1]
            
            # Set uncertainty values
            for t in 1:num_stages
                for param in window.uncertainty_params[t]
                    set_parameter_value(param, 0.4)
                end
            end
            
            # Test gradients using a neural network
            nn = Chain(Dense(2, 4, relu), Dense(4, 1))
            
            s_in = Float32[1.5]
            uncertainties_vec = [Float32[0.4], Float32[0.4]]
            
            loss, grads = Flux.withgradient(nn) do model
                targets = DecisionRules.predict_window_targets(model, s_in, uncertainties_vec)
                obj = DecisionRules.solve_window(
                    window.model,
                    window.state_in_params,
                    window.state_out_params,
                    s_in,
                    targets
                )
                return Float32(obj)
            end
            
            @test loss > 0
            @test grads[1] !== nothing  # Gradients should flow
        end
        
        # Test full multiple shooting simulation
        @testset "simulate_multiple_shooting" begin
            Random.seed!(111)
            
            num_stages = 4
            subproblems = Vector{JuMP.Model}(undef, num_stages)
            state_params_in = Vector{Vector{Any}}(undef, num_stages)
            state_params_out = Vector{Vector{Tuple{Any, VariableRef}}}(undef, num_stages)
            uncertainty_samples = Vector{Vector{Tuple{VariableRef, Vector{Float64}}}}(undef, num_stages)
            
            for t in 1:num_stages
                subproblems[t] = DiffOpt.diff_model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
                @variable(subproblems[t], x[1:4] >= 0)
                @variable(subproblems[t], state_in in MOI.Parameter(1.0))
                @variable(subproblems[t], uncertainty in MOI.Parameter(0.5))
                @variable(subproblems[t], state_out in MOI.Parameter(1.0))
                @variable(subproblems[t], state_out_var)
                @constraint(subproblems[t], sum(x) >= state_in + uncertainty)
                @constraint(subproblems[t], state_out_var == sum(x[1:2]))
                @constraint(subproblems[t], state_out_var >= state_out - 5.0)
                @constraint(subproblems[t], state_out_var <= state_out + 5.0)
                @objective(subproblems[t], Min, sum(x) + 10 * (state_out - state_out_var)^2)
                
                state_params_in[t] = [state_in]
                state_params_out[t] = [(state_out, state_out_var)]
                uncertainty_samples[t] = [(subproblems[t][:uncertainty], [0.3, 0.5, 0.7])]
            end
            
            initial_state = [1.5]
            
            # Setup windows (2 windows of 2 stages each)
            windows = DecisionRules.setup_shooting_windows(
                subproblems,
                state_params_in,
                state_params_out,
                initial_state,
                uncertainty_samples;
                window_size=2,
                model_factory=() -> DiffOpt.nonlinear_diff_model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
            )
            
            @test length(windows) == 2
            
            # Sample uncertainties
            uncertainty_sample = [[[(windows[w].uncertainty_params[t][1], 0.4)] 
                for t in 1:length(windows[w].stage_range)] 
                for w in 1:length(windows)]
            # Flatten to per-stage format
            flat_uncertainty_sample = vcat(uncertainty_sample...)
            uncertainties_vec = [Float32[0.4] for _ in 1:num_stages]
            
            # Simple decision rule
            simple_rule(x) = x[2:2] .+ 0.1f0  # state_out = state + 0.1
            
            # Manually set uncertainty values in windows
            for (w, window) in enumerate(windows)
                for t in 1:length(window.stage_range)
                    for param in window.uncertainty_params[t]
                        set_parameter_value(param, 0.4)
                    end
                end
            end
            
            # Simulate
            total_obj = DecisionRules.simulate_multiple_shooting(
                windows,
                simple_rule,
                Float32.(initial_state),
                flat_uncertainty_sample,
                uncertainties_vec
            )
            
            @test total_obj > 0
        end
        
        # Test gradients flow across windows in simulate_multiple_shooting
        @testset "simulate_multiple_shooting gradients" begin
            Random.seed!(222)
            
            num_stages = 4
            subproblems = Vector{JuMP.Model}(undef, num_stages)
            state_params_in = Vector{Vector{Any}}(undef, num_stages)
            state_params_out = Vector{Vector{Tuple{Any, VariableRef}}}(undef, num_stages)
            uncertainty_samples = Vector{Vector{Tuple{VariableRef, Vector{Float64}}}}(undef, num_stages)
            
            for t in 1:num_stages
                subproblems[t] = DiffOpt.nonlinear_diff_model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
                @variable(subproblems[t], x[1:4] >= 0)
                @variable(subproblems[t], state_in in MOI.Parameter(1.0))
                @variable(subproblems[t], uncertainty in MOI.Parameter(0.5))
                @variable(subproblems[t], state_out in MOI.Parameter(1.0))
                @variable(subproblems[t], state_out_var)
                @constraint(subproblems[t], sum(x) >= state_in + uncertainty)
                @constraint(subproblems[t], state_out_var == sum(x[1:2]))
                @constraint(subproblems[t], state_out_var >= state_out - 5.0)
                @constraint(subproblems[t], state_out_var <= state_out + 5.0)
                @objective(subproblems[t], Min, sum(x) + 10 * (state_out - state_out_var)^2)
                
                state_params_in[t] = [state_in]
                state_params_out[t] = [(state_out, state_out_var)]
                uncertainty_samples[t] = [(subproblems[t][:uncertainty], [0.3, 0.5, 0.7])]
            end
            
            initial_state = [1.5]
            
            windows = DecisionRules.setup_shooting_windows(
                subproblems,
                state_params_in,
                state_params_out,
                initial_state,
                uncertainty_samples;
                window_size=2,
                model_factory=() -> DiffOpt.nonlinear_diff_model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
            )
            
            # Set uncertainty values
            for window in windows
                for t in 1:length(window.stage_range)
                    for param in window.uncertainty_params[t]
                        set_parameter_value(param, 0.4)
                    end
                end
            end
            
            flat_uncertainty_sample = [[(windows[ceil(Int, t/2)].uncertainty_params[mod1(t, 2)][1], 0.4)] for t in 1:num_stages]
            uncertainties_vec = [Float32[0.4] for _ in 1:num_stages]
            
            # Use neural network
            nn = Chain(Dense(2, 4, relu), Dense(4, 1))
            
            loss, grads = Flux.withgradient(nn) do model
                DecisionRules.simulate_multiple_shooting(
                    windows,
                    model,
                    Float32.(initial_state),
                    flat_uncertainty_sample,
                    uncertainties_vec
                )
            end
            
            @test loss > 0
            @test grads[1] !== nothing  # Gradients should flow across windows
        end

        @testset "multiple_shooting_vector_uncertainties" begin
            num_stages = 2
            subproblems = Vector{JuMP.Model}(undef, num_stages)
            state_params_in = Vector{Vector{Any}}(undef, num_stages)
            state_params_out = Vector{Vector{Tuple{Any, VariableRef}}}(undef, num_stages)
            uncertainty_samples = Vector{Vector{Tuple{VariableRef, Vector{Float64}}}}(undef, num_stages)

            for t in 1:num_stages
                subproblems[t] = DiffOpt.diff_model(Ipopt.Optimizer)
                @variable(subproblems[t], x)
                @variable(subproblems[t], state_in in MOI.Parameter(1.0))
                @variable(subproblems[t], u1 in MOI.Parameter(0.1))
                @variable(subproblems[t], u2 in MOI.Parameter(0.2))
                @variable(subproblems[t], state_out in MOI.Parameter(1.0))
                @variable(subproblems[t], state_out_var)
                @constraint(subproblems[t], state_out_var == state_in + u1 + u2)
                @constraint(subproblems[t], x == state_out_var)
                @objective(subproblems[t], Min, x)

                state_params_in[t] = [state_in]
                state_params_out[t] = [(state_out, state_out_var)]
                uncertainty_samples[t] = [(u1, [0.1, 0.2, 0.3]), (u2, [0.2, 0.4, 0.6])]
            end

            initial_state = [1.0]
            windows = DecisionRules.setup_shooting_windows(
                subproblems,
                state_params_in,
                state_params_out,
                Float64.(initial_state),
                uncertainty_samples;
                window_size=2,
                model_factory=() -> DiffOpt.conic_diff_model(Ipopt.Optimizer)
            )

            # Policy expects flat [u1, u2, state] input
            decision_rule(x) = [x[3] + x[1] + x[2]]

            uncertainty_sample = DecisionRules.sample(uncertainty_samples)
            uncertainties_vec = [[Float32(u[2]) for u in stage_u] for stage_u in uncertainty_sample]

            obj = DecisionRules.simulate_multiple_shooting(
                windows,
                decision_rule,
                Float32.(initial_state),
                uncertainty_sample,
                uncertainties_vec
            )

            @test obj > 0
        end

        @testset "train_multiple_shooting samples are ignored by AD" begin
            num_stages = 1
            subproblems = Vector{JuMP.Model}(undef, num_stages)
            state_params_in = Vector{Vector{Any}}(undef, num_stages)
            state_params_out = Vector{Vector{Tuple{Any, VariableRef}}}(undef, num_stages)
            uncertainty_samples = Vector{Vector{Tuple{VariableRef, Vector{Float64}}}}(undef, num_stages)

            subproblems[1] = DiffOpt.diff_model(Ipopt.Optimizer)
            @variable(subproblems[1], x)
            @variable(subproblems[1], state_in in MOI.Parameter(0.0))
            @variable(subproblems[1], uncertainty in MOI.Parameter(0.0))
            @variable(subproblems[1], state_out in MOI.Parameter(0.0))
            @variable(subproblems[1], state_out_var)
            # Extra free variable so Ipopt has >=1 degree of freedom (n_vars > n_eq_constraints);
            # otherwise it throws TOO_FEW_DOF before attempting to solve.
            @variable(subproblems[1], _dof_slack)
            @constraint(subproblems[1], x == state_in + uncertainty)
            @constraint(subproblems[1], state_out_var == x)
            @objective(subproblems[1], Min, x)

            state_params_in[1] = [state_in]
            state_params_out[1] = [(state_out, state_out_var)]
            uncertainty_samples[1] = [(uncertainty, [0.1, 0.2, 0.3])]

            # Model that returns uncertainty + state (inputs are [uncertainty; state])
            model = Dense(2, 1; bias=false)
            model.weight .= 1.0f0

            windows = DecisionRules.setup_shooting_windows(
                subproblems,
                state_params_in,
                state_params_out,
                [1.0],
                uncertainty_samples;
                window_size=1,
                model_factory=() -> DiffOpt.conic_diff_model(Ipopt.Optimizer),
            )

            DecisionRules.train_multiple_shooting(
                model,
                [1.0],
                windows,
                () -> uncertainty_samples;
                num_batches=1,
                num_train_per_batch=1,
                optimizer=Flux.Descent(0.0),
                record_loss=(iter, m, loss, tag) -> false,
            )

            @test true  # no mutation error during AD
        end

        @testset "consistent_state_paths_across_methods" begin
            function build_consistent_subproblems(num_stages)
                subproblems = Vector{JuMP.Model}(undef, num_stages)
                state_params_in = Vector{Vector{Any}}(undef, num_stages)
                state_params_out = Vector{Vector{Tuple{Any, VariableRef}}}(undef, num_stages)
                uncertainty_samples = Vector{Vector{Tuple{VariableRef, Vector{Float64}}}}(undef, num_stages)

                for t in 1:num_stages
                    subproblems[t] = DiffOpt.diff_model(Ipopt.Optimizer)
                    @variable(subproblems[t], reservoir_in)
                    @variable(subproblems[t], reservoir_out)
                    @variable(subproblems[t], inflow)
                    @variable(subproblems[t], deficit >= 0)
                    @variable(subproblems[t], release >= 0)
                    @constraint(subproblems[t], reservoir_out == reservoir_in + inflow - release)
                    @constraint(subproblems[t], reservoir_out >= 0)
                    @objective(subproblems[t], Min, release + 0.1 * reservoir_out + 100.0 * deficit)

                    state_in_param = DecisionRules.variable_to_parameter(subproblems[t], reservoir_in)
                    state_out_param, realized_out = DecisionRules.variable_to_parameter(
                        subproblems[t], reservoir_out; deficit=deficit
                    )
                    inflow_param = DecisionRules.variable_to_parameter(subproblems[t], inflow)

                    state_params_in[t] = [state_in_param]
                    state_params_out[t] = [(state_out_param, realized_out)]
                    uncertainty_samples[t] = [(inflow_param, [0.1 * t, 0.2 * t])]
                end

                return subproblems, state_params_in, state_params_out, uncertainty_samples
            end

            num_stages = 28
            initial_state = [1.0]

            decision_rule(x) = [x[2] + 0.5 * x[1]]  # next_state = prev_state + 0.5*uncertainty

            # Per-stage simulation
            subproblems_s, state_in_s, state_out_s, uncertainty_samples_s =
                build_consistent_subproblems(num_stages)
            # Use a single draw of uncertainty values across all three methods
            Random.seed!(1234)
            base_sample = DecisionRules.sample(uncertainty_samples_s)
            base_values = [[u[2] for u in stage_u] for stage_u in base_sample]
            uncertainties_s = [[(stage_u[i][1], base_values[t][i]) for i in eachindex(stage_u)]
                               for (t, stage_u) in enumerate(uncertainty_samples_s)]
            obj_stage = DecisionRules.simulate_multistage(
                subproblems_s,
                state_in_s,
                state_out_s,
                initial_state,
                uncertainties_s,
                decision_rule,
            )

            states_stage = Vector{Vector{Float64}}(undef, num_stages + 1)
            states_stage[1] = initial_state
            # decisions_stage = Vector{Float64}(undef, num_stages)
            for t in 1:num_stages
                states_stage[t + 1] = [value(state_out_s[t][1][2])]
                # decisions_stage[t] = value(subproblems_s[t][:x])
            end

            # Deterministic equivalent
            subproblems_d, state_in_d, state_out_d, uncertainty_samples_d =
                build_consistent_subproblems(num_stages)
            det_model = JuMP.Model(Ipopt.Optimizer)
            det_model, uncertainty_samples_d = DecisionRules.deterministic_equivalent!(
                det_model,
                subproblems_d,
                state_in_d,
                state_out_d,
                Float64.(initial_state),
                uncertainty_samples_d,
            )
            uncertainties_d = [[(stage_u[i][1], base_values[t][i]) for i in eachindex(stage_u)]
                               for (t, stage_u) in enumerate(uncertainty_samples_d)]
            states_policy = DecisionRules.simulate_states(initial_state, uncertainties_d, decision_rule)
            obj_det = DecisionRules.simulate_multistage(det_model, state_in_d, state_out_d, uncertainties_d, states_policy)

            states_det = Vector{Vector{Float64}}(undef, num_stages + 1)
            states_det[1] = initial_state
            decisions_det = Float64[]
            # x_det = DecisionRules.find_variables(det_model, ["x"])
            for t in 1:num_stages
                states_det[t + 1] = [value(state_out_d[t][1][2])]
                # push!(decisions_det, value(x_det[t]))
            end

            # Multiple shooting
            subproblems_w, state_in_w, state_out_w, uncertainty_samples_w =
                build_consistent_subproblems(num_stages)
            windows = DecisionRules.setup_shooting_windows(
                subproblems_w,
                state_in_w,
                state_out_w,
                Float64.(initial_state),
                uncertainty_samples_w;
                window_size=6,
                model_factory=() -> DiffOpt.conic_diff_model(Ipopt.Optimizer),
            )
            # Variable count checks
            stage_var_count = sum(length.(all_variables.(subproblems_s)))
            det_var_count = length(all_variables(det_model))
            windows_var_count = sum(length(all_variables(w.model)) for w in windows)
            state_dim = length(state_in_s[1])
            expected_det = stage_var_count - (num_stages - 1) * state_dim
            expected_windows = stage_var_count - (num_stages - length(windows)) * state_dim
            @test det_var_count == expected_det
            @test windows_var_count == expected_windows
            uncertainties_w = [[(stage_u[i][1], base_values[t][i]) for i in eachindex(stage_u)]
                               for (t, stage_u) in enumerate(uncertainty_samples_w)]
            uncertainties_vec = [[Float32(u[2]) for u in stage_u] for stage_u in uncertainties_w]

            obj_shoot = DecisionRules.simulate_multiple_shooting(
                windows,
                decision_rule,
                Float32.(initial_state),
                uncertainties_w,
                uncertainties_vec
            )

            states_shoot = Vector{Vector{Float64}}()
            push!(states_shoot, Float64.(initial_state))
            decisions_shoot = Float64[]
            current_state = Float64.(initial_state)
            for window in windows
                window_range = window.stage_range
                window_uncertainties_vec = uncertainties_vec[window_range]
                targets = DecisionRules.predict_window_targets(
                    decision_rule,
                    current_state,
                    window_uncertainties_vec,
                )
                DecisionRules.set_window_uncertainties!(window, uncertainties_w)
                DecisionRules.solve_window(
                    window.model,
                    window.state_in_params,
                    window.state_out_params,
                    current_state,
                    targets,
                )

                # x_win = DecisionRules.find_variables(window.model, ["x"])
                for local_t in 1:length(window_range)
                    push!(states_shoot, [value(window.state_out_params[local_t][1][2])])
                    # push!(decisions_shoot, value(x_win[local_t]))
                end
                current_state = states_shoot[end]
            end

            @test length(states_stage) == length(states_det) == length(states_shoot)
            for t in 1:length(states_stage)
                @test states_stage[t][1] ≈ states_det[t][1] rtol=1.0e-4
                @test states_stage[t][1] ≈ states_shoot[t][1] rtol=1.0e-4   
            end

            @test obj_stage ≈ obj_det rtol=1.0e-4
            @test obj_stage ≈ obj_shoot rtol=1.0e-4

            # @test length(decisions_stage) == length(decisions_det) == length(decisions_shoot)
            # for t in 1:length(decisions_stage)
            #     @test decisions_stage[t] ≈ decisions_det[t]
            #     @test decisions_stage[t] ≈ decisions_shoot[t]
            # end
        end
    end

    @testset "dense_multilayer_nn" begin
        # Dense layers
        m = dense_multilayer_nn(3, 2, [8, 4]; activation=relu, dense=Dense)
        @test size(m(rand(Float32, 3))) == (2,)

        # LSTM layers (Flux 0.16 LSTM requires batched 2D+ input)
        m_lstm = dense_multilayer_nn(3, 2, [8, 4]; dense=LSTM)
        Flux.reset!(m_lstm)
        @test size(m_lstm(rand(Float32, 3, 1))) == (2, 1)

        # Empty layers (single layer)
        m_empty = dense_multilayer_nn(3, 2, Int[]; activation=relu, dense=Dense)
        @test size(m_empty(rand(Float32, 3))) == (2,)

        # Empty layers LSTM
        m_empty_lstm = dense_multilayer_nn(3, 2, Int[]; dense=LSTM)
        Flux.reset!(m_empty_lstm)
        @test size(m_empty_lstm(rand(Float32, 3, 1))) == (2, 1)

        # Single hidden layer
        m_single = dense_multilayer_nn(3, 2, [8]; activation=tanh, dense=Dense)
        @test size(m_single(rand(Float32, 3))) == (2,)

        # Single hidden layer LSTM
        m_single_lstm = dense_multilayer_nn(3, 2, [8]; dense=LSTM)
        Flux.reset!(m_single_lstm)
        @test size(m_single_lstm(rand(Float32, 3, 1))) == (2, 1)
    end

    @testset "policy_input_dim" begin
        @test policy_input_dim(5, 3) == 8
        @test policy_input_dim(0, 4) == 4

        uncertainty_samples = [[(nothing, [1.0, 2.0]), (nothing, [3.0])]]
        initial_state = [0.0, 0.0, 0.0]
        @test policy_input_dim(uncertainty_samples, initial_state) == 5
    end

    @testset "normalize_recur_state" begin
        plain = (a=1.0, b=[2.0, 3.0])
        @test normalize_recur_state(plain) == plain

        s0 = [0.0f0, 0.0f0]
        cell_nt = (state0=s0, Wi=[1.0f0], Wh=[2.0f0])
        recur_nt = (cell=cell_nt, state=[999.0f0, 999.0f0])
        result = normalize_recur_state(recur_nt)
        @test result.state == s0
        @test result.cell === cell_nt

        nested = (layer1=recur_nt, layer2=(x=42,))
        result_nested = normalize_recur_state(nested)
        @test result_nested.layer1.state == s0
        @test result_nested.layer2 == (x=42,)

        @test normalize_recur_state((1, 2, 3)) == (1, 2, 3)
        @test normalize_recur_state(42.0) == 42.0
    end

    @testset "simulate_states" begin
        uncertainties = [[(nothing, 1.0)], [(nothing, 2.0)]]
        initial_state = [5.0]
        dr(x) = [x[1] + x[2]]  # uncertainty + state
        states = DecisionRules.simulate_states(initial_state, uncertainties, dr)
        @test length(states) == 3
        @test states[1] == [5.0]
        @test states[2] ≈ [6.0]
        @test states[3] ≈ [8.0]

        # Vector of decision rules
        dr1(x) = [x[1] + x[2]]
        dr2(x) = [x[1] * 2 + x[2]]
        states2 = DecisionRules.simulate_states(initial_state, uncertainties, [dr1, dr2])
        @test states2[1] == [5.0]
        @test states2[2] ≈ [6.0]
        @test states2[3] ≈ [10.0]
    end

    @testset "StallingCriterium" begin
        sc = StallingCriterium(3, 100.0, 0)
        @test sc(1, nothing, 90.0) == false
        @test sc.best_loss == 90.0
        @test sc.stall_count == 0

        @test sc(2, nothing, 95.0) == false
        @test sc.stall_count == 1
        @test sc(3, nothing, 96.0) == false
        @test sc.stall_count == 2
        @test sc(4, nothing, 97.0) == true
        @test sc.stall_count == 3

        sc2 = StallingCriterium(2, 100.0, 0)
        @test sc2(1, nothing, 50.0) == false  # improvement
        @test sc2(2, nothing, 60.0) == false   # stall 1
        @test sc2(3, nothing, 40.0) == false   # improvement resets
        @test sc2.stall_count == 0
    end

    @testset "SaveBest" begin
        tmpdir = mktempdir()
        path = joinpath(tmpdir, "test_model.jld2")
        sb = SaveBest(100.0, path)

        m = Chain(Dense(2, 3))
        @test sb(1, m, 110.0) == false
        @test !isfile(path)

        @test sb(2, m, 90.0) == false
        @test isfile(path)
        @test sb.best_loss == 90.0

        @test sb(3, m, 95.0) == false
        @test sb.best_loss == 90.0
    end

    @testset "_linear_objective_coefficient" begin
        m = Model()
        @variable(m, a)
        @variable(m, b)
        @objective(m, Min, 3.0 * a + 5.0 * b)
        @test DecisionRules._linear_objective_coefficient(m, a) == 3.0
        @test DecisionRules._linear_objective_coefficient(m, b) == 5.0

        m2 = Model()
        @variable(m2, x2)
        @variable(m2, y2)
        @objective(m2, Min, x2^2 + 2.0 * y2)
        @test DecisionRules._linear_objective_coefficient(m2, y2) == 2.0
        @test DecisionRules._linear_objective_coefficient(m2, x2) == 0.0
    end

    @testset "find_variables" begin
        m = Model()
        @variable(m, reservoir_in[1:3])
        @variable(m, reservoir_out[1:3])
        @variable(m, other_var)
        # Single result (unique match)
        found_single = find_variables(m, ["other_var"])
        @test length(found_single) == 1
        # Multiple results with indexed variables
        found = find_variables(m, ["reservoir_in"])
        @test length(found) == 3
        @test all(occursin("reservoir_in", JuMP.name(v)) for v in found)
    end

    @testset "get_objective_no_target_deficit (vector)" begin
        sp1 = build_subproblem(10; subproblem=DiffOpt.conic_diff_model(Ipopt.Optimizer))[1]
        sp2 = build_subproblem(10; state_i_val=1.0, state_out_val=9.0, subproblem=DiffOpt.conic_diff_model(Ipopt.Optimizer))[1]
        optimize!(sp1)
        optimize!(sp2)
        total = DecisionRules.get_objective_no_target_deficit([sp1, sp2])
        indiv = DecisionRules.get_objective_no_target_deficit(sp1) + DecisionRules.get_objective_no_target_deficit(sp2)
        @test total ≈ indiv
    end

    @testset "materialize_tangent edge cases" begin
        @test materialize_tangent(3.14) == 3.14
        @test materialize_tangent([1, 2, 3]) == [1, 2, 3]
        @test materialize_tangent(nothing) === nothing
        @test materialize_tangent(ChainRulesCore.NoTangent()) === nothing
        @test materialize_tangent(ChainRulesCore.ZeroTangent()) === nothing

        nt = (a=1.0, b=[2.0], c=nothing)
        @test materialize_tangent(nt) == (a=1.0, b=[2.0], c=nothing)

        t = (1.0, [2.0], nothing)
        @test materialize_tangent(t) == (1.0, [2.0], nothing)

        ref = Ref(42.0)
        @test materialize_tangent(ref) == 42.0
    end

    @testset "create_deficit! auto penalty" begin
        m = Model()
        @variable(m, x)
        @objective(m, Min, 100.0 * x)
        nd, d = create_deficit!(m, 2)
        @test length(d) == 2
        coef = coefficient(objective_function(m), nd)
        @test coef ≈ 100.0

        m2 = Model()
        @variable(m2, y)
        @objective(m2, Min, 50.0 * y)
        nd2, d2 = create_deficit!(m2, 1; penalty_l1=:auto)
        coef2 = coefficient(objective_function(m2), nd2)
        @test coef2 ≈ 50.0

        m3 = Model()
        @variable(m3, z)
        @objective(m3, Min, 75.0 * z)
        nd3, d3 = create_deficit!(m3, 1; penalty_l2=:auto)
        coef3 = coefficient(objective_function(m3), nd3)
        @test coef3 ≈ 75.0
    end

    @testset "MadNLP solver compatibility" begin
        @testset "compute_parameter_dual with MadNLP" begin
            model = Model(MadNLP.Optimizer)
            set_optimizer_attribute(model, "print_level", MadNLP.ERROR)
            @variable(model, x >= 0)
            @variable(model, p in MOI.Parameter(2.0))
            @constraint(model, x - p >= 0)
            @objective(model, Min, x)
            optimize!(model)
            @test termination_status(model) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]
            @test value(x) ≈ 2.0 atol=1e-3
            @test compute_parameter_dual(model, p) ≈ 1.0 rtol=0.2
        end

        @testset "simulate_stage with MadNLP" begin
            sp, si, so, sov, u = build_subproblem(10; subproblem=Model(MadNLP.Optimizer))
            set_optimizer_attribute(sp, "print_level", MadNLP.ERROR)
            state_param_in = Vector{Any}([si])
            state_param_out = Vector{Tuple{Any, VariableRef}}([(so, sov)])
            uncertainty_sample = [(u, 2.0)]
            obj = DecisionRules.simulate_stage(sp, state_param_in, state_param_out, uncertainty_sample, [5.0], [4.0])
            @test obj ≈ 210 rtol=0.1
        end

        @testset "create_deficit! with MadNLP" begin
            model = Model(MadNLP.Optimizer)
            set_optimizer_attribute(model, "print_level", MadNLP.ERROR)
            @variable(model, y >= 0)
            @objective(model, Min, 10 * y)
            @constraint(model, y >= 1)
            nd, d = create_deficit!(model, 2; penalty_l2=100.0)
            @constraint(model, d[1] == 2.0)
            @constraint(model, d[2] == 3.0)
            optimize!(model)
            @test termination_status(model) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]
            @test value(nd) ≈ 13.0 rtol=0.05
        end

        @testset "deterministic_equivalent with MadNLP" begin
            sp1, si1, so1, sov1, u1 = build_subproblem(10)
            sp2, si2, so2, sov2, u2 = build_subproblem(10; state_i_val=4.0, state_out_val=3.0, uncertainty_val=1.0)
            sps = [sp1, sp2]
            spi = Vector{Vector{Any}}(undef, 2); spi .= [[si1], [si2]]
            spo = Vector{Vector{Tuple{Any, VariableRef}}}(undef, 2)
            spo .= [[(so1, sov1)], [(so2, sov2)]]
            usamples = [[(u1, [2.0])], [(u2, [1.0])]]

            det_eq = Model(MadNLP.Optimizer)
            set_optimizer_attribute(det_eq, "print_level", MadNLP.ERROR)
            det_eq, usamples = DecisionRules.deterministic_equivalent!(
                det_eq, sps, spi, spo, [5.0], usamples)

            obj_val = DecisionRules.simulate_multistage(det_eq, spi, spo,
                sample(usamples), [[9.0], [7.], [4.0]])
            @test obj_val ≈ 359 rtol=0.1
        end

        @testset "train_multistage (subproblems) with MadNLP" begin
            sp1, si1, so1, sov1, u1 = build_subproblem(10; subproblem=Model(MadNLP.Optimizer))
            sp2, si2, so2, sov2, u2 = build_subproblem(10; state_i_val=1.0, state_out_val=9.0, subproblem=Model(MadNLP.Optimizer))
            set_optimizer_attribute(sp1, "print_level", MadNLP.ERROR)
            set_optimizer_attribute(sp2, "print_level", MadNLP.ERROR)
            sps = [sp1, sp2]
            spi = Vector{Vector{Any}}(undef, 2); spi .= [[si1], [si2]]
            spo = Vector{Vector{Tuple{Any, VariableRef}}}(undef, 2)
            spo .= [[(so1, sov1)], [(so2, sov2)]]
            usamples = [[(u1, [2.0])], [(u2, [1.0])]]

            Random.seed!(222)
            m = Chain(Dense(2, 4), Dense(4, 1))
            obj_before = DecisionRules.simulate_multistage(
                sps, spi, spo, [5.0], sample(usamples), m)
            train_multistage(m, [5.0], sps, spi, spo, usamples;
                num_batches=3, num_train_per_batch=1)
            obj_after = DecisionRules.simulate_multistage(
                sps, spi, spo, [5.0], sample(usamples), m)
            @test obj_after <= obj_before + 50.0  # some tolerance for stochastic training
        end

        @testset "train_multistage (det_eq) with MadNLP and penalty_schedule" begin
            sp1, si1, so1, sov1, u1 = build_subproblem(10)
            sp2, si2, so2, sov2, u2 = build_subproblem(10; state_i_val=4.0, state_out_val=3.0, uncertainty_val=1.0)
            spi = Vector{Vector{Any}}(undef, 2); spi .= [[si1], [si2]]
            spo = Vector{Vector{Tuple{Any, VariableRef}}}(undef, 2)
            spo .= [[(so1, sov1)], [(so2, sov2)]]
            usamples = [[(u1, [2.0])], [(u2, [1.0])]]

            det_eq = Model(MadNLP.Optimizer)
            set_optimizer_attribute(det_eq, "print_level", MadNLP.ERROR)
            set_optimizer_attribute(det_eq, "tol", 1e-6)
            det_eq, usamples_det = DecisionRules.deterministic_equivalent!(
                det_eq, [sp1, sp2], spi, spo, [5.0], usamples)

            Random.seed!(222)
            m = Chain(Dense(2, 10), Dense(10, 1))
            train_multistage(m, [5.0], det_eq, spi, spo, usamples_det;
                num_batches=4, num_train_per_batch=1,
                penalty_schedule=[(1, 2, 1.0), (3, 4, 3.0)])
            nd_copies = [v for v in all_variables(det_eq) if occursin("norm_deficit", JuMP.name(v))]
            @test all(coefficient(objective_function(det_eq), v) ≈ 3.0e4 for v in nd_copies)
        end

        @testset "StateConditionedPolicy + MadNLP det_eq training" begin
            sp1, si1, so1, sov1, u1 = build_subproblem(10)
            sp2, si2, so2, sov2, u2 = build_subproblem(10; state_i_val=4.0, state_out_val=3.0, uncertainty_val=1.0)
            spi = Vector{Vector{Any}}(undef, 2); spi .= [[si1], [si2]]
            spo = Vector{Vector{Tuple{Any, VariableRef}}}(undef, 2)
            spo .= [[(so1, sov1)], [(so2, sov2)]]
            usamples = [[(u1, [2.0])], [(u2, [1.0])]]

            det_eq = Model(MadNLP.Optimizer)
            set_optimizer_attribute(det_eq, "print_level", MadNLP.ERROR)
            set_optimizer_attribute(det_eq, "tol", 1e-6)
            det_eq, usamples_det = DecisionRules.deterministic_equivalent!(
                det_eq, [sp1, sp2], spi, spo, [5.0], usamples)

            Random.seed!(222)
            policy = state_conditioned_policy(1, 1, 1, [8];
                activation=sigmoid, encoder_type=Flux.LSTM)
            obj_before = DecisionRules.simulate_multistage(
                det_eq, spi, spo, [5.0],
                sample(usamples_det), policy)
            train_multistage(policy, [5.0], det_eq, spi, spo, usamples_det;
                num_batches=4, num_train_per_batch=1,
                penalty_schedule=:default_annealed)
            obj_after = DecisionRules.simulate_multistage(
                det_eq, spi, spo, [5.0],
                sample(usamples_det), policy)
            @test isfinite(obj_after)
        end

        @testset "full training pipeline with MadNLP det_eq" begin
            sp1, si1, so1, sov1, u1 = build_subproblem(10)
            sp2, si2, so2, sov2, u2 = build_subproblem(10; state_i_val=4.0, state_out_val=3.0, uncertainty_val=1.0)
            spi = Vector{Vector{Any}}(undef, 2); spi .= [[si1], [si2]]
            spo = Vector{Vector{Tuple{Any, VariableRef}}}(undef, 2)
            spo .= [[(so1, sov1)], [(so2, sov2)]]
            usamples = [[(u1, [2.0])], [(u2, [1.0])]]

            det_eq = Model(MadNLP.Optimizer)
            set_optimizer_attribute(det_eq, "print_level", MadNLP.ERROR)
            set_optimizer_attribute(det_eq, "tol", 1e-6)
            det_eq, usamples_det = DecisionRules.deterministic_equivalent!(
                det_eq, [sp1, sp2], spi, spo, [5.0], usamples)

            Random.seed!(222)
            m = Chain(Dense(2, 10), Dense(10, 1))
            obj_before = DecisionRules.simulate_multistage(
                det_eq, spi, spo, [5.0],
                sample(usamples_det), m)
            train_multistage(m, [5.0], det_eq, spi, spo, usamples_det;
                num_batches=5, num_train_per_batch=1,
                penalty_schedule=:default_annealed)
            obj_after = DecisionRules.simulate_multistage(
                det_eq, spi, spo, [5.0],
                sample(usamples_det), m)
            @test isfinite(obj_after)
            @test obj_after < obj_before + 100.0
        end
    end

    @testset "GPU (CUDA) solver" begin
        gpu_available = try
            @eval using CUDA
            CUDA.functional()
        catch
            false
        end

        if gpu_available
            @eval using CUDSS
            @eval using MadNLPGPU

            @testset "MadNLP+CUDSS GPU solve and duals" begin
                gpu_model = Model(MadNLP.Optimizer)
                set_optimizer_attribute(gpu_model, "array_type", CUDA.CuArray)
                set_optimizer_attribute(gpu_model, "linear_solver", MadNLPGPU.CUDSSSolver)
                set_optimizer_attribute(gpu_model, "print_level", MadNLP.ERROR)
                @variable(gpu_model, gx >= 0)
                @variable(gpu_model, gp in MOI.Parameter(3.0))
                @constraint(gpu_model, gx >= gp)
                @objective(gpu_model, Min, gx)
                optimize!(gpu_model)
                @test termination_status(gpu_model) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]
                @test value(gx) ≈ 3.0 atol=1e-3
                @test compute_parameter_dual(gpu_model, gp) ≈ -1.0 rtol=0.2
            end

            @testset "deterministic_equivalent GPU training" begin
                sp1, si1, so1, sov1, u1 = build_subproblem(10)
                sp2, si2, so2, sov2, u2 = build_subproblem(10; state_i_val=4.0, state_out_val=3.0, uncertainty_val=1.0)
                spi = Vector{Vector{Any}}(undef, 2); spi .= [[si1], [si2]]
                spo = Vector{Vector{Tuple{Any, VariableRef}}}(undef, 2)
                spo .= [[(so1, sov1)], [(so2, sov2)]]
                usamples = [[(u1, [2.0])], [(u2, [1.0])]]

                det_eq = Model(MadNLP.Optimizer)
                set_optimizer_attribute(det_eq, "array_type", CUDA.CuArray)
                set_optimizer_attribute(det_eq, "linear_solver", MadNLPGPU.CUDSSSolver)
                set_optimizer_attribute(det_eq, "print_level", MadNLP.ERROR)
                det_eq, usamples_det = DecisionRules.deterministic_equivalent!(
                    det_eq, [sp1, sp2], spi, spo, [5.0], usamples)

                Random.seed!(222)
                policy = state_conditioned_policy(1, 1, 1, [8];
                    activation=sigmoid, encoder_type=Flux.LSTM)
                obj_before = DecisionRules.simulate_multistage(
                    det_eq, spi, spo, [5.0],
                    sample(usamples_det), policy)
                train_multistage(policy, [5.0], det_eq, spi, spo, usamples_det;
                    num_batches=4, num_train_per_batch=1,
                    penalty_schedule=:default_annealed)
                obj_after = DecisionRules.simulate_multistage(
                    det_eq, spi, spo, [5.0],
                    sample(usamples_det), policy)
                @test isfinite(obj_after)
            end
        else
            @info "Skipping GPU (CUDA) tests: CUDA not available or not functional"
            @test_skip false
        end
    end
end
