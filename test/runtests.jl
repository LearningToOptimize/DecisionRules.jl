using DecisionRules
using Test
using SCS
using JuMP
import MathOptInterface as MOI
using Zygote
using Flux
using Random
using DiffOpt

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
    @constraint(subproblem, [norm_deficit; _deficit] in SecondOrderCone())
    @objective(subproblem, Min, 30 * y + norm_deficit * 10^4)
    return subproblem, state_in, state_out, state_out_var, uncertainty
end

@testset "DecisionRules.jl" begin
    @testset "pdual at infeasibility" begin
        subproblem1, state_in_1, state_out_1, state_out_var_1, uncertainty_1 = build_subproblem(10; subproblem=DiffOpt.conic_diff_model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0)), state_out_val=9.0)
        optimize!(subproblem1)
        @test DecisionRules.pdual(state_in_1) ≈ -1.0e4 rtol=1.0e-1
        @test DecisionRules.pdual(state_out_1) ≈ 1.0e4 rtol=1.0e-1
    end

    subproblem1, state_in_1, state_out_1, state_out_var_1, uncertainty_1 = build_subproblem(10; subproblem=DiffOpt.conic_diff_model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0)))

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
        subproblem1, state_in_1, state_out_1, state_out_var_1, uncertainty_1 = build_subproblem(10; subproblem=DiffOpt.conic_diff_model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0)))
        subproblem2, state_in_2, state_out_2, state_out_var_2, uncertainty_2 = build_subproblem(10; state_i_val=1.0, state_out_val=9.0, uncertainty_val=2.0, subproblem=DiffOpt.conic_diff_model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0)))

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
            DiffOpt.diff_model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0)),
            subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples
        )

        obj_val = DecisionRules.simulate_multistage(det_equivalent, state_params_in, state_params_out, sample(uncertainty_samples), [[9.0], [7.], [4.000]])
        @test obj_val ≈ 359 rtol=1.0e-1
        grad = Zygote.gradient(DecisionRules.simulate_multistage, det_equivalent, state_params_in, state_params_out, sample(uncertainty_samples), [[9.0], [7.], [4.0]])
        @test grad[5][1][1] ≈ -30.0 rtol=1.0e-1
        @test grad[5][3][1] ≈ 30.0 rtol=1.0e-1

        Random.seed!(222)
        uncertainty_sample = sample(uncertainty_samples)
        
        m = Chain(Dense(1, 10), Dense(10, 1))
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
        model1 = Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0))
        @variable(model1, x1 >= 0)
        @variable(model1, p1 in MOI.Parameter(2.0))
        @constraint(model1, con1, x1 - p1 >= 0)
        @objective(model1, Min, x1)
        optimize!(model1)
        @test compute_parameter_dual(model1, p1) ≈ 1.0 rtol=1.0e-2

        # Test 2: Parameter in objective
        # min x + 2*p  s.t. x >= 1
        # ∂obj/∂p = 2 (from objective directly, minimization)
        model2 = Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0))
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
        model3 = Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0))
        @variable(model3, x3 >= 0)
        @variable(model3, p3 in MOI.Parameter(1.0))
        @constraint(model3, con3, x3 - 2 * p3 >= 0)
        @objective(model3, Min, x3 + p3)
        optimize!(model3)
        @test compute_parameter_dual(model3, p3) ≈ 3.0 rtol=1.0e-2

        # Test 4: Maximization problem
        # max -x + p  s.t. x >= 1
        # Equivalent to min x - p, so ∂obj/∂p = -(-1) = 1 for max
        model4 = Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0))
        @variable(model4, x4 >= 0)
        @variable(model4, p4 in MOI.Parameter(1.0))
        @constraint(model4, x4 >= 1)
        @objective(model4, Max, -x4 + p4)
        optimize!(model4)
        @test compute_parameter_dual(model4, p4) ≈ -1.0 rtol=1.0e-2

        # Test 5: SOC constraint with parameter (similar to existing pdual tests)
        model5 = Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0))
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
        @constraint(model5, [norm_deficit5; _deficit5] in SecondOrderCone())
        @objective(model5, Min, 30 * y5 + norm_deficit5 * 10^4)
        optimize!(model5)
        @test compute_parameter_dual(model5, state_in5) ≈ -30.0 rtol=1.0e-1
        @test compute_parameter_dual(model5, state_out5) ≈ 30.0 rtol=1.0e-1
    end
end
