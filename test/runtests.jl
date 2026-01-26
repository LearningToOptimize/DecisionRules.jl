using DecisionRules
using Test
using SCS
using JuMP
import MathOptInterface as MOI
using Zygote
using Flux
using Random
using DiffOpt
using ChainRules: @ignore_derivatives

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

    @testset "create_deficit!" begin
        # Test 1: L1 norm only (default/legacy behavior)
        model1 = Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0))
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
        @test termination_status(model1) == MOI.OPTIMAL
        @test value(norm_deficit1) ≈ 3.0 rtol=1.0e-2  # L1 norm = |0.5| + |1.0| + |1.5| = 3.0
        
        # Test 2: L2 squared norm only (sum of squares)
        model2 = Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0))
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
        @test termination_status(model2) == MOI.OPTIMAL
        # L2 squared = 0.5^2 + 1.0^2 + 1.5^2 = 3.5
        @test value(norm_deficit2) ≈ 3.5 rtol=1.0e-2
        
        # Test 3: Both L1 and L2 squared norms
        model3 = Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0))
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
        @test termination_status(model3) == MOI.OPTIMAL
        # Combined: 100 * 3.0 + 50 * 3.5 = 300 + 175 = 475
        expected_combined = 100.0 * 3.0 + 50.0 * 3.5
        @test value(norm_deficit3) ≈ expected_combined rtol=1.0e-2
        
        # Test 4: Verify backward compatibility with legacy 'penalty' argument
        model4 = Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0))
        @variable(model4, x4)
        @objective(model4, Min, x4)
        fix(x4, 1.0)
        norm_deficit4, _deficit4 = create_deficit!(model4, 2; penalty=500.0)
        @test length(_deficit4) == 2
        # Should create L1 norm constraint (backwards compatible)
        optimize!(model4)
        @test termination_status(model4) == MOI.OPTIMAL
        
        # Test 5: Verify objective contribution with L1 norm
        model5 = Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0))
        @variable(model5, y5 >= 0)
        @objective(model5, Min, 10 * y5)
        @constraint(model5, y5 >= 1)  # Forces y5 = 1
        norm_deficit5, _deficit5 = create_deficit!(model5, 2; penalty_l1=100.0)
        @constraint(model5, _deficit5[1] == 2.0)  # Fixed deficit
        @constraint(model5, _deficit5[2] == 3.0)  # Fixed deficit
        optimize!(model5)
        @test termination_status(model5) == MOI.OPTIMAL
        @test value(norm_deficit5) ≈ 5.0 rtol=1.0e-2  # L1 = 2 + 3 = 5
        # Total objective = 10 * 1 + 100 * 5 = 510
        @test objective_value(model5) ≈ 510.0 rtol=1.0e-2
        
        # Test 6: Verify objective contribution with L2 squared norm
        model6 = Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0))
        @variable(model6, y6 >= 0)
        @objective(model6, Min, 10 * y6)
        @constraint(model6, y6 >= 1)
        norm_deficit6, _deficit6 = create_deficit!(model6, 2; penalty_l2=100.0)
        @constraint(model6, _deficit6[1] == 3.0)
        @constraint(model6, _deficit6[2] == 4.0)
        optimize!(model6)
        @test termination_status(model6) == MOI.OPTIMAL
        @test value(norm_deficit6) ≈ 25.0 rtol=1.0e-2  # L2 squared = 9 + 16 = 25
        # Total objective = 10 * 1 + 100 * 25 = 2510
        @test objective_value(model6) ≈ 2510.0 rtol=1.0e-2
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
    end

    @testset "Multiple Shooting" begin
        # Test create_shooting_window! first (creates models needed for other tests)
        @testset "create_shooting_window!" begin
            Random.seed!(456)
            
            # Create 6 subproblems to test windowing with multiple windows
            num_stages = 6
            subproblems = Vector{JuMP.Model}(undef, num_stages)
            state_params_in = Vector{Vector{Any}}(undef, num_stages)
            state_params_out = Vector{Vector{Tuple{Any, VariableRef}}}(undef, num_stages)
            uncertainty_samples = Vector{Vector{Tuple{VariableRef, Vector{Float64}}}}(undef, num_stages)
            
            for t in 1:num_stages
                subproblems[t] = DiffOpt.diff_model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0))
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
            
            # Create a shooting window for stages 1-2
            window_model = DiffOpt.diff_model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0))
            window_model, uncertainties_new = DecisionRules.create_shooting_window!(
                window_model,
                subproblems,
                state_params_in,
                state_params_out,
                initial_state,
                uncertainty_samples,
                1,  # window_start
                2   # window_size
            )
            
            # Verify the window model was created
            @test window_model !== nothing
            @test length(uncertainties_new) == 2  # Only 2 stages in the window
            
            # Verify we can solve the window model
            optimize!(window_model)
            @test termination_status(window_model) in [MOI.OPTIMAL, MOI.ALMOST_OPTIMAL]
            
            # Create window for stages 3-4
            window_model_2 = DiffOpt.diff_model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0))
            window_model_2, uncertainties_new_2 = DecisionRules.create_shooting_window!(
                window_model_2,
                subproblems,
                state_params_in,
                state_params_out,
                initial_state,
                uncertainty_samples,
                3,  # window_start
                2   # window_size
            )
            
            @test window_model_2 !== nothing
            @test length(uncertainties_new_2) == 2
            optimize!(window_model_2)
            @test termination_status(window_model_2) in [MOI.OPTIMAL, MOI.ALMOST_OPTIMAL]
            
            # Create window for stages 5-6
            window_model_3 = DiffOpt.diff_model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0))
            window_model_3, uncertainties_new_3 = DecisionRules.create_shooting_window!(
                window_model_3,
                subproblems,
                state_params_in,
                state_params_out,
                initial_state,
                uncertainty_samples,
                5,  # window_start
                2   # window_size
            )
            
            @test window_model_3 !== nothing
            @test length(uncertainties_new_3) == 2
            
            # Test edge case: window extends beyond num_stages
            window_model_edge = DiffOpt.diff_model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0))
            window_model_edge, uncertainties_edge = DecisionRules.create_shooting_window!(
                window_model_edge,
                subproblems,
                state_params_in,
                state_params_out,
                initial_state,
                uncertainty_samples,
                5,  # window_start
                5   # window_size (larger than remaining stages)
            )
            
            # Should only include stages 5-6 (2 stages)
            @test length(uncertainties_edge) == 2
        end
        
        # Test predict_window_states
        @testset "predict_window_states" begin
            # Simple decision rule that adds uncertainty to state
            simple_rule(x) = x[1:1] .+ x[2:2]  # state + uncertainty
            
            initial_state = Float32[0.0]
            uncertainties = [Float32[1.0], Float32[2.0], Float32[3.0]]
            
            states = DecisionRules.predict_window_states(simple_rule, initial_state, uncertainties)
            
            # states[1] = initial = [0]
            # states[2] = simple_rule([0, 1]) = [1]
            # states[3] = simple_rule([1, 2]) = [3]
            # states[4] = simple_rule([3, 3]) = [6]
            @test length(states) == 4
            @test states[1] ≈ [0.0f0]
            @test states[2] ≈ [1.0f0]
            @test states[3] ≈ [3.0f0]
            @test states[4] ≈ [6.0f0]
        end
        
        # Test gradients flow through predict_window_states
        @testset "predict_window_states gradients" begin
            # Use a simple neural network
            m = Chain(Dense(2, 1, identity))
            
            initial_state = Float32[1.0]
            uncertainties = [Float32[0.5], Float32[0.3]]
            
            loss, grads = Flux.withgradient(m) do model
                states = DecisionRules.predict_window_states(model, initial_state, uncertainties)
                sum(sum.(states))  # Simple loss to get gradients
            end
            
            @test loss > 0
            @test grads[1] !== nothing
        end
        
        # Test simulate_window_stages with multiple windows each having multiple stages
        @testset "simulate_window_stages with multiple windows" begin
            Random.seed!(789)
            
            # Create 6 subproblems: 3 windows of 2 stages each
            num_stages = 6
            window_size = 2
            num_windows = 3
            
            subproblems = Vector{JuMP.Model}(undef, num_stages)
            state_params_in = Vector{Vector{Any}}(undef, num_stages)
            state_params_out = Vector{Vector{Tuple{Any, VariableRef}}}(undef, num_stages)
            uncertainty_samples = Vector{Vector{Tuple{VariableRef, Vector{Float64}}}}(undef, num_stages)
            
            for t in 1:num_stages
                subproblems[t] = DiffOpt.diff_model(optimizer_with_attributes(SCS.Optimizer, "verbose" => 0))
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
            
            # Sample uncertainties for this test
            sampled_uncertainties = [[(subproblems[t][:uncertainty], 0.4)] for t in 1:num_stages]
            
            # Test each window separately using simulate_window_stages
            total_objective = 0.0
            current_state = Float32.(initial_state)
            
            for w in 1:num_windows
                window_start = (w - 1) * window_size + 1
                window_end = w * window_size
                
                # Get window data
                window_subproblems = subproblems[window_start:window_end]
                window_state_params_in = state_params_in[window_start:window_end]
                window_state_params_out = state_params_out[window_start:window_end]
                window_uncertainties = sampled_uncertainties[window_start:window_end]
                window_uncertainties_vec = [Float32[0.4] for _ in 1:window_size]
                
                # Predict states for this window using a simple decision rule
                simple_rule(x) = x[2:2] .+ 0.5f0  # state_out = uncertainty + 0.5
                window_states = DecisionRules.predict_window_states(simple_rule, current_state, window_uncertainties_vec)
                
                # Simulate this window
                window_obj = DecisionRules.simulate_window_stages(
                    window_subproblems,
                    window_state_params_in,
                    window_state_params_out,
                    window_uncertainties,
                    window_states
                )
                
                @test window_obj > 0  # Each window should have positive objective
                total_objective += window_obj
                
                # Get real state for next window from last stage's optimization result
                if w < num_windows
                    current_state = Float32[value(var) for (param, var) in window_state_params_out[end]]
                end
            end
            
            @test total_objective > 0  # Total objective across all windows should be positive
            
            # Test with neural network decision rule
            nn_rule = Chain(Dense(2, 4, relu), Dense(4, 1))
            
            # Verify gradients flow through multiple windows
            loss, grads = Flux.withgradient(nn_rule) do model
                total = 0.0f0
                curr_state = Float32.(initial_state)
                
                for w in 1:num_windows
                    window_start = (w - 1) * window_size + 1
                    window_end = w * window_size
                    
                    window_subproblems = subproblems[window_start:window_end]
                    window_state_params_in = state_params_in[window_start:window_end]
                    window_state_params_out = state_params_out[window_start:window_end]
                    window_uncertainties = sampled_uncertainties[window_start:window_end]
                    window_uncertainties_vec = [Float32[0.4] for _ in 1:window_size]
                    
                    # Predict states using the model
                    window_states = DecisionRules.predict_window_states(model, curr_state, window_uncertainties_vec)
                    
                    # Simulate window
                    window_obj = DecisionRules.simulate_window_stages(
                        window_subproblems,
                        window_state_params_in,
                        window_state_params_out,
                        window_uncertainties,
                        window_states
                    )
                    
                    total += window_obj
                    
                    # Get real state (stop gradients here)
                    if w < num_windows
                        curr_state = @ignore_derivatives Float32[value(var) for (param, var) in window_state_params_out[end]]
                    end
                end
                
                return total
            end
            
            @test loss > 0
            @test grads[1] !== nothing  # Gradients should flow through
        end
    end
end
