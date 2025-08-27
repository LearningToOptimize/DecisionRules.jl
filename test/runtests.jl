using DecisionRules
using Test
using Gurobi
import ParametricOptInterface as POI
using JuMP
using Zygote
using Flux
using Random
using DiffOpt

function build_subproblem(d; state_i_val=5.0, state_out_val=4.0, uncertainty_val=2.0)
    subproblem = JuMP.Model(DiffOpt.conic_diff_model(Gurobi.Optimizer))
    set_attributes(subproblem, "output_flag" => false)
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
    @objective(subproblem, Min, 30 * y + norm_deficit * 10^7)
    return subproblem, state_in, state_out, state_out_var, uncertainty
end

@testset "DecisionRules.jl" begin
    subproblem1, state_in_1, state_out_1, state_out_var_1, uncertainty_1 = build_subproblem(10)

    optimize!(subproblem1)

    @testset "pdual" begin
        @test DecisionRules.pdual(state_in_1) ≈ -30.0
        @test DecisionRules.pdual(state_out_1) ≈ 30.0
    end

    @testset "simulate_stage" begin
        inflow = 2.0
        state_param_in = [state_in_1]
        state_param_out = [(state_out_1, state_out_var_1)]
        uncertainty_sample = Dict(uncertainty_1 => inflow)
        state_in_val = [5.0]
        state_out_val = [4.0]
        # Test simulate_stage
        @test DecisionRules.simulate_stage(subproblem1, state_param_in, state_param_out, uncertainty_sample, state_in_val, state_out_val) ≈ 210
        grad = gradient(DecisionRules.simulate_stage, subproblem1, state_param_in, state_param_out, uncertainty_sample, state_in_val, state_out_val)
        @test grad[5] ≈ [-30.0]
        @test grad[6] ≈ [30.0]
        # Train flux DR
        subproblem = subproblem1
        Random.seed!(222)
        m = Chain(Dense(1, 10), Dense(10, 1))
        @test DecisionRules.simulate_stage(subproblem, state_param_in, state_param_out, uncertainty_sample, state_in_val, m([inflow])) > 90.0
        for _ in 1:2050
            _inflow = rand(1.:5)
            uncertainty_samp = Dict(uncertainty => _inflow)
            Flux.train!((m, inflow) -> DecisionRules.simulate_stage(subproblem, state_param_in, state_param_out, uncertainty_sample, state_in_val, m(inflow)), m, [[_inflow] for _ =1:10], Flux.Adam())
        end
        @test DecisionRules.simulate_stage(subproblem, state_param_in, state_param_out, uncertainty_sample, state_in_val, m([inflow])) <= 92
    end

    # @testset "get_next_state" begin # WRONG get_next_state needs DiffOpt.jl
    #     inflow = 2.0
    #     state_param_in = [state_in_1]
    #     state_param_out = [(state_out_1, state_out_var_1)]
    #     uncertainty_sample = Dict(uncertainty_1 => inflow)
    #     state_in_val = [5.0]
    #     state_out_val = [4.0]
    #     DecisionRules.simulate_stage(subproblem1, state_param_in, state_param_out, uncertainty_sample, state_in_val, state_out_val)
    #     # Test 1st case
    #     state_out_res1 = DecisionRules.get_next_state(subproblem1, state_param_out, max_state_out, state_out_val)
    #     @test state_out_res1 ≈ [4.0]
    #     jacob = jacobian(DecisionRules.get_next_state, subproblem1, state_param_out, max_state_out, state_out_val)
    #     state_out_val = state_out_val .+ 0.0001
    #     DecisionRules.simulate_stage(subproblem1, state_param_in, state_param_out, uncertainty_sample, state_in_val, state_out_val)
    #     state_out_res2 = DecisionRules.get_next_state(subproblem1, state_param_out, max_state_out, state_out_val)
    #     @test state_out_res2 ≈ state_out_res1 .+ 0.0001* jacob[4][1,1]
    #     # Test 2nd case
    #     state_out_val = max_state_out .+ 1.0
    #     DecisionRules.simulate_stage(subproblem1, state_param_in, state_param_out, uncertainty_sample, state_in_val, state_out_val)
    #     state_out_res3 = DecisionRules.get_next_state(subproblem1, state_param_out, max_state_out, state_out_val)
    #     jacob = jacobian(DecisionRules.get_next_state, subproblem1, state_param_out, max_state_out, state_out_val)
    # end

    @test "deterministic_equivalent" begin
        subproblem2, state_in_2, state_out_2, state_out_var_2, uncertainty_2 = build_subproblem(10; state_i_val=4.0, state_out_val=3.0, uncertainty_val=1.0)
        optimize!(subproblem2)
        objective_value(subproblem2)
        value(state_out_var_2)

        subproblems = [subproblem1, subproblem2]
        state_params_in = [[state_in_1], [state_in_2]]
        state_params_out = [[(state_out_1, state_out_var_1)], [(state_out_2, state_out_var_2)]]
        uncertainty_samples = [Dict(uncertainty_1 => [2.0]), Dict(uncertainty_2 => [1.0])]
        initial_state = [5.0]

        det_equivalent, uncertainty_samples = DecisionRules.deterministic_equivalent(subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples)
        set_optimizer(det_equivalent, DiffOpt.diff_model(Gurobi.Optimizer))
        JuMP.optimize!(det_equivalent)
        obj_val = objective_value(det_equivalent)
        DecisionRules.pdual.(state_params_in[1])
        DecisionRules.pdual.(state_params_out[1][1][1])
        DecisionRules.simulate_multistage(det_equivalent, state_params_in, state_params_out, sample(uncertainty_samples), [[9.0], [7.], [4.000]])
        grad = gradient(DecisionRules.simulate_multistage, det_equivalent, state_params_in, state_params_out, sample(uncertainty_samples), [[9.0], [7.], [4.0]])

        m = Chain(Dense(1, 10), Dense(10, 1))
        obj_val = DecisionRules.simulate_multistage(
            det_equivalent, state_params_in, state_params_out, 
            initial_state, sample(uncertainty_samples), 
            m
        )

        train_multistage(m, initial_state, det_equivalent, state_params_in, state_params_out, uncertainty_samples)
    end
end
