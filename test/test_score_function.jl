using Statistics

raw"""
    _score_function_stage_model(; kwargs...)

Build a one-dimensional linear stage model for score-function tests.

The continuous version is

```math
\begin{aligned}
\min_{u,x,\delta}\quad
    & 30u + 10^4 |\delta| \\
\text{s.t.}\quad
    & x = x^{in} + \xi - u, \\
    & u \ge d, \\
    & \delta = x - \hat{x}.
\end{aligned}
```

When `integer = true`, the model adds a binary setup variable `z`,

```math
u \le 10 z,\qquad z \in \{0,1\},
```

and the objective becomes ``5z + 30u + 10^4|\delta|``.

# Keywords
- `state_value::Real`: initial value for the input-state parameter.
- `target_value::Real`: initial value for the target parameter.
- `uncertainty_value::Real`: initial value for the uncertainty parameter.
- `demand::Real`: lower bound that forces positive ordering cost.
- `integer::Bool`: whether to include a binary setup decision.

# Examples
```julia
model, state_in, target, state_out, uncertainty =
    _score_function_stage_model(; integer = true)
```
"""
function _score_function_stage_model(;
    state_value::Real = 5.0,
    target_value::Real = 4.0,
    uncertainty_value::Real = 2.0,
    demand::Real = 1.0,
    integer::Bool = false,
)
    # HiGHS keeps these tests fast and supports both LP and small MIP cases.
    model = quiet_highs_model()

    # The order quantity is the operational decision whose cost we measure.
    @variable(model, order >= 0.0)

    # The output state is what the policy target tries to guide.
    @variable(model, state_out >= 0.0)

    # Parameters are updated by rollout helpers before every stage solve.
    @variable(model, state_in in MOI.Parameter(Float64(state_value)))
    @variable(model, target in MOI.Parameter(Float64(target_value)))
    @variable(model, uncertainty in MOI.Parameter(Float64(uncertainty_value)))

    # The transition is intentionally simple so expected states are easy to audit.
    @constraint(model, state_out == state_in + uncertainty - order)

    # A positive lower bound prevents the zero-cost solution from hiding bugs.
    @constraint(model, order >= Float64(demand))

    if integer
        # The binary variable gives integer-strategy tests a real discrete object.
        @variable(model, setup, Bin)

        # This links setup to order without changing the simple state equation.
        @constraint(model, order <= 10.0 * setup)

        # The objective includes operational cost and the generated deficit cost.
        _norm_deficit, deficit = create_deficit!(model, 1; penalty = 1.0e4)
        @constraint(model, deficit[1] == state_out - target)
        @objective(model, Min, 5.0 * setup + 30.0 * order + objective_function(model))
    else
        # Continuous tests use the same deficit structure without a setup binary.
        _norm_deficit, deficit = create_deficit!(model, 1; penalty = 1.0e4)
        @constraint(model, deficit[1] == state_out - target)
        @objective(model, Min, 30.0 * order + objective_function(model))
    end

    return model, state_in, target, state_out, uncertainty
end

raw"""
    _two_stage_score_function_fixture(; integer = false)

Create a reusable two-stage score-function fixture.

The fixture represents a two-stage rollout

```math
Q(\hat{x}_{1:2})
    =
    q_1(x_0,\xi_1;\hat{x}_1)
    +
    q_2(x_1,\xi_2;\hat{x}_2),
```

where each ``q_t`` is the one-dimensional stage model built by
[`_score_function_stage_model`](@ref). The two stages intentionally use
different parameter defaults so indexing mistakes change the solved model.

# Keywords
- `integer::Bool`: whether stage models contain binary setup variables.

# Examples
```julia
config, initial_state, uncertainties, targets =
    _two_stage_score_function_fixture()
```
"""
function _two_stage_score_function_fixture(; integer::Bool = false)
    # Stage 1 starts from inventory 5 and observes uncertainty 2.
    stage_1, state_in_1, target_1, state_out_1, uncertainty_1 =
        _score_function_stage_model(; integer)

    # Stage 2 uses different parameter defaults to catch stage-index mistakes.
    stage_2, state_in_2, target_2, state_out_2, uncertainty_2 =
        _score_function_stage_model(;
            state_value = 4.0,
            target_value = 3.0,
            uncertainty_value = 1.0,
            integer,
        )

    # The config mirrors the shape used by train_multistage.
    state_params_in = [[state_in_1], [state_in_2]]
    state_params_out = [[(target_1, state_out_1)], [(target_2, state_out_2)]]
    config = ScoreFunctionConfig(
        [stage_1, stage_2],
        state_params_in,
        state_params_out;
        num_rollouts = 4,
        perturbation_std = 0.5,
    )

    # Targets include the initial state at index 1.
    initial_state = [5.0]
    targets = [[5.0], [4.0], [3.0]]
    uncertainties = [
        [(uncertainty_1, 2.0)],
        [(uncertainty_2, 1.0)],
    ]

    return config, initial_state, uncertainties, targets
end

@testset "Score-function gradient mixing" begin
    @testset "ScoreFunctionConfig validates public arguments" begin
        config, _, _, _ = _two_stage_score_function_fixture()

        @test config.dual_weight == 0.5
        @test config.perturbation_std == 0.5
        @test config.num_rollouts == 4
        @test config.baseline == :mean

        @test_throws ArgumentError ScoreFunctionConfig(
            config.subproblems,
            config.state_params_in[1:1],
            config.state_params_out,
        )
        @test_throws ArgumentError ScoreFunctionConfig(
            config.subproblems,
            config.state_params_in,
            config.state_params_out;
            dual_weight = -0.1,
        )
        @test_throws ArgumentError ScoreFunctionConfig(
            config.subproblems,
            config.state_params_in,
            config.state_params_out;
            perturbation_std = 0.0,
        )
        @test_throws ArgumentError ScoreFunctionConfig(
            config.subproblems,
            config.state_params_in,
            config.state_params_out;
            num_rollouts = 0,
        )
        @test_throws ArgumentError ScoreFunctionConfig(
            config.subproblems,
            config.state_params_in,
            config.state_params_out;
            baseline = :median,
        )
    end

    @testset "rollout_with_perturbation returns operational cost" begin
        config, initial_state, uncertainties, targets =
            _two_stage_score_function_fixture()

        # Zero perturbation should still solve the staged rollout successfully.
        zero_cost = DecisionRules.rollout_with_perturbation(
            config,
            initial_state,
            uncertainties,
            targets,
            [[0.0], [0.0]],
        )

        # A nonzero perturbation exercises target parameter updates.
        perturbed_cost = DecisionRules.rollout_with_perturbation(
            config,
            initial_state,
            uncertainties,
            targets,
            [[0.1], [-0.2]],
        )

        @test isfinite(zero_cost)
        @test isfinite(perturbed_cost)
        @test zero_cost > 0.0
        @test perturbed_cost > 0.0
    end

    @testset "_score_function_rollouts samples centered advantages" begin
        config, initial_state, uncertainties, targets =
            _two_stage_score_function_fixture()

        Random.seed!(42)
        advantages, perturbations = DecisionRules._score_function_rollouts(
            config,
            initial_state,
            uncertainties,
            targets;
            perturbation_std = 0.5,
            num_rollouts = 6,
        )

        @test length(advantages) == 6
        @test length(perturbations) == 6
        @test all(isfinite, advantages)
        @test all(length(rollout) == 2 for rollout in perturbations)
        @test all(length(stage) == 1 for rollout in perturbations for stage in rollout)

        # The default :mean baseline centers advantages by construction.
        @test sum(advantages) ≈ 0.0 atol = 1.0e-8
    end

    @testset "_score_function_surrogate matches Gaussian location score" begin
        # These targets are differentiable arrays in the training loop.
        targets = [[1.0f0], [2.0f0], [4.0f0]]

        # Perturbations are actual target perturbations, not standard normals.
        perturbations = [[0.5], [-0.25]]

        surrogate = DecisionRules._score_function_surrogate(
            3.0,
            perturbations,
            targets,
            0.5,
        )

        # 3 * ((0.5 / 0.25) * 2 + (-0.25 / 0.25) * 4) == 0.
        @test surrogate ≈ 0.0f0
    end

    @testset "sf_params reports scheduled ASCII-named fields" begin
        config, _, _, _ = _two_stage_score_function_fixture()
        schedule = ScoreFunctionSchedule(
            config;
            sf_start = 10,
            ramp_batches = 20,
            perturbation_std_initial = 0.1,
            num_rollouts_initial = 2,
        )

        before_start = sf_params(schedule, 9)
        @test before_start.active == false
        @test before_start.alpha == 1.0
        @test before_start.score_weight == 0.0

        at_start = sf_params(schedule, 10)
        @test at_start.active == true
        @test at_start.alpha == 1.0
        @test at_start.num_rollouts == 2

        halfway = sf_params(schedule, 20)
        @test halfway.active == true
        @test 0.0 < halfway.score_weight < 0.5
        @test halfway.perturbation_std > 0.1

        after_ramp = sf_params(schedule, 30)
        @test after_ramp.alpha ≈ config.dual_weight
        @test after_ramp.score_weight ≈ 1.0 - config.dual_weight
        @test after_ramp.perturbation_std ≈ config.perturbation_std
        @test after_ramp.num_rollouts == config.num_rollouts

        static_params = sf_params(config, 1)
        @test static_params.active == true
        @test static_params.alpha == config.dual_weight
    end

    @testset "rollout solves integer models exactly as written" begin
        config, initial_state, uncertainties, targets =
            _two_stage_score_function_fixture(; integer = true)

        cost = DecisionRules.rollout_with_perturbation(
            config,
            initial_state,
            uncertainties,
            targets,
            [[0.1], [0.0]],
        )

        @test isfinite(cost)
        @test any(JuMP.is_binary, JuMP.all_variables(config.subproblems[1]))
    end

    @testset "train_multistage accepts ScoreFunctionConfig on deterministic equivalent" begin
        # Build the deterministic-equivalent problem used by the dual path.
        stage_1, state_in_1, target_1, state_out_1, uncertainty_1 =
            build_subproblem(10; subproblem = quiet_highs_model())
        stage_2, state_in_2, target_2, state_out_2, uncertainty_2 =
            build_subproblem(
                10;
                state_i_val = 4.0,
                state_out_val = 3.0,
                uncertainty_val = 1.0,
                subproblem = quiet_highs_model(),
        )
        subproblems = [stage_1, stage_2]
        state_params_in = Vector{Vector{Any}}(undef, 2)
        state_params_in .= [[state_in_1], [state_in_2]]
        state_params_out = Vector{Vector{Tuple{Any,VariableRef}}}(undef, 2)
        state_params_out .= [[(target_1, state_out_1)], [(target_2, state_out_2)]]
        uncertainty_samples = [[(uncertainty_1, [2.0])], [(uncertainty_2, [1.0])]]

        det_equivalent, deterministic_sampler = DecisionRules.deterministic_equivalent!(
            quiet_highs_model(),
            subproblems,
            state_params_in,
            state_params_out,
            [5.0],
            uncertainty_samples,
        )

        # Build separate rollout models so the score-function solves do not
        # mutate the deterministic-equivalent model.
        score_config, _, _, _ = _two_stage_score_function_fixture()
        score_config = ScoreFunctionConfig(
            score_config.subproblems,
            score_config.state_params_in,
            score_config.state_params_out;
            dual_weight = 0.5,
            perturbation_std = 0.3,
            num_rollouts = 2,
        )

        Random.seed!(222)
        policy = Chain(Dense(2, 8, relu), Dense(8, 1))

        Random.seed!(42)
        train_multistage(
            policy,
            [5.0],
            det_equivalent,
            state_params_in,
            state_params_out,
            deterministic_sampler;
            num_batches = 4,
            num_train_per_batch = 2,
            score_function = score_config,
        )

        objective = simulate_multistage(
            det_equivalent,
            state_params_in,
            state_params_out,
            [5.0],
            sample(deterministic_sampler),
            policy,
        )
        @test isfinite(objective)
    end
end
