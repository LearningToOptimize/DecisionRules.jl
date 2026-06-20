"""
Train TS-DDR policies for the inventory-control benchmark.

The benchmark compares four target-state decision-rule variants:

1. relaxed continuous subproblems with ordinary LP duals;
2. integer subproblems with fixed-discrete local duals;
3. integer subproblems with continuous-relaxation duals; and
4. integer subproblems with fixed-discrete duals plus score-function rollouts.
"""

using CSV
using DataFrames
using DecisionRules
using Flux
using JLD2
using JuMP
using Random
using Statistics

include(joinpath(@__DIR__, "build_inventory_problem.jl"))

# The script keeps generated models and CSV files out of the source directory.
const EXAMPLE_DIR = @__DIR__
const MODEL_DIR = joinpath(EXAMPLE_DIR, "models")
const RESULT_DIR = joinpath(EXAMPLE_DIR, "results")

# Create output directories before any training run tries to write into them.
mkpath(MODEL_DIR)
mkpath(RESULT_DIR)

# Use one fixed training sample size for every TS-DDR variant.
const N_TRAIN_SCENARIOS = 50

# Use one held-out evaluation size for every reported cost distribution.
const N_TEST_SCENARIOS = 300

"""
    InventoryTrainingVariant

Configuration for one TS-DDR inventory training run.

Mathematically, every variant trains a policy ``\\pi_\\theta`` by stochastic
gradient descent on sampled deterministic-equivalent objectives
``Q(w; \\theta)``. The fields choose the model family and the gradient estimator:

- `integer`: whether the JuMP model contains binary setup variables ``z_t``;
- `training_integer_strategy`: how local dual information is read for
  ``\\nabla_\\theta Q(w; \\theta)`` when the model is mixed-integer;
- `score_function`: optional Monte Carlo correction using perturbed target
  rollouts.

# Fields
- `tag::String`: prefix used for saved models and CSV files.
- `integer::Bool`: whether to build the fixed-cost MIP formulation.
- `num_batches::Int`: number of SGD updates.
- `train_per_batch::Int`: sampled trajectories per SGD update.
- `learning_rate::Float64`: Adam learning rate.
- `warmup_batches::Int`: last batch of the low target-penalty phase.
- `training_integer_strategy::AbstractIntegerStrategy`: dual-path strategy.
- `score_function::Union{Nothing,ScoreFunctionConfig,ScoreFunctionSchedule}`:
  optional score-function estimator.

# Examples
```julia
variant = InventoryTrainingVariant(
    "integer",
    true,
    800,
    10,
    8.0e-4,
    120,
    FixedDiscreteIntegerStrategy(),
    nothing,
)
```
"""
struct InventoryTrainingVariant
    tag::String
    integer::Bool
    num_batches::Int
    train_per_batch::Int
    learning_rate::Float64
    warmup_batches::Int
    training_integer_strategy::AbstractIntegerStrategy
    score_function::Union{Nothing,ScoreFunctionConfig,ScoreFunctionSchedule}
end

"""
    penalty_schedule_for(variant::InventoryTrainingVariant)

Return the two-phase target-penalty schedule used by one inventory variant.

The target penalty is multiplied by `0.4` during warmup and by `1.0`
afterward:

```math
m_k =
\\begin{cases}
0.4, & 1 \\le k \\le k_{warm}, \\\\
1.0, & k_{warm} < k \\le K.
\\end{cases}
```

# Arguments
- `variant::InventoryTrainingVariant`: training configuration.

# Examples
```julia
schedule = penalty_schedule_for(variant)
```
"""
function penalty_schedule_for(variant::InventoryTrainingVariant)
    # The first tuple covers the gentler warmup phase.
    warmup_phase = (1, variant.warmup_batches, 0.4)

    # The second tuple restores the nominal target penalty.
    full_penalty_phase = (
        variant.warmup_batches + 1,
        variant.num_batches,
        1.0,
    )

    return [warmup_phase, full_penalty_phase]
end

"""
    method_label(variant::InventoryTrainingVariant) -> String

Return the table label for one TS-DDR variant.

# Arguments
- `variant::InventoryTrainingVariant`: training configuration.

# Examples
```julia
label = method_label(variant)
```
"""
function method_label(variant::InventoryTrainingVariant)
    # Score-function variants are named by the mixed-gradient estimator.
    !isnothing(variant.score_function) && return "TS-DDR (MixedGrad)"

    # ContinuousRelaxation means the dual path solves an LP relaxation.
    variant.training_integer_strategy isa ContinuousRelaxationIntegerStrategy &&
        return "TS-DDR (ContRelax)"

    # FixedDiscrete means the dual path fixes the MIP incumbent, then reads LP duals.
    variant.training_integer_strategy isa FixedDiscreteIntegerStrategy &&
        return "TS-DDR (FixedDiscrete)"

    return "TS-DDR (trained)"
end

"""
    operational_stage_cost(model::JuMP.Model, integer::Bool) -> Float64

Return the realized inventory cost of one solved stage model.

For the integer formulation, the operational cost is

```math
K z_t + c q_t + h \\max(s_t,0) + p \\max(-s_t,0).
```

For the relaxed formulation, the setup term ``K z_t`` is absent.

# Arguments
- `model::JuMP.Model`: solved inventory stage model.
- `integer::Bool`: whether `model` contains the binary setup variable `z`.

# Examples
```julia
cost = operational_stage_cost(stage_model, true)
```
"""
function operational_stage_cost(model::JuMP.Model, integer::Bool)
    # The order quantity is common to both formulations.
    order_quantity = value(model[:q])

    # Net inventory after demand determines holding or backlog cost.
    next_inventory = value(model[:s_out])

    # Positive inventory pays holding cost.
    holding_cost = INVENTORY_H * max(next_inventory, 0.0)

    # Negative inventory pays backlog cost.
    backlog_cost = INVENTORY_P * max(-next_inventory, 0.0)

    # Continuous formulations pay only variable ordering, holding, and backlog.
    variable_cost = INVENTORY_C * order_quantity + holding_cost + backlog_cost

    if integer
        # MIP solves return an integral setup value; round removes solver noise.
        setup_value = round(value(model[:z]))

        return INVENTORY_K * setup_value + variable_cost
    end

    return variable_cost
end

"""
    rollout_policy(policy, subproblems, state_params_in, state_params_out,
                   uncertainty_sampler, initial_state; kwargs...)

Evaluate a trained policy by stage-wise rollout on held-out trajectories.

At each stage the policy proposes a target state. The stage model then solves

```math
\\min f_t(x_t,y_t) + \\lambda |x_t^{mid} - \\hat{x}_t|
```

subject to the inventory transition and capacity constraints. Only the
operational term ``f_t`` is reported as cost, because target slack is a training
device rather than a deployed cost.

# Arguments
- `policy`: Flux-compatible target-state policy.
- `subproblems`: one solved-and-reused JuMP model per stage.
- `state_params_in`: input-state parameters for each stage.
- `state_params_out`: `(target_parameter, realized_state_variable)` pairs.
- `uncertainty_sampler`: sampler for held-out demand trajectories.
- `initial_state`: inventory state entering stage 1.

# Keywords
- `num_scenarios::Int`: number of held-out rollouts.
- `seed::Int`: random seed for evaluation trajectories.
- `integer::Bool`: whether to use the MIP operational cost formula.

# Examples
```julia
costs, inventory, setup, order = rollout_policy(
    policy,
    subproblems,
    state_params_in,
    state_params_out,
    uncertainty_sampler,
    initial_state;
    integer = true,
)
```
"""
function rollout_policy(
    policy,
    subproblems,
    state_params_in,
    state_params_out,
    uncertainty_sampler,
    initial_state;
    num_scenarios::Int = N_TEST_SCENARIOS,
    seed::Int = 555,
    integer::Bool = true,
)
    # Fix the evaluation sample so variants see the same demand distribution.
    Random.seed!(seed)

    # Store net-inventory trajectories, including the initial inventory at t=0.
    inventory_paths = Matrix{Float64}(undef, num_scenarios, INVENTORY_T + 1)

    # Store setup indicators for integer runs and order indicators for relaxed runs.
    setup_paths = Matrix{Float64}(undef, num_scenarios, INVENTORY_T)

    # Store order quantities for diagnostics.
    order_paths = Matrix{Float64}(undef, num_scenarios, INVENTORY_T)

    # Store operational cost for each scenario.
    operational_costs = Vector{Float64}(undef, num_scenarios)

    for scenario in 1:num_scenarios
        # Draw one demand path for this rollout.
        uncertainty_sample = sample(uncertainty_sampler)

        # Start from the benchmark initial state.
        state = Float64.(initial_state)

        # Record inventory at t=0.
        inventory_paths[scenario, 1] = state[1]

        # Reset the scenario cost accumulator.
        operational_costs[scenario] = 0.0

        for stage in 1:INVENTORY_T
            # Demand is the single uncertainty value in this inventory model.
            demand_value = uncertainty_sample[stage][1][2]

            # The policy maps observed demand plus current state to a target state.
            target = Float64.(policy(Float32[demand_value, state...]))

            # Input-state parameters receive the realized state entering this stage.
            for index in eachindex(state_params_in[stage])
                set_parameter_value(state_params_in[stage][index], state[index])
            end

            # Uncertainty parameters receive this stage's realized demand.
            for (parameter, value) in uncertainty_sample[stage]
                set_parameter_value(parameter, value)
            end

            # Target parameters receive the policy output.
            for index in eachindex(state_params_out[stage])
                set_parameter_value(state_params_out[stage][index][1], target[index])
            end

            # Solve the deployment stage exactly as modeled.
            optimize!(subproblems[stage])

            # Read decisions and realized inventory from the solved stage.
            order_quantity = value(subproblems[stage][:q])
            next_inventory = value(subproblems[stage][:s_out])

            # Add the operational cost, excluding target-deficit penalty.
            operational_costs[scenario] +=
                operational_stage_cost(subproblems[stage], integer)

            # Store setup or order activity for later diagnostics.
            setup_paths[scenario, stage] = integer ?
                round(value(subproblems[stage][:z])) :
                Float64(order_quantity > 1.0e-7)

            # Store order quantity and realized inventory trajectory.
            order_paths[scenario, stage] = order_quantity
            inventory_paths[scenario, stage + 1] = next_inventory

            # The next state carries current inventory and demand history.
            state = [next_inventory, demand_value, state[2]]
        end
    end

    return operational_costs, inventory_paths, setup_paths, order_paths
end

"""
    build_training_problem(variant::InventoryTrainingVariant)

Build the deterministic-equivalent model used for training a variant.

# Arguments
- `variant::InventoryTrainingVariant`: variant configuration.

# Examples
```julia
det_eq, state_in, state_out, sampler, initial_state =
    build_training_problem(variant)
```
"""
function build_training_problem(variant::InventoryTrainingVariant)
    # Training uses a deterministic equivalent so target-dual gradients are coupled.
    return build_inventory_det_equivalent(;
        num_scenarios = N_TRAIN_SCENARIOS,
        penalty = INVENTORY_PENALTY,
        seed = 42,
        integer = variant.integer,
    )
end

"""
    build_evaluation_problem(variant::InventoryTrainingVariant)

Build the stage-wise models used for held-out rollout evaluation.

# Arguments
- `variant::InventoryTrainingVariant`: variant configuration.

# Examples
```julia
subproblems, state_in, state_out, sampler, initial_state =
    build_evaluation_problem(variant)
```
"""
function build_evaluation_problem(variant::InventoryTrainingVariant)
    # Evaluation uses stage-wise deployment semantics, not the training DE solve.
    return build_inventory_subproblems(;
        num_scenarios = N_TEST_SCENARIOS,
        penalty = INVENTORY_PENALTY,
        seed = 99,
        integer = variant.integer,
    )
end

"""
    estimate_initial_loss(policy, det_eq, state_params_in, state_params_out,
                          uncertainty_sampler, initial_state, variant)

Estimate pre-training deterministic-equivalent cost for checkpoint initialization.

# Arguments
- `policy`: policy evaluated before training.
- `det_eq::JuMP.Model`: deterministic-equivalent training model.
- `state_params_in`: input-state parameters.
- `state_params_out`: target-state parameters.
- `uncertainty_sampler`: training sampler.
- `initial_state`: state entering stage 1.
- `variant::InventoryTrainingVariant`: training configuration.

# Examples
```julia
loss = estimate_initial_loss(policy, det_eq, spi, spo, sampler, x0, variant)
```
"""
function estimate_initial_loss(
    policy,
    det_eq::JuMP.Model,
    state_params_in,
    state_params_out,
    uncertainty_sampler,
    initial_state,
    variant::InventoryTrainingVariant,
)
    # Use a small fixed sample only to seed SaveBest with a finite baseline.
    Random.seed!(111)

    return mean(
        let uncertainty_sample = sample(uncertainty_sampler)
            # Deterministic-equivalent simulation needs the full target trajectory.
            target_states = simulate_states(initial_state, uncertainty_sample, policy)

            simulate_multistage(
                det_eq,
                state_params_in,
                state_params_out,
                uncertainty_sample,
                target_states;
                integer_strategy = variant.training_integer_strategy,
            )
        end for _ in 1:12
    )
end

"""
    train_variant!(policy, variant, det_eq, state_params_in, state_params_out,
                   uncertainty_sampler, initial_state, model_path, curve_path)

Train one policy and write its training curve.

# Arguments
- `policy`: mutable Flux policy updated in place.
- `variant::InventoryTrainingVariant`: training configuration.
- `det_eq::JuMP.Model`: deterministic-equivalent training model.
- `state_params_in`: input-state parameters.
- `state_params_out`: target-state parameters.
- `uncertainty_sampler`: training sampler.
- `initial_state`: state entering stage 1.
- `model_path::String`: path for the best model checkpoint.
- `curve_path::String`: path for the training-curve CSV.

# Examples
```julia
train_variant!(policy, variant, det_eq, spi, spo, sampler, x0, model_path, curve_path)
```
"""
function train_variant!(
    policy,
    variant::InventoryTrainingVariant,
    det_eq::JuMP.Model,
    state_params_in,
    state_params_out,
    uncertainty_sampler,
    initial_state,
    model_path::String,
    curve_path::String,
)
    # Estimate a baseline loss before any optimizer step.
    initial_loss = estimate_initial_loss(
        policy,
        det_eq,
        state_params_in,
        state_params_out,
        uncertainty_sampler,
        initial_state,
        variant,
    )

    # SaveBest stores the best policy according to the recorded operational loss.
    save_best = SaveBest(initial_loss, model_path)

    # Keep a small CSV trace for plots and sanity checks.
    training_log = DataFrame(batch = Int[], loss = Float64[])

    println("=" ^ 60)
    println("Training TS-DDR [$(variant.tag)]  integer=$(variant.integer)")
    println("  $(variant.num_batches) batches x $(variant.train_per_batch) scenarios")
    println("  learning rate: $(variant.learning_rate)")
    println("  training integer strategy: $(typeof(variant.training_integer_strategy))")
    !isnothing(variant.score_function) &&
        println("  score function: $(typeof(variant.score_function))")
    println("  pre-training cost: $(round(initial_loss, digits = 1))")
    println("=" ^ 60)

    # Fix optimizer randomness for repeatability.
    Random.seed!(2024)

    elapsed_seconds = @elapsed train_multistage(
        policy,
        initial_state,
        det_eq,
        state_params_in,
        state_params_out,
        uncertainty_sampler;
        num_batches = variant.num_batches,
        num_train_per_batch = variant.train_per_batch,
        optimizer = Flux.Adam(variant.learning_rate),
        integer_strategy = variant.training_integer_strategy,
        penalty_schedule = penalty_schedule_for(variant),
        score_function = variant.score_function,
        record = (sample_log, iteration, current_policy) -> begin
            # Prefer operational cost, because target slack is a training aid.
            loss = isempty(sample_log.objectives_no_deficit) ?
                NaN :
                mean(sample_log.objectives_no_deficit)

            # Store one row per SGD batch.
            push!(training_log, (batch = iteration, loss = loss))

            # Print sparse progress so long runs are inspectable.
            if iteration == 1 || mod(iteration, 50) == 0
                println(
                    "  batch $(lpad(iteration, 4))/$(variant.num_batches)  " *
                    "loss=$(round(loss, digits = 1))",
                )
            end

            # Save the best policy seen so far.
            save_best(iteration, current_policy, loss)

            return false
        end,
    )

    # Persist the training curve after training finishes.
    CSV.write(curve_path, training_log)

    println("Training time: $(round(elapsed_seconds, digits = 1))s")

    return elapsed_seconds
end

"""
    save_evaluation_outputs(variant, costs, inventory_paths, train_seconds, eval_seconds)

Write rollout costs, inventory trajectories, and timing rows for one variant.

# Arguments
- `variant::InventoryTrainingVariant`: variant configuration.
- `costs::AbstractVector{<:Real}`: held-out operational costs.
- `inventory_paths::AbstractMatrix{<:Real}`: inventory trajectory matrix.
- `train_seconds::Real`: total training time.
- `eval_seconds::Real`: total rollout evaluation time.

# Examples
```julia
save_evaluation_outputs(variant, costs, inventory_paths, train_time, eval_time)
```
"""
function save_evaluation_outputs(
    variant::InventoryTrainingVariant,
    costs::AbstractVector{<:Real},
    inventory_paths::AbstractMatrix{<:Real},
    train_seconds::Real,
    eval_seconds::Real,
)
    # Name trajectory columns by period t=0,...,T.
    time_columns = [Symbol("t$(period)") for period in 0:INVENTORY_T]

    # Write inventory paths for plotting.
    CSV.write(
        joinpath(RESULT_DIR, "$(variant.tag)_dr_trajectories.csv"),
        DataFrame(inventory_paths, time_columns),
    )

    # Write one operational-cost row per held-out scenario.
    CSV.write(
        joinpath(RESULT_DIR, "$(variant.tag)_dr_costs.csv"),
        DataFrame(
            scenario = 1:length(costs),
            operational_cost = costs,
        ),
    )

    # Write timing in the shared schema consumed by compare_results.jl.
    CSV.write(
        joinpath(RESULT_DIR, "$(variant.tag)_dr_timing.csv"),
        DataFrame(
            method = [method_label(variant)],
            fit_seconds = [train_seconds],
            eval_seconds = [eval_seconds / (N_TEST_SCENARIOS * INVENTORY_T)],
            n_eval = [N_TEST_SCENARIOS],
        ),
    )

    return nothing
end

"""
    train_and_evaluate(variant::InventoryTrainingVariant)

Train one TS-DDR variant and evaluate it by stage-wise rollout.

# Arguments
- `variant::InventoryTrainingVariant`: variant configuration.

# Examples
```julia
costs = train_and_evaluate(variant)
```
"""
function train_and_evaluate(variant::InventoryTrainingVariant)
    # Keep model and curve paths tied to the variant tag.
    model_path = joinpath(MODEL_DIR, "$(variant.tag)_policy.jld2")
    curve_path = joinpath(RESULT_DIR, "$(variant.tag)_training_curve.csv")

    # Build the training deterministic equivalent.
    det_eq, train_state_in, train_state_out, train_sampler, initial_state =
        build_training_problem(variant)

    # Build separate stage-wise models for deployment evaluation.
    eval_subproblems, eval_state_in, eval_state_out, eval_sampler, _ =
        build_evaluation_problem(variant)

    # Start every variant from the same policy initialization.
    policy = build_exante_policy(; seed = 2024)

    # Train the policy and save the best checkpoint.
    train_seconds = train_variant!(
        policy,
        variant,
        det_eq,
        train_state_in,
        train_state_out,
        train_sampler,
        initial_state,
        model_path,
        curve_path,
    )

    # Reload the best checkpoint before evaluation.
    Flux.loadmodel!(policy, JLD2.load(model_path, "model_state"))

    # Allocate outer variables so the timed block can assign them.
    costs = Float64[]
    inventory_paths = Matrix{Float64}(undef, 0, 0)

    # Evaluate under deployment semantics and time only the rollout solve work.
    eval_seconds = @elapsed begin
        rollout_costs, rollout_inventory_paths, _setup_paths, _order_paths =
            rollout_policy(
            policy,
            eval_subproblems,
            eval_state_in,
            eval_state_out,
            eval_sampler,
            initial_state;
            integer = variant.integer,
        )

        # Copy the rollout results into the outer scope.
        costs = rollout_costs
        inventory_paths = rollout_inventory_paths
    end

    # Write all result files after the elapsed time is known.
    save_evaluation_outputs(
        variant,
        costs,
        inventory_paths,
        train_seconds,
        eval_seconds,
    )

    # Print the headline cost distribution for this variant.
    mean_cost = mean(costs)
    std_cost = std(costs)
    seconds_per_stage = eval_seconds / (N_TEST_SCENARIOS * INVENTORY_T)
    println(
        "Result: $(round(mean_cost, digits = 1)) +- " *
        "$(round(std_cost, digits = 1))  " *
        "(eval/stage: $(round(seconds_per_stage, digits = 4))s)",
    )

    return costs
end

"""
    score_function_variant() -> InventoryTrainingVariant

Build the mixed-gradient integer variant.

The dual path uses `FixedDiscreteIntegerStrategy`. The score-function path uses
separate integer rollout subproblems, so the Monte Carlo costs are true MIP
rollout costs.

# Examples
```julia
variant = score_function_variant()
```
"""
function score_function_variant()
    # Score-function rollouts use separate models so training solves do not
    # mutate the deterministic-equivalent model.
    rollout_subproblems, rollout_state_in, rollout_state_out, _sampler, _ =
        build_inventory_subproblems(;
            num_scenarios = N_TRAIN_SCENARIOS,
            penalty = INVENTORY_PENALTY,
            seed = 77,
            integer = true,
        )

    # The score-function config describes the final estimator settings.
    score_config = ScoreFunctionConfig(
        rollout_subproblems,
        rollout_state_in,
        rollout_state_out;
        dual_weight = 0.5,
        perturbation_std = 1.0,
        num_rollouts = 8,
    )

    # The schedule phases the Monte Carlo correction in after dual-only warmup.
    score_schedule = ScoreFunctionSchedule(
        score_config;
        sf_start = 200,
        ramp_batches = 300,
        perturbation_std_initial = 0.1,
        num_rollouts_initial = 2,
    )

    return InventoryTrainingVariant(
        "integer_sf",
        true,
        800,
        10,
        8.0e-4,
        120,
        FixedDiscreteIntegerStrategy(),
        score_schedule,
    )
end

"""
    inventory_training_variants() -> Vector{InventoryTrainingVariant}

Return all TS-DDR variants used in the benchmark.

# Examples
```julia
for variant in inventory_training_variants()
    train_and_evaluate(variant)
end
```
"""
function inventory_training_variants()
    return [
        InventoryTrainingVariant(
            "relaxed",
            false,
            400,
            5,
            1.5e-3,
            80,
            NoIntegerStrategy(),
            nothing,
        ),
        InventoryTrainingVariant(
            "integer",
            true,
            800,
            10,
            8.0e-4,
            120,
            FixedDiscreteIntegerStrategy(),
            nothing,
        ),
        InventoryTrainingVariant(
            "integer_cr",
            true,
            800,
            10,
            8.0e-4,
            120,
            ContinuousRelaxationIntegerStrategy(),
            nothing,
        ),
        score_function_variant(),
    ]
end

"""
    main() -> Nothing

Run the full inventory TS-DDR training benchmark.

# Examples
```julia
main()
```
"""
function main()
    # Train variants sequentially so output files are deterministic.
    for variant in inventory_training_variants()
        train_and_evaluate(variant)
        println()
    end

    println("All TS-DDR results saved to $(relpath(RESULT_DIR, EXAMPLE_DIR))")

    return nothing
end

# Run the script only when invoked directly, not when included by tests.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
