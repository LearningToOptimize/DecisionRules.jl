"""
    ScoreFunctionConfig(
        subproblems::AbstractVector{<:JuMP.Model},
        state_params_in::AbstractVector,
        state_params_out::AbstractVector;
        dual_weight::Real = 0.5,
        perturbation_std::Real = 1.0,
        num_rollouts::Integer = 8,
        baseline::Symbol = :mean,
    )

Configure the score-function correction used by [`train_multistage`](@ref).

The deterministic-equivalent training path differentiates the target policy
through dual information. For mixed-integer subproblems, those duals are local
to a fixed integer assignment. This configuration adds a REINFORCE-style
correction estimated from stage-wise rollouts with perturbed targets.

The rollout models are solved exactly as they are built. If `subproblems`
contain binary variables, the score-function rollouts solve MIPs. If they
contain relaxed variables, the score-function rollouts solve the relaxation.
This is intentionally separate from the `integer_strategy` keyword of
[`train_multistage`](@ref), which controls only how the differentiable
dual-gradient path reads local sensitivity information from the deterministic
equivalent.

If ``\\hat{x}_{t+1}(\\theta)`` is the target emitted by the policy and
``\\delta_t \\sim \\mathcal{N}(0, \\sigma^2 I)``, the perturbed rollout solves
with target ``\\hat{x}_{t+1}(\\theta) + \\delta_t``. The score-function
surrogate loss is

```math
L_{sf}(\\theta)
    = \\frac{1}{M} \\sum_{m=1}^{M}
      (R_m - b)
      \\sum_{t=1}^{T}
      \\left\\langle \\frac{\\delta_{m,t}}{\\sigma^2},
      \\hat{x}_{t+1}(\\theta) \\right\\rangle ,
```

and the mixed gradient is

```math
\\nabla L
    = \\alpha \\nabla L_{dual}
      + (1 - \\alpha) \\nabla L_{sf}.
```

# Arguments
- `subproblems::AbstractVector{<:JuMP.Model}`: stage-wise rollout models used to
  estimate realized costs under perturbed targets.
- `state_params_in::AbstractVector`: stage input-state parameters.
- `state_params_out::AbstractVector`: pairs `(target_parameter, state_variable)`
  for every stage output state.

# Keywords
- `dual_weight::Real`: mixing weight ``\\alpha`` on the dual-gradient term.
- `perturbation_std::Real`: Gaussian standard deviation ``\\sigma``.
- `num_rollouts::Integer`: number of perturbed rollouts ``M`` per sample.
- `baseline::Symbol`: either `:mean` for mean-centering costs or `:none`.

# Examples
```julia
score_function = ScoreFunctionConfig(
    rollout_subproblems,
    state_params_in,
    state_params_out;
    dual_weight = 0.5,
    perturbation_std = 1.0,
    num_rollouts = 8,
)

train_multistage(
    policy,
    initial_state,
    det_equivalent,
    state_params_in,
    state_params_out,
    uncertainty_sampler;
    score_function,
)
```
"""
struct ScoreFunctionConfig
    subproblems::Vector{JuMP.Model}
    state_params_in::AbstractVector
    state_params_out::AbstractVector
    dual_weight::Float64
    perturbation_std::Float64
    num_rollouts::Int
    baseline::Symbol
end

function ScoreFunctionConfig(
    subproblems::AbstractVector{<:JuMP.Model},
    state_params_in::AbstractVector,
    state_params_out::AbstractVector;
    dual_weight::Real = 0.5,
    perturbation_std::Real = 1.0,
    num_rollouts::Integer = 8,
    baseline::Symbol = :mean,
)
    # Validate dimensions first so later rollout errors point at modeling issues.
    length(subproblems) == length(state_params_in) ||
        throw(ArgumentError("state_params_in must have one entry per subproblem."))
    length(subproblems) == length(state_params_out) ||
        throw(ArgumentError("state_params_out must have one entry per subproblem."))

    # Convert scalar configuration values once, at construction time.
    dual_weight_value = Float64(dual_weight)
    perturbation_std_value = Float64(perturbation_std)
    num_rollouts_value = Int(num_rollouts)

    # Keep validation messages tied to the public keyword names.
    0.0 <= dual_weight_value <= 1.0 ||
        throw(ArgumentError("dual_weight must lie in [0, 1]."))
    perturbation_std_value > 0.0 ||
        throw(ArgumentError("perturbation_std must be positive."))
    num_rollouts_value >= 1 ||
        throw(ArgumentError("num_rollouts must be at least 1."))
    baseline in (:mean, :none) ||
        throw(ArgumentError("baseline must be either :mean or :none."))

    # Store a plain Vector of models so iteration order is concrete and stable.
    return ScoreFunctionConfig(
        collect(subproblems),
        state_params_in,
        state_params_out,
        dual_weight_value,
        perturbation_std_value,
        num_rollouts_value,
        baseline,
    )
end

"""
    ScoreFunctionSchedule(config::ScoreFunctionConfig; <keyword arguments>)

Ramp a [`ScoreFunctionConfig`](@ref) into training after a pure-dual warmup.

The schedule delays score-function rollouts until `sf_start`, then linearly
increases the score-function weight, perturbation scale, and rollout count until
the final values stored in `config` are reached.

Let ``k`` be the current iteration and
``\\rho_k = \\operatorname{clip}((k - k_0) / r, 0, 1)``. The effective
score-function weight is
``\\rho_k (1 - \\alpha)``. The effective dual weight is one minus that value.

# Arguments
- `config::ScoreFunctionConfig`: final score-function configuration.

# Keywords
- `sf_start::Integer`: first iteration at which score-function rollouts are
  considered.
- `ramp_batches::Integer`: number of iterations in the linear ramp.
- `perturbation_std_initial::Real`: initial ``\\sigma`` at ramp start.
- `num_rollouts_initial::Integer`: initial rollout count at ramp start.

# Examples
```julia
schedule = ScoreFunctionSchedule(
    score_function;
    sf_start = 200,
    ramp_batches = 300,
    perturbation_std_initial = 0.1,
    num_rollouts_initial = 2,
)
```
"""
struct ScoreFunctionSchedule
    config::ScoreFunctionConfig
    sf_start::Int
    ramp_batches::Int
    final_dual_weight::Float64
    initial_perturbation_std::Float64
    final_perturbation_std::Float64
    initial_num_rollouts::Int
    final_num_rollouts::Int
end

function ScoreFunctionSchedule(
    config::ScoreFunctionConfig;
    sf_start::Integer = 200,
    ramp_batches::Integer = 200,
    perturbation_std_initial::Real = 0.1,
    num_rollouts_initial::Integer = 2,
)
    # Convert public numeric inputs before validation.
    sf_start_value = Int(sf_start)
    ramp_batches_value = Int(ramp_batches)
    initial_std_value = Float64(perturbation_std_initial)
    initial_rollouts_value = Int(num_rollouts_initial)

    # Reject invalid schedules with keyword-specific messages.
    sf_start_value >= 1 ||
        throw(ArgumentError("sf_start must be at least 1."))
    ramp_batches_value >= 1 ||
        throw(ArgumentError("ramp_batches must be at least 1."))
    initial_std_value > 0.0 ||
        throw(ArgumentError("perturbation_std_initial must be positive."))
    initial_rollouts_value >= 1 ||
        throw(ArgumentError("num_rollouts_initial must be at least 1."))

    return ScoreFunctionSchedule(
        config,
        sf_start_value,
        ramp_batches_value,
        config.dual_weight,
        initial_std_value,
        config.perturbation_std,
        initial_rollouts_value,
        config.num_rollouts,
    )
end

const _ScoreFunctionParameters = @NamedTuple{
    alpha::Float64,
    score_weight::Float64,
    perturbation_std::Float64,
    num_rollouts::Int,
    active::Bool,
}

"""
    sf_params(config::ScoreFunctionConfig, iteration::Integer)
    sf_params(schedule::ScoreFunctionSchedule, iteration::Integer)

Return the effective score-function parameters for `iteration`.

# Arguments
- `config::ScoreFunctionConfig`: unscheduled score-function configuration.
- `schedule::ScoreFunctionSchedule`: scheduled score-function configuration.
- `iteration::Integer`: one-based training iteration.

# Returns
A named tuple with fields:
- `alpha::Float64`: weight on the dual-gradient term.
- `score_weight::Float64`: weight on the score-function term.
- `perturbation_std::Float64`: Gaussian standard deviation ``\\sigma``.
- `num_rollouts::Int`: number of perturbed rollouts.
- `active::Bool`: whether rollout estimation should run.

# Examples
```julia
params = sf_params(schedule, 250)
params.active && @show params.score_weight
```
"""
function sf_params(
    config::ScoreFunctionConfig,
    ::Integer,
)::_ScoreFunctionParameters
    # Static configurations are active at every training iteration.
    return (
        alpha = config.dual_weight,
        score_weight = 1.0 - config.dual_weight,
        perturbation_std = config.perturbation_std,
        num_rollouts = config.num_rollouts,
        active = true,
    )
end

function sf_params(
    schedule::ScoreFunctionSchedule,
    iteration::Integer,
)::_ScoreFunctionParameters
    # Before warmup ends, keep the original deterministic-equivalent gradient.
    if iteration < schedule.sf_start
        return (
            alpha = 1.0,
            score_weight = 0.0,
            perturbation_std = 0.0,
            num_rollouts = 0,
            active = false,
        )
    end

    # A clipped ramp fraction keeps all interpolated values inside bounds.
    ramp_fraction = clamp(
        (iteration - schedule.sf_start) / schedule.ramp_batches,
        0.0,
        1.0,
    )

    # The score-function weight follows the linear ramp.
    uncapped_score_weight = ramp_fraction * (1.0 - schedule.final_dual_weight)

    # Interpolate the perturbation scale and rollout count over the same ramp.
    perturbation_std = schedule.initial_perturbation_std +
        ramp_fraction *
        (schedule.final_perturbation_std - schedule.initial_perturbation_std)
    num_rollouts = round(
        Int,
        schedule.initial_num_rollouts +
            ramp_fraction *
            (schedule.final_num_rollouts - schedule.initial_num_rollouts),
    )

    return (
        alpha = 1.0 - uncapped_score_weight,
        score_weight = uncapped_score_weight,
        perturbation_std = perturbation_std,
        num_rollouts = max(1, num_rollouts),
        active = true,
    )
end

"""
    _sf_config(score_function) -> Union{Nothing,ScoreFunctionConfig}

Extract the underlying [`ScoreFunctionConfig`](@ref), if one exists.

# Arguments
- `score_function::Nothing`: score-function correction is disabled.
- `score_function::ScoreFunctionConfig`: returned as-is.
- `score_function::ScoreFunctionSchedule`: unwraps `score_function.config`.

# Examples
```julia
config = DecisionRules._sf_config(score_function)
```
"""
_sf_config(::Nothing) = nothing
_sf_config(config::ScoreFunctionConfig) = config
_sf_config(schedule::ScoreFunctionSchedule) = schedule.config

"""
    _set_score_function_stage_parameters!(
        state_params_in,
        state_params_out,
        uncertainties,
        state,
        target,
    ) -> Nothing

Set the JuMP parameters needed for one perturbed rollout stage.

# Arguments
- `state_params_in::AbstractVector`: parameters receiving the current state.
- `state_params_out::AbstractVector`: `(target_parameter, state_variable)`
  pairs receiving the target state.
- `uncertainties::AbstractVector`: `(parameter, value)` pairs for stage
  uncertainty.
- `state::AbstractVector{<:Real}`: realized state entering this stage.
- `target::AbstractVector{<:Real}`: perturbed target for the output state.

# Examples
```julia
DecisionRules._set_score_function_stage_parameters!(
    spi[t],
    spo[t],
    uncertainty_sample[t],
    state,
    target,
)
```
"""
function _set_score_function_stage_parameters!(
    state_params_in,
    state_params_out,
    uncertainties,
    state::AbstractVector{<:Real},
    target::AbstractVector{<:Real},
)
    # Input-state parameters receive the realized state from the prior stage.
    for index in eachindex(state_params_in)
        set_parameter_value(state_params_in[index], state[index])
    end

    # Uncertainty parameters receive the sampled exogenous values.
    for (parameter, value) in uncertainties
        set_parameter_value(parameter, value)
    end

    # Output target parameters receive the perturbed policy targets.
    for index in eachindex(state_params_out)
        set_parameter_value(state_params_out[index][1], target[index])
    end

    return nothing
end

"""
    rollout_with_perturbation(
        config::ScoreFunctionConfig,
        initial_state::AbstractVector,
        uncertainties,
        targets,
        perturbations,
    ) -> Float64

Run one stage-wise rollout with fixed target perturbations.

The rollout target at stage `t` is `targets[t + 1] + perturbations[t]`. The
returned cost excludes the target-deficit penalty so the score-function signal
estimates operational cost rather than target-following slack.

# Arguments
- `config::ScoreFunctionConfig`: rollout models and parameter mappings.
- `initial_state::AbstractVector`: state entering stage 1.
- `uncertainties`: sampled uncertainty trajectory.
- `targets`: target trajectory, including `targets[1] == initial_state`.
- `perturbations`: one perturbation vector for each stage target.

# Examples
```julia
cost = DecisionRules.rollout_with_perturbation(
    score_function,
    initial_state,
    uncertainty_sample,
    targets,
    perturbations,
)
```
"""
function rollout_with_perturbation(
    config::ScoreFunctionConfig,
    initial_state::AbstractVector,
    uncertainties,
    targets,
    perturbations,
)::Float64
    # Rollouts always start from the true initial state.
    state = Float64.(initial_state)

    # Accumulate operational cost over the horizon.
    total_cost = 0.0

    for stage in eachindex(config.subproblems)
        # The deterministic target sequence includes the initial state at index 1.
        target = Float64.(targets[stage + 1]) .+ perturbations[stage]

        # Set all model parameters before solving this stage.
        _set_score_function_stage_parameters!(
            config.state_params_in[stage],
            config.state_params_out[stage],
            uncertainties[stage],
            state,
            target,
        )

        # Score-function rollouts need realized costs, not duals, so solve the
        # model exactly as it was built.
        optimize!(config.subproblems[stage])

        # Fail loudly when a sampled rollout is not solved to a usable status.
        _assert_successful_solve(
            config.subproblems[stage];
            context = "score-function rollout solve",
        )

        # Read the operational cost after the successful solve.
        stage_cost = get_objective_no_target_deficit(config.subproblems[stage])

        # Read the realized output state that becomes the next input state.
        next_state = Float64.([
            JuMP.value(config.state_params_out[stage][index][2])
            for index in eachindex(config.state_params_out[stage])
        ])

        # Feed the realized output state to the next stage.
        total_cost += stage_cost
        state = next_state
    end

    return total_cost
end

"""
    _sample_target_perturbations(num_stages::Integer, state_dimension::Integer, sigma::Real)

Draw Gaussian target perturbations for one score-function rollout.

# Arguments
- `num_stages::Integer`: number of stage targets to perturb.
- `state_dimension::Integer`: length of each target state vector.
- `sigma::Real`: Gaussian standard deviation ``\\sigma``.

# Examples
```julia
perturbations = DecisionRules._sample_target_perturbations(3, 2, 0.5)
```
"""
function _sample_target_perturbations(
    num_stages::Integer,
    state_dimension::Integer,
    sigma::Real,
)
    # Multiplying standard normal draws by sigma stores actual perturbations.
    return [Float64(sigma) .* randn(Int(state_dimension)) for _ in 1:Int(num_stages)]
end

"""
    _center_rollout_costs(costs::AbstractVector{<:Real}, baseline::Symbol)

Convert rollout costs into score-function advantages.

# Arguments
- `costs::AbstractVector{<:Real}`: operational costs from perturbed rollouts.
- `baseline::Symbol`: either `:mean` or `:none`.

# Examples
```julia
advantages = DecisionRules._center_rollout_costs([10.0, 12.0], :mean)
```
"""
function _center_rollout_costs(
    costs::AbstractVector{<:Real},
    baseline::Symbol,
)
    # A mean baseline reduces variance without changing the expected gradient.
    baseline_value = baseline === :mean ? mean(costs) : 0.0

    return Float64.(costs) .- baseline_value
end

"""
    _score_function_rollouts(
        config::ScoreFunctionConfig,
        initial_state::AbstractVector,
        uncertainties,
        targets;
        perturbation_std = config.perturbation_std,
        num_rollouts = config.num_rollouts,
    ) -> (advantages, perturbations)

Estimate rollout advantages for the score-function term.

# Arguments
- `config::ScoreFunctionConfig`: score-function rollout configuration.
- `initial_state::AbstractVector`: state entering stage 1.
- `uncertainties`: sampled uncertainty trajectory.
- `targets`: target trajectory, including the initial state.
- `perturbation_std::Real`: Gaussian standard deviation ``\\sigma``.
- `num_rollouts::Integer`: number of perturbed rollouts to sample.

# Examples
```julia
advantages, perturbations = DecisionRules._score_function_rollouts(
    score_function,
    initial_state,
    uncertainty_sample,
    targets;
    perturbation_std = 0.5,
    num_rollouts = 4,
)
```
"""
function _score_function_rollouts(
    config::ScoreFunctionConfig,
    initial_state::AbstractVector,
    uncertainties,
    targets;
    perturbation_std::Real = config.perturbation_std,
    num_rollouts::Integer = config.num_rollouts,
)
    # Use the first target after the initial state to infer the state dimension.
    state_dimension = length(targets[2])
    num_stages = length(config.subproblems)

    # Allocate both arrays up front so each rollout has a visible slot.
    costs = Vector{Float64}(undef, Int(num_rollouts))
    perturbations = Vector{Vector{Vector{Float64}}}(undef, Int(num_rollouts))

    for rollout in eachindex(costs)
        # Draw perturbations once, then reuse them in the surrogate gradient.
        perturbations[rollout] = _sample_target_perturbations(
            num_stages,
            state_dimension,
            perturbation_std,
        )

        # Evaluate the realized cost under the perturbed target trajectory.
        costs[rollout] = rollout_with_perturbation(
            config,
            initial_state,
            uncertainties,
            targets,
            perturbations[rollout],
        )
    end

    return _center_rollout_costs(costs, config.baseline), perturbations
end

"""
    _score_function_surrogate(
        advantage::Real,
        perturbations,
        targets,
        perturbation_std::Real,
    ) -> Real

Build the differentiable scalar whose gradient is the Gaussian score estimate.

For fixed rollout cost advantage ``A`` and perturbations ``\\delta_t``, the
surrogate is

```math
A \\sum_t \\left\\langle
    \\delta_t / \\sigma^2, \\hat{x}_{t+1}(\\theta)
\\right\\rangle .
```

# Arguments
- `advantage::Real`: centered rollout cost ``R - b``.
- `perturbations`: stage perturbations ``\\delta_t``.
- `targets`: differentiable target trajectory produced by the policy.
- `perturbation_std::Real`: Gaussian standard deviation ``\\sigma``.

# Examples
```julia
loss = DecisionRules._score_function_surrogate(
    3.0,
    perturbations,
    targets,
    0.5,
)
```
"""
function _score_function_surrogate(
    advantage::Real,
    perturbations,
    targets,
    perturbation_std::Real,
)
    # The Gaussian location score divides actual perturbations by sigma squared.
    inverse_variance = inv(Float32(perturbation_std)^2)

    # Targets include the initial state, so stage t uses targets[t + 1].
    score = sum(eachindex(perturbations)) do stage
        sum(Float32.(perturbations[stage]) .* targets[stage + 1]) *
            inverse_variance
    end

    return Float32(advantage) * score
end
