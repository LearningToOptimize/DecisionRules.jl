"""
    variable_to_parameter(model, variable; initial_value=0.0, deficit=nothing)

Replace a decision variable with an `MOI.Parameter` and bind them via an equality
constraint.  When `deficit` is provided, the constraint becomes
`variable + deficit == parameter` and both the parameter and the deficit variable
are returned.
"""
function variable_to_parameter(
    model::JuMP.Model, variable::JuMP.VariableRef; initial_value=0.0, deficit=nothing
)
    parameter = @variable(
        model; base_name="_" * name(variable), set=MOI.Parameter(initial_value)
    )
    # bind the parameter to the variable
    if isnothing(deficit)
        @constraint(model, variable == parameter)
        return parameter
    else
        @constraint(model, variable + deficit == parameter)
        return parameter, variable
    end
end

"""
    create_deficit!(model::JuMP.Model, len::Int; penalty_l1=nothing, penalty_l2=nothing, penalty=nothing)

Create deficit variables to penalize state deviations in a JuMP model.

Supports three modes controlled by the penalty keywords. Let
``d \\in \\mathbb{R}^n`` be the deficit vector (`len = n`).

**L1 norm only** (default when no penalty keyword is given, or `penalty_l1`
alone):

```math
\\text{norm\\_deficit} \\geq \\| d \\|_1 = \\sum_{i=1}^{n} |d_i|,
\\quad \\text{objective} \\mathrel{+}= \\lambda_1 \\cdot \\text{norm\\_deficit}.
```

Implemented via `MOI.NormOneCone(1 + n)`.

**L2 squared norm only** (`penalty_l2` alone):

```math
\\text{norm\\_deficit} \\geq \\| d \\|_2^2 = \\sum_{i=1}^{n} d_i^2,
\\quad \\text{objective} \\mathrel{+}= \\lambda_2 \\cdot \\text{norm\\_deficit}.
```

**Both norms** (`penalty_l1` and `penalty_l2`):

```math
\\text{norm\\_deficit}
    \\geq \\lambda_1 \\| d \\|_1 + \\lambda_2 \\| d \\|_2^2,
\\quad \\text{objective} \\mathrel{+}= 1 \\cdot \\text{norm\\_deficit}.
```

# Arguments
- `model`: The JuMP model to add deficit variables to.
- `len`: Number of deficit variables (typically dimension of state).
- `penalty_l1`: Penalty coefficient ``\\lambda_1`` for the L1 norm
  (`NormOneCone`). Pass `:auto` to use `max |objective coefficients|`.
- `penalty_l2`: Penalty coefficient ``\\lambda_2`` for the L2 squared norm
  (sum of squares). Pass `:auto` to use `max |objective coefficients|`.
- `penalty`: Legacy argument. If provided and `penalty_l1`/`penalty_l2` are
  both `nothing`, uses this for L1 norm only.

# Returns
- `norm_deficit`: Single variable representing total penalized deviation (for
  logging compatibility).
- `_deficit`: Vector of deficit variables for each state dimension.

# Examples
```julia
# L1 norm only (default behavior, backwards compatible)
norm_deficit, _deficit = create_deficit!(model, 3; penalty=1000.0)

# L2 norm only
norm_deficit, _deficit = create_deficit!(model, 3; penalty_l2=1000.0)

# Both L1 and L2 norms
norm_deficit, _deficit = create_deficit!(model, 3; penalty_l1=1000.0, penalty_l2=500.0)
```
"""
function create_deficit!(
    model::JuMP.Model, len::Int; penalty_l1=nothing, penalty_l2=nothing, penalty=nothing
)
    # Handle legacy 'penalty' argument for backwards compatibility
    if isnothing(penalty_l1) && isnothing(penalty_l2)
        if !isnothing(penalty)
            penalty_l1 = penalty  # Use legacy penalty for L1 only
        else
            # Default: L1 norm with auto-computed penalty
            obj = objective_function(model)
            penalty_l1 = maximum(abs.(values(obj.terms)))
        end
    end

    # Auto-compute penalties if needed
    if !isnothing(penalty_l1) && penalty_l1 === :auto
        obj = objective_function(model)
        penalty_l1 = maximum(abs.(values(obj.terms)))
    end
    if !isnothing(penalty_l2) && penalty_l2 === :auto
        obj = objective_function(model)
        penalty_l2 = maximum(abs.(values(obj.terms)))
    end

    # Create deficit variables
    _deficit = @variable(model, _deficit[1:len])

    # Create individual norm variables for each cone type
    use_l1 = !isnothing(penalty_l1)
    use_l2 = !isnothing(penalty_l2)

    # Create norm_deficit as the total penalized deviation (for logging compatibility)
    @variable(model, norm_deficit >= 0.0)

    if use_l1 && use_l2
        # Both L1 and L2 squared norms
        @variable(model, norm_l1 >= 0.0)
        @variable(model, norm_l2_sq >= 0.0)  # L2 squared (sum of squares)
        @constraint(model, [norm_l1; _deficit] in MOI.NormOneCone(1 + len))
        @constraint(model, norm_l2_sq >= sum(_deficit[i]^2 for i in 1:len))
        # norm_deficit = penalty_l1 * norm_l1 + penalty_l2 * norm_l2_sq
        @constraint(model, norm_deficit >= penalty_l1 * norm_l1 + penalty_l2 * norm_l2_sq)
        set_objective_coefficient(model, norm_deficit, 1.0)
    elseif use_l1
        # L1 norm only
        @constraint(model, [norm_deficit; _deficit] in MOI.NormOneCone(1 + len))
        set_objective_coefficient(model, norm_deficit, penalty_l1)
    elseif use_l2
        # L2 squared norm only (sum of squares)
        @constraint(model, norm_deficit >= sum(_deficit[i]^2 for i in 1:len))
        set_objective_coefficient(model, norm_deficit, penalty_l2)
    else
        error("At least one of penalty_l1 or penalty_l2 must be specified")
    end

    return norm_deficit, _deficit
end

"""
    default_annealed_schedule(num_batches::Int)

Build the default annealed target-penalty schedule over `num_batches` training batches:
multipliers `0.1 -> 1.0 -> 10.0 -> 30.0` with phase lengths proportional to `2/2/4/16`
of the horizon (the last phase takes the remainder; every phase keeps at least one batch).
For `num_batches < 4` the last `num_batches` multipliers are used, one batch each, so the
run always ends at the strong-penalty phase.

Returns a `Vector{Tuple{Int,Int,Float64}}` of `(first_batch, last_batch, multiplier)`
entries suitable for the `penalty_schedule` keyword of [`train_multistage`](@ref) and
`train_multiple_shooting`. Multipliers are applied **relative to the penalty the model
was built with** (the objective coefficient of the `norm_deficit` variables created by
[`create_deficit!`](@ref)), so with `penalty=:auto` the effective penalty is
`multiplier * max |objective coefficient|`.
"""
function default_annealed_schedule(num_batches::Int)
    num_batches >= 1 || throw(ArgumentError("num_batches must be >= 1"))
    multipliers = [0.1, 1.0, 10.0, 30.0]
    n = length(multipliers)
    if num_batches < n
        mults = multipliers[(n - num_batches + 1):n]
        return [(i, i, mults[i]) for i in 1:num_batches]
    end
    # Phase lengths proportional to 2/2/4/16 over 24; remainder goes to the last phase.
    lengths = [max(1, round(Int, num_batches * f)) for f in (2 / 24, 2 / 24, 4 / 24)]
    excess = sum(lengths) + 1 - num_batches
    i = length(lengths)
    while excess > 0 && i >= 1
        take = min(lengths[i] - 1, excess)
        lengths[i] -= take
        excess -= take
        i -= 1
    end
    push!(lengths, num_batches - sum(lengths))
    schedule = Vector{Tuple{Int,Int,Float64}}(undef, n)
    lo = 1
    for k in 1:n
        hi = lo + lengths[k] - 1
        schedule[k] = (lo, hi, multipliers[k])
        lo = hi + 1
    end
    return schedule
end

"""
    _validate_penalty_schedule(schedule) -> typeof(schedule)

Validate an explicit `penalty_schedule`, a `Vector` of `(first_batch, last_batch,
multiplier)` tuples: phases must be non-empty, contiguous, start at batch 1, satisfy
`first_batch <= last_batch`, and have finite positive multipliers. Return `schedule`
unchanged, or throw `ArgumentError` describing the first violation found.
"""
function _validate_penalty_schedule(schedule)
    isempty(schedule) && throw(ArgumentError("penalty_schedule must not be empty"))
    expected_lo = 1
    for (lo, hi, mult) in schedule
        lo == expected_lo || throw(
            ArgumentError(
                "penalty_schedule phases must be contiguous starting at batch 1; got phase ($lo, $hi, $mult) where first_batch $expected_lo was expected",
            ),
        )
        lo <= hi || throw(
            ArgumentError(
                "penalty_schedule phase ($lo, $hi, $mult) has first_batch > last_batch"
            ),
        )
        (isfinite(mult) && mult > 0) || throw(
            ArgumentError(
                "penalty_schedule multipliers must be finite and positive; got $mult"
            ),
        )
        expected_lo = hi + 1
    end
    return schedule
end

"""
    _resolve_penalty_schedule(penalty_schedule, num_batches::Int)

Resolve the `penalty_schedule` keyword of [`train_multistage`](@ref) and
`train_multiple_shooting` into a `Vector{Tuple{Int,Int,Float64}}` of `(first_batch,
last_batch, multiplier)` phases, or `nothing` if penalty scaling is disabled:

- `nothing` returns `nothing` (no scaling);
- `:default_annealed` returns [`default_annealed_schedule`](@ref)`(num_batches)`;
- any other value is checked with [`_validate_penalty_schedule`](@ref) and returned
  as-is.
"""
_resolve_penalty_schedule(::Nothing, num_batches::Int) = nothing
function _resolve_penalty_schedule(penalty_schedule::Symbol, num_batches::Int)
    penalty_schedule === :default_annealed && return default_annealed_schedule(num_batches)
    throw(
        ArgumentError(
            "unknown penalty_schedule symbol :$penalty_schedule; use :default_annealed, an explicit vector of (first_batch, last_batch, multiplier) tuples, or nothing",
        ),
    )
end
function _resolve_penalty_schedule(penalty_schedule, num_batches::Int)
    return _validate_penalty_schedule(penalty_schedule)
end

"""
    _penalty_multiplier_for(schedule, iter::Int) -> Float64

Return the multiplier of the phase containing batch `iter`. If `iter` is past the
schedule's last phase, return that phase's multiplier (hold the final value steady).
"""
function _penalty_multiplier_for(schedule, iter::Int)
    for (lo, hi, mult) in schedule
        lo <= iter <= hi && return mult
    end
    return schedule[end][3]
end

"""
    _linear_objective_coefficient(model::JuMP.Model, variable::VariableRef) -> Float64

Return `variable`'s linear coefficient in `model`'s objective, or `0.0` if `variable`
does not appear in it. Supports affine and quadratic objectives (quadratic terms are
ignored); throw `ArgumentError` for any other objective type.
"""
function _linear_objective_coefficient(model::JuMP.Model, variable::VariableRef)
    obj = objective_function(model)
    if obj isa GenericAffExpr
        return get(obj.terms, variable, 0.0)
    elseif obj isa GenericQuadExpr
        return get(obj.aff.terms, variable, 0.0)
    end
    throw(
        ArgumentError(
            "penalty_schedule requires an affine or quadratic objective; got $(typeof(obj))"
        ),
    )
end

"""
    _deficit_penalty_bases(model::JuMP.Model; deficit_name="norm_deficit") -> Dict{VariableRef,Float64}
    _deficit_penalty_bases(models::Vector{JuMP.Model}; deficit_name="norm_deficit") -> Vector{Dict{VariableRef,Float64}}

Capture the current objective coefficient of every deficit variable as the multiplier
base for [`_apply_deficit_penalty_multiplier!`](@ref). A variable counts as a deficit
variable if `deficit_name` occurs in its name and its linear objective coefficient
(see [`_linear_objective_coefficient`](@ref)) is nonzero.

Must be called **before** any `penalty_schedule` multiplier is applied, so the
captured coefficients reflect the as-built penalties.
"""
function _deficit_penalty_bases(
    model::JuMP.Model; deficit_name::AbstractString="norm_deficit"
)
    bases = Dict{VariableRef,Float64}()
    for variable in all_variables(model)
        if occursin(deficit_name, JuMP.name(variable))
            coef = _linear_objective_coefficient(model, variable)
            iszero(coef) || (bases[variable] = coef)
        end
    end
    return bases
end

function _deficit_penalty_bases(
    models::Vector{JuMP.Model}; deficit_name::AbstractString="norm_deficit"
)
    return [_deficit_penalty_bases(model; deficit_name=deficit_name) for model in models]
end

"""
    _check_deficit_penalty_bases(bases) -> typeof(bases)

Return `bases` (from [`_deficit_penalty_bases`](@ref)) unchanged if it has at least one
entry; otherwise throw `ArgumentError`, since a `penalty_schedule` would then have
nothing to scale.
"""
function _check_deficit_penalty_bases(bases)
    total = bases isa Dict ? length(bases) : sum(length, bases)
    total > 0 || throw(
        ArgumentError(
            "penalty_schedule was given but no variable matching \"norm_deficit\" with a nonzero objective coefficient was found; build the model(s) with create_deficit! (or equivalent) to use a penalty schedule",
        ),
    )
    return bases
end

"""
    _apply_deficit_penalty_multiplier!(model::JuMP.Model, bases::Dict{VariableRef,Float64}, multiplier::Real) -> JuMP.Model
    _apply_deficit_penalty_multiplier!(models::Vector{JuMP.Model}, bases::Vector, multiplier::Real) -> Vector{JuMP.Model}

Mutate `model` (or each model in `models`) in place, setting every deficit variable's
objective coefficient to `multiplier * base` using the bases from
[`_deficit_penalty_bases`](@ref). Return the mutated model(s).
"""
function _apply_deficit_penalty_multiplier!(
    model::JuMP.Model, bases::Dict{VariableRef,Float64}, multiplier::Real
)
    for (variable, base) in bases
        set_objective_coefficient(model, variable, multiplier * base)
    end
    return model
end

function _apply_deficit_penalty_multiplier!(
    models::Vector{JuMP.Model}, bases::Vector, multiplier::Real
)
    for (model, base) in zip(models, bases)
        _apply_deficit_penalty_multiplier!(model, base, multiplier)
    end
    return models
end

"""
    SaveBest(best_loss::Float64, model_path::String)

Callback that saves the best policy state seen during training.

`SaveBest` is a small callable object used as a training callback. When called
as `callback(iter, model, loss)`, it compares `loss` with the best loss stored
so far. If the new loss is smaller, it copies `model` to CPU, normalizes any
recurrent layer state, and writes the Flux state to `model_path` with JLD2.
It returns `false`, so it records checkpoints without stopping training.

# Arguments
- `best_loss::Float64`: incumbent loss. Use `Inf` to save the first observed
  model.
- `model_path::String`: path of the JLD2 file that receives the best model
  state.

# Examples
```julia
callback = SaveBest(Inf, "best_policy.jld2")
train_multistage(policy, x0, subproblems, state_in, state_out, sampler;
    record = (log, iter, model) -> callback(iter, model, mean(log.losses)))
```
"""
mutable struct SaveBest <: Function
    best_loss::Float64
    model_path::String
end

"""
    normalize_recur_state(state)

Return a copy of a `Flux.state` object where any Recur-like nodes have their
`state` field set to `cell.state0`. This avoids `Flux.loadmodel!` tie errors
when loading into freshly constructed recurrent layers.
"""
function normalize_recur_state(state)
    if state isa NamedTuple
        vals = map(normalize_recur_state, values(state))
        nt = NamedTuple{keys(state)}(vals)
        if (:cell in keys(nt)) &&
            (:state in keys(nt)) &&
            (getproperty(nt, :cell) isa NamedTuple) &&
            (:state0 in keys(getproperty(nt, :cell)))
            ks = keys(nt)
            newvals = map(
                k -> (k === :state ? getproperty(nt.cell, :state0) : getproperty(nt, k)), ks
            )
            return NamedTuple{ks}(newvals)
        end
        return nt
    elseif state isa Tuple
        return map(normalize_recur_state, state)
    else
        return state
    end
end

"""
    (callback::SaveBest)(iter, model, loss) -> Bool

Compare `loss` against the incumbent `callback.best_loss`. When `loss` is
strictly smaller, copy `model` to CPU, normalize recurrent state via
[`normalize_recur_state`](@ref), and write the Flux state to
`callback.model_path` with JLD2. Always returns `false` (never stops training).
"""
function (callback::SaveBest)(iter, model, loss)
    if loss < callback.best_loss
        m = cpu(model)
        @info "best model change" callback.best_loss loss
        callback.best_loss = loss
        model_state = normalize_recur_state(Flux.state(m))
        jldsave(callback.model_path; model_state=model_state)
    end
    return false
end

"""
    StallingCriterium(patience::Int, best_loss::Float64, stall_count::Int)

Early-stopping callback that halts training when the loss stalls.

Tracks the number of consecutive iterations without improvement. When
`stall_count` reaches `patience`, the callback returns `true` to signal that
training should stop. Use `best_loss = Inf` and `stall_count = 0` for a fresh
start.

# Arguments
- `patience::Int`: maximum consecutive non-improving iterations before stopping.
- `best_loss::Float64`: incumbent best loss. Use `Inf` to accept the first value.
- `stall_count::Int`: current stall counter (typically initialized to `0`).
"""
mutable struct StallingCriterium <: Function
    patience::Int
    best_loss::Float64
    stall_count::Int
end

"""
    (callback::StallingCriterium)(iter, model, loss) -> Bool

Update the stall counter and return `true` when `stall_count >= patience`.

If `loss < best_loss`, reset the counter to zero and update the incumbent.
Otherwise increment `stall_count`. Returns `true` (stop training) once the
patience budget is exhausted; `false` otherwise.
"""
function (callback::StallingCriterium)(iter, model, loss)
    if loss < callback.best_loss
        callback.best_loss = loss
        callback.stall_count = 0
    else
        callback.stall_count += 1
    end
    if callback.stall_count >= callback.patience
        return true
    end
    return false
end

"""
    SampleLog(; on_sample=(s, models, sample_log) -> nothing,
              objective_no_deficit_fn=get_objective_no_target_deficit)

Per-sample logger with local cache state for the training loops, patterned after
[`SaveBest`](@ref SaveBest). During each batch the training loop calls
`sample_log(s, det_equivalent_or_subproblems)` right after sample `s` has been simulated
(successful solves only; a failed solve throws exactly as before). The default behavior
caches, per sample, the full objective (`objective_value`) and the objective excluding
the target-slack penalty term (`objective_no_deficit_fn`). The cache is cleared at the
start of every batch and handed to the per-batch `record(sample_log, iter, model)`
callback.

`on_sample` is an optional hook called as `on_sample(s, models, sample_log)` after the
default caching. It receives the live JuMP model(s), so it can inspect termination
statuses or dump the details of a suspicious sample for debugging — without paying any
per-sample logging cost in the default configuration.
"""
mutable struct SampleLog <: Function
    on_sample::Function
    objective_no_deficit_fn::Function
    objectives::Vector{Float64}
    objectives_no_deficit::Vector{Float64}
end

function SampleLog(;
    on_sample=(s, models, sample_log) -> nothing,
    objective_no_deficit_fn=get_objective_no_target_deficit,
)
    return SampleLog(on_sample, objective_no_deficit_fn, Float64[], Float64[])
end

function Base.empty!(sample_log::SampleLog)
    empty!(sample_log.objectives)
    empty!(sample_log.objectives_no_deficit)
    return sample_log
end

"""
    _reset_sample_log!(sample_log::SampleLog) -> SampleLog
    _reset_sample_log!(sample_log) -> typeof(sample_log)

Clear the per-batch cache of a [`SampleLog`](@ref) before the next training
batch. For a `SampleLog`, delegates to `Base.empty!`; for any other type the
call is a no-op and returns `sample_log` unchanged.
"""
_reset_sample_log!(sample_log::SampleLog) = empty!(sample_log)
_reset_sample_log!(sample_log) = sample_log

"""
    _total_objective_value(model::JuMP.Model) -> Float64
    _total_objective_value(models::Vector{JuMP.Model}) -> Float64

Return the objective value of `model`, falling back to the cached value
`model.ext[:_last_obj]` (default `0.0`) when the model is dirty or
`objective_value` throws. The multi-model method sums across all models.
"""
function _total_objective_value(model::JuMP.Model)
    if model.is_model_dirty
        return get(model.ext, :_last_obj, 0.0)
    end
    try
        return objective_value(model)
    catch
        return get(model.ext, :_last_obj, 0.0)
    end
end
function _total_objective_value(models::Vector{JuMP.Model})
    total = 0.0
    for model in models
        total += _total_objective_value(model)
    end
    return total
end

function (sample_log::SampleLog)(s::Int, models)
    push!(sample_log.objectives, _total_objective_value(models))
    push!(sample_log.objectives_no_deficit, sample_log.objective_no_deficit_fn(models))
    sample_log.on_sample(s, models, sample_log)
    return nothing
end

"""
    _sequential_mean(values) -> Float64

Compute the arithmetic mean of `values` using sequential accumulation (left
fold). This preserves the same floating-point summation order as the historical
`loss += ...; loss /= n` pattern inside the training loops.
"""
# Sequential accumulation keeps the same floating-point summation order as the
# historical `loss += ...; loss /= n` pattern inside the training loops.
function _sequential_mean(values)
    total = 0.0
    for v in values
        total += v
    end
    return total / length(values)
end

"""
    default_record(sample_log, iter, model)

Default per-batch recording callback: prints the same two per-batch lines as the
historical `record_loss` default (`metrics/loss` = mean objective excluding the
target-slack penalty, then `metrics/training_loss` = mean full objective) and returns
`false` (training continues). Return `true` from a custom `record` to stop training.
"""
function default_record(sample_log::SampleLog, iter, model)
    println(
        "tag: metrics/loss, Iter: $iter, Loss: $(_sequential_mean(sample_log.objectives_no_deficit))",
    )
    println(
        "tag: metrics/training_loss, Iter: $iter, Loss: $(_sequential_mean(sample_log.objectives))",
    )
    return false
end

"""
    _record_loss_adapter(record_loss)

Adapt the deprecated 4-argument `record_loss(iter, model, loss, tag)` callback to the
`record(sample_log, iter, model)` interface, reproducing the historical two-call
contract: `record_loss` is called first with `tag = "metrics/loss"`, and only if that
returns `false` is it called again with `tag = "metrics/training_loss"`. Return the
result of whichever call is made last.
"""
function _record_loss_adapter(record_loss)
    return function (sample_log::SampleLog, iter, model)
        record_loss(
            iter, model, _sequential_mean(sample_log.objectives_no_deficit), "metrics/loss"
        ) && return true
        return record_loss(
            iter, model, _sequential_mean(sample_log.objectives), "metrics/training_loss"
        )
    end
end

"""
    _resolve_record(record, record_loss)

Resolve the `record`/`record_loss` keywords of the training loops into a single
per-batch callback. Return `record` unchanged if `record_loss` is `nothing`;
otherwise require `record === default_record` and return
[`_record_loss_adapter`](@ref)`(record_loss)`. Throw `ArgumentError` if both a custom
`record` and a `record_loss` are given.
"""
function _resolve_record(record, record_loss)
    isnothing(record_loss) && return record
    record === default_record || throw(
        ArgumentError("pass either `record` or the deprecated `record_loss`, not both")
    )
    return _record_loss_adapter(record_loss)
end

"""
    _target_violation_share(objective::Real, objective_no_deficit::Real) -> Float64

Return the target-violation share, `(objective - objective_no_deficit) / objective`.
Return `NaN` if either input is nonfinite or `abs(objective) <= 1e-12` (share
undefined).
"""
function _target_violation_share(objective::Real, objective_no_deficit::Real)
    slack = objective - objective_no_deficit
    (isfinite(objective) && isfinite(slack) && abs(objective) > 1e-12) || return NaN
    return slack / objective
end

"""
    RolloutEvaluation(subproblems, state_params_in, state_params_out, initial_state,
                      scenarios; stride=1, policy_state=:realized)

Evaluation helper that assesses the policy with a **stage-wise rollout** (the
deployment semantics of a target-trajectory policy) on a fixed held-out scenario set.
Deterministic-equivalent evaluation re-optimizes all stages jointly and can absorb
stage-wise-unfollowable targets through the slack penalty, silently overstating policy
quality; the rollout metric is the guard that detects this.

`policy_state` controls what state is passed back into the policy:

- `:realized` pipes the previous realized state into the policy. This is the
  deployment/closed-loop rollout semantics.
- `:target` pipes the previous target state into the policy, matching the
  deterministic-equivalent target-generation semantics from [`simulate_states`](@ref)
  while still solving the stage subproblems sequentially.

`scenarios` must be a vector of **materialized** scenarios, sampled once before
training (e.g. `[DecisionRules.sample(uncertainty_samples) for _ in 1:n]`), so every
evaluation uses the same fixed set. `subproblems` may be the training subproblems (all
stage parameters are rewritten on every solve) or a separately built copy; when
training on a deterministic equivalent, pass the stage-wise subproblems here.

Call `evaluation(iter, model)`, e.g. from within a `record` callback. Every `stride`
calls it rolls the policy out over the fixed set and reports:

- `metrics/rollout_objective_no_deficit`: the rollout objective excluding the
  target-slack penalty term (the operational cost), and
- `metrics/rollout_target_violation_share`: the realized slack penalty divided by the
  full rollout objective (`NaN` when undefined).

Policy comparisons should only be trusted when the violation share is small
(≤ ~0.05); a larger share means the policy's targets are not followable stage by stage
and the reported cost is not what deployment would realize. The latest values are kept
in `last_objective_no_deficit` / `last_violation_share` for custom logging. Calls on
batches that are not a multiple of `stride` are a no-op and leave the cached values
unchanged.
"""
mutable struct RolloutEvaluation <: Function
    subproblems::Vector{JuMP.Model}
    state_params_in
    state_params_out
    initial_state
    scenarios::Vector
    stride::Int
    policy_state::Symbol
    integer_strategy::AbstractIntegerStrategy
    gradient_fallback::AbstractGradientFallback
    last_objective_no_deficit::Float64
    last_violation_share::Float64
end

function RolloutEvaluation(
    subproblems,
    state_params_in,
    state_params_out,
    initial_state,
    scenarios;
    stride=1,
    policy_state::Symbol=:realized,
    integer_strategy::AbstractIntegerStrategy=NoIntegerStrategy(),
    gradient_fallback::AbstractGradientFallback=ZeroGradientFallback(),
)
    isempty(scenarios) && throw(
        ArgumentError(
            "scenarios must be a nonempty vector of materialized scenarios; sample them once before training so every evaluation uses the same fixed set",
        ),
    )
    stride >= 1 || throw(ArgumentError("stride must be >= 1"))
    policy_state in (:realized, :target) || throw(
        ArgumentError("policy_state must be either :realized or :target; got :$policy_state")
    )
    return RolloutEvaluation(
        subproblems,
        state_params_in,
        state_params_out,
        initial_state,
        collect(scenarios),
        stride,
        policy_state,
        integer_strategy,
        gradient_fallback,
        NaN,
        NaN,
    )
end

"""
    _simulate_multistage_target_feedback(subproblems, state_params_in,
        state_params_out, initial_state, uncertainties, decision_rules,
        integer_strategy) -> Float64

Run a stage-wise rollout where the **target** state (not the realized state) is
fed back into the policy at each stage, matching the deterministic-equivalent
target-generation semantics from [`simulate_states`](@ref). Each stage
subproblem is solved sequentially; the accumulated objective value (including
target-deficit penalties) is returned.

Used by [`RolloutEvaluation`](@ref) when `policy_state == :target`.
"""
function _simulate_multistage_target_feedback(
    subproblems::Vector{JuMP.Model},
    state_params_in,
    state_params_out,
    initial_state,
    uncertainties,
    decision_rules,
    integer_strategy::AbstractIntegerStrategy=NoIntegerStrategy(),
)
    Flux.reset!(decision_rules)
    target_states = simulate_states(initial_state, uncertainties, decision_rules)

    objective = 0.0
    state_in = initial_state
    for stage in 1:length(subproblems)
        subproblem = subproblems[stage]
        state_param_in = state_params_in[stage]
        state_param_out = state_params_out[stage]
        uncertainty = uncertainties[stage]
        target = target_states[stage + 1]
        objective += simulate_stage(
            subproblem,
            state_param_in,
            state_param_out,
            uncertainty,
            state_in,
            target;
            integer_strategy=integer_strategy,
        )
        state_in = get_next_state(
            subproblem,
            state_param_in,
            state_param_out,
            state_in,
            target;
            integer_strategy=integer_strategy,
        )
    end

    return objective
end

"""
    (evaluation::RolloutEvaluation)(iter, model) -> Nothing

Evaluate the policy on the held-out scenario set every `stride` iterations.

On active iterations (`iter % stride == 0`), rolls `model` out over all fixed
scenarios using either closed-loop (`:realized`) or target-feedback (`:target`)
semantics. Prints `metrics/rollout_objective_no_deficit` and
`metrics/rollout_target_violation_share`, and caches the values in
`evaluation.last_objective_no_deficit` / `evaluation.last_violation_share`.
Scenarios that fail to solve are skipped with a warning if all fail.
"""
function (evaluation::RolloutEvaluation)(iter, model)
    iter % evaluation.stride == 0 || return nothing
    total = 0.0
    total_no_deficit = 0.0
    n_success = 0
    for scenario in evaluation.scenarios
        obj = try
            if evaluation.policy_state === :realized
                simulate_multistage(
                    evaluation.subproblems,
                    evaluation.state_params_in,
                    evaluation.state_params_out,
                    evaluation.initial_state,
                    scenario,
                    model;
                    integer_strategy=evaluation.integer_strategy,
                )
            else
                _simulate_multistage_target_feedback(
                    evaluation.subproblems,
                    evaluation.state_params_in,
                    evaluation.state_params_out,
                    evaluation.initial_state,
                    scenario,
                    model,
                    evaluation.integer_strategy,
                )
            end
        catch e
            handle_rollout_error(evaluation.gradient_fallback, e, iter)
            nothing
        end
        isnothing(obj) && continue
        total += obj
        total_no_deficit += get_objective_no_target_deficit(evaluation.subproblems)
        n_success += 1
    end
    if n_success == 0
        @warn "All rollout scenarios failed at iter $iter"
        return nothing
    end
    objective = total / n_success
    evaluation.last_objective_no_deficit = total_no_deficit / n_success
    evaluation.last_violation_share = _target_violation_share(
        objective, evaluation.last_objective_no_deficit
    )
    println(
        "tag: metrics/rollout_objective_no_deficit, Iter: $iter, Loss: $(evaluation.last_objective_no_deficit)",
    )
    println(
        "tag: metrics/rollout_target_violation_share, Iter: $iter, Loss: $(evaluation.last_violation_share)",
    )
    return nothing
end

"""
    var_set_name!(src::JuMP.VariableRef, dest::JuMP.VariableRef, t::Int) -> Nothing

Name `dest` after `src` with a `#t` stage suffix. If `src` has a JuMP name, the
result is `"<name>#<t>"`; otherwise the MOI variable index is used as fallback,
producing `"_[<index>]#<t>"`.
"""
function var_set_name!(src::JuMP.VariableRef, dest::JuMP.VariableRef, t::Int)
    name = JuMP.name(src)
    if !isempty(name)
        # append node index to original variable name
        JuMP.set_name(dest, string(name, "#", t))
    else
        # append node index to original variable index
        var_name = string("_[", index(src).value, "]")
        JuMP.set_name(dest, string(var_name, "#", t))
    end
end

"""
    add_child_model_vars!(model, subproblem, t, state_params_in, state_params_out,
                          initial_state, var_src_to_dest) -> Dict{VariableRef,VariableRef}

Copy decision variables from stage-`t` `subproblem` into the deterministic-
equivalent `model`, populating the source-to-destination mapping
`var_src_to_dest`. State-coupling variables (incoming parameters and outgoing
realized-state/target pairs) are handled specially: at `t == 1` they become
fresh parameters in `model`; at `t > 1` incoming state parameters are linked to
the previous stage's realized state variables. Each copied variable is renamed
via [`var_set_name!`](@ref) with a `#t` suffix. Mutates `state_params_in`,
`state_params_out`, and `var_src_to_dest` in place; returns `var_src_to_dest`.
"""
function add_child_model_vars!(
    model::JuMP.Model,
    subproblem::JuMP.Model,
    t::Int,
    state_params_in::Vector{Vector{Any}},
    state_params_out::Vector{Vector{Tuple{Any,VariableRef}}},
    initial_state::Vector{Float64},
    var_src_to_dest::Dict{VariableRef,VariableRef},
)
    allvars = all_variables(subproblem)
    allvars = setdiff(allvars, state_params_in[t])
    if state_params_out[t][1][1] isa VariableRef # not MadNLP
        allvars = setdiff(allvars, [x[1] for x in state_params_out[t]])
    end
    allvars = setdiff(allvars, [x[2] for x in state_params_out[t]])
    x = @variable(model, [1:length(allvars)])
    for (src, dest) in zip(allvars, x)
        var_src_to_dest[src] = dest
        var_set_name!(src, dest, t)
    end

    for (i, src) in enumerate(state_params_out[t])
        dest_var = @variable(model)
        var_src_to_dest[src[2]] = dest_var
        var_set_name!(src[2], dest_var, t)

        if state_params_out[t][1][1] isa VariableRef
            dest_param = @variable(model)
            var_src_to_dest[src[1]] = dest_param
            var_set_name!(src[1], dest_param, t)
            state_params_out[t][i] = (dest_param, dest_var)
        else
            state_params_out[t][i] = (state_params_out[t][i][1], dest_var)
        end
    end
    if t == 1
        for (i, src) in enumerate(state_params_in[t])
            if src isa VariableRef
                dest = @variable(model)
                var_src_to_dest[src] = dest
                var_set_name!(src, dest, t)
                state_params_in[t][i] = dest
            end
        end
    else
        for (i, src) in enumerate(state_params_in[t])
            if src isa VariableRef
                var_src_to_dest[src] = state_params_out[t - 1][i][2]
            end
            state_params_in[t][i] = state_params_out[t - 1][i][2]
            # delete parameter constraint associated with src
            if src isa VariableRef
                for con in
                    JuMP.all_constraints(subproblem, VariableRef, MOI.Parameter{Float64})
                    obj = JuMP.constraint_object(con)
                    if obj.func == src
                        JuMP.delete(subproblem, con)
                    end
                end
            end
        end
    end
    return var_src_to_dest
end

"""
    copy_and_replace_variables(src, map::Dict{VariableRef,VariableRef})

Deep-copy a JuMP expression `src`, substituting every `VariableRef` key in
`map` with its destination value. Dispatches on the concrete expression type:

- `Vector`: element-wise recursive call.
- `Real`: returned as-is (no variables to replace).
- `VariableRef`: direct lookup in `map`.
- `GenericAffExpr`: rebuild with remapped variable keys and same coefficients.
- `GenericQuadExpr`: rebuild affine part and remapped `UnorderedPair` keys.
- `GenericNonlinearExpr`: recursively remap arguments, then reconstruct via
  `@expression`.

Throws an error for unrecognized expression types.
"""
function copy_and_replace_variables(
    src::Vector, map::Dict{JuMP.VariableRef,JuMP.VariableRef}
)
    return copy_and_replace_variables.(src, Ref(map))
end

function copy_and_replace_variables(src::Real, ::Dict{JuMP.VariableRef,JuMP.VariableRef})
    return src
end

function copy_and_replace_variables(
    src::JuMP.VariableRef, src_to_dest_variable::Dict{JuMP.VariableRef,JuMP.VariableRef}
)
    return src_to_dest_variable[src]
end

function copy_and_replace_variables(
    src::JuMP.GenericAffExpr, src_to_dest_variable::Dict{JuMP.VariableRef,JuMP.VariableRef}
)
    return JuMP.GenericAffExpr(
        src.constant,
        Pair{VariableRef,Float64}[
            src_to_dest_variable[key] => val for (key, val) in src.terms
        ],
    )
end

function copy_and_replace_variables(
    src::JuMP.GenericQuadExpr, src_to_dest_variable::Dict{JuMP.VariableRef,JuMP.VariableRef}
)
    return JuMP.GenericQuadExpr(
        copy_and_replace_variables(src.aff, src_to_dest_variable),
        Pair{UnorderedPair{VariableRef},Float64}[
            UnorderedPair{VariableRef}(
                src_to_dest_variable[pair.a], src_to_dest_variable[pair.b]
            ) => coef for (pair, coef) in src.terms
        ],
    )
end

function copy_and_replace_variables(
    src::JuMP.GenericNonlinearExpr{V},
    src_to_dest_variable::Dict{JuMP.VariableRef,JuMP.VariableRef},
) where {V}
    num_args = length(src.args)
    args = Vector{Any}(undef, num_args)
    for i in 1:num_args
        args[i] = copy_and_replace_variables(src.args[i], src_to_dest_variable)
    end

    return @expression(owner_model(first(src_to_dest_variable)[2]), eval(src.head)(args...))
end

function copy_and_replace_variables(src::Any, ::Dict{JuMP.VariableRef,JuMP.VariableRef})
    return error(
        "`copy_and_replace_variables` is not implemented for functions like `$(src)`."
    )
end

"""
    create_constraint(model, obj, var_src_to_dest) -> ConstraintRef

Add a constraint to `model` whose function is `obj.func` with all `VariableRef`
keys replaced via `var_src_to_dest` (see [`copy_and_replace_variables`](@ref)).

Four methods handle different constraint types:
- Generic `ScalarConstraint`: uses `@constraint(model, new_func in obj.set)`.
- `ScalarConstraint{NonlinearExpr, MOI.EqualTo}`: `new_func == obj.set.value`.
- `ScalarConstraint{NonlinearExpr, MOI.LessThan}`: `new_func <= obj.set.upper`.
- `ScalarConstraint{NonlinearExpr, MOI.GreaterThan}`: `new_func >= obj.set.lower`.
"""
function create_constraint(model, obj, var_src_to_dest)
    new_func = copy_and_replace_variables(obj.func, var_src_to_dest)
    return @constraint(model, new_func in obj.set)
end

function create_constraint(
    model, obj::ScalarConstraint{NonlinearExpr,MOI.EqualTo{Float64}}, var_src_to_dest
)
    new_func = copy_and_replace_variables(obj.func, var_src_to_dest)
    return @constraint(model, new_func == obj.set.value)
end

function create_constraint(
    model, obj::ScalarConstraint{NonlinearExpr,MOI.LessThan{Float64}}, var_src_to_dest
)
    new_func = copy_and_replace_variables(obj.func, var_src_to_dest)
    return @constraint(model, new_func <= obj.set.upper)
end

function create_constraint(
    model, obj::ScalarConstraint{NonlinearExpr,MOI.GreaterThan{Float64}}, var_src_to_dest
)
    new_func = copy_and_replace_variables(obj.func, var_src_to_dest)
    return @constraint(model, new_func >= obj.set.lower)
end

"""
    add_child_model_exps!(model, subproblem, var_src_to_dest, state_params_out,
                          state_params_in, t) -> Dict

Copy all constraints and the objective contribution from stage-`t` `subproblem`
into the deterministic-equivalent `model`, remapping variables through
`var_src_to_dest` via [`create_constraint`](@ref) and
[`copy_and_replace_variables`](@ref). Constraint-based state parameters and
input parameters at `t == 1` are updated to point to the new model's
constraint refs. The subproblem objective is added to `model`'s existing
objective. Returns a `Dict` mapping source `ConstraintRef` to destination
`ConstraintRef`.
"""
function add_child_model_exps!(
    model::JuMP.Model,
    subproblem::JuMP.Model,
    var_src_to_dest::Dict{VariableRef,VariableRef},
    state_params_out,
    state_params_in,
    t,
)
    # Add constraints:
    # for (F, S) in JuMP.list_of_constraint_types(subproblem)
    cons_to_cons = Dict()
    for con in JuMP.all_constraints(subproblem; include_variable_in_set_constraints=true) #, F, S)
        obj = JuMP.constraint_object(con)
        c = create_constraint(model, obj, var_src_to_dest)
        cons_to_cons[con] = c
        if (state_params_out[t][1][1] isa ConstraintRef)
            for (i, _con) in enumerate(state_params_out[t])
                if con == _con[1]
                    state_params_out[t][i] = (c, state_params_out[t][i][2])
                    continue;
                end
            end
        end
        if (t==1) && (state_params_in[t][1] isa ConstraintRef)
            for (i, _con) in enumerate(state_params_in[t])
                if con == _con
                    state_params_in[t][i] = c
                    continue;
                end
            end
        end
    end
    # end
    # Add objective:
    current = JuMP.objective_function(model)
    subproblem_objective = copy_and_replace_variables(
        JuMP.objective_function(subproblem), var_src_to_dest
    )
    JuMP.set_objective_function(model, current + subproblem_objective)
    return cons_to_cons
end

"""
    deterministic_equivalent!(model, subproblems, state_params_in, state_params_out,
                              initial_state, uncertainties)

Build the deterministic-equivalent (direct transcription) JuMP model by copying all
stage subproblems into `model`.  Variables are renamed with a `#t` suffix to avoid
conflicts.  Stage coupling is enforced by identifying the realized state variable of
stage `t` with the incoming state parameter of stage `t+1`.

`uncertainties` accepts both sampling formats (see [`sample`](@ref)):

- **Per-unit pools**: `Vector{Vector{Tuple{VariableRef, Vector{T}}}}` — one pool per
  parameter, drawing independently per parameter.
- **Joint-scenario pools**: `Vector{Vector{Vector{Tuple{VariableRef, T}}}}` — pre-built
  joint scenarios preserving cross-parameter correlations.

Returns `(model, uncertainties_new)` where `uncertainties_new` has the same format as
the input but with variable refs remapped to the deterministic-equivalent model.
"""
function deterministic_equivalent!(
    model::JuMP.Model,
    subproblems::Vector{JuMP.Model},
    state_params_in::Vector{Vector{Any}},
    state_params_out::Vector{Vector{Tuple{Any,VariableRef}}},
    initial_state::Vector{Float64},
    uncertainties,
)
    set_objective_sense(model, objective_sense(subproblems[1]))
    var_src_to_dest = Dict{VariableRef,VariableRef}()
    for t in 1:length(subproblems)
        DecisionRules.add_child_model_vars!(
            model,
            subproblems[t],
            t,
            state_params_in,
            state_params_out,
            initial_state,
            var_src_to_dest,
        )
    end

    cons_to_cons = Vector{Dict}(undef, length(subproblems))
    for t in 1:length(subproblems)
        cons_to_cons[t] = DecisionRules.add_child_model_exps!(
            model, subproblems[t], var_src_to_dest, state_params_out, state_params_in, t
        )
    end

    uncertainties_new = _remap_uncertainties(uncertainties, var_src_to_dest, cons_to_cons)
    return model, uncertainties_new
end

"""
    _remap_uncertainties(uncertainties, var_src_to_dest, cons_to_cons)

Replace source-model `VariableRef` keys in an uncertainty pool with their
destination-model counterparts (using the variable or constraint mapping built
by [`deterministic_equivalent!`](@ref)).

Two methods dispatch on the pool format:

- **Per-unit pools** (`Vector{Vector{Tuple{VariableRef, Vector{T}}}}`):
  each stage maps `[(param₁, [v₁, …]), …]` independently.
- **Joint-scenario pools** (`Vector{Vector{Vector{Tuple{VariableRef, T}}}}`):
  each stage maps `[[scenario₁…], [scenario₂…], …]` preserving the grouped
  structure.

This is an internal helper; users interact with it indirectly through
[`deterministic_equivalent!`](@ref).
"""
function _remap_uncertainties(
    uncertainties::Vector{Vector{Tuple{VariableRef,Vector{T}}}},
    var_src_to_dest, cons_to_cons,
) where {T<:Real}
    remap = if uncertainties[1][1][1] isa VariableRef
        ky -> var_src_to_dest[ky]
    else
        ky -> cons_to_cons[1][ky]
    end
    return [
        [(remap(ky), val) for (ky, val) in uncertainties[t]]
        for t in eachindex(uncertainties)
    ]
end

function _remap_uncertainties(
    uncertainties::Vector{Vector{Vector{Tuple{VariableRef,T}}}},
    var_src_to_dest, cons_to_cons,
) where {T<:Real}
    remap = ky -> haskey(var_src_to_dest, ky) ? var_src_to_dest[ky] : cons_to_cons[1][ky]
    return [
        [[(remap(ky), val) for (ky, val) in scenario] for scenario in uncertainties[t]]
        for t in eachindex(uncertainties)
    ]
end

"""
    find_variables(model::JuMP.Model, variable_name_parts::Vector{<:AbstractString})

Return variables from `model` whose JuMP name contains **all** substrings in
`variable_name_parts`. When the initial filter yields more than one variable,
results are reordered by matching `"<first_part>[i]"` for `i = 1, 2, ...` to
produce a consistently indexed vector.
"""
function find_variables(model::JuMP.Model, variable_name_parts::Vector{S}) where {S}
    all_vars = all_variables(model)
    interest_vars = all_vars[findall(
        x -> all([occursin(part, JuMP.name(x)) for part in variable_name_parts]), all_vars
    )]
    if length(interest_vars) == 1
        return interest_vars
    end
    return [
        interest_vars[findfirst(
            x -> occursin(variable_name_parts[1] * "[$i]", JuMP.name(x)), interest_vars
        )] for i in 1:length(interest_vars)
    ]
end
