# Gradient Fallback

```@meta
CurrentModule = DecisionRules
```

## Motivation

TS-DDR training relies on solving an optimization subproblem at every stage and
differentiating through it (via Lagrange duals or DiffOpt). In practice, some
solves may fail — the solver hits numerical trouble, DiffOpt encounters
degenerate duals, or the subproblem is infeasible for a particular sample. A
single uncaught error kills the entire training run.

The **gradient fallback** system provides a principled, extensible way to handle
these errors at three levels:

| Level | Where it fires | What it controls |
|-------|----------------|------------------|
| **rrule pullback** | Inside the ChainRules `rrule` for `get_next_state` | Whether a bad-solver-status pullback returns zeros or throws |
| **Training loop** | Around `Flux.gradient(...)` in `train_multistage` / `train_multiple_shooting` | Whether a DiffOpt error skips the iteration or crashes |
| **Rollout evaluation** | Inside [`RolloutEvaluation`](@ref) per scenario | Whether a failed scenario is excluded from the metric or crashes |

## Built-in fallback types

```@docs
AbstractGradientFallback
ZeroGradientFallback
ErrorGradientFallback
```

## Usage

### Default behavior (zero gradients)

By default, all training functions use [`ZeroGradientFallback`](@ref). Failed
iterations log a warning and skip the parameter update:

```julia
train_multistage(
    model, x0, subproblems, spi, spo, uncertainty;
    num_batches=500,
    # gradient_fallback=ZeroGradientFallback()  # this is the default
)
```

At training start you will see:

```
[ Info: Training with ZeroGradientFallback: solver/differentiation errors
will be caught and the iteration skipped (zero gradient). Pass
`gradient_fallback=ErrorGradientFallback()` to throw instead, or implement
a custom `AbstractGradientFallback` subtype.
```

### Strict mode (for tests)

Use [`ErrorGradientFallback`](@ref) when you want errors to surface
immediately — typically in unit tests where every solve should succeed:

```julia
train_multistage(
    model, x0, subproblems, spi, spo, uncertainty;
    num_batches=10,
    gradient_fallback=ErrorGradientFallback(),
)
```

The same keyword works for [`train_multiple_shooting`](@ref) and
[`RolloutEvaluation`](@ref):

```julia
rollout = RolloutEvaluation(
    subproblems, spi, spo, x0, scenarios;
    gradient_fallback=ErrorGradientFallback(),
)
```

## Custom fallbacks (extending the type system)

Subtype [`AbstractGradientFallback`](@ref) and implement three methods:

```julia
struct LoggingFallback <: DecisionRules.AbstractGradientFallback
    log::Vector{Any}
end

# Called when the rrule pullback (DiffOpt / dual extraction) fails.
# Return a tuple of cotangents matching the rrule signature, or rethrow.
function DecisionRules.handle_gradient_error(fb::LoggingFallback, e, n_in, n_out)
    push!(fb.log, (:gradient, e))
    return DecisionRules._zero_cotangents(n_in, n_out)
end

# Called when Flux.gradient(...) throws in the training loop.
# Return `true` to skip this iteration, or rethrow.
function DecisionRules.handle_training_error(fb::LoggingFallback, e, iter)
    push!(fb.log, (:training, iter, e))
    return true
end

# Called when a rollout scenario fails during evaluation.
# Return `true` to exclude this scenario from the metric, or rethrow.
function DecisionRules.handle_rollout_error(fb::LoggingFallback, e, iter)
    push!(fb.log, (:rollout, iter, e))
    return true
end
```

Then pass it to any training function:

```julia
fb = LoggingFallback(Any[])
train_multistage(model, x0, subs, spi, spo, unc;
    gradient_fallback=fb,
)
println("Caught $(length(fb.log)) errors during training")
```

This is useful for:
- **Monitoring**: count how often solves fail and on which iterations
- **Adaptive recovery**: adjust solver tolerances, restart from a checkpoint, etc.
- **Selective rethrowing**: catch known benign errors but let unexpected ones through

## Relationship to `STRICT_GRADIENTS`

The global [`STRICT_GRADIENTS`](@ref DecisionRules.STRICT_GRADIENTS) flag controls a separate, lower-level mechanism:
inside the `rrule` pullback, when the forward solver terminates with a
non-optimal status (e.g., `ITERATION_LIMIT`), the pullback returns zero
gradients (if `STRICT_GRADIENTS[] == false`, the default) or throws (if `true`).

The `gradient_fallback` keyword operates at a higher level — it catches errors
from DiffOpt's `reverse_differentiate!` (assertion errors, degenerate duals,
etc.) and from the training loop itself. Both mechanisms are independent and
complementary:

```
Forward solve
  └─ bad termination status → STRICT_GRADIENTS controls behavior
  └─ good status → DiffOpt reverse_differentiate!
       └─ error (assertion, numerical) → gradient_fallback catches it
            └─ in rrule pullback: handle_gradient_error
            └─ in training loop: handle_training_error
            └─ in rollout eval: handle_rollout_error
```
