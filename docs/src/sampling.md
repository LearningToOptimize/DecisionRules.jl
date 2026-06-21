# Uncertainty Sampling

```@meta
CurrentModule = DecisionRules
```

## Why sampling matters in TS-DDR

In the TS-DDR training loop, each SGD step approximates the stochastic objective

```math
\min_\theta \; \mathbb{E}_{w_{1:T}} \left[
    \sum_{t=1}^{T} q_t\bigl(x_{t-1}, w_t;\, \hat{x}_t(\theta)\bigr)
\right]
```

by drawing **sample trajectories** ``w_{1:T}^{(s)},\; s = 1,\ldots,S`` and
differentiating through the subproblem solves. The `uncertainty_sampler`
argument in [`train_multistage`](@ref) and [`train_multiple_shooting`](@ref)
controls how these trajectories are generated.

Once a trajectory ``w_{1:T}`` is realized — a concrete numeric value per
uncertain parameter per stage — the rest of the training pipeline
(policy rollout, subproblem solve, gradient computation) is **identical**
regardless of how the trajectory was sampled. The sampler is therefore a
pluggable component that lets you match the correlation structure of your
problem.

## Three sampling formats

DecisionRules.jl supports three ways to specify uncertainty, offering
increasing levels of correlation control.

### 1. Independent (per-unit) pools

Each uncertain parameter has its own finite support and is sampled
**independently** at each stage.

```
                    ┌─ param₁: draw from {v₁₁, v₁₂, v₁₃}  ←── independent
Stage t ────────────┤
                    └─ param₂: draw from {v₂₁, v₂₂}        ←── independent
```

**Julia type**: `Vector{Vector{Tuple{VariableRef, Vector{T}}}}`

```julia
# uncertainty_pool[t][i] = (param_ref, [possible_values...])
independent_pool = [
    # stage 1
    [(demand_param_1, [10.0, 15.0, 12.0]),
     (demand_param_2, [8.0, 12.0, 9.0])],
    # stage 2
    [(demand_param_1, [11.0, 14.0, 13.0]),
     (demand_param_2, [7.0, 11.0, 10.0])],
]
```

Each call to `sample(independent_pool)` draws one value per parameter per
stage, independently. With ``n`` parameters each having ``k`` scenarios, this
samples from ``k^n`` possible combinations per stage — most of which may
never have occurred in reality.

**Use when**: parameters are genuinely independent, or you have a single
uncertain parameter per stage.

### 2. Joint-scenario pools (spatial correlation)

Pre-defined joint realizations across **all** parameters. Sampling picks one
complete scenario per stage, preserving cross-parameter (spatial) correlations.

```
                         ω=1: (param₁=v₁₁, param₂=v₂₁)
                        ╱
Stage t ── draw one ω ─── ω=2: (param₁=v₁₂, param₂=v₂₂)
                        ╲
                         ω=3: (param₁=v₁₃, param₂=v₂₃)
```

**Julia type**: `Vector{Vector{Vector{Tuple{VariableRef, T}}}}`

```julia
# uncertainty_pool[t][ω] = [(param₁, val), (param₂, val), ...]
joint_pool = [
    # stage 1: 3 scenarios
    [[(inflow_1, 10.0), (inflow_2, 80.0)],   # ω=1
     [(inflow_1, 20.0), (inflow_2, 120.0)],   # ω=2
     [(inflow_1, 15.0), (inflow_2, 90.0)]],   # ω=3
    # stage 2: 3 scenarios
    [[(inflow_1, 11.0), (inflow_2, 70.0)],
     [(inflow_1, 14.0), (inflow_2, 110.0)],
     [(inflow_1, 13.0), (inflow_2, 100.0)]],
]
```

Each call to `sample(joint_pool)` picks one scenario index ``\omega``
per stage and returns all parameters from that scenario. Only historically
observed combinations appear.

!!! warning "Matching SDDP semantics"
    SDDP.jl's `SDDP.parameterize` draws one ``\omega`` for all random
    variables in a stage. If you compare TS-DDR against SDDP, you **must**
    use joint-scenario pools to avoid a distributional mismatch.

**Use when**: parameters are correlated (e.g., inflows across a river basin),
or your benchmark uses joint scenarios (SDDP, scenario trees).

### 3. Trajectory sampler (spatial + temporal correlation)

A callable that generates each stage's realization **conditioned on previous
stages**. This enables autoregressive (AR), Markovian, or any custom temporal
dependence — something the data-pool formats above cannot express because
they sample stages independently.

```
Stage 1 ── sampler(1, []) ──────────────────── w₁
                                                │
Stage 2 ── sampler(2, [w₁]) ───────────────── w₂
                                                │
Stage 3 ── sampler(3, [w₁, w₂]) ──────────── w₃
```

**Julia type**: `Function` with signature `(t::Int, past::Vector{...}) -> Vector{Tuple{VariableRef, T}}`

```julia
# AR(1) inflow sampler with spatial correlation
function ar1_sampler(t, past)
    if isempty(past)
        # Stage 1: draw from marginal distribution
        ω = rand(1:nScenarios)
        return [(params[t][r], data[r][t, ω]) for r in 1:nHyd]
    else
        # Stage t > 1: AR(1) conditioned on previous stage
        prev = [pair[2] for pair in past[end]]
        noise = randn(nHyd) .* σ
        vals = ρ .* prev .+ (1 .- ρ) .* μ .+ noise
        return [(params[t][r], vals[r]) for r in 1:nHyd]
    end
end

# Generate one trajectory
trajectory = sample(ar1_sampler, T)

# Use in training — wrap as zero-arg callable
train_multistage(
    policy, x0, subproblems,
    state_in, state_out,
    () -> sample(ar1_sampler, T);  # pass as callable
    num_batches=500,
)
```

**Use when**: your uncertainty process has temporal dependence (e.g.,
autoregressive inflows, mean-reverting prices, regime-switching demands).

## Comparison table

| Feature | Independent | Joint-scenario | Trajectory sampler |
|:--------|:-----------|:--------------|:-------------------|
| Spatial correlation | ✗ | ✓ | ✓ |
| Temporal correlation | ✗ | ✗ | ✓ |
| Data format | Finite supports | Pre-built scenarios | Callable |
| Combinations per stage | ``k^n`` | ``k`` | unlimited |
| SDDP-compatible | only if ``n=1`` | ✓ | depends on model |
| Ease of use | simplest | moderate | most flexible |

## Building uncertainty pools

### From historical data (joint scenarios — recommended)

When your data comes as a matrix where columns are scenarios:

```julia
# data[r] is a (T × nScenarios) matrix for reservoir r
nHyd = length(data)
nCen = size(data[1], 2)

uncertainty_pool = Vector{Any}(undef, T)
for t in 1:T
    uncertainty_pool[t] = [
        [(inflow_params[t][r], data[r][t, ω] + 0.0) for r in 1:nHyd]
        for ω in 1:nCen      # ω is the OUTER loop — all units share it
    ]
end
```

### From independent distributions

```julia
uncertainty_pool = [
    [(demand_param[t], [low, mid, high])]
    for t in 1:T
]
```

### From an AR(1) process

```julia
# Estimate AR(1) parameters from data
μ = mean.(eachrow(hcat(data...)))    # long-run mean per unit
σ = std.(eachrow(hcat(data...)))     # innovation std per unit
ρ = 0.7                              # autocorrelation coefficient

function ar1_sampler(t, past)
    if isempty(past)
        vals = μ .+ σ .* randn(nHyd)
    else
        prev = [p[2] for p in past[end]]
        vals = ρ .* prev .+ (1 .- ρ) .* μ .+ σ .* randn(nHyd)
    end
    return [(inflow_params[t][r], max(0.0, vals[r])) for r in 1:nHyd]
end
```

## Sampling in practice

All three formats produce the same **realized trajectory** type:
`Vector{Vector{Tuple{VariableRef, Float64}}}`. This is what gets passed to
`simulate_multistage`, `simulate_stage`, and all internal training code.

```julia
using DecisionRules

# 1. From a data pool (independent or joint):
trajectory = sample(uncertainty_pool)

# 2. From a trajectory sampler:
trajectory = sample(ar1_sampler, T)

# Both produce the same type — downstream code is identical:
objective = simulate_multistage(
    subproblems, state_params_in, state_params_out,
    initial_state, trajectory, policy,
)
```

### Passing to training functions

```julia
# Data pools are passed directly:
train_multistage(policy, x0, subs, s_in, s_out, uncertainty_pool; ...)

# Trajectory samplers are wrapped as zero-arg callables:
train_multistage(policy, x0, subs, s_in, s_out,
    () -> sample(ar1_sampler, T); ...)
```

This works because `train_multistage` calls `sample(uncertainty_sampler)` to
draw each trajectory. For data pools, `sample` dispatches on the pool type.
For callables, `sample(f::Function)` simply calls `f()`.

## Demonstrating the difference

Consider 3 hydro reservoirs with 4 historical inflow scenarios:

```
Historical inflow data (columns = scenarios):

         ω=1    ω=2    ω=3    ω=4
Res 1:   10     20     15     25
Res 2:   80    120     90    110
Res 3:    5      8      6      9
```

**Independent sampling** draws one value per row independently. A sample
might be `(10, 120, 9)` — reservoir 1 from ω=1, reservoir 2 from ω=2,
reservoir 3 from ω=4. This combination never occurred historically and
may violate the drought-affects-all-basins correlation.

**Joint sampling** picks one column: `(10, 80, 5)` or `(25, 110, 9)` —
always a historically observed combination.

**Trajectory sampling** can additionally model temporal persistence:
if ω=1 (dry year) was drawn at stage 1, the AR(1) sampler will likely
produce below-average inflows at stage 2 as well.

```
Joint sampling (k=4 possible outcomes per stage):

    Res 1 ──┐
    Res 2 ──┼── same ω ──→ one of 4 historical vectors
    Res 3 ──┘

Independent sampling (k³=64 possible outcomes per stage):

    Res 1 ── ω₁ ──┐
    Res 2 ── ω₂ ──┼──→ one of 64 combinations (most never observed)
    Res 3 ── ω₃ ──┘

Trajectory sampling (conditioned on past):

    Stage 1: same as joint ──→ w₁
                                │
    Stage 2: AR(1)(w₁) ──────→ w₂  (temporal correlation preserved)
```

## Internal functions

The following internal helpers process uncertainty pools in different training
formulations. They are not part of the public API but documented here for
maintainability.

- [`_remap_uncertainties`](@ref): Remap JuMP `VariableRef` keys when
  copying uncertainty pools into a deterministic-equivalent model.
  Two methods dispatch on per-unit vs. joint-scenario pool types.
- [`extract_uncertainty_params`](@ref): Extract just the `VariableRef`
  parameters from an uncertainty pool, discarding the scenario values.
  Used by `setup_shooting_windows` for multiple-shooting training.

## API Reference

```@docs
sample
```
