# GPU Acceleration with DecisionRulesExa.jl

```@meta
CurrentModule = DecisionRules
```

[DecisionRulesExa.jl](https://github.com/LearningToOptimize/DecisionRulesExa.jl) is a
companion package that implements the same TS-DDR algorithm using
[ExaModels.jl](https://github.com/exanauts/ExaModels.jl) instead of JuMP for the
optimization backend. It targets GPU-accelerated training via
[MadNLP.jl](https://github.com/MadNLP/MadNLP.jl) with CUDSS-backed interior-point
solves.

## When to use DecisionRulesExa.jl

| | DecisionRules.jl (JuMP) | DecisionRulesExa.jl (ExaModels) |
|:---|:---|:---|
| **Backend** | JuMP + DiffOpt | ExaModels + MadNLP |
| **Hardware** | CPU | CPU or GPU (CUDA) |
| **Training modes** | DE, stage-wise, multiple shooting | Deterministic equivalent |
| **Gradient source** | DiffOpt implicit diff + duals | Envelope theorem (duals only) |
| **Best for** | Moderate NLPs, integer variables, stage-wise decomposition | Large NLPs (AC-OPF), GPU speedup, many samples per batch |

**Choose DecisionRulesExa.jl when** the inner NLP is large enough that GPU
acceleration matters (e.g., AC-OPF with hundreds of buses and thousands of
variables per stage) and you want to run many training samples per gradient
step on a single GPU.

**Choose DecisionRules.jl when** you need stage-wise or multiple-shooting
decomposition, integer variable support, or DiffOpt-based solution sensitivities.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/LearningToOptimize/DecisionRulesExa.jl.git")
```

For GPU support, also install CUDA.jl and MadNLPGPU:

```julia
Pkg.add(["CUDA", "MadNLPGPU"])
```

## Quick start: CPU

The simplest way to get started is with the built-in linear tracking problem:

```julia
using DecisionRulesExa
using ExaModels, Flux, MadNLP, Random

Random.seed!(1)

T  = 8   # horizon
nx = 1   # state dimension

# Build a parametric deterministic-equivalent NLP on CPU
prob = build_linear_tracking_problem(
    horizon       = T,
    nx            = nx,
    backend       = nothing,       # CPU
    slack_penalty = 10.0,
    u_bounds      = (-2.0, 2.0),
)

# LSTM policy: maps [w_t ; x_{t-1}] → target x̂_t at each stage
policy = StateConditionedPolicy(nx, nx, nx, [64, 64])

# Uncertainty sampler: returns a flat vector of length T * nw
sampler() = Float32.(0.1 .* randn(T * nx))

# Train with TS-DDR policy gradient (envelope theorem)
train_tsddr(
    policy,
    Float32.([1.0]),               # initial state
    prob,
    prob.p_x0,
    prob.p_target,
    prob.p_w,
    sampler;
    num_batches         = 100,
    num_train_per_batch = 4,
    optimizer           = Flux.Adam(1f-3),
    madnlp_kwargs       = (print_level = MadNLP.ERROR, tol = 1e-6),
)
```

## Moving to GPU

To run the same problem on GPU, change the backend and add a GPU-native
linear solver:

```julia
using CUDA, MadNLPGPU

prob_gpu = build_linear_tracking_problem(
    horizon       = T,
    nx            = nx,
    backend       = CUDABackend(),
    slack_penalty = 10.0,
    u_bounds      = (-2.0, 2.0),
)

train_tsddr(
    policy,
    Float32.([1.0]),
    prob_gpu,
    prob_gpu.p_x0,
    prob_gpu.p_target,
    prob_gpu.p_w,
    sampler;
    num_batches         = 100,
    num_train_per_batch = 4,
    optimizer           = Flux.Adam(1f-3),
    madnlp_kwargs       = (
        print_level   = MadNLP.ERROR,
        tol           = 1e-6,
        linear_solver = CUDSSSolver,
    ),
)
```

The policy (Flux model) stays on CPU; only the NLP solve runs on GPU.
Parameter updates (`ExaModels.set_parameter!`) and multiplier extraction
handle CPU↔GPU transfers automatically.

## Custom problems

For domain-specific models (power systems, robotics, etc.), build the
ExaModels NLP directly instead of using `build_linear_tracking_problem`.
The key requirements are:

1. **Add target constraints last** so their multipliers form a contiguous
   slice of `result.multipliers`.
2. **Parameterize** the initial state (`p_x0`), uncertainty trajectory
   (`p_w`), and target trajectory (`p_target`) as ExaModels parameters.
3. **Return** a struct with fields `.core`, `.model`, `.horizon`, and
   `.target_con_range`.

The `HydroPowerModels` example in DecisionRulesExa.jl demonstrates this
pattern for a full AC-OPF problem with reservoir dynamics:

```julia
# In examples/HydroPowerModels/hydro_power_exa.jl
prob = build_hydro_de(
    data;
    num_stages     = 96,
    backend        = CUDABackend(),
    formulation    = :ac_polar,
    deficit_cost   = 1e5,
    target_penalty = :auto,
)
```

## Parallel GPU solves

When training samples are independent, multiple NLP instances can be
solved concurrently on the same GPU. Build a pool of independent problem
copies and pass it to `train_tsddr`:

```julia
pool = [(prob, prob.p_x0, prob.p_target, prob.p_w)]
for _ in 2:num_workers
    p = build_my_problem(backend = CUDABackend())
    push!(pool, (p, p.p_x0, p.p_target, p.p_w))
end

train_tsddr(policy, x0, prob, prob.p_x0, prob.p_target, prob.p_w, sampler;
    problem_pool        = pool,
    num_train_per_batch = num_workers,
)
```

Each pool entry gets its own MadNLP solver instance. Samples are
distributed round-robin across the pool and solved via `Threads.@spawn`.

## Penalty annealing

DecisionRulesExa.jl supports penalty annealing through the
`adjust_hyperparameters` callback. The target penalty coefficient
``\rho`` is stored as an ExaModels parameter and can be updated at
runtime:

```julia
adjust_hyperparameters = function(iter, opt_state, num_train)
    phase = iter < 100 ? 0.1 :
            iter < 200 ? 1.0 :
            iter < 300 ? 10.0 : 30.0
    ρ = base_penalty * phase
    penalty_vals = fill(ρ / 2, T * nx)
    ExaModels.set_parameter!(prob.core, prob.p_penalty_half, penalty_vals)
    return num_train
end
```

This mirrors the `penalty_schedule` keyword in DecisionRules.jl's
[`train_multistage`](@ref).

## Rollout evaluation

[`RolloutEvaluation`](@ref) in DecisionRules.jl evaluates policies
stage-by-stage under deployment semantics. DecisionRulesExa.jl provides
an analogous `RolloutEvaluation` that solves stage subproblems
sequentially:

```julia
eval = RolloutEvaluation(
    stage_problem, x0, eval_scenarios;
    horizon              = T,
    n_uncertainty        = nw,
    set_stage_parameters! = my_stage_setter!,
    realized_state       = my_realized_state,
    stride               = 25,
    policy_state         = :realized,
)
```

Both packages report the same metrics: operational cost excluding
target-deficit penalty, and target-violation share.

## Mapping between packages

| DecisionRules.jl | DecisionRulesExa.jl | Notes |
|:---|:---|:---|
| `train_multistage` | `train_tsddr` | Main training loop |
| `state_conditioned_policy` | `StateConditionedPolicy` | LSTM policy |
| `dense_multilayer_nn` | `MLPPolicy` | MLP policy |
| `state_params_in` | `p_x0` | Initial state parameter |
| `state_params_out` | `p_target` | Target parameter |
| `uncertainty_samples` | `p_w` + sampler | Uncertainty parameter |
| `SampleLog` / `record` | `record_loss` | Per-iteration callback |
| `RolloutEvaluation` | `RolloutEvaluation` | Stage-wise eval |
| `penalty_schedule` | `adjust_hyperparameters` | Penalty annealing |
| `ScoreFunctionConfig` | — | Not yet ported to ExaModels |
| Stage-wise decomposition | — | JuMP only |
| Multiple shooting | — | JuMP only |

## Embedded deterministic equivalent

The standard `DeterministicEquivalentProblem` treats the policy's target
trajectory as an external parameter: the training loop generates
``\hat{x}_{1:T}`` outside the NLP and passes it in via `set_targets!`.
This is **open-loop** — the policy does not see the realized states
from the coupled solve.

`EmbeddedDeterministicEquivalentProblem` embeds the policy *inside*
the NLP via a `VectorNonlinearOracle`.  The NLP constraint becomes:

```math
\pi_\theta(w_t,\, x_{t-1}^*) - x_t - \delta_t = 0 \quad \forall t
```

where ``x_{t-1}^*`` is the solver's realized state.  This is
**closed-loop**: the policy sees realized states from the coupled solve,
and the duals ``\lambda_t`` reflect the joint (policy + physics) system.

```julia
prob = build_embedded_deterministic_equivalent(
    policy;
    horizon       = T,
    nx            = nx,
    nu            = nu,
    nw            = nw,
    dynamics_eq   = my_dynamics,
    stage_cost    = my_cost,
    backend       = CUDABackend(),
)

train_tsddr_embedded(
    policy, x0, prob, sampler;
    num_batches         = 500,
    num_train_per_batch = 4,
    optimizer           = Flux.Adam(1f-3),
    madnlp_kwargs       = (print_level = MadNLP.ERROR, tol = 1e-6),
)
```

The oracle closures capture the policy **by reference** — updating Flux
parameters between solves automatically changes the NLP without
rebuilding it.  Use `invalidate_policy_cache!` if your oracle caches
policy-dependent intermediates.

### Strict embedded targets

When the policy is guaranteed to produce feasible targets (e.g., via a
reachable-set mapping), the slack variables ``\delta_t`` can be removed
entirely:

```julia
prob = build_embedded_hydro_de(policy, power_data, hydro_data, T;
    formulation    = :ac_polar,
    strict_targets = true,
)
```

In strict mode the constraint is simply ``x_t = \pi_\theta(w_t, x_{t-1}^*)``,
the dual ``\lambda_t`` is the pure economic shadow price, and there is no
target penalty to tune.

## Sequential training

`train_tsddr` solves the full deterministic equivalent in one shot.
For problems where this is too large (or where stage-wise gradient
accumulation is preferred), `train_tsddr_sequential!` decomposes the
training into sequential stage solves:

```julia
train_tsddr_sequential!(
    policy, x0, stage_problem, sampler;
    horizon             = T,
    n_uncertainty        = nw,
    set_stage_parameters! = my_setter!,
    realized_state       = my_state_reader,
    num_batches         = 500,
)
```

This mirrors the stage-wise decomposition in DecisionRules.jl but uses
ExaModels + MadNLP per stage.

## Critic control variate

`train_tsddr` optionally trains a scalar critic ``C(w, \hat{x})`` that
provides a learned control variate for the dual gradient signal.  The
critic does not replace the NLP solve — dual multipliers remain the
primary actor gradient.  The critic reduces gradient variance by
subtracting a correlated baseline.

```julia
critic = Chain(Dense(input_dim => 128, tanh), Dense(128 => 128, tanh), Dense(128 => 1))

cv = ScalarCriticControlVariate(critic;
    featurizer          = default_critic_featurizer,
    value_loss_weight   = 1.0,
    gradient_loss_weight = 0.0,
)

critic_target = RolloutCriticTarget(stage_problem;
    horizon            = T,
    n_uncertainty      = nw,
    set_stage_parameters! = my_setter!,
    realized_state     = my_state_reader,
    policy_state       = :target,
)

train_tsddr(policy, x0, prob, prob.p_x0, prob.p_target, prob.p_w, sampler;
    control_variate              = cv,
    critic_training_target       = critic_target,
    actor_gradient_mode          = :control_variate,
    critic_cv_weight             = 1.0,
    critic_optimizer             = Flux.Adam(1f-3),
)
```

Two actor modes are supported:

- `:control_variate` — subtracts ``\nabla_{\hat{x}} C`` from the dual
  signal and adds it back as a differentiable surrogate.  Unbiased when
  the critic is exact; reduces variance otherwise.
- `:surrogate` — blends dual and critic actor gradients via explicit
  weights (`dual_actor_weight`, `critic_actor_weight`).  Useful when raw
  duals are noisy, but no longer strictly unbiased.

## Full example: HydroPowerModels

The `examples/HydroPowerModels/` directory in DecisionRulesExa.jl contains
a complete AC-OPF hydrothermal scheduling example for the Bolivia test case
— the same problem solved by DecisionRules.jl in the
[Hydropower Scheduling](@ref) tutorial. It demonstrates:

- Parsing PowerModels.jl network data and hydro reservoir parameters
- Building a multi-stage deterministic-equivalent NLP in ExaModels
  (DC or AC polar OPF formulations)
- L1 + L2 penalty on target slack (δ⁺/δ⁻ splitting for smooth NLP)
- GPU training with parallel MadNLP solves
- Warm-start caching to prevent cascade solver failures
- Penalty and sample-count annealing schedules
- Embedded closed-loop training with strict reachable targets
- Optional critic control variate with rollout-based value targets
- W&B metric logging
