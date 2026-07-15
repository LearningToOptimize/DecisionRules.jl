# Getting started

```@meta
CurrentModule = DecisionRules
```

This chapter covers installation, the anatomy of a DecisionRules.jl
training run, and how to choose among the training formulations. The
theory behind every step is in Part I; complete worked problems are in
Part III.

## Installation

```julia
using Pkg
Pkg.add("DecisionRules")
```

DecisionRules.jl builds policies with [Flux.jl](https://fluxml.ai) and
subproblems with [JuMP](https://jump.dev); training requires a
DiffOpt-compatible solver for the stage problems (Ipopt for smooth NLPs,
HiGHS for LPs/MIPs). For GPU-accelerated training of large NLPs, install
the companion package
[DecisionRulesExa.jl](https://github.com/LearningToOptimize/DecisionRulesExa.jl)
(see [GPU Acceleration with DecisionRulesExa.jl](@ref)).

## Anatomy of a training run

Every TS-DDR training run assembles the same five ingredients:

1. **Stage subproblems** — one JuMP model per stage, wrapped with
   `DiffOpt.diff_optimizer` so Lagrange duals and sensitivities are
   available. The incoming state, the uncertainty, and the policy's
   target state enter as *parameters*.
2. **A policy** — a Flux model mapping ``[w_t;\, x_{t-1}]`` to a target
   state ``\hat{x}_t`` (see [Target-state policies](@ref)).
3. **An uncertainty sampler** — how trajectories ``w_{1:T}`` are drawn;
   the three supported formats (independent pools, joint-scenario pools,
   trajectory samplers) are the subject of [Uncertainty Sampling](@ref).
4. **A training formulation** — deterministic equivalent, stage-wise,
   multiple shooting, or strict (see
   [Three training formulations](@ref) and
   [Strict mode: penalty-free gradient signal](@ref)).
5. **An evaluation protocol** — out-of-sample stage-wise rollout via
   [`RolloutEvaluation`](@ref), with target- or realized-state feedback
   (see [Evaluation semantics](@ref)).

## Quick start

```julia
using DecisionRules, JuMP, DiffOpt, Flux, Ipopt

# Build per-stage subproblems in JuMP (DiffOpt-enabled)
# subproblems, state_params_in, state_params_out, uncertainty_samples, initial_state = ...

# Define a policy: maps [uncertainty; state] → target state
policy = Chain(
    Dense(policy_input_dim(num_uncertainties, num_states), 64, relu),
    Dense(64, num_states),
)

# Train via stage-wise decomposition
train_multistage(
    policy, initial_state, subproblems,
    state_params_in, state_params_out, uncertainty_samples;
    num_batches=100, optimizer=Flux.Adam(1e-3),
)
```

## Choosing a training formulation

| Formulation | Horizon coupling | Gradient source |
|:---|:---|:---|
| **Deterministic Equivalent** | Full horizon, one large NLP | Duals on the coupled problem |
| **Stage-wise (single shooting)** | Sequential rollout | Duals + DiffOpt per stage |
| **Multiple Shooting** | Windowed sub-horizons | DiffOpt per window, continuity penalties |
| **Strict subproblems** | Sequential rollout, no slack | Pure shadow-price duals |

As a rule of thumb:

- start with **stage-wise** training — closed-loop, smallest solves,
  fewest assumptions;
- move to the **deterministic equivalent** (or its GPU implementation)
  when the per-stage solves are large and horizon-coupled gradient signal
  pays off;
- use **multiple shooting** as the middle ground on long horizons;
- switch to **strict** mode whenever you can construct a
  feasibility-guaranteeing policy (e.g. `HydroReachablePolicy` for
  reservoir systems): it eliminates the target-slack penalty and its
  tuning entirely, and the dual ``\lambda_t`` becomes the pure shadow
  price. This is the configuration used in the flagship
  [Hydropower Scheduling](@ref) case study.

## Robustness and hardware

- Solver or differentiation failures during training are handled by the
  pluggable [gradient fallback](@ref "Gradient Fallback") system —
  by default a failed iteration logs a warning and is skipped.
- Large problems (e.g. multi-stage AC-OPF) train an order of magnitude
  faster on GPU through
  [DecisionRulesExa.jl](@ref "GPU Acceleration with DecisionRulesExa.jl"),
  which implements the strict deterministic equivalent with
  ExaModels + MadNLP/cuDSS.

## Where to go next

- Part I develops the theory: the
  [general multistage problem](@ref "Multistage stochastic optimization"),
  [the TS-DDR framework](@ref "The TS-DDR framework"),
  [SDDP and inconsistent formulations](@ref "Stochastic dual dynamic programming"),
  and [extensions](@ref "Extensions: mixed gradients, critics, and risk").
- Part III works three problems end to end, led by the hydrothermal
  case study
  ([problem](@ref "The long-term hydrothermal planning problem"),
  [instance](@ref "The Bolivian interconnected system"),
  [walkthrough](@ref "Hydropower Scheduling")).
- The [API Reference](@ref) documents every exported symbol.
