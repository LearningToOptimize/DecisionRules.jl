# Algorithm

```@meta
CurrentModule = DecisionRules
```

This page summarizes the TS-DDR (Two-Stage Deep Decision Rules) training algorithm.
For the full derivation, see [arXiv:2405.14973](https://arxiv.org/abs/2405.14973).

## Problem setting

Consider a ``T``-stage stochastic control problem where at each stage ``t`` we observe
an uncertainty realization ``w_t`` and must choose an action ``u_t`` that satisfies
stage constraints ``(u_t, x_t) \in \mathcal{X}_t(x_{t-1}, w_t)``. The goal is to
minimize the expected total cost:

```math
\min_\theta \; \mathbb{E}_{w_{1:T}} \left[ \sum_{t=1}^{T} c_t(x_t, u_t) \right]
```

where ``x_t`` evolves according to the constrained dynamics and ``\theta`` parameterizes
the policy.

## Target-state policies

Instead of mapping observations directly to actions, the policy outputs **target
states**:

```math
\hat{x}_{1:T} = \pi_\theta(w_{1:T})
```

A projection subproblem enforces feasibility by solving:

```math
\min_{x_t, u_t} \; c_t(x_t, u_t) + \lambda \| x_t - \hat{x}_t \|
\quad \text{s.t.} \quad (u_t, x_t) \in \mathcal{X}_t(x_{t-1}, w_t)
```

The target ``\hat{x}_t`` enters as a **parameter** (not a decision variable). The
penalty ``\lambda`` on the slack ``\| x_t - \hat{x}_t \|`` ensures that when the target
is feasible, the optimizer follows it exactly; when infeasible, it deviates minimally.

## Gradient computation

The policy gradient with respect to ``\theta`` decomposes via the chain rule:

```math
\nabla_\theta \mathcal{L} =
  \sum_{t=1}^{T} \frac{\partial \mathcal{L}}{\partial \hat{x}_t}
  \cdot \frac{\partial \hat{x}_t}{\partial \theta}
```

The first factor — sensitivity of the loss to the target — comes from the **Lagrange
duals** of the target constraints (or equivalently, from implicit differentiation of the
KKT conditions via DiffOpt). The second factor is a standard neural-network backprop.

This two-stage structure avoids differentiating through the full optimization solver:
dual information provides a first-order signal, and DiffOpt handles the implicit
function theorem when needed (e.g., for state-transition sensitivities).

## Three training formulations

### Deterministic equivalent

All stages are coupled into a single NLP for a sampled trajectory ``w_{1:T}``:

```math
\min_{x, u} \; \sum_{t=1}^T c_t(x_t, u_t) + \lambda \| x_t - \hat{x}_t \|
\quad \text{s.t.} \quad \text{dynamics + constraints for all } t
```

The policy generates targets in a single forward pass, and the coupled solve determines
the realized states. DiffOpt differentiates through the full NLP.

**Pros**: strongest gradient signal (full horizon coupling).
**Cons**: largest subproblem per sample; targets generated without realized-state feedback.

### Stage-wise decomposition (single shooting)

Each stage is solved independently in sequence:

```
for t = 1, ..., T:
    x̂_t = π_θ(w_{1:t}, x_{t-1})       # policy predicts target
    solve stage-t subproblem            # project onto feasible set
    x_t = realized state from solver    # feed back to next stage
```

Gradients combine dual information for targets with DiffOpt sensitivities along the
rollout chain.

**Pros**: closed-loop policy (sees realized states); smaller per-stage solves.
**Cons**: sequential; gradient signal weakens over long horizons.

### Multiple shooting

The horizon is partitioned into windows of ``W`` stages. Each window solves a
deterministic equivalent over its stages, then passes the realized end-state to the
next window:

```
for k = 1, ..., ⌈T/W⌉:
    solve window-k deterministic equivalent (stages (k-1)W+1 to kW)
    pass realized end-state to window k+1
```

**Pros**: balances coupling (within windows) with tractability; parallelizable windows.
**Cons**: continuity gaps between windows require penalty tuning.

## Penalty annealing

The target penalty ``\lambda`` is critical: too small and the optimizer ignores
targets (no gradient); too large and the problem becomes ill-conditioned. DecisionRules.jl
supports a **penalty annealing schedule** that ramps ``\lambda`` during training:

```
Phase 1 (warmup):  λ × 0.1   — let the policy explore
Phase 2 (nominal): λ × 1.0   — standard training
Phase 3 (tighten): λ × 10.0  — sharpen target tracking
Phase 4 (lock):    λ × 30.0  — final precision
```

This is the `default_annealed` schedule, activated with `penalty_schedule=:default_annealed`.

## Evaluation semantics

A policy trained on the deterministic equivalent generates targets using **target-state
feedback** (each target depends on the previous *predicted* target, not the realized
state). Evaluating such a policy with **realized-state feedback** (deployment semantics)
tests a different closed-loop path and will generally report higher cost.

[`RolloutEvaluation`](@ref) supports both modes via the `policy_state` keyword:
- `:target` — matches DE training semantics (fair in-sample comparator)
- `:realized` — deployment/closed-loop semantics (the true test)

The **target-violation share** measures how much of the rollout objective comes from
the slack penalty rather than operational cost. A small share (≤ 5%) means the policy's
targets are followable stage-by-stage; a large share signals that the coupled DE solve
was absorbing infeasible targets through slack.
