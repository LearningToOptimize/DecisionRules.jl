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

## Mixed gradient: score-function (REINFORCE) correction

For problems with integer variables or non-smooth subproblems, the dual
gradient can be biased — it is local to a fixed integer assignment and cannot
see the effect of discrete switches (e.g., opening a setup variable).

DecisionRules provides a **score-function (REINFORCE)** correction that mixes
the dual gradient with a model-free policy gradient estimated from stage-wise
rollouts under perturbed targets.

### How the score-function estimator works

1. **Perturb**: add Gaussian noise to the policy targets:
   ``\tilde{x}_t = \hat{x}_t(\theta) + \delta_t``, where
   ``\delta_t \sim \mathcal{N}(0, \sigma^2 I)``.

2. **Rollout**: solve the stage-wise subproblems with the perturbed targets to
   obtain realized costs ``R_m`` for ``m = 1, \ldots, M`` rollouts. These
   rollouts solve the models exactly as built (MIPs stay MIPs), so the costs
   reflect true integer-feasible decisions.

3. **Advantage**: center the costs ``A_m = R_m - \bar{R}`` (mean baseline
   reduces variance without changing the expected gradient).

4. **Surrogate loss**: the differentiable scalar whose gradient recovers the
   REINFORCE estimate:

```math
L_{\text{sf}}(\theta)
\;=\;
\frac{1}{M} \sum_{m=1}^{M}
  A_m
  \sum_{t=1}^{T}
  \left\langle
    \frac{\delta_{m,t}}{\sigma^2},\;
    \hat{x}_{t+1}(\theta)
  \right\rangle.
```

This is the standard score-function estimator for Gaussian perturbations.
The key identity is
``\nabla_\theta \log p(\delta_t \mid \theta) = \delta_t / \sigma^2``
for a Gaussian centered at ``\hat{x}_t(\theta)``.

### Mixed gradient

The final training gradient combines both signals:

```math
\nabla L
\;=\;
\alpha\, \nabla L_{\text{dual}}
+ (1 - \alpha)\, \nabla L_{\text{sf}},
```

where ``\alpha \in [0, 1]`` is the `dual_weight`.

There are two separate solve paths in the mixed-gradient training loop:

- **Dual path**: controlled by `integer_strategy`, which determines how local
  dual information is read from the deterministic equivalent
  (e.g., [`FixedDiscreteIntegerStrategy`](@ref) solves the MIP, fixes integers,
  re-solves the LP, and reads LP duals).
- **Score-function path**: controlled by [`ScoreFunctionConfig`](@ref), which
  owns separate rollout subproblems. These are solved exactly as built, and
  their realized costs define the Monte Carlo score-function term.

### Scheduled ramp-in

A [`ScoreFunctionSchedule`](@ref) can ramp ``\alpha`` from 1 (pure dual) to
its final value over a warmup period.  Let ``k`` be the current iteration and
``\rho_k = \operatorname{clip}((k - k_0) / r,\, 0,\, 1)``.  The effective
score-function weight is ``\rho_k (1 - \alpha)``.

This lets the DE dual gradient establish a good initial policy before
introducing the higher-variance REINFORCE signal.

See the [Stochastic Lot-Sizing with Fixed Ordering Costs](@ref) example for a
complete worked example with integer variables and mixed gradients.

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

## Strict mode: penalty-free gradient signal

The standard TS-DDR formulation uses a penalty ``C_\delta \|\delta_t\|`` to
penalize deviations from the policy's targets. While effective, the penalty
introduces a trade-off: the dual ``\lambda_t`` conflates the **economic shadow
price** with a **penalty-correction term**. At high penalty, the gradient
signal tells the policy "reduce ``\delta``" rather than "be economically
optimal."

**Strict mode** eliminates this coupling entirely by replacing the slack
constraint ``x_t + \delta_t = \hat{x}_t`` with a **hard equality**:

```math
x_t = \hat{x}_t \quad :\lambda_t
```

There are no deficit variables, no penalty term, and no penalty to tune. The
dual ``\lambda_t`` is the **pure shadow price** ``\partial Q_t / \partial
\hat{x}_t`` — the marginal value of changing the target, uncontaminated by
any regularization.

### Feasibility-guaranteeing policies

Strict mode requires that the policy always produce feasible targets — if the
target is outside the feasible set, the subproblem has no slack to absorb the
violation and becomes infeasible. This is guaranteed by construction through a
**reachable-set policy** that bounds its output to the one-stage reachable set.

For hydro scheduling with reservoir ``r``, the one-stage reachable set from
current volume ``v_r`` under inflow ``w_r`` is:

```math
\hat{v}_r \in
\Bigl[
  \max\bigl(\underline{v}_r,\; v_r + K w_r - K \overline{q}_r - \overline{s}_r
        + \sum_{u \in U_r^{\text{turn}}} K \underline{q}_u\bigr),\;
  \min\bigl(\overline{v}_r,\; v_r + K w_r - K \underline{q}_r
        + \sum_{u \in U_r^{\text{turn}}} K \overline{q}_u\bigr)
\Bigr],
```

where ``K`` is the water-balance conversion factor, ``\underline{q}_r,
\overline{q}_r`` are turbine bounds, ``\overline{s}_r`` is the spill bound,
and ``U_r^{\text{turn}}`` is the set of upstream units feeding into ``r``.

The [`HydroReachablePolicy`] implements this by passing the LSTM encoder output
through a sigmoid activation and scaling the result to ``[\ell_r, u_r]``:

```math
\hat{v}_r = \ell_r + (u_r - \ell_r) \cdot \sigma(z_r).
```

The bounds ``\ell_r, u_r`` are computed from physics (no gradient flows through
them); the gradient path is solely through ``\sigma(z_r)``, exactly as in the
standard TS-DDR pipeline.

### When to use strict mode

Strict mode is the preferred approach when:

1. The problem has **absolute recourse** — any state within the physical bounds
   is feasible for the subproblem solver.
2. The reachable set is **easy to compute** — e.g., reservoir water balance with
   known turbine/spill bounds.
3. You want to avoid **penalty tuning** — strict mode has no penalty hyperparameter.

In the [Hydropower Scheduling](@ref) example, strict mode with a
`HydroReachablePolicy` achieves competitive simulation costs out of the box,
with no penalty schedule, no annealing, and no hyperparameter search.

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
