# Extensions: mixed gradients, critics, and risk

```@meta
CurrentModule = DecisionRules
```

The dual gradient of [The TS-DDR framework](@ref) is exact for smooth
subproblems and, over fresh samples, an unbiased estimator of the
expected-cost gradient. Two practical situations call for more:
**discrete decisions**, where the dual is blind to integer switches and a
score-function (REINFORCE) correction restores the missing signal, and
**small sample budgets**, where a control-variate critic cuts the
estimator's variance without moving its optimum. Both extensions, and a
risk-averse change-of-measure variant of the gradient, ship with the
package.

!!! note "Scope"
    The battery-storage case study is continuous and is designed to use the
    pure strict dual gradient after its feasibility gates; it uses none of these
    extensions. The score-function correction is exercised in
    [Stochastic Lot-Sizing with Fixed Ordering Costs](@ref), whose
    fixed-charge (binary) ordering decisions are exactly the situation it
    addresses.

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

3. **Advantage**: center the costs ``A_m = R_m - \bar{R}``. Because the mean
   baseline ``\bar{R}`` is computed from the same ``M`` rollouts, mean-centering
   reduces variance but introduces a small ``O(1/M)`` bias (effectively scaling
   the estimator by ``(M-1)/M``) that vanishes as `num_rollouts` grows; a
   leave-one-out baseline would be exactly unbiased.

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

## Variance reduction: control-variate critic

The dual gradient over a batch of ``N`` sampled trajectories is the
sample-average

```math
g \;=\; \frac{1}{N}\sum_{s=1}^{N}\sum_{t=1}^{T}
        \bigl\langle \lambda^{s}_t,\; \partial \hat{x}^{s}_t/\partial\theta \bigr\rangle ,
\qquad \lambda^{s}_t = \partial Q_s/\partial \hat{x}^{s}_t .
```

With fresh independent samples each step this is an **unbiased** estimator of
``\nabla_\theta\,\mathbb{E}[Q]`` for any ``N``. A small batch does not bias it —
it only inflates its **variance**, which sets the SGD noise floor and keeps the
policy short of the optimum. Rather than paying for a large ``N``, a
`ScalarCriticControlVariate` subtracts a learned, state-conditioned
baseline ``b^{s}_t = \nabla_{\hat{x}_t} C \approx \lambda^{s}_t`` and adds it back
as an independent-sample expectation:

```math
g_{\text{cv}} \;=\;
  \frac{1}{N}\sum_{s}\sum_t \bigl\langle \lambda^{s}_t - b^{s}_t,\; \partial\hat{x}^{s}_t/\partial\theta\bigr\rangle
  \;+\;
  \frac{1}{M}\sum_{j}\sum_t \bigl\langle b^{j}_t,\; \partial\hat{x}^{j}_t/\partial\theta\bigr\rangle .
```

**Unbiasedness.** The subtracted and added terms are two Monte-Carlo estimates of
the same expectation ``\mathbb{E}\bigl[\sum_t\langle b_t,\partial\hat{x}_t/\partial\theta\rangle\bigr]``,
so ``\mathbb{E}[g_{\text{cv}}] = \mathbb{E}[g] = \nabla_\theta\mathbb{E}[Q]``: the
critic **cannot move the optimum**. (If the add-back reuses the same samples with
``M=N``, the two terms cancel and the critic is a no-op — a fresh add-back batch,
`num_cheap_critic_samples_per_batch > 0`, is what activates it.)

**Variance.** The reduction is governed by how well the baseline tracks the dual,

```math
\text{Var reduction} \;\approx\; \frac{1}{1 - R^2}, \qquad
R^2 = \text{explained variance of } \lambda_t \text{ by } b_t ,
```

so the baseline must be trained to **match the dual** (`gradient_loss_weight > 0`),
not only the scalar value (`value_loss_weight`): a value-only critic leaves
``\nabla_{\hat{x}} C`` unconstrained, ``R^2\approx 0``, and yields no reduction. The
control-variate critic is therefore a **sample-efficiency lever** — it reaches the
same risk-neutral optimum a much larger batch would, at a modest batch size.

## Risk-averse extension (nested change-of-measure)

The same per-stage critic supports a **risk-averse** objective by replacing the
expectation with a nested, time-consistent coherent risk measure (e.g. conditional
``\mathrm{CVaR}_\alpha``). Each stage weight becomes ``\Xi^{s}_t\,\lambda^{s}_t`` with
``\Xi^{s}_t = \prod_{u\le t}\zeta^{s}_u``, the causal product of per-node
change-of-measure densities ``\zeta_u`` (``\zeta\equiv 1`` recovers the risk-neutral
gradient above). Because each ``\zeta_u`` depends only on the distribution
*conditional* on stage ``u``, the policy never hedges against a tail already
precluded by realized uncertainty — the time-consistency property that a global
scenario re-weighting would violate. This targets tail cost at the expense of the
mean, and is a distinct objective from the risk-neutral formulation above.
