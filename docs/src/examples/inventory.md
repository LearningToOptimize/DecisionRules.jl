```@meta
EditURL = "inventory.jl"
```

# Inventory Control with Fixed Ordering Costs

This example is a finite-horizon inventory-control problem with the feature that
makes lot-sizing difficult in practice: placing an order has a fixed setup cost.
The controller therefore faces a genuine batching decision. It may be better to
order a large quantity now and carry inventory, or to skip an order and risk
backlog later.

The example is designed to stress methods that rely on stagewise-independent
uncertainty. Each simulated demand trajectory has a latent seasonal phase,
persistent high/low demand regimes, and autocorrelated shocks. The phase is
randomized independently across sample paths, so a method cannot solve the
problem by memorizing that a particular calendar period is always the peak.
Useful decisions require reacting to observed demand history.

## Information Pattern

At the start of period ``t``, the controller observes:

- current net inventory ``s_{t-1}``,
- the last two realized demands.

It does **not** observe the demand that will arrive in the current period.
The order decision is therefore ex-ante. After the order is placed, current
demand is realized and becomes part of the history available next period.

TS-DDR uses the same neural policy at every period. The policy is time-invariant:
it receives demand history and inventory, but not the period index.

## Inventory Model

The binary variable has the standard lot-sizing interpretation:

```math
z_t =
\begin{cases}
1, & \text{place an order in period } t,\\
0, & \text{do not place an order.}
\end{cases}
```

If ``z_t=0``, no units can be ordered. If ``z_t=1``, the model pays the fixed
setup cost ``K`` and may order up to capacity:

```math
0 \le q_t \le Q_{\max} z_t.
```

The ordered units arrive before demand:

```math
s^{mid}_t = s_{t-1} + q_t,
\qquad
s_t = s^{mid}_t - d_t.
```

Positive ``s_t`` is inventory on hand; negative ``s_t`` is backlog. The period
cost is

```math
K z_t + c q_t + h \max(s_t,0) + p \max(-s_t,0).
```

The numerical parameters are:

| Parameter | Value | Meaning |
|:--|--:|:--|
| ``T`` | 12 | periods |
| ``K`` | 100 | fixed order/setup cost |
| ``c`` | 2 | unit ordering cost |
| ``h`` | 1 | holding cost |
| ``p`` | 15 | backlog penalty |
| ``Q_{\max}`` | 200 | order capacity |
| ``s_0`` | 30 | initial inventory |

## TS-DDR Formulation

The TS-DDR policy predicts a pre-demand order-up-to target ``\hat{s}^{mid}_t``.
The mixed-integer subproblem chooses ``z_t`` and ``q_t`` to track that target
while respecting setup and capacity constraints. The target penalty is imposed on
``s^{mid}_t-\hat{s}^{mid}_t``, not on post-demand inventory.

Although current demand is present as an uncertainty parameter in the JuMP model,
the policy does not use it to choose the order target. The policy output has two
roles:

- choose the current pre-demand inventory target,
- copy realized demand into the state so it can be used next period.

The state carried between periods is:

```julia
[net_inventory, last_demand, previous_demand]
```

This lets a time-invariant policy infer the latent demand regime from recent
observations without receiving a period counter or synthetic seasonal features.

## Demand Process

Let ``\phi`` be a path-level phase shift, sampled uniformly from
``\{0,\ldots,T-1\}``, and let
``\kappa_t = 1 + ((t+\phi-1) \bmod T)`` denote the shifted seasonal index. The
nominal lower and upper seasonal bands are ``D^{lo}_{\kappa_t}`` and
``D^{hi}_{\kappa_t}``, with midpoint ``m_{\kappa_t}`` and half-width
``w_{\kappa_t}``.

Demand also contains a persistent latent regime ``r_t\in\{-1,0,1\}`` and an
autoregressive shock ``\epsilon_t``. In the implementation,

```math
\epsilon_t = 0.84\,\epsilon_{t-1}+0.35\,\eta_t,
\qquad
d_t =
\operatorname{clip}\!\left(
  m_{\kappa_t}
  + w_{\kappa_t}
    (0.78 r_t + 0.42 \epsilon_t + 0.12 \eta'_t)
\right),
```

where the regime is resampled with probability ``0.08`` each period. The phase,
regime, and shocks are not observed directly. The controller sees their effect
only through realized demand history.

The plot below shows 24 sampled demand paths. The dashed curve is the nominal
seasonal center before the random phase shift. Because each trajectory has a
different phase and persistent regime, the same period can correspond to high,
medium, or low demand across scenarios.

![Demand process](../assets/inventory_demand_process.png)

## Benchmarks

The comparison uses four baselines:

- **SDDP.jl**: a 24-stage order/demand graph trained with a stagewise sampling
  approximation. It sees the stage index through the policy graph, but not the
  latent phase of each sample path.
- **Base-stock**: a tuned constant order-up-to policy.
- **Marginal DP**: a specialized dynamic program for the nominal marginal
  seasonal model. It is intentionally included as a structure-aware reference,
  but it is not exact for the true random-phase latent process.
- **Random**: an untrained neural policy with the same ex-ante information
  pattern as TS-DDR.

## Results

All costs below are out-of-sample operational costs, excluding the auxiliary
TS-DDR target-tracking penalty used during training. Fit time is the wall-clock
time spent training or tuning the method. Evaluation time is reported per
simulated scenario.

TS-DDR training:

![Training curve](../assets/inventory_training_curve.png)

SDDP learning curve:

![SDDP learning](../assets/inventory_sddp_learning.png)

Net-inventory trajectories:

![Inventory trajectories](../assets/inventory_trajectories.png)

Cost distribution:

![Cost comparison](../assets/inventory_cost_comparison.png)

SDDP LP relaxation bound: **2449.1**

| Method                  | N    | Mean cost | Std    | 95% CI | vs TS-DDR | Fit (s) | Eval ms/scen |
|:------------------------|-----:|----------:|-------:|-------:|----------:|--------:|-------------:|
| TS-DDR (trained)        |  300 |    3152.9 |  375.2 |   42.5 |     +0.0% |    72.2 |       133.24 |
| SDDP.jl integer rollout |  300 |    3460.4 |  653.3 |   73.9 |     +9.8% |    18.5 |         0.82 |
| Base-stock (S*=112)     |  300 |    3433.7 |  387.8 |   43.9 |     +8.9% |     0.4 |         0.15 |
| Marginal DP policy      | 3000 |    3706.0 | 1301.1 |   46.6 |    +17.5% |     1.4 |         0.04 |
| Random (untrained)      |  300 |    3453.8 |  445.4 |   50.4 |     +9.5% |     0.0 |       207.05 |

The main qualitative point is not that TS-DDR is faster at rollout; it still
solves mixed-integer subproblems during evaluation. The point is that a
time-invariant policy trained through the deterministic equivalent learns a
useful reaction to demand history, whereas methods built around stagewise
independence or fixed order-up-to rules are misled by the random phase and
persistent latent regimes.

The scripts used to generate these numbers are in
`examples/inventory_control/`.
