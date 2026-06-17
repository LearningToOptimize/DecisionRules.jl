# # Inventory Control with Fixed Ordering Costs
#
# This example is a finite-horizon inventory-control problem with a fixed setup
# cost for ordering. The binary variable ``z_t`` indicates whether an order is
# placed in period ``t``. If ``z_t=0``, the order quantity must be zero; if
# ``z_t=1``, the model pays a setup cost and may order up to capacity.
#
# Demand trajectories have a latent seasonal phase, persistent high/low regimes,
# and autocorrelated shocks. The phase is randomized across sample paths, so the
# calendar period is not a reliable signal of where the trajectory is in the
# cycle. TS-DDR is used as a time-invariant policy that reacts to inventory and
# realized demand history.

using DecisionRules
using JuMP, HiGHS
using Flux
using Statistics, Random

# ## Information Pattern
#
# At the beginning of a period, the controller observes current inventory and
# recent realized demand. It does **not** observe current demand before ordering.
# The order is therefore ex-ante. After ordering, demand is realized and becomes
# part of the state for the next period.

# ## Inventory Model
#
# ```math
# 0 \le q_t \le Q_{\max} z_t,\quad z_t\in\{0,1\}
# ```
#
# ```math
# s^{mid}_t = s_{t-1}+q_t,\qquad s_t=s^{mid}_t-d_t.
# ```
#
# Positive ``s_t`` is inventory and negative ``s_t`` is backlog. The stage cost is
#
# ```math
# Kz_t+cq_t+h\max(s_t,0)+p\max(-s_t,0).
# ```
#
# The example uses ``T=12``, ``K=100``, ``c=2``, ``h=1``, ``p=15``,
# ``Q_{\max}=200``, and ``s_0=30``.

# ## TS-DDR Policy
#
# The policy predicts a pre-demand order-up-to target. The target penalty is
# imposed on ``s^{mid}_t-\hat{s}^{mid}_t``. Current demand is a model uncertainty
# parameter, but the policy does not use it to choose the order target. It only
# copies realized demand into the history state for the next period.
#
# The state is:
#
# ```julia
# [net_inventory, last_demand, previous_demand]
# ```

# ## Demand Process
#
# Each trajectory has a path-level phase shift, a persistent latent high/low
# regime, and an autoregressive shock. The implementation samples
# ``\phi\sim\mathrm{Unif}\{0,\ldots,T-1\}``, shifts the seasonal demand band by
# ``\phi``, and then perturbs the shifted midpoint using the latent regime and
# AR shock. None of these latent variables is observed directly; the policy sees
# only inventory and realized demand history.
#
# ![Demand process](../assets/inventory_demand_process.png)

# ## Benchmarks
#
# - **SDDP.jl**: stagewise sampling on a 24-stage order/demand graph.
# - **Base-stock**: tuned constant order-up-to level.
# - **Marginal DP**: dynamic program for the nominal marginal seasonal model,
#   not an exact optimum for the random-phase latent process.
# - **Random**: untrained neural policy with the same ex-ante information pattern.

# ## Results
#
# Costs are out-of-sample operational costs. Fit time is wall-clock training or
# tuning time, and evaluation time is reported per simulated scenario.
#
# ![Training curve](../assets/inventory_training_curve.png)
#
# ![SDDP learning](../assets/inventory_sddp_learning.png)
#
# ![Inventory trajectories](../assets/inventory_trajectories.png)
#
# ![Cost comparison](../assets/inventory_cost_comparison.png)
#
# SDDP LP relaxation bound: **2449.1**
#
# | Method                  | N    | Mean cost | Std    | 95% CI | vs TS-DDR | Fit (s) | Eval ms/scen |
# |:------------------------|-----:|----------:|-------:|-------:|----------:|--------:|-------------:|
# | TS-DDR (trained)        |  300 |    3152.9 |  375.2 |   42.5 |     +0.0% |    72.2 |       133.24 |
# | SDDP.jl integer rollout |  300 |    3460.4 |  653.3 |   73.9 |     +9.8% |    18.5 |         0.82 |
# | Base-stock (S*=112)     |  300 |    3433.7 |  387.8 |   43.9 |     +8.9% |     0.4 |         0.15 |
# | Marginal DP policy      | 3000 |    3706.0 | 1301.1 |   46.6 |    +17.5% |     1.4 |         0.04 |
# | Random (untrained)      |  300 |    3453.8 |  445.4 |   50.4 |     +9.5% |     0.0 |       207.05 |
