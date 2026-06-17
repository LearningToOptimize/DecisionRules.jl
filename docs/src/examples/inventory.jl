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

# ## Integer Postprocessing Strategy
#
# DecisionRules.jl provides `FixedDiscreteIntegerStrategy` for problems with
# binary variables. At each training step: (1) solve the MIP for incumbent
# binary values ``z^*_t``; (2) fix ``z_t = z^*_t`` and relax integrality;
# (3) re-solve the resulting LP; (4) read LP duals as gradient signal.
# This is the same principle as SDDP.jl's `FixedDiscreteDuality`: both fix
# the binary incumbent and extract LP duals as subgradients.  SDDP uses them
# to build Benders cuts; TS-DDR uses them to back-propagate through the
# neural policy.
#
# ## Benchmarks
#
# - **SDDP.jl**: a 24-stage order/demand graph trained with a stagewise
#   sampling approximation. It sees the stage index through the policy graph,
#   but not the latent phase, regime, or demand history of each sample path.
#   Because the integer ordering variable ``z_t`` is relaxed to ``[0,1]``
#   during training, the LP cuts systematically underestimate the fixed
#   ordering cost — the rollout rounds ``z`` back to binary.
# - **Base-stock**: a tuned constant order-up-to policy (``S^*`` found by
#   grid search).
# - **Marginal DP**: backward dynamic program on the nominal seasonal model
#   (demand uniform over ``[D^{lo}_t, D^{hi}_t]`` per stage without phase
#   shifts, regime switching, or autocorrelation).  This gives the optimal
#   policy for the single-cycle nominal problem but cannot adapt to the
#   latent state — its stage-specific ordering rule may apply a peak-season
#   policy during a trough and vice versa.
# - **Random**: untrained neural policy with the same ex-ante information
#   pattern as TS-DDR.  It still solves MIP subproblems per stage, so it
#   isolates the benefit of training from the benefit of the MIP structure.

# ## Results
#
# All costs below are out-of-sample operational costs, excluding the auxiliary
# TS-DDR target-tracking penalty used during training.  All methods are
# evaluated on the same 300 demand scenarios (seed 555) for fair comparison.
#
# **Fit** is the one-time offline cost: TS-DDR neural-network training, SDDP
# cut building, base-stock grid search, or DP backward induction.
# **Eval** is the online deployment cost per decision point.  For TS-DDR and
# Random, this is the time to solve one stage MIP subproblem.  For SDDP,
# the full algorithm must be re-run because its LP-relaxation cuts cannot be
# pre-computed for the latent demand process (see [arXiv:2405.14973](https://arxiv.org/abs/2405.14973)).
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
# | Method                  | N   | Mean cost | Std    | 95% CI | vs TS-DDR | Fit (s) | Eval (s) |
# |:------------------------|----:|----------:|-------:|-------:|----------:|--------:|---------:|
# | TS-DDR (trained)        | 300 |    3152.9 |  375.2 |   42.5 |     +0.0% |    70.6 |   0.0112 |
# | SDDP.jl integer rollout | 300 |    3459.2 |  669.3 |   75.7 |     +9.7% |     0.0 |  17.4000 |
# | Base-stock (S*=110)     | 300 |    3456.3 |  435.8 |   49.3 |     +9.6% |     0.2 |   0.0002 |
# | Marginal DP policy      | 300 |    3759.0 | 1318.9 |  149.2 |    +19.2% |     1.5 |   0.0003 |
# | Random (untrained)      | 300 |    3453.8 |  445.4 |   50.4 |     +9.5% |     0.0 |   0.0111 |
#
# The main qualitative point is not that TS-DDR is faster at rollout — it
# still solves mixed-integer subproblems during evaluation.  The point is
# that a time-invariant policy trained through the deterministic equivalent
# learns a useful reaction to demand history, whereas methods built around
# stagewise independence (SDDP) or fixed seasonal structure (Marginal DP)
# are misled by the random phase and persistent latent regimes.
#
# ## Runnable Scripts
#
# The complete experiment lives in `examples/inventory_control/`:
#
# | Script | Purpose |
# |:-------|:--------|
# | `build_inventory_problem.jl` | JuMP subproblem and det-equivalent builders, demand process, policy architecture |
# | `train_dr_inventory.jl` | TS-DDR training and trajectory evaluation |
# | `evaluate_inventory.jl` | Base-stock grid-search and random baseline evaluation |
# | `solve_sddp.jl` | SDDP (2T-stage) training and integer rollout |
# | `solve_optimal_dp.jl` | Marginal DP backward induction and simulation |
# | `compare_results.jl` | Load all CSVs, print summary table, save plots |
#
# ```bash
# julia --project=examples/inventory_control examples/inventory_control/train_dr_inventory.jl
# julia --project=examples/inventory_control examples/inventory_control/evaluate_inventory.jl
# julia --project=examples/inventory_control examples/inventory_control/solve_sddp.jl
# julia --project=examples/inventory_control examples/inventory_control/solve_optimal_dp.jl
# julia --project=examples/inventory_control examples/inventory_control/compare_results.jl
# ```
