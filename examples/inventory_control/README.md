# Inventory Control Example

This example studies a 12-period stochastic lot-sizing problem with fixed order
costs. It is meant to be a hard mixed-integer inventory benchmark for TS-DDR,
not a toy newsvendor example.

## Problem

At the start of each period, the controller knows:

- current net inventory,
- demand observed in the previous two periods.

It does **not** know the demand that will arrive in the current period. The
controller chooses whether to order and how much to order before current demand
is realized.

The binary variable `z_t` means "place an order in period `t`." If `z_t = 0`,
then the order quantity `q_t` must be zero. If `z_t = 1`, the model pays the
fixed setup cost `K` and may order up to `Q_max` units:

```julia
0 <= q_t <= Q_max * z_t
```

After the order arrives, demand is realized:

```julia
s_mid[t] = s_in[t] + q[t]
s_out[t] = s_mid[t] - demand[t]
```

Positive `s_out` is held as inventory; negative `s_out` is backlog. The realized
state for the next period is `[s_out[t], demand[t], demand[t-1]]`, so policies
may react to demand history without seeing current demand before ordering.

## Demand Process

Demand has a seasonal peak around period 6 plus a persistent latent high/low
regime and autocorrelated shocks. Each sample path receives an independent
random phase shift, so the peak can arrive at a different calendar period in
different scenarios. This is intentionally difficult for methods that assume
stagewise-independent uncertainty or rely on a fixed period label. The only
stochastic observation available to real policies is realized demand history.

In code, each path samples a phase shift `phi`, a latent regime in
`{-1, 0, 1}`, and an AR(1) shock. Period `t` then uses the shifted seasonal band
`D_LO[mod1(t + phi, T)]` to `D_HI[mod1(t + phi, T)]`. The latent variables are
never passed to policies; they must infer the process from recent demands.

## Scripts

Run from the repository root:

```bash
julia --project=examples/inventory_control examples/inventory_control/train_dr_inventory.jl
julia --project=examples/inventory_control examples/inventory_control/evaluate_inventory.jl
julia --project=examples/inventory_control examples/inventory_control/solve_sddp.jl
julia --project=examples/inventory_control examples/inventory_control/solve_optimal_dp.jl
julia --project=examples/inventory_control examples/inventory_control/compare_results.jl
```

The first four scripts can be run independently. `compare_results.jl` should be
run last.

## Outputs

Results are written to `examples/inventory_control/results/`:

- `dr_costs.csv`, `dr_trajectories.csv`, `dr_orders.csv`
- `basestock_costs.csv`, `basestock_trajectories.csv`, `basestock_S_star.txt`
- `random_costs.csv`
- `sddp_costs.csv`, `sddp_bound.txt`, `sddp_training_log.csv`
- `optimal_costs.csv`, `optimal_dp_value.txt`
- `training_curve.csv`
- `dr_timing.csv`, `baseline_timing.csv`, `sddp_timing.csv`,
  `optimal_timing.csv`

Figures are written to `docs/src/assets/`:

- `inventory_demand_process.png`
- `inventory_training_curve.png`
- `inventory_sddp_learning.png`
- `inventory_trajectories.png`
- `inventory_cost_comparison.png`

The final comparison table in the documentation reports out-of-sample
operational cost, wall-clock fit time, and evaluation milliseconds per
scenario.

## Benchmarks

- **TS-DDR** learns an ex-ante order target from inventory and demand history,
  using the same time-invariant policy at every period.
- **SDDP.jl** uses a 24-stage order/demand graph and is evaluated by integer
  rollout on the same latent demand process.
- **Base-stock** is a tuned constant order-up-to policy.
- **Marginal DP** is a specialized reference policy for a simplified marginal
  seasonal model. It is not an exact optimum for the latent demand process.
- **Random** is an untrained ex-ante neural policy.

The expected qualitative result is that TS-DDR should beat the practical
baselines, especially SDDP and base-stock. The marginal-DP policy is useful as a
structure-aware reference for a simplified model, but it is not an oracle for the
true random-phase process and should not be interpreted as an exact lower bound.
