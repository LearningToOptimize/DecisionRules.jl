# Inventory Control Example

This example studies a 12-period stochastic lot-sizing problem with two
formulations — relaxed (continuous LP) and integer (MIP with fixed ordering
costs). It benchmarks TS-DDR against SDDP, base-stock, and random policies.

## Problem

At the start of each period, the controller knows:

- current net inventory,
- demand observed in the previous two periods.

It does **not** know the demand that will arrive in the current period. The
controller chooses whether to order and how much to order before current demand
is realized.

### Relaxed formulation

No binary variable. The order quantity `q_t ∈ [0, Q_max]` is continuous.
Cost per period: `c·q + h·max(s,0) + p·max(-s,0)`.

### Integer formulation

The binary variable `z_t` means "place an order in period `t`." If `z_t = 0`,
then the order quantity `q_t` must be zero. If `z_t = 1`, the model pays the
fixed setup cost `K` and may order up to `Q_max` units:

```julia
0 <= q_t <= Q_max * z_t
```

Cost per period: `K·z + c·q + h·max(s,0) + p·max(-s,0)`.

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

SDDP uses a PAR(1) approximation fitted to the true process, carrying
`d_lag` as a state variable to capture autocorrelation.

## Integer Postprocessing Strategies

TS-DDR uses `AbstractIntegerStrategy` to handle discrete variables in
subproblems. Two strategies are available:

- **`FixedDiscreteIntegerStrategy`**: Solve the MIP, fix binary/integer
  variables at their incumbent values, relax integrality, re-solve the
  fixed continuous LP, and read duals. The gradient is local to the
  incumbent integer assignment.

- **`ContinuousRelaxationIntegerStrategy`**: Relax all binary/integer
  constraints to continuous bounds (binary → [0,1]), solve the resulting
  LP, and read duals directly. Faster (one LP solve instead of MIP + LP)
  with smoother gradients, but the solution may have fractional integer
  variables.

For the relaxed formulation (no integer variables), `NoIntegerStrategy`
is used — subproblems are solved and duals read as-is.

## Scripts

Run from the repository root:

```bash
julia --project=examples/inventory_control examples/inventory_control/train_dr_inventory.jl
julia --project=examples/inventory_control examples/inventory_control/evaluate_inventory.jl
julia --project=examples/inventory_control examples/inventory_control/solve_sddp.jl
julia --project=examples/inventory_control examples/inventory_control/compare_results.jl
```

The first three scripts can be run independently. `compare_results.jl` should be
run last.

## Outputs

Results are written to `examples/inventory_control/results/` with `relaxed_`
and `integer_` prefixes:

- `{tag}_dr_costs.csv`, `{tag}_dr_trajectories.csv`
- `{tag}_basestock_costs.csv`, `{tag}_basestock_trajectories.csv`, `{tag}_basestock_S_star.txt`
- `{tag}_random_costs.csv`
- `{tag}_sddp_costs.csv`, `{tag}_sddp_bound.txt`, `{tag}_sddp_training_log.csv`
- `{tag}_training_curve.csv`
- `{tag}_dr_timing.csv`, `{tag}_baseline_timing.csv`, `{tag}_sddp_timing.csv`

Figures are written to `docs/src/assets/`:

- `inventory_demand_process.png`
- `inventory_relaxed_results.png`
- `inventory_integer_results.png`

## Benchmarks

- **TS-DDR** learns an ex-ante order target from inventory and demand history,
  using the same time-invariant neural policy at every period.
- **SDDP** uses a PAR(1) demand approximation in a 24-stage order/demand graph.
  For the integer case, it uses LP relaxation with integer rounding at rollout.
- **Base-stock** is a tuned constant order-up-to policy.
- **Random** is an untrained ex-ante neural policy.

The expected qualitative result is:
- **Relaxed**: SDDP dominates (near-optimal for convex problems with Markov noise).
- **Integer**: TS-DDR dominates (handles MIP subproblems natively via integer
  postprocessing strategies, while SDDP's LP relaxation underestimates fixed costs).
