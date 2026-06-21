# DecisionRules.jl

```@meta
CurrentModule = DecisionRules
```

DecisionRules.jl trains parametric decision rules through multi-stage optimization,
implementing the **Two-Stage Deep Decision Rules (TS-DDR)** framework from
[arXiv:2405.14973](https://arxiv.org/abs/2405.14973).

## How it works

In multi-stage stochastic control, the feasible action at each stage comes from solving
a constrained optimization problem (OPF, MPC, hydrothermal dispatch, …). Rather than
outputting actions directly, the neural-network policy outputs **target states**.
An optimization subproblem then projects these targets onto the feasible set defined by
dynamics and constraints. Lagrange duals and implicit differentiation (via
[DiffOpt.jl](https://github.com/jump-dev/DiffOpt.jl)) provide the gradient signal to
update the policy end-to-end.

Three training formulations are supported:

| Formulation | Horizon coupling | Gradient source |
|:---|:---|:---|
| **Deterministic Equivalent** | Full horizon, one large NLP | Duals on the coupled problem |
| **Stage-wise (single shooting)** | Sequential rollout | Duals + DiffOpt per stage |
| **Multiple Shooting** | Windowed sub-horizons | DiffOpt per window, continuity penalties |

## Installation

```julia
using Pkg
Pkg.add("DecisionRules")
```

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

See the [Algorithm](@ref) page for the mathematical formulation, the
[Uncertainty Sampling](@ref) guide for how to prepare your scenario data, and the
examples for complete worked problems.

## Citation

```bibtex
@article{rosemberg2024efficiently,
  title={Efficiently Training Deep-Learning Parametric Policies using Lagrangian Duality},
  author={Rosemberg, Andrew and Street, Alexandre and Vallad{\~a}o, Davi M and Van Hentenryck, Pascal},
  journal={arXiv preprint arXiv:2405.14973},
  year={2024}
}
```
