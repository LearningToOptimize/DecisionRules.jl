# DecisionRules.jl (WIP)

[![Build Status](https://github.com/LearningToOptimize/DecisionRules.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/LearningToOptimize/DecisionRules.jl/actions/workflows/CI.yml?query=branch%3Amain)

![Diagram](https://github.com/LearningToOptimize/DecisionRules.jl/blob/main/diagram_tsddr.png)

DecisionRules.jl provides a differentiable workflow for **multistage stochastic optimization**. It couples JuMP/DiffOpt models with machine-learned decision rules (e.g., Flux neural networks) so you can simulate, differentiate, and train policies end-to-end.

## Features
- Build multistage JuMP models with differentiable subproblems (DiffOpt).
- Simulate single- or multi-stage problems with stochastic scenarios.
- Train decision rules (Flux models) directly through reverse-mode differentiation.
- Utilities for sampling uncertainties, computing deterministic equivalents, and extracting duals.

## Installation
```julia
using Pkg
Pkg.add(url = "https://github.com/LearningToOptimize/DecisionRules.jl.git")
```
This package relies on DiffOpt, JuMP, and Flux; see the `Project.toml` for full dependencies.

## Quickstart

### Single-stage simulation
```julia
using DecisionRules, JuMP, DiffOpt, SCS

subproblem = DiffOpt.conic_diff_model(SCS.Optimizer)
# ... build your JuMP model with parameterized state/uncertainty vars ...
obj = DecisionRules.simulate_stage(subproblem, state_in_params, state_out_params,
    sampled_uncertainty, current_state, candidate_action)
```

### Multistage with a learned policy
```julia
using DecisionRules, Flux

# Assume subproblems, state_params_in/out, initial_state, uncertainty_samples are defined
policy = Chain(Dense(2, 16, relu), Dense(16, 1))

loss = DecisionRules.simulate_multistage(
    subproblems, state_params_in, state_params_out,
    initial_state, DecisionRules.sample(uncertainty_samples),
    policy
)

# Train end-to-end
DecisionRules.train_multistage(policy, initial_state,
    subproblems, state_params_in, state_params_out, uncertainty_samples;
    num_batches=50, optimizer=Flux.Adam(1e-3)
)
```

## Examples
- **Rocket control**: see `examples/rocket_control/train_dr_rocket.jl` for a full differentiable control workflow (deterministic equivalent and per-stage policies).

## Running tests
```julia
julia --project -e 'using Pkg; Pkg.test()'
```

## Citing DecisionRules
If you use DecisionRules.jl, please cite:
```
@article{rosemberg2024two,
  title={Two-Stage ML-Guided Decision Rules for Sequential Decision Making under Uncertainty},
  author={Rosemberg, Andrew and Street, Alexandre and Vallad{\~a}o, Davi M and Van Hentenryck, Pascal},
  journal={arXiv preprint arXiv:2405.14973},
  year={2024}
}
```
[Presentation](https://www.youtube.com/watch?v=JhgHKYvga2s&list=PLP8iPy9hna6TAzZJvloYK9NBD5qgFJ1PQ&index=14&ab_channel=TheJuliaProgrammingLanguage).

## License
MIT. See `LICENSE` for details.
