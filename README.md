<div align="center">

# DecisionRules.jl

<p><sub><b>MULTISTAGE POLICIES · STATE TARGETS · DUAL GRADIENTS</b></sub></p>

[![Build Status](https://github.com/LearningToOptimize/DecisionRules.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/LearningToOptimize/DecisionRules.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/LearningToOptimize/DecisionRules.jl/graph/badge.svg)](https://app.codecov.io/gh/LearningToOptimize/DecisionRules.jl)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://LearningToOptimize.github.io/DecisionRules.jl/dev/)

</div>

<p align="center">
  <picture>
    <source media="(max-width: 640px)" srcset="docs/src/assets/diagrams/tsddr-compact.svg">
    <img src="docs/src/assets/diagrams/tsddr.svg" width="760" alt="TS-DDR training. A sampled uncertainty omega drives per-stage policies pi_t that emit a state target x-hat_t for every stage. One deterministic-equivalent problem over the whole horizon minimizes the stage costs plus a penalty on the slack delta_t, subject to the dynamics, the constraints h, and the target constraints x_t + delta_t = x-hat_t, whose multiplier is lambda. By duality lambda is the gradient of the optimal value Q with respect to the targets, and automatic differentiation multiplies it by the policy Jacobian to update theta.">
  </picture>
</p>
<p align="center"><i>Policies emit a state target for every stage, one deterministic-equivalent problem realizes them, duality prices them, and automatic differentiation carries the price back to the policy weights (redrawn from Fig. 1 of <a href="https://arxiv.org/abs/2405.14973">arXiv:2405.14973</a>).</i></p>

A policy for a constrained multistage system must return actions that satisfy the operating model at every stage.
DecisionRules.jl trains policies whose outputs are state targets, realized by an optimization over the operating
model written in JuMP, with gradients read from the multipliers of the target constraints instead of differentiated
through solver iterations. It implements two-stage generalized decision rules (TS-GDR) and their deep instantiation,
two-stage deep decision rules (TS-DDR), introduced in [arXiv:2405.14973](https://arxiv.org/abs/2405.14973); the
companion package [DecisionRulesExa.jl](https://github.com/LearningToOptimize/DecisionRulesExa.jl) trains the same
policies on GPUs.

## Installation

```julia
using Pkg
Pkg.add("DecisionRules")
```

## What you need to provide

The package is model-first: you describe the problem in JuMP (DiffOpt-enabled for every formulation except the
deterministic equivalent), and the package handles simulation and training.

| Input | What it is |
|:--|:--|
| `subproblems::Vector{JuMP.Model}` | one JuMP model per stage (DiffOpt-enabled) |
| `state_params_in[t]` | parameter variables for the incoming state of stage `t` |
| `state_params_out[t]` | `(target_param, realized_state_var)` pairs of stage `t`: the parameter the policy sets, and the decision variable whose value becomes the realized state |
| `uncertainty_sampler` | returns per-stage samples in the format used by `DecisionRules.sample(...)` |
| policy | a differentiable model (Flux.jl or compatible); input size = uncertainty components per stage + initial-state size, output size = target-state size |

Working patterns are in [`examples/`](examples/).

## Quick start: the deterministic equivalent

A training step samples an uncertainty trajectory $w_{1:T}$, predicts the targets $\hat x_{1:T} = \pi(\cdot;\theta)$,
solves an optimization problem that enforces them against the stage models (with a penalized slack, or as hard
equalities in strict mode), and updates $\theta$ with the multipliers of the target constraints. The deterministic
equivalent, the formulation in the figure above, couples all stages in one problem: build it with
`DecisionRules.deterministic_equivalent!` and train with the matching overload of `train_multistage`.

```julia
using DecisionRules, JuMP, DiffOpt, Flux
using SCS

# 1) Build per-stage subproblems (DiffOpt-enabled) and collect:
#    subproblems, state_params_in, state_params_out, uncertainty_samples

# 2) Build the deterministic equivalent over the full horizon
det = DiffOpt.diff_model(() -> DiffOpt.diff_optimizer(SCS.Optimizer))

det, uncertainty_samples_det = DecisionRules.deterministic_equivalent!(
    det,
    subproblems,
    state_params_in,
    state_params_out,
    Float64.(initial_state),
    uncertainty_samples,
)

# deterministic_equivalent! remaps state_params_in/state_params_out in place.
# Copy those arrays first if you also need the original stage-wise refs later.

# 3) Train a TS-DDR policy end-to-end
num_uncertainties = length(uncertainty_samples[1])  # number of uncertainty components per stage
policy = Chain(
    Dense(DecisionRules.policy_input_dim(num_uncertainties, length(initial_state)), 64, relu),
    Dense(64, length(initial_state)),
)

DecisionRules.train_multistage(
    policy,
    initial_state,
    det,
    state_params_in,
    state_params_out,
    uncertainty_samples_det;
    num_batches=100,
    num_train_per_batch=32,
    optimizer=Flux.Adam(1e-3),
)
```

This mode gives the most faithful gradient signal (full coupling across the horizon), at the price of the largest
inner problem per sample.

## Three ways to couple the horizon

| Formulation | Solved per sampled trajectory | Gradient from |
|:--|:--|:--|
| Deterministic equivalent (direct transcription) | one problem over the whole horizon | multipliers of the target constraints |
| Stage-wise decomposition (single shooting) | one problem per stage, in sequence | multipliers + DiffOpt sensitivities along the rollout |
| Windowed decomposition (multiple shooting) | one problem per window of `window_size` stages, chained by the realized end state | multipliers within a window, sensitivities across windows |

**Stage-wise (single shooting).** Each stage is solved in sequence from the realized state returned by the solver, so
the policy is closed-loop: it receives the realized state $x_{t-1}$ when it predicts $\hat x_t$.

```julia
using DecisionRules, Flux

num_uncertainties = length(uncertainty_samples[1])
policy = Chain(
    Dense(DecisionRules.policy_input_dim(num_uncertainties, length(initial_state)), 64, relu),
    Dense(64, length(initial_state)),
)

DecisionRules.train_multistage(
    policy,
    initial_state,
    subproblems,
    state_params_in,
    state_params_out,
    uncertainty_samples;
    num_batches=100,
    num_train_per_batch=32,
    optimizer=Flux.Adam(1e-3),
)
```

**Windowed (multiple shooting).** Each window solves a deterministic equivalent over its stages and passes the
realized end state to the next one. This can stabilize learning over long horizons compared with single shooting,
while staying cheaper than a full-horizon deterministic equivalent.

```julia
using DecisionRules, Flux, DiffOpt
using SCS

num_uncertainties = length(uncertainty_samples[1])
policy = Chain(
    Dense(DecisionRules.policy_input_dim(num_uncertainties, length(initial_state)), 64, relu),
    Dense(64, length(initial_state)),
)

windows = DecisionRules.setup_shooting_windows(
    subproblems,
    state_params_in,
    state_params_out,
    Float64.(initial_state),
    uncertainty_samples;
    window_size=24,
    model_factory=() -> DiffOpt.nonlinear_diff_model(optimizer_with_attributes(
        SCS.Optimizer,
        "verbose" => 0,
    )),
)

DecisionRules.train_multiple_shooting(
    policy,
    initial_state,
    windows,
    uncertainty_samples;
    num_batches=100,
    num_train_per_batch=32,
    optimizer=Flux.Adam(1e-3),
)
```

## Evaluation: stage-wise rollout and target-violation share

Evaluating a policy only through the deterministic equivalent can overstate its quality: the coupled solve
re-optimizes all stages jointly and absorbs, through the slack penalty, targets that cannot be followed stage by
stage, which deployment cannot do. The stage-wise rollout is the deployment semantics of a target-trajectory policy,
so report it as the headline metric together with a target-violation measure.

The training loops record metrics through a per-sample `SampleLog` cache and a per-batch
`record(sample_log, iter, model)` callback. `RolloutEvaluation` evaluates the policy stage-wise on a fixed held-out
scenario set; call it from `record`:

```julia
using DecisionRules, Random

# Materialize a FIXED held-out evaluation set once, before training.
# Use stage-wise subproblems and parameter refs, not DE-remapped refs.
Random.seed!(1234)
eval_scenarios = [DecisionRules.sample(uncertainty_samples) for _ in 1:8]

rollout_eval = RolloutEvaluation(
    subproblems, state_params_in, state_params_out, initial_state, eval_scenarios;
    stride=25,  # evaluate every 25 batches
    policy_state=:realized,
)

train_multistage(policy, initial_state, det, state_params_in, state_params_out, uncertainty_samples_det;
    num_batches=100,
    record=(sample_log, iter, model) -> begin
        rollout_eval(iter, model)
        return false
    end)
```

| `policy_state` | Fed back to the policy between stage solves | Use it for |
|:--|:--|:--|
| `:realized` (default) | the previous realized optimizer state | the closed-loop, deployment rollout |
| `:target` | the previous target state | comparing with a deterministic-equivalent training loss, which calls the policy on the same target history; log `:realized` separately as the deployment diagnostic |

Each evaluation reports the rollout objective **excluding** the target-slack penalty (the operational cost) and the
**target-violation share**, the realized slack penalty divided by the full objective. Policy comparisons are
trustworthy only when the share is small (≤ ~0.05); a larger share means the targets cannot be followed stage by
stage and the reported cost is not what deployment would realize. When training drives the share to ~0, the
deterministic-equivalent and rollout views are expected to coincide; the rollout metric detects when they don't.

Per-sample debugging hooks attach with `SampleLog(on_sample=(s, models, log) -> ...)`, called after each sample's
solve with the live JuMP model(s). The previous `record_loss=(iter, model, loss, tag) -> ...` keyword keeps working
as a deprecated adapter.

## Strict mode and reachable policies

The standard TS-DDR target constraint uses slack:

```math
x_t + \delta_t = \hat{x}_t,
\qquad
\text{objective} += C_\delta \|\delta_t\|.
```

Slack makes training robust to unreachable targets, but it also makes the dual
signal depend on the target-penalty calibration. Strict mode removes the slack:

```math
x_t = \hat{x}_t.
```

The resulting dual is the clean shadow price of imposing the target. The price
of that cleaner signal is feasibility: every policy target must be reachable
from the state used to condition the policy.

This is automatic in the hydro strict subproblem path because each stage is
solved sequentially and the policy receives the realized previous reservoir
state. It is also possible in regular deterministic equivalents when the target
trajectory is rolled out from the true initial state using a reachable policy:

```math
\hat{x}_0 = x_0,\qquad
\hat{x}_t = \pi_\theta(w_t, \hat{x}_{t-1}),\qquad
\hat{x}_t \in R(\hat{x}_{t-1}, w_t).
```

Every target is then reachable from the state the policy conditioned on, and the
strict equalities make the realized trajectory follow that reachable path. See
[`examples/HydroPowerModels`](examples/HydroPowerModels) and the
DecisionRulesExa.jl companion for the GPU strict regular-DE implementation.

The reachable map $R(\hat x_{t-1}, w_t)$ depends on the state, and that
dependence **must be differentiated**. Treating the interval endpoints as
constants still trains and still lowers the loss while descending a materially
different direction — on the hydro case, a gradient carrying 6% of the true
magnitude and pointing 48 degrees away from it. Verify the complete actor
gradient against finite differences before trusting any hyperparameter
conclusion drawn on top of it.

The policy helpers separate two architectural choices:

- recurrent `layers` / `DR_ENCODER_LAYERS` process uncertainty history only;
- `combiner_layers` / `DR_HEAD_LAYERS` add a nonlinear feed-forward
  state-to-target head without recurrence over the state input.

## GPU acceleration with DecisionRulesExa.jl

When the inner NLP solve is the bottleneck (e.g., AC-OPF with hundreds of buses),
[DecisionRulesExa.jl](https://github.com/LearningToOptimize/DecisionRulesExa.jl) runs the same TS-DDR algorithm
(deterministic-equivalent mode, same envelope-theorem gradient) with
[ExaModels.jl](https://github.com/exanauts/ExaModels.jl) and [MadNLP.jl](https://github.com/MadNLP/MadNLP.jl) + CUDSS:
GPU-native interior-point solves, parallel GPU solves for the samples of a gradient step, and runtime parameter
updates with `ExaModels.set_parameter!` instead of model reconstruction. The
[GPU acceleration](https://LearningToOptimize.github.io/DecisionRules.jl/dev/gpu_acceleration/) page of the
documentation is the tutorial.

## Examples and tests

The case studies of the paper and smaller demonstrations live in `examples/`.
To reproduce a case study, start from [`examples/README.md`](examples/README.md),
which says where each case lives, which engine trains which method, and what to
expect from a run. The [documentation](https://LearningToOptimize.github.io/DecisionRules.jl/dev/)
explains the method, the case studies and how the code is organized. Run the
tests with:

```julia
julia --project -e 'using Pkg; Pkg.test()'
```

## Repository Map

| Path | Purpose |
|---|---|
| `src/DecisionRules.jl` | Module entrypoint and exports |
| `src/dense_multilayer_nn.jl` | MLP helpers, state-conditioned recurrent policies, nonlinear target heads |
| `src/simulate_multistage.jl` | Stage-wise and deterministic-equivalent simulation logic |
| `src/multiple_shooting.jl` | Windowed multiple-shooting setup, simulation, and training |
| `src/utils.jl` | Target-parameter utilities, deficit construction, rollout evaluation |
| `src/integer_strategies.jl` | Strategies for extracting gradients from integer/mixed-integer models |
| `src/score_function.jl` | Score-function gradient correction for nonsmooth/integer problems |
| `src/parameter_duals.jl` | Dual/sensitivity helpers for parameterized JuMP models |
| `docs/src/` | Documenter.jl manual pages |
| `examples/HydroPowerModels/` | Bolivian hydrothermal case study: case, JuMP engine, SDDP and RL baselines |
| `examples/BatteryStorageOPF/` | Battery-storage AC-OPF case study: case construction, JuMP engine, SDDP |
| `examples/inventory_control/` | Inventory-control example and dynamic-programming/SDDP comparisons |
| `examples/rocket_control/` | Rocket MPC/control example |
| `examples/Experimental/` | Research prototypes and robotics/control explorations |
| `test/runtests.jl` | Package test suite |

## Citation

If you use this package in academic work, please cite:

```bibtex
@article{rosemberg2024efficiently,
  title={Efficiently Training Deep-Learning Parametric Policies using Lagrangian Duality},
  author={Rosemberg, Andrew and Street, Alexandre and Vallad{\~a}o, Davi M and Van Hentenryck, Pascal},
  journal={arXiv preprint arXiv:2405.14973},
  year={2024}
}
```

DiffOpt (for differentiating through optimization): https://github.com/jump-dev/DiffOpt.jl

## License

MIT. See `LICENSE`.
