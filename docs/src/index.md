# DecisionRules.jl

```@meta
CurrentModule = DecisionRules
```

DecisionRules.jl is a Julia package for training parametric decision rules for
multistage stochastic programs in which every stage is an optimization model,
such as an optimal power flow or a model-predictive control problem. It
implements two-stage generalized decision rules (TS-GDR) and their deep
instantiation, two-stage deep decision rules (TS-DDR), introduced in
[arXiv:2405.14973](https://arxiv.org/abs/2405.14973). A policy, from an affine
rule to a deep recurrent network built with Flux.jl, maps the observed history of
the uncertainty to a target for the state at the end of every stage; an
optimization problem enforces the targets against the stage models; and the
multipliers of the target constraints provide the gradient with which the policy
is trained by stochastic gradient descent.

The user describes the problem as one JuMP model per stage, marks the incoming
state, the targets and the uncertainty as parameters, and supplies a sampler of
uncertainty trajectories. The package provides three training formulations: the
deterministic equivalent, in which one problem over the whole horizon is solved
per sampled trajectory; the stage-wise rollout, in which stages are solved in
sequence and the gradient is propagated through the realized states with
DiffOpt.jl; and the windowed decomposition, which solves one problem per window
and chains windows through their end states. Targets are enforced either through
a penalized slack, which applies to any model, or, when the states reachable in
one stage are known in closed form, as hard equalities on targets that the
policy is constructed to keep reachable. It also provides recurrent and
state-conditioned policy architectures, dual-gradient extraction for
mixed-integer stage models with an optional score-function correction, and
rollout evaluation with checkpoint selection on fixed scenarios. The companion
package [DecisionRulesExa.jl](https://github.com/LearningToOptimize/DecisionRulesExa.jl)
trains the same policies with the deterministic-equivalent formulation on GPUs,
using ExaModels.jl and MadNLP.jl.

## The documentation

**Part I, Theory**, states the
[multistage stochastic optimization problem](@ref "Multistage stochastic optimization")
and the solution families it admits, develops
[the TS-DDR framework](@ref "The TS-DDR framework") and its strict formulation,
presents [stochastic dual dynamic programming](@ref "Stochastic dual dynamic programming")
as used by the baselines, including its variant for nonconvex stage problems, and
discusses [extensions](@ref "Extensions: mixed gradients, critics, and risk") to
integer decisions, control variates and risk measures.

**Part II, Package guide**, covers [getting started](@ref "Getting started"),
[uncertainty sampling formats](@ref "Uncertainty Sampling"),
[gradient fallback](@ref "Gradient Fallback"),
[GPU acceleration](@ref "GPU Acceleration with DecisionRulesExa.jl"), a guide to
[reading the code](@ref "Reading the code") of both packages and their examples,
and the [API Reference](@ref).

**Part III, Case studies**, documents
[long-term hydrothermal planning](@ref "Long-term hydrothermal planning") on the
Bolivian system,
[stochastic battery-storage AC optimal power flow](@ref "Stochastic battery-storage AC optimal power flow"),
[rocket control](@ref "Rocket Control") and
[stochastic lot-sizing](@ref "Stochastic Lot-Sizing with Fixed Ordering Costs").
The code that reproduces the case studies is in the `examples` directory of this
package and of DecisionRulesExa.jl; its
[README](https://github.com/LearningToOptimize/DecisionRules.jl/tree/main/examples)
is the entry point.

## Installation

```julia
using Pkg
Pkg.add("DecisionRules")
```

[Getting started](@ref) covers solver requirements, a first example, and the
choice among the training formulations.

## Citation

```bibtex
@article{rosemberg2024efficiently,
  title={Efficiently Training Deep-Learning Parametric Policies using Lagrangian Duality},
  author={Rosemberg, Andrew and Street, Alexandre and Vallad{\~a}o, Davi M and Van Hentenryck, Pascal},
  journal={arXiv preprint arXiv:2405.14973},
  year={2024}
}
```
