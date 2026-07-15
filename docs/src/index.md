# DecisionRules.jl

```@meta
CurrentModule = DecisionRules
```

DecisionRules.jl trains parametric decision rules — from affine policies
to deep recurrent networks — for multistage stochastic optimization
problems whose actions come from constrained optimization subproblems
(optimal power flow, MPC, inventory control, …). It implements the
**Two-Stage Deep Decision Rules (TS-DDR)** framework of
[arXiv:2405.14973](https://arxiv.org/abs/2405.14973): the policy outputs
**target states**, a projection subproblem restores exact feasibility, and
Lagrange duals (with implicit differentiation via
[DiffOpt.jl](https://github.com/jump-dev/DiffOpt.jl) where needed) provide
the end-to-end training gradient — no differentiation through solver
iterations, no feasibility violations at deployment.

In the **strict** formulation, the target constraints are hard equalities
and the policy is built to emit only reachable targets: no slack, no
penalty hyperparameter, and the dual ``\lambda_t`` is the pure shadow
price of the target. A GPU companion package,
[DecisionRulesExa.jl](https://github.com/LearningToOptimize/DecisionRulesExa.jl),
trains the same policies through full-horizon deterministic equivalents
with ExaModels + MadNLP/cuDSS.

## The documentation

**Theory.** The
[multistage stochastic optimization problem](@ref "Multistage stochastic optimization")
and where decision rules sit among solution methods;
[the TS-DDR framework](@ref "The TS-DDR framework") — target-state
policies, dual gradients, the training formulations, and strict mode with
its reachability-based feasibility guarantee;
[stochastic dual dynamic programming](@ref "Stochastic dual dynamic programming"),
including the inconsistent-formulation variant for nonconvex stage problems and
the bound-versus-forward gap; and
[extensions](@ref "Extensions: mixed gradients, critics, and risk") —
score-function corrections for integer decisions, control-variate
critics, risk-averse objectives.

**Package guide.** [Getting started](@ref);
[uncertainty sampling formats](@ref "Uncertainty Sampling");
[gradient fallback](@ref "Gradient Fallback");
[GPU acceleration](@ref "GPU Acceleration with DecisionRulesExa.jl");
[API Reference](@ref).

**Case studies.** The flagship is hydrothermal scheduling on the Bolivian
interconnected system, in three chapters —
[the planning problem](@ref "The long-term hydrothermal planning problem")
(multistage AC-OPF with cascaded reservoir dynamics),
[the instance](@ref "The Bolivian interconnected system")
(a real grid whose counter-cyclical, storage-critical operation sits in
the regime where convex value-of-water surrogates misprice the network),
and [the walkthrough](@ref "Hydropower Scheduling")
(strict TS-DDR on CPU and GPU against an SDDP baseline, under a paired
evaluation protocol). Two further studies,
[rocket control](@ref "Rocket Control") and
[stochastic lot-sizing](@ref "Stochastic Lot-Sizing with Fixed Ordering Costs"),
exercise continuous control and mixed-integer recourse.

## Installation

```julia
using Pkg
Pkg.add("DecisionRules")
```

[Getting started](@ref) covers solver requirements, a quick-start example,
and how to choose among the training formulations.

## Citation

```bibtex
@article{rosemberg2024efficiently,
  title={Efficiently Training Deep-Learning Parametric Policies using Lagrangian Duality},
  author={Rosemberg, Andrew and Street, Alexandre and Vallad{\~a}o, Davi M and Van Hentenryck, Pascal},
  journal={arXiv preprint arXiv:2405.14973},
  year={2024}
}
```
