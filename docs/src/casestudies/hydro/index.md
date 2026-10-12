# Long-term hydrothermal planning

```@meta
CurrentModule = DecisionRules
```

This case study applies TS-DDR to the long-term hydrothermal dispatch of the
Bolivian interconnected system, with every stage subject to the AC power-flow
equations, and compares it with stochastic dual dynamic programming, with
two-stage linear decision rules, and with four model-free reinforcement-learning
algorithms. All policies are trained on the same model and evaluated on a common
set of paired inflow scenarios. The case, the JuMP and ExaModels engines, the
training schedules, the baselines and the evaluation protocol are distributed with
the examples of this package and of DecisionRulesExa.jl.

## Motivation

Long-term hydrothermal scheduling determines, at weekly stages over a horizon of
several years, the releases of the hydro plants and the dispatch of the thermal
units that minimize the expected operating cost under uncertain inflows. Hydro
generation has no direct cost; its economic value is the opportunity cost of the
stored water, the expected thermal cost it displaces in later stages. The problem
is the one for which SDDP was developed, and SDDP remains the method used in the
operational planning of large hydro-dominated systems. Its cuts are valid lower approximations of the
cost-to-go only when the stage problems are convex. The AC power-flow equations
are not, so in practice the cost-to-go is computed on a convex relaxation or
approximation of the network while decisions are taken on the physical one. The
resulting mismatch between planning and operation appears as unexpected
reservoir depletion and price spikes, and the convex model that yields the best
policy is not known before each candidate has been trained and simulated.

The Bolivian system is chosen so that this mismatch is small. Its transfers
follow essentially radial corridors, and on radial networks the second-order-cone
relaxation of the power-flow equations is known to be accurate. SDDP with conic cuts is therefore a strong
benchmark on this system, and the case study measures how closely a policy trained
against the AC equations, with no convex model in its training, approaches it. The
linear decision rule is trained with the same gradient, schedule and feasibility
map as TS-DDR and isolates the role of the functional form of the policy; the
reinforcement-learning agents act on the same stage problems and isolate the role
of the dual gradient as the learning signal.

## Organization

[The problem](@ref "The long-term hydrothermal planning problem") states the
reservoir dynamics, the AC network model, the planning problem and the stage
problem solved by a target-state policy, and describes the Bolivian instance.
[Valuing water: two approaches](@ref) presents SDDP with inconsistent
formulations and TS-DDR, the two further baselines, the training schedule and the
selection protocol, and lists what is held identical across methods.
[Reproducing the study](@ref) describes how the policies are trained and
evaluated with the distributed code, and the [Walkthrough](@ref "Walkthrough")
builds the stage problems, the policy and a few gradient steps on a short horizon
on the CPU. The entry point to the code is the
[examples README](https://github.com/LearningToOptimize/DecisionRules.jl/tree/main/examples).
