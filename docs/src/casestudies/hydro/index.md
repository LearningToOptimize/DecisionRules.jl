# Long-term hydrothermal planning

```@meta
CurrentModule = DecisionRules
```

A hydrothermal power system is operated by deciding, every week for years, how
much water to release and how much fuel to burn. Water is free but finite; fuel
is expensive but available. The whole problem is the price of water — a price
that no market quotes and that has to be inferred from what the water will be
worth later, elsewhere on the network, under inflows nobody has seen yet.

This case study puts two ways of inferring that price against each other on a
real system, under the full nonconvex AC power flow, on identical inflow
scenarios.

## The question

**Stochastic dual dynamic programming** is the standard answer, and a very good
one. It builds an explicit value function from cutting planes. But cuts are only
valid if the stage problem is convex, and AC power flow is not — so in practice
the value of water is computed against a *relaxed* network and then applied to
the real one.

**TS-DDR** needs no such relaxation. It trains a policy that outputs a target
reservoir level for each stage; the stage problem projects that target onto the
true AC feasible set, and the multiplier of the target constraint *is* the
marginal value of water, handed over by the solver at no extra cost. There is no
value function to build and nothing to convexify.

So: **can a policy learned this way operate a real system as well as a converged
SDDP policy?**

## The answer

Trained from random initialisation in about eleven GPU-hours, and evaluated
against SDDP on 500 shared inflow scenarios under true AC physics:

| | operating cost |
|---|---|
| SDDP | **313,546** |
| TS-DDR, from scratch | **314,023** |

A difference of **+0.152%** — statistically unambiguous, practically small, and
in SDDP's favour. Not a tie, and not a win: the [Results](@ref) page says so in
those words and refuses the three obvious overstatements.

The interesting part is not the number but the mechanism. The learned policy
**under-hedges**: it carries less water than SDDP, runs cheaper for most of the
horizon, and pays the difference back in the closing weeks when it arrives short.
That is visible stage by stage, in storage, in thermal dispatch, and in the
marginal price of energy — which is what makes the result diagnosable rather
than merely reported.

## How to read this case study

| page | what it covers |
|---|---|
| [The problem](@ref "The long-term hydrothermal planning problem") | the planning problem itself: reservoir dynamics, cascades, AC network physics, and why the value of water is both locational and temporal |
| [Valuing water: two approaches](@ref) | how SDDP and TS-DDR each arrive at a price for water, what each assumes, and what is held identical so the comparison is about the methods |
| [Results](@ref "Results") | the measured comparison, its statistics, the physical mechanism behind the difference, and an honest reading |
| [Walkthrough](@ref "Walkthrough") | a runnable, few-minute version on CPU: build the stage problems, construct the policy, roll out, read the value of water, take some gradient steps |

Everything needed to reproduce the published numbers — the case, both trained
policies, the evaluation protocol and the figures — ships with
`examples/HydroPowerModels` in this package and its companion,
DecisionRulesExa.jl. Their READMEs carry the commands.
