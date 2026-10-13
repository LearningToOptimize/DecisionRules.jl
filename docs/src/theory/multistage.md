# Multistage stochastic optimization

```@meta
CurrentModule = DecisionRules
```

Multistage stochastic programming models sequential decisions under uncertainty
in which information is revealed progressively and each decision may depend only
on the information available when it is taken. Its applications include the
scheduling of hydrothermal systems, for which much of the computational
methodology was developed, the operation of energy storage, asset and
liability management, and inventory and supply-chain planning. In all of them the
stages are coupled through a state, such as stored water, stored energy or
inventory, and the economic content of the problem lies in the value of carrying
that state forward. This chapter fixes the notation used throughout the
documentation, recalls the dynamic-programming formulation, reviews the main
families of solution methods, and states how policies are evaluated.

## Problem statement

Consider a horizon of ``T`` stages. At the beginning of stage ``t`` an exogenous
random vector ``w_t`` is observed; the decision maker, knowing the incoming state
``x_{t-1}`` and the history ``w_{[t]} = (w_1, \ldots, w_t)``, chooses controls
``u_t`` and the outgoing state ``x_t``, subject to

```math
(u_t, x_t) \in \mathcal{X}_t(x_{t-1}, w_t)
= \bigl\{ (u_t, x_t) : x_t = \mathcal{T}_t(x_{t-1}, w_t, u_t),\;
          h_t(x_t, u_t, w_t) \ge 0 \bigr\},
```

where the transition ``\mathcal{T}_t`` describes the dynamics of the state and
``h_t`` collects the constraints internal to the stage, for instance the
power-flow equations of a network, which may be nonconvex. A stage cost
``c_t(x_t, u_t)`` is incurred. A policy ``\pi`` is a sequence of maps from the
information available at each stage to a feasible decision, and it is
nonanticipative if the decision at stage ``t`` depends on ``w_{[t]}`` and not on
later realizations. The problem is

```math
\min_{\pi \in \Pi} \;
\mathbb{E}\Bigl[ \sum_{t=1}^{T} c_t\bigl(x_t^\pi, u_t^\pi\bigr) \Bigr]
\quad \text{s.t.} \quad
\bigl(u_t^\pi, x_t^\pi\bigr) \in \mathcal{X}_t\bigl(x_{t-1}^\pi, w_t\bigr), \;\; t = 1, \ldots, T,
```

over the set ``\Pi`` of nonanticipative policies. Two features of this
formulation distinguish the solution methods discussed below. The stages
interact only through the state, so the effect of a decision on the future is
summarized by the state it leaves behind. And the feasible set of a stage is
itself an optimization model, so any policy must return decisions that satisfy
it exactly; in operational applications a violated constraint is not a small
error but an infeasible operation.

## The dynamic-programming formulation

When the uncertainty is stagewise independent, or when the state is augmented so
that ``(x_{t-1}, w_t)`` summarizes the relevant history, the problem admits the
recursion

```math
V_t(x_{t-1}, w_t) = \min_{(u_t, x_t) \in \mathcal{X}_t(x_{t-1}, w_t)}
\; c_t(x_t, u_t) + \mathcal{V}_{t+1}(x_t),
\qquad
\mathcal{V}_{t+1}(x_t) = \mathbb{E}\bigl[ V_{t+1}(x_t, w_{t+1}) \bigr],
```

with ``\mathcal{V}_{T+1} \equiv 0``. The expected cost-to-go ``\mathcal{V}_{t+1}``
prices the outgoing state, and its negative gradient with respect to a stored
quantity is the marginal value of that quantity: the water value of hydrothermal
scheduling, or the value of charge of a battery. An optimal policy trades the
immediate cost against this value at every stage. The recursion is exact but, in
general, intractable: the cost-to-go is a function on the whole state space, and
any representation of it on a grid grows exponentially with the dimension of the
state. Practical methods approximate either the
value function or the policy.

## Solution approaches

### Scenario trees and the deterministic equivalent

If the uncertainty is discretized into a scenario tree, the problem becomes a
single mathematical program with one copy of the decision variables per node, the
deterministic equivalent, whose solution is optimal for the tree. The tree grows exponentially with the number of stages, so the
approach is restricted to short horizons or coarse discretizations. The
deterministic equivalent of a single sampled trajectory, by contrast, has the
size of one scenario, and it reappears in
[Three training formulations](@ref) as the problem through which a decision rule
is evaluated and differentiated.

### Value-function approximation

Stochastic dual dynamic programming approximates each
expected cost-to-go from below by the maximum of affine functions, or cuts,
generated from the multipliers of the stage problems along sampled trajectories.
Under convexity of the stage problems, stagewise independence and relatively
complete recourse, the method converges, and it has become the standard tool of
long-term hydrothermal planning. Time-dependent uncertainty can be accommodated
by a Markov chain or a lattice on which the cuts are indexed, and integer state
variables by Lagrangian cuts, at the cost of solving the stage problems to global
optimality. For
nonconvex continuous stage problems, such as those defined by the AC power-flow
equations, the cuts are in practice computed on a convex model of the stage.
[Stochastic dual dynamic programming](@ref) develops the method and the
consequences of that choice.

### [Decision rules](@id decision-rules-sec)

Decision rules restrict the policy to a parametric family and optimize its
parameters. For a rule
``x_t = \pi_t(w_{[t]}, x_0; \theta)`` that reads only the observed history,
nonanticipativity holds by construction, and the problem reduces to an
optimization over the finite-dimensional parameter ``\theta``. Affine rules make
this reduction tractable for linear programs and robust optimization, and their
suboptimality can be bounded by solving primal and dual approximations; nonlinear
rules can be obtained by lifting the uncertainty into a higher dimension.
Two-stage linear decision rules apply the rule to the state variables only, so that,
for fixed parameters, the remaining decisions form a deterministic problem per
scenario and training the rule is a two-stage stochastic program. Decision rules
require neither stagewise independence nor convexity of the cost-to-go, and they
return an explicit policy that is evaluated by solving one stage problem per
stage.

Replacing the affine map by a deep network raises two difficulties. A network
output has no reason to be feasible, and penalizing infeasibility introduces
coefficients that must be tuned and a policy that may still violate constraints.
And if feasibility is restored by an optimization problem, training requires the
derivative of that problem's solution with respect to the network output.
[The TS-DDR framework](@ref) addresses both: the network emits target states, an
optimization problem enforces them against the stage models, and the multipliers
of the target constraints give the derivative of the cost with respect to the
targets. When the targets can be restricted to the states reachable in one stage,
the [strict formulation](@ref "Strict mode: penalty-free gradient signal")
imposes them as equalities and removes the penalty altogether.

### Approximate dynamic programming and reinforcement learning

Approximate dynamic programming and reinforcement learning fit a value function,
a policy, or both, from simulated transitions. Model-free methods use the optimization
model of the stage only as a simulator: they estimate the effect of a decision on
the future from sampled returns, and they must learn feasibility from penalties or
enforce it by projection. Credit assignment over long horizons and hard
constraints are the main obstacles to applying them to the operation of physical
systems.

## Evaluating a policy: bounds and simulation

Since every practical method approximates, comparisons rest on two quantities. A
lower bound on the optimal expected cost is available from value-function methods:
the first-stage value of SDDP under its cuts bounds the optimal cost of the
problem the cuts were built on, which, when the cuts were computed on a convex
model of a nonconvex stage, is the convex model and not the original problem
([The bound and the forward cost](@ref)). The expected cost of a policy is
estimated by simulating it on the true stage problems over sampled scenarios.
Simulation is the only quantity that treats value-function methods, linear rules
and deep rules on the same footing, and it is the metric of the case studies,
which evaluate every policy on a common set of paired scenarios.
