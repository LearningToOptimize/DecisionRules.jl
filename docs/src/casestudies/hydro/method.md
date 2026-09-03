# Valuing water: two approaches

```@meta
CurrentModule = DecisionRules
```

Both methods solve the same planning problem and differ in exactly one respect:
how they decide what stored water is worth. Everything else — the network, the
demand, the inflow scenarios, the cost of shedding load, the horizon, the solver
tolerances, and the requirement that the reported dispatch satisfy the true AC
equations — is held identical, so the measured difference is attributable to the
method rather than to the setup.

## SDDP: build the value function, but convexify to do it

Stochastic dual dynamic programming approximates the future cost of leaving
water behind by an outer envelope of cutting planes, refined by sweeping forward
and backward through the horizon. It is the standard method for this problem and
it converges to a genuine lower bound on the cost.

The bound is only valid if each stage problem is **convex in the incoming
state**, which AC power flow is not. The universal practical compromise, used
here, is to run the backward pass — the one that generates cuts — on a
second-order-cone relaxation of the network, and to simulate the resulting policy
forward on the true AC model.

That compromise has a price, and naming it is the point of comparing at all. The
value of water is computed against a network that is easier to deliver power
across than the real one. Where the relaxation is loose — meshed corridors,
binding voltage limits, load far from generation — delivery is underpriced, and
storage decisions inherit the mispricing exactly where they matter most.

Nothing else about the baseline is tuned in TS-DDR's favour or against it: cut
generation is stock SDDP, with no regularisation ladder and no retry-until-optimal
loop that would change which duals become cuts.

## TS-DDR: skip the value function, read the price off the solver

TS-DDR trains a policy that emits a **target** reservoir level for each stage.
The stage problem is then solved with the outgoing storage pinned to that target
by a hard equality — no slack, no penalty term. Two things follow.

First, the dual of that equality is precisely ``\partial Q_t / \partial \hat v_t``:
the marginal value of water, delivered by the solver as a by-product of solving
the stage. There is no value function to construct, and therefore nothing to
convexify. Training and evaluation both use the **true AC model** end to end.

Second, hard equalities are only well posed if every target the policy emits is
actually attainable in one stage. That is what the reachable policy guarantees:
a network output is mapped into the one-stage reachable interval of each
reservoir, and then clamped down the cascade so a downstream unit can never be
told to hold water that the unit above it did not release. The reachable
interval itself is a property of the water balance and is derived in
[The problem](@ref "One-stage reachable sets"); what matters for training is that
its endpoints depend on the state, and that the derivative has to go through
them. That is the next section.

## The gradient must flow through the reachable map

Strict mode makes the stage a projection: the policy emits a target
``\hat v_t`` and the stage problem is solved with ``v_t = \hat v_t`` enforced as
an equality. The multiplier ``\lambda_t`` of that equality is exactly
``\partial Q_t / \partial \hat v_t`` — the marginal value of water, delivered by
the solver at no extra cost. TS-DDR's actor gradient is then

```math
\nabla_\theta \; \sum_t \bigl\langle \lambda_t,\; \hat v_t(\theta) \bigr\rangle ,
```

so everything hinges on differentiating the map ``\theta \mapsto \hat v_t``
**completely**. That map is not just the network and a sigmoid: it is

```math
\hat v_{r,t}
  \;=\; \ell_r(v_{t-1}, w_t) \;+\;
        \bigl(u_r(v_{t-1}, w_t) - \ell_r(v_{t-1}, w_t)\bigr)\,\sigma(z_r),
```

followed by the cascade clamp. The bounds carry the state, so
``\partial \hat v_t / \partial v_{t-1}`` is nonzero *through them* even when the
network output ``z`` is held fixed — and since ``v_{t-1}`` is itself the previous
stage's target, this term is precisely what couples the stages.

This is worth spelling out because getting it wrong is silent. Declaring the
bounds non-differentiable still produces a gradient, still trains, and still
reduces the loss; it simply descends a different direction. Measured on this
case over the full horizon:

| | truncated | complete |
|---|---|---|
| ``\cos(\nabla_{\text{AD}}, \nabla_{\text{FD}})`` | 0.93 | **1.000000** |
| ``\|\nabla_{\text{AD}}\| / \|\nabla_{\text{FD}}\|`` | 0.059 | **1.000000** |
| ``\|\nabla\|`` at the same point | 28,991 | **491,674** |

The truncated gradient is a 17-times-too-short vector pointing 48 degrees off.
The practical consequence was not a failure to train but a *wrong conclusion
about the method*: with a gradient carrying 6% of the magnitude, raising the
learning rate could not help, and a learning-rate sweep duly reported "learning
rate is not the lever". On the repaired gradient the ordering inverts and the
learning rate becomes the dominant lever. **Verify the complete actor gradient
against finite differences before spending a campaign on hyperparameters.**

Two properties make the finite-difference check trustworthy here. The map is
piecewise affine in ``z``, so a difference taken across a kink is meaningless;
the check therefore measures the distance to the nearest kink and asserts that
the perturbation stays inside it. And the forward map must be *unchanged* by the
repair — a gradient fix that moves the policy's output is a different policy, so
the forward value is pinned bit-for-bit before the derivative is compared.

## Training, and why it is staged

The policy is trained from a random initialisation. Training runs in **phases**,
each a separate process that restarts from the policy the previous phase
selected. A restart is the point, not an artefact: the optimiser state, the
learning-rate schedule and its warm-up all begin again.

The phases move two knobs in opposite directions:

| phase | sampling per gradient step | learning rate | role |
|---|---|---|---|
| 1 | low | high | bulk descent — a noisy, cheap gradient is enough to make fast progress |
| 2 | low | high | continued descent from a better initialisation |
| 3 | raised | dropped | convergence — a precise gradient and a small step |

The rule behind this is worth stating because it generalises: a **small sample
gives a noisy but cheap gradient**, which is what bulk descent wants; a **large
sample gives a precise one**, which is what final convergence wants. Pairing a
large sample with a small learning rate from the start is the flat quadrant — it
buys precision the optimiser cannot yet use and makes almost no progress.

The schedule is declared as configuration and executed by a driver, so the
published run is reproduced by running the declared schedule rather than by
following a narrative.

## Selection, and why this is not overfitting

Learned policies invite a fair suspicion: that the reported number is the best of
many attempts on the data it was chosen with. Three properties of this study are
designed to answer it.

**Checkpoints are selected on a small fixed panel, never on the training loss.**
The panel is a handful of scenarios with common random numbers, evaluated over
the reported horizon. Two requirements are enforced in code rather than by
convention: the evaluation must be **complete** — a mean over a scenario that
failed to solve is a mean over a different denominator, and no tolerance makes
that comparable — and it must show **no load shedding**.

**The published claim is measured on a different, much larger protocol** — 500
paired scenarios that no checkpoint was ever selected against. The selection
panel turns out to have been directionally right and slightly optimistic about
the level, which is what a small screening set should be expected to be, and is
why the claim does not rest on it.

**The discarded work is reported.** One additional phase was attempted and
produced no selectable policy; it is excluded from the lineage and its cost is
included in the honest accounting of how long the result took. A time-to-policy
figure that quietly omits the attempts that failed is not a time-to-policy
figure.

## What is held identical

| | SDDP | TS-DDR |
|---|---|---|
| network, hydro topology, inflow scenarios | same | same |
| demand profile | same | same |
| uncertainty | inflow only | inflow only |
| initial reservoir state | same | same |
| water balance | same | same |
| price of shedding load | same | same |
| reactive balance | hard, no slack | hard, no slack |
| branch limits | apparent power, both ends | apparent power, both ends |
| horizon simulated and reported | same | same |
| evaluation scenarios | the shared paired protocol | the same scenarios |
| **model the dispatch must satisfy** | **true AC** | **true AC** |
| | | |
| model used to *value water* | convex relaxation | true AC |
| how the future enters | cutting planes | a learned target |

The line in the middle is the whole experiment. Above it, the two are the same
problem; below it, they are two different answers to what water is worth.
