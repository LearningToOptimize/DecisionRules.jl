# Valuing water: two approaches

```@meta
CurrentModule = DecisionRules
```

Every method for the planning problem must, in one way or another, decide what
stored water is worth. Two families do so in opposite ways. Value-function
methods approximate the cost-to-go and derive the decisions from it; SDDP is the
established member of this family and the standard method for hydrothermal
planning. Policy methods restrict the form of the decision and optimize its
parameters; TS-DDR belongs to this family. The case study compares them on the
same problem, with everything except the way water is valued held identical, so
that a difference in cost is a difference between the methods.

## SDDP: a value function built on a convex network

SDDP approximates the expected cost-to-go of each stage by the maximum of a set of
affine functions of the state, its cuts. A forward pass simulates the policy along
sampled inflows and records the reservoir volumes it visits; a backward pass
re-solves each stage at those volumes for every inflow realization and adds a cut
built from the multipliers of the water balance. The value of water is the slope
of the cuts, and the first-stage value under the cuts is a lower bound on the
optimal cost that rises as cuts accumulate.

The construction rests on convexity. When the stage problem is convex, its
cost-to-go is convex in the state and a cut built from the multipliers supports it
from below. Under the AC equations the cost-to-go is nonconvex, and the
multipliers of a locally optimal solution define a linearization that need not
lie below it. Cuts that remain valid for nonconvex stages exist, but they require
solving nonconvex subproblems to global optimality at every backward step. In
practice SDDP is therefore trained on a convex model of the network, and its
cuts are used to operate the real one. The variant used as the baseline here runs
the two passes on different models: the backward pass, which builds the cuts, on
a convex model, and the forward pass, which decides the states at which the cuts
are built, on the AC equations. The cuts are then built at the states visited under AC operation, which yields a
stronger policy than training entirely on the convex model and deploying the
result.

Two convex models are used. The second-order-cone relaxation keeps losses and
reactive power in a relaxed form and is nearly exact on essentially radial
networks such as this one, which makes SDDP with conic cuts a strong benchmark on this system. The DC approximation drops losses,
reactive power and voltages altogether. In both cases water is valued on a network model less constrained than the
physical one, and the cost of the
resulting policy is known only by simulating it under the AC equations. Cut
generation is the stock algorithm of SDDP.jl, with no modification that would
change which multipliers become cuts.

## TS-DDR: a policy trained against the true network

TS-DDR does not build a cost-to-go. A recurrent network reads the inflows observed
so far and the previous target and emits, at every stage, a target volume for each
reservoir inside its one-stage reachable interval (see
[The problem](@ref "One-stage reachable sets")). For a sampled inflow trajectory,
the targets of all stages are imposed as equalities in one deterministic problem
over the horizon, solved on the AC equations, and the multiplier
``\lambda_{r,t}`` of each target equality is the sensitivity of the cost of that
trajectory to the target. These multipliers are prices for water obtained from the AC model as a
by-product of the solve. They are pulled back through the
policy by automatic differentiation, and the policy is updated by stochastic
gradient descent over sampled trajectories. No convex model of the network enters
training or evaluation.

## The gradient must flow through the reachable map

The gradient of the trajectory cost with respect to the policy parameters is

```math
\nabla_\theta \sum_t \bigl\langle \lambda_t,\; \hat v_t(\theta) \bigr\rangle ,
```

and it is only as good as the derivative of the map ``\theta \mapsto \hat v_t``.
That map is not the network followed by a sigmoid. It is

```math
\hat v_{r,t} = \ell_r(v_{t-1}, w_t) + \bigl(u_r(v_{t-1}, w_t) - \ell_r(v_{t-1}, w_t)\bigr)\,\sigma(z_r),
```

followed by the cascade limit, and the bounds ``\ell_r`` and ``u_r`` depend on
the incoming volume, which is the previous stage's target. The derivative of a
target with respect to the previous one therefore passes through the bounds even
when the network output is held fixed, and this term is what carries the effect
of an early target to the later stages. Treating the bounds as constants is a natural simplification, because they
resemble a projection, but it is incorrect: the truncated gradient still lowers
the training loss, yet it points in a different direction, and the error grows
with the horizon. The complete derivative adds little cost, and it should be verified against
finite differences before any long training run. The check perturbs one target at a time, keeps every perturbation
inside the piecewise-affine region of the map so that a kink is never crossed,
and confirms that the derivative is taken of the same map the forward pass
evaluates.

## Two further baselines

Two baselines isolate the parts of TS-DDR. The two-stage linear decision rule
restricts the policy to an affine function of the observed inflows; here it is
trained with the same duality scheme, the same schedule and the same reachable
map as TS-DDR, so that the comparison isolates the functional form of the policy.
Four model-free reinforcement-learning algorithms, REINFORCE, PPO, TD3 and SAC,
act in an environment that wraps the same strict stage problems and the same
reachable map, observe the incoming volumes, the current inflow and the stage,
and are trained on the sampled stage costs. The only difference from TS-DDR is
the learning signal: returns estimated from samples instead of multipliers. The
value of water released in one week materializes many weeks later, and a return-based estimator must learn this credit assignment from noisy
samples, whereas the multiplier
of a target is the local effect of that target on the cost of the whole
trajectory.

## Training, and why it is staged

The policy is trained from a random initialization in phases. Each phase is a
separate process that restarts from the policy the previous phase selected, with
a fresh optimizer state and a fresh learning-rate schedule. The phases move two
quantities in opposite directions. A small number of sampled trajectories per
update gives a cheap and noisy gradient, which is what fast early descent needs,
and a high learning rate lets that descent make progress. A larger sample gives
the precise gradient that the last part of the descent needs, and a lower
learning rate keeps the policy from overshooting it. Combining a large sample with a small learning rate from the start yields a
precise gradient that the optimizer cannot yet exploit, and progress is slow. The schedule is declared as configuration and executed by a
driver, so a run is reproduced by executing the schedule rather than by
following a narrative.

## Selection, and why this is not overfitting

Because a learned policy is chosen among many checkpoints, its reported cost
could be biased toward the scenarios used for the choice. Three rules prevent
this. Checkpoints are
selected on a small fixed panel of inflow scenarios over the reported horizon,
never on the training loss, and an evaluation counts only if every scenario
solved and no load was shed, because a mean over a failed scenario is a mean over
a different denominator. The reported comparison is measured on a different and
much larger protocol of 500 paired scenarios that no checkpoint is ever selected
against; a small panel can be expected to rank policies correctly while being
optimistic about their level, which is why no claim rests on it. And a phase that
produces no selectable policy is excluded from the lineage, but its cost counts
when stating how long the policy took to train.

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
| model the dispatch must satisfy | true AC | true AC |
| model used to value water | convex relaxation or approximation | true AC |
| how the future enters | cutting planes | a learned target |

Above the last two rows the two methods solve the same problem; the last two rows
are the two different answers to what water is worth.
