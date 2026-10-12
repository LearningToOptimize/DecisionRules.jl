# Reproducing the study

```@meta
CurrentModule = DecisionRules
```

This page explains how to train a policy for the hydrothermal case and how to
compare it with the baselines, so that a reader can obtain results of the same quality as those reported for the
method, although not the same numbers. The stage problems are nonconvex AC power flows solved to local
optimality, the policy is a deep recurrent network trained by stochastic
gradient descent, and the interior-point solver, the sampled trajectories and
the GPU arithmetic all influence where a run comes to rest. Two runs of the same schedule reach different local optima of similar cost, and
comparisons are meaningful at that resolution. The commands live in the
[examples README](https://github.com/LearningToOptimize/DecisionRules.jl/tree/main/examples)
and in the READMEs of `examples/HydroPowerModels` in this package and in
DecisionRulesExa.jl; this page explains the choices behind them.

## What is fixed

The case is frozen and verified. A manifest records the hash
of every input file and the invariants derived from them (the weekly water
balance, the load factor, the price of shedding load, the empty initial
reservoirs), and both engines refuse to build a stage model from a case that does
not verify. Policies are trained over 126 weekly stages and evaluated over the
first 96, so that the reported window is free of the end-of-horizon dumping of
water that any finite horizon invites.

Evaluation uses one shared protocol of 500 paired inflow scenarios, generated
from a fixed seed rather than stored. Every method is scored on the same
scenarios, so a comparison between any two policies is a paired comparison: the
difference in their costs scenario by scenario has a spread far smaller than the
spread of either cost, which is what makes small differences decidable with a
few hundred scenarios. A policy evaluated by either engine on this protocol is
directly comparable to every other policy evaluated on it.

## Training as a deep-learning problem

Training a policy over a long horizon with nonconvex stage problems is a
difficult deep-learning problem, and it requires the corresponding practices. A
single uninterrupted run with one learning-rate schedule rarely reaches a good
policy. Training is instead organized in phases, each a separate process that
restarts from the checkpoint the previous phase selected, with a fresh optimizer
state and a fresh learning-rate schedule, as described in
[Training, and why it is staged](@ref). Each phase saves the best policy found on
the selection panel and a recovery snapshot of its full state, so an interrupted
phase continues its own trajectory instead of starting over. A restart of the learning-rate schedule temporarily increases the cost before
decreasing it; a stopping rule that does not account for this terminates the
phase prematurely, which is why the stale-evaluation stop is off by default.

The default schedule, declared in `lineage_from_scratch.json` and executed by
`run_tsddr_lineage.jl`, is a good start rather than a tuned optimum. It begins
with a small number of sampled trajectories per update and a high learning rate,
which gives cheap and noisy gradients suited to fast descent, and ends with more
trajectories and a lower learning rate, which gives the precise gradient the last
part of the descent needs. On a different machine, a different budget or a
modified case, the same principle applies with different values: when the
selection metric stops improving, either the step is too large for the precision
of the gradient, and the learning rate should fall, or the gradient is too noisy
for the step, and the sample size should grow.

## The baselines

SDDP is trained by `sddp/run_sddp_inconsistent.jl`, which builds cuts on a convex
relaxation of the network and runs its forward pass on the true AC model. The
default backward model is the SOC-WR relaxation; `DR_SDDP_BACKWARD=dc` builds
the cuts on the linearized DC network instead, solved with Gurobi. SDDP is run
until its bound has stopped moving and its forward costs have settled, and its
policy is then evaluated with the cuts frozen on the paired protocol, exactly
like a learned policy.

The linear decision rule of the two-stage linear decision-rule literature is
trained with the same driver and the same schedule as TS-DDR, from
`lineage_tsldr_from_scratch.json`; only the architecture differs, so the
comparison isolates the value of the nonlinear policy. The reinforcement-learning
baselines in `examples/HydroPowerModels/rl` (REINFORCE, PPO, TD3 and SAC) act in
an environment that wraps the same strict stage problems and the same reachable
target map, and are trained on the stage cost alone. They isolate the value of
the dual gradient as the learning signal. Their environment should be checked
against a TS-DDR checkpoint with the parity mode of `train_rl.jl` before any
training budget is spent.

## Practical lessons

Verify the complete policy gradient against finite differences before a long run.
A truncated gradient still trains and still lowers the loss; it descends in the
wrong direction, and every hyperparameter conclusion drawn on top of it is wrong
too. The check is described in
[The gradient must flow through the reachable map](@ref).

Keep the training signal and the selection signal apart. The training loss is one
noisy sample over the full training horizon; the selection metric is a fixed
panel over the reported horizon, and only the latter should select a checkpoint.
Select on complete evaluations only, because a mean over a scenario that failed
to solve is a mean over a different denominator. Finally, report a policy on a
protocol it was never selected on, and count the phases that produced nothing
when stating how long training took.
