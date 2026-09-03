# Multistage stochastic optimization

```@meta
CurrentModule = DecisionRules
```

Everything downstream — the TS-DDR framework, the SDDP baseline, the case
studies — is an attempt to solve one problem: sequential decision making
under uncertainty, with actions constrained by an optimization-level
feasible set. What follows fixes
notation, recalls the dynamic-programming view and why it is intractable
in general, and places **decision rules** among the classical solution
families; readers fluent in multistage stochastic programming can skim to
[Decision rules](@ref decision-rules-sec) and continue with
[The TS-DDR framework](@ref).

## Problem statement

Consider a sequential decision problem over a finite horizon of ``T``
stages. At each stage ``t = 1, \ldots, T``:

1. an exogenous **uncertainty realization** ``w_t \in \mathcal{W}_t`` is
   revealed (river inflows, demands, prices, disturbances);
2. the decision maker, knowing the current **state** ``x_{t-1}`` and the
   realization ``w_t`` (and, in general, the whole history
   ``w_{1:t} = (w_1, \ldots, w_t)``), chooses a **control**
   ``u_t`` and a next state ``x_t``;
3. the pair must satisfy the stage feasibility constraints,
   ``(u_t, x_t) \in \mathcal{X}_t(x_{t-1}, w_t)``, which encode both the
   **dynamics** (how the state evolves) and the **static constraints** of
   the stage (a network, a budget, a capacity — whatever the application
   imposes within a single period);
4. a **stage cost** ``c_t(x_t, u_t)`` is incurred.

The objective is to choose a *policy* — a rule for making each decision
from the information available when it must be made — minimizing expected
total cost:

```math
\min_{\pi \in \Pi} \;
\mathbb{E}_{w_{1:T}} \left[ \sum_{t=1}^{T} c_t\bigl(x_t^\pi, u_t^\pi\bigr) \right]
\qquad \text{s.t.} \quad
\bigl(u_t^\pi, x_t^\pi\bigr) \in \mathcal{X}_t\bigl(x_{t-1}^\pi, w_t\bigr)
\;\; \forall t,
```

where ``\Pi`` is the set of **nonanticipative** policies: ``u_t^\pi`` may
depend on ``w_{1:t}`` but not on future realizations ``w_{t+1:T}``.
Nonanticipativity is what makes the problem *stochastic control* rather
than a family of deterministic problems — every decision is a hedge
against a distribution of futures, committed before those futures are
revealed.

Two structural features of this formulation deserve emphasis, because the
solution methods differ precisely in how they treat them:

- **Intertemporal coupling through the state.** The only channel through
  which stage ``t`` affects stage ``t+1`` is ``x_t``. A resource stored in
  the state (energy in a battery, inventory on a shelf, fuel in a tank)
  has an *opportunity cost* — the expected future cost avoided by carrying
  it forward — that no single-stage view can price.
- **Constrained actions.** The feasible set ``\mathcal{X}_t`` is itself an
  optimization-level object — possibly a full nonconvex program, as in
  the battery-storage AC-OPF case study. Any learned policy must produce
  decisions that *satisfy it exactly*, not approximately.

## The dynamic-programming recursion

Under the Markovian assumption that ``(x_{t-1}, w_t)`` summarizes the
history (stagewise-independent ``w_t``, or an augmented state otherwise),
the problem admits Bellman's recursion. Define the **cost-to-go**
(or *value*) **function** at the end of stage ``t``:

```math
V_t(x_{t-1}, w_t) \;=\;
\min_{(u_t, x_t) \in \mathcal{X}_t(x_{t-1}, w_t)}
\; c_t(x_t, u_t) + \mathbb{E}_{w_{t+1}}\bigl[ V_{t+1}(x_t, w_{t+1}) \bigr],
```

with ``V_{T+1} \equiv 0``. The optimal policy acts greedily against the
expected cost-to-go: at each stage it trades the immediate cost
``c_t`` against the expected future cost
``\mathbb{E}[V_{t+1}(x_t, \cdot)]`` of the state it leaves behind. In
resource-storage problems this expected cost-to-go *is* the "value of
water" (or of inventory): its negative gradient with respect to the stored
quantity is the marginal price at which storing beats releasing.

The recursion is conceptually complete and computationally hopeless in
general: ``V_t`` is a function on the full state space, and any grid-based
representation grows exponentially with the state dimension — Bellman's
*curse of dimensionality*. Every practical method is a way of
approximating either the value function or the policy.

## Solution families

Three broad families dominate practice; the third is the one this package
implements.

### Scenario trees and the deterministic equivalent

Discretize the uncertainty into a finite **scenario tree** and attach one
copy of the decision variables to every node. The result is a single —
typically enormous — mathematical program, the **deterministic
equivalent** (DE), whose solution is exact *for the tree*. The tree grows
exponentially in ``T``, so pure scenario-tree methods are confined to
short horizons or coarse discretizations. The DE returns in
[Three training formulations](@ref) in a different role — not as a
solution method but as a *differentiable training oracle* for a policy,
evaluated one sampled trajectory at a time, which sidesteps the
exponential growth entirely.

### Value-function approximation: SDDP

**Stochastic dual dynamic programming** (SDDP) exploits convexity: when
each stage problem is convex in ``x_{t-1}``, the cost-to-go
``\mathbb{E}[V_{t+1}]`` is convex and can be outer-approximated by
supporting hyperplanes ("cuts") generated from stage duals. SDDP is the
workhorse of long-horizon planning under uncertainty and the baseline the
case studies compare against;
[Stochastic dual dynamic programming](@ref) develops it in detail —
including what must be done, and what is silently given up, when the true
stage problem is *nonconvex*.

### [Decision rules](@id decision-rules-sec)

The third family approximates the **policy** directly: restrict ``\Pi`` to
a parametric class

```math
u_t = \pi_\theta\bigl(w_{1:t}, x_{t-1}\bigr), \qquad \theta \in \Theta,
```

and optimize over the finite-dimensional parameter ``\theta`` instead of
over the space of all measurable policies. Nonanticipativity holds *by
construction* — the rule only ever reads the history. The classical
instance is the **linear decision rule** (LDR/affine policy), where
``\pi_\theta`` is affine in the observations: tractable, sometimes
provably near-optimal, but limited in expressiveness. Replacing the affine
map with a deep network gives a **deep decision rule** with the opposite
profile: expressive, but raising two difficulties that the naive
"learn a network that outputs actions" approach does not survive in
constrained physical systems:

1. **Feasibility.** A network output has no reason to satisfy
   ``\mathcal{X}_t`` — and in operations, constraint violation is not a
   soft error. Penalizing violations reintroduces exactly the kind of
   hyperparameter tuning decision rules were supposed to avoid.
2. **Gradient signal.** If feasibility is enforced by an optimization
   layer, training requires differentiating through a solver — expensive
   and fragile if done by unrolling or generic implicit differentiation at
   scale.

[The TS-DDR framework](@ref) resolves both at
once: the network outputs *target states* rather than actions, a
projection subproblem restores feasibility exactly, and Lagrangian duality
supplies the training gradient at the cost of the solve itself. The
[strict variant](@ref "Strict mode: penalty-free gradient signal")
sharpens this further when targets can be guaranteed reachable by
construction.

## What "solving" means: bounds and simulation

Because all practical methods approximate, empirical comparisons rest on
two complementary quantities, used throughout the case studies:

- A **lower bound** (for minimization): SDDP's cut model provides a valid
  lower bound on the expected cost *of the problem its cuts actually
  model*. When the cut model is a convex relaxation of a nonconvex stage
  problem,
  the bound is a bound on the *relaxed* problem — an important subtlety
  developed in [The bound and the forward cost](@ref).
- A **simulation (forward) cost**: the expected cost of a concrete policy,
  estimated by rolling it out on the *true* stage problems over sampled
  scenarios. This is the only number that treats every method — cuts,
  linear rules, deep rules — on identical footing, and it is the primary
  metric of the case studies (see the
  paired evaluation protocol in the
  [battery-storage AC-OPF study](@ref "Stochastic battery-storage AC optimal power flow")).

The gap between the two jointly measures the suboptimality of the policy
*and* the fidelity of the model used to bound it — and keeping those two
contributions separate is a recurring theme, made precise for SDDP in
[The bound and the forward cost](@ref).

## Further reading

- Shapiro, Dentcheva, Ruszczyński, *Lectures on Stochastic Programming*
  (SIAM) — the standard reference for the general theory.
- Bertsekas, *Dynamic Programming and Optimal Control* — the
  control-theoretic view of the same recursion.
- Ben-Tal et al., *Adjustable robust solutions of uncertain linear
  programs* (2004) — the origin of affine decision rules.
- Rosemberg, Street, Valladão, Van Hentenryck,
  [*Efficiently Training Deep-Learning Parametric Policies using
  Lagrangian Duality*](https://arxiv.org/abs/2405.14973) — the TS-DDR
  paper this package implements.
