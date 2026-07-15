# Stochastic dual dynamic programming

```@meta
CurrentModule = DecisionRules
```

Stochastic dual dynamic programming (SDDP) is the industrial standard for
long-horizon planning under uncertainty and the baseline against which the
case studies are evaluated — a fair comparison requires both methods at
the same depth. Its guarantees rest on one structural assumption,
convexity of the stage problem in the state; making that assumption
precise leads directly to the **inconsistent-formulation** variant used
when the true stage problem is nonconvex, and to the
*bound-versus-forward gap*, the quantity that measures what the
convexification gives up. (DecisionRules.jl does not implement SDDP; the
baselines use [SDDP.jl](https://github.com/odow/SDDP.jl).)

## Cutting-plane approximation of the cost-to-go

Recall the dynamic-programming recursion of
[Multistage stochastic optimization](@ref): the optimal stage decision
trades immediate cost against the expected cost-to-go
``\mathcal{V}_{t+1}(x_t) := \mathbb{E}_{w_{t+1}}[V_{t+1}(x_t, w_{t+1})]``.
SDDP (Pereira & Pinto, 1991) replaces ``\mathcal{V}_{t+1}`` by a
polyhedral outer approximation built from **cuts**,

```math
\mathcal{V}_{t+1}(x) \;\ge\; \underline{\mathcal{V}}_{t+1}(x)
\;=\; \max_{k = 1, \ldots, K} \;\alpha_k + \langle \beta_k,\, x \rangle ,
```

and iterates two passes over a sampled scenario lattice:

- **Forward pass.** Sample a trajectory ``w_{1:T}``; solve the stage
  problems in sequence with ``\underline{\mathcal{V}}_{t+1}`` in place of
  the true cost-to-go, recording the visited states ``x_t``. The
  accumulated stage costs of many forward passes estimate the expected
  cost of the *current cut policy* — an upper-bound estimator (in
  expectation) for minimization.
- **Backward pass.** At each visited state ``x_t``, re-solve the
  stage-``(t{+}1)`` problems for every uncertainty realization, and read
  the **dual multipliers** of the constraints through which ``x_t``
  enters (the state-coupling rows). Averaging over realizations
  yields a subgradient ``\beta`` of ``\underline{\mathcal{V}}_{t+1}`` at
  ``x_t`` and an intercept ``\alpha`` — a new cut, appended to the model.

The value of the first-stage problem under the current cuts is a valid
**lower bound** on the optimal expected cost, monotonically nondecreasing
as cuts accumulate. Under standard assumptions (finite support,
stagewise independence, relatively complete recourse), the bound and the
forward-cost estimate converge to the common optimal value.

## Where convexity enters

Every step above leans on convexity of the stage problem in the incoming
state ``x_{t-1}``:

1. **Cut validity.** A cut is a supporting hyperplane; it under-estimates
   ``\mathcal{V}_{t+1}`` everywhere only if ``\mathcal{V}_{t+1}`` is
   convex. Convexity of ``V_{t+1}(\cdot, w)`` in the state follows from
   convexity of the stage feasible set and cost — and is *inherited
   backwards* through the recursion.
2. **Dual attainment.** The subgradient ``\beta`` is a Lagrange
   multiplier; strong duality (no duality gap) is what makes the
   multiplier a subgradient of the value function rather than merely a
   local sensitivity.

If the stage problem is **nonconvex** in the state, both properties
fail: duals of a nonconvex solve are local objects, and a "cut" built
from them can *cut off* the true value function. SDDP as stated simply
does not apply.

## Inconsistent formulations: convex cuts, nonconvex stage problems

The pragmatic and widely used response is to run the two passes on
**different formulations** of the same stage:

- the **backward pass** (cut generation) uses a **convex relaxation**
  ``\mathcal{X}_t^{\mathrm{rel}} \supseteq \mathcal{X}_t`` of the stage
  feasible set;
- the **forward pass** (state sampling and policy simulation) uses the
  **true nonconvex stage problem** ``\mathcal{X}_t``.

We refer to this as SDDP with **inconsistent formulations**. It is
well defined: the relaxed stage problem is convex in the state, so the
cuts are valid *for the relaxed problem*, and the recursion converges on
that surrogate. The forward pass then evaluates the resulting
value-function approximation against the stage problem that will
actually be operated. Concretely, the operating policy is

```math
u_t^{\mathrm{SDDP}}(x_{t-1}, w_t) \;\in\;
\arg\min_{(u_t, x_t) \in \mathcal{X}_t(x_{t-1}, w_t)}
\; c_t(x_t, u_t) + \underline{\mathcal{V}}_{t+1}^{\mathrm{rel}}(x_t) :
```

true feasibility inside the stage, *relaxation-priced* future outside
it.

### What the surrogate misprices

The quality of this policy hinges on how well the relaxed cost-to-go
``\underline{\mathcal{V}}^{\mathrm{rel}}`` prices the *true* marginal
value of the state. Relaxation only widens the stage feasible set, so
the surrogate can realize transitions the true system cannot — it
systematically **underestimates the cost of future operation** wherever
the relaxation is loose, and therefore undervalues precisely the states
whose worth derives from relieving that future stress. Whether the
resulting error is negligible or material is a property of the
*instance and its operating regime*, not of the algorithm. The
[hydropower case study](@ref "The Bolivian interconnected system")
works through a concrete mechanism — a conic relaxation of the network
constraints mispricing stored energy — on an instance family chosen to
sit in the regime where the question is live.

## The bound and the forward cost

The inconsistent scheme produces two headline numbers with different
epistemic status:

- ``\underline{z}^{\mathrm{rel}}`` — the converged **backward bound**: a
  valid lower bound on the expected cost of the *relaxed* multistage
  problem. Because relaxation only widens each stage's feasible set, it
  is also a valid lower bound on the true problem — but a *slack* one:
  it is attained (if at all) by relaxed trajectories that no feasible
  policy can reproduce.
- ``\hat{z}`` — the **forward simulation cost**: the Monte Carlo
  estimate of the expected cost of the actual operating policy on the
  true stage problems.

Their relative difference,

```math
\mathrm{gap} \;=\;
\frac{\hat{z} - \underline{z}^{\mathrm{rel}}}
     {\underline{z}^{\mathrm{rel}}},
```

is the **bound-versus-forward gap**. It conflates two contributions that
cannot be separated without further work: ordinary SDDP suboptimality
(finitely many cuts) and the **cost of convexification** — the systematic
error of pricing the future on a relaxed model. On instances where the
relaxation is nearly tight, the gap collapses to the first contribution
and SDDP is close to unbeatable; as the relaxation loosens, the gap grows
and becomes *headroom*: expected cost that a method free of the
convexification assumption is, at least in principle, able to recover.
TS-DDR trains directly on the nonconvex stage problems — its gradient
comes from duals of the *true* stage solves, not from a relaxation — so
the gap is the natural ex-ante measure of how much room such a method has
on a given instance. The hydropower case study reports this gap
explicitly for its instance family.

Two disciplines keep the comparison honest, and both are enforced in the
case studies:

1. **Bounds are horizon-specific.** A bound computed on a
   ``T``-stage problem does not bound a ``T' < T``-stage simulation
   metric; training and evaluation horizons must be stated and matched.
2. **Policies are compared on the forward metric only.** The only number
   comparable across SDDP, TS-DDR, and any other method is the simulated
   expected cost under identical stage problems and identical scenarios —
   hence the paired-scenario protocol of the case studies.

## Complementarity with decision rules

SDDP and TS-DDR occupy dual corners of the design space. SDDP
approximates the *value function* and recovers actions by re-solving a
stage problem at operation time; its strength is a self-certifying bound
and decades of industrial hardening, and its structural commitment is
convexity of the stage model that generates cuts. TS-DDR approximates the
*policy* and needs no convexity — the projection subproblem may be an
arbitrary NLP — but it certifies nothing by itself: its quality is
established empirically, by simulation against a baseline. This is why
the case studies always report both: SDDP supplies the yardstick (a bound
and a strong incumbent policy), and the decision rule is measured against
it on the true stage problems.

## Further reading

- Pereira & Pinto, *Multi-stage stochastic optimization applied to energy
  planning*, Mathematical Programming 52 (1991) — the original SDDP paper.
- Dowson & Kapelevich, *SDDP.jl: a Julia package for stochastic dual
  dynamic programming*, INFORMS Journal on Computing 33 (2021).
- Shapiro, *Analysis of stochastic dual dynamic programming method*,
  EJOR 209 (2011) — convergence analysis and statistical stopping.
