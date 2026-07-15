# Stochastic dual dynamic programming

```@meta
CurrentModule = DecisionRules
```

Stochastic dual dynamic programming (SDDP) is the industrial standard for
long-horizon hydrothermal planning and the baseline against which the
case studies in Part III are evaluated. This chapter reviews the
algorithm, makes precise the convexity assumption on which its guarantees
rest, and develops the **inconsistent-formulation** variant used when the
true stage physics is nonconvex — together with the *bound-versus-forward
gap*, the quantity that measures what the convexification gives up. None
of this machinery is implemented in DecisionRules.jl itself (we use
[SDDP.jl](https://github.com/odow/SDDP.jl)); it is presented here because
a fair comparison requires understanding both methods at the same depth.

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
  enters (in hydro, the water-balance rows). Averaging over realizations
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

If the stage physics is **nonconvex** — the flagship example being AC
optimal power flow, whose feasible injection region is nonconvex — both
properties fail: duals of a nonconvex solve are local objects, and a
"cut" built from them can *cut off* the true value function. SDDP as
stated simply does not apply.

## Inconsistent formulations: convex cuts, nonconvex physics

The pragmatic and widely used response is to run the two passes on
**different formulations** of the same stage:

- the **backward pass** (cut generation) uses a **convex relaxation** of
  the stage physics — in the hydrothermal case study, the second-order
  cone relaxation of AC power flow (`SOCWRConicPowerModel`), solved with
  a conic interior-point method (Clarabel);
- the **forward pass** (state sampling and policy simulation) uses the
  **true nonconvex physics** — the AC polar formulation
  (`ACPPowerModel`), solved with a nonlinear interior-point method
  (MadNLP/Ipopt).

We refer to this as SDDP with **inconsistent formulations**. It is
well defined: the relaxed stage problem is convex in the state, so the
cuts are valid *for the relaxed problem*, and the recursion converges on
that surrogate. The forward pass then evaluates the resulting
value-of-water surface against the physics that will actually be
operated. Concretely, the operating policy is

```math
u_t^{\mathrm{SDDP}}(x_{t-1}, w_t) \;\in\;
\arg\min_{(u_t, x_t) \in \mathcal{X}_t^{\mathrm{AC}}(x_{t-1}, w_t)}
\; c_t(x_t, u_t) + \underline{\mathcal{V}}_{t+1}^{\mathrm{SOC}}(x_t) :
```

true AC feasibility inside the stage, *relaxation-priced* future outside
it. (In the reference scripts, this is
`examples/HydroPowerModels/sddp/run_sddp_inconsistent.jl`.)

### What the surrogate misprices

The quality of this policy hinges on how well the relaxed cost-to-go
``\underline{\mathcal{V}}^{\mathrm{SOC}}`` prices the *true* marginal
value of the state. The SOC relaxation is exact on radial, lightly loaded
networks; on meshed networks with high resistance-to-reactance ratios,
binding voltage bands, and spatially concentrated load, it is not — it
systematically **underestimates the cost of delivering power** across the
stressed part of the grid, because it can realize flows the physical
network cannot. A value-of-water surface computed on that surrogate then
misprices storage in exactly the states where storage matters most: the
relaxation "believes" dry-season delivery is cheaper than it is, so it
undervalues the water that would relieve it. Whether the resulting error
is negligible or material is a property of the *network and the operating
regime*, not of the algorithm — the Bolivian case study of Part III sits
deliberately in the regime where the question is live, because that is
the regime real storage-critical systems occupy (see
[The Bolivian interconnected system](@ref)).

## The bound and the forward cost

The inconsistent scheme produces two headline numbers with different
epistemic status:

- ``\underline{z}^{\mathrm{SOC}}`` — the converged **backward bound**: a
  valid lower bound on the expected cost of the *relaxed* multistage
  problem. Because relaxation only widens each stage's feasible set, it
  is also a valid lower bound on the true AC problem — but a *slack* one:
  it is attained (if at all) by relaxed trajectories that no physically
  feasible policy can reproduce.
- ``\hat{z}^{\mathrm{AC}}`` — the **forward simulation cost**: the Monte
  Carlo estimate of the expected cost of the actual operating policy
  under true AC physics.

Their relative difference,

```math
\mathrm{gap} \;=\;
\frac{\hat{z}^{\mathrm{AC}} - \underline{z}^{\mathrm{SOC}}}
     {\underline{z}^{\mathrm{SOC}}},
```

is the **bound-versus-forward gap**. It conflates two contributions that
cannot be separated without further work: ordinary SDDP suboptimality
(finitely many cuts) and the **cost of convexification** — the systematic
error of pricing the future on a relaxed network. On instances where the
relaxation is nearly tight, the gap collapses to the first contribution
and SDDP is close to unbeatable; as the relaxation loosens, the gap grows
and becomes *headroom*: expected cost that a method free of the
convexification assumption is, at least in principle, able to recover.
TS-DDR trains directly on the nonconvex stage problems — its gradient
comes from duals of the *true* AC solves, not from a relaxation — so the
gap is the natural ex-ante measure of how much room such a method has on
a given instance. The hydrothermal case study reports this gap
explicitly for its instance family.

Two disciplines keep the comparison honest, and both are enforced in the
case studies:

1. **Bounds are horizon-specific.** A bound computed on a
   ``T``-stage problem does not bound a ``T' < T``-stage simulation
   metric; training and evaluation horizons must be stated and matched
   (the hydro study trains on 126 stages and evaluates on the first 96
   precisely to keep end-of-horizon effects out of the metric, for every
   method equally).
2. **Policies are compared on the forward metric only.** The only number
   comparable across SDDP, TS-DDR, and any other method is the simulated
   expected cost under identical physics and identical scenarios — hence
   the paired-scenario protocol of the case study.

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
it under the true physics.

## Further reading

- Pereira & Pinto, *Multi-stage stochastic optimization applied to energy
  planning*, Mathematical Programming 52 (1991) — the original SDDP paper.
- Dowson & Kapelevich, *SDDP.jl: a Julia package for stochastic dual
  dynamic programming*, INFORMS Journal on Computing 33 (2021).
- Shapiro, *Analysis of stochastic dual dynamic programming method*,
  EJOR 209 (2011) — convergence analysis and statistical stopping.
- Molzahn & Hiskens, *A survey of relaxations and approximations of the
  power flow equations* (2019) — where and why conic relaxations of AC
  power flow are (in)exact.
