# The long-term hydrothermal planning problem

```@meta
CurrentModule = DecisionRules
```

**Long-term hydrothermal dispatch (LTHD)** is the coordinated operation of
hydro reservoirs and thermal generation on an AC transmission network over
a multi-year horizon under inflow and demand uncertainty — an instance of
the general problem of [Multistage stochastic optimization](@ref) with the
state given by stored water, the uncertainty by river inflows, and the
stage feasibility set by a nonconvex AC optimal power flow. The system the
case study runs on is presented in
[The Bolivian interconnected system](@ref); the training and evaluation
walkthrough is [Hydropower Scheduling](@ref).

## Stages, state, uncertainty, decisions

Time is discretized into **weekly stages** ``t = 1, \ldots, T`` (each stage
represents 168 hours of operation; the case study trains on ``T = 126``
stages, roughly two and a half years). At each stage:

- **State** — the vector of reservoir volumes
  ``x_t = (v_{r,t})_{r \in \mathcal{R}} \in \mathbb{R}^{n_{\mathrm{hyd}}}``,
  the only quantity carried between stages.
- **Uncertainty** — the vector of river inflows
  ``w_t = (w_{r,t})_{r \in \mathcal{R}}``, revealed at the start of the
  stage. Inflows are strongly seasonal and spatially correlated across the
  basin, so realizations are drawn as *joint* scenarios (see
  [Uncertainty Sampling](@ref)); demand may follow a deterministic
  seasonal profile or be sampled as an additional uncertainty.
- **Decisions** — the stage dispatch ``u_t``: thermal generation
  ``p_{g,t}`` (and reactive ``q_{g,t}``), turbined outflow ``q_{r,t}``,
  spillage ``s_{r,t}``, load-shedding (deficit) variables, and the AC
  power-flow variables (bus voltage magnitudes and angles).

## Reservoir dynamics and cascades

Volumes evolve by **water balance**. Rivers form **cascades**: water
released by an upstream plant arrives at its downstream neighbour within
the same weekly stage (travel times are short relative to the stage
length). For each reservoir ``r``,

```math
v_{r,t} \;=\; v_{r,t-1}
  + K \Bigl( w_{r,t} - q_{r,t} - s_{r,t}
  + \sum_{u \in \mathcal{U}_r} q_{u,t}
  + \sum_{u \in \mathcal{S}_r} s_{u,t} \Bigr),
\qquad
v_{r,t} \in [\underline{v}_r,\, \overline{v}_r],
```

where

- ``K`` is the **flow-to-volume conversion factor** (in the case study,
  ``K = 0.0036``: a flow of 1 m³/s sustained for one hour is
  0.0036 hm³);
- ``\mathcal{U}_r`` is the set of plants whose **turbined** water feeds
  ``r``, and ``\mathcal{S}_r`` the set whose **spilled** water does — the
  two sets need not coincide (some diversions bypass the downstream
  turbine intake);
- turbine flow and spill obey their own bounds,
  ``q_{r,t} \in [\underline{q}_r, \overline{q}_r]`` and
  ``s_{r,t} \ge 0``.

In the Bolivian system three cascade links are active: **COR → SIS**
(turbine-only: only COR's turbined water reaches SIS) and
**ZON → CHU** and **TAQ1 → TAQ2** (turbine *and* spill). COR is the
system's one large seasonal reservoir; SIS, immediately downstream, is a
run-of-river plant with negligible storage, so every hectometre COR
releases is worth SIS's production factor *in addition to* COR's own —
storage decisions at the head of a cascade are leveraged decisions.

The water balance is the **only intertemporal coupling** in the problem:
water not released this week is available next week. Everything else —
power flow, generation limits — is contained within the stage.

## Hydro-to-electric coupling

A hydro plant converts outflow to active power through its
**production factor** ``\rho_r`` (MW per unit of turbined flow):

```math
p_{r,t} \;=\; \rho_r \, q_{r,t},
\qquad 0 \le p_{r,t} \le \rho_r\, \overline{q}_r .
```

The production factor differs by an order of magnitude across plants
(in the case study from ``\rho = 1.2`` to ``9.7``), which is why *where*
the system stores and releases water matters as much as *how much*: a
hectometre of water is not a fungible commodity but a location- and
plant-specific quantity of energy.

## Network physics: AC optimal power flow

Within each stage, the dispatch must satisfy the full **AC power-flow**
equations on the transmission network ``(\mathcal{N}, \mathcal{E})``. In
polar form, with complex voltage ``V_i = |V_i| e^{j\theta_i}`` at bus
``i`` and admittances ``G, B``:

```math
\begin{aligned}
&\sum_{g \in \mathcal{G}_i} p_{g,t} + \sum_{r \in \mathcal{R}_i} p_{r,t}
  - P^{d}_{i,t} + \Delta_{i,t}
  = |V_i| \sum_{k} |V_k| \bigl( G_{ik} \cos\theta_{ik} + B_{ik} \sin\theta_{ik} \bigr),
  \\[2pt]
&\sum_{g \in \mathcal{G}_i} q_{g,t} - Q^{d}_{i,t}
  = |V_i| \sum_{k} |V_k| \bigl( G_{ik} \sin\theta_{ik} - B_{ik} \cos\theta_{ik} \bigr),
\end{aligned}
\qquad \forall i \in \mathcal{N},
```

with ``\theta_{ik} = \theta_i - \theta_k``, together with voltage bands
``|V_i| \in [\underline{V}_i, \overline{V}_i]``, branch thermal (apparent
power) limits, and generator capability bounds. ``P^d_{i,t}, Q^d_{i,t}``
are the stage-``t`` bus loads and ``\Delta_{i,t} \ge 0`` is the **deficit**
(unserved load) at bus ``i``.

These equations are **nonconvex** in the voltage variables. That single
fact drives the methodological fork of this case study: the true cost of
delivering power across a stressed network — losses, reactive support,
voltage margin — is a property of this nonconvex set, and any method that
replaces it with a convex surrogate is pricing delivery on a network that
does not quite exist. We write the whole within-stage feasible set
compactly as ``(x_t, u_t) \in \mathcal{F}_t(w_t)``.

## Stage cost and objective

The stage cost is thermal fuel plus a penalty on unserved load:

```math
c_t(x_t, u_t) \;=\;
\sum_{g \in \mathcal{G}} C_g\bigl(p_{g,t}\bigr)
\;+\; C_{\Delta} \sum_{i \in \mathcal{N}} \Delta_{i,t},
```

with ``C_g`` the (convex, typically affine or quadratic) fuel cost of
thermal unit ``g`` and ``C_\Delta`` the deficit cost, set well above the
most expensive generator so that shedding load is always the last resort.
Hydro production itself is free at the stage level — its cost is
*opportunity cost*, visible only through the intertemporal coupling.

The planning problem is then exactly the general problem of
[Multistage stochastic optimization](@ref):

```math
\min_{\pi \in \Pi} \;\;
\mathbb{E}_{w_{1:T}} \Bigl[ \sum_{t=1}^{T} c_t\bigl(x_t^\pi, u_t^\pi\bigr) \Bigr]
\quad \text{s.t.} \quad
\text{water balance},\;\;
(x^\pi_t, u^\pi_t) \in \mathcal{F}_t(w_t) \;\; \forall t,
```

over nonanticipative policies ``\pi``.

## Why this is a planning problem: the value of water

The economics of LTHD are concentrated in one quantity: the marginal
**value of water**,
``-\partial\, \mathbb{E}[V_{t+1}]/\partial v_{r,t}`` — the expected future
fuel cost avoided by holding one more unit of volume in reservoir ``r``
now. A *myopic* (greedy) operator, minimizing each week in isolation,
implicitly sets this value to zero and fails in three distinct ways:

1. **Water has a time value.** Free hydro spent to shave this week's fuel
   bill is hydro missing at the seasonal demand peak, when its replacement
   is the most expensive thermal unit on the system. When the demand peak
   falls in the *dry* season — as in the Bolivian case — the mistake is
   maximal: the water most tempting to spend is exactly the water that
   will be scarcest when needed.
2. **Relief is locational.** Stored hydro relieves network stress only if
   it is stored *upstream of the right plants* and released *in the right
   weeks*; through the production factors and the cascade topology, the
   same volume is worth different energy in different places. A greedy
   dispatch cannot see the future congestion it should be positioning
   against.
3. **The forecast is a fan.** Each release is committed before the next
   inflow is known. A planning policy hedges across the scenario
   distribution; a greedy rule effectively bets on a point forecast and
   is caught out by dry sequences.

The case study does not train a greedy baseline — these failure modes are
structural, not empirical claims — but they explain what any competent
method must accomplish: **bank wet-season inflow, carry it across the
network, and release it against the dry-season peak, hedged across
scenarios.**

## One-stage reachable sets

A concept used throughout the strict TS-DDR formulation
(see [Strict mode: penalty-free gradient signal](@ref)) is the
**one-stage reachable set** of the water balance: the set of next-stage
volume vectors attainable from state ``x_{t-1}`` under inflow ``w_t`` by
*some* admissible choice of turbine flows and spills,

```math
R(x_{t-1}, w_t) \;=\;
\Bigl\{ x_t \;:\; \exists\, (q_t, s_t) \in
  [\underline{q}, \overline{q}] \times [0, \overline{s}]
  \;\text{ s.t. water balance holds and } x_t \in [\underline{v}, \overline{v}]
\Bigr\}.
```

Because the water balance is *linear* in ``(q_t, s_t)``, the per-reservoir
reachable set is an interval whose endpoints follow from substituting the
extreme releases — the property that makes penalty-free (strict) training
practical for hydro.

### Per-unit reachable bounds

For reservoir ``r`` at state ``v_{r}`` under inflow ``w_{r}``, the highest
attainable next volume corresponds to minimum outflow plus the worst-case
(maximal) upstream contribution, and the lowest to maximum outflow:

```math
u_r \;=\; \min\Bigl(\overline{v}_r,\;
    v_r + K w_r - K \underline{q}_r
    + \sum_{u \in \mathcal{U}_r} K \overline{q}_u\Bigr),
\qquad
\ell_r \;=\; \max\bigl(\underline{v}_r,\;
    v_r + K w_r - K \overline{q}_r - \overline{s}_r\bigr),
```

with ``\overline{s}_r`` the spill bound; when spillage is unbounded,
``\ell_r = \underline{v}_r`` — the reservoir can always be drawn down to its
physical minimum. A feasibility-guaranteeing policy
(`HydroReachablePolicy` in the walkthrough) maps its network output
``z_r`` into this interval through a sigmoid,

```math
\hat{v}_r \;=\; \ell_r + (u_r - \ell_r)\,\sigma(z_r),
```

with the bounds ``\ell_r, u_r`` computed from the physics and excluded from
differentiation (`@non_differentiable`): the gradient path is solely through
``\sigma(z_r)``.

### Cascade-aware clamping

The fixed upstream term ``\sum_u K \overline{q}_u`` in ``u_r`` is an
**overestimate** whenever an upstream unit stores water: its actual release
is then smaller than ``K \overline{q}_u``, so the fixed bound can exceed the
true reachable set and render a strict subproblem infeasible. After
computing the raw sigmoid targets for all units, the policy therefore clamps
downstream targets against the release actually implied upstream. For each
cascade link ``u \to d``, the implied upstream release is

```math
R_u \;=\; K w_u + v_u - \hat{v}_u ,
```

and the maximum contribution reaching ``d`` is ``\max(0, R_u)`` for
turbine-plus-spill links and ``\min(K \overline{q}_u,\, \max(0, R_u))`` for
turbine-only links. The downstream target is clamped to

```math
\hat{v}_d \;\le\; \min\bigl(\overline{v}_d,\;
    v_d + K w_d - K \underline{q}_d + \text{max\_contrib}\bigr).
```

Two assumptions are documented for this scheme:

- **Single-level cascades**: the release formula ``R_u`` omits the upstream
  unit's own incoming cascade contribution, which is conservative
  (underestimates the release) for multi-level chains — and exact for the
  Bolivian topology, whose three links are all single-level.
- **No gradient through binding clamps**: like the bounds, the clamping step
  is `@non_differentiable`; when a clamp binds, the dependence of the
  downstream target on the upstream target is not differentiated (a
  projected-gradient signal).

With these bounds and clamps, every target is one-stage reachable from the
state the policy conditioned on, which is precisely the condition under
which strict training is well-posed in every formulation (see
[Validity in every formulation, by induction](@ref)).

## Convexification and what it misprices

The classical solution method for LTHD is SDDP
([Stochastic dual dynamic programming](@ref)), which requires each stage
problem to be **convex** in the incoming state so that value-function cuts
are valid. With AC physics in the stage, practice substitutes a convex
relaxation — here the second-order-cone (SOC-WR) relaxation — in the
backward pass, and simulates the resulting policy under the true AC
equations in the forward pass: SDDP with *inconsistent formulations*.

The substitution is not free. The SOC relaxation is exact on radial,
lightly loaded networks; on meshed networks with high ``r/x`` ratios,
binding voltage bands, and load concentrated far from generation, it
admits flows the physical network cannot realize and therefore
**underprices delivery** into the stressed region. A value-of-water
surface computed against the relaxed network inherits that mispricing
precisely where storage decisions are most consequential. The measurable
symptom is the **bound-versus-forward gap** — the relative difference
between the converged relaxation bound and the simulated AC forward
cost — developed in [The bound and the forward cost](@ref). The size of
that gap on a given instance is an honest, method-agnostic estimate of how
much the convexification assumption costs there; the Bolivian system, by
its physical geography alone, sits in the regime where it is material
(see [The Bolivian interconnected system](@ref)).

## Further reading

- Molzahn & Hiskens, *A survey of relaxations and approximations of the
  power flow equations*, Foundations and Trends in Electric Energy
  Systems (2019) — where and why conic relaxations of AC power flow are
  (in)exact.
- Pereira & Pinto, *Multi-stage stochastic optimization applied to energy
  planning*, Mathematical Programming 52 (1991) — the origin of SDDP, in
  exactly this application domain.
