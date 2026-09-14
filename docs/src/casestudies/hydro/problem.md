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
[Results](@ref "Results: TS-DDR versus SDDP"); how each policy arrives at a price for water is
[Valuing water: two approaches](@ref); a runnable version is
[Walkthrough](@ref).

## Stages, state, uncertainty, decisions

Time is discretized into **weekly stages** ``t = 1, \ldots, T`` (each stage
the case study trains on ``T`` stages spanning several years, and reports over
a shorter window so the reported horizon is free of end-of-horizon effects). At
each stage:

- **State** — the vector of reservoir volumes
  ``x_t = (v_{r,t})_{r \in \mathcal{R}} \in \mathbb{R}^{n_{\mathrm{hyd}}}``,
  the only quantity carried between stages.
- **Uncertainty** — the vector of river inflows
  ``w_t = (w_{r,t})_{r \in \mathcal{R}}``, revealed at the start of the
  stage. Inflows are strongly seasonal and spatially correlated across the
  basin, so realizations are drawn as *joint* scenarios (see
  [Uncertainty Sampling](@ref)). Demand follows a fixed profile: inflow is the
  only uncertainty, which keeps the comparison a statement about how the two
  methods value **water**.
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
  + K \Bigl( w_{r,t} - q_{r,t}
  + \sum_{u \in \mathcal{U}_r} q_{u,t} \Bigr)
  - s_{r,t} + \sum_{u \in \mathcal{S}_r} s_{u,t},
\qquad
v_{r,t} \in [\underline{v}_r,\, \overline{v}_r],
```

where

- ``K`` is the **flow-to-volume conversion factor**: the volume accumulated by
  a unit flow sustained over one stage. It therefore scales with the stage
  duration, which for long-term planning is long — the case study uses weekly
  stages — and the whole water balance is proportional to it;
- note that **turbine flow is scaled by ``K`` and spill is not**: inflow and
  turbined outflow are rates (m³/s) while spill is already carried as a volume
  in this formulation. The asymmetry is HydroPowerModels' convention and is
  reproduced exactly by every engine here;
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
physical minimum. A policy that must emit attainable targets therefore has a
natural construction available: squash an unconstrained output into this
interval,

```math
\hat{v}_r \;=\; \ell_r + (u_r - \ell_r)\,\sigma(z_r),
```

The bounds ``\ell_r, u_r`` are functions of the incoming state and the realized
inflow, and they **are differentiated**. An earlier implementation declared them
non-differentiable, which silently truncated ``\partial \hat v_r / \partial
v_r`` to the ``\sigma`` term alone; measured against finite differences over the
full 126-stage horizon, that truncated gradient carried 5.9% of the true
magnitude and pointed 48 degrees away from it, and the error compounds with the
horizon. Restoring the path through ``\ell_r`` and ``u_r`` reproduces the finite
difference to `cos = 1.000000` and `‖AD‖/‖FD‖ = 1.000000`. See
[The gradient must flow through the reachable map](@ref).

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
- **The clamp is a real dependence, not a projection.** Where it binds, the
  downstream reachable set genuinely moves with the upstream decision — one more
  unit released above is one more unit the unit below can hold — and where the
  turbine cap binds instead, it does not. Both branches matter to any method that
  differentiates through this map; see
  [Valuing water: two approaches](@ref "The gradient must flow through the reachable map").

With these bounds and clamps, a target chosen inside the interval is reachable in
one stage from the state it was conditioned on — the condition under which a hard
target equality is well posed at all (see
[Validity in every formulation, by induction](@ref)).

## Further reading

- Molzahn & Hiskens, *A survey of relaxations and approximations of the
  power flow equations*, Foundations and Trends in Electric Energy
  Systems (2019) — where and why conic relaxations of AC power flow are
  (in)exact.
- Pereira & Pinto, *Multi-stage stochastic optimization applied to energy
  planning*, Mathematical Programming 52 (1991) — the origin of SDDP, in
  exactly this application domain.
