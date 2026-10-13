# Stochastic battery-storage AC optimal power flow

```@meta
CurrentModule = DecisionRules
```

This chapter is the canonical specification of the battery-storage case study.
It defines the physical problem, information pattern, target-state formulations,
and comparison protocol. The full network equations are collected in
[Appendix A: AC polar formulation](@ref) and
[Appendix B: SOC-WR relaxation](@ref) so the main text can be read as an
experimental design.

!!! note "Implementation status"
    The deterministic ExaModels ACP foundation and reproducible PGLib battery
    generator are implemented in the `BatteryStorageOPF` example of
    [DecisionRulesExa.jl](https://github.com/LearningToOptimize/DecisionRulesExa.jl).
    The stochastic TS-DDR and JuMP/SDDP implementations must conform to this
    specification before they are treated as accepted results.

## Purpose

The experiment studies whether a policy trained with the true nonconvex AC
network can outperform an SDDP policy whose backward passes use a convex network
relaxation. The intended mechanism is **locational storage mispricing**:

1. demand uncertainty moves scarcity between network regions;
2. congestion, losses, voltage constraints, and reactive-power limits make one
   MWh of battery energy worth different amounts at different buses;
3. SOC-WR can assign different marginal values to those stored MWh than ACP;
4. the resulting SDDP policy can charge or discharge the wrong batteries even
   when its algorithm has converged.

The final comparison is therefore:

- SDDP: SOC-WR backward passes and ACP forward simulation;
- TS-DDR: training and evaluation with ACP;
- perfect foresight (PF/WS): full-horizon ACP with the complete demand path known.

Here **battery SoC** always means state of charge. **SOC-WR** always means the
second-order-cone relaxation in lifted voltage-product space. The abbreviation
“SOC” alone is avoided.

## One-stage chronology

At the start of stage ``t``:

1. the previous battery state ``e_t`` is known;
2. the current demand atom ``\xi_t`` is observed;
3. future atoms ``\xi_{t+1:T}`` remain unknown;
4. the policy produces a target next state ``\hat e_{t+1}``;
5. an operational OPF chooses generation, network flows, charging, discharging,
   the two-sided active recourse, and ``e_{t+1}``.

Thus a nonanticipative policy has the form

```math
\hat e_{t+1}=\pi_\theta(e_t,\xi_{1:t}),
```

implemented with a recurrent encoder. PF alone observes the complete path before
making its first decision. SDDP and TS-DDR receive exactly the same current
observation and state.

The physical interstage state is the vector of battery energies. Generator
dispatch, voltages, branch flows, charge and discharge powers, and the two-sided active recourse
are stage decisions, not states.

## Data, units, and reproducible case construction

The network is an unmodified PGLib-OPF case. The initial public benchmark uses
`case300_ieee`; the same constructor works for every compatible PGLib case.

All model power quantities are per unit on the case base ``S^{base}`` in MVA:

| Quantity | Model unit | Physical conversion |
|:--|:--|:--|
| active/reactive power | pu | ``S^{base}`` MW/MVAr |
| apparent-power limit | pu | ``S^{base}`` MVA |
| battery energy | pu·h | ``S^{base}`` MWh |
| stage duration ``\Delta t`` | h | unchanged |
| voltage magnitude | pu | unchanged |
| voltage angle | rad | unchanged |

Conversion occurs once in the data layer. Component identifiers from PGLib are
never assumed consecutive or equal to array positions.

The default battery fleet is constructed by:

1. selecting in-service buses with positive active load;
2. sorting the eligible original bus identifiers;
3. sampling distinct buses without replacement with `StableRNG`;
4. sizing total fleet power as a declared fraction of base active demand;
5. splitting that power equally across batteries;
6. setting energy capacity from the declared duration in hours.

The manifest records the exact MATPOWER filename and SHA-256, PGLib release and
license, package versions, placement rule and seed, ordered battery buses, every
battery parameter, demand process, horizons, scenario seeds, and hashes. The
source-network hash, ordered placement, and battery parameters are verified on
reconstruction.

No generator price, generator limit, branch parameter, voltage limit, or network
topology may be changed during the declared candidate ladder. Any future overlay
is a new, separately approved experiment.

## Demand uncertainty

Demand is the only exogenous uncertainty in the first experiment. Batteries have
no inflow: they charge by purchasing energy from the grid.

Each bus is assigned to one of ``R`` deterministic, topology-derived regions.
At stage ``t``, atom ``a_t`` supplies a system factor ``L^{a_t}`` and regional
factors ``R_r^{a_t}``. With deterministic daily shape ``h_t``,

```math
\begin{aligned}
p^d_{t,i}
  &= p^{d,0}_i\,h_t L^{a_t}R_{r(i)}^{a_t},\\
q^d_{t,i}
  &= q^{d,0}_i\,h_t L^{a_t}R_{r(i)}^{a_t}.
\end{aligned}
```

The same multiplier is applied to active and reactive demand, preserving the
base power factor at every bus. The finite joint support contains a calm atom
and regional-scarcity atoms, so the location of high demand changes without
changing network data or prices. The initial process is stagewise independent,
which permits ordinary finite-support SDDP backward passes.

Scenario generation is a pure seeded operation. Training and evaluation use
different seeds. Paired evaluation stores the stage-major atom-index matrix;
every method reads that same matrix rather than regenerating scenarios.

## Battery model

For battery ``b`` at stage ``t``, charge and discharge powers are continuous and
nonnegative:

```math
0\le p^{ch}_{t,b}\le \bar p^{ch}_b,\qquad
0\le p^{dis}_{t,b}\le \bar p^{dis}_b.
```

The active injection at its host bus is

```math
p^{bat}_{t,b}=p^{dis}_{t,b}-p^{ch}_{t,b}.
```

The first model uses unity power factor: a battery neither injects nor absorbs
reactive power. Its energy balance is

```math
e_{t+1,b}
=(1-\sigma_b\Delta t)e_{t,b}
+\eta^{ch}_b\Delta t\,p^{ch}_{t,b}
-\frac{\Delta t}{\eta^{dis}_b}p^{dis}_{t,b},
```

with

```math
\underline e_b\le e_{t,b}\le\bar e_b.
```

Here ``\eta^{ch}_b,\eta^{dis}_b\in(0,1]`` are efficiencies and ``\sigma_b`` is
the hourly self-discharge rate; the data must satisfy
``0\le\sigma_b\Delta t<1``.

The continuous formulation has no binary charge/discharge mode. A nonnegative
throughput price penalizes
``p^{ch}_{t,b}+p^{dis}_{t,b}``, making simultaneous operation economically
dominated when the remaining costs are well formed. Simultaneous operation must
still be measured and reported; binary mode variables are introduced only if
that audit invalidates the continuous model.

### One-stage reachable energy

Ignoring the network but enforcing battery power and energy limits, the reachable
interval from ``e_{t,b}`` is

```math
\begin{aligned}
\ell_{t,b}
&=\max\left\{\underline e_b,\,
  (1-\sigma_b\Delta t)e_{t,b}
  -\frac{\Delta t}{\eta^{dis}_b}\bar p^{dis}_b\right\},\\
u_{t,b}
&=\min\left\{\bar e_b,\,
  (1-\sigma_b\Delta t)e_{t,b}
  +\eta^{ch}_b\Delta t\,\bar p^{ch}_b\right\}.
\end{aligned}
```

These bounds prove battery-dynamic reachability only. They do **not** prove that
the associated charge or discharge is feasible under generator, voltage,
reactive-power, or branch limits.

## Two-sided active recourse — the complete-recourse slack

The physical model carries a **two-sided active nodal slack**: a bounded-below,
unbounded-above nonnegative pair at **every** bus, not a fraction of local demand.
For each bus ``i`` and stage ``t``,

```math
d^{+}_{t,i}\ge 0,\qquad d^{-}_{t,i}\ge 0,
```

entering only the **active** balance:

```math
p^{d}_{t,i}-d^{+}_{t,i}+d^{-}_{t,i}+g^{s}_i v_{t,i}^2
-\!\!\sum_{g\in i}\! p^{g}_{t,g}-\!\!\sum_{b\in i}\!(p^{dis}_{t,b}-p^{ch}_{t,b})
+\!\!\sum_{\text{from }i}\! p^{fr}+\!\!\sum_{\text{to }i}\! p^{to}=0 .
```

``d^{+}`` (active deficit / injection) covers an active-power **shortfall**;
``d^{-}`` (active surplus / absorption) absorbs an active-power **excess**. Because ``d^{+}``
can inject and ``d^{-}`` can absorb arbitrary local power, the stage subproblem
has **relatively complete recourse**: it is feasible for every incoming SoC and
every dynamically reachable battery target, in **both** the charging and
discharging directions. A target that forces a battery to *charge* at a
network-constrained bus is served by local ``d^{+}``; a target that forces it to
*discharge* into a bus with saturated outgoing branches is absorbed by local
``d^{-}``. This is the classical multistage load-deficit device that lets any
non-anticipative algorithm converge without hitting an infeasible subproblem.

The slack is **active-only**: it does not touch reactive power, so reactive KCL
remains a **hard equality** with no reactive slack. Batteries are unity-power-
factor, so the battery target moves only active injection; reactive feasibility
is a property of the base network and the feasible demand process, independent of
the target.

The value of lost load is ``c^{VOLL}=10{,}000`` USD/MWh, giving stage cost

```math
C^{shed}_t
=c^{VOLL}S^{base}\Delta t
\sum_i \bigl(d^{+}_{t,i}+d^{-}_{t,i}\bigr).
```

Both slacks are included in physical operating cost. They are a safety valve: an
accepted scientific run leaves both at zero within its declared numerical
tolerance. A nonzero deficit or surplus on an otherwise sensible target is a
case-design signal (the target is not network-deliverable at that operating
point)—not a solver failure. **The strict stage always solves**; the cost, not
the solver status, reports whether the target was deliverable.

### The nodal slack is not target slack

The two mechanisms have different meanings:

| Mechanism | Relaxes | Unit | Included in reported physical cost? |
|:--|:--|:--|:--|
| nodal slack ``d^{\pm}`` | active nodal power balance | pu; cost from MWh | yes |
| target slack ``\delta^\pm`` | agreement with policy target | pu·h | no |

Code, output schemas, and prose use `active_deficit`/`active_surplus` for the
first (``d^{+}``/``d^{-}``) and `target_slack` for the second. The active
recourse is an artificial active-balance device, **not** curtailed customer load:
``d^{+}`` may exceed local demand and may be positive where ``p^d = 0``, so it is
never named "load shedding" nor reported as a per-load fraction. A scientific
candidate path requires **both** directions numerically zero within tolerance.

## Target-state projection

The policy outputs the desired outgoing battery energy
``\hat e_{t+1,b}``; the OPF determines whether and how to realize it.

### Strict mode

Strict mode adds the hard equality

```math
\hat e_{t+1,b}-e_{t+1,b}=0.
```

It has no target slack and no target penalty. With this orientation, the
equality multiplier is defined and finite-difference tested as

```math
\lambda_{t,b}
=\frac{\partial Q_t}{\partial\hat e_{t+1,b}},
```

up to the solver interface's documented dual convention. The implementation
must test the sign and magnitude rather than infer them from a convention.

Strict mode is the **primary, default** production and training target because
its multiplier is an economic shadow price uncontaminated by a penalty. Backed by
the two-sided active nodal slack, strict has **complete recourse**: for every
supported PGLib case and every dynamically reachable target—including the exact
reachable endpoints—the strict stage NLP solves and reproduces the target to
within ``10^{-5}``. Battery reachability alone is not a network-feasibility proof,
but the nodal slack makes the strict solve feasible regardless: an
under-deliverable target simply carries a deficit/surplus cost.

### Soft diagnostic mode

Soft mode uses two nonnegative target slacks:

```math
\hat e_{t+1,b}-e_{t+1,b}
-\delta^+_{t,b}+\delta^-_{t,b}=0,\qquad
\delta^+_{t,b},\delta^-_{t,b}\ge0.
```

A documented training-only penalty may combine L1 and L2 terms:

```math
C^{target}_t
=\rho_1\sum_b(\delta^+_{t,b}+\delta^-_{t,b})
+\frac{\rho_2}{2}\sum_b
\left[(\delta^+_{t,b})^2+(\delta^-_{t,b})^2\right].
```

Soft mode is for diagnosis, warm starts, and penalty sensitivity studies. Its
penalty is excluded from physical operating cost and final policy comparisons;
target violations are reported separately.

### Reachable target policy

For a raw network output ``z_{t,b}``, the normalized output is mapped into the
battery interval:

```math
\hat e_{t+1,b}
=\ell_{t,b}+(u_{t,b}-\ell_{t,b})\,y_{t,b}.
```

The canonical default is the project-tested stretched sigmoid

```math
y_{t,b}
=\operatorname{clamp}\left(
\frac{\operatorname{sigmoid}(z_{t,b})-0.03}{0.94},
0,\;1-10^{-3}\right).
```

It can reach the lower edge while keeping a small margin below the exact upper
edge, which avoids a known interior-point degeneracy at a store-max strict
target. A separately named `hardsigmoidsafe` activation may be retained as an
option, but must use the same safe upper margin and be tested independently.

Reachability bounds are physical projection data, not learned functions. The
canonical gradient stops through ``\ell`` and ``u``; gradients flow through the
normalized policy output. Recurrent state is reset at every scenario boundary.

## Stage objective and cost accounting

For quadratic generator costs expressed in USD/hour at per-unit dispatch, the
physical stage cost is

```math
\begin{aligned}
C^{phys}_t
={}&\Delta t\sum_g
\left(c_{2g}(p^g_{t,g})^2+c_{1g}p^g_{t,g}+c_{0g}\right)\\
&+S^{base}\Delta t\sum_b c^{cycle}_b
\left(p^{ch}_{t,b}+p^{dis}_{t,b}\right)
+C^{shed}_t.
\end{aligned}
```

Every term, including ``c_{0g}``, is duration-scaled. The complete training
objective is ``C^{phys}_t+C^{target}_t`` in soft mode and ``C^{phys}_t`` in
strict mode.

Every result reports at least:

- generator cost;
- battery throughput cost;
- VOLL nodal-slack cost, deficit and surplus MWh;
- target penalty and target violation, if soft;
- physical operating cost;
- reporting-window and look-ahead physical costs separately.

No target penalty is mixed into PF, SDDP-ACP, or TS-DDR-ACP operating cost.

## Horizon and terminal treatment

The total horizon is

```math
T=T^{report}+T^{lookahead}.
```

Every method optimizes both portions. Statistical comparisons use physical cost
only over stages ``1:T^{report}``; the look-ahead cost and terminal battery SoC
are reported separately. The look-ahead buffer discourages end-of-horizon
depletion without adding a salvage value or terminal target.

There is no default terminal salvage term or terminal energy constraint. If
either is introduced later, it must be fixed before production and identical in
PF, SDDP, and TS-DDR.

## Methods

### TS-DDR

TS-DDR trains a recurrent policy for next-battery-SoC targets. Each training
sample embeds those targets in ACP projection problems. Strict training uses the
target-constraint multipliers as envelope gradients; soft training additionally
requires the declared target-penalty derivatives.

Training and evaluation use true ACP. Checkpoints contain model parameters,
normalization, architecture, activation and upper margin, battery/process
manifests, horizon, stage duration, seeds, and source hashes.

### SDDP

The SDDP baseline uses the battery SoC as the resource state:

- backward subproblems use SOC-WR and ordinary stock cuts;
- forward simulations use ACP;
- cuts are rebuilt for each frozen candidate;
- no custom cut acceptance, tolerance ladder, or penalty rewrite is allowed.

The SDDP bound belongs to the relaxed backward model. It is not the SDDP policy's
ACP operating cost and is not the room available for TS-DDR.

### Perfect foresight

For each evaluation path, PF solves the full-horizon ACP after seeing every
demand atom. It provides an information-relaxation benchmark:

```math
\operatorname{room}
=\frac{\mathbb E[C^{SDDP\text{-}ACP}]
-\mathbb E[C^{PF\text{-}ACP}]}
{\mathbb E[C^{SDDP\text{-}ACP}]}.
```

This room is only an upper bound on the improvement a nonanticipative policy
could attain. Because ACP is nonconvex, a locally solved PF model is an empirical
benchmark, not a rigorous mathematical lower bound unless global optimality is
certified.

## Experimental acceptance

A candidate proceeds to production only if:

1. ExaModels and JuMP agree with an independent PowerModels ACP reference on
   deterministic cases;
2. battery balances, AC residuals, bounds, and target equations close within
   declared tolerances;
3. strict case14 and case300 paths solve (always feasible via the nodal slack)
   with zero active deficit and zero active surplus within tolerance;
4. simultaneous charge/discharge is negligible;
5. ACP and SOC-WR assign materially different marginal values to energy at
   relevant battery buses, and that difference changes battery behavior;
6. SDDP runs normally with clean ACP forward passes;
7. paired PF room is large enough to justify training.

Final evaluation uses frozen manifests and the same stored scenario matrix for
all methods. It reports per-path PF, SDDP-ACP, and TS-DDR-ACP physical costs,
paired differences, a 95% confidence interval, solver failures, active recourse
(deficit and surplus), battery trajectories, binding network constraints, runtime, hardware, seeds,
and hashes. Scientific success requires the paired TS-DDR-minus-SDDP mean to be
negative with a 95% paired confidence interval excluding zero.

## Implementation invariants

The following are model requirements rather than tunable choices:

- one shared ACP constraint implementation underlies deterministic, stochastic,
  training, and evaluation builders;
- active and reactive demand use the same atom multiplier (power factor preserved);
- reactive balance has no independent slack;
- the active nodal slack is two-sided (deficit ``d^{+}`` and surplus ``d^{-}``),
  nonnegative, unbounded above at every bus, and priced at VOLL — it gives the
  strict target formulation relatively complete recourse;
- strict and soft target formulations have different variable sets;
- target penalties never enter reported physical cost;
- generator and network data remain the original PGLib values;
- CPU and GPU builders represent the same equations;
- no solver status is relabeled and failed paths are never silently discarded.

## Appendix A: AC polar formulation

This appendix states the complete per-stage ACP projection. Time subscripts are
omitted where unambiguous.

### Sets and voltage variables

Let ``N`` be buses, ``G_i`` generators at bus ``i``, ``B_i`` batteries at bus
``i``, and ``A_i`` directed branch ends leaving bus ``i``. Complex bus voltage is

```math
V_i=v_i e^{\mathrm j\theta_i},
\qquad \underline v_i\le v_i\le\bar v_i.
```

One angle is fixed in each connected reference component:

```math
\theta_i=0,\qquad i\in N^{ref}.
```

### Generator limits

```math
\underline p^g_g\le p^g_g\le\bar p^g_g,\qquad
\underline q^g_g\le q^g_g\le\bar q^g_g.
```

The limits and polynomial costs are taken directly from PGLib after the single
per-unit conversion.

### General branch model

For branch ``k=(i,j)`` let its pi-model—including series admittance, asymmetric
line charging, complex transformer tap, and phase shift—be represented by

```math
\begin{bmatrix}I_{ij}\\I_{ji}\end{bmatrix}
=
\begin{bmatrix}
Y^{ff}_k & Y^{ft}_k\\
Y^{tf}_k & Y^{tt}_k
\end{bmatrix}
\begin{bmatrix}V_i\\V_j\end{bmatrix}.
```

The two complex branch flows are

```math
S_{ij}=p_{ij}+\mathrm jq_{ij}=V_i I_{ij}^*,\qquad
S_{ji}=p_{ji}+\mathrm jq_{ji}=V_j I_{ji}^*.
```

These equations are the four real ACP branch-flow equalities. They retain
transformer taps and shifts and both end shunts; replacing them with a lossless
or single-ended approximation changes the model.

For clarity, if ``Y^{ff}=a+\mathrm jb`` and
``Y^{ft}=c+\mathrm jd``, the from-end equations are

```math
\begin{aligned}
p_{ij}
&=a v_i^2+v_iv_j[c\cos(\theta_i-\theta_j)
                 +d\sin(\theta_i-\theta_j)],\\
q_{ij}
&=-b v_i^2+v_iv_j[c\sin(\theta_i-\theta_j)
                  -d\cos(\theta_i-\theta_j)].
\end{aligned}
```

The to-end equations follow identically from ``Y^{tt}``, ``Y^{tf}``, and
``\theta_j-\theta_i``.

### Branch limits

Apparent-power limits are enforced at both ends:

```math
p_{ij}^2+q_{ij}^2\le(\bar s_k)^2,\qquad
p_{ji}^2+q_{ji}^2\le(\bar s_k)^2.
```

Voltage angle differences satisfy

```math
\underline\theta^\Delta_k
\le\theta_i-\theta_j
\le\bar\theta^\Delta_k.
```

An absent PGLib thermal limit adds no artificial finite bound.

### Nodal power balance

Let bus shunt admittance be ``Y_i^s=g_i^s+\mathrm jb_i^s``. Using branch flows
directed away from the bus, active and reactive KCL are

```math
\begin{aligned}
\sum_{g\in G_i}p^g_g
+\sum_{b\in B_i}(p^{dis}_b-p^{ch}_b)
-p^{served}_i-g_i^s v_i^2
&=\sum_{(i,j,k)\in A_i}p_{ij},\\
\sum_{g\in G_i}q^g_g
-q^{served}_i+b_i^s v_i^2
&=\sum_{(i,j,k)\in A_i}q_{ij}.
\end{aligned}
```

The battery energy equation, power and energy bounds, two-sided active-recourse
terms, target equation for the selected mode, and stage objective from the main
text complete the model.

## Appendix B: SOC-WR relaxation

SOC-WR retains the OPF specification—generation, batteries, demand, two-sided
active recourse, costs, KCL, thermal limits, and target equations—but replaces
the nonconvex voltage representation.

Define the Hermitian voltage-product matrix

```math
W=VV^*,\qquad
W_{ii}=w_i,\qquad
W_{ij}=w^R_{ij}+\mathrm jw^I_{ij}.
```

Voltage bounds become

```math
(\underline v_i)^2\le w_i\le(\bar v_i)^2.
```

Branch flows are affine in ``W``:

```math
\begin{aligned}
S_{ij}
&=(Y^{ff}_k)^*W_{ii}+(Y^{ft}_k)^*W_{ij},\\
S_{ji}
&=(Y^{tt}_k)^*W_{jj}+(Y^{tf}_k)^*W_{ji},\\
W_{ji}&=W_{ij}^*.
\end{aligned}
```

The exact lifted ACP model requires

```math
(w^R_{ij})^2+(w^I_{ij})^2=w_iw_j
```

for every branch, together with globally consistent voltage angles around all
network cycles. SOC-WR relaxes the rank-one equality to

```math
(w^R_{ij})^2+(w^I_{ij})^2\le w_iw_j,
```

which is second-order-cone representable, and does not impose global rank-one
cycle consistency. When the angle-difference interval lies inside
``(-\pi/2,\pi/2)``, its standard lifted form is

```math
\tan(\underline\theta^\Delta_k)w^R_{ij}
\le w^I_{ij}\le
\tan(\bar\theta^\Delta_k)w^R_{ij},
```

with the corresponding valid-domain conditions and strengthening used by
PowerModels. Apparent-power limits at both ends and nodal KCL are unchanged and
remain convex in the lifted variables.

The canonical implementation is PowerModels' `SOCWRPowerModel`; independent
hand-written versions must match it on objective, bounds, flows, and storage
marginal values before use. SOC-WR is a relaxation, not an AC-feasible network
model. Its solution must be evaluated by a separate ACP forward solve.

## Appendix C: symbols and references

| Symbol | Meaning |
|:--|:--|
| ``p^g,q^g`` | generator active/reactive power |
| ``v,\theta,V`` | voltage magnitude, angle, complex voltage |
| ``p_{ij},q_{ij},S_{ij}`` | directed branch-end power flow |
| ``p^d,q^d`` | realized active/reactive demand |
| ``d^{+},d^{-}`` | two-sided active recourse: deficit / surplus (pu) |
| ``p^{ch},p^{dis}`` | battery charge/discharge power |
| ``e`` | battery energy state |
| ``\hat e`` | policy target for outgoing battery energy |
| ``\delta^+,\delta^-`` | soft target slacks |
| ``\Delta t`` | stage duration in hours |
| ``S^{base}`` | network power base in MVA |
| ``W`` | lifted voltage-product matrix |

Primary references:

- C. Coffrin et al.,
  [“PowerModels.jl: An Open-Source Framework for Exploring Power Flow Formulations”](https://doi.org/10.23919/PSCC.2018.8442948),
  PSCC 2018.
- S. Babaeinejadsarookolaee et al.,
  [“The Power Grid Library for Benchmarking AC Optimal Power Flow Algorithms”](https://doi.org/10.48550/arXiv.1908.02788),
  IEEE PES Task Force report.
- R. A. Jabr,
  [“Radial Distribution Load Flow Using Conic Programming”](https://doi.org/10.1109/TPWRS.2006.879234),
  IEEE Transactions on Power Systems, 2006.
- M. V. F. Pereira and L. M. V. G. Pinto,
  [“Multi-stage Stochastic Optimization Applied to Energy Planning”](https://doi.org/10.1007/BF01582895),
  Mathematical Programming, 1991.
- A. Rosemberg et al.,
  [“Efficiently Training Deep-Learning Parametric Policies Using Lagrangian Duality”](https://arxiv.org/abs/2405.14973),
  2024.
