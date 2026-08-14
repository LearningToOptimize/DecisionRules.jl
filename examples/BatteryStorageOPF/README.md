# Battery-storage AC-OPF — JuMP / PowerModels / SDDP engine

This example is the CPU half of the multistage battery-storage study. It does
two things:

1. it is a **reusable toolkit** for building and diagnosing battery cases on top
   of any PGLib benchmark — acquire a case, place batteries, author a demand
   process, freeze it into finite support, probe the physics and the value of
   stored energy, solve perfect-foresight equivalents, and train the SDDP
   baseline;
2. it is the **construction of this study's own benchmark**, built with exactly
   those tools and nothing else.

The scientific narrative — what the experiment asks and what it found — lives in
the documentation. This file says how to run things and what each file is for.

The GPU half (the ExaModels true-ACP model, the strict reachable policy and the
TS-DDR trainer) lives in `DecisionRulesExa.jl/examples/BatteryStorageOPF`. The
two packages are independent: neither loads the other. They share the frozen
case bytes and two source files (`battery_case.jl`, `battery_solution_schema.jl`)
as copies whose byte identity is asserted whenever the case is rebuilt.

Everything below runs from this directory with `julia --project=.`, needs no
cluster scheduler and writes no telemetry.

## The boundary between the packages and this example

| owned by | what |
|---|---|
| PGLib.jl | the benchmark case and its parser |
| PowerModels.jl | buses, generators, branches, voltage variables, reference angle, Ohm's law at both ends with taps and phase shifts, shunts, angle-difference limits, apparent-power limits at both ends, nodal active and reactive balances, generator bounds and costs, and the `ACPPowerModel` / `SOCWRConicPowerModel` formulations themselves |
| SDDP.jl | policy graph, state variables, sampling, cut generation, training, simulation |
| Distributions.jl | the laws an authoring demand sampler is built from |
| this example | batteries: energy state, charge/discharge, state transition, unity-power-factor injection, throughput cost, the strict outgoing-energy target equality, the two-sided nodal active-power recourse, and the demand parameterization |

No AC trigonometry and no SOC-WR lifted branch equation is written here. The
regression suite scans the model-building files and the diagnostics and fails if
one appears.

## Files

| file | role |
|---|---|
| `battery_case.jl` | the frozen case contract: canonical JSON, artifact hashes, battery parameters, the one-stage reachable interval, the **frozen finite demand support**, the battery-placement API, and the reproducible evaluation protocol. **Byte-identical copy in the Exa package.** |
| `battery_solution_schema.jl` | the shared long-format solution schema, and an engine-neutral recomputation of the physical residuals. **Byte-identical copy in the Exa package.** |
| `battery_demand.jl` | the **authoring** side of demand: the composable sampler abstraction, the deterministic profile, sampler validation, and `freeze_demand_support` |
| `build_battery_case.jl` | PGLib acquisition and validation, recourse pricing, the general case constructor, the correctness case, the research case, and the Exa mirror |
| `battery_powermodels.jl` | the battery layer on top of PowerModels: the shared problem specification, stage-model construction, the strict stage solve with its multipliers, nodal prices, binding limits, solution extraction, and the runtime provenance assertion |
| `battery_diagnostics.jl` | the diagnostic toolkit: incoming-energy sampling, the targetless probe, the fixed-outgoing-energy value curve, the multiperiod deterministic equivalent and the perfect-foresight panel |
| `battery_analysis.jl` | tables and figures over the shared schema |
| `battery_sddp.jl` | the SDDP baseline: SOC-WR backward graph, true-ACP forward graph, stock cuts, paired simulation on a protocol |
| `test/runtests.jl` | the consolidated regression suite |
| `case/<name>/` | the constructed artifacts: `network.json`, `batteries.json`, `demand.json`, `case_manifest.json`. **Not committed** — see below. |

---

# The user workflow

The whole workflow is nine calls. Everything below is a single Julia session
started with `julia --project=.`.

```julia
include("build_battery_case.jl")     # → battery_case.jl, battery_demand.jl
include("battery_diagnostics.jl")    # → battery_powermodels.jl
include("battery_analysis.jl")
```

## 1. Obtain any PGLib case

```julia
src = acquire_pglib_case("pglib_opf_case30_ieee")
src.report      # counts, totals, connected component, generation headroom
src.versions    # the PGLib / PowerModels / Julia versions the bytes came from
```

`acquire_pglib_case` preserves the benchmark's own component identifiers — PGLib
cases contain nonconsecutive ones — and validates that every in-service component
sits on an existing bus, that every in-service bus is connected to the reference
bus, and that the case has a positively priced generator and positive load.

PGLib ships three libraries per benchmark and they are **different systems, not
different settings of one**. The suffix selects it:

```julia
acquire_pglib_case("pglib_opf_case30_ieee")         # typical operating condition
acquire_pglib_case("pglib_opf_case30_ieee__api")    # congested (active power increase)
acquire_pglib_case("pglib_opf_case30_ieee__sad")    # small angle difference
```

## 2. Specify or randomly sample battery locations

```julia
# explicit
buses, prec = select_battery_buses(src.network, ExplicitPlacement([5, 12, 30]))

# a reproducible uniform draw from the load buses
buses, prec = select_battery_buses(src.network, SampledPlacement(3; seed = 7))

# weighted by nominal demand
load_at = nominal_load_at_bus(src.network)
buses, prec = select_battery_buses(src.network,
    SampledPlacement(3; seed = 7, weight = b -> max(0.0, get(load_at, b, 0.0))))

# a custom eligibility rule, and a custom placement rule
select_battery_buses(src.network, SampledPlacement(2; seed = 7);
                     eligible = bus -> Float64(bus["vmax"]) >= 1.06)
select_battery_buses(src.network, CallablePlacement(
    (candidates, meta) -> candidates[1:2]; name = "two-lowest-ids"))
```

Eligibility rejects nonexistent, out-of-service and disconnected buses, and a
sampled draw is without replacement. `prec` is the manifest record: the full
eligible pool, the strategy, the seed, the weights and the selection.

## 3. Configure capacities and initial energy

```julia
total = sum(max(0.0, v) for v in values(nominal_load_at_bus(src.network)))
fleet, crec = battery_fleet(src.network, buses;
                            power = 0.04 * total,   # pu, per battery
                            energy_hours = 2.0,     # duration at full power
                            charge_efficiency = 0.95, discharge_efficiency = 0.95,
                            self_discharge = 0.999, throughput_cost = 5.0,
                            initial_fraction = 0.5)
```

Every rating accepts a scalar, a `Dict` keyed by bus, or a callable — which is
how a **sampled** capacity is expressed without the case contract depending on
Distributions.jl:

```julia
rng = StableRNG(4)
fleet, _ = battery_fleet(src.network, buses;
                         power = _ -> rand(rng, Uniform(0.2, 0.4)), energy_hours = 2.0)
```

## 4. Define a demand sampler

Demand is the only uncertainty. For the original PGLib load values, the realized
demand of load `i` at stage `t` is

```
pd[i,t] = h[i,t] * m[i,t] * pd0[i]
qd[i,t] = h[i,t] * m[i,t] * qd0[i]
```

with `h` a deterministic profile and `m` the uncertain multiplier. The **same**
multiplier scales active and reactive demand, so every realization preserves each
load's own power factor.

A sampler's fundamental output is a **joint multiplier vector** over the case's
loads — not a scalar, and not an implicit collection of independent draws.

```julia
meta = demand_meta(src.network)      # load_ids, load_bus, nominal_pd/qd, num_loads
```

**A `Distribution`, system-wide:**

```julia
sampler = SystemMultiplier(Uniform(0.9, 1.1))
sampler = SystemMultiplier(DiscreteNonParametric([0.95, 1.0, 1.05], fill(1/3, 3)))
```

**Independent per load (a convenience, not the general interface):**

```julia
sampler = IndependentMultiplier(LogNormal(0.0, 0.05))
sampler = IndependentMultiplier(Dict(3 => Uniform(0.8, 1.2)))   # others deterministic
```

**Regional finite atoms — a genuinely correlated joint vector:**

```julia
pocket = [j for j in 1:meta.num_loads if meta.load_bus[j] in [1,3,5,6,7,8,9,14]]
sampler = GroupMultiplier(
    [meta.load_ids[pocket], setdiff(meta.load_ids, meta.load_ids[pocket])],
    [DiscreteNonParametric([0.96, 1.04], [0.5, 0.5]),
     DiscreteNonParametric([0.99, 1.01], [0.5, 0.5])])
```

**A custom callable** — the general interface every other sampler is a
convenience over:

```julia
sampler = CallableMultiplier(
    (rng, t, meta) -> 1.0 .+ 0.05 .* randn(rng, meta.num_loads) .* (t > 12);
    name = "late-stage jitter")
```

**Explicit atoms, and stage dependence:**

```julia
sampler = FiniteMultiplier([0.9, 1.0, 1.1], [0.25, 0.5, 0.25])
sampler = StageMultiplier([t <= 12 ? DeterministicMultiplier(1.0) : sampler
                           for t in 1:24])
```

**Composition** multiplies element-wise, so a system-wide level, a regional
effect and a per-load term compose without knowing about each other:

```julia
sampler = ProductMultiplier(SystemMultiplier(Uniform(0.98, 1.02)),
                            GroupMultiplier(...), IndependentMultiplier(...))
```

Check it before freezing:

```julia
validate_sampler(sampler, src.network, 24)
```

which asserts the output dimension and load order, finiteness and
nonnegativity, exact seeded reproduction, that the sampler does not reach for the
global RNG, that every load keeps its power factor, and that any declared finite
support normalizes.

## 5. Freeze the sampler into finite support

**Both SDDP and TS-DDR train from finite support.** A general sampler or a
continuous `Distribution` is an *authoring mechanism*; before either method
trains, it is materialized into a frozen, hashed, stage-major support

```
W_t = { (w_{t,1}, p_{t,1}), …, (w_{t,K_t}, p_{t,K_t}) }
```

and both engines read the **same bytes**. Neither method may resample or
rediscretize the authoring law on its own — that is the boundary that makes "the
two methods faced the same stochastic program" a checkable statement.

```julia
support = freeze_demand_support(sampler, src.network, 30;
                                seed = 20260805,
                                method = :auto,        # :exact | :empirical | :auto
                                atoms_per_stage = 8,   # for an empirical freeze
                                profile = diurnal_profile(30),
                                profile_period = 24,
                                stage_hours = 1.0,
                                protocol_seed = 20260805)
support_digest(support)
```

- an **explicitly discrete** sampler keeps its support and probabilities exactly
  — no reweighting, no resampling, only exact-duplicate merging;
- a **continuous or general** sampler is discretized transparently: draw
  `atoms_per_stage` joint vectors per stage from `StableRNG(seed)` and weight them
  equally. It is reproducible from `(seed, atoms_per_stage)` alone and claims no
  moment matching. A different discretizer is declared by giving the sampler a
  `support` callable; there is no hidden quadrature rule.

The frozen support records the authoring sampler's description, the seed, the
atom count, the method and the resulting atoms in `demand.json`.

**Four distinct objects, and the README will not conflate them:**

| object | what it is | where it lives |
|---|---|---|
| authoring sampler | an arbitrary joint law, possibly continuous | your code, never a case |
| frozen training support | finite atoms + probabilities per stage, hashed | `demand.json`, mirrored to both engines |
| screening protocol | a small set of global scenario IDs drawn from that support, used while choosing a case and selecting checkpoints | regenerated from `protocol_seed` |
| final unseen protocol | a fresh, larger paired protocol no policy was ever selected on | regenerated from the manifest, evaluated once |

## 6. Build and verify the manifest

```julia
case = build_case(src; dir = "case/my_case", batteries = fleet, support = support,
                  placement = Dict("buses" => prec, "capacity" => crec),
                  protocol_stages = 30, protocol_scenarios = 500)
verify("case/my_case")
```

`build_case` writes the four artifacts, records every hash, and reads the case
back **through the verifier** — a case that cannot be re-read is a build failure
rather than a later mystery. Two properties are deliberately fail-closed, because
both have historically corrupted a study silently: the stage duration must be
present, positive and agree with the manifest, and every artifact must hash to
what the manifest records.

## 7. Sample incoming battery energies

```julia
states = sample_incoming_energy(case; kind = :uniform, seed = 3, batch = 8)
sample_incoming_energy(case; kind = :fixed, level = 0.5)
sample_incoming_energy(case; kind = :distribution, dist = Beta(2, 2), seed = 3, batch = 4)
sample_incoming_energy(case; kind = :callable, callable = (rng, b, m) -> 0.25, seed = 1)
```

The state is sampled in NORMALIZED terms and mapped into each battery's own
bounds, so a sampler transfers unchanged to a case with different ratings. Every
resulting energy is validated. This samples **diagnostic initial states only** —
it is not the demand process and plays no part in training or evaluation.

## 8. Run targetless single-stage probes

```julia
p = targetless_probe(case, PowerModels.ACPPowerModel;
                     energy_in = states[1], stage = 19, atom = 2)
p.cost_stage, p.cost_generation, p.cost_deficit, p.cost_surplus
p.energy_out, p.p_ch, p.p_dis, p.p_bat, p.simultaneous
p.vm, p.va, p.p_fr, p.q_to, p.price_active, p.price_reactive
p.energy_in_dual        # ∂Q/∂e_in — what the INHERITED energy was worth
p.residuals             # recomputed independently of the engine
binding_table(p)        # what stopped the network
```

The same call with `PowerModels.SOCWRConicPowerModel` runs the relaxation, and
`targetless_batch` sweeps incoming-energy vectors × stages × atoms, keeping every
combination including the failures.

> **A targetless one-stage solve is myopic.** Nothing in it prices the energy
> left in a battery at the end of the stage, so it will rationally discharge as
> much as is useful and leave the battery empty. That is the definition of a
> one-stage problem, not a finding. Read it for the physics — what the network
> could deliver, what limit stopped it, what the energy the stage *inherited* was
> worth — and read the next section for the value of what is left behind.

## 9. Inspect fixed-target value curves and nodal prices

```julia
b = first(case.batteries)
lo, hi = reachable_interval(b, states[1][b.index], stage_hours(case))
cmp = compare_value_curves(case; battery = b.index, energy_in = states[1],
                           stage = 19, atom = 2,
                           grid = collect(range(lo, hi; length = 9)))
value_curve_table(cmp)
plot_value_curves(cmp, "value_curves.png")
```

The outgoing energy is fixed by the **same hard equality the strict formulation
uses** — there is no target slack anywhere — so the reported multiplier is
`∂Q/∂e_out`, the price the model puts on carrying one more unit of energy out of
the stage.

Three properties of the check matter:

- **finite differences confirm the interior multipliers.** The value curve is
  convex and *piecewise* smooth — a binding limit puts a kink in it — so each
  interior point is classified first, and `λ` is compared against a central
  difference only where the curve is locally smooth. At a kink the correct
  statement is that `λ` lies *between* the one-sided slopes, and `bracketed`
  records it.
- **endpoints are not evidence.** At an endpoint of the reachable interval the
  target sits on a bound, `λ` is a subgradient, and two solvers may report two
  valid values orders of magnitude apart.
- **points that used physical recourse are excluded.** There, part of the
  marginal value is the recourse *price* — chosen to be far above any generator —
  rather than the network's valuation of stored energy.

The quantity that carries a finding is `dlambda = λ_SOC − λ_ACP`, and its
locational *ranking*: a uniform level shift changes nothing about where a policy
puts energy, a change in the order of buses changes everything.

```julia
marginal_value_table([cmp1, cmp2, ...])    # rank_acp, rank_soc, rank_changed
```

## 10. Solve one multiperiod deterministic equivalent

```julia
proto = scenario_index_matrix(case.demand, 30, 500)
de = deterministic_equivalent(case, proto[:, 1])       # true ACP
de.total_cost, de.stage_cost, de.cumulative_cost
de.energy_terminal, de.throughput, de.simultaneous
de.worst_deficit, de.worst_surplus, de.residuals
stage_cost_table(de); battery_table(de, case); price_table(de, case)
plot_energy_trajectory(de, case, "energy.png")
```

The whole horizon is solved at once, knowing the entire demand path: no policy,
no target, no target slack, no future-cost approximation. It is assembled from
the same per-stage builder the study trains on — one PowerModels model per stage,
all in one JuMP model, coupled only by `e_in[t+1] == e_out[t]`.

The same call with `model_type = PowerModels.SOCWRConicPowerModel` gives the
**relaxed** perfect-foresight solve. That is a diagnostic, not a physical
reference: its cost is not attainable.

## 11. Calculate a small perfect-foresight panel

```julia
panel = perfect_foresight_panel(case, proto; ids = 1:8)
panel.complete, panel.mean, panel.std, panel.sem
panel_table(panel)
```

Every requested identifier is retained: none is replaced, dropped or renumbered,
a first-attempt failure is re-solved once with a completely fresh model, and both
statuses are recorded. If any identifier is unsolved the panel is marked
incomplete — a mean over the scenarios that happened to succeed is a mean over a
different problem.

> **What the mean is, and is not.** The true-ACP perfect-foresight mean is a
> **wait-and-see lower bound** on the nonanticipative stochastic problem: a
> clairvoyant operator cannot be beaten by one who must decide before seeing the
> future. The gap between a policy and this bound is diagnostic **headroom**, and
> it *contains the value of future information*, which no nonanticipative policy
> — TS-DDR or SDDP — can recover. It is not a target and it is not attainable.
> The bound may be computed while screening candidates; its relationship to a
> trained policy is assessed only on **paired** paths, after that policy exists.

## 12. Train and evaluate SDDP

```julia
trained = train_battery_sddp(case; num_stages = 30, iteration_limit = 300)
trained.bound                      # over the SOC-WR relaxation, over 30 stages
assert_graph_provenance(trained.backward, PowerModels.SOCWRConicPowerModel)
assert_graph_provenance(trained.forward, PowerModels.ACPPowerModel)

sims = simulate_battery_sddp_on(trained, proto;
                                ids = [b.index for b in case.batteries], columns = 1:8)
costs = [sum(s[t][:stage_objective] for t in 1:30) for s in sims]
paired_difference(costs, [panel.results[c].total_cost for c in 1:8])
```

The backward pass is an actual `PowerModels.SOCWRConicPowerModel`, the forward
pass an actual `PowerModels.ACPPowerModel`, and the two are joined by SDDP.jl's
own `AlternativeForwardPass` / `AlternativePostIterationCallback`. No
`duality_handler` is overridden and no cut is filtered, retried or reweighted.

The bound is a bound on the **SOC-WR relaxation over the horizon it was trained
on**. Quoting it beside a forward cost accumulated over a different number of
stages compares two different quantities, and nothing here invites that.

`simulate_battery_sddp_on` replays a fixed protocol through `SDDP.Historical`,
which is what makes an SDDP cost **paired** with a perfect-foresight cost and,
later, with a TS-DDR cost.

## 13. Where strict TS-DDR enters

Nothing above trains a policy. Strict TS-DDR is the GPU engine's business:
`DecisionRulesExa.jl/examples/BatteryStorageOPF` reads the **same frozen case
bytes and the same frozen support**, builds the multistage true-ACP
deterministic equivalent in ExaModels, evaluates the reachable policy before each
stage, and differentiates the stage value through the strict target multiplier.
This engine's role afterwards is to replay and verify what that engine produced,
through the shared solution schema.

---

## Commands

```bash
# rebuild (and mirror) the correctness case from PGLib
DR_BAT_MIRROR=/path/to/DecisionRulesExa.jl/examples/BatteryStorageOPF \
  julia --project=. build_battery_case.jl

# re-verify a frozen case in place: hashes, schemas, stage duration, support, protocol
julia --project=. build_battery_case.jl --verify

# stock SDDP: SOC-WR backward, true-ACP forward, construction smoke
julia --project=. battery_sddp.jl

# the consolidated regression suite
julia --project=. test/runtests.jl
```

## Environment variables

Case construction (`build_battery_case.jl`):

| variable | meaning |
|---|---|
| `DR_BAT_CASE` | PGLib case name (default `pglib_opf_case14_ieee`) |
| `DR_BAT_DIR` | output directory for the artifacts |
| `DR_BAT_NUM` | number of batteries to place |
| `DR_BAT_BUSES` | comma-separated explicit bus override; bypasses the seeded draw |
| `DR_BAT_SEED` | seed of the deterministic placement draw |
| `DR_BAT_HORIZON` | frozen horizon |
| `DR_BAT_PROTOCOL_SEED`, `DR_BAT_PROTOCOL_STAGES`, `DR_BAT_PROTOCOL_SCENARIOS` | the evaluation protocol the manifest records a digest for |
| `DR_BAT_MIRROR` | Exa example directory to mirror the case and shared sources into |

SDDP smoke (`battery_sddp.jl`): `DR_BAT_SDDP_STAGES` (3), `DR_BAT_SDDP_ITERATIONS`
(10), `DR_BAT_SDDP_SIMS` (6).

## What the recourse variables are, and are not

Every bus carries two nonnegative, UNCAPPED variables — a deficit injection `d`
and a surplus sink `s` — which enter the ACTIVE nodal balance with opposite
signs and are priced far above any generator. They are physical operating
recourse: they are what makes a dynamically reachable battery target attainable
under the true network, and they are charged identically in the ACP model, the
SOC-WR model, both SDDP passes and the Exa engine.

They are NOT target slack. There is no target-slack variable, no target penalty
and no soft-target formulation anywhere in the supported workflow; the strict
target is a hard equality whose multiplier is the actor signal. A policy that
uses either recourse variable materially is rejected, not priced — and a
diagnostic point that used one is excluded from the evidence rather than
reported.

## Expected outputs

`build_battery_case.jl` prints the case summary and the artifact, support and
protocol digests. `battery_sddp.jl` prints the SOC-WR bound over its own horizon,
the number of stock cuts created, the true-ACP forward cost over the simulated
paths, the worst recourse on any simulated stage, and one battery's energy
trajectory. The bound and the forward cost are quoted over the SAME horizon; a
bound never bounds a metric accumulated over a different number of stages.
