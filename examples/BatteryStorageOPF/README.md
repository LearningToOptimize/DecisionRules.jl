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
| PowerModels.jl | buses, generators, branches, voltage variables, reference angle, Ohm's law at both ends with taps and phase shifts, shunts, angle-difference limits, apparent-power limits at both ends, nodal active and reactive balances, generator bounds and costs, and the `ACPPowerModel` / `SOCWRConicPowerModel` / `DCPPowerModel` formulations themselves |
| SDDP.jl | policy graph, state variables, sampling, cut generation, training, simulation |
| Distributions.jl | the laws an authoring demand sampler is built from |
| this example | batteries: energy state, charge/discharge, state transition, unity-power-factor injection, throughput cost, the strict outgoing-energy target equality, the two-sided nodal active-power recourse, and the demand parameterization |

No AC trigonometry, no SOC-WR lifted branch equation and no DC susceptance-times-
angle-difference flow is written here. The regression suite scans the
model-building files and the diagnostics and fails if one appears.

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
| `battery_sddp.jl` | the SDDP baseline: a SELECTABLE convex backward graph (SOC-WR or DC), true-ACP forward graph, stock cuts, paired simulation on a protocol, the standard cost report, and the study's four method identifiers |
| `battery_portfolio.jl` | the **preregistered PGLib panel**: hash-keyed storage placement, PTDF-sensitivity regions, the finite joint demand support, the ACP headroom calibration, and the manifest a reader regenerates every case from |
| `battery_portfolio.json` | the frozen panel manifest. **Byte-identical copy in the Exa package.** |
| `portfolio_runner.jl` | the production runner: one preemptible SEGMENT of one long SDDP run (`sddp_soc` or `sddp_dc`), with identity binding, verified checkpoints that embed the cuts, resume and a stop protocol |
| `test/runtests.jl` | the consolidated regression suite |
| `case/<name>/` | the constructed artifacts: `network.json`, `batteries.json`, `demand.json`, `case_manifest.json`. **Not committed** — see below. |


## Running a long study: `portfolio_runner.jl`

`train_battery_sddp` is one training call in one process. A study run is longer
than any queue reservation and can be killed at any moment, so it is executed as
a sequence of SEGMENTS, each a separate invocation of `portfolio_runner.jl` that
continues the previous one from a verified checkpoint. The runner adds no
science: the policy graphs, the stock `AlternativeForwardPass` /
`AlternativePostIterationCallback` pair, the cuts, the cost and the paired
protocol evaluation are the same objects `battery_sddp.jl` describes.

```bash
julia --project=. portfolio_runner.jl \
    --case-manifest case/pglib_opf_case118_ieee/case_manifest.json \
    --method        sddp_soc \
    --config        config.toml \
    --protocol      screening.toml \
    --output        run/seg001 \
    --resume-from   none
```

Those six flags are the whole contract; `--run-id`, `--segment`, `--attempt`,
`--stop-file` and `--max-seconds` exist for an automated caller and all default.
The protocol descriptor is written once per case with

```julia
include("portfolio_runner.jl")
write_protocol_descriptor("case/pglib_opf_case118_ieee", "screening.toml")
```

and a descriptor naming the FINAL protocol is refused, both when writing one and
when a run is launched against one — before any scenario is solved.

**Continuation is the only path.** The runner trains in chunks of
`checkpoint_every` iterations, and every chunk rebuilds both graphs from the
frozen case and restores the previous chunk's cuts through
`SDDP.read_cuts_from_file` — whether or not the process was ever interrupted. A
resumed run therefore does not merely resemble the uninterrupted one, it takes
the identical path. `train_battery_sddp` gained one keyword for this,
`resume_cuts`, which reads a tagged cut file into the backward graph AND the ACP
forward graph before the first iteration; a forward graph resumed without the
cuts would decide against an empty cost-to-go. No solver object is serialized
and no SDDP internal is parsed by hand.

Sampling is made a function of the global iteration index the same way: each
chunk's seed is derived from `(seed, iterations already completed)`.

**What a segment writes.** `checkpoints/ck_XXXXXXXX.<tag>.json`, which embeds
the cut set exactly as `SDDP.write_cuts_to_file` produced it together with the
iteration count, the sampling position, the convergence and evaluation histories
and the best admissible true-ACP forward result — plus a `.meta.toml` sidecar
naming its digest, written second so no sidecar can vouch for an unfinished
file. Then `history.csv`, `trajectory.csv`, `evaluation.csv`, `result.toml` and
`identity.toml`. The arm's scalar travels with `bound_name` and
`bound_bounds_acp` in every one of them, so the DC arm's number never acquires
the word "bound" from a column heading.

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
assert_graph_provenance(trained.backward_graph, PowerModels.SOCWRConicPowerModel)
assert_graph_provenance(trained.forward, PowerModels.ACPPowerModel)

sims = simulate_battery_sddp_on(trained, proto;
                                ids = [b.index for b in case.batteries], columns = 1:8)
costs = [sum(s[t][:stage_objective] for t in 1:30) for s in sims]
paired_difference(costs, [panel.results[c].total_cost for c in 1:8])
```

The backward pass is an actual convex PowerModels formulation, the forward pass
an actual `PowerModels.ACPPowerModel`, and the two are joined by SDDP.jl's own
`AlternativeForwardPass` / `AlternativePostIterationCallback`. No
`duality_handler` is overridden and no cut is filtered, retried or reweighted.

`simulate_battery_sddp_on` replays a fixed protocol through `SDDP.Historical`,
which is what makes an SDDP cost **paired** with a perfect-foresight cost and,
later, with a TS-DDR cost.

### Two backward formulations, and what their scalars mean

`backward = :soc` (the default) builds the backward nodes as
`PowerModels.SOCWRConicPowerModel`; `backward = :dc` builds them as
`PowerModels.DCPPowerModel`. That is the only thing the keyword changes: the
forward pass, the demand support and its probabilities, the storage state, the
battery layer, the generator data and the generator cost polynomials, the
uncapped physical recourse and the SDDP machinery are shared, not duplicated.

```julia
dc = train_battery_sddp(case; backward = :dc, num_stages = 24, iteration_limit = 300,
                        cut_path = sddp_cut_path("out", case, :dc))
dc.backward                # :dc
dc.backward_formulation    # PowerModels.DCPPowerModel
dc.bound_name              # "DC-approximation training bound"
dc.bound_bounds_acp        # false
sddp_method_id(dc)         # :sddp_dc
```

The two SCALARS are not the same kind of object and this example never lets them
be printed as if they were:

| arm | scalar | is it a lower bound on the true ACP problem? |
|---|---|---|
| `:soc` | **SOC-WR relaxation bound** | **yes** — the relaxation lower-bounds ACP, so its bound does too |
| `:dc` | **DC-approximation training bound** | **no** — the DC approximation drops the reactive balance and fixes voltage magnitudes, so it is neither a relaxation nor a restriction of ACP and its value bounds nothing in either direction |

The DC scalar is the internal convergence scalar of the approximation the cuts
came from, reported to say whether that training converged. It changes arithmetic
in exactly one place: `cost_report`'s recoverable ceiling `f` drops the bound
term from `max(bound, pf_mean)` on the DC arm, because a number that does not
bound the best nonanticipative cost has no right to tighten a cap on it.

Neither scalar is comparable to a forward objective accumulated over a different
number of stages, and nothing here invites that.

**The arms cannot be confused.** `backward`, `backward_formulation`, `bound_name`
and `bound_bounds_acp` travel with every result; `cost_report` prints the method
identifier and labels row `b` from `bound_name`; and a cut file must carry the
arm's tag in its name — `sddp_cut_path` builds one and `train_battery_sddp`
refuses a path that does not.

The DC arm is validated against a **direct PowerModels DC oracle**: an ordinary
`solve_opf(net, DCPPowerModel, …)` on a network with no storage table at all,
whose per-bus load is the realized demand minus the battery's net injection. It
must agree on generation, generation cost and every branch flow, and the DC nodal
balance and line flows are recomputed independently from the reported angles.

## 13. Where strict TS-DDR enters

Nothing above trains a policy. Strict TS-DDR is the GPU engine's business:
`DecisionRulesExa.jl/examples/BatteryStorageOPF` reads the **same frozen case
bytes and the same frozen support**, builds the multistage true-ACP
deterministic equivalent in ExaModels, evaluates the reachable policy before each
stage, and differentiates the stage value through the strict target multiplier.
This engine's role afterwards is to replay and verify what that engine produced,
through the shared solution schema.

---

# The preregistered PGLib portfolio

Sections 1–13 are the toolkit. The **study** is not one case built by hand with
it: it is a panel of canonical PGLib systems whose every construction choice is
a documented function of the benchmark's own bytes. `battery_portfolio.jl` is
that function, and `battery_portfolio.json` is what it produced.

## Regenerating a case

```bash
julia --project=. battery_portfolio.jl --list                        # the panel
julia --project=. battery_portfolio.jl --verify                      # check the manifest
julia --project=. battery_portfolio.jl --case pglib_opf_case118_ieee --out /tmp/panel
```

The third command acquires the canonical PGLib case, recomputes the regions and
the placement, rebuilds the frozen support at the recorded demand level, writes
the four case artifacts into `/tmp/panel/pglib_opf_case118_ieee/`, and **fails
closed** if the acquired network, the recomputed regions, the recomputed
placement, the frozen support or any written artifact does not hash to what the
manifest records. Add `--verify-all` to re-run the full `24 × 6` headroom gate
as well. No private repository, no cluster scheduler and no pre-generated JSON
is involved.

## What is frozen, and how

| choice | rule |
|---|---|
| horizon | `T = 24`, one hour per stage |
| profile | one common normalized 24-value daily profile, multiplying `pd` AND `qd`, so every realization keeps each load's own power factor |
| eligible buses | in service, positive nominal active demand, and reachable from the reference bus over in-service branches |
| battery count | `min(#eligible, clamp(round(0.20 n_bus), 24, 240))` |
| placement | weighted sampling without replacement, proportional to nominal active demand, with each bus's key drawn from `SHA-256(schema, "placement", seed, network digest, bus)` — no RNG, so the panel survives a reimplementation |
| ratings | 10 % of the calibrated peak active demand, split by nominal demand capped at 3× the selected-bus median; 8 h duration, 5 % reserve, 50 % initial, 0.95/0.95 efficiency, 0.999 self-discharge, throughput cost 5.0 |
| regions | six, **demand-balanced** assignment over unit-norm PTDF sensitivity signatures on the highest-reach rated corridors, relabelled by descending demand; each region carries 8–28 % of nominal demand |
| uncertainty | six equiprobable joint atoms; atom `r` gives region `r` a multiplier of `1.15` and every other region `0.97`, so each region's support mean is exactly `(1.15 + 5×0.97)/6 = 1` and the regions are **negatively** correlated |
| demand level | `κ_case = 0.95 κ_max`, where `κ_max` is the largest level in `[0.50, 1.25]` at which every atom of the peak-profile stage solves in base ACP, unmodified and battery-free, with residual ≤ 1e-7 and no meaningful recourse |
| protocols | a 500-column final panel from one seed, and a 32-column screening panel from an independent seed, **repaired against the final one so the two share no scenario by construction** |

The demand level is the only quantity the manifest carries that a reader cannot
cheaply recompute — it costs a bisection plus a `24 × 6` verification of true-ACP
solves per case — so it is recorded and `--verify-all` re-derives it on demand.
Everything else in the manifest is a digest of something the reader regenerates.

## Why the regions are balanced

Six regions are six LEVERS only if they carry comparable demand. The atoms move
demand by region, so a region holding 0.2 % of the load is an atom that moves
nothing — and an unconstrained clustering does exactly that: on the first freeze
it put 80.8 % of `case1951_rte`'s demand in one region and 61.5 % of
`case300_ieee`'s, collapsing six atoms toward two directions.

The assignment step is therefore an integer program (HiGHS, through JuMP) that
keeps the same PTDF objective and adds the demand bounds:

```math
\min_x \sum_{i,r} w_i \lVert s_i - c_r \rVert^2 x_{ir}
\quad\text{s.t.}\quad
\sum_r x_{ir} = 1,\;
0.08\,W \le \sum_i w_i x_{ir} \le 0.28\,W,\;
x_{ir} \in \{0,1\},
```

refined over Lloyd sweeps so the bounds hold at every sweep rather than only at
the end. A bus is indivisible, so if ONE bus alone exceeds 28 % the cap rises to
exactly that bus's share and a `Σy ≤ 1` constraint lets a single region use it —
the minimum necessary exception, recorded in the manifest.

Balancing did not cost PTDF coherence. Measured over the panel, weighted
within-region signature dispersion went to 0.79–1.10× of the unconstrained
value — better on seven cases — because both are local searches and solving each
assignment step to global optimality lands in a better basin. Both numbers are
recorded per case so the tradeoff is visible rather than assumed.

## The reported cost is not the solver's objective

An interior-point method does not leave a nonnegative variable at zero; it
leaves it a barrier tolerance away, and the sign depends on the solver. The
recourse price is 1e5–1e6 per pu, so 1e-8 pu on a couple of thousand buses is
tens of cost units of pure numerical residue — on a stage where neither engine
used any recourse at all.

`physical_stage_cost`, in the byte-identical `battery_solution_schema.jl`, is the
only function either engine may use to produce a headline cost:

```julia
c = physical_stage_cost(sol, case.recourse)
c.raw          # the solver's own objective, preserved for diagnostics
c.corrected    # generation + throughput + recourse actually charged
c.correction   # the barrier artifact, reported rather than discovered
c.admissible   # false if any element exceeded the physical tolerance
```

Every recourse element within `PHYSICAL_RECOURSE_TOL = 1e-6` pu of zero is
projected to exactly zero, element by element. An element OUTSIDE it is not
projected: the solve is marked inadmissible and the caller rejects it. The
projection changes what is reported, never what was solved — the stage problem
still carries the recourse at full price.

> **The margin is not a knob.** `0.95` is a constant of `battery_portfolio.jl`,
> fixed before any method was run, identical for every case. So is the bracket,
> so is the tolerance, and so is the solver configuration: a case is never
> "helped" to a higher level. A case whose unmodified base ACP fails at `0.50` is
> **replaced** from a preregistered reserve list, and the replacement is recorded
> in the manifest. A SOC, DC or method failure never causes a replacement.

## Validating a frozen case

```julia
include("battery_portfolio.jl")
case = materialize_portfolio_case("pglib_opf_case118_ieee"; dir = "/tmp/panel/c118")
validate_portfolio_case(case; stages = 1:24, atoms = 1:1)   # strict-stage gate
aggressive_charge_probe(case)                                # the diagnostic
```

`validate_portfolio_case` checks that every initial state is inside its bounds,
that every reachable interval is nonempty, that the idle/hold target is feasible,
that every strict ACP solve completes, that the independently recomputed physical
residual is at most 1e-7 and that no solve uses meaningful recourse.

`aggressive_charge_probe` is the opposite: it aims the whole fleet at the top of
its reachable interval on one stage, which is an admissible target the network
may not be able to serve. When it draws recourse, the right outcome is that
admissibility **rejects** the solution — not that the case is changed.

---

## Commands

```bash
# rebuild (and mirror) the correctness case from PGLib
DR_BAT_MIRROR=/path/to/DecisionRulesExa.jl/examples/BatteryStorageOPF \
  julia --project=. build_battery_case.jl

# the portfolio panel
julia --project=. battery_portfolio.jl --list
julia --project=. battery_portfolio.jl --verify
julia --project=. battery_portfolio.jl --case <PGLIB CASE> --out <DIR>

# re-verify a frozen case in place: hashes, schemas, stage duration, support, protocol
julia --project=. build_battery_case.jl --verify

# stock SDDP: SOC-WR backward, true-ACP forward, construction smoke
julia --project=. battery_sddp.jl

# the same smoke with DC backward nodes
DR_BAT_SDDP_BACKWARD=dc julia --project=. battery_sddp.jl

# the consolidated regression suite
julia --project=. test/runtests.jl
```

## The study's four method identifiers

`BATTERY_METHODS` carries the four the study compares, with the same rows and the
same invariant fields in **both** public engines:

| identifier | engine | what varies |
|---|---|---|
| `tsddr_nonlinear` | the Exa engine | LSTM encoder, nonlinear head |
| `tsldr_recurrent_linear` | the Exa engine | affine recurrence, affine head |
| `sddp_soc` | this one | `SOCWRConicPowerModel` backward cuts |
| `sddp_dc` | this one | `DCPPowerModel` backward cuts |

```julia
battery_method(:sddp_dc)                 # the descriptor and the shared invariants
run_battery_method(:sddp_dc, case; num_stages = 24, iteration_limit = 300)
run_battery_method(:tsddr_nonlinear, case)   # refused here: it is the Exa engine's
```

Every row declares the same horizon (24), protocol (screening), strict target
semantics, recourse and admissibility rule, cost contract
(`physical_stage_cost`) and comparison path (true ACP on paired protocol
columns), and each suite asserts it. `run_battery_method` is a dispatch layer,
not a campaign runner: it selects an implementation and forwards keyword
arguments, and it schedules nothing.

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

SDDP smoke (`battery_sddp.jl`): `DR_BAT_SDDP_BACKWARD` (`soc`),
`DR_BAT_SDDP_STAGES` (3), `DR_BAT_SDDP_ITERATIONS` (10), `DR_BAT_SDDP_SIMS` (6).

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
protocol digests. `battery_sddp.jl` prints the method identifier of the arm it
ran, its backward scalar under that arm's own name together with whether that
scalar bounds the true ACP problem, the number of stock cuts created, the
true-ACP forward cost over the simulated paths, the worst recourse on any
simulated stage, and one battery's energy trajectory. The scalar and the forward
cost are quoted over the SAME horizon; a bound never bounds a metric accumulated
over a different number of stages.
