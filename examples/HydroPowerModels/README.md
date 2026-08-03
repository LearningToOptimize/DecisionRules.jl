# Bolivia long-term hydrothermal scheduling — JuMP engine

The published case study: a strict TS-DDR policy, trained from random
initialisation, operating the Bolivian interconnected system over 96 weekly
stages under true AC power flow, compared against an SDDP baseline on the same
500 inflow scenarios.

This directory holds the case, the JuMP/MathOptFormat engine, the SDDP baseline,
the paired evaluation, and the figures. The GPU trainer that produced the
published policy lives in the companion package,
`DecisionRulesExa.jl/examples/HydroPowerModels`; the two share the case bytes,
the protocol, and several source files byte for byte.

Narrative and mathematics: [`docs/src/casestudies/hydro_problem.md`](../../docs/src/casestudies/hydro_problem.md)
and [`hydro_bolivia.md`](../../docs/src/casestudies/hydro_bolivia.md). An
executable walkthrough is [`docs/src/examples/hydro.jl`](../../docs/src/examples/hydro.jl).

## The frozen case

| | |
|---|---|
| network, generators, costs, limits, nominal load | `bolivia/PowerModels.json` |
| hydro topology, bounds, production factors, `stage_hours` | `bolivia/hydro.json` |
| historical inflow scenarios | `bolivia/inflows.csv` |
| machine-readable contract and its verifier | `bolivia/case_manifest.json`, `generate_canonical_case_artifacts.jl` |

- 28 buses, 31 branches, 34 generators, 11 hydro units.
- Weekly stages: `stage_hours = 168`, so the water balance converts flow to
  volume with `K = 0.0036 × 168 = 0.6048`. A run that leaves `stage_hours` at
  its default of 1 silently models a week as an hour; both the manifest verifier
  and the stage-model loader fail closed on that.
- Demand is **deterministic**: `0.6 ×` the `PowerModels.json` active *and*
  reactive load at every stage. There are no demand atoms and no demand file —
  the verifier fails if one appears.
- Uncertainty is **inflow only**.
- Reservoirs start **empty**. `hydro.json` carries denormal `initial_volume`
  values near `9e-316`; both engines clamp the initial state into
  `[min_volume, max_volume]` and evaluate it at working precision, which leaves
  every reservoir at zero. That is the state the published result was produced
  from, and the input bytes are not repaired.
- Physical load shedding is the per-bus active-balance slack `deficit[b]`,
  priced at `6000 USD/(pu·stage)` (`60 USD/MWh × 100 MVA`). Reactive balance is
  **hard** — there is no reactive slack anywhere.
- 126 stages are simulated; costs are reported over the first 96. The 30-stage
  tail is a look-ahead buffer that keeps the reported window free of
  end-of-horizon reservoir dumping.
- The paired protocol is reproducible by construction rather than stored:
  entry `[t, s]` of `rand(StableRNG(20260706), 1:nCen, 126, 500)` is the inflow
  scenario realized at stage `t` of paired column `s`. The manifest records a
  SHA-256 of that index matrix.

## Layout

| file | role |
|---|---|
| `generate_canonical_case_artifacts.jl` | the frozen-case contract and its verifier; writes `bolivia/case_manifest.json`; byte-identical in both packages |
| `export_subproblem_mof.jl` | the ONLY supported producer of `bolivia/*.mof.json` — builds the case through HydroPowerModels and serializes one stage subproblem per formulation |
| `load_hydropowermodels.jl` | reads a serialized stage model per stage and re-parameterizes it into incoming state, inflow and target; the JuMP engine's model builder |
| `hydro_reachable_policy.jl` | the feasibility-guaranteeing policy: LSTM encoder over inflow, state-conditioned head, targets mapped into the one-stage reachable interval |
| `hydro_reachable_reference.jl` | an engine-independent oracle for that map, used by the gradient regression tests; byte-identical in both packages |
| `hydro_solution_schema.jl` | the shared long-format solution schema both engines write, so their full solutions can be differenced; byte-identical in both packages |
| `train_dr_hydropowermodels_strict.jl` | strict TS-DDR training on CPU (the smoke path; the published policy was trained with the GPU engine) |
| `eval_paired_tsddr.jl` | paired evaluation of a checkpoint through the JuMP stage models |
| `eval_jump_de.jl` | full-horizon deterministic-equivalent cross-check |
| `verify_full_solution_parity.jl` | the full-solution parity gate — replays a recorded decision trace and differences every physical variable |
| `plot_hydro_results.jl` | the publication figures, from `results/` |
| `sddp/run_sddp_inconsistent.jl` | the SDDP baseline: SOC-WR backward, true-ACP forward |
| `sddp/eval_paired_sddp.jl` | paired evaluation of the frozen cut policy |
| `sddp/merge_sddp_shards.jl` | shard merge; refuses gaps, duplicates and partial sets |
| `sddp/sddp_ac_starts.jl` | non-singular voltage starts for the ACP forward graph |
| `test/runtests.jl` | the example's regression suite |
| `results/` | the compact published evidence the figures and the documentation are built from |

## Commands

Every command below is run from this directory. `--project=.` uses
`Project.toml`; the SDDP scripts use `--project=sddp`, which additionally
carries HydroPowerModels, PowerModels, SDDP and Clarabel.

**1. Verify the case and regenerate the stage models.**

```bash
julia --project=. generate_canonical_case_artifacts.jl --verify
julia --project=sddp export_subproblem_mof.jl \
    --exa-root=/path/to/DecisionRulesExa.jl
```

The exporter verifies the three input hashes, applies the 0.6 load factor at
model construction, passes `stage_hours` into HydroPowerModels, re-reads each
serialized model and asserts its invariants (including `K = 0.6048`), rewrites
the manifest from the generated bytes, and mirrors the whole case into the other
engine. It is byte-reproducible: two runs produce identical files.

**2. Run the regression suite.**

```bash
julia --project=. test/runtests.jl              # everything
julia --project=. test/runtests.jl case merge   # the fast groups
```

**3. Small CPU smoke test** — a few stages, a few updates, no GPU:

```bash
DR_NUM_STAGES=4 DR_NUM_EPOCHS=2 DR_NUM_BATCHES=5 \
  julia --project=. train_dr_hydropowermodels_strict.jl
```

**4. SDDP baseline.** Training writes cuts to
`bolivia/ACPPowerModel/SOCWRConicPowerModel-ACPPowerModel.cuts.json`:

```bash
julia --project=sddp -t auto sddp/run_sddp_inconsistent.jl
```

**5. Paired evaluation of the frozen SDDP policy** (shardable; ids are GLOBAL
protocol columns, so shards and a full run agree exactly):

```bash
DR_SCENARIO_FIRST=1 DR_SCENARIO_LAST=25 DR_PHYSICAL_AUDIT=1 \
  julia --project=sddp -t auto sddp/eval_paired_sddp.jl
julia --project=sddp sddp/merge_sddp_shards.jl \
    --dir=bolivia/ACPPowerModel --first=1 --last=500
```

**6. Paired evaluation of a TS-DDR checkpoint through the JuMP stage models:**

```bash
julia --project=. -t auto eval_paired_tsddr.jl /path/to/checkpoint.jld2
```

**7. Figures:**

```bash
julia --project=. plot_hydro_results.jl
```

## Full-solution parity

Aggregate agreement between engines is not evidence that they solve the same
model: two different feasible sets can price the same operating point
identically. `verify_full_solution_parity.jl` replays a recorded decision
trace — incoming state, realized inflow and reservoir target, one row per
`(scenario, stage, reservoir)` — through the serialized stage model and
differences EVERY physical primal variable against the engine that produced the
trace: storage in and out, inflow, turbine outflow, spill, thermal active and
reactive dispatch, bus voltage magnitudes and angles, nodal generation
aggregates, branch flows at both ends, load shedding, the strict target
multipliers, the stage and cumulative objectives, constraint residuals and
solver status.

```bash
julia --project=. verify_full_solution_parity.jl \
    --trace=<engine>_trace.csv \
    --reference=<engine>_solution.csv \
    --out=replay_solution.csv \
    --label=my_run
```

Traces come from `eval_paired_sddp.jl` with `DR_SOLUTION_DUMP=1` and from the
ExaModels engine's `eval_paired_exa.jl` with the same flag.
