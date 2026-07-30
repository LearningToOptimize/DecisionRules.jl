# Bolivia hydro example

This directory contains the canonical public Bolivia MAIN hydro case. Phase 2A
restores the mature strict-target JuMP implementation and its paired evaluation
infrastructure; it does not contain new SDDP cuts, production training results,
or density comparisons.

The canonical contract is:

- byte-exact MAIN `PowerModels.json`, `inflows.csv`, and repaired `hydro.json`;
- active and reactive MAIN loads both scaled explicitly by `0.6`;
- 168-hour stages and water conversion `K = 0.0036 * 168 = 0.6048`;
- operational active deficit cost `6000 USD/(pu·stage)`, derived from
  `60 USD/MWh * 100 MVA`;
- 96 reporting stages plus 30 look-ahead stages;
- strict reservoir targets mapped into the one-stage reachable interval with
  the stretched sigmoid and a `1e-3` safe upper margin;
- hard reactive nodal balance and apparent-power limits at both branch ends;
- stage-major joint inflow-and-demand paths, with demand atoms
  `{0.9, 1.0, 1.1}`.

`bolivia/case_manifest.json` records canonical hashes, constants, topology,
MOF metadata, and protocol provenance. `bolivia/joint_protocol_500.csv` is the
single 126-by-500 paired protocol for later SDDP and TS-DDR evaluation.

The three Bolivia MOFs are generated once by
`generate_canonical_case_artifacts.jl` and copied byte-for-byte to
DecisionRulesExa.jl. `export_subproblem_mof.jl` fails closed on input hashes,
stage duration, water conversion, topology, and the active-deficit coefficient.

Run Julia commands from an allocated compute node with a prepared project:

```bash
julia --project generate_canonical_case_artifacts.jl \
  --exa-root=/path/to/DecisionRulesExa.jl
julia --project generate_joint_protocol.jl --verify
julia --project test_sampling_consistency.jl
julia --project test_strict_mode.jl
```

The retained July 2026 checkpoint and cost files are provenance references
only. The archived SDDP cost dump sampled inflow alone, and the historical cuts
file is missing, so those artifacts are not final paired evidence.
