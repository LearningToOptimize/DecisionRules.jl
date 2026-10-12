# Examples

The examples reproduce the case studies of the paper and include smaller
demonstrations of the package. Each directory has its own `Project.toml`; the
two power-system case studies are split between this package, which holds the
JuMP engine and the baselines, and
[DecisionRulesExa.jl](https://github.com/LearningToOptimize/DecisionRulesExa.jl),
which holds the GPU trainer, under the same directory names.

## Reproducing the case studies

The scripts reproduce the method, not the exact numbers of the paper. The stage
problems are nonconvex AC power flows solved to local optimality, and the policy
is a deep network trained by stochastic gradient descent, so the solver, the
sampled trajectories and the hardware all influence where a run comes to rest.
Two runs of the same schedule reach different policies of similar quality. As in
any hard deep-learning problem, a good policy comes from a schedule of training
phases rather than from one uninterrupted run: each phase restarts from the best
checkpoint of the previous one with a fresh learning-rate schedule, saves its own
best policy on a fixed selection panel, and can be resumed exactly if it is
interrupted. The default schedules shipped with the examples are a good start and
give a run a fair chance of finding a good policy; the documentation page
[Reproducing the study](https://LearningToOptimize.github.io/DecisionRules.jl/dev/casestudies/hydro/reproducing/)
explains how to adjust them when a run stalls. Every method is evaluated on the
same paired scenario protocol, so results obtained by different methods, engines
or runs are directly comparable.

## Hydrothermal planning

`HydroPowerModels/` holds the Bolivian hydrothermal case, its verifier, the JuMP
stage models, the SDDP baseline in `sddp/` and the reinforcement-learning
baselines in `rl/`. The TS-DDR and linear decision-rule policies are trained on a
GPU by the companion package's `examples/HydroPowerModels`, whose lineage driver
runs the schedules `lineage_from_scratch.json` and
`lineage_tsldr_from_scratch.json`. A complete study verifies the case, trains
SDDP and the learned policies, and evaluates all of them on the shared 500
scenario protocol; the README of each directory gives the commands in that order.

## Battery storage

`BatteryStorageOPF/` builds battery-storage cases on PGLib networks, defines the
demand uncertainty and the battery physics, and trains SDDP with a relaxed or
linearized backward model and an AC forward pass. Its counterpart in
DecisionRulesExa.jl trains TS-DDR and the linear decision rule on the same
frozen cases on a GPU.

## Smaller examples

`inventory_control/` is a stochastic lot-sizing problem with fixed ordering
costs, in a relaxed and an integer version, compared with dynamic programming and
SDDP; it shows how the dual gradient is combined with a score-function estimate
when the stage problem has integer variables. `rocket_control/` maximizes the
altitude of a Goddard rocket under stochastic wind and compares the learned
policy with model-predictive control. `Experimental/` holds research prototypes
that are not documented, including an earlier reinforcement-learning benchmark
on the hydro case.

## Running an example

Activate and instantiate the example's environment before running it:

```julia
using Pkg
Pkg.activate("examples/inventory_control")
Pkg.instantiate()
include("examples/inventory_control/train_dr_inventory.jl")
```

`slurm.jl` starts Distributed workers on a SLURM allocation through
`ClusterManagers` and then includes a target script.
