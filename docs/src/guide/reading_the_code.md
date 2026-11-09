# Reading the code

```@meta
CurrentModule = DecisionRules
```

The framework is split between two packages that implement the same algorithm.
DecisionRules.jl expresses every stage problem in JuMP and covers all three
training formulations: the deterministic equivalent, the stage-wise rollout and
the windowed decomposition. DecisionRulesExa.jl expresses the deterministic
equivalent in ExaModels and solves it with MadNLP, on the CPU or on a GPU. The
examples sit on top of both and are where the case studies live. This page is a
reading guide: it follows one training step through the code and then says where
each piece of an example is.

## One training step

Every training step does the same four things, whichever package runs it. It
samples an uncertainty trajectory; it runs the policy along that trajectory to
produce a target for every stage; it solves the optimization problem that
projects those targets onto the feasible set, either one stage at a time or over
the whole horizon at once; and it reads the multipliers of the target
constraints, which are the derivatives of the realized cost with respect to the
targets, and pulls them back through the policy with automatic differentiation.
The optimizer then updates the policy parameters. Nothing is differentiated
through the solver's iterations: the solver supplies a value and its multipliers,
and the network supplies the rest of the chain rule.

## DecisionRules.jl

The module file `src/DecisionRules.jl` lists the exports and includes the other
files, and is the natural place to start. The modelling helpers are in
`src/utils.jl`: `variable_to_parameter` turns a state variable of a stage model
into a target parameter tied to it by a constraint, `create_deficit!` adds the
slack used by the penalized formulation, and `deterministic_equivalent!` stitches
the stage models into one model over the horizon. The same file holds the
evaluation tools, `RolloutEvaluation` and `SaveBest`, which score a policy on a
fixed set of scenarios and keep the best checkpoint.

The training loops are in `src/simulate_multistage.jl`. `simulate_multistage`
rolls a policy out, and `train_multistage` trains it, with one method for a vector
of stage models and one for a deterministic equivalent. In the stage-wise
formulation the realized state of each stage feeds the next, so the gradient has
to flow through the stage solution itself; this is the reverse rule on
`get_next_state`, which seeds DiffOpt with the incoming derivative and reads the
sensitivities of the parameters. In the deterministic equivalent only the target
multipliers are needed, and `src/parameter_duals.jl` computes them from the duals
of the constraints in which each parameter appears. The windowed decomposition,
which solves a deterministic equivalent per window and chains windows through
their realized end states, is self-contained in `src/multiple_shooting.jl`.

The policies are in `src/dense_multilayer_nn.jl`: dense heads, recurrent
encoders of the uncertainty, and the state-conditioned policy that combines an
encoding of the uncertainty observed so far with the current state. Problems with
integer decisions are handled by `src/integer_strategies.jl`, which decides how
dual information is read from a mixed-integer stage model, and by
`src/score_function.jl`, which mixes the dual gradient with a score-function
estimate where duals are not informative.

## DecisionRulesExa.jl

The GPU package keeps the deterministic equivalent and drops everything else, so
it is shorter. `src/deterministic_equivalent.jl` builds a generic deterministic
equivalent as an ExaModel with three parameter blocks, for the initial state, the
targets and the uncertainty, and exposes `set_x0!`, `set_targets!`,
`set_uncertainty!` and `solve!`; updating a parameter changes the problem without
rebuilding it, which is what makes repeated solves cheap. The training loop is
`train_tsddr` in `src/training.jl`. Besides the step described above, it handles
what a long run on a nonconvex problem needs: warm starts between solves, a retry
from a fresh solver state when a solve fails, several independent problems solved
in parallel for the samples of one step, and the hooks through which a caller
checkpoints the optimizer state and continues an interrupted run exactly.
`src/rollout.jl` evaluates a policy stage by stage on fixed scenarios, and
`src/policy.jl` holds the same policy families as the JuMP package. The two
remaining files are extensions: `src/embedded_deterministic_equivalent.jl` embeds
the policy inside the optimization model, and `src/critic_control_variate.jl`
adds a learned critic as a control variate on the dual gradient.

## How an example is organized

The hydrothermal example, `examples/HydroPowerModels` in both packages, is the
reference for how a case study is laid out, and the battery example follows the
same pattern. A case is frozen data together with a manifest that records the
hash of every input and the invariants derived from them; a verifier checks the
manifest, and every builder refuses a case that does not verify. Each engine has
its own model builder: the JuMP engine reads a serialized stage model per
formulation and reparameterizes it, while the ExaModels engine builds the
deterministic equivalent directly from the case data. The policy that guarantees
feasibility is a separate file, because it is specific to the physics of the
case: it maps the network's output into the interval of targets that the stage
can actually reach from its incoming state.

Training is split between a trainer and a driver. The trainer runs one training
phase and is configured entirely by environment variables, so that a phase is a
self-contained process that can be interrupted, resumed and recorded. The driver
reads a schedule of phases from a JSON file, runs them in order, chains only the
checkpoints that a complete evaluation selected, and keeps a record of every
phase. Evaluation is a separate script that scores a checkpoint on the case's
paired scenario protocol, so that every method, learned or not, is measured on
the same scenarios. The baselines sit in subdirectories of the example: SDDP in
`sddp/`, with its own environment, and the reinforcement-learning agents in
`rl/`.

The two smaller examples of this package, `examples/inventory_control` and
`examples/rocket_control`, are single-directory scripts that build the problem,
train a policy and compare it with a reference method. They are the quickest way
to see the package's interface at work before reading a case study.
