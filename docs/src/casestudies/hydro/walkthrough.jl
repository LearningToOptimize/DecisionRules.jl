# # Walkthrough
#
# This walkthrough builds the Bolivian hydrothermal planning problem, constructs
# the feasibility-guaranteeing policy that strict TS-DDR needs, and takes a few
# gradient steps — on CPU, in a few minutes, on a horizon short enough to watch.
#
# It is deliberately *not* the published run. That took eleven GPU-hours over 126
# stages. The point here is that every piece of it is visible in a script you can
# execute, and that the pieces are the ones the result depends on. Where this
# simplifies, it says so.
#
# The case, the numbers and the honest reading of the comparison are in
# [Results](@ref); the mathematics is in
# [The long-term hydrothermal planning problem](@ref).

# ## Setup
#
# Run from `examples/HydroPowerModels`, whose `Project.toml` carries everything
# used below.

using DecisionRules
using JuMP, DiffOpt, Ipopt
using Flux
using Random
using Statistics

HYDRO_DIR = joinpath(dirname(dirname(dirname(@__DIR__))), "examples", "HydroPowerModels") #hide
nothing #hide

# ## 1. The system
#
# The case is the Bolivian national grid operated over weekly stages: a
# transmission network with thermal units, and a set of reservoirs linked into
# cascades, driven by historical inflow scenarios. Three files describe it — the
# network, the hydro topology, and the inflows.

CASE_DIR = joinpath(HYDRO_DIR, "bolivia")

# ## 2. Build the stage problems
#
# `build_hydropowermodels` reads one serialized stage model per stage — produced
# from the case by `export_subproblem_mof.jl` through HydroPowerModels — and
# re-parameterizes it: the incoming reservoir state becomes a parameter, the
# inflow becomes a parameter, and in **strict** mode the outgoing state is bound
# to a target parameter by a hard equality.
#
# A short horizon keeps this runnable; the published run uses 126.

include(joinpath(HYDRO_DIR, "load_hydropowermodels.jl"))
include(joinpath(HYDRO_DIR, "hydro_reachable_policy.jl"))

NUM_STAGES = 3

diff_optimizer = () -> DiffOpt.diff_optimizer(
    optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0),
)

subproblems, state_params_in, state_params_out, uncertainty_samples,
    initial_volumes, max_volume, hydro_meta = build_hydropowermodels(
    CASE_DIR, "ACPPowerModel.mof.json";
    num_stages = NUM_STAGES, optimizer = diff_optimizer, strict = true,
)

(stages = length(subproblems), reservoirs = hydro_meta.nHyd,
 inflow_scenarios = length(uncertainty_samples[1]), K = hydro_meta.K)

# In strict mode there are **no slack variables on the target** and no penalty
# term. The dual of `reservoir_out == target` is therefore the clean marginal
# value of water, ``\partial Q_t / \partial \hat v_t``, with no penalty noise
# mixed into it. That is the entire reason strict mode exists.
#
# It is only well posed if every target the policy emits is reachable in one
# stage — otherwise the equality makes the stage infeasible. Hence the policy.

# ## 3. The reachable policy
#
# `hydro_reachable_policy` maps an unconstrained network output into the
# one-stage reachable interval of each reservoir,
#
# ```math
# \hat v_r \;=\; \ell_r(v, w) + \bigl(u_r(v, w) - \ell_r(v, w)\bigr)\,\sigma(z_r),
# ```
#
# and then applies a cascade clamp, so a downstream target can never assume more
# water than the upstream unit actually released.
#
# Two details carry the published result:
#
# * the activation is a **stretched** sigmoid onto `[0, 1 - 1e-3]`, not a plain
#   one. A plain sigmoid cannot attain the ends of the interval, and the good
#   policy on this case puts a substantial share of its targets exactly at a
#   feasibility extreme;
# * the bounds ``\ell_r, u_r`` depend on the incoming state, and that dependence
#   **is differentiated**. Treating it as constant still trains and still lowers
#   the loss, while descending a direction 48 degrees off the true gradient.

Random.seed!(42)
policy = hydro_reachable_policy(hydro_meta, [128, 128]; combiner_layers = [256, 256])
nothing #hide

# The encoder is an LSTM over the **inflow** sequence only; the reservoir state
# enters through the state-conditioned head, not through the recurrence.

# ## 4. One rollout
#
# A rollout threads the realized state: the policy sees the state it actually
# reached, emits a target, the stage problem projects that target onto the
# feasible set, and the realized outgoing state becomes the next stage's input.

# Written as a function rather than a bare loop: at top level (and inside a
# documentation `@example` block) a `for` introduces its own scope, so a
# loop-carried `total_cost += ...` would fail with `UndefVarError`. This is a
# recurring Julia trap in exactly this kind of script.

function rollout(policy, scenario)
    state = Float64.(initial_volumes)
    total = 0.0
    for t in 1:NUM_STAGES
        for (j, param) in enumerate(state_params_in[t])
            set_parameter_value(param, state[j])
        end
        for (param, val) in scenario[t]
            set_parameter_value(param, val)
        end

        w = Float32.([val for (_, val) in scenario[t]])
        target = policy(vcat(w, Float32.(state)))
        for j in 1:hydro_meta.nHyd
            set_parameter_value(state_params_out[t][j][1], Float64(target[j]))
        end

        optimize!(subproblems[t])
        @assert termination_status(subproblems[t]) in
                (MOI.LOCALLY_SOLVED, MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
        total += objective_value(subproblems[t])

        for j in 1:hydro_meta.nHyd
            state[j] = value(state_params_out[t][j][2])
        end
    end
    return total, state
end

scenario = [uncertainty_samples[t][1] for t in 1:NUM_STAGES]
total_cost, final_state = rollout(policy, scenario)
total_cost

# Every stage solved, from an untrained policy: the reachable map did its job and
# no emitted target was infeasible. That property is what makes hard target
# equalities usable at all.

# ## 5. The marginal value of water
#
# The multiplier of the target equality is what TS-DDR differentiates, and it
# comes straight off the solved stage — no extra machinery:

lambda = [DecisionRules.pdual(state_params_out[NUM_STAGES][j][1])
          for j in 1:hydro_meta.nHyd]
round.(lambda; digits = 3)

# A negative entry means holding one more unit in that reservoir *lowers* future
# cost — water has value there. The spread across reservoirs is the locational
# content that the production factors and the cascade topology create: a cubic
# metre of water is not a fungible commodity.

# ## 6. A few training steps
#
# `train_multistage` assembles the loop: sample a scenario, roll out, collect the
# multipliers, backpropagate through the policy, step the optimizer. Three
# iterations here; the published run took 890 across three restarted stages.

# `uncertainty_samples` is passed DIRECTLY: `DecisionRules.sample` has an
# overload for it that draws one inflow scenario per stage, which is exactly the
# per-stage joint sampling this case needs.

DecisionRules.train_multistage(
    policy, initial_volumes, subproblems,
    state_params_in, state_params_out, uncertainty_samples;
    num_train_per_batch = 2,
    num_batches = 3,
    optimizer = Flux.Adam(1e-3),
)

# Three updates on three stages will not produce a good policy, and the number
# above is not meaningful on its own — it is one noisy sample of a stochastic
# objective, and the trap this case study keeps returning to: the training loss,
# the training objective and the fixed-panel evaluation are three different
# quantities, and only the last one selects a policy.

# ## Where to go from here
#
# The published run is the same construction at full scale — a longer horizon,
# a training schedule of several phases, and a GPU. How each method arrives at a
# price for water is [Valuing water: two approaches](@ref); what the comparison
# measured, and what it does and does not say, is [Results](@ref).
#
# To actually run it, the example READMEs of `DecisionRules.jl` and
# `DecisionRulesExa.jl` carry the commands, from verifying the case through to
# regenerating the figures.
