# hydro_rl_env.jl — the Bolivia hydrothermal problem as a reinforcement-learning
# environment, built on EXACTLY the problem TS-DDR is evaluated on.
#
# Same problem: the strict JuMP stage subproblems of
# `examples/HydroPowerModels/load_hydropowermodels.jl` (`build_hydropowermodels`,
# strict targets, load factor 0.6, deficit cost 6000, water balance
# K = 0.0036·stage_hours read from hydro.json = 0.6048), the same ACP formulation,
# the same paired evaluation protocol (`paired_scenario_indices`) and the same
# ten-column selection panel TS-DDR is selected on.
#
# Same feasibility layer: an RL action is a vector y ∈ [0,1]^nHyd that is mapped
# into the one-stage reachable interval exactly as the TS-DDR policy maps its
# head output,
#
#     x̂_t = l_t(x_{t-1}, w_t) + (u_t(x_{t-1}, w_t) − l_t(x_{t-1}, w_t)) ⊙ y_t ,
#
# followed by the same cascade clamp, so every action is feasible by construction
# and the only difference between the methods is how y is learned.
#
# One environment per Distributed worker: Ipopt with MUMPS is not thread-safe, so
# parallel episodes run in separate processes, each with its own JuMP models.

using DecisionRules
using JuMP, Ipopt
using Flux
using JSON, JLD2, StableRNGs, Random, Statistics

const HYDRO_DIR = dirname(@__DIR__)   # examples/HydroPowerModels
include(joinpath(HYDRO_DIR, "load_hydropowermodels.jl"))
include(joinpath(HYDRO_DIR, "hydro_reachable_policy.jl"))

"Ten paired-protocol columns TS-DDR is selected on (results/paired500_C3)."
const PANEL_IDS = [2, 39, 81, 119, 130, 156, 200, 206, 378, 493]
"Training horizon (TS-DDR trains on 126 stages: 96 reported + 30 look-ahead)."
const TRAIN_STAGES = 126
"Reported horizon: every cost on the figure is a 96-stage cost."
const REPORT_STAGES = 96

"""
    HydroEnv

One process-local copy of the strict stage problems and the reachability metadata.

# Fields
- `subproblems`, `pin`, `pout`: per-stage JuMP models and their state parameters
  (`pin[t][j]` the incoming volume parameter, `pout[t][j] = (target_param, realized_var)`).
- `samples`: per-stage inflow support, `samples[t][k]` a vector of `(param, value)`.
- `x0`: initial volumes. `xmax`: physical maximum volumes (observation scale).
- `wscale`: per-reservoir largest inflow over the support (observation scale).
- `bounds`: a `HydroReachablePolicy` used ONLY for its reachability metadata.
- `nhyd`, `ncen`: reservoirs and inflow atoms per stage.
"""
struct HydroEnv
    subproblems::Vector{JuMP.Model}
    pin::Vector
    pout::Vector
    samples::Vector
    x0::Vector{Float64}
    xmax::Vector{Float64}
    wscale::Vector{Float64}
    bounds::Any
    meta::Any            # hydro metadata from the builder (to rebuild a TS-DDR policy)
    nhyd::Int
    ncen::Int
end

"""
    build_env(; stages = TRAIN_STAGES) -> HydroEnv

Build the strict ACP stage problems with Ipopt (MUMPS), exactly as
`eval_paired_tsddr.jl` does, and the reachability metadata.
"""
function build_env(; stages::Integer = TRAIN_STAGES)
    opt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "linear_solver" => "mumps")
    sp, pin, pout, samples, x0, xmax, meta = build_hydropowermodels(
        joinpath(HYDRO_DIR, "bolivia"), "ACPPowerModel.mof.json";
        num_stages = stages, optimizer = opt, strict = true)
    nhyd = length(x0)
    wscale = [maximum(abs(samples[t][k][j][2]) for t in 1:stages for k in eachindex(samples[t])) + 1e-9
              for j in 1:nhyd]
    bounds = hydro_reachable_policy(meta, [4])          # metadata carrier only
    return HydroEnv(sp, pin, pout, samples, Float64.(x0), Float64.(xmax), wscale, bounds,
                    meta, nhyd, length(samples[1]))
end

"Inflow values of a stage sample, in sample order (the order TS-DDR's policy reads)."
inflow_values(w) = Float64[v for (_, v) in w]

"""
    observation(env, x_prev, w, t) -> Vector{Float32}

`[x_{t-1}/x_max; w_t/w_scale; t/TRAIN_STAGES]`: the incoming volumes, the realized
inflow of the stage (known when the decision is taken, as for TS-DDR and SDDP), and
the clock normalized by the TRAINING horizon in every episode, so a 96-stage
evaluation sees exactly the clock values training saw. Markov in (state, stage)
for stagewise-independent inflows.
"""
observation(env::HydroEnv, x_prev, w, t) =
    Float32.(vcat(x_prev ./ max.(env.xmax, 1e-9), inflow_values(w) ./ env.wscale, t / TRAIN_STAGES))

"""
    feasible_target(env, y, x_prev, w) -> Vector{Float64}

Map `y ∈ [0,1]^nHyd` into the reachable interval with the TS-DDR feasibility
layer: affine map into `[l, u]` from `_hydro_reachable_bounds`, then the cascade
clamp `_cascade_upper_bounds`. Identical to `(::HydroReachablePolicy)(x)` after
its head.
"""
function feasible_target(env::HydroEnv, y, x_prev, w)
    infl = inflow_values(w)
    lo, up = _hydro_reachable_bounds(env.bounds, infl, Float64.(x_prev))
    # Same range as TS-DDR's output (stretchedsigmoid stops at 1 − 1e-3): a target AT
    # the top of the reachable interval is not solvable in the strict stage. TD3's
    # clipped exploration noise lands exactly on 1.0, and with [0, 1] here every TD3
    # training episode failed from its first exploration iteration on.
    raw = lo .+ (up .- lo) .* clamp.(Float64.(y), 0.0, 1.0 - 1e-3)
    isempty(env.bounds.cascade) && return raw
    return min.(raw, _cascade_upper_bounds(env.bounds, raw, infl, Float64.(x_prev)))
end

"""
    solve_stage!(model) -> Bool

Solve one stage; on failure retry ONCE from a cold start (the evaluation's rule).
"""
function solve_stage!(model)
    accepted(m) = JuMP.termination_status(m) in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
    JuMP.optimize!(model)
    accepted(model) && return true
    for v in JuMP.all_variables(model)
        JuMP.set_start_value(v, nothing)
    end
    JuMP.optimize!(model)
    return accepted(model)
end

"""
    step!(env, t, x_prev, w, target) -> (ok, cost, x_next)

Impose the incoming volumes, the stage inflow and the target, solve the strict
stage, and return the stage cost and the realized outgoing volumes.
"""
function step!(env::HydroEnv, t, x_prev, w, target)
    for (j, p) in enumerate(env.pin[t]); JuMP.set_parameter_value(p, x_prev[j]); end
    for (p, v) in w; JuMP.set_parameter_value(p, v); end
    for j in 1:env.nhyd; JuMP.set_parameter_value(env.pout[t][j][1], target[j]); end
    ok = solve_stage!(env.subproblems[t])
    ok || return (false, NaN, x_prev)
    return (true, JuMP.objective_value(env.subproblems[t]),
            Float64[JuMP.value(env.pout[t][j][2]) for j in 1:env.nhyd])
end

"Training scenario: an independent inflow atom per stage, as TS-DDR's sampler draws."
sample_scenario(env::HydroEnv, rng, T) = [env.samples[t][rand(rng, 1:env.ncen)] for t in 1:T]

"The paired-protocol scenario of global column `col`, first `T` stages."
function panel_scenario(env::HydroEnv, col, T)
    idx = paired_scenario_indices(500, env.ncen)
    return [env.samples[t][idx[t, col]] for t in 1:T]
end

"""
    rollout(env, act, scenario; T, record) -> NamedTuple

Run one episode. `act(obs, x_prev, w, t)` returns `(y, extra)`: the [0,1] action and
anything the algorithm wants stored with it (pre-squash sample, log-probability).
`extra === :target` means `y` is already a TARGET (the parity check drives the
environment with the TS-DDR policy this way). With `record = true` every
transition is returned.

Returns `(ok, cost, cost96, transitions)`: `cost` over all `T` stages, `cost96`
over the first `REPORT_STAGES` (the reported quantity).
"""
function rollout(env::HydroEnv, act, scenario; T = length(scenario), record::Bool = true)
    x = copy(env.x0); cost = 0.0; cost96 = 0.0
    tr = record ? Vector{NamedTuple}(undef, 0) : nothing
    for t in 1:T
        w = scenario[t]
        o = observation(env, x, w, t)
        y, extra = act(o, x, w, t)
        target = extra === :target ? Float64.(y) : feasible_target(env, y, x, w)
        ok, c, xn = step!(env, t, x, w, target)
        ok || return (ok = false, cost = NaN, cost96 = NaN, transitions = tr)
        cost += c; t <= REPORT_STAGES && (cost96 += c)
        if record
            on = t < T ? observation(env, xn, scenario[t + 1], t + 1) : o
            push!(tr, (obs = o, y = Float32.(y), extra = extra, cost = c, next_obs = on, done = t == T))
        end
        x = xn
    end
    return (ok = true, cost = cost, cost96 = cost96, transitions = tr)
end
