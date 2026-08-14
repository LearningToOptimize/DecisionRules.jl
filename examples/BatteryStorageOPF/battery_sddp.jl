# battery_sddp.jl
#
# The SDDP baseline: stock SDDP.jl over the shared PowerModels battery problem
# specification, with a SOC-WR backward pass and a true-ACP forward pass.
#
# WHAT IS STOCK AND WHY IT MATTERS.
# SDDP.jl owns the policy graph, the state variables, the sampling scheme, cut
# generation, training and simulation. This file constructs two policy graphs
# from the SAME `build_battery_opf` specification — one instantiated as an
# actual `PowerModels.SOCWRConicPowerModel`, one as an actual
# `PowerModels.ACPPowerModel` — and hands them to SDDP's own
# `AlternativeForwardPass`/`AlternativePostIterationCallback` pair, which is
# SDDP.jl's documented mechanism for training against a convex relaxation while
# operating a nonconvex model. No `duality_handler` is overridden, no cut is
# filtered, retried or reweighted: which duals become cuts is SDDP's decision.
#
# The bound produced here is a bound on the SOC-WR relaxation over the horizon
# it was trained on. It is NOT comparable to a forward objective over a shorter
# horizon, and nothing in this file invites that comparison.
#
# Usage
#   julia --project=. battery_sddp.jl                  # short construction smoke
#   DR_BAT_SDDP_STAGES=3 DR_BAT_SDDP_ITERATIONS=10 julia --project=. battery_sddp.jl

using SDDP
using JuMP
using PowerModels
using Ipopt
using Clarabel
using Printf
using Random
using Statistics

@isdefined(BatterySpecification) || include(joinpath(@__DIR__, "battery_powermodels.jl"))
@isdefined(perfect_foresight_panel) || include(joinpath(@__DIR__, "battery_diagnostics.jl"))

"The relaxation SDDP's backward pass values stored energy with."
const BACKWARD_FORMULATION = PowerModels.SOCWRConicPowerModel

"The true model SDDP's forward pass operates."
const FORWARD_FORMULATION = PowerModels.ACPPowerModel

"""
    sddp_path_cost(case, path, T) -> Float64

The cost of one simulated path: its `T` stage objectives summed.

# Notes
Both policy graphs are built by the shared specification, so every stage
objective SDDP sees — and therefore every cut, every `stage_objective` a
simulation records and the bound `SDDP.calculate_bound` returns — is the physical
stage cost of the case, in the objective units of the underlying PGLib network.
Stage costs, cuts and bounds live in that one unit system on both graphs, so a
path cost is a plain sum and a number leaving this file needs no conversion.

`case` is taken for the signature every consumer already calls and to keep the
path bound to the case it was simulated on; the sum itself does not need it.
"""
sddp_path_cost(case::BatteryCase, path, T::Integer) =
    sum(path[t][:stage_objective] for t in 1:Int(T))

"""
    battery_policy_graph(case, model_type; num_stages, optimizer) -> SDDP.PolicyGraph

Build a `num_stages`-stage linear policy graph whose stage problem is the shared
battery problem specification instantiated as `model_type`.

# Arguments
- `case::BatteryCase`: frozen case.
- `model_type::Type`: `SOCWRConicPowerModel` or `ACPPowerModel`.

# Keywords
- `num_stages::Integer`: horizon.
- `optimizer`: solver factory matching `model_type`.

# Returns
- The `SDDP.PolicyGraph`. Each node's JuMP model additionally carries
  `sp.ext[:battery]` (the battery layer's variable references) and
  `sp.ext[:pm]` (the live PowerModels object, so provenance can be asserted
  from the graph itself).

# Notes
Battery energy is the genuine SDDP state: `@variable(sp, ..., SDDP.State)`
creates the incoming/outgoing pair, SDDP equates one stage's outgoing value to
the next stage's incoming value, and the battery transition — written by the
shared specification, not here — is what ties that state to the dispatch.

Demand is the node's NOISE. `SDDP.parameterize` moves the fixed demand-deviation
variables the specification installed; the atoms and their probabilities come
from the frozen case, so the forward graph, the backward graph and the Exa
engine all face the same stochastic program.

Information timing is the plan's: the incoming energy and the realized demand
are both known when the stage decision is taken, and the outgoing energy is the
decision that becomes the next stage's state.
"""
function battery_policy_graph(case::BatteryCase, model_type::Type;
                              num_stages::Integer,
                              optimizer)
    ids = [b.index for b in case.batteries]
    byid = Dict(b.index => b for b in case.batteries)
    num_stages <= horizon(case.demand) ||
        throw(ArgumentError("policy graph asks for $num_stages stages but the frozen support covers $(horizon(case.demand))"))

    graph = SDDP.LinearPolicyGraph(
        stages = num_stages,
        sense = :Min,
        # Every stage cost is a sum of nonnegative terms (generation cost with
        # nonnegative coefficients, throughput cost, priced recourse), so 0 is a
        # valid and honest lower bound on the value function.
        lower_bound = 0.0,
        optimizer = optimizer,
    ) do sp, t
        JuMP.@variable(sp, e[k in ids], SDDP.State,
                       initial_value = byid[k].energy_initial,
                       lower_bound = byid[k].energy_min,
                       upper_bound = byid[k].energy_max)
        state_in = Dict{Int,Any}(k => e[k].in for k in ids)
        state_out = Dict{Int,Any}(k => e[k].out for k in ids)

        pm = battery_stage_model(case, model_type;
                                 jump_model = sp,
                                 stage = t, atom = 1, mode = :sddp,
                                 state_in = state_in, state_out = state_out)
        sp.ext[:pm] = pm
        sp.ext[:battery] = pm.ext[:battery]

        # The stage cost is exactly the objective the shared specification
        # assembled: stock PowerModels generation cost plus the battery layer's
        # throughput and recourse prices.
        SDDP.@stageobjective(sp, JuMP.objective_function(sp))

        bat = pm.ext[:battery]
        pd_nom, qd_nom = bat[:pd_nom], bat[:qd_nom]
        buses = bat[:buses]
        # The stage's atoms come from the FROZEN support, per stage: the support
        # is stage-dependent in general, so a graph-wide atom list would be
        # wrong on any case whose late stages carry a different support. The
        # realized per-bus demand of each atom is materialized ONCE here, so a
        # parameterize call is a few `fix`es rather than a re-aggregation of the
        # case's loads on every node solve of every iteration.
        atoms = collect(1:num_atoms(case.demand, t))
        probs = copy(atom_probabilities(case.demand, t))
        realized = [realized_bus_demand(case, t, k) for k in atoms]
        SDDP.parameterize(sp, atoms, probs) do ω
            pd, qd = realized[ω]
            for i in buses
                JuMP.fix(bat[:dpd][i], pd[i] - pd_nom[i]; force = true)
                JuMP.fix(bat[:dqd][i], qd[i] - qd_nom[i]; force = true)
            end
            return
        end
    end
    return graph
end

"""
    assert_graph_provenance(graph, expected::Type)

Assert that every node of `graph` was instantiated as an actual `expected`
PowerModels model.

# Notes
Checking the type on one node would not do: a policy graph builds one model per
stage and a formulation switch that only fired on some stages would be a
different stochastic program with the same name.
"""
function assert_graph_provenance(graph::SDDP.PolicyGraph, expected::Type)
    for (key, node) in graph.nodes
        haskey(node.subproblem.ext, :pm) ||
            error("node $key carries no PowerModels object")
        assert_powermodels_provenance(node.subproblem.ext[:pm], expected)
    end
    return nothing
end

"""
    train_battery_sddp(case; num_stages, iteration_limit, time_limit, seed,
                       print_level, backward_optimizer, forward_optimizer,
                       protocol, evaluate_columns, evaluate_every, cut_path)
        -> NamedTuple

Train the SDDP policy: SOC-WR backward, true-ACP forward.

# Keywords
- `num_stages::Integer`: horizon.
- `iteration_limit`, `time_limit`: the training budget. Either may be `nothing`.
- `seed::Integer`: seed of the forward sampling scheme.
- `backward_optimizer`, `forward_optimizer`: solver factories. These are
  SOLVER settings; nothing here may change the mathematical model.
- `protocol`, `evaluate_columns`, `evaluate_every`: when all three are given,
  the TRUE-ACP forward cost is evaluated on those fixed protocol columns every
  `evaluate_every` iterations and recorded in `history`.
- `cut_path`: when given, the cuts are written there after training.

# Returns
A `NamedTuple` with the two graphs, the termination status, the SOC-WR bound, the
elapsed wall time, and `history` — one row per evaluation with the iteration, the
bound at that point, the true-ACP mean cost on the fixed columns, the worst
recourse and the elapsed time.

# Notes
`SDDP.AlternativeForwardPass(forward)` performs each forward pass on the ACP
graph and copies the resulting states back into the convex graph, and
`AlternativePostIterationCallback(forward)` copies the newly created cuts into
the ACP graph. Both are SDDP.jl's own; the pair is what makes "value with a
relaxation, operate the true model" a stock configuration rather than an
intervention. The periodic evaluation is composed AROUND that callback, never
in place of it.

**Three signals, never compared in level.** The SOC-WR bound is a bound on the
RELAXATION over `num_stages` stages. The true-ACP forward cost on the fixed
columns is a physical cost of the current policy on a small, fixed sample. A
final protocol cost is a third thing again. Quoting the bound beside a forward
cost accumulated over a different number of stages compares two different
quantities, and nothing here invites that.

The evaluation is a MEASUREMENT, not a stopping rule: training runs its declared
budget. Choosing when to stop by watching the evaluation would select a policy on
the same numbers used to report it.
"""
function train_battery_sddp(case::BatteryCase;
                            num_stages::Integer = 3,
                            iteration_limit = 10,
                            time_limit = nothing,
                            seed::Integer = 20260804,
                            print_level::Integer = 0,
                            backward_optimizer = socwr_optimizer(),
                            forward_optimizer = acp_optimizer(),
                            protocol = nothing,
                            evaluate_columns = nothing,
                            evaluate_every = nothing,
                            cut_path = nothing)
    Random.seed!(seed)
    backward = battery_policy_graph(case, BACKWARD_FORMULATION;
                                    num_stages = num_stages, optimizer = backward_optimizer)
    forward = battery_policy_graph(case, FORWARD_FORMULATION;
                                   num_stages = num_stages, optimizer = forward_optimizer)
    assert_graph_provenance(backward, BACKWARD_FORMULATION)
    assert_graph_provenance(forward, FORWARD_FORMULATION)

    ids = [b.index for b in case.batteries]
    history = NamedTuple[]
    stock = SDDP.AlternativePostIterationCallback(forward)
    iteration = Ref(0)
    t0 = time()

    # The stock callback FIRST — the periodic evaluation is composed around it,
    # so the cuts the ACP graph carries are exactly the ones stock SDDP made.
    callback = function (result)
        stock(result)
        iteration[] += 1
        (protocol === nothing || evaluate_every === nothing) && return
        iteration[] % Int(evaluate_every) == 0 || return
        sims = simulate_battery_sddp_on((forward = forward, num_stages = Int(num_stages)),
                                        protocol; ids = ids, columns = evaluate_columns)
        costs = [sddp_path_cost(case, s, num_stages) for s in sims]
        rec = maximum(max(sims[i][t][:deficit], sims[i][t][:surplus])
                      for i in eachindex(sims), t in 1:Int(num_stages))
        push!(history, (iteration = iteration[],
                        bound = SDDP.calculate_bound(backward),
                        acp_mean = mean(costs), acp_min = minimum(costs),
                        acp_max = maximum(costs), worst_recourse = rec,
                        elapsed = time() - t0))
        return
    end

    kwargs = Dict{Symbol,Any}(:print_level => print_level,
                              :forward_pass => SDDP.AlternativeForwardPass(forward),
                              :post_iteration_callback => callback)
    iteration_limit === nothing || (kwargs[:iteration_limit] = Int(iteration_limit))
    time_limit === nothing || (kwargs[:time_limit] = Float64(time_limit))
    SDDP.train(backward; kwargs...)
    elapsed = time() - t0

    cut_path === nothing || SDDP.write_cuts_to_file(backward, cut_path)

    return (backward = backward, forward = forward,
            status = SDDP.termination_status(backward),
            # In the objective units of the PGLib case, like every stage cost the
            # backward graph was built from. Every consumer of `trained.bound` —
            # `cost_report`, the smoke and any downstream analysis — reads it as
            # it stands, directly comparable with `sddp_path_cost`.
            bound = SDDP.calculate_bound(backward),
            case = case,
            elapsed = elapsed, num_stages = Int(num_stages),
            iterations = iteration[], history = history,
            cut_path = cut_path)
end

"""
    simulate_battery_sddp(trained, num_replications; ids, seed) -> Vector

Simulate the trained policy on the TRUE-ACP graph.

# Arguments
- `ids`: battery identifiers, in the order the recorded energy vectors use.

# Notes
Simulation records, per stage: the stage objective, the outgoing battery energy,
the charge/discharge split and the two recourse totals. The recourse totals are
recorded on every path because a policy that leans on them is rejected, and a
rejection rule that is only evaluated at the end of a campaign is not a rule.

Everything is recorded through `custom_recorders`, which read the live
subproblem after it is solved, so what is stored is the value the solver
actually returned rather than a re-derivation of it.
"""
function simulate_battery_sddp(trained, num_replications::Integer;
                               ids::AbstractVector{Int},
                               seed::Integer = 20260804)
    Random.seed!(seed)
    return SDDP.simulate(trained.forward, num_replications;
                         custom_recorders = Dict{Symbol,Function}(
                             :energy_out => (sp::JuMP.Model) -> [JuMP.value(sp[:e][k].out) for k in ids],
                             :energy_in => (sp::JuMP.Model) -> [JuMP.value(sp[:e][k].in) for k in ids],
                             :deficit => (sp::JuMP.Model) -> sum(JuMP.value.(sp.ext[:battery][:d])),
                             :surplus => (sp::JuMP.Model) -> sum(JuMP.value.(sp.ext[:battery][:s])),
                             :p_ch => (sp::JuMP.Model) -> sum(JuMP.value(sp.ext[:battery][:p_ch][k]) for k in ids; init = 0.0),
                             :p_dis => (sp::JuMP.Model) -> sum(JuMP.value(sp.ext[:battery][:p_dis][k]) for k in ids; init = 0.0),
                         ))
end

"""
    simulate_battery_sddp_on(trained, protocol; ids, columns=nothing) -> Vector

Simulate the trained policy on the true-ACP graph over EXACTLY the demand paths
of a protocol, in the protocol's own order.

# Arguments
- `protocol::AbstractMatrix{<:Integer}`: the `(stages × scenarios)` atom-index
  matrix.

# Keywords
- `columns`: the global scenario identifiers to simulate; every column by
  default.

# Notes
This is what makes an SDDP cost PAIRED with a perfect-foresight cost or with a
TS-DDR cost: all three are then accumulated over the same demand realizations,
and the difference between two policies is measured scenario by scenario rather
than between two independent samples of a distribution whose spread dwarfs it.

`SDDP.Historical` is SDDP.jl's own mechanism for replaying a fixed set of
realizations; nothing about the policy or the graph changes, only which noises
the forward pass sees.
"""
function simulate_battery_sddp_on(trained, protocol::AbstractMatrix{<:Integer};
                                  ids::AbstractVector{Int},
                                  columns = nothing)
    T = trained.num_stages
    T <= size(protocol, 1) ||
        throw(ArgumentError("protocol has $(size(protocol, 1)) stages but the policy has $T"))
    cols = columns === nothing ? collect(1:size(protocol, 2)) : collect(Int.(columns))
    scenarios = [[(t, Int(protocol[t, c])) for t in 1:T] for c in cols]
    return SDDP.simulate(trained.forward, length(scenarios);
                         sampling_scheme = SDDP.Historical(scenarios),
                         custom_recorders = Dict{Symbol,Function}(
                             :energy_out => (sp::JuMP.Model) -> [JuMP.value(sp[:e][k].out) for k in ids],
                             :energy_in => (sp::JuMP.Model) -> [JuMP.value(sp[:e][k].in) for k in ids],
                             :deficit => (sp::JuMP.Model) -> max(0.0, sum(JuMP.value.(sp.ext[:battery][:d]))),
                             :surplus => (sp::JuMP.Model) -> max(0.0, sum(JuMP.value.(sp.ext[:battery][:s]))),
                             :p_ch => (sp::JuMP.Model) -> sum(JuMP.value(sp.ext[:battery][:p_ch][k]) for k in ids; init = 0.0),
                             :p_dis => (sp::JuMP.Model) -> sum(JuMP.value(sp.ext[:battery][:p_dis][k]) for k in ids; init = 0.0),
                         ))
end

"""
    sddp_smoke(; case_dir, num_stages, iteration_limit, replications) -> NamedTuple

The Phase-1 SDDP construction smoke.

# Notes
This is a CONSTRUCTION test, not a performance experiment. It runs the smallest
horizon and iteration count that still exercises every mechanism the study
depends on — actual SOC-WR backward nodes, actual ACP forward nodes, battery
energy as a genuine state that propagates, demand realized from the frozen
atoms, cuts created by stock SDDP, and clean ACP forward solves — and it reports
nothing about policy quality.
"""
function sddp_smoke(; case_dir::AbstractString = joinpath(@__DIR__, "case", "pglib_opf_case14_ieee"),
                      num_stages::Integer = parse(Int, get(ENV, "DR_BAT_SDDP_STAGES", "3")),
                      iteration_limit::Integer = parse(Int, get(ENV, "DR_BAT_SDDP_ITERATIONS", "10")),
                      replications::Integer = parse(Int, get(ENV, "DR_BAT_SDDP_SIMS", "6")))
    case = read_battery_case(case_dir)
    ids = [b.index for b in case.batteries]
    trained = train_battery_sddp(case; num_stages = num_stages, iteration_limit = iteration_limit)
    sims = simulate_battery_sddp(trained, replications; ids = ids)
    @printf("SDDP smoke: %d stages, %d iterations, status %s\n",
            num_stages, iteration_limit, trained.status)
    @printf("  backward formulation %s   forward formulation %s\n",
            BACKWARD_FORMULATION, FORWARD_FORMULATION)
    # `trained.bound` and `sddp_path_cost` are both in the objective units of the
    # PGLib case, so the bound below and the forward costs printed after it are
    # directly comparable.
    @printf("  SOC-WR bound over %d stages: %.6f   (elapsed %.1f s)\n",
            trained.num_stages, trained.bound, trained.elapsed)
    ncuts = sum(length(node.bellman_function.global_theta.cuts)
                for (_, node) in trained.backward.nodes)
    @printf("  stock cuts created: %d\n", ncuts)
    costs = [sddp_path_cost(case, s, num_stages) for s in sims]
    @printf("  ACP forward cost over %d paths: mean %.6f  min %.6f  max %.6f\n",
            length(costs), mean(costs), minimum(costs), maximum(costs))
    worst_d = maximum(sims[i][t][:deficit] for i in eachindex(sims), t in 1:num_stages)
    worst_s = maximum(sims[i][t][:surplus] for i in eachindex(sims), t in 1:num_stages)
    @printf("  worst recourse on any simulated stage: deficit %.3e  surplus %.3e\n", worst_d, worst_s)
    for (j, k) in enumerate(ids)
        # `energy_out` is recorded in `ids` order, so column j is battery k.
        traj = [sims[1][t][:energy_out][j] for t in 1:num_stages]
        @printf("  battery %d energy path (path 1): %.5f -> %s\n", k,
                sims[1][1][:energy_in][j],
                join((@sprintf("%.5f", x) for x in traj), " -> "))
    end
    return (trained = trained, sims = sims, cuts = ncuts,
            worst_deficit = worst_d, worst_surplus = worst_s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    sddp_smoke()
end


# ─────────────────────────────────────────────────────────────────────────────
# The standard cost report
# ─────────────────────────────────────────────────────────────────────────────

"""
    cost_report(case, trained, protocol; columns, alpha=0.95, io=stdout) -> NamedTuple

The four quantities this study reports about a case, always together and always
with the same meaning.

# Returns / prints

| | |
|---|---|
| `sddp_mean`, `sddp_cvar`, `sddp_max` | the SDDP policy's TRUE-ACP cost over `columns`: mean, the `alpha`-CVaR (mean of the worst `1-alpha` tail) and the worst path |
| `bound` | the SOC-WR relaxation's bound over the SAME horizon |
| `pf_mean`, `pf_cvar`, `pf_max` | the true-ACP perfect-foresight cost over the SAME paths |
| `gap_bound` | `(sddp_mean − bound) / sddp_mean` |
| `gap_pf` | `(sddp_mean − pf_mean) / pf_mean` |

# Notes
The two gaps answer different questions and are never added or conflated:

- `gap_bound` is the SDDP policy's true cost against a LOWER BOUND on the
  RELAXED problem. It is dominated by how loose the SOC-WR relaxation is on this
  network, and it is not a statement about the policy.
- `gap_pf` is the policy against CLAIRVOYANCE on the same demand paths. It is
  the policy headroom, and it contains the value of perfect information, which no
  nonanticipative method can recover.

Every quantity in the table is in the objective units of the PGLib case, the one
unit system this study has. The policy costs come through
[`sddp_path_cost`](@ref), the bound off the backward graph built from those same
stage costs, and the perfect-foresight and forecast costs are sums of
`cost_stage`. No quantity here is rescaled at any point, so the rows are
comparable as they stand.

Both are reported on the SAME paths and the SAME horizon as the bound, because a
bound never bounds a quantity accumulated over a different horizon and a paired
comparison is only paired if both sides saw the same scenarios. The tail metric
is reported beside every mean because a storage policy that is good on average
and bad in the tail is a different object from one that is good in both.

# The supporting metrics, and why they exist
A large `gap_bound` is easy to mistake for room a better policy could take. These
two say how much of each gap is actually available:

- **`e` — the deterministic-forecast policy** and `VSS = (forecast − a)/a`. If
  ignoring the uncertainty entirely costs nothing, the case has no stochastic
  content and no two methods can be told apart on it, whatever the other gaps.
- **`f` — the recoverable ceiling**, `(a − max(bound, pf_mean))/a`. The best
  nonanticipative cost `RP` is unknown, but it is at least the relaxation bound
  and at least the wait-and-see mean, so this is a hard cap on what TS-DDR could
  ever take from `a`. It is an upper bound on the prize, not the prize.

**The policy is never run in the SOC model.** This SDDP is deliberately
INCONSISTENT — cuts built in the relaxation, states visited in the true model —
so operating the same cuts inside the relaxation gives the cost of a different
trajectory of a different problem, not a decomposition of anything about the
policy we have.
"""
function cost_report(case::BatteryCase, trained, protocol::AbstractMatrix{<:Integer};
                     columns, alpha::Real = 0.95, forecast::Bool = true,
                     io::IO = stdout)
    T = trained.num_stages
    ids = [b.index for b in case.batteries]
    cols = collect(Int.(columns))

    sims = simulate_battery_sddp_on(trained, protocol; ids = ids, columns = cols)
    sddp = [sddp_path_cost(case, s, T) for s in sims]
    rec = maximum(max(sims[i][t][:deficit], sims[i][t][:surplus])
                  for i in eachindex(sims), t in 1:T)

    panel = perfect_foresight_panel(case, protocol; ids = cols)
    panel.complete ||
        error("perfect-foresight panel incomplete; a gap against a partial panel is a gap against a different problem")
    pf = [r.cost for r in panel.rows]

    fc = forecast ? forecast_policy_cost(case, protocol; columns = cols) : nothing

    tail(v) = (n = max(1, ceil(Int, (1 - alpha) * length(v))); mean(sort(v; rev = true)[1:n]))
    a, bnd, cc = mean(sddp), trained.bound, mean(pf)
    # RP, the best nonanticipative cost, is unknown but is bounded below by BOTH
    # the relaxation bound and the wait-and-see mean. Whichever is tighter caps
    # what ANY better policy — TS-DDR included — could ever recover from `a`.
    rp_lower = max(bnd, cc)

    r = (n = length(cols), horizon = T, alpha = alpha,
         sddp_mean = a, sddp_cvar = tail(sddp), sddp_max = maximum(sddp),
         sddp_sem = std(sddp) / sqrt(length(sddp)), sddp_recourse = rec,
         bound = bnd,
         pf_mean = cc, pf_cvar = tail(pf), pf_max = maximum(pf),
         pf_sem = std(pf) / sqrt(length(pf)),
         forecast_mean = fc === nothing ? NaN : fc.mean,
         forecast_max = fc === nothing ? NaN : fc.max,
         gap_bound = (a - bnd) / a,
         gap_pf = (a - cc) / cc,
         vss = fc === nothing ? NaN : (fc.mean - a) / a,
         recoverable_ceiling = (a - rp_lower) / a)

    @printf(io, "\n%d stages, %d paths, CVaR at %.0f%%\n", T, r.n, 100alpha)
    @printf(io, "  a) SDDP ac-soc cost   mean %14.2f   CVaR %14.2f   max %14.2f   (sem %.2f)\n",
            r.sddp_mean, r.sddp_cvar, r.sddp_max, r.sddp_sem)
    @printf(io, "  b) SDDP ac-soc bound  %14.2f\n", r.bound)
    @printf(io, "  c) perfect foresight  mean %14.2f   CVaR %14.2f   max %14.2f   (sem %.2f)\n",
            r.pf_mean, r.pf_cvar, r.pf_max, r.pf_sem)
    @printf(io, "  d) gap a-b %8.4f%%     gap a-c %8.4f%%     worst recourse %.2e\n",
            100r.gap_bound, 100r.gap_pf, r.sddp_recourse)
    @printf(io, "\n  supporting metrics\n")
    if fc !== nothing
        @printf(io, "  e) deterministic-forecast policy     %14.2f   max %14.2f\n",
                r.forecast_mean, r.forecast_max)
        @printf(io, "       VSS = (forecast - a)/a = %7.4f%%  — how much the uncertainty is worth\n",
                100r.vss)
    end
    @printf(io, "  f) recoverable ceiling %7.4f%%  — (a - max(b,c))/a, the most ANY better\n",
            100r.recoverable_ceiling)
    @printf(io, "       nonanticipative policy, TS-DDR included, could win from a\n")
    return r
end
