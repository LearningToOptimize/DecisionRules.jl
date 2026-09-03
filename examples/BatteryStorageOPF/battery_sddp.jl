# battery_sddp.jl
#
# The SDDP baseline: stock SDDP.jl over the shared PowerModels battery problem
# specification, with a SELECTABLE convex backward pass and a true-ACP forward
# pass.
#
# WHAT IS STOCK AND WHY IT MATTERS.
# SDDP.jl owns the policy graph, the state variables, the sampling scheme, cut
# generation, training and simulation. This file constructs two policy graphs
# from the SAME `build_battery_opf` specification — one instantiated as an
# actual convex PowerModels formulation, one as an actual
# `PowerModels.ACPPowerModel` — and hands them to SDDP's own
# `AlternativeForwardPass`/`AlternativePostIterationCallback` pair, which is
# SDDP.jl's documented mechanism for training against a convex approximation
# while operating a nonconvex model. No `duality_handler` is overridden, no cut
# is filtered, retried or reweighted: which duals become cuts is SDDP's decision.
#
# TWO BACKWARD FORMULATIONS, AND WHAT THEIR SCALARS MEAN.
# `:soc` builds the backward nodes as `PowerModels.SOCWRConicPowerModel` and
# `:dc` as `PowerModels.DCPPowerModel`. Both are PowerModels' own formulations:
# nothing in this study writes a DC network equation, exactly as nothing in it
# writes an AC one. The generator data and the generator cost polynomials are
# the PGLib case's own in both, unrepriced and unlinearized, and the battery
# layer added on top is the same one — state, transition, charge/discharge,
# unity-power-factor injection, demand parameterization, uncapped physical
# recourse — in both.
#
# The SCALARS they produce are NOT the same kind of object, and this file never
# lets them be printed as if they were:
#
#   * the SOC-WR arm's scalar is a lower bound on the SOC-WR RELAXATION over the
#     horizon it was trained on. The relaxation lower-bounds the true ACP
#     problem, so the scalar does too;
#   * the DC arm's scalar is the DC-APPROXIMATION TRAINING BOUND. The DC
#     approximation is neither a relaxation nor a restriction of the nonconvex
#     ACP problem — it drops the reactive balance and fixes voltage magnitudes —
#     so its value bounds NOTHING about ACP, in either direction. It is the
#     internal convergence scalar of the approximation the cuts came from, and
#     it is reported to say whether that training converged.
#
# Neither is comparable to a forward objective over a different horizon, and
# nothing in this file invites that comparison.
#
# Usage
#   julia --project=. battery_sddp.jl                  # short construction smoke
#   DR_BAT_SDDP_BACKWARD=dc DR_BAT_SDDP_STAGES=3 julia --project=. battery_sddp.jl

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

"The true model SDDP's forward pass operates. Not selectable: the study compares methods, not physics."
const FORWARD_FORMULATION = PowerModels.ACPPowerModel

"""
    BACKWARD_SPECS

The selectable backward formulations, keyed by the identifier that names them
everywhere — in a keyword, in a result field, in a printed line and in a cut
file's name.

# Fields of each row
- `formulation::Type`: the PowerModels model type the backward nodes really are.
- `optimizer`: the solver factory that matches it.
- `bounds_acp::Bool`: whether the scalar SDDP converges to is a valid lower
  bound on the true ACP problem. TRUE for the SOC-WR relaxation, FALSE for the
  DC approximation.
- `bound_name::String`: what that scalar may be CALLED. Every place this file
  prints or labels it reads this field, so the DC arm cannot acquire the word
  "bound" on its own anywhere.
- `tag::String`: the token a serialized artifact must carry.
"""
const BACKWARD_SPECS = Dict{Symbol,NamedTuple}(
    :soc => (formulation = PowerModels.SOCWRConicPowerModel,
             optimizer = socwr_optimizer,
             bounds_acp = true,
             bound_name = "SOC-WR relaxation bound",
             tag = "socwr"),
    :dc => (formulation = PowerModels.DCPPowerModel,
            optimizer = dc_optimizer,
            bounds_acp = false,
            bound_name = "DC-approximation training bound",
            tag = "dc"),
)

"""
    backward_spec(backward::Symbol) -> NamedTuple

The [`BACKWARD_SPECS`](@ref) row for `backward`, or an error naming the ones
that exist.
"""
function backward_spec(backward::Symbol)
    haskey(BACKWARD_SPECS, backward) || throw(ArgumentError(
        "backward formulation must be one of $(sort!(collect(keys(BACKWARD_SPECS)))), got :$backward"))
    return BACKWARD_SPECS[backward]
end

"""
    sddp_cut_path(dir, case, backward) -> String

The path a cut file for this case and this backward formulation is written to.

# Notes
The formulation's tag is in the FILE NAME, not only in a sibling record. A cut
set is the one artifact of a run that outlives the session that made it, and a
directory holding `case118.cuts.json` twice — once from SOC and once from DC —
is a directory in which the two can be confused by nothing worse than a
`readdir`. [`train_battery_sddp`](@ref) refuses a `cut_path` whose name does not
carry the tag as a dot-delimited component, so the convention is enforced rather
than merely offered.
"""
sddp_cut_path(dir::AbstractString, case::BatteryCase, backward::Symbol) =
    joinpath(dir, "$(case.name).$(backward_spec(backward).tag).cuts.json")

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
    battery_recorders(ids) -> Dict{Symbol,Function}

The `custom_recorders` every simulation in this file uses.

# What is recorded, and why each one
| key | value |
|---|---|
| `:energy_out`, `:energy_in` | the battery state, in `ids` order |
| `:deficit`, `:surplus` | the recourse SUMMED over buses, kept for the existing convergence log |
| `:deficit_by_bus`, `:surplus_by_bus` | the recourse PER BUS, raw and unprojected |
| `:bus_ids` | the bus ordering the two per-bus vectors use |
| `:cost_generation`, `:cost_throughput` | the stage cost decomposition |
| `:p_ch`, `:p_dis` | throughput |

# Notes
**The per-bus vectors are what make an element-wise cost contract possible.**
The shared `physical_stage_cost` projects and validates recourse element by
element; a recorder that returned only the totals would leave a simulation
record from which the contract cannot be evaluated at all, and the aggregate is
not a conservative stand-in for it — positive and negative tolerance-scale
residues at different buses CANCEL in a sum, so a total of exactly zero is
consistent with every bus being off, and one bus off by a lot can hide behind
other buses' surpluses. `sddp_physical_path_cost` is the consumer.

Everything is read off the live subproblem after it is solved, through
`stage_cost_decomposition` — the same decomposition `extract_stage_solution`
uses — so what is stored is the value the solver actually returned, decomposed
once, in one place.

The per-bus values are stored RAW. Projection belongs to the cost contract and
happens there; a recorder that pre-projected would destroy the evidence the
contract has to judge.
"""
function battery_recorders(ids::AbstractVector{Int})
    dec(sp::JuMP.Model) = stage_cost_decomposition(sp.ext[:pm])
    order(sp::JuMP.Model) = sp.ext[:pm].ext[:battery][:buses]
    return Dict{Symbol,Function}(
        :energy_out => (sp::JuMP.Model) -> [JuMP.value(sp[:e][k].out) for k in ids],
        :energy_in => (sp::JuMP.Model) -> [JuMP.value(sp[:e][k].in) for k in ids],
        :deficit => (sp::JuMP.Model) -> max(0.0, sum(JuMP.value.(sp.ext[:battery][:d]))),
        :surplus => (sp::JuMP.Model) -> max(0.0, sum(JuMP.value.(sp.ext[:battery][:s]))),
        :bus_ids => (sp::JuMP.Model) -> collect(Int, order(sp)),
        :deficit_by_bus => (sp::JuMP.Model) -> (D = dec(sp).deficit; [D[i] for i in order(sp)]),
        :surplus_by_bus => (sp::JuMP.Model) -> (S = dec(sp).surplus; [S[i] for i in order(sp)]),
        :cost_generation => (sp::JuMP.Model) -> dec(sp).cost_generation,
        :cost_throughput => (sp::JuMP.Model) -> dec(sp).cost_throughput,
        :p_ch => (sp::JuMP.Model) -> sum(JuMP.value(sp.ext[:battery][:p_ch][k]) for k in ids; init = 0.0),
        :p_dis => (sp::JuMP.Model) -> sum(JuMP.value(sp.ext[:battery][:p_dis][k]) for k in ids; init = 0.0),
    )
end

"""
    sddp_physical_path_cost(case, path, T; tol=PHYSICAL_RECOURSE_TOL)
        -> (cost, worst_recourse, admissible)

The CORRECTED physical cost of one simulated path, under the shared cost
contract.

``C = \\sum_{t=1}^{T} \\mathrm{physical\\_stage\\_cost}(\\mathrm{stage}_t)_{\\mathrm{corrected}}``

# Returns
- `cost`: the sum of the per-stage CORRECTED costs — generation, throughput and
  the recourse actually charged after element-wise projection.
- `worst_recourse`: the largest absolute RAW per-bus recourse over every bus of
  every stage, before projection.
- `admissible`: whether every individual element of every stage was inside
  `tol`.

# Notes
Each stage goes through `physical_stage_cost`, the byte-identical contract both
engines carry, fed the PER-BUS recourse the recorders stored. Only after every
element has been validated and projected is anything summed — validate, project,
then sum, in that order. Summing first and testing the total would accept a
stage whose bus-level violations happen to cancel, which is the failure this
ordering exists to exclude.

`sddp_path_cost` remains the RAW sum of the solver's stage objectives and is
still what the training log and the convergence history report. This is the
quantity a policy is SELECTED and REPORTED on, exactly as on the Exa side.
"""
function sddp_physical_path_cost(case::BatteryCase, path, T::Integer;
                                 tol::Real = PHYSICAL_RECOURSE_TOL)
    total = 0.0
    worst = 0.0
    admissible = true
    for t in 1:Int(T)
        st = path[t]
        bus = Int.(st[:bus_ids])
        c = physical_stage_cost(
            (cost_generation = Float64(st[:cost_generation]),
             cost_throughput = Float64(st[:cost_throughput]),
             deficit = Dict(bus[i] => Float64(st[:deficit_by_bus][i]) for i in eachindex(bus)),
             surplus = Dict(bus[i] => Float64(st[:surplus_by_bus][i]) for i in eachindex(bus)),
             objective = Float64(st[:stage_objective])), case.recourse; tol = tol)
        total += c.corrected
        worst = max(worst, c.worst_recourse)
        admissible &= c.admissible
    end
    return (cost = total, worst_recourse = worst, admissible = admissible)
end

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
    train_battery_sddp(case; backward=:soc, num_stages, iteration_limit,
                       time_limit, seed, print_level, backward_optimizer,
                       forward_optimizer, protocol, evaluate_columns,
                       evaluate_every, cut_path, resume_cuts)
        -> NamedTuple

Train the SDDP policy: convex backward, true-ACP forward.

# Keywords
- `backward::Symbol`: `:soc` (default) or `:dc`; see [`BACKWARD_SPECS`](@ref).
  It selects the PowerModels formulation the backward nodes are instantiated as,
  and nothing else about the stochastic program.
- `num_stages::Integer`: horizon.
- `iteration_limit`, `time_limit`: the training budget. Either may be `nothing`.
- `seed::Integer`: seed of the forward sampling scheme.
- `backward_optimizer`: solver factory, or `nothing` for the backward
  formulation's own. `forward_optimizer`: solver factory for the ACP graph.
  These are SOLVER settings; nothing here may change the mathematical model.
- `protocol`, `evaluate_columns`, `evaluate_every`: when all three are given,
  the TRUE-ACP forward cost is evaluated on those fixed protocol columns every
  `evaluate_every` iterations and recorded in `history`.
- `cut_path`: when given, the cuts are written there after training. It must
  carry the backward formulation's tag — see [`sddp_cut_path`](@ref).
- `resume_cuts`: when given, the cut file a previous training stage wrote, read
  into the freshly built graphs BEFORE training so this call CONTINUES that
  policy instead of starting a new one. It must carry the same tag, so a SOC cut
  set can never be read into a DC graph.

# Returns
A `NamedTuple` with the two graphs, the termination status, the backward
formulation's scalar, the elapsed wall time, and `history` — one row per
evaluation with the iteration, the scalar at that point, the true-ACP mean cost
on the fixed columns, the worst recourse and the elapsed time.

Four fields identify the arm and travel with every downstream consumer:
`backward` (the identifier), `backward_formulation` (the actual PowerModels
type), `bound_name` (what `bound` may be called) and `bound_bounds_acp` (whether
`bound` is a valid lower bound on the true ACP problem — TRUE only for `:soc`).

# Notes
`SDDP.AlternativeForwardPass(forward)` performs each forward pass on the ACP
graph and copies the resulting states back into the convex graph, and
`AlternativePostIterationCallback(forward)` copies the newly created cuts into
the ACP graph. Both are SDDP.jl's own; the pair is what makes "value with a
convex approximation, operate the true model" a stock configuration rather than
an intervention, and it is SHARED by the two arms rather than reimplemented for
the second one. The periodic evaluation is composed AROUND that callback, never
in place of it.

**Three signals, never compared in level.** The backward scalar is a property of
the formulation the cuts came from over `num_stages` stages — a bound on the
relaxation for `:soc`, an internal convergence scalar for `:dc`. The true-ACP
forward cost on the fixed columns is a physical cost of the current policy on a
small, fixed sample. A final protocol cost is a third thing again.

The evaluation is a MEASUREMENT, not a stopping rule: training runs its declared
budget. Choosing when to stop by watching the evaluation would select a policy on
the same numbers used to report it.

CONTINUATION. `resume_cuts` exists so a long run can survive preemption: the
graphs are REBUILT from the frozen case and SDDP's own `read_cuts_from_file`
restores the policy into them. No solver object is serialized and no SDDP
internal is parsed. The file is read into BOTH graphs, because
`AlternativePostIterationCallback` is what normally puts each iteration's cuts
into the ACP graph, and a forward graph resumed without them would take its
decisions against an empty cost-to-go while the backward graph believed
otherwise. The two graphs carry the same state variables, so the same file is the
right file for both.
"""
function train_battery_sddp(case::BatteryCase;
                            backward::Symbol = :soc,
                            num_stages::Integer = 3,
                            iteration_limit = 10,
                            time_limit = nothing,
                            seed::Integer = 20260804,
                            print_level::Integer = 0,
                            backward_optimizer = nothing,
                            forward_optimizer = acp_optimizer(),
                            protocol = nothing,
                            evaluate_columns = nothing,
                            evaluate_every = nothing,
                            cut_path = nothing,
                            resume_cuts = nothing)
    spec = backward_spec(backward)
    # The tag must be a dot-delimited COMPONENT of the file name, not merely a
    # substring of it: a PGLib case name that happened to contain "dc" would
    # otherwise let a DC run write over a path built for the SOC arm.
    for (label, p) in (("cut_path", cut_path), ("resume_cuts", resume_cuts))
        p === nothing || spec.tag in split(basename(String(p)), '.') ||
            throw(ArgumentError(
                "$label file $(basename(String(p))) does not carry the \"$(spec.tag)\" tag of the " *
                ":$backward backward formulation; use sddp_cut_path to build one"))
    end
    backward_optimizer === nothing && (backward_optimizer = spec.optimizer())
    Random.seed!(seed)
    backward_graph = battery_policy_graph(case, spec.formulation;
                                          num_stages = num_stages, optimizer = backward_optimizer)
    forward = battery_policy_graph(case, FORWARD_FORMULATION;
                                   num_stages = num_stages, optimizer = forward_optimizer)
    assert_graph_provenance(backward_graph, spec.formulation)
    assert_graph_provenance(forward, FORWARD_FORMULATION)

    # CONTINUATION, before the first iteration: SDDP's own reader restores the
    # cuts into both graphs. `add_to_existing_cuts` is not needed below — these
    # graphs were built a moment ago and carry no training results of their own,
    # so `SDDP.train` sees a model it has never trained.
    if resume_cuts !== nothing
        isfile(String(resume_cuts)) || error("no cut file to resume from: $resume_cuts")
        SDDP.read_cuts_from_file(backward_graph, String(resume_cuts))
        SDDP.read_cuts_from_file(forward, String(resume_cuts))
    end

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
        # The CORRECTED physical cost, element-wise: each stage through the
        # shared contract, validated and projected per bus before anything is
        # summed. `acp_raw_mean` keeps the solver's raw objective sum beside it
        # as a diagnostic, never as the headline.
        phys = [sddp_physical_path_cost(case, s, num_stages) for s in sims]
        costs = [p.cost for p in phys]
        raw = [sddp_path_cost(case, s, num_stages) for s in sims]
        rec = maximum(p.worst_recourse for p in phys)
        push!(history, (iteration = iteration[], backward = backward,
                        bound = SDDP.calculate_bound(backward_graph),
                        acp_mean = mean(costs), acp_min = minimum(costs),
                        acp_max = maximum(costs), worst_recourse = rec,
                        admissible = all(p.admissible for p in phys),
                        acp_raw_mean = mean(raw),
                        elapsed = time() - t0))
        return
    end

    kwargs = Dict{Symbol,Any}(:print_level => print_level,
                              :forward_pass => SDDP.AlternativeForwardPass(forward),
                              :post_iteration_callback => callback)
    iteration_limit === nothing || (kwargs[:iteration_limit] = Int(iteration_limit))
    time_limit === nothing || (kwargs[:time_limit] = Float64(time_limit))
    SDDP.train(backward_graph; kwargs...)
    elapsed = time() - t0

    cut_path === nothing || SDDP.write_cuts_to_file(backward_graph, cut_path)

    return (backward_graph = backward_graph, forward = forward,
            status = SDDP.termination_status(backward_graph),
            # Which arm this is, in four fields, so no consumer has to infer it
            # and none can print the DC scalar under the SOC one's name.
            backward = backward,
            backward_formulation = spec.formulation,
            bound_name = spec.bound_name,
            bound_bounds_acp = spec.bounds_acp,
            # In the objective units of the PGLib case, like every stage cost the
            # backward graph was built from. Every consumer of `trained.bound` —
            # `cost_report`, the smoke and any downstream analysis — reads it as
            # it stands, directly comparable with `sddp_path_cost`.
            bound = SDDP.calculate_bound(backward_graph),
            case = case,
            elapsed = elapsed, num_stages = Int(num_stages),
            iterations = iteration[], history = history,
            cut_path = cut_path)
end

"""
    sddp_method_id(trained) -> Symbol

The study's method identifier of a trained run: `:sddp_soc` or `:sddp_dc`.

# Notes
Derived from the run itself rather than from what a caller remembers asking for,
so a result table, a log line and a serialized artifact all name the same arm.
"""
sddp_method_id(trained) = Symbol("sddp_", trained.backward)

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
                         custom_recorders = battery_recorders(collect(Int, ids)))
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
                         custom_recorders = battery_recorders(collect(Int, ids)))
end

"""
    sddp_smoke(; case_dir, backward, num_stages, iteration_limit, replications)
        -> NamedTuple

The SDDP construction smoke, for either backward formulation.

# Notes
This is a CONSTRUCTION test, not a performance experiment. It runs the smallest
horizon and iteration count that still exercises every mechanism the study
depends on — actual backward nodes of the requested formulation, actual ACP
forward nodes, battery energy as a genuine state that propagates, demand
realized from the frozen atoms, cuts created by stock SDDP, and clean ACP
forward solves — and it reports nothing about policy quality.

Every line it prints names the arm, and the backward scalar is printed under the
name [`BACKWARD_SPECS`](@ref) gives it, so a DC log can never be read as a
statement about a bound on the ACP problem.
"""
function sddp_smoke(; case_dir::AbstractString = joinpath(@__DIR__, "case", "pglib_opf_case14_ieee"),
                      backward::Symbol = Symbol(get(ENV, "DR_BAT_SDDP_BACKWARD", "soc")),
                      num_stages::Integer = parse(Int, get(ENV, "DR_BAT_SDDP_STAGES", "3")),
                      iteration_limit::Integer = parse(Int, get(ENV, "DR_BAT_SDDP_ITERATIONS", "10")),
                      replications::Integer = parse(Int, get(ENV, "DR_BAT_SDDP_SIMS", "6")))
    case = read_battery_case(case_dir)
    ids = [b.index for b in case.batteries]
    trained = train_battery_sddp(case; backward = backward, num_stages = num_stages,
                                 iteration_limit = iteration_limit)
    sims = simulate_battery_sddp(trained, replications; ids = ids)
    @printf("SDDP smoke [%s]: %d stages, %d iterations, status %s\n",
            sddp_method_id(trained), num_stages, iteration_limit, trained.status)
    @printf("  backward formulation %s   forward formulation %s\n",
            trained.backward_formulation, FORWARD_FORMULATION)
    # `trained.bound` and `sddp_path_cost` are both in the objective units of the
    # PGLib case. Whether the scalar BOUNDS the forward costs printed after it is
    # a property of the formulation, and is stated rather than implied.
    @printf("  %s over %d stages: %.6f   (elapsed %.1f s)\n",
            trained.bound_name, trained.num_stages, trained.bound, trained.elapsed)
    @printf("    (%s a lower bound on the true ACP problem)\n",
            trained.bound_bounds_acp ? "is" : "is NOT")
    ncuts = sum(length(node.bellman_function.global_theta.cuts)
                for (_, node) in trained.backward_graph.nodes)
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
| `bound` | the backward formulation's own scalar over the SAME horizon, under the name `trained.bound_name` gives it |
| `pf_mean`, `pf_cvar`, `pf_max` | the true-ACP perfect-foresight cost over the SAME paths |
| `gap_bound` | `(sddp_mean − bound) / sddp_mean` |
| `gap_pf` | `(sddp_mean − pf_mean) / pf_mean` |

# Notes
The two gaps answer different questions and are never added or conflated:

- `gap_bound` is the SDDP policy's true cost against the backward formulation's
  scalar. For the SOC-WR arm that scalar is a LOWER BOUND on the relaxed problem
  and the difference is dominated by how loose the relaxation is on this network;
  it is not a statement about the policy. For the DC arm it is not a bound at
  all, and the difference is the distance between a true-ACP cost and the
  internal scalar of a different approximation — a diagnostic of the DC training,
  never a headroom.
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

  **On the DC arm the bound term is dropped from that maximum**, and this is the
  one place where the difference between the two scalars changes arithmetic
  rather than only a label. The DC-approximation training bound does not bound
  `RP`, so admitting it into `max(·)` could tighten the ceiling with a number
  that has no right to tighten it, and could report a prize smaller than the one
  that exists. The ceiling then rests on the wait-and-see mean alone, which is a
  valid lower bound on `RP` under either arm.

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
    # Row a is the CORRECTED physical cost under the shared contract, so it is
    # the same estimand the Exa arm reports and the same one a checkpoint was
    # selected on. The raw objective sum is kept beside it as `sddp_raw_mean`,
    # a solver diagnostic; comparing two engines on a raw objective compares
    # their barrier parameters.
    phys = [sddp_physical_path_cost(case, s, T) for s in sims]
    sddp = [p.cost for p in phys]
    sddp_raw = [sddp_path_cost(case, s, T) for s in sims]
    rec = maximum(p.worst_recourse for p in phys)
    all(p.admissible for p in phys) || error(
        "the SDDP forward panel is INADMISSIBLE: worst individual per-bus recourse " *
        "$(rec) pu exceeds $(PHYSICAL_RECOURSE_TOL); a cost report over a panel the " *
        "physical contract rejects would report a policy that does not exist")

    panel = perfect_foresight_panel(case, protocol; ids = cols)
    panel.complete ||
        error("perfect-foresight panel incomplete; a gap against a partial panel is a gap against a different problem")
    pf = [r.cost for r in panel.rows]

    fc = forecast ? forecast_policy_cost(case, protocol; columns = cols) : nothing

    tail(v) = (n = max(1, ceil(Int, (1 - alpha) * length(v))); mean(sort(v; rev = true)[1:n]))
    a, bnd, cc = mean(sddp), trained.bound, mean(pf)
    # RP, the best nonanticipative cost, is unknown but is bounded below by the
    # wait-and-see mean always, and by the backward scalar only when that scalar
    # is a bound on the true ACP problem. Whichever admissible one is tighter
    # caps what ANY better policy — TS-DDR included — could recover from `a`.
    rp_lower = trained.bound_bounds_acp ? max(bnd, cc) : cc

    r = (n = length(cols), horizon = T, alpha = alpha,
         method = sddp_method_id(trained), backward = trained.backward,
         sddp_mean = a, sddp_cvar = tail(sddp), sddp_max = maximum(sddp),
         sddp_sem = std(sddp) / sqrt(length(sddp)), sddp_recourse = rec,
         sddp_raw_mean = mean(sddp_raw),
         bound = bnd, bound_name = trained.bound_name,
         bound_bounds_acp = trained.bound_bounds_acp,
         pf_mean = cc, pf_cvar = tail(pf), pf_max = maximum(pf),
         pf_sem = std(pf) / sqrt(length(pf)),
         forecast_mean = fc === nothing ? NaN : fc.mean,
         forecast_max = fc === nothing ? NaN : fc.max,
         gap_bound = (a - bnd) / a,
         gap_pf = (a - cc) / cc,
         vss = fc === nothing ? NaN : (fc.mean - a) / a,
         recoverable_ceiling = (a - rp_lower) / a)

    @printf(io, "\n%s: %d stages, %d paths, CVaR at %.0f%%\n", r.method, T, r.n, 100alpha)
    @printf(io, "  a) SDDP true-ACP cost mean %14.2f   CVaR %14.2f   max %14.2f   (sem %.2f)\n",
            r.sddp_mean, r.sddp_cvar, r.sddp_max, r.sddp_sem)
    @printf(io, "  b) %-20s %14.2f  (%s a lower bound on the true ACP problem)\n",
            r.bound_name, r.bound, r.bound_bounds_acp ? "is" : "is NOT")
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
    @printf(io, "  f) recoverable ceiling %7.4f%%  — (a - %s)/a, the most ANY better\n",
            100r.recoverable_ceiling, r.bound_bounds_acp ? "max(b,c)" : "c")
    @printf(io, "       nonanticipative policy, TS-DDR included, could win from a\n")
    return r
end

# ─────────────────────────────────────────────────────────────────────────────
# The study's four method identifiers
#
# One table, carried by BOTH public engines with the same four rows and the same
# invariant fields, because the comparison the study makes is between four
# methods and not between two packages. Each engine can RUN the two methods it
# owns and refuses the other two by name, pointing at the engine that owns them —
# neither package loads the other, and a dispatch layer that pretended otherwise
# would fail somewhere less legible than here. The peer copy of this block lives
# in `DecisionRulesExa.jl/examples/BatteryStorageOPF/train_battery_exa_strict.jl`
# and each suite asserts the identifiers and the invariants independently.
# ─────────────────────────────────────────────────────────────────────────────

"""
    BATTERY_METHODS

The four policies this study compares, keyed by their stable identifier.

# The rows

| identifier | family | engine | what varies |
|---|---|---|---|
| `:tsddr_nonlinear` | `:tsddr` | `:exa` | LSTM encoder, nonlinear head |
| `:tsldr_recurrent_linear` | `:tsddr` | `:exa` | affine recurrence, affine head |
| `:sddp_soc` | `:sddp` | `:jump` | `SOCWRConicPowerModel` backward cuts |
| `:sddp_dc` | `:sddp` | `:jump` | `DCPPowerModel` backward cuts |

# The invariants

Every row declares the SAME `horizon`, `stage_semantics`, `recourse`,
`cost_contract` and `comparison` fields, and a test in each engine's suite
asserts it. They are recorded rather than assumed because the four methods are
only comparable if they share them: the frozen case and protocol identities,
`T = 24`, strict reachable targets with no target slack, uncapped physical nodal
recourse with the same admissibility rule, `physical_stage_cost` as the only
headline cost, and a true-ACP evaluation on paired protocol columns.

`protocol` is `"screening"` for every row at this phase. The final 500-column
protocol is not opened by anything in this file.
"""
const BATTERY_METHODS = Dict{Symbol,NamedTuple}(
    :tsddr_nonlinear => (
        family = :tsddr, engine = :exa, architecture = :tsddr_nonlinear,
        backward = nothing,
        summary = "strict TS-DDR with an LSTM encoder and a nonlinear bounded head"),
    :tsldr_recurrent_linear => (
        family = :tsddr, engine = :exa, architecture = :tsldr_recurrent_linear,
        backward = nothing,
        summary = "strict recurrent TSLDR: affine recurrence and affine head, " *
                  "raw target affine in the observed demand history"),
    :sddp_soc => (
        family = :sddp, engine = :jump, architecture = nothing,
        backward = :soc,
        summary = "SDDP with SOCWRConicPowerModel backward cuts and ACP forward decisions"),
    :sddp_dc => (
        family = :sddp, engine = :jump, architecture = nothing,
        backward = :dc,
        summary = "SDDP with DCPPowerModel backward cuts and ACP forward decisions"),
)

"""
The properties every one of [`BATTERY_METHODS`](@ref)' four rows shares.
"""
const BATTERY_METHOD_INVARIANTS = (
    horizon = 24,
    protocol = "screening",
    stage_semantics = "strict reachable outgoing-energy target, no target slack",
    recourse = "uncapped two-sided physical nodal active recourse, admissibility at 1e-6 pu",
    cost_contract = "physical_stage_cost from battery_solution_schema.jl",
    comparison = "true-ACP cost on paired frozen protocol columns",
)

"""
    battery_method(id::Symbol) -> NamedTuple

The descriptor of one method identifier, with the shared invariants merged in.

# Notes
An unknown identifier raises and lists the four, rather than returning
`nothing`: a campaign driver that silently skipped a misspelled method would
report three-quarters of a study as a whole one.
"""
function battery_method(id::Symbol)
    haskey(BATTERY_METHODS, id) || throw(ArgumentError(
        "unknown battery method :$id; the study's methods are $(sort!(collect(keys(BATTERY_METHODS))))"))
    return merge(BATTERY_METHODS[id], (id = id,), BATTERY_METHOD_INVARIANTS)
end

"""
    run_battery_method(id::Symbol, case::BatteryCase; kwargs...) -> NamedTuple

Dispatch a method identifier to this engine's implementation.

# Notes
This engine owns the two `:sddp` rows and runs them through the ONE
[`train_battery_sddp`](@ref) entry point, differing only in `backward`. The two
`:exa` rows are the ExaModels engine's: this package has neither Flux nor
ExaModels, so they are refused by name here rather than half-implemented.

This is a dispatch layer, not a campaign runner. It selects an implementation
and forwards keyword arguments; it schedules nothing, resumes nothing and writes
no ledger.
"""
function run_battery_method(id::Symbol, case::BatteryCase; kwargs...)
    m = battery_method(id)
    m.engine === :jump || error(
        "method :$id runs on the $(m.engine) engine (DecisionRulesExa.jl/examples/BatteryStorageOPF), " *
        "not on this one; this package loads neither Flux nor ExaModels")
    return train_battery_sddp(case; backward = m.backward, kwargs...)
end
