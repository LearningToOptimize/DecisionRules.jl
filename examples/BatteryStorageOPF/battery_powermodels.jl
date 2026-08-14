# battery_powermodels.jl
#
# The battery layer on top of PowerModels.jl, and nothing else.
#
# THE BOUNDARY, stated once and enforced by the tests:
#
#   PowerModels.jl owns  buses, generators, branches, voltage variables, the
#                        reference angle, Ohm's law at both branch ends,
#                        transformer taps and phase shifts, shunts, angle-
#                        difference limits, apparent-power limits at both ends,
#                        the nodal active and reactive balances, generator
#                        bounds and generator cost.
#   this file owns       batteries: their energy state, charge/discharge
#                        controls, state transition, unity-power-factor active
#                        injection, throughput cost, the strict outgoing-energy
#                        target equality, the two-sided nodal active-power
#                        recourse, and the demand parameterization.
#
# There is no handwritten AC trigonometry and no handwritten SOC-WR lifted
# branch equation anywhere below. Selecting `PowerModels.ACPPowerModel` or
# `PowerModels.SOCWRConicPowerModel` selects the network formulation and nothing
# in this file changes.
#
# HOW THE BATTERY REACHES THE NODAL BALANCE.
# PowerModels' `constraint_power_balance` is form-specific and closed: it writes
# the whole balance in one `@constraint`. Its ONLY documented extension point
# for an additional nodal injection is the `storage` component, which enters
# every form's balance as `- sum(ps[s] for s in bus_storage)` on the injection
# side and `- sum(qs[s] ...)` on the reactive side. This file therefore installs
# exactly ONE storage element per bus as an INJECTION CARRIER and defines what
# flows through it:
#
#     ps[i] = Δp^d_i - p^{bat}_i - d_i + s_i          qs[i] = Δq^d_i
#
# so the stock balance PowerModels writes reads, at bus i,
#
#     Σ p_arcs = Σ pg + p^{bat}_i + d_i - s_i - (p^d_i + Δp^d_i) - g^s_i |V_i|²
#     Σ q_arcs = Σ qg                          - (q^d_i + Δq^d_i) + b^s_i |V_i|²
#
# which is the model in the plan: the battery injects at unity power factor, the
# recourse pair enters the ACTIVE balance with opposite signs, the reactive
# balance is HARD, and demand is parameterized by the fixed deviations Δp^d, Δq^d
# that carry the realized stage/atom demand. `ps` and `qs` are JuMP EXPRESSIONS,
# not variables, so this adds no variable and no equality row of its own.

using JuMP
using PowerModels
using Ipopt
using Clarabel
using LinearAlgebra
using Printf
import MathOptInterface as MOI

# Include guards: several public files include the same shared sources, and a
# second `include` of a file that defines a struct is an error, not a no-op.
@isdefined(BatterySpec) || include(joinpath(@__DIR__, "battery_case.jl"))
@isdefined(SolutionRecorder) || include(joinpath(@__DIR__, "battery_solution_schema.jl"))

PowerModels.silence()

# ─────────────────────────────────────────────────────────────────────────────
# Solvers
#
# One definition each, used by the diagnostics, by the SDDP baseline and by the
# regression suite. Two engines that disagree because they were solved at
# different tolerances is a failure mode this study has already paid for once.
# ─────────────────────────────────────────────────────────────────────────────

"""
    acp_optimizer(; tol=1e-10, max_iter=3000) -> JuMP optimizer factory

Interior-point NLP solver for the true-ACP model.

# Notes
Tolerance is tightened well below the physical tolerances the study reports at,
because a variable parked a solver tolerance below its bound shifts a
positively-priced objective by a near-constant amount every stage, which then
looks like a systematic model difference between two engines.
"""
acp_optimizer(; tol::Real = 1e-10, max_iter::Integer = 3000) =
    JuMP.optimizer_with_attributes(Ipopt.Optimizer,
                                   "print_level" => 0,
                                   "tol" => tol,
                                   "max_iter" => Int(max_iter),
                                   "sb" => "yes")

"""
    socwr_optimizer(; tol=1e-8, equilibrate=true, max_iter=10_000)
        -> JuMP optimizer factory

Conic solver for the SOC-WR relaxation.

# Notes
Nothing here retries a failed solve at a different setting: a retry ladder
chooses which duals become cuts, and cut generation must stay stock. The
settings are therefore chosen ONCE, by measurement, to be the ones under which
the solver does not fail in the first place. Two independent measurements fix
them, and they pull in opposite directions:

**Tolerance and iteration cap, measured 2026-08-06.** These are ONE frozen
correctness configuration, chosen once and applied to every candidate. They
correct an invalid inherited setting; they are not tuned per case.

The earlier default of `tol = 1e-6, max_iter = 500` did not merely lose
precision — it returned the WRONG ANSWER while reporting `OPTIMAL`. Solves are
bit-reproducible (five repeats, spread exactly 0.00), so this is not noise. On
one cycle-module case with the storage target pinned:

| tolerance | max_iter | status | objective |
|---|---|---|---|
| `1e-6` | 500 … 50,000 | OPTIMAL | 296,951 |
| `1e-8` | 500 | ALMOST_OPTIMAL | 323,913 |
| `1e-8` | 2,000 | ALMOST_OPTIMAL | 350,788 |
| **`1e-8`** | **10,000** | **OPTIMAL** | **351,411** |
| `1e-8` | 50,000 | OPTIMAL | 351,411 |

The true optimum is 351,411, against ACP's 358,806 — a 2.1 % relaxation gap. At
`1e-6` the reported value is 15.5 % BELOW it, and no iteration budget helps,
because termination is on the loose tolerance itself. `max_iter = 500` was a
budget inherited from a 57-bus case; a several-hundred-bus conic program needs
far more.

Why this matters beyond precision: SDDP consumes these solves' DUALS as cuts.
Objectives wrong by percent-level amounts, varying with the pinned state, make
successive cuts mutually inconsistent — which is how individually-healthy
subproblems become a collectively infeasible SDDP.

`max_iter` is a CAP, not a target; record the iterations and time actually used.

**Objective convergence is necessary but NOT sufficient.** A production
subproblem must satisfy all three: `OPTIMAL` status, a stable objective, and
stable STATE DUALS validated against finite differences (or one-sided brackets
at a kink). The duals can still move after the objective has settled, and the
duals are what become cuts. `ALMOST_OPTIMAL` may be kept as diagnostic evidence
but can never generate an accepted cut or a headline number.

**Equilibration is ON, and the objective is PHYSICAL, measured 2026-08-10.**
The two settings were tested together, as a 2×2×2 sweep over objective scaling,
equilibration and tolerance, at two BLAS thread counts, on `case500_goc` bare
and with one cycle module added to it. Only one corner solves the problem that
was posed:

| objective | equilibration | status | objective | max violation |
|---|---|---|---|---|
| **physical** | **on** | **OPTIMAL**, 27–30 iterations | **453 838.4553 / 617 807.7916** | **≤6.3e-9** |
| physical | off | OPTIMAL, 97–208 iterations | 453 590.70 / 617 387.25 | 3.5e-4 |
| divided by the recourse price | on | NUMERICAL_ERROR / SLOW_PROGRESS | — | — |
| divided by the recourse price | off | ALMOST_OPTIMAL / NUMERICAL_ERROR | — | — |

Read the second row before the third: with equilibration off, the lower
objective is not a better optimum, it is a point that violates the conic model
by 2.4e-4–3.5e-4 and is not feasible. Only the first row returns a converged
value at a residual worth quoting, and it is the only one that is `OPTIMAL` at
both thread counts on both arms. Tightening to `1e-10` reproduces its objective
to four decimals in one extra iteration.

Rescaling the objective is therefore rejected as a numerical device: a solver
whose own equilibration already normalizes rows and columns gains nothing from
a second, cruder rescaling of one block, and Clarabel measurably fails on the
rescaled program. Everything this file builds, solves and reports is in one
physical unit system — the objective units of the PGLib case — with no
conversion anywhere.

The lesson stands, only pointed the other way: a setting that makes an isolated
probe stop failing is not thereby the setting that makes the whole method run.
"""
socwr_optimizer(; tol::Real = 1e-8, equilibrate::Bool = true,
                  max_iter::Integer = 10_000) =
    JuMP.optimizer_with_attributes(Clarabel.Optimizer,
                                   "verbose" => false,
                                   "tol_gap_abs" => tol,
                                   "tol_gap_rel" => tol,
                                   "tol_feas" => tol,
                                   "equilibrate_enable" => equilibrate,
                                   "max_iter" => Int(max_iter))

const ACCEPTED_STATUSES = (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)

"""
    worst_recourse(sol) -> Float64

The largest USE of physical active-power recourse in a solved stage, in pu.

# Notes
Clamped at zero on purpose. An interior-point solver parks a nonnegative variable
a tolerance BELOW its zero bound — this study routinely sees `-1e-8` — and a
"worst recourse" of `-1e-8` is not a negative amount of unserved load, it is
zero. Reporting the raw minimum would also make the admissibility rule compare a
negative number against a positive tolerance and pass for the wrong reason.
"""
worst_recourse(sol) = max(0.0,
                          maximum(values(sol.deficit); init = 0.0),
                          maximum(values(sol.surplus); init = 0.0))

# ─────────────────────────────────────────────────────────────────────────────
# Injection carriers
# ─────────────────────────────────────────────────────────────────────────────

"""
    network_with_carriers(case::BatteryCase) -> Dict{String,Any}

Return a copy of the frozen network augmented with one PowerModels `storage`
element per bus.

# Arguments
- `case::BatteryCase`: the frozen case.

# Returns
- A deep copy of `case.network` whose `"storage"` table has exactly one entry
  per bus, keyed by the bus identifier as a string.

# Notes
The storage elements carry no dynamics of their own: this file never calls
`PowerModels.variable_storage_power`, `constraint_storage_state`,
`constraint_storage_losses`, `constraint_storage_complementarity_*` or
`constraint_storage_thermal_limit`. Their sole role is to make `ref[:bus_storage]`
nonempty so that the stock nodal balance contains a `ps`/`qs` term this file can
bind to an expression (see the file header).

Their numeric fields are filled with neutral, valid values only so that
PowerModels' own data checks accept the table; they are never read by any
constraint that this problem specification builds. The battery ratings that
matter live in `case.batteries` and are enforced by this file's own bounds.

The network is COPIED because PowerModels mutates the dictionaries it is given;
sharing one parse across two formulations is how a case export once stopped
being byte-reproducible.

**Generator costs come through UNCHANGED.** The polynomial coefficients this
returns are the case's own, so `PowerModels.objective_min_fuel_and_flow_cost`
builds the physical generation cost and every stage model this file assembles is
in the case's physical objective units. A stock `PowerModels.solve_opf` on
`case.network` is then a valid external reference for any of them, comparable
without conversion.
"""
function network_with_carriers(case::BatteryCase)
    net = deepcopy(case.network)
    # Fail closed on component classes this study's cost accounting does not
    # decompose. PowerModels would happily build them and their cost would then
    # sit inside the objective but outside `cost_generation`, so a cross-engine
    # objective comparison would disagree with the sum of its own parts.
    isempty(get(net, "dcline", Dict())) ||
        error("network_with_carriers: HVDC lines are not supported by this study's cost decomposition")
    isempty(get(net, "switch", Dict())) ||
        error("network_with_carriers: switches are not supported by this study")
    haskey(net, "storage") && !isempty(net["storage"]) &&
        error("network_with_carriers: the case already declares storage; the injection carriers would collide with it")

    # The cost decomposition in `extract_stage_solution` re-evaluates each
    # generator's polynomial at the reported dispatch, so a piecewise-linear cost
    # model would land inside PowerModels' objective but outside that sum.
    for (_, gen) in net["gen"]
        haskey(gen, "cost") || continue
        Int(get(gen, "model", 2)) == 2 ||
            error("network_with_carriers: only polynomial (model 2) generator costs are supported")
    end

    storage = Dict{String,Any}()
    for (_, bus) in net["bus"]
        i = Int(bus["index"])
        storage[string(i)] = Dict{String,Any}(
            "index" => i,
            "storage_bus" => i,
            "status" => 1,
            # Neutral, valid, and unused: no constraint built here reads them.
            "energy" => 0.0, "energy_rating" => 0.0,
            "charge_rating" => 0.0, "discharge_rating" => 0.0,
            "charge_efficiency" => 1.0, "discharge_efficiency" => 1.0,
            "thermal_rating" => 0.0, "qmin" => 0.0, "qmax" => 0.0,
            "r" => 0.0, "x" => 0.0, "p_loss" => 0.0, "q_loss" => 0.0,
            "ps" => 0.0, "qs" => 0.0,
        )
    end
    net["storage"] = storage
    return net
end

# ─────────────────────────────────────────────────────────────────────────────
# Problem specification
# ─────────────────────────────────────────────────────────────────────────────

"""
    BatteryStateMode

How the battery energy state is carried by a stage model.

- `:sddp`   the outgoing energy is an `SDDP.State` variable supplied by the
            caller; the stage model has no target and no target multiplier.
            This is the TARGETLESS formulation stock SDDP trains on.
- `:strict` the incoming energy is data, and the outgoing energy is pinned by
            the HARD equality ``e_{b} = \\hat e_b`` whose multiplier is the
            actor signal. There is no target slack and no target penalty.
- `:free`   the outgoing energy is a free decision inside its own bounds and
            there is no target at all. The incoming energy is data when
            `state_in` is supplied, and a free variable — to be linked
            externally, as a multiperiod deterministic equivalent does — when it
            is not. This is the DIAGNOSTIC mode: a myopic one-stage solve, or one
            stage of a perfect-foresight solve.
- `:none`   the case's batteries are not installed at all. Used to show that
            this problem specification reduces to ordinary PowerModels OPF.

`:free` is not a third scientific formulation. It carries no target and
therefore no target multiplier, so no policy can be trained on it; it exists so
that the physics of a stage, and the value of the energy a stage inherits, can be
probed with the SAME builders the study's two formulations use, instead of with a
second model that would have to be validated all over again.
"""
const BATTERY_STATE_MODES = (:sddp, :strict, :free, :none)

"""
    BatterySpecification

Everything the battery layer needs in order to build one stage model.

# Fields
- `case::BatteryCase`: the frozen case.
- `stage::Int`: stage index ``t``, used only to select the stage's entry of the
  frozen demand support.
- `atom::Int`: index of the realized demand atom at this stage. In an SDDP node
  it is a placeholder overwritten by `SDDP.parameterize`.
- `mode::Symbol`: one of [`BATTERY_STATE_MODES`](@ref).
- `state_in`: `nothing`, or a `Dict{Int,Float64}` of incoming energies
  (`:strict`, and optionally `:free`), or a `Dict{Int,Any}` of JuMP
  incoming-state references (`:sddp`).
- `state_out`: `nothing`, or a `Dict{Int,Any}` of JuMP outgoing-state
  references (`:sddp`).
- `target`: `nothing`, or a `Dict{Int,Float64}` of outgoing-energy targets
  (`:strict`).

# Notes
One specification builds either formulation of the network: the model
constructor passed to `PowerModels.instantiate_model` decides whether the
network equations are `ACPPowerModel`'s or `SOCWRConicPowerModel`'s, and this
struct is unchanged by that choice.
"""
struct BatterySpecification
    case::BatteryCase
    stage::Int
    atom::Int
    mode::Symbol
    state_in::Any
    state_out::Any
    target::Any
end

function BatterySpecification(case::BatteryCase;
                              stage::Integer = 1,
                              atom::Integer = 1,
                              mode::Symbol = :strict,
                              state_in = nothing,
                              state_out = nothing,
                              target = nothing)
    mode in BATTERY_STATE_MODES ||
        throw(ArgumentError("mode must be one of $BATTERY_STATE_MODES, got :$mode"))
    if mode === :strict
        state_in === nothing && throw(ArgumentError(":strict requires incoming energies"))
        target === nothing && throw(ArgumentError(":strict requires outgoing targets"))
    elseif mode === :sddp
        (state_in === nothing || state_out === nothing) &&
            throw(ArgumentError(":sddp requires the SDDP state references"))
    elseif mode === :free
        # A target in a targetless mode would be silently ignored, which is the
        # one outcome worse than an error: the caller would believe a target was
        # imposed and read a value that was never constrained.
        target === nothing || throw(ArgumentError(":free is targetless; pass mode = :strict to impose a target"))
    end
    return BatterySpecification(case, Int(stage), Int(atom), mode, state_in, state_out, target)
end

"""
    build_battery_opf(pm::PowerModels.AbstractPowerModel, spec::BatterySpecification)

The shared PowerModels problem specification for the battery-storage study.

# Arguments
- `pm`: a model instantiated by `PowerModels.instantiate_model`; its concrete
  type — `ACPPowerModel` or `SOCWRConicPowerModel` — selects the network
  formulation and is not inspected here.
- `spec::BatterySpecification`: the battery layer's data for this stage.

# Notes
The body is `PowerModels.build_opf` with the storage block replaced. Everything
electrical is a call into PowerModels; everything battery-related is a call into
this file. That split is the point of the whole design and is asserted by the
provenance test: `pm` must be an actual PowerModels model type, and no AC
trigonometric or SOC-WR lifted branch equation is written here.

Build ORDER matters and is not cosmetic. The nodal balance is closed over the
`ps`/`qs` entries that exist when it is written, so the battery layer's
variables and its injection expressions are installed BEFORE
`constraint_power_balance` runs. Writing them afterwards would silently produce
a network with no battery in it and no error anywhere.
"""
function build_battery_opf(pm::PowerModels.AbstractPowerModel, spec::BatterySpecification)
    # ── 1. Stock PowerModels variables ───────────────────────────────────────
    PowerModels.variable_bus_voltage(pm)
    PowerModels.variable_gen_power(pm)
    PowerModels.variable_branch_power(pm)
    PowerModels.variable_dcline_power(pm)

    # ── 2. Battery layer variables and nodal injection carriers ──────────────
    _battery_variables!(pm, spec)

    # ── 3. Stock generator/dcline cost, then the battery layer's own costs ───
    PowerModels.objective_min_fuel_and_flow_cost(pm)
    _battery_objective!(pm, spec)

    # ── 4. Stock PowerModels constraints ─────────────────────────────────────
    PowerModels.constraint_model_voltage(pm)

    for i in PowerModels.ids(pm, :ref_buses)
        PowerModels.constraint_theta_ref(pm, i)
    end

    for i in PowerModels.ids(pm, :bus)
        PowerModels.constraint_power_balance(pm, i)
    end

    for i in PowerModels.ids(pm, :branch)
        PowerModels.constraint_ohms_yt_from(pm, i)
        PowerModels.constraint_ohms_yt_to(pm, i)
        PowerModels.constraint_voltage_angle_difference(pm, i)
        PowerModels.constraint_thermal_limit_from(pm, i)
        PowerModels.constraint_thermal_limit_to(pm, i)
    end

    for i in PowerModels.ids(pm, :dcline)
        PowerModels.constraint_dcline_power_losses(pm, i)
    end

    # ── 5. Battery dynamics and the strict target equality ───────────────────
    _battery_constraints!(pm, spec)
    return nothing
end

"""
    _battery_variables!(pm, spec)

Declare the battery layer's variables and bind the nodal injection carriers.

# Notes
Declared here, in this order:

1. `dpd[i]`, `dqd[i]` — FIXED variables carrying the realized demand deviation
   at bus `i`. Fixed rather than baked in as constants because the realized
   demand is the stage's noise: `SDDP.parameterize` and the strict evaluator
   both move them with `JuMP.fix`, which is the only way a formulation whose
   balance is written once can see a new demand.
2. `d[i] ≥ 0`, `s[i] ≥ 0` — the two-sided physical active recourse. Neither has
   an upper bound of any kind. Capping `d` by the realized demand — natural if
   `d` were only curtailed load — would destroy relatively complete recourse for
   strict charging targets, so it is deliberately absent.
3. `p_ch[b] ∈ [0, \\overline p^{ch}_b]`, `p_dis[b] ∈ [0, \\overline p^{dis}_b]`,
   and the energy state, whose form depends on `spec.mode`.
4. `ps[i]`, `qs[i]` — EXPRESSIONS, registered into `PowerModels.var` so the
   stock balance picks them up.
"""
function _battery_variables!(pm::PowerModels.AbstractPowerModel, spec::BatterySpecification)
    model = pm.model
    case = spec.case
    Δt = stage_hours(case)
    buses = sort!(collect(PowerModels.ids(pm, :bus)))

    # ── Demand parameterization ──────────────────────────────────────────────
    # `pd_nom` and `qd_nom` are already inside the stock balance (PowerModels
    # reads them from `ref`), so the deviation carries the realized minus the
    # nominal demand. With every multiplier equal to 1 every deviation is 0 and
    # the model is exactly ordinary PowerModels OPF on the untouched case.
    #
    # The realized demand is computed per LOAD from the frozen support and then
    # aggregated to the bus, so a per-load or regional multiplier is carried
    # exactly; a bus-level factor would already have averaged it away.
    pd_nom, qd_nom = nominal_bus_demand(case)
    pd_real, qd_real = realized_bus_demand(case, spec.stage, spec.atom)
    dpd = JuMP.@variable(model, [i in buses], base_name = "dpd")
    dqd = JuMP.@variable(model, [i in buses], base_name = "dqd")
    for i in buses
        JuMP.fix(dpd[i], pd_real[i] - pd_nom[i]; force = true)
        JuMP.fix(dqd[i], qd_real[i] - qd_nom[i]; force = true)
    end

    # ── Two-sided uncapped physical active recourse ──────────────────────────
    d = JuMP.@variable(model, [i in buses], lower_bound = 0.0, base_name = "d")
    s = JuMP.@variable(model, [i in buses], lower_bound = 0.0, base_name = "s")

    # ── Battery controls and state ───────────────────────────────────────────
    batteries = spec.mode === :none ? BatterySpec[] : case.batteries
    ids = [b.index for b in batteries]
    byid = Dict(b.index => b for b in batteries)
    p_ch = JuMP.@variable(model, [k in ids], lower_bound = 0.0,
                          upper_bound = byid[k].charge_max, base_name = "p_ch")
    p_dis = JuMP.@variable(model, [k in ids], lower_bound = 0.0,
                           upper_bound = byid[k].discharge_max, base_name = "p_dis")

    e_in = Dict{Int,Any}()
    e_out = Dict{Int,Any}()
    if spec.mode === :sddp
        # The caller owns the SDDP.State pair; the battery layer only reads it.
        for k in ids
            e_in[k] = spec.state_in[k]
            e_out[k] = spec.state_out[k]
        end
    elseif spec.mode === :strict || spec.mode === :free
        # Incoming energy is a free variable here and is pinned by an equality in
        # `_battery_constraints!` when data was supplied, so that its multiplier
        # is available: that multiplier is ∂Q_t/∂e_{t-1}, the second half of the
        # actor signal for the PREVIOUS stage's target, and — in `:free` mode —
        # the marginal value of the energy a stage inherits.
        ein = JuMP.@variable(model, [k in ids], base_name = "e_in")
        eout = JuMP.@variable(model, [k in ids],
                              lower_bound = byid[k].energy_min,
                              upper_bound = byid[k].energy_max, base_name = "e_out")
        for k in ids
            e_in[k] = ein[k]
            e_out[k] = eout[k]
        end
    end

    # ── Nodal injection carriers, as expressions ─────────────────────────────
    # p^bat_i aggregates every battery at bus i; unity power factor means the
    # reactive carrier holds only the demand deviation.
    pbat = Dict{Int,Any}(i => JuMP.AffExpr(0.0) for i in buses)
    for b in batteries
        JuMP.add_to_expression!(pbat[b.bus], 1.0, p_dis[b.index])
        JuMP.add_to_expression!(pbat[b.bus], -1.0, p_ch[b.index])
    end
    ps = Dict{Int,Any}()
    qs = Dict{Int,Any}()
    for i in buses
        ps[i] = dpd[i] - pbat[i] - d[i] + s[i]
        qs[i] = JuMP.AffExpr(0.0) + dqd[i]
    end
    PowerModels.var(pm)[:ps] = ps
    PowerModels.var(pm)[:qs] = qs

    # Keep everything reachable for constraints, objective and extraction. The
    # battery layer's own bookkeeping lives in `pm.ext`, never in `ref`, so it
    # cannot collide with a PowerModels key.
    pm.ext[:battery] = Dict{Symbol,Any}(
        :spec => spec, :buses => buses, :ids => ids, :byid => byid,
        :dpd => dpd, :dqd => dqd, :d => d, :s => s,
        :p_ch => p_ch, :p_dis => p_dis, :e_in => e_in, :e_out => e_out,
        :pbat => pbat, :Δt => Δt,
        :pd_nom => pd_nom, :qd_nom => qd_nom,
        # The REALIZED demand of this stage/atom, kept so extraction reports what
        # the model was actually solved at rather than re-deriving it.
        :pd_real => pd_real, :qd_real => qd_real,
    )
    return nothing
end

"""
    _battery_objective!(pm, spec)

Add the battery layer's cost terms to the stock PowerModels objective.

# Notes
The added term is

```math
\\sum_b c^{deg}_b \\Delta t\\,(p^{ch}_b + p^{dis}_b)
+ C^{def}\\sum_i d_i + C^{sur}\\sum_i s_i,
```

every coefficient taken from the frozen case, so the ACP model, the SOC-WR model
and both SDDP passes charge for the same thing at the same price. The recourse
prices are far above any generator's marginal cost: they guarantee the strict
target is attainable, they do not make attaining it economical.

**Units.** The generator cost already in the objective is the case's own
physical polynomial ([`network_with_carriers`](@ref) does not touch it), and the
prices added here are the case's own, so the assembled objective IS the physical
stage cost. Nothing in it is rescaled and nothing mixes unit systems, which is
why every quantity `extract_stage_solution` reads off the solved model — cost,
price, multiplier alike — needs no conversion.
"""
function _battery_objective!(pm::PowerModels.AbstractPowerModel, spec::BatterySpecification)
    bat = pm.ext[:battery]
    case = spec.case
    Δt = bat[:Δt]
    extra = JuMP.AffExpr(0.0)
    for k in bat[:ids]
        b = bat[:byid][k]
        JuMP.add_to_expression!(extra, b.throughput_cost * Δt, bat[:p_ch][k])
        JuMP.add_to_expression!(extra, b.throughput_cost * Δt, bat[:p_dis][k])
    end
    for i in bat[:buses]
        JuMP.add_to_expression!(extra, case.recourse.deficit, bat[:d][i])
        JuMP.add_to_expression!(extra, case.recourse.surplus, bat[:s][i])
    end
    JuMP.set_objective_function(pm.model, JuMP.objective_function(pm.model) + extra)
    return nothing
end

"""
    _battery_constraints!(pm, spec)

Add the battery state transition and, in `:strict` mode, the target equality.

# Notes
The transition is

```math
e_{b} - \\alpha_b e_{b}^{in}
  - \\eta^{ch}_b \\Delta t\\, p^{ch}_b
  + \\frac{\\Delta t}{\\eta^{dis}_b} p^{dis}_b = 0,
```

with ``e_b`` the END-of-stage energy. Neither ``d`` nor ``s`` appears in it:
the recourse is a property of the NETWORK balance, not of the battery, and a
recourse term inside the state equation would be target slack wearing a
physical name.

In `:strict` mode two further equalities are added, and both are recorded so
their duals can be read:

- `energy_in[b]`: pins the incoming energy to data. Its dual is
  ``\\partial Q_t / \\partial e_{b,t-1}``.
- `target[b]`: pins the outgoing energy to the policy's target. Its dual is
  ``\\lambda_{b,t} = \\partial Q_t / \\partial \\hat e_{b,t}``.

The actor signal for target ``\\hat e_{b,t}`` is the SUM of the `target` dual at
stage ``t`` and the `energy_in` dual at stage ``t+1``, because in strict mode
the emitted target IS the next stage's incoming state. Nothing else couples the
stages: with both ends of every battery pinned, the strict trajectory separates
into `T` independent stage problems. That separation is what lets the JuMP
engine replay an Exa trajectory stage by stage.
"""
function _battery_constraints!(pm::PowerModels.AbstractPowerModel, spec::BatterySpecification)
    bat = pm.ext[:battery]
    model = pm.model
    Δt = bat[:Δt]

    transition = Dict{Int,Any}()
    for k in bat[:ids]
        b = bat[:byid][k]
        transition[k] = JuMP.@constraint(model,
            bat[:e_out][k] - b.self_discharge * bat[:e_in][k]
            - b.charge_efficiency * Δt * bat[:p_ch][k]
            + (Δt / b.discharge_efficiency) * bat[:p_dis][k] == 0.0)
    end
    bat[:transition] = transition

    if spec.mode === :strict || spec.mode === :free
        # The incoming-energy equality is written whenever incoming energy is
        # DATA. In `:free` mode with `state_in === nothing` the incoming energy
        # stays a free variable that the caller links to the previous stage.
        if spec.state_in !== nothing
            energy_in_con = Dict{Int,Any}()
            for k in bat[:ids]
                energy_in_con[k] = JuMP.@constraint(model,
                    bat[:e_in][k] == Float64(spec.state_in[k]))
            end
            bat[:energy_in_con] = energy_in_con
        end
        if spec.mode === :strict
            target_con = Dict{Int,Any}()
            for k in bat[:ids]
                target_con[k] = JuMP.@constraint(model,
                    bat[:e_out][k] == Float64(spec.target[k]))
            end
            bat[:target_con] = target_con
        end
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Stage model construction and solution
# ─────────────────────────────────────────────────────────────────────────────

"""
    apply_stage_availability!(net, stage) -> Int

Scale the limits of every scheduled generator of `net` by its own availability at
`stage`, IN PLACE, and return how many generators were scaled.

# Arguments
- `net::AbstractDict`: a parsed network, about to be handed to PowerModels.
- `stage::Integer`: the stage being built, `1`-based.

# Notes
`pmin`, `pmax`, `qmin` and `qmax` are all scaled by the same multiplier, so an
availability of `0` takes the unit out of service completely — active AND
reactive — rather than leaving a generator that cannot generate but can still
hold up a voltage for free. The multiplier form generalises past that one case: a
resource whose capacity varies through the day is the same statement with
fractions in it.

The generator is scaled rather than deleted, and `gen_status` is deliberately not
touched. PowerModels drops out-of-service generators from `ref`, so flipping the
status would change the variable set — and therefore the model's shape — from one
stage to the next. Scaling keeps every stage's model structurally identical, with
`pg` pinned at zero where the unit is unavailable, and a polynomial cost
evaluated at zero contributes exactly zero to the objective.

Fail-closed on a schedule that does not cover the requested stage: a case that
declares a two-stage schedule and is then solved at stage 3 has been mixed up
with a different case, and silently reusing the last entry would hide that.
"""
function apply_stage_availability!(net::AbstractDict, stage::Integer)
    stage >= 1 || throw(ArgumentError("stage must be 1-based, got $stage"))
    scaled = 0
    for (id, gen) in net["gen"]
        haskey(gen, STAGE_AVAILABILITY_KEY) || continue
        sched = gen[STAGE_AVAILABILITY_KEY]
        (sched isa AbstractVector && !isempty(sched)) ||
            error("generator $id: \"$STAGE_AVAILABILITY_KEY\" must be a non-empty vector of multipliers")
        stage <= length(sched) ||
            error("generator $id: \"$STAGE_AVAILABILITY_KEY\" covers $(length(sched)) stages but stage $stage was requested")
        a = Float64(sched[stage])
        (isfinite(a) && a >= 0) ||
            error("generator $id: availability $a at stage $stage is not a finite nonnegative multiplier")
        for k in ("pmin", "pmax", "qmin", "qmax")
            haskey(gen, k) && (gen[k] = Float64(gen[k]) * a)
        end
        scaled += 1
    end
    return scaled
end

"""
    battery_stage_model(case, model_type; optimizer, kwargs...)
        -> PowerModels.AbstractPowerModel

Instantiate one stage model of the battery study.

# Arguments
- `case::BatteryCase`: frozen case.
- `model_type::Type`: `PowerModels.ACPPowerModel` (the TRUE model) or
  `PowerModels.SOCWRConicPowerModel` (the relaxation SDDP's backward pass uses).

# Keywords
- `optimizer`: JuMP optimizer factory; attached to the model that is created.
- `jump_model`: an existing JuMP model to build into (SDDP passes its
  subproblem here). Defaults to a fresh model.
- everything else is forwarded to [`BatterySpecification`](@ref).

# Returns
- The instantiated `pm`. `pm.model` is the JuMP model; `pm.ext[:battery]` holds
  the battery layer's variable references.

# Notes
`model_type` is passed straight to `PowerModels.instantiate_model`, so the
network formulation actually instantiated is PowerModels' own. Nothing in this
function branches on it.

Every stage model in the study is built here — the strict evaluator, the
diagnostics, and each SDDP subproblem — so the per-stage generator availability
applied below is applied once, for all of them. A consumer that assembled a stage
from `network_with_carriers` on its own would silently ignore it, which is why
nothing else in this study does that.

Dual reporting is requested through PowerModels' own `setting`, which makes it
store the nodal balance CONSTRAINT REFERENCES in `sol(pm, :bus, i)` under
`:lam_kcl_r` and `:lam_kcl_i`. That is the only supported way to reach the
balance rows: the balance is written by a single closed PowerModels function
that returns nothing, so a diagnostic that wants nodal prices either asks
PowerModels to keep the reference or rewrites the balance — and rewriting it is
exactly what this file does not do.
"""
function battery_stage_model(case::BatteryCase, model_type::Type;
                             optimizer = nothing,
                             jump_model = nothing,
                             kwargs...)
    spec = BatterySpecification(case; kwargs...)
    net = network_with_carriers(case)
    apply_stage_availability!(net, spec.stage)
    model = jump_model === nothing ?
            (optimizer === nothing ? JuMP.Model() : JuMP.Model(optimizer)) : jump_model
    if jump_model !== nothing && optimizer !== nothing
        JuMP.set_optimizer(model, optimizer)
    end
    return PowerModels.instantiate_model(net, model_type,
                                         pm -> build_battery_opf(pm, spec);
                                         jump_model = model,
                                         setting = Dict("output" => Dict("duals" => true)))
end

"""
    solve_strict_stage(case, model_type; stage, atom, energy_in, target,
                       optimizer, silent=true) -> NamedTuple

Build and solve one STRICT stage problem, returning its full physical solution.

# Arguments / Keywords
- `stage::Integer`, `atom::Integer`: which demand realization to impose.
- `energy_in::AbstractDict{Int,<:Real}`: incoming energy per battery identifier.
- `target::AbstractDict{Int,<:Real}`: outgoing-energy target per battery.
- `optimizer`: JuMP optimizer factory (Ipopt for ACP).
- `silent::Bool`: suppress solver output.

# Returns
A `NamedTuple` with the solver status, the objective decomposed into its
physical components, every physical variable keyed by NETWORK identifier, the
strict target multipliers, and the incoming-energy multipliers.

# Notes
The returned `target_dual` is the multiplier of ``e_b = \\hat e_b`` as JuMP
reports it for a minimization problem, i.e. the sensitivity of the stage
objective to the target. `energy_in_dual` is the corresponding sensitivity to
the incoming state. Together they give the full actor signal — see
[`_battery_constraints!`](@ref).

The solve is accepted only when the solver returns a status in
`(OPTIMAL, LOCALLY_SOLVED)`; anything else is reported as-is and the caller must
reject it. A solver status alone is still not sufficient, which is why the
physical residuals are computed separately from the returned solution.
"""
function solve_strict_stage(case::BatteryCase, model_type::Type;
                            stage::Integer,
                            atom::Integer,
                            energy_in::AbstractDict,
                            target::AbstractDict,
                            optimizer,
                            silent::Bool = true,
                            active_batteries::Bool = true)
    mode = active_batteries ? :strict : :none
    pm = battery_stage_model(case, model_type;
                             optimizer = optimizer,
                             stage = stage, atom = atom, mode = mode,
                             state_in = active_batteries ? Dict(Int(k) => Float64(v) for (k, v) in energy_in) : nothing,
                             target = active_batteries ? Dict(Int(k) => Float64(v) for (k, v) in target) : nothing)
    silent && JuMP.set_silent(pm.model)
    JuMP.optimize!(pm.model)
    return extract_stage_solution(pm)
end

"""
    extract_stage_solution(pm) -> NamedTuple

Read every physical quantity of a solved stage model, keyed by network id.

# Notes
Generator reactive power is reported BOTH per generator and aggregated to the
bus. Several generators may share a bus and the nodal balance constrains only
their sum, so per-generator values live in a null space that two engines are
free to split differently; the nodal aggregate is the physically determined
quantity, and comparing only the per-generator values would report a difference
that the model does not determine.

**Nothing here is converted, because nothing was rescaled.** The model that was
solved carries the case's physical stage cost, so its objective value, its
generation-cost polynomial, its nodal prices and its state multipliers are all
already in the case's objective units and are read off as they stand. A residual
compared against the solver's own KKT residuals, or a cost compared against a
stock `PowerModels.solve_opf`, therefore compares like with like — there is no
second unit system anywhere in this file for one of them to slip into.
"""
function extract_stage_solution(pm::PowerModels.AbstractPowerModel)
    model = pm.model
    bat = pm.ext[:battery]
    spec = bat[:spec]
    case = spec.case
    Δt = bat[:Δt]
    status = JuMP.termination_status(model)
    ok = status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)

    val(x) = JuMP.value(x)
    buses = bat[:buses]
    gen_ids = sort!(collect(PowerModels.ids(pm, :gen)))
    branch_ids = sort!(collect(PowerModels.ids(pm, :branch)))

    vm = Dict{Int,Float64}()
    va = Dict{Int,Float64}()
    if haskey(PowerModels.var(pm), :vm)
        # Polar forms carry the magnitude directly.
        for i in buses
            vm[i] = val(PowerModels.var(pm, :vm, i))
            va[i] = val(PowerModels.var(pm, :va, i))
        end
    else
        # W-space forms (SOC-WR) carry |V|²; the angle is not a variable there.
        for i in buses
            vm[i] = sqrt(max(val(PowerModels.var(pm, :w, i)), 0.0))
            va[i] = NaN
        end
    end

    pg = Dict{Int,Float64}(g => val(PowerModels.var(pm, :pg, g)) for g in gen_ids)
    qg = Dict{Int,Float64}(g => val(PowerModels.var(pm, :qg, g)) for g in gen_ids)
    pg_bus = Dict{Int,Float64}(i => 0.0 for i in buses)
    qg_bus = Dict{Int,Float64}(i => 0.0 for i in buses)
    for g in gen_ids
        b = Int(PowerModels.ref(pm, :gen, g)["gen_bus"])
        pg_bus[b] += pg[g]
        qg_bus[b] += qg[g]
    end

    p_fr = Dict{Int,Float64}(); q_fr = Dict{Int,Float64}()
    p_to = Dict{Int,Float64}(); q_to = Dict{Int,Float64}()
    for l in branch_ids
        br = PowerModels.ref(pm, :branch, l)
        f = (l, Int(br["f_bus"]), Int(br["t_bus"]))
        t = (l, Int(br["t_bus"]), Int(br["f_bus"]))
        p_fr[l] = val(PowerModels.var(pm, :p, f)); q_fr[l] = val(PowerModels.var(pm, :q, f))
        p_to[l] = val(PowerModels.var(pm, :p, t)); q_to[l] = val(PowerModels.var(pm, :q, t))
    end

    d = Dict{Int,Float64}(i => val(bat[:d][i]) for i in buses)
    s = Dict{Int,Float64}(i => val(bat[:s][i]) for i in buses)
    pd = Dict{Int,Float64}(i => bat[:pd_real][i] for i in buses)
    qd = Dict{Int,Float64}(i => bat[:qd_real][i] for i in buses)

    ids = bat[:ids]
    p_ch = Dict{Int,Float64}(k => val(bat[:p_ch][k]) for k in ids)
    p_dis = Dict{Int,Float64}(k => val(bat[:p_dis][k]) for k in ids)
    p_bat = Dict{Int,Float64}(k => p_dis[k] - p_ch[k] for k in ids)
    e_in = Dict{Int,Float64}(k => val(bat[:e_in][k]) for k in ids)
    e_out = Dict{Int,Float64}(k => val(bat[:e_out][k]) for k in ids)

    # Multipliers of the physical objective: objective units per pu-hour of
    # stored energy, read off with no conversion.
    target_dual = Dict{Int,Float64}()
    energy_in_dual = Dict{Int,Float64}()
    if JuMP.has_duals(model)
        if haskey(bat, :target_con)
            for k in ids
                target_dual[k] = JuMP.dual(bat[:target_con][k])
            end
        end
        if haskey(bat, :energy_in_con)
            for k in ids
                energy_in_dual[k] = JuMP.dual(bat[:energy_in_con][k])
            end
        end
    end

    # ── Nodal prices ─────────────────────────────────────────────────────────
    # PowerModels writes the active balance at bus i as
    #
    #     Σ p_arcs − Σ pg + Σ ps + p^d_i + g^s_i |V_i|² = 0,
    #
    # i.e. `h(x) = −p^d_i`. JuMP's multiplier λ of `h(x) == b` in a minimization
    # is ∂obj/∂b, so the price of demand — the quantity anyone means by a nodal
    # price — is ∂obj/∂p^d_i = −λ. The sign is derived here once rather than
    # guessed, and it is checked in the regression suite against a finite
    # difference of the realized demand.
    price_active = Dict{Int,Float64}()
    price_reactive = Dict{Int,Float64}()
    if JuMP.has_duals(model)
        for i in buses
            bus_sol = PowerModels.sol(pm, :bus, i)
            haskey(bus_sol, :lam_kcl_r) && (price_active[i] = -JuMP.dual(bus_sol[:lam_kcl_r]))
            haskey(bus_sol, :lam_kcl_i) && (price_reactive[i] = -JuMP.dual(bus_sol[:lam_kcl_i]))
        end
    end

    # ── Cost decomposition ───────────────────────────────────────────────────
    # `ref` holds the case's own polynomial and the three remaining components
    # are built from the case's own prices, so all four are in the same units as
    # the objective they must sum to.
    cost_generation = 0.0
    for g in gen_ids
        gen = PowerModels.ref(pm, :gen, g)
        cost_generation += _polynomial_cost(gen, pg[g])
    end
    cost_throughput = sum(bat[:byid][k].throughput_cost * Δt * (p_ch[k] + p_dis[k])
                          for k in ids; init = 0.0)
    cost_deficit = case.recourse.deficit * sum(values(d); init = 0.0)
    cost_surplus = case.recourse.surplus * sum(values(s); init = 0.0)

    return (
        status = status, solved = ok,
        objective = ok ? JuMP.objective_value(model) : NaN,
        cost_generation = cost_generation,
        cost_throughput = cost_throughput,
        cost_deficit = cost_deficit,
        cost_surplus = cost_surplus,
        cost_stage = cost_generation + cost_throughput + cost_deficit + cost_surplus,
        vm = vm, va = va, pg = pg, qg = qg, pg_bus = pg_bus, qg_bus = qg_bus,
        p_fr = p_fr, q_fr = q_fr, p_to = p_to, q_to = q_to,
        deficit = d, surplus = s, pd = pd, qd = qd,
        p_ch = p_ch, p_dis = p_dis, p_bat = p_bat,
        energy_in = e_in, energy_out = e_out,
        target_dual = target_dual, energy_in_dual = energy_in_dual,
        price_active = price_active, price_reactive = price_reactive,
        simultaneous = Dict{Int,Float64}(k => min(p_ch[k], p_dis[k]) for k in ids),
        pm = pm,
    )
end

"""
    binding_constraints(pm, sol; tol=1e-6) -> Vector{NamedTuple}

Every physical limit of a solved stage that is binding or nearly binding.

# Arguments
- `pm`: the solved model (used for its network reference).
- `sol`: the named tuple returned by [`extract_stage_solution`](@ref).

# Keywords
- `tol::Real`: absolute slack, in the natural unit of each limit, below which a
  limit counts as binding.

# Returns
A vector of `(kind, index, side, value, limit, slack)` rows, sorted by slack, for

- generator active and reactive bounds (pu);
- bus voltage magnitude bounds (pu);
- branch apparent-power limits at BOTH ends (pu, compared on ``|S|`` rather than
  on ``|S|^2`` so the tolerance means the same thing on every branch);
- branch angle-difference limits (rad), where the formulation has angles.

# Notes
This answers "what stopped the network from delivering the energy" — the question
that separates a real relaxation-driven storage-value difference from a numerical
one. It reads the reported solution rather than the solver's active set, so it
says the same thing about a solution produced by either engine.
"""
function binding_constraints(pm::PowerModels.AbstractPowerModel, sol; tol::Real = 1e-6)
    rows = NamedTuple[]
    push_row!(kind, index, side, value, limit) = begin
        slack = abs(limit - value)
        slack <= tol && push!(rows, (kind = kind, index = index, side = side,
                                     value = float(value), limit = float(limit),
                                     slack = float(slack)))
    end

    for (g, val) in sol.pg
        gen = PowerModels.ref(pm, :gen, g)
        push_row!("pg", g, "min", val, Float64(gen["pmin"]))
        push_row!("pg", g, "max", val, Float64(gen["pmax"]))
    end
    for (g, val) in sol.qg
        gen = PowerModels.ref(pm, :gen, g)
        push_row!("qg", g, "min", val, Float64(gen["qmin"]))
        push_row!("qg", g, "max", val, Float64(gen["qmax"]))
    end
    for (i, val) in sol.vm
        bus = PowerModels.ref(pm, :bus, i)
        push_row!("vm", i, "min", val, Float64(bus["vmin"]))
        push_row!("vm", i, "max", val, Float64(bus["vmax"]))
    end
    for l in keys(sol.p_fr)
        br = PowerModels.ref(pm, :branch, l)
        rate = Float64(get(br, "rate_a", Inf))
        isfinite(rate) || continue
        push_row!("thermal", l, "from", hypot(sol.p_fr[l], sol.q_fr[l]), rate)
        push_row!("thermal", l, "to", hypot(sol.p_to[l], sol.q_to[l]), rate)
        if !isnan(sol.va[Int(br["f_bus"])])
            θ = sol.va[Int(br["f_bus"])] - sol.va[Int(br["t_bus"])]
            push_row!("angle", l, "min", θ, Float64(get(br, "angmin", -pi)))
            push_row!("angle", l, "max", θ, Float64(get(br, "angmax", pi)))
        end
    end
    sort!(rows; by = r -> r.slack)
    return rows
end

"""
    _polynomial_cost(gen, pg) -> Float64

Evaluate a PowerModels polynomial cost model at `pg`.

# Notes
PowerModels stores the polynomial highest-order-first with `ncost` coefficients,
so the value is ``\\sum_{j} c_j\\, pg^{ncost-j}``. Only `model == 2` (polynomial)
is supported; a piecewise-linear cost model would need its own evaluation and
none of the benchmarks used here carries one.

The result is in the units of the coefficients it was handed. Every model this
file builds carries the case's own coefficients, so calling it on `case.network`
and calling it on a model's `ref` return the same number for the same dispatch.
"""
function _polynomial_cost(gen::AbstractDict, pg::Real)
    Int(get(gen, "model", 2)) == 2 ||
        error("only polynomial (model 2) generator costs are supported")
    cost = Float64.(gen["cost"])
    n = Int(gen["ncost"])
    total = 0.0
    for (j, c) in enumerate(cost[(end - n + 1):end])
        total += c * pg^(n - j)
    end
    return total
end

"""
    record_stage_solution!(rec, sol, case; scenario, stage)

Write one solved stage into a [`SolutionRecorder`](@ref) in the shared schema.
"""
function record_stage_solution!(rec::SolutionRecorder, sol, case::BatteryCase;
                                scenario::Integer, stage::Integer)
    record_map!(rec, scenario, stage, "vm", sol.vm)
    record_map!(rec, scenario, stage, "va", sol.va)
    record_map!(rec, scenario, stage, "pg", sol.pg)
    record_map!(rec, scenario, stage, "qg", sol.qg)
    record_map!(rec, scenario, stage, "pg_bus", sol.pg_bus)
    record_map!(rec, scenario, stage, "qg_bus", sol.qg_bus)
    record_map!(rec, scenario, stage, "p_fr", sol.p_fr)
    record_map!(rec, scenario, stage, "q_fr", sol.q_fr)
    record_map!(rec, scenario, stage, "p_to", sol.p_to)
    record_map!(rec, scenario, stage, "q_to", sol.q_to)
    record_map!(rec, scenario, stage, "deficit", sol.deficit)
    record_map!(rec, scenario, stage, "surplus", sol.surplus)
    record_map!(rec, scenario, stage, "pd", sol.pd)
    record_map!(rec, scenario, stage, "qd", sol.qd)
    record_map!(rec, scenario, stage, "p_ch", sol.p_ch)
    record_map!(rec, scenario, stage, "p_dis", sol.p_dis)
    record_map!(rec, scenario, stage, "p_bat", sol.p_bat)
    record_map!(rec, scenario, stage, "energy_in", sol.energy_in)
    record_map!(rec, scenario, stage, "energy_out", sol.energy_out)
    isempty(sol.target_dual) || record_map!(rec, scenario, stage, "target_dual", sol.target_dual)
    isempty(sol.price_active) || record_map!(rec, scenario, stage, "price_active", sol.price_active)
    isempty(sol.price_reactive) || record_map!(rec, scenario, stage, "price_reactive", sol.price_reactive)
    record!(rec, scenario, stage, "cost_generation", 0, sol.cost_generation)
    record!(rec, scenario, stage, "cost_throughput", 0, sol.cost_throughput)
    record!(rec, scenario, stage, "cost_deficit", 0, sol.cost_deficit)
    record!(rec, scenario, stage, "cost_surplus", 0, sol.cost_surplus)
    record!(rec, scenario, stage, "cost_stage", 0, sol.cost_stage)
    record!(rec, scenario, stage, "objective", 0, sol.objective)
    record!(rec, scenario, stage, "solved", 0, sol.solved ? 1.0 : 0.0)
    return rec
end

# ─────────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────────

"""
    assert_powermodels_provenance(pm, expected::Type) -> Nothing

Assert, from the LIVE object, that the network formulation actually built is the
intended PowerModels one.

# Notes
This is the runtime half of the provenance gate. Numerical agreement with a
handwritten model is explicitly NOT an acceptable substitute: the claim the
study makes is that SDDP's backward pass is an actual
`PowerModels.SOCWRConicPowerModel` and its forward pass an actual
`PowerModels.ACPPowerModel`, and only the type of the instantiated object can
establish that.

Beyond the type, the assertion checks that the model carries the variables the
intended formulation is defined by — `vm`/`va` for the polar AC form, the lifted
`w`/`wr`/`wi` for the SOC-WR form — so that a type that had been made to alias
something else would still be caught.
"""
function assert_powermodels_provenance(pm::PowerModels.AbstractPowerModel, expected::Type)
    typeof(pm) === expected ||
        error("expected an actual $expected, got $(typeof(pm))")
    pm isa PowerModels.AbstractPowerModel ||
        error("$(typeof(pm)) is not a PowerModels.AbstractPowerModel")
    vars = keys(PowerModels.var(pm))
    if expected === PowerModels.ACPPowerModel
        (:vm in vars && :va in vars) ||
            error("ACPPowerModel is missing its polar voltage variables")
    elseif expected === PowerModels.SOCWRConicPowerModel
        (:w in vars && :wr in vars && :wi in vars) ||
            error("SOCWRConicPowerModel is missing its lifted voltage variables")
    end
    return nothing
end
