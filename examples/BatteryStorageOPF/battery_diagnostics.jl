# battery_diagnostics.jl
#
# The diagnostic toolkit: what a user runs on a case BEFORE training anything.
#
# Everything here is built out of the production builders in
# `battery_powermodels.jl`. There is no second network model, no duplicated
# balance equation and no separate cost accounting: a diagnostic that measures a
# different model from the one the study trains on measures nothing about the
# study. The only thing these functions add is which variables are pinned and
# which are free.
#
# The five tools, and the question each answers:
#
#   sample_incoming_energy   what states should I probe from?
#   targetless_probe         what does the network do at this state, right now?
#   energy_value_curve       what is the stored energy WORTH here, and do the
#                            true model and the relaxation disagree about it?
#   deterministic_equivalent what would a clairvoyant operator have done?
#   perfect_foresight_panel  how much headroom is there, over a protocol?
#
# The three are deliberately ordered from myopic to clairvoyant, because the
# first is the one that is easiest to over-read: a targetless ONE-STAGE solve is
# myopic and will rationally empty a battery, since nothing in it prices the
# energy it leaves behind. It diagnoses this stage's physics. It does not, by
# itself, measure the value of preserving energy — that is what
# `energy_value_curve` is for.

using JuMP
using PowerModels
using StableRNGs
using Statistics
using Printf
import MathOptInterface as MOI

@isdefined(BatterySpecification) || include(joinpath(@__DIR__, "battery_powermodels.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# B1 — incoming-energy sampling
# ─────────────────────────────────────────────────────────────────────────────

"""
    sample_incoming_energy(case; kind=:fixed, level=0.5, dist=nothing,
                           callable=nothing, explicit=nothing, seed=1, batch=1)
        -> Vector{Dict{Int,Float64}}

Reproducibly sample incoming battery-energy vectors inside their own bounds.

# Keywords
- `kind::Symbol`: one of
  - `:fixed` — every battery at the normalized state `level`;
  - `:uniform` — independent uniform normalized states in `[0, 1]`;
  - `:distribution` — independent draws of the NORMALIZED state from `dist`
    (anything supporting `rand(rng, dist)`), clamped into `[0, 1]`;
  - `:callable` — `callable(rng, battery, meta) -> normalized state`;
  - `:explicit` — `explicit` is a vector of `Dict{Int,Float64}` energy vectors,
    validated and returned.
- `level::Real`: the normalized state used by `:fixed`.
- `seed::Integer`: seed of the `StableRNG`.
- `batch::Integer`: how many vectors to draw.

# Returns
- A vector of `batch` dictionaries keyed by BATTERY identifier, in pu·h.

# Notes
The NORMALIZED state ``u_b \\in [0,1]`` maps to energy by
``e_b = \\underline e_b + u_b(\\overline e_b - \\underline e_b)``, so a sampler
expressed in normalized terms transfers unchanged to a case whose batteries have
different ratings. Every resulting energy is validated against its own bounds
before being returned.

This samples DIAGNOSTIC INITIAL STATES only. It is not the demand process and it
plays no part in training or in evaluation; a state drawn here is a place to
stand while measuring the network, not a scenario.
"""
function sample_incoming_energy(case::BatteryCase;
                                kind::Symbol = :fixed,
                                level::Real = 0.5,
                                dist = nothing,
                                callable = nothing,
                                explicit = nothing,
                                seed::Integer = 1,
                                batch::Integer = 1)
    kind in (:fixed, :uniform, :distribution, :callable, :explicit) ||
        throw(ArgumentError("kind must be :fixed, :uniform, :distribution, :callable or :explicit"))
    rng = StableRNG(seed)
    out = Dict{Int,Float64}[]

    if kind === :explicit
        explicit === nothing && throw(ArgumentError(":explicit requires the energy vectors"))
        for e in explicit
            push!(out, Dict{Int,Float64}(Int(k) => Float64(v) for (k, v) in e))
        end
    else
        for _ in 1:Int(batch)
            e = Dict{Int,Float64}()
            for b in case.batteries
                u = if kind === :fixed
                    Float64(level)
                elseif kind === :uniform
                    rand(rng)
                elseif kind === :distribution
                    dist === nothing && throw(ArgumentError(":distribution requires `dist`"))
                    clamp(Float64(rand(rng, dist)), 0.0, 1.0)
                else
                    callable === nothing && throw(ArgumentError(":callable requires `callable`"))
                    Float64(callable(rng, b, (case = case,)))
                end
                e[b.index] = b.energy_min + u * (b.energy_max - b.energy_min)
            end
            push!(out, e)
        end
    end

    byid = Dict(b.index => b for b in case.batteries)
    for e in out
        Set(keys(e)) == Set(keys(byid)) ||
            error("incoming-energy vector covers batteries $(sort!(collect(keys(e)))), expected $(sort!(collect(keys(byid))))")
        for (k, v) in e
            b = byid[k]
            isfinite(v) || error("battery $k: incoming energy is not finite")
            b.energy_min - 1e-12 <= v <= b.energy_max + 1e-12 ||
                error("battery $k: incoming energy $v outside [$(b.energy_min), $(b.energy_max)]")
        end
    end
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# B2 — the single-stage targetless probe
# ─────────────────────────────────────────────────────────────────────────────

"""
    targetless_probe(case, model_type; energy_in, stage, atom,
                     optimizer=nothing, pins=nothing, silent=true) -> NamedTuple

Solve ONE stage with the battery's charge, discharge and OUTGOING energy all
optimized, and return everything measurable about the result.

# Arguments
- `case::BatteryCase`, `model_type::Type`: `PowerModels.ACPPowerModel` or
  `PowerModels.SOCWRConicPowerModel`.

# Keywords
- `energy_in::AbstractDict`: incoming energy per battery identifier (pu·h).
- `stage::Integer`, `atom::Integer`: which demand realization to impose.
- `optimizer`: defaults to [`acp_optimizer`](@ref) / [`socwr_optimizer`](@ref)
  according to `model_type`.
- `pins::Union{Nothing,AbstractDict}`: outgoing energies to PIN by a hard
  equality, per battery identifier. `nothing` leaves every battery free — the
  targetless probe proper. Any battery not listed stays free.

# Returns
A `NamedTuple` extending [`extract_stage_solution`](@ref) with

- `residuals` — the engine-neutral physical residuals of the reported solution;
- `binding` — the binding and nearly binding limits;
- `simultaneous` — ``\\min(p^{ch}_b, p^{dis}_b)`` per battery, which is zero
  exactly when the solution does not charge and discharge at once;
- `energy_in_dual` — ``\\partial Q/\\partial e_{b}^{in}``, the marginal value of
  the energy the stage INHERITED;
- `pin_dual` — the multiplier of each pinned outgoing energy, i.e.
  ``\\partial Q/\\partial e_b^{out}``.

# Notes
Pinning is done with the same hard equality the strict formulation uses, so when
`pins` covers every battery the model built here is the strict stage problem.
The regression suite asserts exactly that, which is what lets one diagnostic
path serve both the free and the fixed probe without a second formulation.

**A targetless one-stage solve is myopic.** Nothing in it prices the energy left
in a battery at the end of the stage, so it will rationally discharge as much as
is useful and leave the battery empty. That is not a defect and it is not a
finding: it is the definition of a one-stage problem. Read it for the physics —
what the network could deliver, what bound stopped it, what the energy the stage
INHERITED was worth — and read [`energy_value_curve`](@ref) for the value of what
is left behind.
"""
function targetless_probe(case::BatteryCase, model_type::Type;
                          energy_in::AbstractDict,
                          stage::Integer,
                          atom::Integer,
                          optimizer = nothing,
                          pins::Union{Nothing,AbstractDict} = nothing,
                          silent::Bool = true)
    opt = optimizer === nothing ? _default_optimizer(model_type) : optimizer
    pm = battery_stage_model(case, model_type;
                             optimizer = opt,
                             stage = stage, atom = atom, mode = :free,
                             state_in = Dict(Int(k) => Float64(v) for (k, v) in energy_in))
    assert_powermodels_provenance(pm, model_type)

    pin_con = Dict{Int,Any}()
    if pins !== nothing
        bat = pm.ext[:battery]
        for (k, v) in pins
            haskey(bat[:e_out], Int(k)) || error("pin refers to unknown battery $k")
            pin_con[Int(k)] = JuMP.@constraint(pm.model, bat[:e_out][Int(k)] == Float64(v))
        end
    end

    silent && JuMP.set_silent(pm.model)
    JuMP.optimize!(pm.model)
    sol = extract_stage_solution(pm)

    # The pin is written on the model whose objective is the physical stage cost,
    # so its multiplier is already in the same units as the value curve a finite
    # difference of `sol.objective` would produce — the units the strict target
    # dual is reported in too.
    pin_dual = Dict{Int,Float64}()
    if JuMP.has_duals(pm.model)
        for (k, con) in pin_con
            pin_dual[k] = JuMP.dual(con)
        end
    end

    residuals = sol.solved ?
                physical_residuals(case.network, case.batteries, stage_hours(case), sol) :
                nothing
    binding = sol.solved ? binding_constraints(pm, sol) : NamedTuple[]

    return merge(sol, (residuals = residuals, binding = binding,
                       pin_dual = pin_dual, stage = Int(stage), atom = Int(atom),
                       model_type = model_type))
end

"""
    base_feasibility(case, model_type; stage, atom, optimizer=nothing, silent=true)
        -> NamedTuple

Solve the case's network with NO batteries installed, at one stage/atom demand
realization.

# Returns
The stage solution extended with `residuals`, `binding`, `worst_recourse` and the
nodal-price spread.

# Notes
This is the admissibility precondition of the entire strict-target construction:
the two-sided physical recourse guarantees that a dynamically reachable battery
target is attainable GIVEN that the base dispatch is feasible at the realized
demand. A case whose base problem already leans on the recourse has no such
guarantee, and every number measured on it would be measuring the recourse price
rather than the network.

It is also the cheapest possible screen of a candidate benchmark: it answers
"does this system serve this demand at all, and what stops it" before any
battery, any horizon or any policy exists.

`price_spread` is the ratio of the largest to the smallest active nodal price. It
is the direct locational signal: a system where every bus prices energy the same
cannot reward putting storage in one place rather than another, whatever its
topology looks like.
"""
function base_feasibility(case::BatteryCase, model_type::Type;
                          stage::Integer, atom::Integer,
                          optimizer = nothing, silent::Bool = true)
    opt = optimizer === nothing ? _default_optimizer(model_type) : optimizer
    t0 = time()
    pm = battery_stage_model(case, model_type;
                             optimizer = opt, stage = stage, atom = atom, mode = :none)
    assert_powermodels_provenance(pm, model_type)
    silent && JuMP.set_silent(pm.model)
    JuMP.optimize!(pm.model)
    elapsed = time() - t0
    sol = extract_stage_solution(pm)
    residuals = sol.solved ?
                physical_residuals(case.network, BatterySpec[], stage_hours(case), sol) :
                nothing
    binding = sol.solved ? binding_constraints(pm, sol) : NamedTuple[]
    prices = collect(values(sol.price_active))
    spread = isempty(prices) || minimum(prices) <= 0 ? NaN :
             maximum(prices) / minimum(prices)
    return merge(sol, (residuals = residuals, binding = binding,
                       worst_recourse = worst_recourse(sol),
                       price_spread = spread, elapsed = elapsed,
                       stage = Int(stage), atom = Int(atom), model_type = model_type))
end

"The solver each formulation is solved with unless the caller says otherwise."
function _default_optimizer(model_type::Type)
    model_type === PowerModels.ACPPowerModel && return acp_optimizer()
    model_type === PowerModels.SOCWRConicPowerModel && return socwr_optimizer()
    error("no default optimizer for $model_type; pass one explicitly")
end

"""
    targetless_batch(case, model_type; states, stages, atoms, optimizer)
        -> Vector{NamedTuple}

Run [`targetless_probe`](@ref) over the Cartesian product of incoming-energy
vectors, stages and atoms.

# Notes
Every combination is retained, including the ones that failed to solve: a batch
that silently drops its failures reports a success rate of 100 % by construction.
"""
function targetless_batch(case::BatteryCase, model_type::Type;
                          states::AbstractVector{<:AbstractDict},
                          stages::AbstractVector{<:Integer},
                          atoms::AbstractVector{<:Integer},
                          optimizer = nothing)
    out = NamedTuple[]
    for (si, e) in enumerate(states), t in stages, k in atoms
        p = targetless_probe(case, model_type; energy_in = e, stage = t, atom = k,
                             optimizer = optimizer)
        push!(out, merge(p, (state_index = si,)))
    end
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# B3 — the fixed-outgoing-energy value probe
# ─────────────────────────────────────────────────────────────────────────────

"""
    energy_value_curve(case, model_type; battery, energy_in, stage, atom, grid,
                       hold=nothing, optimizer=nothing, fd_tolerance=5e-3)
        -> NamedTuple

The stage value as a function of the outgoing energy of ONE battery, together
with its multiplier and a finite-difference check of that multiplier.

# Keywords
- `battery::Integer`: which battery's outgoing energy is swept.
- `energy_in::AbstractDict`: the incoming state to sweep from.
- `grid::AbstractVector{<:Real}`: outgoing energies to solve at. Values outside
  the battery's one-stage reachable interval are reported as UNREACHABLE rather
  than solved, because a hard equality on an unreachable target is an infeasible
  model, not an expensive one.
- `hold`: `nothing` to leave the other batteries free, or a
  `Dict{Int,Float64}` pinning them too.
- `fd_tolerance::Real`: relative agreement required between an interior
  multiplier and its central finite difference at a SMOOTH point.
- `kink_tolerance::Real`: relative disagreement between the one-sided slopes
  above which an interior point is classified nonsmooth.

# Returns
A `NamedTuple` with

- `energy::Vector{Float64}`, `value::Vector{Float64}` — the value curve;
- `lambda::Vector{Float64}` — the strict target multiplier at each grid point,
  i.e. ``\\partial Q/\\partial e^{out}``;
- `fd::Vector{Float64}` — the central finite difference of `value` at interior
  points (`NaN` at the ends);
- `fd_error::Vector{Float64}` — relative disagreement between `lambda` and `fd`
  at SMOOTH interior points;
- `fd_ok::Bool` — every smooth interior point agreed within `fd_tolerance`;
- `nonsmooth::Vector{Bool}` — the one-sided slopes disagree by more than
  `kink_tolerance`, so the value curve has a kink at this point;
- `bracketed::Vector{Bool}` — at a nonsmooth point, whether the reported
  multiplier lies between the two one-sided slopes, which is the correct
  statement for a subgradient of a convex value function;
- `endpoint::Vector{Bool}` — the grid point sits at an endpoint of the reachable
  interval, where the multiplier is a subgradient and finite differences bracket
  rather than match it;
- `solved`, `status`, `residuals`, `worst_recourse`, `binding`, `prices` per
  point.

# Notes
This is the principal tool for locating buses where the SOC-WR relaxation
misprices stored energy. The interesting quantity is not the objective GAP
between the two formulations — a relaxation is below the true model everywhere
and that says nothing about decisions — but the difference in
``\\partial Q/\\partial e^{out}``: the price the two models put on carrying one
more unit of energy out of this stage. A bus where the two models disagree about
that price is a bus where a cut built on the relaxation will steer storage
differently from the truth.

**Endpoints are not evidence.** At an endpoint of the reachable interval the
target sits on a bound, the multiplier is a subgradient, and two solvers may
report two valid values that differ by orders of magnitude. Interior points are
where a multiplier comparison means something, which is why they are the ones
finite differences are checked against.
"""
function energy_value_curve(case::BatteryCase, model_type::Type;
                            battery::Integer,
                            energy_in::AbstractDict,
                            stage::Integer,
                            atom::Integer,
                            grid::AbstractVector{<:Real},
                            hold::Union{Nothing,AbstractDict} = nothing,
                            optimizer = nothing,
                            fd_tolerance::Real = 5e-3,
                            kink_tolerance::Real = 5e-2)
    b = only(filter(x -> x.index == Int(battery), case.batteries))
    lo, hi = reachable_interval(b, Float64(energy_in[b.index]), stage_hours(case))
    n = length(grid)

    energy = Float64.(collect(grid))
    value = fill(NaN, n)
    lambda = fill(NaN, n)
    solved = falses(n)
    status = Vector{Any}(undef, n)
    reachable = falses(n)
    endpoint = falses(n)
    recourse_used = fill(NaN, n)
    residuals = Vector{Any}(undef, n)
    binding = Vector{Any}(undef, n)
    prices = Vector{Any}(undef, n)

    for (i, e) in enumerate(energy)
        # A tolerance on reachability, not on feasibility: the interval endpoints
        # are computed in floating point and a grid built from them would
        # otherwise fall a rounding error outside itself.
        reachable[i] = lo - 1e-9 <= e <= hi + 1e-9
        endpoint[i] = reachable[i] && (abs(e - lo) <= 1e-9 || abs(e - hi) <= 1e-9)
        reachable[i] || (status[i] = :unreachable; residuals[i] = nothing;
                         binding[i] = NamedTuple[]; prices[i] = nothing; continue)

        pins = Dict{Int,Float64}(b.index => clamp(e, lo, hi))
        hold === nothing || for (k, v) in hold
            Int(k) == b.index && continue
            pins[Int(k)] = Float64(v)
        end
        p = targetless_probe(case, model_type; energy_in = energy_in, stage = stage,
                             atom = atom, optimizer = optimizer, pins = pins)
        status[i] = p.status
        solved[i] = p.solved
        residuals[i] = p.residuals
        binding[i] = p.binding
        prices[i] = (active = p.price_active, reactive = p.price_reactive)
        if p.solved
            value[i] = p.cost_stage
            lambda[i] = get(p.pin_dual, b.index, NaN)
            recourse_used[i] = worst_recourse(p)
        end
    end

    # ── Finite differences at interior grid points ───────────────────────────
    # The stage value is convex and PIECEWISE smooth in the outgoing energy: a
    # binding generator, branch or voltage limit puts a kink in it. A central
    # difference taken ACROSS a kink is not an approximation of anything, so the
    # check first classifies each interior point by comparing its one-sided
    # slopes and only compares λ against a central difference where the curve is
    # locally smooth. At a kink the correct statement about a subgradient is that
    # it lies BETWEEN the one-sided slopes, and that is what is recorded.
    fd = fill(NaN, n)
    fd_error = fill(NaN, n)
    nonsmooth = falses(n)
    bracketed = falses(n)
    ok = true
    for i in 2:(n - 1)
        (solved[i - 1] && solved[i] && solved[i + 1]) || continue
        endpoint[i] && continue
        h⁻ = energy[i] - energy[i - 1]
        h⁺ = energy[i + 1] - energy[i]
        slope⁻ = (value[i] - value[i - 1]) / h⁻
        slope⁺ = (value[i + 1] - value[i]) / h⁺
        if abs(slope⁺ - slope⁻) / max(abs(slope⁻), abs(slope⁺), 1e-8) > kink_tolerance
            nonsmooth[i] = true
            bracketed[i] = min(slope⁻, slope⁺) - 1e-6 <= lambda[i] <= max(slope⁻, slope⁺) + 1e-6
            continue
        end
        # Non-uniform central difference: exact for a quadratic, which is what
        # makes the check meaningful on a grid that is denser near an endpoint.
        fd[i] = (h⁻^2 * value[i + 1] - h⁺^2 * value[i - 1] -
                 (h⁻^2 - h⁺^2) * value[i]) / (h⁻ * h⁺ * (h⁻ + h⁺))
        scale = max(abs(fd[i]), abs(lambda[i]), 1e-8)
        fd_error[i] = abs(fd[i] - lambda[i]) / scale
        fd_error[i] <= fd_tolerance || (ok = false)
    end

    return (battery = b.index, bus = b.bus, stage = Int(stage), atom = Int(atom),
            model_type = model_type,
            reachable_lo = lo, reachable_hi = hi,
            energy = energy, value = value, lambda = lambda,
            fd = fd, fd_error = fd_error, fd_ok = ok,
            nonsmooth = nonsmooth, bracketed = bracketed,
            reachable = reachable, endpoint = endpoint,
            solved = solved, status = status,
            worst_recourse = recourse_used, residuals = residuals,
            binding = binding, prices = prices)
end

"""
    compare_value_curves(case; battery, energy_in, stage, atom, grid, kwargs...)
        -> NamedTuple

Run [`energy_value_curve`](@ref) under BOTH formulations and difference their
marginal stored-energy values.

# Returns
`(acp, soc, energy, dlambda, dlambda_interior, max_abs_dlambda,
  max_rel_dlambda, reversal)` where `dlambda = λ_SOC − λ_ACP`, restricted to grid
points both formulations solved.

# Notes
`reversal` is `true` when the two formulations disagree about the SIGN of the
marginal value at some interior point — the strongest form of mispricing, because
it means the relaxation would store energy where the true model would spend it.
A magnitude difference changes how much a policy hedges; a sign difference
changes what it does.

`dlambda_interior` excludes two kinds of grid point, and the exclusions are the
difference between evidence and an artefact:

- **reachable-interval endpoints**, where a multiplier is a subgradient and a
  difference between two valid subgradients is not evidence of anything;
- **points that used physical recourse**, above `max_recourse` in either
  formulation. There, part of the marginal value is the recourse PRICE — a number
  chosen to be far above any generator — rather than the network's valuation of
  stored energy, and it produces enormous, meaningless disagreements. A candidate
  whose apparent mechanism lives only at such points is exactly what this phase's
  review conditions say to reject.
"""
function compare_value_curves(case::BatteryCase;
                              battery::Integer,
                              energy_in::AbstractDict,
                              stage::Integer,
                              atom::Integer,
                              grid::AbstractVector{<:Real},
                              max_recourse::Real = 1e-6,
                              kwargs...)
    acp = energy_value_curve(case, PowerModels.ACPPowerModel;
                             battery = battery, energy_in = energy_in, stage = stage,
                             atom = atom, grid = grid, kwargs...)
    soc = energy_value_curve(case, PowerModels.SOCWRConicPowerModel;
                             battery = battery, energy_in = energy_in, stage = stage,
                             atom = atom, grid = grid, kwargs...)
    n = length(acp.energy)
    dλ = fill(NaN, n)
    for i in 1:n
        (acp.solved[i] && soc.solved[i]) || continue
        dλ[i] = soc.lambda[i] - acp.lambda[i]
    end
    clean = [i for i in 1:n if acp.solved[i] && soc.solved[i] &&
                 acp.worst_recourse[i] <= max_recourse &&
                 soc.worst_recourse[i] <= max_recourse]
    interior = [i for i in clean if !isnan(dλ[i]) && !acp.endpoint[i] && !soc.endpoint[i]]
    reversal = any(i -> sign(acp.lambda[i]) != sign(soc.lambda[i]) &&
                        abs(acp.lambda[i]) > 1e-6 && abs(soc.lambda[i]) > 1e-6,
                   interior)
    rel = [abs(dλ[i]) / max(abs(acp.lambda[i]), 1e-8) for i in interior]
    return (acp = acp, soc = soc, energy = acp.energy, dlambda = dλ,
            dlambda_interior = dλ[interior],
            max_abs_dlambda = isempty(interior) ? NaN : maximum(abs, dλ[interior]),
            max_rel_dlambda = isempty(rel) ? NaN : maximum(rel),
            reversal = reversal, interior = interior, clean = clean,
            num_recourse_excluded = count(i -> acp.solved[i] && soc.solved[i], 1:n) -
                                    length(clean))
end

# ─────────────────────────────────────────────────────────────────────────────
# B4 — the multiperiod deterministic equivalent
# ─────────────────────────────────────────────────────────────────────────────

"""
    deterministic_equivalent(case, atoms; model_type=ACPPowerModel,
                             optimizer=nothing, energy_initial=nothing,
                             silent=true, time_limit=nothing) -> NamedTuple

Solve the WHOLE horizon jointly for one complete demand path: the
perfect-foresight (wait-and-see) solution.

# Arguments
- `atoms::AbstractVector{<:Integer}`: the atom index realized at each stage. Its
  length is the horizon solved.

# Keywords
- `model_type::Type`: `ACPPowerModel` — the physical reference — or
  `SOCWRConicPowerModel`, which is a RELAXED perfect-foresight solve and is not
  a physically attainable cost.
- `energy_initial`: `nothing` for the case's own ``e_{b,0}``, or a dictionary.

# Returns
A `NamedTuple` with the solver status, the total and per-stage costs, the full
stagewise solution in the shared schema, the terminal energy, the throughput, the
simultaneous-charge audit, the recourse totals and the physical residuals.

# Notes
There is no policy here, no target, no target slack and no future-cost
approximation: every charge, discharge and outgoing energy over the whole horizon
is chosen at once, knowing the entire demand path. The battery dynamics, the
bounds, the two-sided physical recourse and the cost accounting are the study's
own, because the model is assembled from the same per-stage builder the study
trains on — one `pm` per stage, all in ONE JuMP model, coupled only by
``e^{in}_{b,t+1} = e^{out}_{b,t}``.

That coupling is the only thing this function writes. In particular it does not
write a network equation, and `assert_powermodels_provenance` is run on every
stage.

**What the resulting number is.** For a single path it is the cost a clairvoyant
operator would have paid. Averaged over a protocol of paths it is a WAIT-AND-SEE
LOWER BOUND on the nonanticipative stochastic problem — see
[`perfect_foresight_panel`](@ref) for what may and may not be concluded from the
gap to a policy.

Each stage's build replaces the JuMP objective (PowerModels' own
`objective_min_fuel_and_flow_cost` does), so the per-stage objective is captured
immediately after each build and the horizon objective is set from their sum at
the end. Reading the objective any later would return the last stage's cost.
"""
function deterministic_equivalent(case::BatteryCase,
                                  atoms::AbstractVector{<:Integer};
                                  model_type::Type = PowerModels.ACPPowerModel,
                                  optimizer = nothing,
                                  energy_initial = nothing,
                                  silent::Bool = true,
                                  time_limit = nothing)
    T = length(atoms)
    T >= 1 || throw(ArgumentError("a deterministic equivalent needs at least one stage"))
    T <= horizon(case.demand) ||
        throw(ArgumentError("path has $T stages but the frozen support covers $(horizon(case.demand))"))
    opt = optimizer === nothing ? _default_optimizer(model_type) : optimizer
    e0 = energy_initial === nothing ?
         Dict{Int,Float64}(b.index => b.energy_initial for b in case.batteries) :
         Dict{Int,Float64}(Int(k) => Float64(v) for (k, v) in energy_initial)

    model = JuMP.Model(opt)
    silent && JuMP.set_silent(model)
    time_limit === nothing || JuMP.set_time_limit_sec(model, Float64(time_limit))

    pms = Vector{Any}(undef, T)
    stage_obj = Vector{Any}(undef, T)
    for t in 1:T
        pms[t] = battery_stage_model(case, model_type;
                                     jump_model = model,
                                     stage = t, atom = atoms[t], mode = :free,
                                     state_in = t == 1 ? e0 : nothing)
        assert_powermodels_provenance(pms[t], model_type)
        # Captured NOW: the next stage's build will overwrite the model objective.
        stage_obj[t] = JuMP.objective_function(model)
    end

    # The one thing this function writes: the interstage state coupling.
    link = Dict{Tuple{Int,Int},Any}()
    for t in 2:T, b in case.batteries
        link[(t, b.index)] = JuMP.@constraint(model,
            pms[t].ext[:battery][:e_in][b.index] ==
            pms[t - 1].ext[:battery][:e_out][b.index])
    end

    JuMP.@objective(model, Min, sum(stage_obj))
    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    ok = status in ACCEPTED_STATUSES

    stages = [extract_stage_solution(pms[t]) for t in 1:T]
    residuals = ok ? [physical_residuals(case.network, case.batteries, stage_hours(case), s)
                      for s in stages] : nothing
    cost = ok ? [s.cost_stage for s in stages] : fill(NaN, T)
    total = ok ? sum(cost) : NaN
    Δt = stage_hours(case)

    throughput = Dict{Int,Float64}(b.index =>
        sum(Δt * (stages[t].p_ch[b.index] + stages[t].p_dis[b.index]) for t in 1:T)
        for b in case.batteries)
    simultaneous = ok ? maximum(min(stages[t].p_ch[b.index], stages[t].p_dis[b.index])
                                for t in 1:T, b in case.batteries) : NaN
    worst_deficit = ok ? maximum(max(0.0, maximum(values(stages[t].deficit); init = 0.0)) for t in 1:T) : NaN
    worst_surplus = ok ? maximum(max(0.0, maximum(values(stages[t].surplus); init = 0.0)) for t in 1:T) : NaN

    return (status = status, solved = ok, total_cost = total,
            stage_cost = cost, cumulative_cost = ok ? cumsum(cost) : fill(NaN, T),
            stages = stages, residuals = residuals,
            energy_initial = e0,
            energy_terminal = Dict{Int,Float64}(b.index => stages[T].energy_out[b.index]
                                                for b in case.batteries),
            throughput = throughput, simultaneous = simultaneous,
            worst_deficit = worst_deficit, worst_surplus = worst_surplus,
            cost_deficit = ok ? sum(s.cost_deficit for s in stages) : NaN,
            cost_surplus = ok ? sum(s.cost_surplus for s in stages) : NaN,
            atoms = collect(Int.(atoms)), horizon = T, model_type = model_type,
            link = link, model = model)
end

"""
    record_trajectory!(rec, result, case; scenario)

Write every stage of a [`deterministic_equivalent`](@ref) result into a
[`SolutionRecorder`](@ref) in the shared schema.
"""
function record_trajectory!(rec::SolutionRecorder, result, case::BatteryCase;
                            scenario::Integer)
    for t in 1:result.horizon
        record_stage_solution!(rec, result.stages[t], case; scenario = scenario, stage = t)
        result.residuals === nothing && continue
        r = result.residuals[t]
        record!(rec, scenario, t, "residual_equality", 0,
                max(r.branch_flow, r.active_balance, r.reactive_balance, r.transition))
    end
    return rec
end

# ─────────────────────────────────────────────────────────────────────────────
# B5 — the perfect-foresight panel
# ─────────────────────────────────────────────────────────────────────────────

"""
    perfect_foresight_panel(case, protocol; ids=nothing, model_type=ACPPowerModel,
                            optimizer=nothing, retry=true, verbose=false)
        -> NamedTuple

Solve one true-ACP deterministic equivalent per GLOBAL scenario identifier and
summarize the panel.

# Arguments
- `protocol::AbstractMatrix{<:Integer}`: the `(stages × scenarios)` atom-index
  matrix, from [`scenario_index_matrix`](@ref).

# Keywords
- `ids`: the global scenario identifiers to solve, defaulting to every column.
  A column's identifier is its INDEX IN THE PROTOCOL and never changes.
- `retry::Bool`: on a first-attempt failure, re-solve once with a completely
  fresh model and solver.

# Returns
A `NamedTuple` with one row per requested identifier — cost, status, first-attempt
status, recourse, terminal energy, throughput — plus the panel mean, standard
deviation and standard error, and the full per-scenario results.

# Notes
**Fail closed.** Every requested identifier is retained: none is replaced,
dropped or renumbered, and if any is unsolved after the retry the panel is marked
incomplete. A mean over the scenarios that happened to succeed is a mean over a
different problem, and no tolerance makes it comparable to a mean over all of
them. The first-attempt status is recorded separately from the retry status,
because "solved on the second try" is information about the case.

**What the mean is.** The true-ACP perfect-foresight mean is a WAIT-AND-SEE LOWER
BOUND on the nonanticipative stochastic problem: a clairvoyant operator cannot be
beaten by one who must decide before seeing the future. The gap between a
policy's cost and this bound is therefore diagnostic HEADROOM, and it contains
the value of future information — which no nonanticipative policy, TS-DDR or
SDDP, can recover. It is not a target, it is not attainable, and a policy that
closes half of it has not necessarily left anything on the table.

The bound may be computed during screening, on the screening protocol. Its
relationship to a trained policy is assessed only on PAIRED paths, after that
policy exists.
"""
function perfect_foresight_panel(case::BatteryCase, protocol::AbstractMatrix{<:Integer};
                                 ids = nothing,
                                 model_type::Type = PowerModels.ACPPowerModel,
                                 optimizer = nothing,
                                 retry::Bool = true,
                                 verbose::Bool = false)
    columns = ids === nothing ? collect(1:size(protocol, 2)) : collect(Int.(ids))
    all(c -> 1 <= c <= size(protocol, 2), columns) ||
        throw(ArgumentError("scenario identifier outside the protocol's 1:$(size(protocol, 2))"))

    rows = NamedTuple[]
    results = Dict{Int,Any}()
    for c in columns
        path = protocol[:, c]
        res = deterministic_equivalent(case, path; model_type = model_type,
                                       optimizer = optimizer)
        first_status = res.status
        if !res.solved && retry
            # A completely fresh model and solver, not a re-solve: a re-solve
            # from a failed interior point is a different algorithm, not a
            # second attempt at the same one.
            res = deterministic_equivalent(case, path; model_type = model_type,
                                           optimizer = optimizer)
        end
        results[c] = res
        push!(rows, (scenario = c, solved = res.solved,
                     first_status = first_status, status = res.status,
                     cost = res.total_cost,
                     worst_deficit = res.worst_deficit, worst_surplus = res.worst_surplus,
                     simultaneous = res.simultaneous,
                     terminal_energy = sum(values(res.energy_terminal)),
                     throughput = sum(values(res.throughput))))
        verbose && @printf("  scenario %4d  %-16s  cost %14.4f\n", c, string(res.status),
                           res.total_cost)
    end

    complete = all(r -> r.solved, rows)
    costs = [r.cost for r in rows if r.solved]
    n = length(costs)
    return (rows = rows, results = results, complete = complete,
            num_requested = length(columns), num_solved = n,
            mean = n == 0 ? NaN : mean(costs),
            std = n < 2 ? NaN : std(costs),
            sem = n < 2 ? NaN : std(costs) / sqrt(n),
            worst_deficit = maximum(r -> r.worst_deficit, rows; init = 0.0),
            worst_surplus = maximum(r -> r.worst_surplus, rows; init = 0.0),
            model_type = model_type)
end

"""
    paired_difference(a::AbstractVector, b::AbstractVector) -> NamedTuple

Paired mean difference `a − b` with its standard error, `t` statistic and 95 %
confidence interval.

# Notes
Pairing is what makes a comparison of two policies on a common demand protocol
decidable: the spread of cost ACROSS scenarios is typically an order of magnitude
larger than the difference BETWEEN policies on the same scenario, so an unpaired
comparison of the same sample size would resolve nothing.

The interval uses the normal quantile 1.96 rather than a `t` quantile; at the
sample sizes this study reports on (hundreds of paired paths) the difference is
in the third decimal of the interval and is not worth a distributions dependency
in a file every consumer loads.
"""
function paired_difference(a::AbstractVector, b::AbstractVector)
    length(a) == length(b) || throw(ArgumentError("paired vectors must have equal length"))
    d = Float64.(a) .- Float64.(b)
    n = length(d)
    n >= 2 || throw(ArgumentError("a paired difference needs at least two pairs"))
    m = mean(d)
    se = std(d) / sqrt(n)
    return (n = n, mean = m, std = std(d), sem = se,
            t = se == 0 ? NaN : m / se,
            ci = (m - 1.96 * se, m + 1.96 * se),
            wins = count(<(0), d))
end


# ─────────────────────────────────────────────────────────────────────────────
# The deterministic-forecast reference policy
# ─────────────────────────────────────────────────────────────────────────────

"""
    central_atom_path(case, T) -> Vector{Int}

The atom whose total multiplier is closest to the stage's PROBABILITY-WEIGHTED
mean, for each stage — i.e. the single scenario a forecaster would use.
"""
function central_atom_path(case::BatteryCase, T::Integer)
    return [begin
                p = atom_probabilities(case.demand, t)
                μ = sum(p[k] * mean(demand_multipliers(case.demand, t, k))
                        for k in eachindex(p))
                argmin([abs(mean(demand_multipliers(case.demand, t, k)) - μ)
                        for k in eachindex(p)])
            end for t in 1:Int(T)]
end

"""
    forecast_policy_cost(case, protocol; columns, optimizer=nothing) -> NamedTuple

The cost of the DETERMINISTIC-FORECAST policy: optimise once against a single
forecast path, then operate those storage targets on every realized path.

# How it is built
1. Solve one true-ACP deterministic equivalent against
   [`central_atom_path`](@ref) — the forecaster's single scenario.
2. Take its outgoing battery energies as a fixed target schedule.
3. On each evaluation path, walk forward imposing those targets as STRICT
   equalities, clamping each into the one-stage reachable interval of the state
   actually reached. Clamping is not a fudge: a real forecast-based operator
   cannot charge past its rating either, and the reachable map is the same one
   the strict policy uses.

# Why it is reported
`VSS = forecast_cost − sddp_cost` is the **value of the stochastic solution**: how
much is lost by ignoring uncertainty and operating a single forecast. It is the
direct test of whether a case's uncertainty structure requires a policy at all.
If VSS is near zero, every method — SDDP, TS-DDR, a spreadsheet — costs the same,
and no comparison between them can resolve anything, however large the other
gaps look.
"""
function forecast_policy_cost(case::BatteryCase, protocol::AbstractMatrix{<:Integer};
                              columns, optimizer = nothing)
    T = size(protocol, 1)
    opt = optimizer === nothing ? acp_optimizer() : optimizer
    plan = deterministic_equivalent(case, central_atom_path(case, T))
    plan.solved || error("the forecast deterministic equivalent did not solve")
    schedule = [Dict{Int,Float64}(b.index => plan.stages[t].energy_out[b.index]
                                  for b in case.batteries) for t in 1:T]

    Δt = stage_hours(case)
    costs = Float64[]
    worst_rec = 0.0
    for c in collect(Int.(columns))
        e = Dict{Int,Float64}(b.index => b.energy_initial for b in case.batteries)
        total = 0.0
        for t in 1:T
            tgt = Dict{Int,Float64}()
            for b in case.batteries
                lo, hi = reachable_interval(b, e[b.index], Δt)
                tgt[b.index] = clamp(schedule[t][b.index], lo, hi)
            end
            sol = solve_strict_stage(case, PowerModels.ACPPowerModel; stage = t,
                                     atom = protocol[t, c], energy_in = e,
                                     target = tgt, optimizer = opt)
            sol.solved || error("forecast policy: stage $t of scenario $c did not solve")
            total += sol.cost_stage
            worst_rec = max(worst_rec, worst_recourse(sol))
            e = tgt
        end
        push!(costs, total)
    end
    return (costs = costs, mean = mean(costs), max = maximum(costs),
            worst_recourse = worst_rec, plan = plan)
end
