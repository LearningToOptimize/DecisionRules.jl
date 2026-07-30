# sddp_ac_starts.jl — robust primal starts for ACP forward-pass subproblems.
#
# HydroPowerModels' `rainfall_noises` parameterize block fills a primal start of
# `sp.ext[:lower_bound] = 0.0` into every variable whose start is `nothing`
# (see HydroPowerModels/src/constraint.jl). For the polar-AC forward graph this
# includes the voltage magnitudes, and ``v_m = 0`` is a singular point of the
# polar power-flow equations (every ``v_i v_j \cos(\theta_i-\theta_j)`` term and
# its Jacobian vanish), so MadNLP can fail from that start at stressed
# operating points even though the subproblem is feasible — the identical
# subproblem dumped to MOF (which carries no starts) solves in milliseconds.
# Observed as `Unable to retrieve solution from node 22` on the seasonal-demand
# case (2026-07-15).
#
# The fix: preset every start BEFORE training so the package's fill-in loop
# never runs — flat-voltage ``v_m = 1`` for magnitudes, ``0`` for everything
# else (angles, flows, generation, deficit).

"""
    preset_ac_starts!(graph::SDDP.PolicyGraph)

Set primal start values on every subproblem of `graph`: variables whose name
contains `"_vm"` (polar-AC voltage magnitudes, PowerModels naming `0_vm[...]`)
start at the flat-voltage point ``v_m = 1``; every other variable with no
start yet gets ``0``. Idempotent; call once after `hydro_thermal_operation`
and before `HydroPowerModels.train`.

# Arguments

- `graph::SDDP.PolicyGraph`: the (forward) policy graph whose stage
  subproblems are polar-AC OPF models.

# Example

```julia
m = hydro_thermal_operation(alldata, params)
preset_ac_starts!(m.forward_graph)   # forward pass = ACPPowerModel/MadNLP
HydroPowerModels.train(m; iteration_limit = 100)
```
"""
function preset_ac_starts!(graph)
    # count how many voltage-magnitude variables were preset (sanity print)
    n_vm = 0
    # every stage node holds one JuMP subproblem
    for (_, node) in graph.nodes
        sp = node.subproblem
        # walk all variables of the stage subproblem once
        for v in JuMP.all_variables(sp)
            nm = JuMP.name(v)
            if occursin("_vm", nm)
                # flat-voltage start: the standard non-singular AC initial point
                JuMP.set_start_value(v, 1.0)
                n_vm += 1
            elseif JuMP.start_value(v) === nothing
                # preserve the package's historical default for the rest
                JuMP.set_start_value(v, 0.0)
            end
        end
    end
    println("preset_ac_starts!: $n_vm voltage-magnitude starts set to 1.0")
    return nothing
end

"""
    make_robust_recovery(ac_optimizer::Function, conic_optimizer::Function)

Return a `numerical_difficulty_callback` for SDDP (install the result with
`graph.ext[:numerical_difficulty_callback] = cb` on BOTH policy graphs).

SDDP's default recovery makes a single `MOI.Utilities.reset_optimizer` +
re-solve attempt. On the stressed seasonal-demand case two failure modes
survive it: (a) intermittent MadNLP non-convergence from bad iterates at
near-peak load, and (b) a swallowed solver exception that leaves the caching
layer optimizer-less (`OPTIMIZE_NOT_CALLED` / `NoOptimizer`). The returned
callback escalates through:

1. plain `reset_optimizer` + re-solve (the SDDP default);
2. **re-attach a brand-new solver instance** (`ac_optimizer()` for polar-AC
   subproblems — detected by `_vm` variables — `conic_optimizer()` otherwise)
   and restart from the flat-voltage point (``v_m = 1``, all else ``0``);
3. for CONIC subproblems only, one further rung per factory in
   `conic_variants`: a brand-new solver that differs **only** in its static
   regularization, at unchanged feasibility/gap tolerances.

Rung 3 exists because rungs 1 and 2 are numerically IDENTICAL — same tolerances,
same regularization — so a solve that fails on a knife-edge reproduces the same
failure on both. Diagnosis of the v67damp node-122 `ALMOST_OPTIMAL` abort
(PROJECT.md §19.12/R1) showed the failure is a narrow, instance-specific
resonance of the cut-augmented KKT system: on that exact instance
`static_regularization_constant` values of 1e-11, 3e-11, 3e-10 and 1e-9 all
terminate literal `OPTIMAL` while 1e-10 alone returns `ALMOST_OPTIMAL`, and the
objectives agree to ~1e-7 relative. Because the resonance is instance-specific,
no single global regularization is durable; retrying the SAME solve at a
different regularization is.

**This is not a tolerance relaxation.** `tol_feas`, `tol_gap_abs` and
`tol_gap_rel` are unchanged across every rung, `StrictConicDuality` still
requires literal `OPTIMAL` before a cut is read, and a rung is accepted only if
it genuinely reaches `OPTIMAL`. An earlier revision of this docstring described a
rung that loosened the tolerance to ``10^{-4}``; that rung does not exist and
must not be reintroduced — loose duals invalidate the SDDP lower bound.

Every mutation is `try/catch`-guarded so one broken attempt never masks the
next; the callback itself never throws — if all attempts fail, SDDP proceeds
to its own diagnostics dump.

# Arguments
- `ac_optimizer::Function`: factory for polar-AC (forward) subproblems.
- `conic_optimizer::Function`: factory for conic (backward) subproblems.
- `conic_variants::Vector{<:Function}`: optional extra conic factories, tried in
  order after rung 2, differing only in regularization.
"""
function make_robust_recovery(ac_optimizer::Function, conic_optimizer::Function;
                              conic_variants::Vector{<:Function} = Function[])
    return function (model, node; require_dual::Bool = false)
        sp = node.subproblem
        # polar-AC subproblems carry voltage-magnitude variables named `*_vm*`
        is_ac = any(v -> occursin("_vm", JuMP.name(v)), JuMP.all_variables(sp))
        # a solve "worked" when SDDP can read a primal (and, if required, dual).
        # For CUT solves the dual must come from a solve that terminated
        # OPTIMAL — SDDP's own predicate accepts NEARLY_FEASIBLE_POINT duals,
        # which is exactly the invalid-cut hazard we are excluding.
        solved() = SDDP._has_primal_solution(node) &&
                   !(require_dual && !(SDDP._has_dual_solution(node) &&
                       JuMP.termination_status(sp) == JuMP.OPTIMAL))
        # No TOLERANCE ladder is permitted: loose cut duals invalidate the SDDP
        # lower bound and loose forward solves contaminate the states on which
        # cuts are sampled. Regularization variants (rung 3+) are a different
        # thing — they change only the KKT regularization, leaving every
        # feasibility/gap tolerance at its strict value, and are accepted only on
        # a literal OPTIMAL termination. See the docstring for the node-122
        # evidence motivating them.
        # AC voltage-magnitude start seeds for the multi-start rungs. Polar-AC is
        # nonconvex, so MadNLP can converge to a LOCALLY-infeasible point from one
        # flat start even when the subproblem is feasible (the wait-and-see solve
        # of the identical horizon confirms feasibility). Retrying from a DIFFERENT
        # voltage level escapes that basin. This changes only the primal START, not
        # any tolerance — a rung is still accepted only on OPTIMAL/LOCALLY_SOLVED
        # with a primal, so it cannot invalidate anything.
        ac_vm_seeds = [1.0, 1.05, 0.95, 1.10, 0.90, 1.03]
        # AC: attempt 1 = reset, attempts 2..(1+nseeds) = fresh solver at each seed.
        # Conic: attempt 1 = reset, 2 = fresh solver, 3.. = regularization variants.
        n_attempts = is_ac ? (1 + length(ac_vm_seeds)) : (2 + length(conic_variants))
        for attempt in 1:n_attempts
            try
                if attempt == 1
                    # cheapest first: fresh copy of the cached problem data
                    MOI.Utilities.reset_optimizer(sp)
                else
                    # hard reset: brand-new solver object (cures NoOptimizer /
                    # poisoned internal state that reset_optimizer keeps).
                    # attempt >= 3 additionally swaps in a regularization variant
                    # (conic) or a fresh voltage seed (AC) so the retry is not
                    # numerically identical to rung 2.
                    # NOTE: a variant attached here PERSISTS on this subproblem
                    # for its later solves (JuMP.set_optimizer is sticky, and
                    # restoring the base factory would discard the solution we
                    # just recovered). That is path-dependent but safe: every
                    # rung uses the same strict tolerances and a cut is still
                    # only built from a literal OPTIMAL termination.
                    factory = if is_ac
                        ac_optimizer                       # AC: same solver, vary the start
                    elseif attempt == 2
                        conic_optimizer
                    else
                        conic_variants[attempt-2]
                    end
                    JuMP.set_optimizer(sp, factory)
                    # AC multi-start: rung 2 uses vm=1 (flat), later rungs sweep
                    # ac_vm_seeds. Conic: flat restart (vm term absent anyway).
                    vm_start = is_ac ? ac_vm_seeds[attempt-1] : 1.0
                    for v in JuMP.all_variables(sp)
                        JuMP.set_start_value(v, occursin("_vm", JuMP.name(v)) ? vm_start : 0.0)
                    end
                end
                JuMP.optimize!(sp)
            catch err
                # loud escalation — silent failures cost debugging rounds
                println("[recovery] node=", node.index, " rung=", attempt,
                        " THREW: ", sprint(showerror, err)[1:min(end, 200)])
            end
            try
                st = JuMP.termination_status(sp)
                println("[recovery] node=", node.index, " rung=", attempt,
                        " status=", st, " primal=", JuMP.primal_status(sp),
                        " dual=", JuMP.dual_status(sp))
                acceptable = is_ac ?
                    (st in (JuMP.OPTIMAL, JuMP.LOCALLY_SOLVED) && SDDP._has_primal_solution(node)) :
                    (st == JuMP.OPTIMAL && solved())
                acceptable && return
            catch
            end
        end
        error("strict numerical recovery failed at node $(node.index); aborting to preserve bound integrity")
    end
end

# ── Fail-closed backward-cut duality handler ────────────────────────────────────
#
# THE HAZARD. SDDP's default `ContinuousConicDuality` reads cut duals whenever
# `SDDP._has_dual_solution(node)` is true, and that predicate accepts BOTH
# `FEASIBLE_POINT` and `NEARLY_FEASIBLE_POINT` (algorithm.jl). A backward
# cut-generating solve that terminates `ALMOST_OPTIMAL` / `ALMOST_SOLVED`
# (Clarabel) reports `dual_status == NEARLY_FEASIBLE_POINT`, so SDDP silently
# builds a cut from the loose dual and NEVER invokes the numerical-difficulty
# recovery callback. Iteration-limit, infeasible, or numerically-failed solves
# with a stale near-feasible dual are the same hazard. A cut from a non-OPTIMAL
# dual is not a valid outer approximation and corrupts the SDDP lower bound.
#
# `StrictConicDuality` closes this: before any cut dual is read it requires the
# backward subproblem to have terminated **literal `OPTIMAL`**. If not, it runs
# the robust recovery ladder (fresh solver, flat-voltage restart — the same one
# installed as the numerical-difficulty callback, which itself requires OPTIMAL),
# then re-checks. If the solve still is not OPTIMAL it raises an explicit
# diagnostic and aborts training instead of emitting an invalid cut.
"""
    StrictConicDuality(optimizer = nothing) <: SDDP.AbstractDualityHandler

Drop-in replacement for `SDDP.ContinuousConicDuality` that refuses to generate a
cut from a backward solve whose termination status is not literal `MOI.OPTIMAL`
(rejecting `ALMOST_OPTIMAL`, iteration-limit, infeasible, and numerically-failed
solves whose near-feasible dual SDDP would otherwise silently accept). Pass it as
`duality_handler = StrictConicDuality()` to `SDDP.train` / `HydroPowerModels.train`.
Delegates `prepare_backward_pass`, `duality_log_key`, and the actual dual read to
an inner `ContinuousConicDuality`.
"""
struct StrictConicDuality <: SDDP.AbstractDualityHandler
    inner::SDDP.ContinuousConicDuality
end
StrictConicDuality(optimizer = nothing) =
    StrictConicDuality(SDDP.ContinuousConicDuality(optimizer))

SDDP.prepare_backward_pass(node::SDDP.Node, h::StrictConicDuality, options::SDDP.Options) =
    SDDP.prepare_backward_pass(node, h.inner, options)

SDDP.duality_log_key(::StrictConicDuality) = "!"  # marks strict-cut mode in the log

function SDDP.get_dual_solution(node::SDDP.Node, h::StrictConicDuality)
    sp = node.subproblem
    st = JuMP.termination_status(sp)
    if st != JuMP.OPTIMAL
        # Non-OPTIMAL cut-generating solve: force the recovery ladder (re-solve to
        # literal OPTIMAL) rather than let SDDP accept a NEARLY_FEASIBLE_POINT dual.
        model = sp.ext[:sddp_policy_graph]
        SDDP.attempt_numerical_recovery(model, node; require_dual = true)
        st = JuMP.termination_status(sp)
    end
    if st != JuMP.OPTIMAL
        error(
            "Fail-closed backward cut: node $(node.index) solve terminated $(st) " *
            "(primal=$(JuMP.primal_status(sp)), dual=$(JuMP.dual_status(sp))). " *
            "Refusing to build a cut from a non-OPTIMAL solve — this would " *
            "invalidate the SDDP lower bound. Investigate solver tolerances / " *
            "conditioning at this node instead of continuing.",
        )
    end
    return SDDP.get_dual_solution(node, h.inner)
end
