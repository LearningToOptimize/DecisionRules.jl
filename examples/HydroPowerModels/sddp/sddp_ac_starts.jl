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
3. same as 2 but with the solver tolerance loosened to ``10^{-4}`` for this
   solve (fresh instance per solve, so later solves keep the tight default).

Every mutation is `try/catch`-guarded so one broken attempt never masks the
next; the callback itself never throws — if all attempts fail, SDDP proceeds
to its own diagnostics dump.
"""
function make_robust_recovery(ac_optimizer::Function, conic_optimizer::Function)
    return function (model, node; require_dual::Bool = false)
        sp = node.subproblem
        # polar-AC subproblems carry voltage-magnitude variables named `*_vm*`
        is_ac = any(v -> occursin("_vm", JuMP.name(v)), JuMP.all_variables(sp))
        # a solve "worked" when SDDP can read a primal (and, if required, dual)
        solved() = SDDP._has_primal_solution(node) &&
                   !(require_dual && !SDDP._has_dual_solution(node))
        for attempt in 1:3
            try
                if attempt == 1
                    # cheapest first: fresh copy of the cached problem data
                    MOI.Utilities.reset_optimizer(sp)
                else
                    # hard reset: brand-new solver object (cures NoOptimizer /
                    # poisoned internal state that reset_optimizer keeps)
                    factory = is_ac ? ac_optimizer : conic_optimizer
                    JuMP.set_optimizer(sp, factory)
                    # flat-voltage restart: vm = 1 is the non-singular AC point
                    for v in JuMP.all_variables(sp)
                        JuMP.set_start_value(v, occursin("_vm", JuMP.name(v)) ? 1.0 : 0.0)
                    end
                    if attempt == 3
                        # last resort: accept a looser tolerance this once
                        if is_ac
                            JuMP.set_optimizer_attribute(sp, "tol", 1e-4)
                        else
                            # Clarabel exposes separate gap/feasibility tolerances
                            for a in ("tol_gap_abs", "tol_gap_rel", "tol_feas")
                                JuMP.set_optimizer_attribute(sp, a, 1e-5)
                            end
                        end
                    end
                end
                JuMP.optimize!(sp)
            catch
                # swallow and escalate — the next rung starts from scratch
            end
            try
                solved() && return
            catch
            end
        end
        return
    end
end
