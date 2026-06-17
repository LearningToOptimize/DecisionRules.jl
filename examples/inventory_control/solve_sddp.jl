"""
Stochastic lot-sizing solved with SDDP.jl using FixedDiscreteDuality.

SDDP.FixedDiscreteDuality and DecisionRules.FixedDiscreteIntegerStrategy share
the same idea: fix the binary ordering decision z to its MIP incumbent, re-solve
the resulting LP, and read LP duals as gradient/cut information. SDDP uses those
duals to build Benders cuts that approximate the value function; TS-DDR uses them
to back-propagate through the neural policy.

Run this script after activate the inventory_control project:
  julia --project=examples/inventory_control examples/inventory_control/solve_sddp.jl
"""

using SDDP
using JuMP
using HiGHS
using CSV, DataFrames
using Statistics
using Random

# ── Same problem parameters as build_inventory_problem.jl ─────────────────────
const T_STAGES    = 12
const K_COST      = 30.0    # fixed ordering cost
const C_COST      = 2.0     # unit ordering cost
const H_COST      = 1.0     # holding cost
const P_COST      = 10.0    # backlog penalty
const Q_MAX       = 80.0    # max order quantity
const I_0         = 20.0    # initial inventory
const D_MIN       = 10.0    # demand lower bound
const D_MAX       = 30.0    # demand upper bound
const N_TRAIN     = 30      # scenarios per stage for training
const N_SIM       = 200     # out-of-sample simulations

rng_train = MersenneTwister(42)
demand_scenarios = [
    D_MIN .+ (D_MAX - D_MIN) .* rand(rng_train, N_TRAIN)
    for _ in 1:T_STAGES
]

# ── Build the SDDP model ───────────────────────────────────────────────────────
model = SDDP.LinearPolicyGraph(
    stages      = T_STAGES,
    sense       = :Min,
    lower_bound = 0.0,
    optimizer   = HiGHS.Optimizer,
) do sp, t
    set_silent(sp)

    # State variable: net inventory (continuous)
    @variable(sp, s, SDDP.State, initial_value = I_0)

    # Local decisions
    @variable(sp, 0 <= q <= Q_MAX)     # order quantity (continuous)
    @variable(sp, z, Bin)              # order indicator (binary)
    @variable(sp, inv_hold >= 0)       # inventory on hand
    @variable(sp, back >= 0)           # backlog

    # Uncertain demand — fixed inside parameterize callback
    @variable(sp, d_par)
    SDDP.parameterize(sp, demand_scenarios[t]) do d_val
        JuMP.fix(d_par, d_val)
    end

    # Constraints
    @constraint(sp, s.out == s.in + q - d_par)     # inventory balance
    @constraint(sp, inv_hold - back == s.out)       # split into on-hand / backlog
    @constraint(sp, q <= Q_MAX * z)                 # order only if z = 1

    @stageobjective(sp, K_COST * z + C_COST * q + H_COST * inv_hold + P_COST * back)
end

# ── Train ──────────────────────────────────────────────────────────────────────
# FixedDiscreteDuality: identical concept to FixedDiscreteIntegerStrategy.
# Fix z to its MIP incumbent → solve LP → read LP duals as Benders subgradients.
# FixedDiscreteDuality keeps z ∈ {0,1} in the forward pass, giving honest
# integer-feasible policies.  ContinuousConicDuality relaxes z in the forward
# pass too, which makes simulation costs artificially low (below DP optimal).
println("Training SDDP with FixedDiscreteDuality...")
SDDP.train(
    model;
    duality_handler = SDDP.FixedDiscreteDuality(),
    iteration_limit = 500,
    stopping_rules  = [SDDP.BoundStalling(50, 1e-3)],
    print_level     = 1,
)

lower_bound = SDDP.calculate_bound(model)
println("\nSDDP lower bound (expected cost): $lower_bound")

# ── Out-of-sample simulation on fresh U[D_MIN, D_MAX] draws ───────────────────
# SDDP.simulate samples from the N_TRAIN training scenarios (possibly biased mean).
# Manual rollout fixes each subproblem's state and demand directly so we can
# use fresh draws from the true U[D_MIN, D_MAX] distribution.
println("\nManual rollout on $N_SIM fresh U[$D_MIN,$D_MAX] scenarios...")

rng_eval = MersenneTwister(99999)

function rollout_sddp(model, n_sim, rng)
    costs = Vector{Float64}(undef, n_sim)
    for sim in 1:n_sim
        state      = I_0
        total_cost = 0.0
        for t in 1:T_STAGES
            d  = D_MIN + (D_MAX - D_MIN) * rand(rng)
            sp = model.nodes[t].subproblem

            JuMP.fix(sp[:s].in,  state; force = true)
            JuMP.fix(sp[:d_par], d;     force = true)
            optimize!(sp)

            z     = round(value(sp[:z]))    # round to enforce {0,1}
            q     = value(sp[:q])
            s_out = value(sp[:s].out)

            total_cost += K_COST * z + C_COST * q +
                          H_COST * max(s_out, 0.0) + P_COST * max(-s_out, 0.0)
            state = s_out
        end
        costs[sim] = total_cost
    end
    return costs
end

sddp_costs = rollout_sddp(model, N_SIM, rng_eval)

μ = mean(sddp_costs)
σ = std(sddp_costs)
println("SDDP policy — mean cost: $(round(μ, digits=1)) ± $(round(σ, digits=1))")
println("SDDP lower bound:        $(round(lower_bound, digits=1))")
println("Optimality gap (upper):  $(round(100*(μ - lower_bound)/μ, digits=1))%")

# ── Save results ───────────────────────────────────────────────────────────────
result_dir = joinpath(@__DIR__, "results")
mkpath(result_dir)
CSV.write(
    joinpath(result_dir, "sddp_costs.csv"),
    DataFrame(operational_cost = sddp_costs),
)
open(joinpath(result_dir, "sddp_bound.txt"), "w") do io
    println(io, lower_bound)
end
println("\nSaved results to $(result_dir)/sddp_costs.csv")
