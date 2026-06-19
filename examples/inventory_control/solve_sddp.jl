"""
Solve the inventory control problem with SDDP.jl.

SDDP uses a PAR(1) approximation of the true latent demand process:
  d_t ≈ μ_t + α·(d_{t-1} - μ_{t-1}) + ω_t
where μ_t, α, and Ω are fitted from simulated demand paths.

Two cases:
1. Relaxed: no binary z, SDDP is near-optimal for convex problems
2. Integer: AlternativeForwardPass — forward pass solves true MIP (z ∈ {0,1}),
   backward pass uses LP relaxation (z ∈ [0,1]) to compute cuts with valid duals.
   Both models share the same PAR(1) demand structure.
"""

using SDDP
using JuMP
using HiGHS
using CSV, DataFrames
using Statistics
using Random

include(joinpath(@__DIR__, "build_inventory_problem.jl"))

const N_SIM = 300
const ITERATION_LIMIT = 500

# ═══════════════════════════════════════════════════════════════════════════════
# Fit PAR(1) to the true demand process
# ═══════════════════════════════════════════════════════════════════════════════
function fit_par1(; n_paths=10000, n_omega=9, seed=123)
    rng = MersenneTwister(seed)
    paths = [sample_inventory_demand_path(rng) for _ in 1:n_paths]

    mu = [mean(p[t] for p in paths) for t in 1:INVENTORY_T]

    num = 0.0
    den = 0.0
    for p in paths
        for t in 2:INVENTORY_T
            x = p[t-1] - mu[t-1]
            y = p[t] - mu[t]
            num += x * y
            den += x * x
        end
    end
    alpha = clamp(num / den, 0.0, 0.99)

    residuals = Float64[]
    for p in paths
        for t in 2:INVENTORY_T
            r = (p[t] - mu[t]) - alpha * (p[t-1] - mu[t-1])
            push!(residuals, r)
        end
    end
    sort!(residuals)
    n = length(residuals)
    omega = [residuals[round(Int, (2k-1)*n / (2*n_omega))] for k in 1:n_omega]

    return mu, alpha, omega
end

println("Fitting PAR(1) to true demand process...")
par_mu, par_alpha, par_omega = fit_par1()
println("  μ (seasonal means): $(round.(par_mu, digits=1))")
println("  α (autocorrelation): $(round(par_alpha, digits=3))")
println("  Ω (innovations, $(length(par_omega)) points): $(round.(par_omega, digits=1))")

# ═══════════════════════════════════════════════════════════════════════════════
# Build SDDP model with PAR(1) demand approximation
# ═══════════════════════════════════════════════════════════════════════════════
function build_sddp_model(; integer::Bool=false, binary::Bool=false,
                            mu=par_mu, alpha=par_alpha, omega=par_omega)
    d_lag_init = mu[1]
    SDDP.LinearPolicyGraph(
        stages=2 * INVENTORY_T,
        sense=:Min,
        lower_bound=0.0,
        optimizer=HiGHS.Optimizer,
    ) do sp, stage
        set_silent(sp)
        @variable(sp, s, SDDP.State, initial_value=INVENTORY_I0)
        @variable(sp, d_lag, SDDP.State, initial_value=d_lag_init)

        if isodd(stage)
            @variable(sp, 0 <= q <= INVENTORY_Q_MAX)
            if integer
                if binary
                    @variable(sp, z, Bin)
                else
                    @variable(sp, 0 <= z <= 1)
                end
                @constraint(sp, q <= INVENTORY_Q_MAX * z)
                @stageobjective(sp, INVENTORY_K * z + INVENTORY_C * q)
            else
                @stageobjective(sp, INVENTORY_C * q)
            end
            @constraint(sp, s.out == s.in + q)
            @constraint(sp, d_lag.out == d_lag.in)
        else
            t = stage ÷ 2
            mu_prev = t == 1 ? mu[INVENTORY_T] : mu[t-1]
            @variable(sp, omega_var)
            @variable(sp, demand)
            @variable(sp, inv_hold >= 0)
            @variable(sp, back >= 0)
            @constraint(sp, demand == mu[t] + alpha * (d_lag.in - mu_prev) + omega_var)
            @constraint(sp, d_lag.out == demand)
            @constraint(sp, s.out == s.in - demand)
            @constraint(sp, inv_hold - back == s.out)
            @stageobjective(sp, INVENTORY_H * inv_hold + INVENTORY_P * back)
            SDDP.parameterize(sp, omega) do ω
                JuMP.fix(omega_var, ω; force=true)
            end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Training log extraction
# ═══════════════════════════════════════════════════════════════════════════════
function training_log_dataframe(model)
    log = model.most_recent_training_results.log
    rows = DataFrame(iteration=Int[], bound=Float64[], simulation_value=Float64[], time=Float64[])
    for row in log
        iter = hasproperty(row, :iteration) ? row.iteration : row[:iteration]
        bound = hasproperty(row, :bound) ? row.bound : row[:bound]
        sim = try
            hasproperty(row, :simulation_value) ? row.simulation_value : row[:simulation_value]
        catch; NaN end
        tm = try
            hasproperty(row, :time) ? row.time : row[:time]
        catch; NaN end
        push!(rows, (iteration=iter, bound=bound, simulation_value=sim, time=tm))
    end
    return rows
end

# ═══════════════════════════════════════════════════════════════════════════════
# Rollout: evaluate SDDP policy on fresh demand from the TRUE process
# ═══════════════════════════════════════════════════════════════════════════════
function rollout_sddp(model, n_sim; integer_round::Bool=false, mu=par_mu)
    costs = Vector{Float64}(undef, n_sim)
    traj_inv = Matrix{Float64}(undef, n_sim, INVENTORY_T + 1)

    for sim in 1:n_sim
        state = INVENTORY_I0
        d_lag = mu[1]
        total_cost = 0.0
        traj_inv[sim, 1] = state
        demand_path = sample_inventory_demand_path()

        for t in 1:INVENTORY_T
            order_sp = model.nodes[2t - 1].subproblem
            JuMP.fix(order_sp[:s].in, state; force=true)
            JuMP.fix(order_sp[:d_lag].in, d_lag; force=true)
            optimize!(order_sp)

            q_val = clamp(value(order_sp[:q]), 0.0, INVENTORY_Q_MAX)
            if integer_round
                z_val = q_val <= 1e-7 ? 0.0 : 1.0
                stage_cost = INVENTORY_K * z_val + INVENTORY_C * q_val
            else
                stage_cost = INVENTORY_C * q_val
            end
            s_mid = state + q_val

            d = demand_path[t]
            s_out = s_mid - d
            stage_cost += INVENTORY_H * max(s_out, 0.0) + INVENTORY_P * max(-s_out, 0.0)

            total_cost += stage_cost
            state = s_out
            d_lag = d
            traj_inv[sim, t+1] = state
        end
        costs[sim] = total_cost
    end
    return costs, traj_inv
end

result_dir = joinpath(@__DIR__, "results")
mkpath(result_dir)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Relaxed (continuous) SDDP
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 60)
println("SECTION 1: SDDP — Relaxed (continuous, PAR(1) approx)")
println("=" ^ 60)

model_relaxed = build_sddp_model(; integer=false)
println("Training relaxed SDDP ($(2*INVENTORY_T) stages)...")
sddp_relax_start = time()
SDDP.train(
    model_relaxed;
    duality_handler=SDDP.ContinuousConicDuality(),
    iteration_limit=ITERATION_LIMIT,
    stopping_rules=[SDDP.BoundStalling(100, 1e-3)],
    print_level=1,
)
sddp_relax_seconds = time() - sddp_relax_start

relax_bound = SDDP.calculate_bound(model_relaxed)
println("\nRelaxed SDDP bound: $(round(relax_bound, digits=1))")

println("Rollout on $N_SIM fresh scenarios (TRUE demand)...")
Random.seed!(555)
relax_eval_start = time()
relax_costs, relax_traj = rollout_sddp(model_relaxed, N_SIM; integer_round=false)
relax_eval_seconds = time() - relax_eval_start

μ_r = mean(relax_costs)
σ_r = std(relax_costs)
println("Relaxed SDDP — mean: $(round(μ_r, digits=1)) ± $(round(σ_r, digits=1))")
println("Gap to bound: $(round(100 * (μ_r - relax_bound) / μ_r, digits=1))%")

CSV.write(joinpath(result_dir, "relaxed_sddp_costs.csv"), DataFrame(operational_cost=relax_costs))
CSV.write(joinpath(result_dir, "relaxed_sddp_trajectories.csv"),
    DataFrame(relax_traj, [Symbol("t$i") for i in 0:INVENTORY_T]))
CSV.write(joinpath(result_dir, "relaxed_sddp_training_log.csv"), training_log_dataframe(model_relaxed))
CSV.write(joinpath(result_dir, "relaxed_sddp_timing.csv"),
    DataFrame(method=["SDDP (PAR)"], fit_seconds=[0.0],
              eval_seconds=[sddp_relax_seconds], n_eval=[N_SIM]))
open(joinpath(result_dir, "relaxed_sddp_bound.txt"), "w") do io
    println(io, relax_bound)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Integer SDDP (MIP forward pass + LP cuts via AlternativeForwardPass)
#
# Two-phase training:
#   Phase 1 — LP forward + LP backward until convergence (warm-start cuts)
#   Phase 2 — MIP forward + LP backward (refine at MIP-realistic trial points)
# Rollout on true MIP model (z ∈ {0,1}) with all accumulated cuts.
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 60)
println("SECTION 2: SDDP — Integer (MIP forward + LP cuts)")
println("=" ^ 60)

model_lp = build_sddp_model(; integer=true, binary=false)
model_mip = build_sddp_model(; integer=true, binary=true)

# --- Phase 1: LP warm-start ---
println("Phase 1: LP warm-start ($(2*INVENTORY_T) stages)...")
sddp_int_start = time()
SDDP.train(
    model_lp;
    duality_handler=SDDP.ContinuousConicDuality(),
    iteration_limit=ITERATION_LIMIT,
    stopping_rules=[SDDP.BoundStalling(100, 1e-3)],
    print_level=1,
)
lp_bound = SDDP.calculate_bound(model_lp)
println("  LP warm-start bound: $(round(lp_bound, digits=1))")

phase1_log = training_log_dataframe(model_lp)
sddp_lp_seconds = time() - sddp_int_start

# --- LP rollout (default SDDP baseline: LP decisions + integer rounding) ---
println("LP rollout (default SDDP) on $N_SIM fresh scenarios...")
Random.seed!(555)
lp_eval_start = time()
lp_costs, lp_traj = rollout_sddp(model_lp, N_SIM; integer_round=true)
lp_eval_seconds = time() - lp_eval_start
μ_lp = mean(lp_costs)
println("  Default SDDP (LP rollout) — mean: $(round(μ_lp, digits=1)) ± $(round(std(lp_costs), digits=1))")

CSV.write(joinpath(result_dir, "integer_sddp_lp_costs.csv"), DataFrame(operational_cost=lp_costs))
CSV.write(joinpath(result_dir, "integer_sddp_lp_trajectories.csv"),
    DataFrame(lp_traj, [Symbol("t$i") for i in 0:INVENTORY_T]))
CSV.write(joinpath(result_dir, "integer_sddp_lp_training_log.csv"), phase1_log)
CSV.write(joinpath(result_dir, "integer_sddp_lp_timing.csv"),
    DataFrame(method=["SDDP (LP relax)"], fit_seconds=[0.0],
              eval_seconds=[sddp_lp_seconds], n_eval=[N_SIM]))

cuts_file = joinpath(result_dir, "integer_lp_cuts.json")
SDDP.write_cuts_to_file(model_lp, cuts_file)
SDDP.read_cuts_from_file(model_mip, cuts_file)
println("  Exported LP cuts to MIP model")

# --- Phase 2: MIP forward + LP backward ---
println("\nPhase 2: AlternativeForwardPass — MIP forward + LP cuts...")
println("  Forward pass: true MIP (z ∈ {0,1})")
println("  Backward pass: LP relaxation (z ∈ [0,1]) for cuts")
SDDP.train(
    model_lp;
    forward_pass=SDDP.AlternativeForwardPass(model_mip),
    post_iteration_callback=SDDP.AlternativePostIterationCallback(model_mip),
    duality_handler=SDDP.ContinuousConicDuality(),
    iteration_limit=ITERATION_LIMIT,
    add_to_existing_cuts=true,
    print_level=1,
)
sddp_int_seconds = time() - sddp_int_start

phase2_log = training_log_dataframe(model_lp)
phase2_log.iteration .+= maximum(phase1_log.iteration)
combined_log = vcat(phase1_log, phase2_log)

int_bound = SDDP.calculate_bound(model_lp)
println("\nLP relaxation bound (after both phases): $(round(int_bound, digits=1))")

println("MIP rollout on $N_SIM fresh scenarios (TRUE demand)...")
Random.seed!(555)
int_eval_start = time()
int_costs, int_traj = rollout_sddp(model_mip, N_SIM; integer_round=true)
int_eval_seconds = time() - int_eval_start

μ_i = mean(int_costs)
σ_i = std(int_costs)
println("Integer SDDP (MIP fwd) — mean: $(round(μ_i, digits=1)) ± $(round(σ_i, digits=1))")
println("Gap to LP bound: $(round(100 * (μ_i - int_bound) / μ_i, digits=1))%")

CSV.write(joinpath(result_dir, "integer_sddp_costs.csv"), DataFrame(operational_cost=int_costs))
CSV.write(joinpath(result_dir, "integer_sddp_trajectories.csv"),
    DataFrame(int_traj, [Symbol("t$i") for i in 0:INVENTORY_T]))
CSV.write(joinpath(result_dir, "integer_sddp_training_log.csv"), combined_log)
CSV.write(joinpath(result_dir, "integer_sddp_timing.csv"),
    DataFrame(method=["SDDP (MIP fwd)"], fit_seconds=[0.0],
              eval_seconds=[sddp_int_seconds], n_eval=[N_SIM]))
open(joinpath(result_dir, "integer_sddp_bound.txt"), "w") do io
    println(io, int_bound)
end

println("\nAll SDDP results saved to $(relpath(result_dir, @__DIR__))")
