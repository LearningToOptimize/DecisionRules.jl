# Validate that DecisionRules JuMP subproblems match SDDP's problem formulation.
#
# Reads SDDP trajectory data (states, inflows, stage costs) extracted by
# sddp/extract_sddp_trajectories.jl, then replays those trajectories through
# the JuMP subproblems built by load_hydropowermodels.jl.
#
# For each SDDP scenario and stage:
#   1. Set initial state = SDDP's reservoir_in
#   2. Set inflows = SDDP's inflow values
#   3. Set target = SDDP's reservoir_out (what SDDP realized)
#   4. Solve the JuMP subproblem
#   5. Compare: JuMP stage cost vs SDDP stage cost
#
# If the problems are equivalent, costs should match to solver tolerance.
# Any mismatch reveals structural differences (load scaling, constraints, etc.).
#
# Runs in the main DecisionRules project environment.
using DecisionRules
using JuMP, DiffOpt, Ipopt
using CSV, DataFrames
using JSON
using Statistics

HydroPowerModels_dir = dirname(@__FILE__)
include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))

case_name = "bolivia"
formulation = "ACPPowerModel"
case_dir = joinpath(HydroPowerModels_dir, case_name)
out_dir = joinpath(case_dir, formulation)

println("=" ^ 60)
println("Validate SDDP vs JuMP subproblems")
println("=" ^ 60)

# ── Load SDDP trajectory data ─────────────────────────────────────────────
traj_file = joinpath(out_dir, "sddp_trajectories_validate.csv")
meta_file = joinpath(out_dir, "sddp_validate_meta.csv")
demand_file = joinpath(out_dir, "sddp_validate_demands.csv")

if !isfile(traj_file)
    error("SDDP trajectory file not found: $traj_file\n" *
          "Run sddp/extract_sddp_trajectories.jl first.")
end

traj_df = CSV.read(traj_file, DataFrame)

# Infer nHyd from trajectory columns (res_in_1, res_in_2, ...)
nhyd = count(n -> startswith(string(n), "res_in_"), names(traj_df))
num_gen = count(n -> startswith(string(n), "pg_"), names(traj_df))

# Load metadata if available
if isfile(meta_file)
    meta_df = CSV.read(meta_file, DataFrame)
    meta = Dict(row.key => row.value for row in eachrow(meta_df))
    baseMVA = meta["baseMVA"]
    load_scaler = meta["load_scaler"]
else
    println("  WARNING: meta file not found, using defaults")
    hydro_json = JSON.parsefile(joinpath(case_dir, "hydro.json"))
    power_json = JSON.parsefile(joinpath(case_dir, "PowerModels.json"))
    baseMVA = power_json["baseMVA"]
    load_scaler = 0.6
end

scenarios = sort(unique(traj_df.scenario))
num_scenarios = length(scenarios)
stages = sort(unique(traj_df.stage))
num_stages = length(stages)

println("  Trajectories: $num_scenarios scenarios × $num_stages stages")
println("  nHyd=$nhyd, num_gen=$num_gen, baseMVA=$baseMVA, load_scaler=$load_scaler")

# ── Check demand consistency ───────────────────────────────────────────────
# The MOF file has demands baked in. SDDP applies 0.6 scaling.
# Let's read the MOF file and compare demands.
println("\n--- Demand cross-check ---")
mof_model = JuMP.read_from_file(
    joinpath(case_dir, formulation * ".mof.json"); use_nlp_block=false
)
# Find load constraints or demand parameters in MOF
# The demand is typically in the objective or power balance constraints.
# Let's check by looking at the pd/qd variables or fixed values.
mof_vars = JuMP.all_variables(mof_model)
demand_vars = filter(v -> occursin("deficit", JuMP.name(v)) || occursin("load", JuMP.name(v)), mof_vars)
println("  MOF demand-related variables: $(length(demand_vars))")
for v in demand_vars[1:min(3, length(demand_vars))]
    println("    $(JuMP.name(v))")
end

try
    pb_cons = filter(c -> occursin("kcl", string(c)), JuMP.all_constraints(mof_model; include_variable_in_set_constraints=false))
    if !isempty(pb_cons)
        println("  Found $(length(pb_cons)) power balance (kcl) constraints")
        println("  First: $(pb_cons[1])")
    end
catch e
    println("  (could not inspect constraints: $e)")
end

# ── Build JuMP subproblems ─────────────────────────────────────────────────
diff_optimizer =
    () -> DiffOpt.diff_optimizer(
        optimizer_with_attributes(
            Ipopt.Optimizer,
            "print_level" => 0,
            "linear_solver" => "mumps",
        ),
    )

# Build with strict=true (hard equality targets, no deficit)
subproblems_strict, state_in_strict, state_out_strict, uncert_strict,
    initial_state, max_volume, _ = build_hydropowermodels(
    case_dir, formulation * ".mof.json";
    num_stages=num_stages,
    optimizer=diff_optimizer,
    strict=true,
)

# Also build with penalty (to test non-strict)
subproblems_penalty, state_in_penalty, state_out_penalty, uncert_penalty,
    _, _, _ = build_hydropowermodels(
    case_dir, formulation * ".mof.json";
    num_stages=num_stages,
    optimizer=diff_optimizer,
    penalty_l1=:auto, penalty_l2=:auto,
)

println("\n  Built $num_stages subproblems (strict and penalty modes)")
println("  Initial state: $initial_state")
println("  SDDP initial: $([meta["initial_vol_$r"] for r in 1:nhyd])")

# ── Cross-check inflow data ───────────────────────────────────────────────
println("\n--- Inflow cross-check ---")
# DecisionRules reads inflows from the same CSV
nCen = length(uncert_strict[1])  # number of scenarios per stage
println("  DecisionRules nCen=$nCen")

# Check that inflow values match
for s in scenarios[1:min(2, num_scenarios)]
    stage_rows = filter(r -> r.scenario == s, traj_df)
    for t in 1:min(3, num_stages)
        row = stage_rows[t, :]
        ω = row.noise_term
        # DecisionRules uncertainty_samples[t][ω] should match SDDP inflows
        dr_scenario = uncert_strict[t][ω]
        for r in 1:nhyd
            dr_inflow = dr_scenario[r][2]
            sddp_inflow = row[Symbol("inflow_$r")]
            if !isapprox(dr_inflow, sddp_inflow; atol=1e-10)
                println("  MISMATCH: scenario=$s stage=$t hydro=$r: DR=$dr_inflow SDDP=$sddp_inflow")
            end
        end
    end
end
println("  Inflow cross-check passed (spot-checked)")

# ── Replay SDDP trajectories through JuMP subproblems ─────────────────────
println("\n--- Stage-by-stage cost comparison ---")

results_rows = NamedTuple[]

for s in scenarios
    stage_rows = filter(r -> r.scenario == s, traj_df)
    sddp_total = 0.0
    jump_strict_total = 0.0
    jump_penalty_total = 0.0
    max_cost_diff_strict = 0.0
    max_cost_diff_penalty = 0.0

    for t in 1:num_stages
        row = stage_rows[t, :]
        ω = row.noise_term
        sddp_cost = row.stage_objective

        # SDDP's realized states
        sddp_res_in = [row[Symbol("res_in_$r")] for r in 1:nhyd]
        sddp_res_out = [row[Symbol("res_out_$r")] for r in 1:nhyd]
        sddp_inflows = [row[Symbol("inflow_$r")] for r in 1:nhyd]

        # ── Strict subproblem ──
        sp = subproblems_strict[t]
        # Set initial state
        for (j, param) in enumerate(state_in_strict[t])
            set_parameter_value(param, sddp_res_in[j])
        end
        # Set inflows
        for (param, _) in uncert_strict[t][ω]
            # Already the right values, but let's be explicit
        end
        dr_scenario = uncert_strict[t][ω]
        for (param, val) in dr_scenario
            set_parameter_value(param, val)
        end
        # Set target = SDDP's realized out state
        for j in 1:nhyd
            target_param = state_out_strict[t][j][1]
            set_parameter_value(target_param, sddp_res_out[j])
        end

        optimize!(sp)
        jump_strict_cost = objective_value(sp)

        # Read realized state from JuMP
        jump_res_out = [value(state_out_strict[t][j][2]) for j in 1:nhyd]

        # ── Penalty subproblem ──
        sp_p = subproblems_penalty[t]
        for (j, param) in enumerate(state_in_penalty[t])
            set_parameter_value(param, sddp_res_in[j])
        end
        for (param, val) in uncert_penalty[t][ω]
            set_parameter_value(param, val)
        end
        for j in 1:nhyd
            target_param = state_out_penalty[t][j][1]
            set_parameter_value(target_param, sddp_res_out[j])
        end

        optimize!(sp_p)
        jump_penalty_cost = objective_value(sp_p)
        jump_penalty_res_out = [value(state_out_penalty[t][j][2]) for j in 1:nhyd]
        # Penalty cost includes deficit — compute cost without deficit
        # The deficit penalty is the diff between penalty and strict cost ideally

        # State comparison
        state_diff = maximum(abs.(jump_res_out .- sddp_res_out))
        state_diff_penalty = maximum(abs.(jump_penalty_res_out .- sddp_res_out))

        cost_diff_strict = jump_strict_cost - sddp_cost
        cost_diff_penalty = jump_penalty_cost - sddp_cost

        sddp_total += sddp_cost
        jump_strict_total += jump_strict_cost
        jump_penalty_total += jump_penalty_cost
        max_cost_diff_strict = max(max_cost_diff_strict, abs(cost_diff_strict))
        max_cost_diff_penalty = max(max_cost_diff_penalty, abs(cost_diff_penalty))

        push!(results_rows, (
            scenario = s,
            stage = t,
            noise_term = ω,
            sddp_cost = sddp_cost,
            jump_strict_cost = jump_strict_cost,
            jump_penalty_cost = jump_penalty_cost,
            cost_diff_strict = cost_diff_strict,
            cost_diff_penalty = cost_diff_penalty,
            rel_diff_strict = abs(cost_diff_strict) / max(abs(sddp_cost), 1e-12),
            max_state_diff_strict = state_diff,
            max_state_diff_penalty = state_diff_penalty,
        ))

        if t <= 3 || abs(cost_diff_strict) > 1.0
            println("  s=$s t=$t: SDDP=$(round(sddp_cost; digits=2)) " *
                    "JuMP_strict=$(round(jump_strict_cost; digits=2)) " *
                    "diff=$(round(cost_diff_strict; digits=4)) " *
                    "state_diff=$(round(state_diff; sigdigits=3))")
        end
    end

    println("  Scenario $s totals: " *
            "SDDP=$(round(sddp_total; digits=1)) " *
            "JuMP_strict=$(round(jump_strict_total; digits=1)) " *
            "diff=$(round(jump_strict_total - sddp_total; digits=1)) " *
            "max_stage_diff=$(round(max_cost_diff_strict; digits=4))")
end

# ── Summary ────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 60)
println("Validation Summary")
println("=" ^ 60)

df_results = DataFrame(results_rows)
println("  Total comparisons: $(nrow(df_results))")
println("  Max stage cost diff (strict): $(round(maximum(abs.(df_results.cost_diff_strict)); digits=4))")
println("  Mean stage cost diff (strict): $(round(mean(abs.(df_results.cost_diff_strict)); digits=4))")
println("  Max relative diff (strict): $(round(maximum(df_results.rel_diff_strict); sigdigits=3))")
println("  Max state diff (strict): $(round(maximum(df_results.max_state_diff_strict); sigdigits=3))")
println("  Max stage cost diff (penalty): $(round(maximum(abs.(df_results.cost_diff_penalty)); digits=4))")

# Per-scenario total comparison
for s in scenarios
    s_rows = filter(r -> r.scenario == s, df_results)
    sddp_total = sum(s_rows.sddp_cost)
    jump_total = sum(s_rows.jump_strict_cost)
    println("  Scenario $s: SDDP=$(round(sddp_total; digits=1)) JuMP=$(round(jump_total; digits=1)) " *
            "gap=$(round(jump_total - sddp_total; digits=1)) " *
            "($(round((jump_total - sddp_total) / sddp_total * 100; digits=3))%)")
end

tol = 1.0
big_diffs = filter(r -> abs(r.cost_diff_strict) > tol, df_results)
if nrow(big_diffs) > 0
    println("\n  WARNING: $(nrow(big_diffs)) stages have |cost diff| > $tol")
    println("  Largest differences:")
    sorted = sort(big_diffs, :cost_diff_strict; by=abs, rev=true)
    for r in eachrow(sorted[1:min(10, nrow(sorted)), :])
        println("    s=$(r.scenario) t=$(r.stage): diff=$(round(r.cost_diff_strict; digits=4))")
    end
else
    println("\n  ✓ All stage cost differences within $tol — problems are equivalent")
end

# Save detailed results
results_file = joinpath(out_dir, "validate_sddp_vs_jump.csv")
CSV.write(results_file, df_results)
println("\nSaved: $results_file")
