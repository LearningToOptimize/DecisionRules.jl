# Evaluate a trained strict-mode HydroReachablePolicy on 96-stage rollouts.
#
# Loads a saved model checkpoint, builds 96-stage strict subproblems, and
# runs stage-wise rollout evaluation on 100 held-out inflow scenarios.
# Reports per-scenario operational costs (no deficit/penalty — strict mode
# has none) and summary statistics.  Also records per-stage reservoir volumes
# and thermal generation for comparison plots (MeanVolume.csv,
# MeanGeneration.csv, costs.csv).
#
# Usage:
#   julia --project -t auto eval_strict_rollout.jl MODEL_PATH
#
# Environment overrides:
#   DR_NUM_EVAL_STAGES=96   number of rollout stages
#   DR_NUM_SCENARIOS=100    number of evaluation scenarios
#   DR_EVAL_SEED=1221       random seed (matches SDDP evaluation)
using DecisionRules
using Flux
using Statistics
using Random
using JuMP, DiffOpt, Ipopt
using JLD2
using CSV, DataFrames
using JSON

HydroPowerModels_dir = dirname(@__FILE__)
include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))
include(joinpath(HydroPowerModels_dir, "hydro_reachable_policy.jl"))

# ── Parse arguments ─────────────────────────────────────────────────────────

model_path = ARGS[1]
num_eval_stages = parse(Int, get(ENV, "DR_NUM_EVAL_STAGES", "96"))
num_scenarios = parse(Int, get(ENV, "DR_NUM_SCENARIOS", "100"))
seed = parse(Int, get(ENV, "DR_EVAL_SEED", "1221"))
layers = Int64[128, 128]

println("=" ^ 60)
println("Strict Rollout Evaluation")
println("  Model:      $model_path")
println("  Stages:     $num_eval_stages")
println("  Scenarios:  $num_scenarios")
println("  Seed:       $seed")
println("=" ^ 60)

# ── Build strict subproblems for evaluation ─────────────────────────────────

case_name = "bolivia"
formulation = "ACPPowerModel"
formulation_file = formulation * ".mof.json"

diff_optimizer =
    () -> DiffOpt.diff_optimizer(
        optimizer_with_attributes(
            Ipopt.Optimizer,
            "print_level" => 0,
            "linear_solver" => "mumps",
        ),
    )

subproblems, state_params_in, state_params_out, uncertainty_samples,
    initial_state, max_volume, hydro_meta = build_hydropowermodels(
    joinpath(HydroPowerModels_dir, case_name),
    formulation_file;
    num_stages=num_eval_stages,
    optimizer=diff_optimizer,
    strict=true,
)

num_hydro = length(initial_state)

# ── Identify thermal generators ─────────────────────────────────────────────

hydro_data = JSON.parsefile(joinpath(HydroPowerModels_dir, case_name, "hydro.json"))
power_data = JSON.parsefile(joinpath(HydroPowerModels_dir, case_name, "PowerModels.json"))
baseMVA = power_data["baseMVA"]
hydro_grid_idx = Set(hg["index_grid"] for hg in hydro_data["Hydrogenerators"])
num_gen = length(power_data["gen"])
thermal_idx = [i for i in 1:num_gen if !(i in hydro_grid_idx)]

volume_to_mw(volume; k=0.0036) = volume / k

# Precompute pg variable references for each stage
pg_vars_per_stage = [DecisionRules.find_variables(subproblems[t], ["pg"]) for t in 1:num_eval_stages]

# ── Build policy and load trained weights ───────────────────────────────────

models = hydro_reachable_policy(hydro_meta, layers)

model_save = JLD2.load(model_path)
model_state = model_save["model_state"]
load_policy_weights!(models, model_state)
println("Loaded model weights from $model_path")

# ── Run rollout evaluation ──────────────────────────────────────────────────

Random.seed!(seed)
eval_scenarios = [DecisionRules.sample(uncertainty_samples) for _ in 1:num_scenarios]

println("\nEvaluating $num_scenarios scenarios on $num_eval_stages stages...")

costs = Float64[]
vol_trajectories = zeros(num_eval_stages, num_scenarios)
gen_trajectories = zeros(num_eval_stages, num_scenarios)

for (i, scenario) in enumerate(eval_scenarios)
    Flux.reset!(models)

    state = Float64.(initial_state)
    scenario_cost = 0.0

    for t in 1:num_eval_stages
        # Set initial state parameters
        for (i, param) in enumerate(state_params_in[t])
            set_parameter_value(param, state[i])
        end

        # Set uncertainty parameters
        w_t = scenario[t]
        for (param, val) in w_t
            set_parameter_value(param, val)
        end

        # Policy forward pass
        w_vals = Float32.([val for (_, val) in w_t])
        x_hat = models(vcat(w_vals, Float32.(state)))

        # Set targets
        for j in 1:num_hydro
            target_param = state_params_out[t][j][1]
            set_parameter_value(target_param, Float64(x_hat[j]))
        end

        # Solve
        optimize!(subproblems[t])
        scenario_cost += objective_value(subproblems[t])

        # Read realized reservoir volumes
        for j in 1:num_hydro
            state[j] = value(state_params_out[t][j][2])
        end

        # Record per-stage data
        vol_trajectories[t, i] = sum(volume_to_mw(state[j]) for j in 1:num_hydro)

        # Thermal generation: sum pg * baseMVA for thermal generators
        gen_trajectories[t, i] = sum(
            value(pg_vars_per_stage[t][j]) * baseMVA for j in thermal_idx
        )
    end

    push!(costs, scenario_cost)
    if i % 10 == 0 || i == num_scenarios
        println("  [$i/$num_scenarios] cost = $(round(scenario_cost; digits=1)), running mean = $(round(mean(costs); digits=1))")
    end
end

# ── Report results ──────────────────────────────────────────────────────────

println("\n" * "=" ^ 60)
println("Results: Strict TS-DDR Rollout ($num_eval_stages stages, $num_scenarios scenarios)")
println("=" ^ 60)
println("  Mean cost:   $(round(mean(costs); digits=1))")
println("  Std:         $(round(std(costs); digits=1))")
println("  Min:         $(round(minimum(costs); digits=1))")
println("  Max:         $(round(maximum(costs); digits=1))")
println("  Median:      $(round(median(costs); digits=1))")
println("  Violation:   0.0% (strict mode)")
println("=" ^ 60)

# ── Save results ────────────────────────────────────────────────────────────

out_dir = joinpath(HydroPowerModels_dir, case_name, formulation)

# Per-scenario costs → costs.csv
const STRICT_COL = "TS-DDR (strict)"
costs_file = joinpath(out_dir, "costs.csv")
if isfile(costs_file)
    df = CSV.read(costs_file, DataFrame)
    df[!, STRICT_COL] = costs
else
    df = DataFrame(Symbol(STRICT_COL) => costs)
end
CSV.write(costs_file, df)
println("Updated: $costs_file")

# Mean volume trajectory → MeanVolume.csv
mean_vol = vec(mean(vol_trajectories; dims=2))
vol_file = joinpath(out_dir, "MeanVolume.csv")
if isfile(vol_file)
    df_vol = CSV.read(vol_file, DataFrame; header=true)
    df_vol[!, STRICT_COL] = mean_vol
else
    df_vol = DataFrame(Symbol(STRICT_COL) => mean_vol)
end
CSV.write(vol_file, df_vol)
println("Updated: $vol_file")

# Mean thermal generation → MeanGeneration.csv
mean_gen = vec(mean(gen_trajectories; dims=2))
gen_file = joinpath(out_dir, "MeanGeneration.csv")
if isfile(gen_file)
    df_gen = CSV.read(gen_file, DataFrame; header=true)
    df_gen[!, STRICT_COL] = mean_gen
else
    df_gen = DataFrame(Symbol(STRICT_COL) => mean_gen)
end
CSV.write(gen_file, df_gen)
println("Updated: $gen_file")

# Full results → JLD2
results_dir = joinpath(out_dir, "results")
mkpath(results_dir)
results_file = joinpath(results_dir, "strict_rollout_$(num_eval_stages)stages_$(num_scenarios)scenarios.jld2")
jldsave(results_file;
    costs=costs,
    mean_cost=mean(costs),
    std_cost=std(costs),
    vol_trajectories=vol_trajectories,
    gen_trajectories=gen_trajectories,
    mean_vol=mean_vol,
    mean_gen=mean_gen,
    num_stages=num_eval_stages,
    num_scenarios=num_scenarios,
    model_path=model_path,
)
println("Saved: $results_file")
