# Paired TS-DDR strict rollout evaluation using pre-sampled scenario indices.
#
# Reads scenario indices from paired_scenario_indices.csv (126×100 matrix of
# integers in 1:nCen) so that the exact same inflow realizations can be fed to
# both this script and the SDDP Historical simulation.
#
# Usage:
#   julia --project -t auto eval_paired_tsddr.jl MODEL_PATH
#
# Environment overrides:
#   DR_NUM_EVAL_STAGES=96
#   DR_NUM_SCENARIOS=100
using DecisionRules
using Flux
using Statistics
using Random
using JuMP, DiffOpt, Ipopt
using JLD2
using CSV, DataFrames
using JSON
using DelimitedFiles

HydroPowerModels_dir = dirname(@__FILE__)
include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))
include(joinpath(HydroPowerModels_dir, "hydro_reachable_policy.jl"))

model_path = ARGS[1]
num_eval_stages = parse(Int, get(ENV, "DR_NUM_EVAL_STAGES", "96"))
num_scenarios = parse(Int, get(ENV, "DR_NUM_SCENARIOS", "100"))
layers = Int64[128, 128]

println("=" ^ 60)
println("Paired TS-DDR Strict Rollout Evaluation")
println("  Model:      $model_path")
println("  Stages:     $num_eval_stages")
println("  Scenarios:  $num_scenarios")
println("=" ^ 60)

# ── Load pre-sampled scenario indices ──────────────────────────────────────
indices_file = joinpath(HydroPowerModels_dir, "bolivia", "paired_scenario_indices.csv")
all_indices = Int.(readdlm(indices_file, ','))
@assert size(all_indices, 1) >= num_eval_stages
@assert size(all_indices, 2) >= num_scenarios
println("Loaded scenario indices: $(size(all_indices))")

# ── Build strict subproblems ───────────────────────────────────────────────
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
nCen = length(uncertainty_samples[1])
println("nHyd=$num_hydro, nCen=$nCen")
@assert all(1 .<= all_indices[1:num_eval_stages, 1:num_scenarios] .<= nCen) "Scenario indices out of range [1, $nCen]"

# ── Identify thermal generators ────────────────────────────────────────────
hydro_data = JSON.parsefile(joinpath(HydroPowerModels_dir, case_name, "hydro.json"))
power_data = JSON.parsefile(joinpath(HydroPowerModels_dir, case_name, "PowerModels.json"))
baseMVA = power_data["baseMVA"]
hydro_grid_idx = Set(hg["index_grid"] for hg in hydro_data["Hydrogenerators"])
num_gen = length(power_data["gen"])
thermal_idx = [i for i in 1:num_gen if !(i in hydro_grid_idx)]

volume_to_mw(volume; k=0.0036) = volume / k

pg_vars_per_stage = [DecisionRules.find_variables(subproblems[t], ["pg"]) for t in 1:num_eval_stages]

# ── Build policy and load weights ──────────────────────────────────────────
models = hydro_reachable_policy(hydro_meta, layers)
model_save = JLD2.load(model_path)
model_state = model_save["model_state"]
load_policy_weights!(models, model_state)
println("Loaded model weights from $model_path")

# ── Construct scenarios from pre-sampled indices ───────────────────────────
eval_scenarios = Vector{Vector{Vector{Tuple{eltype(uncertainty_samples[1][1][1][1]), eltype(uncertainty_samples[1][1][1][2])}}}}(undef, num_scenarios)
for s in 1:num_scenarios
    eval_scenarios[s] = [uncertainty_samples[t][all_indices[t, s]] for t in 1:num_eval_stages]
end

# Verify: print first scenario's first stage inflows
println("\nFirst scenario, stage 1 inflows:")
for (param, val) in eval_scenarios[1][1]
    println("  $(JuMP.name(param)) = $val")
end

# ── Run rollout evaluation ─────────────────────────────────────────────────
println("\nEvaluating $num_scenarios scenarios on $num_eval_stages stages...")

costs = Float64[]
vol_trajectories = zeros(num_eval_stages, num_scenarios)
gen_trajectories = zeros(num_eval_stages, num_scenarios)

for (i, scenario) in enumerate(eval_scenarios)
    Flux.reset!(models)

    state = Float64.(initial_state)
    scenario_cost = 0.0

    for t in 1:num_eval_stages
        for (j, param) in enumerate(state_params_in[t])
            set_parameter_value(param, state[j])
        end

        w_t = scenario[t]
        for (param, val) in w_t
            set_parameter_value(param, val)
        end

        w_vals = Float32.([val for (_, val) in w_t])
        x_hat = models(vcat(w_vals, Float32.(state)))

        for j in 1:num_hydro
            target_param = state_params_out[t][j][1]
            set_parameter_value(target_param, Float64(x_hat[j]))
        end

        optimize!(subproblems[t])
        scenario_cost += objective_value(subproblems[t])

        for j in 1:num_hydro
            state[j] = value(state_params_out[t][j][2])
        end

        vol_trajectories[t, i] = sum(volume_to_mw(state[j]) for j in 1:num_hydro)
        gen_trajectories[t, i] = sum(
            value(pg_vars_per_stage[t][j]) * baseMVA for j in thermal_idx
        )
    end

    push!(costs, scenario_cost)
    if i % 10 == 0 || i == num_scenarios
        println("  [$i/$num_scenarios] cost = $(round(scenario_cost; digits=1)), running mean = $(round(mean(costs); digits=1))")
    end
end

# ── Report results ─────────────────────────────────────────────────────────
println("\n" * "=" ^ 60)
println("Results: Paired TS-DDR Strict ($num_eval_stages stages, $num_scenarios scenarios)")
println("=" ^ 60)
println("  Mean cost:   $(round(mean(costs); digits=1))")
println("  Std:         $(round(std(costs); digits=1))")
println("  Min:         $(round(minimum(costs); digits=1))")
println("  Max:         $(round(maximum(costs); digits=1))")
println("  Median:      $(round(median(costs); digits=1))")
println("  Violation:   0.0% (strict mode)")
println("=" ^ 60)

# ── Save results ───────────────────────────────────────────────────────────
out_dir = joinpath(HydroPowerModels_dir, case_name, formulation)

const COL_NAME = "TS-DDR (strict, paired)"
costs_file = joinpath(out_dir, "paired_costs.csv")
df = DataFrame(Symbol(COL_NAME) => costs)
CSV.write(costs_file, df)
println("Saved: $costs_file")

mean_vol = vec(mean(vol_trajectories; dims=2))
vol_file = joinpath(out_dir, "paired_MeanVolume.csv")
df_vol = DataFrame(Symbol(COL_NAME) => mean_vol)
CSV.write(vol_file, df_vol)
println("Saved: $vol_file")

mean_gen = vec(mean(gen_trajectories; dims=2))
gen_file = joinpath(out_dir, "paired_MeanGeneration.csv")
df_gen = DataFrame(Symbol(COL_NAME) => mean_gen)
CSV.write(gen_file, df_gen)
println("Saved: $gen_file")

results_dir = joinpath(out_dir, "results")
mkpath(results_dir)
results_file = joinpath(results_dir, "paired_strict_rollout.jld2")
jldsave(results_file;
    costs=costs,
    vol_trajectories=vol_trajectories,
    gen_trajectories=gen_trajectories,
    scenario_indices=all_indices[1:num_eval_stages, 1:num_scenarios],
)
println("Saved: $results_file")
