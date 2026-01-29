# Visualize Trained Atlas Balancing Policy (Deterministic Equivalent)
#
# This script loads a trained policy and animates its performance on the Atlas
# balancing task using the deterministic equivalent formulation.

using Flux
using DecisionRules
using Random
using Statistics
using JuMP
import Ipopt, HSL_jll
using JLD2
using DiffOpt

Atlas_dir = dirname(@__FILE__)
include(joinpath(Atlas_dir, "build_atlas_problem.jl"))
include(joinpath(Atlas_dir, "atlas_utils.jl"))
include(joinpath(Atlas_dir, "atlas_visualization.jl"))

# ============================================================================
# Configuration
# ============================================================================

# Model to load (modify this path to your trained model)
model_path = nothing  # Set to path of trained model, or nothing to use latest

# Problem parameters (should match training)
N = 50                          # Number of time steps
h = 0.01                        # Time step
perturbation_scale = 0.05       # Scale of random perturbations
num_scenarios = 10              # Number of scenarios to simulate
penalty = 1e3                   # Penalty for state deviation
perturbation_frequency = 5      # Frequency of perturbations (every k stages)

# Visualization options
animate_robot = true            # Whether to animate in MeshCat

# ============================================================================
# Load Model
# ============================================================================

if isnothing(model_path)
    model_dir = joinpath(Atlas_dir, "models")
    if isdir(model_dir)
        model_files = filter(
            f -> endswith(f, ".jld2") && startswith(f, "atlas-balancing-deteq"),
            readdir(model_dir),
        )
        if !isempty(model_files)
            model_files_full = [joinpath(model_dir, f) for f in model_files]
            model_path = model_files_full[argmax([mtime(f) for f in model_files_full])]
            println("Using latest model: $model_path")
        end
    end
end

if isnothing(model_path) || !isfile(model_path)
    println("No trained model found. Creating a random policy for visualization.")
    use_random_policy = true
else
    use_random_policy = false
    println("Loading model from: $model_path")
end

# ============================================================================
# Setup Problem
# ============================================================================

println("\nSetting up Atlas problem (deterministic equivalent)...")
atlas = Atlas()
@load joinpath(Atlas_dir, "atlas_ref.jld2") x_ref u_ref

nx = atlas.nx
nu = atlas.nu

@time subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples,
      _, _, _, _, _ = build_atlas_subproblems(;
    atlas = atlas,
    x_ref = x_ref,
    u_ref = u_ref,
    N = N,
    h = h,
    perturbation_scale = perturbation_scale,
    num_scenarios = num_scenarios,
    penalty = penalty,
    perturbation_frequency = perturbation_frequency,
)

det_equivalent = DiffOpt.diff_model(optimizer_with_attributes(Ipopt.Optimizer,
    "print_level" => 0,
    "hsllib" => HSL_jll.libhsl_path,
    "linear_solver" => "ma97",
))

det_equivalent, uncertainty_samples_det = DecisionRules.deterministic_equivalent!(
    det_equivalent,
    subproblems,
    state_params_in,
    state_params_out,
    initial_state,
    uncertainty_samples,
)

println("Atlas state dimension: $nx")
println("Atlas control dimension: $nu")
println("Number of stages: $(N - 1)")

# ============================================================================
# Load or Create Policy
# ============================================================================

layers = [64, 64]
activation = sigmoid

n_uncertainties = length(uncertainty_samples_det[1])
models = state_conditioned_policy(n_uncertainties, nx, nx, layers;
                                   activation=activation, encoder_type=Flux.LSTM)

if !use_random_policy
    model_data = JLD2.load(model_path)
    if haskey(model_data, "model_state")
        Flux.loadmodel!(models, model_data["model_state"])
        println("Loaded trained model weights")
    else
        println("Warning: Could not find model_state in file, using random weights")
    end
end

# ============================================================================
# Simulate Multiple Scenarios
# ============================================================================

println("\nSimulating $num_scenarios scenarios...")

all_states = Vector{Vector{Vector{Float64}}}(undef, num_scenarios)
all_objectives = fill(NaN, num_scenarios)

for s in 1:num_scenarios
    Random.seed!(s * 100 + 42)
    Flux.reset!(models)

    perturbation_sample = DecisionRules.sample(uncertainty_samples_det)

    try
        obj = simulate_multistage(
            det_equivalent, state_params_in, state_params_out,
            initial_state, perturbation_sample,
            models
        )
        all_objectives[s] = obj

        states = Vector{Vector{Float64}}(undef, N)
        states[1] = copy(initial_state)
        for t in 1:N-1
            states[t + 1] = [value(state_params_out[t][i][2]) for i in 1:nx]
        end
        all_states[s] = states

        println("Scenario $s: objective = $(round(obj, digits=4))")
    catch e
        println("Scenario $s: FAILED - $e")
        all_states[s] = [copy(initial_state) for _ in 1:N]
    end
end

valid_scenarios = findall(!isnan, all_objectives)
println("\nSuccessful scenarios: $(length(valid_scenarios))/$num_scenarios")
if !isempty(valid_scenarios)
    println("Mean objective: $(round(mean(all_objectives[valid_scenarios]), digits=4))")
    println("Std objective: $(round(std(all_objectives[valid_scenarios]), digits=4))")
end

# ============================================================================
# MeshCat Animation
# ============================================================================

if animate_robot
    println("\nSetting up MeshCat visualizer...")
    vis = Visualizer()
    mvis = init_visualizer(atlas, vis)

    if !isempty(valid_scenarios)
        best_scenario = valid_scenarios[argmin(all_objectives[valid_scenarios])]
        println("Animating best scenario (scenario $best_scenario)...")

        X_animate = all_states[best_scenario]
        animate!(atlas, mvis, X_animate, Δt=h)

        println("\nAnimation ready! Open MeshCat visualizer to view.")
        println("Best scenario objective: $(all_objectives[best_scenario])")
    end
end

println("\nVisualization complete!")
