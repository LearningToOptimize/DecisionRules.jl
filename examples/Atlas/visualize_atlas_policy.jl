# Visualize Trained Atlas Balancing Policies
#
# This script loads trained policies and visualizes their performance
# on the Atlas robot balancing task with stochastic perturbations.

using Flux
using DecisionRules
using Random
using Statistics
using JuMP
import Ipopt, HSL_jll
using JLD2
using LinearAlgebra
using Plots

Atlas_dir = dirname(@__FILE__)
include(joinpath(Atlas_dir, "build_atlas_problem.jl"))
include(joinpath(Atlas_dir, "atlas_utils.jl"))
include(joinpath(Atlas_dir, "atlas_visualization.jl"))

# ============================================================================
# Configuration
# ============================================================================

# Model to load (modify this path to your trained model)
model_path = "./models/atlas-balancing-deteq-N10-2026-02-15T19:49:47.739.jld2"  # Set to path of trained model, or nothing to use latest

# Problem parameters (should match training)
N = 300                          # Number of time steps
h = 0.01                        # Time step  
perturbation_scale = 1.5       # Scale of random perturbations
num_scenarios = 1              # Number of scenarios to simulate
perturbation_frequency = 1000     # Frequency of perturbations (every k stages)

# Visualization options
animate_robot = true            # Whether to animate in MeshCat
save_plots = true               # Whether to save plots to file
show_perturbation_cause_in_meshcat = true
meshcat_cause_arrow_scale = 2.0
meshcat_cause_show_threshold = 1e-6
meshcat_cause_linger_seconds = 2.0
meshcat_cause_min_arrow_length = 0.40
meshcat_cause_shaft_radius = 0.08
meshcat_cause_impact_distance = 0.18
meshcat_cause_retreat_distance = 0.35

# ============================================================================
# Load Model
# ============================================================================

# Find latest model if not specified
if isnothing(model_path)
    model_dir = joinpath(Atlas_dir, "models")
    if isdir(model_dir)
        model_files = filter(f -> endswith(f, ".jld2") && startswith(f, "atlas"), readdir(model_dir))
        if !isempty(model_files)
            # Sort by modification time and get latest
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

println("\nSetting up Atlas problem...")
atlas = Atlas()

# Load reference state
@load joinpath(Atlas_dir, "atlas_ref.jld2") x_ref u_ref

nx = atlas.nx
nu = atlas.nu
perturbation_idx = atlas.nq + 5

println("State dimension: $nx")
println("Control dimension: $nu")

# Build subproblems for simulation
@time subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples,
      X_vars, U_vars, _, _, _ = build_atlas_subproblems(;
    atlas = atlas,
    x_ref = x_ref,
    u_ref = u_ref,
    N = N,
    h = h,
    perturbation_scale = perturbation_scale,
    perturbation_indices = [perturbation_idx],
    num_scenarios = num_scenarios,
    perturbation_frequency = perturbation_frequency,
)

# ============================================================================
# Load or Create Policy
# ============================================================================

# Define model architecture (must match training)
layers = [64, 64]
activation = sigmoid

n_uncertainties = length(uncertainty_samples[1])
models = state_conditioned_policy(n_uncertainties, nx, nx, layers; 
    activation=activation, encoder_type=Flux.LSTM
)

if !use_random_policy
    # Load trained weights
    model_data = JLD2.load(model_path)
    if haskey(model_data, "model_state")
        Flux.loadmodel!(models, normalize_recur_state(model_data["model_state"]))
        println("Loaded trained model weights")
    else
        println("Warning: Could not find model_state in file, using random weights")
    end
end

# ============================================================================
# Simulate Multiple Scenarios
# ============================================================================

println("\nSimulating $num_scenarios scenarios...")

# Storage for trajectories
all_states = Vector{Vector{Vector{Float64}}}(undef, num_scenarios)
all_objectives = Vector{Float64}(undef, num_scenarios)
all_perturbations = Vector{Vector{Float64}}(undef, num_scenarios)

for s in 1:num_scenarios
    Random.seed!(s * 100 + 42)
    
    # Sample perturbations
    perturbation_sample = DecisionRules.sample(uncertainty_samples)
    
    # Record perturbations for this scenario (first perturbation per stage)
    all_perturbations[s] = [stage_u[1][2] for stage_u in perturbation_sample]
    
    # Simulate using the policy
    try
        obj = simulate_multistage(
            subproblems, state_params_in, state_params_out,
            initial_state, perturbation_sample,
            models
        )
        all_objectives[s] = obj
        
        # Extract state trajectory from solved subproblems
        states = [copy(initial_state)]
        for t in 1:N-1
            x_t = [value(X_vars[t][i]) for i in 1:nx]
            push!(states, x_t)
        end
        all_states[s] = states
        
        println("Scenario $s: objective = $(round(obj, digits=4))")
    catch e
        println("Scenario $s: FAILED - $e")
        all_objectives[s] = NaN
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
# Plot Results
# ============================================================================

println("\nGenerating plots...")

# Time vector
t_vec = (0:N-1) .* h

# Plot 1: State trajectories (position states - first nq components)
nq = atlas.nq
p1 = plot(title="Position States Over Time", xlabel="Time (s)", ylabel="Position", legend=:outerright)
colors = palette(:tab10)

# Plot mean and std for a few representative position states
representative_states = [1, 5, 10, min(15, nq)]  # Select a few states to plot
for (idx, state_idx) in enumerate(representative_states)
    if state_idx <= nq
        # Extract this state across all scenarios
        state_traj = zeros(length(valid_scenarios), N)
        for (si, s) in enumerate(valid_scenarios)
            for t in 1:min(N, length(all_states[s]))
                state_traj[si, t] = all_states[s][t][state_idx]
            end
        end
        
        mean_traj = vec(mean(state_traj, dims=1))
        std_traj = vec(std(state_traj, dims=1))
        
        plot!(p1, t_vec, mean_traj, ribbon=std_traj, fillalpha=0.3, 
              label="q[$state_idx]", color=colors[idx])
        hline!(p1, [x_ref[state_idx]], linestyle=:dash, color=colors[idx], 
               label="", alpha=0.5)
    end
end

# Plot 2: Velocity states (second nq components)  
p2 = plot(title="Velocity States Over Time", xlabel="Time (s)", ylabel="Velocity", legend=:outerright)

for (idx, state_idx) in enumerate(representative_states)
    vel_idx = nq + state_idx
    if vel_idx <= nx
        state_traj = zeros(length(valid_scenarios), N)
        for (si, s) in enumerate(valid_scenarios)
            for t in 1:min(N, length(all_states[s]))
                state_traj[si, t] = all_states[s][t][vel_idx]
            end
        end
        
        mean_traj = vec(mean(state_traj, dims=1))
        std_traj = vec(std(state_traj, dims=1))
        
        plot!(p2, t_vec, mean_traj, ribbon=std_traj, fillalpha=0.3,
              label="v[$state_idx]", color=colors[idx])
        hline!(p2, [x_ref[vel_idx]], linestyle=:dash, color=colors[idx],
               label="", alpha=0.5)
    end
end

# Plot 3: State deviation from reference
p3 = plot(title="State Deviation from Reference", xlabel="Time (s)", 
          ylabel="||x - x_ref||", legend=:topright)

for s in valid_scenarios[1:min(5, length(valid_scenarios))]
    deviations = [norm(all_states[s][t] - x_ref) for t in 1:min(N, length(all_states[s]))]
    plot!(p3, t_vec[1:length(deviations)], deviations, alpha=0.7, label="Scenario $s")
end

# Mean deviation
if !isempty(valid_scenarios)
    mean_dev = zeros(N)
    for t in 1:N
        devs = [norm(all_states[s][t] - x_ref) for s in valid_scenarios if t <= length(all_states[s])]
        mean_dev[t] = isempty(devs) ? 0.0 : mean(devs)
    end
    plot!(p3, t_vec, mean_dev, linewidth=3, color=:black, label="Mean")
end

# Plot 4: Perturbations over time
p4 = plot(title="Applied Perturbations", xlabel="Time (s)", 
          ylabel="Perturbation", legend=:topright)

for s in valid_scenarios[1:min(5, length(valid_scenarios))]
    plot!(p4, t_vec[1:end-1], all_perturbations[s], alpha=0.7, label="Scenario $s")
end

# Combine plots
combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))

if save_plots
    plot_dir = joinpath(Atlas_dir, "plots")
    mkpath(plot_dir)
    plot_path = joinpath(plot_dir, "atlas_policy_evaluation.png")
    savefig(combined_plot, plot_path)
    println("Saved plot to: $plot_path")
end

display(combined_plot)

# ============================================================================
# Objective Distribution Plot
# ============================================================================

p_obj = histogram(all_objectives[valid_scenarios], bins=20, 
                  title="Objective Value Distribution",
                  xlabel="Objective Value", ylabel="Count",
                  legend=false, alpha=0.7)
vline!(p_obj, [mean(all_objectives[valid_scenarios])], linewidth=2, 
       color=:red, linestyle=:dash)

if save_plots
    obj_plot_path = joinpath(Atlas_dir, "plots", "atlas_objective_distribution.png")
    savefig(p_obj, obj_plot_path)
    println("Saved objective distribution to: $obj_plot_path")
end

display(p_obj)

# ============================================================================
# MeshCat Animation
# ============================================================================

if animate_robot
    println("\nSetting up MeshCat visualizer...")
    vis = Visualizer()
    mvis = init_visualizer(atlas, vis)
    
    # Animate best scenario
    if !isempty(valid_scenarios)
        best_scenario = valid_scenarios[argmin(all_objectives[valid_scenarios])]
        println("Animating best scenario (scenario $best_scenario)...")
        
        # Convert to format expected by animate!
        X_animate = all_states[best_scenario]

        if show_perturbation_cause_in_meshcat
            animate_with_perturbation_cause!(
                atlas,
                mvis,
                X_animate,
                all_perturbations[best_scenario];
                Δt = h,
                arrow_scale = meshcat_cause_arrow_scale,
                show_threshold = meshcat_cause_show_threshold,
                linger_seconds = meshcat_cause_linger_seconds,
                min_arrow_length = meshcat_cause_min_arrow_length,
                shaft_radius = meshcat_cause_shaft_radius,
                perturbation_state_index = perturbation_idx,
                impact_distance = meshcat_cause_impact_distance,
                retreat_distance = meshcat_cause_retreat_distance,
            )
            println("MeshCat overlay: collision-style perturbation cause enabled (impactor appears at contact and retreats over linger window).")
        else
            animate!(atlas, mvis, X_animate, Δt=h)
        end
        
        println("\nAnimation ready! Open MeshCat visualizer to view.")
        println("Best scenario objective: $(all_objectives[best_scenario])")
    end
end

# ============================================================================
# Compare with Open-Loop (No Policy)
# ============================================================================

println("\n" * "="^60)
println("Comparison: Policy vs Open-Loop Control")
println("="^60)

# Simulate open-loop (just use reference control, no feedback)
openloop_objectives = Float64[]
openloop_final_deviations = Float64[]

for s in 1:num_scenarios
    Random.seed!(s * 100 + 42)
    
    X_openloop = [copy(x_ref) for _ in 1:N]
    
    for k in 1:N-1
        # Integrate with reference control
        X_openloop[k+1] = rk4(atlas, X_openloop[k], u_ref, h)
        # Add same perturbation
        X_openloop[k+1][perturbation_idx] += all_perturbations[s][k]
    end
    
    # Compute "objective" as sum of squared deviations
    obj = sum(sum((X_openloop[t] .- x_ref).^2) for t in 2:N)
    push!(openloop_objectives, obj)
    push!(openloop_final_deviations, norm(X_openloop[end] - x_ref))
end

println("\nOpen-Loop Control:")
println("  Mean objective: $(round(mean(openloop_objectives), digits=4))")
println("  Std objective: $(round(std(openloop_objectives), digits=4))")
println("  Mean final deviation: $(round(mean(openloop_final_deviations), digits=4))")

if !isempty(valid_scenarios)
    policy_final_devs = [norm(all_states[s][end] - x_ref) for s in valid_scenarios]
    
    println("\nTrained Policy:")
    println("  Mean objective: $(round(mean(all_objectives[valid_scenarios]), digits=4))")
    println("  Std objective: $(round(std(all_objectives[valid_scenarios]), digits=4))")
    println("  Mean final deviation: $(round(mean(policy_final_devs), digits=4))")
    
    improvement = (mean(openloop_objectives) - mean(all_objectives[valid_scenarios])) / mean(openloop_objectives) * 100
    println("\nImprovement: $(round(improvement, digits=2))%")
end

println("\nVisualization complete!")
