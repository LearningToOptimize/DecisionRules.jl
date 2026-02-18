# Train DecisionRules.jl policy for Atlas Robot Balancing
# Using Deterministic Equivalent Formulation
#
# This script trains a neural network policy using the deterministic equivalent
# formulation (single large optimization problem) instead of decomposed subproblems.

using Flux
using DecisionRules
using Random
using Statistics
using LinearAlgebra
using JuMP
import Ipopt, HSL_jll
using Wandb, Dates, Logging
using JLD2
using DiffOpt

Atlas_dir = dirname(@__FILE__)
include(joinpath(Atlas_dir, "build_atlas_problem.jl"))

# ============================================================================
# Parameters
# ============================================================================

# Problem parameters
N = 10                          # Number of time steps
h = 0.01                        # Time step
perturbation_scale = 0.5       # Scale of random perturbations
num_scenarios = 10              # Number of uncertainty samples per stage
penalty = 10.0                   # Penalty for state deviation
perturbation_frequency = 1000      # Frequency of perturbations (every k stages)

# Training parameters
num_epochs = 10
num_batches = 10
num_train_per_batch = 1
layers = Int64[64, 64]
activation = sigmoid
optimizers = [Flux.Adam(0.001)]

# Initial-state augmentation via short policy rollouts
# If enabled, each eligible epoch starts training from a reachable state obtained
# by rolling out the current policy for a small random number of stages.
enable_rollout_initial_state_augmentation = true
rollout_start_epoch = 1
rollout_every_epochs = 1
rollout_max_horizon_fraction = 20.0

if enable_rollout_initial_state_augmentation
    rollout_every_epochs < 1 && error("rollout_every_epochs must be >= 1")
    (rollout_max_horizon_fraction <= 0) &&
        error("rollout_max_horizon_fraction must be in (0, 1]")
end

# Save paths
model_dir = joinpath(Atlas_dir, "models")
mkpath(model_dir)
save_file = "atlas-balancing-deteq-N$(N)-$(now())"
# Warm start options:
# - set to `nothing` to train from random initialization
# - set to "latest" to use the most recent deterministic-equivalent model in `model_dir`
# - set to a run name (with or without `.jld2`) or a full path
#   e.g. "atlas-balancing-deteq-N50-2026-02-02T21:16:37.554"
warmstart_model = "atlas-balancing-deteq-N10-2026-02-15T19:49:47.739"
# CLI override:
# julia --project=. examples/Atlas/train_dr_atlas_det_eq.jl <warmstart_model>
if !isempty(ARGS)
    warmstart_model = ARGS[1]
end

# ============================================================================
# Build Deterministic Equivalent Problem
# ============================================================================

println("Building Atlas deterministic equivalent problem...")

# Build one subproblem set dedicated to deterministic-equivalent construction.
# `deterministic_equivalent!` mutates this data.
@time subproblems_det, state_params_in_det, state_params_out_det, initial_state, uncertainty_samples_det_builder,
      _, _, x_ref, u_ref, atlas = build_atlas_subproblems(;
    N = N,
    h = h,
    perturbation_scale = perturbation_scale,
    num_scenarios = num_scenarios,
    penalty = penalty,
    perturbation_frequency = perturbation_frequency,
)

# Build a second, independent subproblem set used only for rollout-based initial-state generation.
@time rollout_subproblems, rollout_state_params_in, rollout_state_params_out, rollout_initial_state, rollout_uncertainty_samples,
      _, _, _, _, _ = build_atlas_subproblems(;
    N = N,
    h = h,
    perturbation_scale = perturbation_scale,
    num_scenarios = num_scenarios,
    penalty = penalty,
    perturbation_frequency = perturbation_frequency,
)

# Build deterministic equivalent
det_equivalent = DiffOpt.nonlinear_diff_model(optimizer_with_attributes(Ipopt.Optimizer, 
    "print_level" => 0,
    "hsllib" => HSL_jll.libhsl_path,
    "linear_solver" => "ma97",
    # "mu_target" => 1e-8,
))

# Convert subproblems to deterministic equivalent using DecisionRules
det_equivalent, uncertainty_samples_det = DecisionRules.deterministic_equivalent!(
    det_equivalent, 
    subproblems_det, 
    state_params_in_det, 
    state_params_out_det, 
    initial_state, 
    uncertainty_samples_det_builder
)

nx = atlas.nx
nu = atlas.nu
n_perturb = length(rollout_uncertainty_samples[1])  # Number of perturbation parameters

println("Atlas state dimension: $nx")
println("Atlas control dimension: $nu")
println("Number of perturbations: $n_perturb")
println("Number of stages: $(N-1)")

# ============================================================================
# Logging
# ============================================================================

lg = WandbLogger(
    project = "DecisionRules-Atlas",
    name = save_file,
    config = Dict(
        "N" => N,
        "h" => h,
        "perturbation_scale" => perturbation_scale,
        "num_scenarios" => num_scenarios,
        "penalty" => penalty,
        "layers" => layers,
        "activation" => string(activation),
        "optimizer" => string(optimizers),
        "nx" => nx,
        "nu" => nu,
        "formulation" => "deterministic_equivalent",
        "warmstart_model" => isnothing(warmstart_model) ? "none" : string(warmstart_model),
        "enable_rollout_initial_state_augmentation" => enable_rollout_initial_state_augmentation,
        "rollout_start_epoch" => rollout_start_epoch,
        "rollout_every_epochs" => rollout_every_epochs,
        "rollout_max_horizon_fraction" => rollout_max_horizon_fraction,
    )
)

function record_loss(iter, model, loss, tag)
    Wandb.log(lg, Dict(tag => loss))
    return false
end

# ============================================================================
# Define Neural Network Policy
# ============================================================================

# Policy architecture: LSTM processes perturbations, Dense combines with previous state
# This design is memory-efficient and allows the LSTM to focus on temporal patterns
n_uncertainties = length(rollout_uncertainty_samples[1])
models = state_conditioned_policy(n_uncertainties, nx, nx, layers; 
                                   activation=activation, encoder_type=Flux.LSTM)

function resolve_warmstart_path(warmstart_model, model_dir)
    isnothing(warmstart_model) && return nothing

    if warmstart_model == "latest"
        model_files = filter(
            f -> endswith(f, ".jld2") && startswith(f, "atlas-balancing-deteq"),
            readdir(model_dir),
        )
        isempty(model_files) && return nothing
        model_files_full = [joinpath(model_dir, f) for f in model_files]
        return model_files_full[argmax([mtime(f) for f in model_files_full])]
    end

    candidates = String[warmstart_model]
    push!(candidates, joinpath(model_dir, warmstart_model))
    if !endswith(warmstart_model, ".jld2")
        push!(candidates, warmstart_model * ".jld2")
        push!(candidates, joinpath(model_dir, warmstart_model * ".jld2"))
    end

    for candidate in candidates
        isfile(candidate) && return candidate
    end
    return nothing
end

warmstart_model_path = resolve_warmstart_path(warmstart_model, model_dir)
if isnothing(warmstart_model)
    println("Warm start: disabled (random initialization)")
elseif isnothing(warmstart_model_path)
    error("Warm start model not found: $(warmstart_model)")
else
    model_data = JLD2.load(warmstart_model_path)
    if haskey(model_data, "model_state")
        Flux.loadmodel!(models, normalize_recur_state(model_data["model_state"]))
        println("Warm start: loaded model weights from $warmstart_model_path")
    else
        error("Warm start model is missing `model_state`: $warmstart_model_path")
    end
end

println("Model architecture: StateConditionedPolicy")
println("  Encoder (LSTM): $n_uncertainties -> $(layers)")
println("  Combiner (Dense): $(layers[end]) + $nx -> $nx")

# ============================================================================
# Initial Evaluation
# ============================================================================

println("\nEvaluating initial policy...")
Random.seed!(8788)
objective_values = [simulate_multistage(
    det_equivalent, state_params_in_det, state_params_out_det, 
    initial_state, DecisionRules.sample(uncertainty_samples_det), 
    models;
) for _ in 1:2]
best_obj = mean(objective_values)
println("Initial objective: $best_obj")

# for testing visualization. fill x with visited states
# X[2:end] = [value.([var[2] for var in stage]) for stage in state_params_out_det]
# calculate distance from reference
# dist = sum((X[t][i] - x_ref[i])^2 for i in 1:length(x_ref) for t in 1:length(X))

model_path = joinpath(model_dir, save_file * ".jld2")
save_control = SaveBest(best_obj, model_path)

# ============================================================================
# Hyperparameter Adjustment
# ============================================================================

adjust_hyperparameters = (iter, opt_state, num_train_per_batch) -> begin
    if iter % 500 == 0
        num_train_per_batch = min(num_train_per_batch * 2, 8)
    end
    return num_train_per_batch
end

function rollout_reachable_initial_state(
    model,
    nominal_initial_state::Vector{Float64},
    subproblems::Vector{JuMP.Model},
    state_params_in,
    state_params_out,
    uncertainty_samples,
    rollout_steps::Int,
)
    uncertainty_sample = DecisionRules.sample(uncertainty_samples)
    max_stages = min(rollout_steps, length(subproblems), length(uncertainty_sample))
    state_in = copy(nominal_initial_state)
    Flux.reset!(model)

    for stage in 1:max_stages
        uncertainty_stage = uncertainty_sample[stage]
        uncertainty_vec = [u[2] for u in uncertainty_stage]
        state_out_target = Float64.(model(vcat(uncertainty_vec, state_in)))
        DecisionRules.simulate_stage(
            subproblems[stage],
            state_params_in[stage],
            state_params_out[stage],
            uncertainty_stage,
            state_in,
            state_out_target,
        )
        state_in = Float64.(DecisionRules.get_next_state(
            subproblems[stage],
            state_params_in[stage],
            state_params_out[stage],
            state_in,
            state_out_target,
        ))
    end

    return state_in
end

# ============================================================================
# Training
# ============================================================================

println("\nStarting training with deterministic equivalent...")
println("Epochs: $num_epochs, Batches per epoch: $num_batches")

for epoch in 1:num_epochs
    println("\n=== Epoch $epoch ===")
    _num_train_per_batch = num_train_per_batch
    epoch_initial_state = copy(initial_state)

    if enable_rollout_initial_state_augmentation &&
       epoch >= rollout_start_epoch &&
       ((epoch - rollout_start_epoch) % rollout_every_epochs == 0)
        max_rollout_steps = max(1, floor(Int, rollout_max_horizon_fraction * (N - 1)))
        rollout_steps = rand(1:max_rollout_steps)
        try
            epoch_initial_state = rollout_reachable_initial_state(
                models,
                rollout_initial_state,
                rollout_subproblems,
                rollout_state_params_in,
                rollout_state_params_out,
                rollout_uncertainty_samples,
                rollout_steps,
            )
            rollout_state_shift = norm(epoch_initial_state .- initial_state)
            println("  Rollout augmentation active: steps=$rollout_steps, ||x0_epoch - x0_nominal||₂=$(round(rollout_state_shift, digits=4))")
            Wandb.log(lg, Dict(
                "metrics/rollout_steps" => rollout_steps,
                "metrics/rollout_state_shift_l2" => rollout_state_shift,
            ))
        catch err
            @warn "Rollout initial-state update failed; falling back to nominal initial_state." exception=(err, catch_backtrace())
            epoch_initial_state = copy(initial_state)
        end
    end
    
    train_multistage(
        models, 
        epoch_initial_state, 
        det_equivalent, 
        state_params_in_det, 
        state_params_out_det, 
        uncertainty_samples_det;
        num_batches = num_batches,
        num_train_per_batch = _num_train_per_batch,
        optimizer = optimizers[min(epoch, length(optimizers))],
        record_loss = (iter, model, loss, tag) -> begin
            if tag == "metrics/training_loss"
                save_control(iter, model, loss)
            end
            return record_loss(iter, model, loss, tag)
        end,
        adjust_hyperparameters = adjust_hyperparameters
    )
end

# ============================================================================
# Final Evaluation
# ============================================================================

println("\n=== Final Evaluation ===")
Random.seed!(8788)
objective_values = [simulate_multistage(
    det_equivalent, state_params_in_det, state_params_out_det, 
    initial_state, DecisionRules.sample(uncertainty_samples_det), 
    models;
) for _ in 1:10]

println("Final objective: $(mean(objective_values)) ± $(std(objective_values))")
println("Best objective during training: $(save_control.best_loss)")

# Finish logging
close(lg)

println("\nModel saved to: $model_path")
println("Training complete!")
