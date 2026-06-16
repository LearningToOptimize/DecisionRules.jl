# Test Atlas Balancing Problem Setup
#
# This script tests the stochastic Atlas balancing problem before training.

using JLD2
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "build_atlas_problem.jl"))
include(joinpath(@__DIR__, "atlas_utils.jl"))

# Setup visualization
atlas = Atlas()
vis = Visualizer()
mvis = init_visualizer(atlas, vis)

# Load reference
@load joinpath(@__DIR__, "atlas_ref.jld2") x_ref u_ref
visualize!(atlas, mvis, x_ref)

println("Atlas robot loaded successfully!")
println("State dimension (nx): $(atlas.nx)")
println("Control dimension (nu): $(atlas.nu)")
println("Number of joints: $(atlas.nq)")

# ============================================================================
# Test 1: Simulate with random perturbations (open loop)
# ============================================================================

println("\n=== Test 1: Open-loop simulation with perturbations ===")

h = 0.01
N = 100
perturbation_scale = 0.05
perturbation_idx = atlas.nq + 5  # Perturb one velocity state

Random.seed!(42)

X_openloop = [zeros(atlas.nx) for _ in 1:N]
X_openloop[1] = copy(x_ref)

for k in 1:N-1
    # Integrate dynamics
    X_openloop[k+1] = rk4(atlas, X_openloop[k], u_ref, h)
    
    # Add random perturbation
    X_openloop[k+1][perturbation_idx] += perturbation_scale * randn()
end

# Check final state deviation
final_deviation = norm(X_openloop[end] - x_ref)
println("Final state deviation from reference: $final_deviation")

# Animate
println("Animating open-loop trajectory (check MeshCat visualizer)...")
animate!(atlas, mvis, X_openloop, Δt=h)

# ============================================================================
# Test 2: Build subproblems and test one
# ============================================================================

println("\n=== Test 2: Build and test subproblems ===")

@time subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples,
      X_vars, U_vars, _, _, _ = build_atlas_subproblems(;
    atlas = atlas,
    x_ref = x_ref,
    u_ref = u_ref,
    N = 10,  # Small for testing
    h = h,
    perturbation_scale = perturbation_scale,
    perturbation_indices = [perturbation_idx],
    num_scenarios = 5,
)

println("Built $(length(subproblems)) subproblems")
println("State params in: $(length(state_params_in[1])) parameters per stage")
println("State params out: $(length(state_params_out[1])) outputs per stage")
println("Uncertainty samples: $(length(uncertainty_samples[1])) uncertain parameters per stage")

# Test solving one subproblem
println("\nSolving first subproblem...")
using JuMP
optimize!(subproblems[1])
println("Termination status: $(termination_status(subproblems[1]))")
println("Objective value: $(objective_value(subproblems[1]))")

# ============================================================================
# Test 3: Simulate using DecisionRules with a dummy policy
# ============================================================================

println("\n=== Test 3: Simulate with dummy (identity) policy ===")

using DecisionRules
using Flux

# Simple policy that outputs the reference state
dummy_policy = Chain(
    x -> x_ref  # Always output reference
)

# More realistic: small perturbation from input
identity_policy = Chain(
    Dense(atlas.nx, atlas.nx, identity, bias=false)  # Linear layer
)

# Initialize to identity
Flux.loadparams!(identity_policy, [Matrix{Float32}(I, atlas.nx, atlas.nx)])

println("Testing simulation with identity policy...")
Random.seed!(123)

try
    obj_value = simulate_multistage(
        subproblems, state_params_in, state_params_out,
        initial_state, DecisionRules.sample(uncertainty_samples),
        identity_policy
    )
    println("Simulation completed! Objective: $obj_value")
catch e
    println("Simulation failed with error: $e")
    println("\nThis is expected if the problem setup needs adjustment.")
end

# ============================================================================
# Test 4: Multiple scenarios
# ============================================================================

println("\n=== Test 4: Multiple scenario simulation ===")

n_test_scenarios = 5
objectives = Float64[]

for s in 1:n_test_scenarios
    Random.seed!(s * 100)
    try
        obj = simulate_multistage(
            subproblems, state_params_in, state_params_out,
            initial_state, DecisionRules.sample(uncertainty_samples),
            identity_policy
        )
        push!(objectives, obj)
        println("Scenario $s: objective = $obj")
    catch e
        println("Scenario $s: failed - $e")
    end
end

if !isempty(objectives)
    println("\nMean objective: $(mean(objectives))")
    println("Std objective: $(std(objectives))")
end

println("\n=== Tests Complete ===")
println("If all tests passed, you can proceed with training using train_dr_atlas.jl")
