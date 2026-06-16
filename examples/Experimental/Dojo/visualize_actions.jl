# Load and visualize saved quadruped actions
# This script replays the saved actions in the environment and visualizes the trajectory

using JLD2
using Dojo
using DojoEnvironments
using Random

# Load the saved actions
data = jldopen("quadruped_actions.jld2", "r")
actions_taken = data["actions"]
test_return = data["test_return"]
final_x = data["final_x"]
close(data)

println("Loaded $(length(actions_taken)) actions")
println("Test return: $test_return")
println("Final x position: $final_x")

# Reconstruct the environment and replay the actions
include("quadruped_env.jl")

println("\n=== Replaying actions ===")
replay_env = QuadrupedEnv(; record=true, rng=MersenneTwister(456))

reset!(replay_env)

for (i, a) in enumerate(actions_taken)
    println("Step $i: action $a")
    replay_env(a)
    
    if is_terminated(replay_env)
        println("Episode terminated at step $i")
        break
    end
end

println("\nReplayed x position: $(replay_env.state[1])")

# Visualize
println("\nVisualizing replay trajectory...")
try
    vis = DojoEnvironments.visualize(replay_env.dojo_env)
    DojoEnvironments.render(vis)
    println("Visualization complete!")
catch e
    println("Visualization not available in this Julia version:")
    println("  Error: $e")
    println("\nTo visualize, you may need to run this script with an older Julia version")
    println("or use an external visualization tool compatible with Dojo.jl")
end
