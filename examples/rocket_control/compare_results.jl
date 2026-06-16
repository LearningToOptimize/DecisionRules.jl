using CSV
using DataFrames
using Plots

example_dir = @__DIR__
dr_dir = joinpath(example_dir, "dr_results")
mpc_dir = joinpath(example_dir, "mpc_results")

dr_h = CSV.read(joinpath(dr_dir, "dr_h.csv"), DataFrame)
mpc_h = CSV.read(joinpath(mpc_dir, "mpc_h.csv"), DataFrame)

successful_seeds = intersect(dr_h.seed, mpc_h.seed)
T = ncol(dr_h) - 1

plt = Plots.plot(; xlabel="Time Step", ylabel="Height", title="Rocket Height: DR vs MPC");
for (j, i) in enumerate(successful_seeds)
    Plots.plot!(1:T, Matrix(dr_h[dr_h.seed .== i, 1:T])'; color=:red,
        label=(j == 1 ? "DR (TS-DDR)" : ""), alpha=0.6);
    T_mpc = ncol(mpc_h) - 1
    Plots.plot!(1:T_mpc, Matrix(mpc_h[mpc_h.seed .== i, 1:T_mpc])'; color=:blue,
        label=(j == 1 ? "MPC" : ""), alpha=0.6);
end
Plots.savefig(plt, joinpath(example_dir, "rocket_height_comparison.png"))
println("Saved rocket_height_comparison.png")
