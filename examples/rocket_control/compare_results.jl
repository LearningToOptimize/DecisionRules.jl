using CSV
using DataFrames

example_dir = joinpath(pwd(), "examples", "rocket_control") #dirname(@__FILE__)
dr_dir = joinpath(example_dir, "dr_results")
mpc_dir = joinpath(example_dir, "mpc_results")

dr_h = CSV.read(joinpath(dr_dir, "dr_h.csv"), DataFrame)
dr_u = CSV.read(joinpath(dr_dir, "dr_u.csv"), DataFrame)

mpc_h = CSV.read(joinpath(mpc_dir, "mpc_h.csv"), DataFrame)
mpc_u = CSV.read(joinpath(mpc_dir, "mpc_u.csv"), DataFrame)

successful_seeds = intersect(dr_h.seed, mpc_h.seed)

# plot height
# mpc in blue, dr in red
# each row is a different seed, plot each seed as a line
using Plots

plt = Plots.plot(; xlabel="Time", ylabel="Height", legend=false);
for i in successful_seeds
    Plots.plot!(1:1000, Matrix(dr_h[dr_h.seed .== i, 1:1000])'; color=:red);
    Plots.plot!(1:1000, Matrix(mpc_h[mpc_h.seed .== i, 1:1000])'; color=:blue);
end
Plots.savefig(plt, joinpath(example_dir, "height_comparison.png"))
