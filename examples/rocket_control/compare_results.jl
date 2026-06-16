using CSV
using DataFrames
using Plots

example_dir = @__DIR__
dr_dir = joinpath(example_dir, "dr_results")
mpc_dir = joinpath(example_dir, "mpc_results")

dr_h = CSV.read(joinpath(dr_dir, "dr_h.csv"), DataFrame)
mpc_h = CSV.read(joinpath(mpc_dir, "mpc_h.csv"), DataFrame)

successful_seeds = intersect(unique(dr_h.seed), unique(mpc_h.seed))

dr_time_cols = names(dr_h, Not(:seed))
mpc_time_cols = names(mpc_h, Not(:seed))

plt = plot(
    xlabel = "Time Step",
    ylabel = "Height",
    title = "Rocket Height: DR vs MPC",
)

for (j, seed) in enumerate(successful_seeds)
    dr_rows = dr_h[dr_h.seed .== seed, dr_time_cols]
    mpc_rows = mpc_h[mpc_h.seed .== seed, mpc_time_cols]

    # Plot the first row for this seed. This avoids accidentally plotting
    # multiple series with the same legend label.
    dr_y = vec(Matrix(dr_rows[1:1, :]))
    mpc_y = vec(Matrix(mpc_rows[1:1, :]))

    plot!(
        plt,
        1:length(dr_y),
        dr_y;
        color = :red,
        label = j == 1 ? "DR (TS-DDR)" : false,
        alpha = 0.6,
    )

    plot!(
        plt,
        1:length(mpc_y),
        mpc_y;
        color = :blue,
        label = j == 1 ? "MPC" : false,
        alpha = 0.6,
    )
end

savefig(plt, joinpath(example_dir, "rocket_height_comparison.png"))
println("Saved rocket_height_comparison.png")