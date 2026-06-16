using CSV
using DataFrames
using Plots
using Printf
using Statistics

example_dir = @__DIR__
dr_dir = joinpath(example_dir, "dr_results")
mpc_dir = joinpath(example_dir, "mpc_results")

dr_h = CSV.read(joinpath(dr_dir, "dr_h.csv"), DataFrame)
mpc_h = CSV.read(joinpath(mpc_dir, "mpc_h.csv"), DataFrame)

successful_seeds = intersect(unique(dr_h.seed), unique(mpc_h.seed))
sort!(successful_seeds)

dr_time_cols = names(dr_h, Not(:seed))
mpc_time_cols = names(mpc_h, Not(:seed))

function first_row_per_seed(df::DataFrame, seeds)
    rows = Int[]
    for seed in seeds
        row = findfirst(==(seed), df.seed)
        isnothing(row) || push!(rows, row)
    end
    return df[rows, :]
end

dr_h = first_row_per_seed(dr_h, successful_seeds)
mpc_h = first_row_per_seed(mpc_h, successful_seeds)

function trajectory_summary(df::DataFrame, time_cols)
    trajectories = Matrix(select(df, time_cols))
    final_height = trajectories[:, end]
    peak_height = vec(maximum(trajectories; dims=2))
    return (
        n=size(trajectories, 1),
        final_mean=mean(final_height),
        final_std=std(final_height),
        final_min=minimum(final_height),
        final_max=maximum(final_height),
        peak_mean=mean(peak_height),
        peak_std=std(peak_height),
    )
end

function print_summary(name, summary)
    @printf(
        "| %-6s | %9d | %.5f +/- %.5f | %.5f-%.5f | %.5f +/- %.5f |\n",
        name,
        summary.n,
        summary.final_mean,
        summary.final_std,
        summary.final_min,
        summary.final_max,
        summary.peak_mean,
        summary.peak_std,
    )
end

dr_summary = trajectory_summary(dr_h, dr_time_cols)
mpc_summary = trajectory_summary(mpc_h, mpc_time_cols)

println("Rocket trajectory summary using first saved row for each common seed.")
println()
println("| Method | Scenarios | Final Height (mean +/- std) | Final Height Range | Peak Height (mean +/- std) |")
println("|:---|---:|---:|---:|---:|")
print_summary("TS-DDR", dr_summary)
print_summary("MPC", mpc_summary)
println()

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

example_plot_path = joinpath(example_dir, "rocket_height_comparison.png")
docs_plot_path = normpath(joinpath(example_dir, "..", "..", "docs", "src", "assets", "rocket_height_comparison.png"))

savefig(plt, example_plot_path)
mkpath(dirname(docs_plot_path))
savefig(plt, docs_plot_path)

println("Saved $(relpath(example_plot_path, example_dir))")
println("Saved $(relpath(docs_plot_path, example_dir))")
