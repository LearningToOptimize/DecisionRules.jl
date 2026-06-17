"""
Compare all policies for the latent seasonal ex-ante lot-sizing problem.
"""

using CSV, DataFrames, Statistics, Printf, Random
using Plots, StatsPlots

include(joinpath(@__DIR__, "build_inventory_problem.jl"))

result_dir = joinpath(@__DIR__, "results")

dr_costs = CSV.read(joinpath(result_dir, "dr_costs.csv"), DataFrame).operational_cost
bs_costs = CSV.read(joinpath(result_dir, "basestock_costs.csv"), DataFrame).operational_cost
rand_costs = CSV.read(joinpath(result_dir, "random_costs.csv"), DataFrame).operational_cost
sddp_costs = CSV.read(joinpath(result_dir, "sddp_costs.csv"), DataFrame).operational_cost
opt_costs = CSV.read(joinpath(result_dir, "optimal_costs.csv"), DataFrame).operational_cost

dr_traj_df = CSV.read(joinpath(result_dir, "dr_trajectories.csv"), DataFrame)
bs_traj_df = CSV.read(joinpath(result_dir, "basestock_trajectories.csv"), DataFrame)
curve_df = CSV.read(joinpath(result_dir, "training_curve.csv"), DataFrame)
sddp_log = CSV.read(joinpath(result_dir, "sddp_training_log.csv"), DataFrame)
timing_df = vcat(
    CSV.read(joinpath(result_dir, "dr_timing.csv"), DataFrame),
    CSV.read(joinpath(result_dir, "sddp_timing.csv"), DataFrame),
    CSV.read(joinpath(result_dir, "baseline_timing.csv"), DataFrame),
    CSV.read(joinpath(result_dir, "optimal_timing.csv"), DataFrame),
    cols=:union,
)
timing = Dict(row.method => row for row in eachrow(timing_df))

S_STAR = parse(Float64, strip(read(joinpath(result_dir, "basestock_S_star.txt"), String)))
sddp_bound = parse(Float64, strip(read(joinpath(result_dir, "sddp_bound.txt"), String)))
dp_value = parse(Float64, strip(read(joinpath(result_dir, "optimal_dp_value.txt"), String)))

function ci95(costs)
    return 1.96 * std(costs) / sqrt(length(costs))
end

println("Latent seasonal ex-ante lot-sizing — operational cost comparison")
println()
println("Marginal DP model value:        $(@sprintf("%.1f", dp_value))")
println("SDDP LP relaxation bound:       $(@sprintf("%.1f", sddp_bound))")
println()
println("| Method                   |   N | Mean cost |   Std | 95% CI | vs TS-DDR | Fit (s) | Eval ms/scen |")
println("|:-------------------------|----:|----------:|------:|-------:|---------:|--------:|-------------:|")

entries = [
    ("TS-DDR (trained)", dr_costs),
    ("SDDP.jl integer rollout", sddp_costs),
    ("Base-stock (S*=$(round(Int, S_STAR)))", bs_costs),
    ("Marginal DP policy", opt_costs),
    ("Random (untrained)", rand_costs),
]

dr_mean = mean(dr_costs)
for (name, costs) in entries
    timing_key = startswith(name, "Base-stock") ? "Base-stock" : name
    row = timing[timing_key]
    gap = (mean(costs) - dr_mean) / dr_mean * 100
    @printf(
        "| %-24s | %3d | %9.1f | %5.1f | %6.1f | %+8.1f%% | %7.1f | %12.2f |\n",
        name,
        length(costs),
        mean(costs),
        std(costs),
        ci95(costs),
        gap,
        row.fit_seconds,
        row.inference_ms_per_scenario,
    )
end
println()

best_practical_baseline = minimum(mean.([sddp_costs, bs_costs, rand_costs]))
if mean(dr_costs) > best_practical_baseline
    @warn "TS-DDR did not beat the best practical baseline on this run" mean_dr=mean(dr_costs) best_practical_baseline
end

example_dir = @__DIR__
docs_dir = normpath(joinpath(example_dir, "..", "..", "docs", "src", "assets"))
mkpath(docs_dir)
time_cols = [Symbol("t$i") for i in 0:INVENTORY_T]

periods = 1:INVENTORY_T
demand_mid = (D_LO .+ D_HI) ./ 2
plt_demand = plot(
    periods,
    demand_mid;
    xlabel="Period",
    ylabel="Demand",
    title="Random-phase latent demand process",
    label="Nominal seasonal center",
    linewidth=2,
    linestyle=:dash,
    color=:purple,
)
rng_plot = MersenneTwister(1234)
for k in 1:24
    path = sample_inventory_demand_path(rng_plot)
    plot!(plt_demand, periods, path; color=:gray, alpha=0.28, label=false)
end
savefig(plt_demand, joinpath(docs_dir, "inventory_demand_process.png"))
println("Saved inventory_demand_process.png")

plt_train = plot(
    curve_df.batch,
    curve_df.loss;
    xlabel="Batch",
    ylabel="Mean operational cost",
    title="TS-DDR training — latent seasonal ex-ante lot-sizing",
    legend=false,
    linewidth=2,
    color=:steelblue,
)
savefig(plt_train, joinpath(docs_dir, "inventory_training_curve.png"))
println("Saved inventory_training_curve.png")

valid_sddp = filter(row -> !ismissing(row.bound) && isfinite(row.bound), sddp_log)
plt_sddp = plot(
    valid_sddp.iteration,
    valid_sddp.bound;
    xlabel="Iteration",
    ylabel="Cost",
    title="SDDP learning curve",
    label="LP lower bound",
    linewidth=2,
    color=:darkgreen,
)
if "simulation_value" in names(valid_sddp)
    sim_rows = filter(row -> !ismissing(row.simulation_value) && isfinite(row.simulation_value), valid_sddp)
    if nrow(sim_rows) > 0
        plot!(
            plt_sddp,
            sim_rows.iteration,
            sim_rows.simulation_value;
            label="Training simulation",
            linewidth=2,
            color=:darkorange,
        )
    end
end
savefig(plt_sddp, joinpath(docs_dir, "inventory_sddp_learning.png"))
println("Saved inventory_sddp_learning.png")

n_show = min(20, nrow(dr_traj_df), nrow(bs_traj_df))
plt_traj = plot(;
    xlabel="Period",
    ylabel="Net inventory",
    title="Net inventory trajectories",
    legend=:topright,
)
for s in 1:n_show
    plot!(plt_traj, 0:INVENTORY_T, Vector(dr_traj_df[s, time_cols]); color=:steelblue, alpha=0.35, label=s == 1 ? "TS-DDR" : false)
end
for s in 1:n_show
    plot!(plt_traj, 0:INVENTORY_T, Vector(bs_traj_df[s, time_cols]); color=:darkorange, alpha=0.35, label=s == 1 ? "Base-stock" : false)
end
hline!(plt_traj, [0.0]; linestyle=:dash, color=:black, label="Zero")
savefig(plt_traj, joinpath(docs_dir, "inventory_trajectories.png"))
println("Saved inventory_trajectories.png")

labels = ["TS-DDR", "SDDP", "Base-stock\n(S*=$(round(Int,S_STAR)))", "Marginal\nDP", "Random"]
data = [dr_costs, sddp_costs, bs_costs, opt_costs, rand_costs]
plt_box = boxplot(
    labels,
    data;
    xlabel="Method",
    ylabel="Operational cost",
    title="Cost comparison — latent seasonal ex-ante lot-sizing",
    legend=false,
    fillcolor=[:gold :darkgreen :steelblue :darkorange :gray],
    linecolor=:black,
)

savefig(plt_box, joinpath(docs_dir, "inventory_cost_comparison.png"))
println("Saved inventory_cost_comparison.png")
println("\nAll assets saved to: $(relpath(docs_dir, example_dir))")
