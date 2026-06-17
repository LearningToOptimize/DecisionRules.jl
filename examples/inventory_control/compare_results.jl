"""
Compare all policies for the stochastic lot-sizing problem.

Loads results saved by:
  train_dr_inventory.jl   → dr_costs.csv, dr_trajectories.csv, training_curve.csv
  evaluate_inventory.jl   → basestock_costs.csv, basestock_trajectories.csv,
                             random_costs.csv, basestock_S_star.txt
  solve_sddp.jl           → sddp_costs.csv, sddp_bound.txt
  solve_optimal_dp.jl     → optimal_costs.csv, optimal_dp_value.txt

Produces:
  inventory_cost_comparison.png
  inventory_trajectories.png
  inventory_training_curve.png
  Summary table printed to stdout
"""

using CSV, DataFrames, Statistics, Printf
using Plots, StatsPlots

result_dir = joinpath(@__DIR__, "results")

# ── Load results ──────────────────────────────────────────────────────────────
dr_costs   = CSV.read(joinpath(result_dir, "dr_costs.csv"),        DataFrame).operational_cost
bs_costs   = CSV.read(joinpath(result_dir, "basestock_costs.csv"), DataFrame).operational_cost
rand_costs = CSV.read(joinpath(result_dir, "random_costs.csv"),    DataFrame).operational_cost
sddp_costs = CSV.read(joinpath(result_dir, "sddp_costs.csv"),      DataFrame).operational_cost
opt_costs  = CSV.read(joinpath(result_dir, "optimal_costs.csv"),   DataFrame).operational_cost

dr_traj_df  = CSV.read(joinpath(result_dir, "dr_trajectories.csv"),       DataFrame)
bs_traj_df  = CSV.read(joinpath(result_dir, "basestock_trajectories.csv"), DataFrame)
curve_df    = CSV.read(joinpath(result_dir, "training_curve.csv"),         DataFrame)

S_STAR     = parse(Float64, strip(read(joinpath(result_dir, "basestock_S_star.txt"),   String)))
sddp_bound = parse(Float64, strip(read(joinpath(result_dir, "sddp_bound.txt"),         String)))
dp_value   = parse(Float64, strip(read(joinpath(result_dir, "optimal_dp_value.txt"),   String)))

# ── Summary table ─────────────────────────────────────────────────────────────
println("Stochastic lot-sizing — operational cost comparison")
println()
println("DP exact optimal expected cost: $(@sprintf("%.1f", dp_value))")
println("SDDP lower bound:               $(@sprintf("%.1f", sddp_bound))")
println()
println("| Method                   |   N | Mean cost |   Std | vs Optimal |")
println("|:-------------------------|----:|----------:|------:|-----------:|")

entries = [
    ("TS-DDR (trained)",         dr_costs),
    ("Base-stock (S*=$(round(Int, S_STAR)))", bs_costs),
    ("Random (untrained)",       rand_costs),
    ("SDDP.jl",                  sddp_costs),
    ("DP Optimal",               opt_costs),
]

opt_mean = mean(opt_costs)
for (name, costs) in entries
    gap = (mean(costs) - opt_mean) / opt_mean * 100
    @printf("| %-24s | %3d | %9.1f | %5.1f | %+9.1f%% |\n",
        name, length(costs), mean(costs), std(costs), gap)
end
println()

# ── Directories ───────────────────────────────────────────────────────────────
example_dir = @__DIR__
docs_dir    = normpath(joinpath(example_dir, "..", "..", "docs", "src", "assets"))
mkpath(docs_dir)

T_STAGES  = 12
time_cols = [Symbol("t$i") for i in 0:T_STAGES]

# ── Training curve ────────────────────────────────────────────────────────────
plt_curve = plot(
    curve_df.batch, curve_df.loss;
    xlabel="Batch", ylabel="Operational cost (mean)",
    title="TS-DDR training — stochastic lot-sizing",
    legend=false, linewidth=2, color=:steelblue,
)
savefig(plt_curve, joinpath(docs_dir, "inventory_training_curve.png"))
println("Saved inventory_training_curve.png")

# ── Net-inventory trajectories ────────────────────────────────────────────────
n_show   = min(20, nrow(dr_traj_df), nrow(bs_traj_df))
plt_traj = plot(;
    xlabel="Period", ylabel="Net inventory",
    title="Net inventory trajectories",
    legend=:topright,
)
for s in 1:n_show
    plot!(plt_traj, 0:T_STAGES, Vector(dr_traj_df[s, time_cols]);
          color=:steelblue, alpha=0.35, label = s == 1 ? "TS-DDR" : false)
end
for s in 1:n_show
    plot!(plt_traj, 0:T_STAGES, Vector(bs_traj_df[s, time_cols]);
          color=:darkorange, alpha=0.35, label = s == 1 ? "Base-stock (S*=$(round(Int,S_STAR)))" : false)
end
hline!(plt_traj, [0.0]; linestyle=:dash, color=:black, label="Zero inventory")
savefig(plt_traj, joinpath(docs_dir, "inventory_trajectories.png"))
println("Saved inventory_trajectories.png")

# ── Cost comparison (box plot) ────────────────────────────────────────────────
labels = ["TS-DDR", "Base-stock\n(S*=$(round(Int,S_STAR)))", "Random", "SDDP.jl", "DP Optimal"]
data   = [dr_costs, bs_costs, rand_costs, sddp_costs, opt_costs]
colors = [:steelblue :darkorange :gray :green :gold]

plt_box = boxplot(
    labels, data;
    xlabel="Method", ylabel="Operational cost",
    title="Cost comparison — stochastic lot-sizing (T=12)",
    legend=false,
    fillcolor=colors,
    linecolor=:black,
    notch=false,
)

savefig(plt_box, joinpath(example_dir, "inventory_cost_comparison.png"))
savefig(plt_box, joinpath(docs_dir,    "inventory_cost_comparison.png"))
println("Saved inventory_cost_comparison.png")
println("\nAll assets saved to: $(relpath(docs_dir, example_dir))")
