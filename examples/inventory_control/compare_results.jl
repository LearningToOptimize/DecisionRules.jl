"""
Compare all policies for both relaxed and integer inventory problem variants.
Produces two-section output: tables + compact 2×2 plot layouts.
"""

using CSV, DataFrames, Statistics, Printf, Random
using Plots, StatsPlots

include(joinpath(@__DIR__, "build_inventory_problem.jl"))

result_dir = joinpath(@__DIR__, "results")
example_dir = @__DIR__
docs_dir = normpath(joinpath(example_dir, "..", "..", "docs", "src", "assets"))
mkpath(docs_dir)

function ci95(costs)
    return 1.96 * std(costs) / sqrt(length(costs))
end

function load_costs(tag, method)
    CSV.read(joinpath(result_dir, "$(tag)_$(method)_costs.csv"), DataFrame).operational_cost
end

function load_timing(tags)
    dfs = DataFrame[]
    for tag in tags
        for f in ["dr_timing", "sddp_timing", "sddp_lp_timing", "baseline_timing"]
            path = joinpath(result_dir, "$(tag)_$(f).csv")
            isfile(path) && push!(dfs, CSV.read(path, DataFrame))
        end
    end
    df = vcat(dfs..., cols=:union)
    return Dict(row.method => row for row in eachrow(df))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Print comparison table
# ═══════════════════════════════════════════════════════════════════════════════
function print_table(entries, timing, sddp_bound; ref_idx=1)
    ref_mean = mean(entries[ref_idx][2])
    println("SDDP LP bound: $(@sprintf("%.1f", sddp_bound))")
    println()
    println("| Method                   |   N | Mean cost |   Std | 95% CI | vs $(entries[ref_idx][1]) | Fit (s) | Eval (s) |")
    println("|:-------------------------|----:|----------:|------:|-------:|----------:|--------:|---------:|")
    for (name, costs) in entries
        timing_key = startswith(name, "Base-stock") ? "Base-stock" : name
        row = timing[timing_key]
        gap = (mean(costs) - ref_mean) / ref_mean * 100
        @printf("| %-24s | %3d | %9.1f | %5.1f | %6.1f | %+9.1f%% | %7.1f | %8.4f |\n",
            name, length(costs), mean(costs), std(costs), ci95(costs),
            gap, row.fit_seconds, row.eval_seconds)
    end
    println()
end

# ═══════════════════════════════════════════════════════════════════════════════
# Compact 2×2 plot
# ═══════════════════════════════════════════════════════════════════════════════
function make_plots(tag, entries, S_star, title_suffix; sddp_tag=tag, dr_tag=tag)
    time_cols = [Symbol("t$i") for i in 0:INVENTORY_T]

    # (1,1) SDDP learning curve
    sddp_log = CSV.read(joinpath(result_dir, "$(sddp_tag)_sddp_training_log.csv"), DataFrame)
    valid = filter(row -> !ismissing(row.bound) && isfinite(row.bound), sddp_log)
    p1 = plot(valid.iteration, valid.bound;
        xlabel="Iteration", ylabel="Cost",
        title="SDDP learning curve", label="LP bound",
        linewidth=2, color=:darkgreen, legend=:right)
    if "simulation_value" in names(valid)
        sim_rows = filter(row -> !ismissing(row.simulation_value) && isfinite(row.simulation_value), valid)
        if nrow(sim_rows) > 0
            plot!(p1, sim_rows.iteration, sim_rows.simulation_value;
                label="Simulation", linewidth=2, color=:darkorange)
        end
    end

    # (1,2) TS-DDR training curve
    curve_df = CSV.read(joinpath(result_dir, "$(dr_tag)_training_curve.csv"), DataFrame)
    p2 = plot(curve_df.batch, curve_df.loss;
        xlabel="Batch", ylabel="Mean operational cost",
        title="TS-DDR training curve", legend=false,
        linewidth=2, color=:steelblue)

    # (2,1) Net-inventory trajectories
    dr_traj = CSV.read(joinpath(result_dir, "$(dr_tag)_dr_trajectories.csv"), DataFrame)
    bs_tag_file = sddp_tag  # baselines share the sddp tag prefix
    bs_traj = CSV.read(joinpath(result_dir, "$(bs_tag_file)_basestock_trajectories.csv"), DataFrame)
    n_show = min(20, nrow(dr_traj), nrow(bs_traj))
    p3 = plot(; xlabel="Period", ylabel="Net inventory",
        title="Inventory trajectories", legend=:topright)
    for s in 1:n_show
        plot!(p3, 0:INVENTORY_T, Vector(dr_traj[s, time_cols]);
            color=:steelblue, alpha=0.35, label=s == 1 ? "TS-DDR" : false)
    end
    for s in 1:n_show
        plot!(p3, 0:INVENTORY_T, Vector(bs_traj[s, time_cols]);
            color=:darkorange, alpha=0.35, label=s == 1 ? "Base-stock" : false)
    end
    hline!(p3, [0.0]; linestyle=:dash, color=:black, label="Zero")

    # (2,2) Cost distribution boxplot
    labels = [e[1] for e in entries]
    short_labels = replace.(labels,
        "TS-DDR (FixedDiscrete)" => "TS-DDR\n(FixedDisc)",
        "TS-DDR (ContRelax)" => "TS-DDR\n(ContRelax)",
        "TS-DDR (trained)" => "TS-DDR",
        "SDDP (PAR)" => "SDDP",
        "SDDP (MIP fwd)" => "SDDP\n(MIP fwd)",
        "SDDP (LP relax)" => "SDDP\n(LP relax)",
        "Random (untrained)" => "Random")
    short_labels = [startswith(l, "Base-stock") ? "Base-stock\n(S*=$(round(Int,S_star)))" : l for l in short_labels]
    data = [e[2] for e in entries]
    n_methods = length(entries)
    method_colors = if n_methods == 4
        [:steelblue, :darkgreen, :gold, :gray]
    elseif n_methods == 5
        [:steelblue, :royalblue, :darkgreen, :gold, :gray]
    elseif n_methods == 6
        [:steelblue, :royalblue, :darkgreen, :seagreen, :gold, :gray]
    else
        palette(:auto, n_methods)
    end
    p4 = plot(; xlabel="Method", ylabel="Operational cost",
        title="Cost comparison", legend=false)
    for i in 1:n_methods
        violin!(p4, fill(short_labels[i], length(data[i])), data[i];
            fillcolor=method_colors[i], linecolor=:black, fillalpha=0.7)
    end

    layout = @layout [a b; c d]
    combined = plot(p1, p2, p3, p4; layout=layout, size=(1100, 800),
        plot_title=title_suffix, plot_titlefontsize=12, margin=5Plots.mm)
    return combined
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demand process plot (shared)
# ═══════════════════════════════════════════════════════════════════════════════
periods = 1:INVENTORY_T
demand_mid = (D_LO .+ D_HI) ./ 2
plt_demand = plot(periods, demand_mid;
    xlabel="Period", ylabel="Demand",
    title="Latent demand process (random phase + regime + AR)",
    label="Nominal seasonal center", linewidth=2, linestyle=:dash, color=:purple)
rng_plot = MersenneTwister(1234)
for k in 1:24
    path = sample_inventory_demand_path(rng_plot)
    plot!(plt_demand, periods, path; color=:gray, alpha=0.28, label=false)
end
savefig(plt_demand, joinpath(docs_dir, "inventory_demand_process.png"))
println("Saved inventory_demand_process.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Relaxed
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 60)
println("SECTION 1: Relaxed (continuous) comparison")
println("=" ^ 60)

r_dr = load_costs("relaxed", "dr")
r_sddp = load_costs("relaxed", "sddp")
r_bs = load_costs("relaxed", "basestock")
r_rand = load_costs("relaxed", "random")
r_timing = load_timing(["relaxed"])
r_S = parse(Float64, strip(read(joinpath(result_dir, "relaxed_basestock_S_star.txt"), String)))
r_bound = parse(Float64, strip(read(joinpath(result_dir, "relaxed_sddp_bound.txt"), String)))

r_entries = [
    ("TS-DDR (trained)", r_dr),
    ("SDDP (PAR)", r_sddp),
    ("Base-stock (S*=$(round(Int, r_S)))", r_bs),
    ("Random (untrained)", r_rand),
]
print_table(r_entries, r_timing, r_bound)

plt_relaxed = make_plots("relaxed", r_entries, r_S, "Relaxed (continuous) problem")
savefig(plt_relaxed, joinpath(docs_dir, "inventory_relaxed_results.png"))
println("Saved inventory_relaxed_results.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Integer
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 60)
println("SECTION 2: Integer (MIP) comparison")
println("=" ^ 60)

i_dr = load_costs("integer", "dr")
i_dr_cr = load_costs("integer_cr", "dr")
i_sddp = load_costs("integer", "sddp")
i_sddp_lp = load_costs("integer", "sddp_lp")
i_bs = load_costs("integer", "basestock")
i_rand = load_costs("integer", "random")
i_timing = load_timing(["integer", "integer_cr"])
i_S = parse(Float64, strip(read(joinpath(result_dir, "integer_basestock_S_star.txt"), String)))
i_bound = parse(Float64, strip(read(joinpath(result_dir, "integer_sddp_bound.txt"), String)))

i_entries = [
    ("TS-DDR (FixedDiscrete)", i_dr),
    ("TS-DDR (ContRelax)", i_dr_cr),
    ("SDDP (MIP fwd)", i_sddp),
    ("SDDP (LP relax)", i_sddp_lp),
    ("Base-stock (S*=$(round(Int, i_S)))", i_bs),
    ("Random (untrained)", i_rand),
]
print_table(i_entries, i_timing, i_bound)

plt_integer = make_plots("integer", i_entries, i_S, "Integer (MIP) problem";
    sddp_tag="integer", dr_tag="integer")
println("  SDDP (LP relax) vs SDDP (MIP fwd): LP=$(round(mean(i_sddp_lp),digits=1)), MIP=$(round(mean(i_sddp),digits=1))")
savefig(plt_integer, joinpath(docs_dir, "inventory_integer_results.png"))
println("Saved inventory_integer_results.png")

println("\nAll assets saved to: $(relpath(docs_dir, example_dir))")
