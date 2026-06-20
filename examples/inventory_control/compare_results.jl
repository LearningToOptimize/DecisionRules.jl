"""
Compare inventory-control benchmark results and regenerate documentation plots.

This script expects the CSV files written by:

- `train_dr_inventory.jl`;
- `evaluate_inventory.jl`; and
- `solve_sddp.jl`.

It prints Markdown tables and writes the figures used by the documentation.
"""

using CSV
using DataFrames
using Plots
using Printf
using Random
using Statistics
using StatsPlots

include(joinpath(@__DIR__, "build_inventory_problem.jl"))

# All benchmark scripts write their raw CSV files here.
const RESULT_DIR = joinpath(@__DIR__, "results")

# Documentation figures are checked into the docs asset directory.
const DOCS_ASSET_DIR = normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "assets"))

# Ensure the figure output directory exists before plotting.
mkpath(DOCS_ASSET_DIR)

"""
    MethodResult

Costs and display metadata for one benchmark method.

# Fields
- `name::String`: label printed in tables.
- `costs::Vector{Float64}`: operational costs, one per evaluation scenario.

# Examples
```julia
result = MethodResult("TS-DDR", [1.0, 2.0, 3.0])
```
"""
struct MethodResult
    name::String
    costs::Vector{Float64}
end

"""
    TimingRecord

Training and evaluation timing for one benchmark method.

# Fields
- `fit_seconds::Float64`: total fitting time.
- `eval_seconds::Float64`: average evaluation time per stage.

# Examples
```julia
timing = TimingRecord(10.0, 0.01)
```
"""
struct TimingRecord
    fit_seconds::Float64
    eval_seconds::Float64
end

"""
    ci95(costs::AbstractVector{<:Real}) -> Float64

Return the normal-approximation half-width of a 95% confidence interval.

The reported value is

```math
1.96 \\frac{s}{\\sqrt{n}},
```

where ``s`` is the sample standard deviation and ``n`` is the number of costs.

# Arguments
- `costs::AbstractVector{<:Real}`: sampled operational costs.

# Examples
```julia
half_width = ci95([10.0, 12.0, 11.0])
```
"""
function ci95(costs::AbstractVector{<:Real})
    # The table reports uncertainty in the sample mean, not in one trajectory.
    return 1.96 * std(costs) / sqrt(length(costs))
end

"""
    percent_gap(costs, reference_costs) -> Float64

Return the mean-cost percent gap relative to a reference method.

The gap is

```math
100 \\frac{\\bar{c} - \\bar{c}_{ref}}{\\bar{c}_{ref}}.
```

# Arguments
- `costs::AbstractVector{<:Real}`: candidate method costs.
- `reference_costs::AbstractVector{<:Real}`: reference method costs.

# Examples
```julia
gap = percent_gap(candidate.costs, reference.costs)
```
"""
function percent_gap(costs, reference_costs)
    # Positive gaps mean the candidate is more expensive than the reference.
    return 100.0 * (mean(costs) - mean(reference_costs)) / mean(reference_costs)
end

"""
    load_costs(tag::AbstractString, method::AbstractString) -> Vector{Float64}

Load one benchmark cost vector from `RESULT_DIR`.

# Arguments
- `tag::AbstractString`: problem prefix, such as `"relaxed"` or `"integer"`.
- `method::AbstractString`: method suffix, such as `"dr"` or `"sddp"`.

# Examples
```julia
costs = load_costs("integer", "dr")
```
"""
function load_costs(tag::AbstractString, method::AbstractString)
    # Every cost file uses the shared `operational_cost` column.
    table = CSV.read(joinpath(RESULT_DIR, "$(tag)_$(method)_costs.csv"), DataFrame)

    return Float64.(table.operational_cost)
end

"""
    optional_costs(tag::AbstractString, method::AbstractString)

Load costs if a result file exists; otherwise return `nothing`.

# Arguments
- `tag::AbstractString`: problem prefix.
- `method::AbstractString`: method suffix.

# Examples
```julia
costs = optional_costs("integer_sf", "dr")
```
"""
function optional_costs(tag::AbstractString, method::AbstractString)
    # Optional variants should not make the comparison script fail.
    path = joinpath(RESULT_DIR, "$(tag)_$(method)_costs.csv")

    return isfile(path) ? Float64.(CSV.read(path, DataFrame).operational_cost) : nothing
end

"""
    read_scalar(path::AbstractString) -> Float64

Read a scalar floating-point value from a text file.

# Arguments
- `path::AbstractString`: text file containing one numeric value.

# Examples
```julia
bound = read_scalar(joinpath(RESULT_DIR, "integer_sddp_bound.txt"))
```
"""
function read_scalar(path::AbstractString)
    # Baseline scripts write scalar values as plain text.
    return parse(Float64, strip(read(path, String)))
end

"""
    timing_key(method_name::AbstractString) -> String

Return the key used to look up timing rows.

# Arguments
- `method_name::AbstractString`: table label.

# Examples
```julia
key = timing_key("Base-stock (S*=160)")
```
"""
function timing_key(method_name::AbstractString)
    # Base-stock labels include S*, while timing files use a stable method name.
    startswith(method_name, "Base-stock") && return "Base-stock"

    return String(method_name)
end

"""
    load_timing(tags) -> Dict{String,TimingRecord}

Load timing rows for a set of result prefixes.

# Arguments
- `tags`: iterable of prefixes, such as `["integer", "integer_cr"]`.

# Examples
```julia
timing = load_timing(["integer", "integer_cr"])
```
"""
function load_timing(tags)
    # These suffixes cover TS-DDR, SDDP, LP-relaxed SDDP, and baselines.
    timing_suffixes = ["dr_timing", "sddp_timing", "sddp_lp_timing", "baseline_timing"]

    # Accumulate all timing CSVs that exist for the requested tags.
    rows = DataFrame[]
    for tag in tags
        for suffix in timing_suffixes
            path = joinpath(RESULT_DIR, "$(tag)_$(suffix).csv")
            isfile(path) && push!(rows, CSV.read(path, DataFrame))
        end
    end

    # A missing timing row is a data-generation error, so keep loading strict.
    combined = vcat(rows...; cols = :union)

    # Convert DataFrame rows to a small typed dictionary.
    return Dict(
        String(row.method) => TimingRecord(row.fit_seconds, row.eval_seconds)
        for row in eachrow(combined)
    )
end

"""
    print_table(results, timing, bound; reference_index = 1)

Print a Markdown comparison table.

# Arguments
- `results::Vector{MethodResult}`: cost vectors and display names.
- `timing::Dict{String,TimingRecord}`: timing rows by method name.
- `bound::Real`: SDDP lower bound printed above the table.
- `reference_index::Integer`: result used for percent-gap comparisons.

# Examples
```julia
print_table(results, timing, bound)
```
"""
function print_table(
    results::Vector{MethodResult},
    timing::Dict{String,TimingRecord},
    bound::Real;
    reference_index::Integer = 1,
)
    # The reference method defines the "vs" column.
    reference = results[reference_index]

    println("SDDP LP bound: $(@sprintf("%.1f", bound))")
    println()
    println(
        "| Method                   |   N | Mean cost |   Std | 95% CI | " *
        "vs $(reference.name) | Fit (s) | Eval (s) |",
    )
    println("|:-------------------------|----:|----------:|------:|-------:|----------:|--------:|---------:|")

    for result in results
        # Timing rows use stable method labels.
        row = timing[timing_key(result.name)]

        # Compute all statistics once so the table row is easy to inspect.
        mean_cost = mean(result.costs)
        std_cost = std(result.costs)
        confidence = ci95(result.costs)
        gap = percent_gap(result.costs, reference.costs)

        @printf(
            "| %-24s | %3d | %9.1f | %5.1f | %6.1f | %+9.1f%% | %7.1f | %8.4f |\n",
            result.name,
            length(result.costs),
            mean_cost,
            std_cost,
            confidence,
            gap,
            row.fit_seconds,
            row.eval_seconds,
        )
    end

    println()

    return nothing
end

"""
    short_method_label(name::AbstractString, base_stock_level::Real) -> String

Return compact labels for plot axes.

# Arguments
- `name::AbstractString`: full table label.
- `base_stock_level::Real`: base-stock order-up-to level.

# Examples
```julia
label = short_method_label("TS-DDR (FixedDiscrete)", 160.0)
```
"""
function short_method_label(name::AbstractString, base_stock_level::Real)
    # Keep repeated labels short enough for the violin plot axis.
    startswith(name, "Base-stock") &&
        return "Base-stock\n(S*=$(round(Int, base_stock_level)))"

    replacements = Dict(
        "TS-DDR (FixedDiscrete)" => "TS-DDR\n(FixedDisc)",
        "TS-DDR (ContRelax)" => "TS-DDR\n(ContRelax)",
        "TS-DDR (MixedGrad)" => "TS-DDR\n(MixedGrad)",
        "TS-DDR (trained)" => "TS-DDR",
        "SDDP (PAR)" => "SDDP",
        "SDDP (MIP fwd)" => "SDDP\n(MIP fwd)",
        "SDDP (LP relax)" => "SDDP\n(LP relax)",
        "Random (untrained)" => "Random",
    )

    return get(replacements, String(name), String(name))
end

"""
    method_colors(num_methods::Integer)

Return stable plot colors for the number of compared methods.

# Arguments
- `num_methods::Integer`: number of methods in the comparison.

# Examples
```julia
colors = method_colors(length(results))
```
"""
function method_colors(num_methods::Integer)
    # Keep colors stable between documentation rebuilds.
    if num_methods == 4
        return [:steelblue, :darkgreen, :gold, :gray]
    elseif num_methods == 5
        return [:steelblue, :royalblue, :darkgreen, :gold, :gray]
    elseif num_methods == 6
        return [:steelblue, :royalblue, :darkgreen, :seagreen, :gold, :gray]
    elseif num_methods == 7
        return [:steelblue, :royalblue, :darkorange, :darkgreen, :seagreen, :gold, :gray]
    end

    return palette(:auto, num_methods)
end

"""
    plot_sddp_learning_curve(tag::AbstractString)

Create the SDDP training-bound subplot.

# Arguments
- `tag::AbstractString`: result prefix used by SDDP output files.

# Examples
```julia
plot_sddp_learning_curve("integer")
```
"""
function plot_sddp_learning_curve(tag::AbstractString)
    # SDDP logs may include failed or missing simulation rows.
    log = CSV.read(joinpath(RESULT_DIR, "$(tag)_sddp_training_log.csv"), DataFrame)

    # Log-scale plots require strictly positive finite values.
    valid_bound_rows = filter(
        row -> !ismissing(row.bound) && isfinite(row.bound) && row.bound > 0,
        log,
    )

    plot_handle = plot(
        valid_bound_rows.iteration,
        valid_bound_rows.bound;
        xlabel = "Iteration",
        ylabel = "Cost (log scale)",
        title = "SDDP learning curve",
        label = "LP bound",
        linewidth = 2,
        color = :darkgreen,
        legend = :right,
        yscale = :log10,
    )

    if "simulation_value" in names(valid_bound_rows)
        # Simulation values are optional and may be recorded sparsely.
        valid_sim_rows = filter(
            row -> !ismissing(row.simulation_value) &&
                isfinite(row.simulation_value) &&
                row.simulation_value > 0,
            valid_bound_rows,
        )

        if nrow(valid_sim_rows) > 0
            plot!(
                plot_handle,
                valid_sim_rows.iteration,
                valid_sim_rows.simulation_value;
                label = "Simulation",
                linewidth = 2,
                color = :darkorange,
            )
        end
    end

    return plot_handle
end

"""
    plot_training_curves(curve_specs)

Create the TS-DDR training-curve subplot.

# Arguments
- `curve_specs`: tuples `(tag, label, color)` for training-curve CSV files.

# Examples
```julia
plot_training_curves([("integer", "FixedDiscrete", :steelblue)])
```
"""
function plot_training_curves(curve_specs)
    # Start with an empty plot so optional curves can be skipped cleanly.
    plot_handle = plot(;
        xlabel = "Batch",
        ylabel = "Mean operational cost",
        title = "TS-DDR training curves",
        legend = :topright,
    )

    for (tag, label, color) in curve_specs
        # Optional variants should not break the plot.
        path = joinpath(RESULT_DIR, "$(tag)_training_curve.csv")
        isfile(path) || continue

        # Training curves store one loss per SGD batch.
        curve = CSV.read(path, DataFrame)

        plot!(
            plot_handle,
            curve.batch,
            curve.loss;
            label = label,
            linewidth = 2,
            color = color,
        )
    end

    return plot_handle
end

"""
    plot_inventory_trajectories(dr_tag, baseline_tag)

Create the inventory-trajectory subplot.

# Arguments
- `dr_tag::AbstractString`: TS-DDR trajectory prefix.
- `baseline_tag::AbstractString`: baseline trajectory prefix.

# Examples
```julia
plot_inventory_trajectories("integer", "integer")
```
"""
function plot_inventory_trajectories(dr_tag, baseline_tag)
    # Trajectory files have columns t0, t1, ..., tT.
    time_columns = [Symbol("t$(period)") for period in 0:INVENTORY_T]

    # Load TS-DDR and base-stock trajectories.
    dr_paths = CSV.read(joinpath(RESULT_DIR, "$(dr_tag)_dr_trajectories.csv"), DataFrame)
    base_stock_paths = CSV.read(
        joinpath(RESULT_DIR, "$(baseline_tag)_basestock_trajectories.csv"),
        DataFrame,
    )

    # Plot a readable subset rather than all trajectories.
    num_paths = min(20, nrow(dr_paths), nrow(base_stock_paths))

    plot_handle = plot(;
        xlabel = "Period",
        ylabel = "Net inventory",
        title = "Inventory trajectories",
        legend = :topright,
    )

    for row in 1:num_paths
        plot!(
            plot_handle,
            0:INVENTORY_T,
            Vector(dr_paths[row, time_columns]);
            color = :steelblue,
            alpha = 0.35,
            label = row == 1 ? "TS-DDR" : false,
        )
    end

    for row in 1:num_paths
        plot!(
            plot_handle,
            0:INVENTORY_T,
            Vector(base_stock_paths[row, time_columns]);
            color = :darkorange,
            alpha = 0.35,
            label = row == 1 ? "Base-stock" : false,
        )
    end

    # Zero inventory separates holding from backlog.
    hline!(plot_handle, [0.0]; linestyle = :dash, color = :black, label = "Zero")

    return plot_handle
end

"""
    plot_cost_distribution(results, base_stock_level)

Create the cost-distribution subplot.

# Arguments
- `results::Vector{MethodResult}`: methods to compare.
- `base_stock_level::Real`: base-stock order-up-to level.

# Examples
```julia
plot_cost_distribution(results, 160.0)
```
"""
function plot_cost_distribution(results::Vector{MethodResult}, base_stock_level::Real)
    # Convert table labels to compact axis labels.
    labels = [short_method_label(result.name, base_stock_level) for result in results]

    # Keep method colors stable across plot rebuilds.
    colors = method_colors(length(results))

    plot_handle = plot(;
        xlabel = "Method",
        ylabel = "Operational cost",
        title = "Cost comparison",
        legend = false,
    )

    for index in eachindex(results)
        violin!(
            plot_handle,
            fill(labels[index], length(results[index].costs)),
            results[index].costs;
            fillcolor = colors[index],
            linecolor = :black,
            fillalpha = 0.7,
        )
    end

    return plot_handle
end

"""
    make_summary_plot(problem; kwargs...)

Build the 2x2 documentation figure for one problem variant.

# Arguments
- `problem::AbstractString`: figure title.
- `results::Vector{MethodResult}`: compared methods.
- `base_stock_level::Real`: base-stock order-up-to level.
- `sddp_tag::AbstractString`: SDDP result prefix.
- `dr_tag::AbstractString`: TS-DDR trajectory prefix.
- `curve_specs`: training-curve plot specifications.

# Examples
```julia
plot = make_summary_plot(
    "Integer problem",
    results = integer_results,
    base_stock_level = 160.0,
    sddp_tag = "integer",
    dr_tag = "integer",
    curve_specs = [("integer", "FixedDiscrete", :steelblue)],
)
```
"""
function make_summary_plot(
    problem::AbstractString;
    results::Vector{MethodResult},
    base_stock_level::Real,
    sddp_tag::AbstractString,
    dr_tag::AbstractString,
    curve_specs,
)
    # Build each panel with a single responsibility.
    sddp_panel = plot_sddp_learning_curve(sddp_tag)
    training_panel = plot_training_curves(curve_specs)
    trajectory_panel = plot_inventory_trajectories(dr_tag, sddp_tag)
    distribution_panel = plot_cost_distribution(results, base_stock_level)

    # Use a fixed layout so generated docs are stable.
    layout = @layout [a b; c d]

    return plot(
        sddp_panel,
        training_panel,
        trajectory_panel,
        distribution_panel;
        layout = layout,
        size = (1100, 800),
        plot_title = problem,
        plot_titlefontsize = 12,
        margin = 5Plots.mm,
    )
end

"""
    plot_demand_process() -> Nothing

Regenerate the demand-process documentation figure.

# Examples
```julia
plot_demand_process()
```
"""
function plot_demand_process()
    # Period numbers run from 1 to T.
    periods = 1:INVENTORY_T

    # The nominal seasonal center is the midpoint of the demand band.
    demand_midpoint = (D_LO .+ D_HI) ./ 2

    plot_handle = plot(
        periods,
        demand_midpoint;
        xlabel = "Period",
        ylabel = "Demand",
        title = "Latent demand process (random phase + regime + AR)",
        label = "Nominal seasonal center",
        linewidth = 2,
        linestyle = :dash,
        color = :purple,
    )

    # Use a fixed RNG so the figure is reproducible.
    rng = MersenneTwister(1234)

    for _ in 1:24
        # Each path has its own hidden phase, regime, and shock sequence.
        path = sample_inventory_demand_path(rng)

        plot!(
            plot_handle,
            periods,
            path;
            color = :gray,
            alpha = 0.28,
            label = false,
        )
    end

    savefig(plot_handle, joinpath(DOCS_ASSET_DIR, "inventory_demand_process.png"))
    println("Saved inventory_demand_process.png")

    return nothing
end

"""
    relaxed_results() -> (results, timing, base_stock_level, bound)

Load all relaxed-problem comparison data.

# Examples
```julia
results, timing, base_stock_level, bound = relaxed_results()
```
"""
function relaxed_results()
    # Load all relaxed operational-cost samples.
    dr_costs = load_costs("relaxed", "dr")
    sddp_costs = load_costs("relaxed", "sddp")
    base_stock_costs = load_costs("relaxed", "basestock")
    random_costs = load_costs("relaxed", "random")

    # Load scalar baseline metadata.
    base_stock_level = read_scalar(joinpath(RESULT_DIR, "relaxed_basestock_S_star.txt"))
    sddp_bound = read_scalar(joinpath(RESULT_DIR, "relaxed_sddp_bound.txt"))

    # Build display records in table order.
    results = [
        MethodResult("TS-DDR (trained)", dr_costs),
        MethodResult("SDDP (PAR)", sddp_costs),
        MethodResult("Base-stock (S*=$(round(Int, base_stock_level)))", base_stock_costs),
        MethodResult("Random (untrained)", random_costs),
    ]

    return results, load_timing(["relaxed"]), base_stock_level, sddp_bound
end

"""
    integer_results() -> (results, timing, base_stock_level, bound)

Load all integer-problem comparison data.

# Examples
```julia
results, timing, base_stock_level, bound = integer_results()
```
"""
function integer_results()
    # Load required integer operational-cost samples.
    fixed_discrete_costs = load_costs("integer", "dr")
    continuous_relaxation_costs = load_costs("integer_cr", "dr")
    sddp_mip_forward_costs = load_costs("integer", "sddp")
    sddp_lp_relaxation_costs = load_costs("integer", "sddp_lp")
    base_stock_costs = load_costs("integer", "basestock")
    random_costs = load_costs("integer", "random")

    # The mixed-gradient variant is optional because it may be expensive to run.
    mixed_gradient_costs = optional_costs("integer_sf", "dr")

    # Load scalar baseline metadata.
    base_stock_level = read_scalar(joinpath(RESULT_DIR, "integer_basestock_S_star.txt"))
    sddp_bound = read_scalar(joinpath(RESULT_DIR, "integer_sddp_bound.txt"))

    # Build the mandatory method list first.
    results = [
        MethodResult("TS-DDR (FixedDiscrete)", fixed_discrete_costs),
        MethodResult("TS-DDR (ContRelax)", continuous_relaxation_costs),
        MethodResult("SDDP (MIP fwd)", sddp_mip_forward_costs),
        MethodResult("SDDP (LP relax)", sddp_lp_relaxation_costs),
        MethodResult("Base-stock (S*=$(round(Int, base_stock_level)))", base_stock_costs),
        MethodResult("Random (untrained)", random_costs),
    ]

    if !isnothing(mixed_gradient_costs)
        # Insert mixed gradients after the two TS-DDR dual-only variants.
        insert!(results, 3, MethodResult("TS-DDR (MixedGrad)", mixed_gradient_costs))
    end

    # Include optional timing tags only when their result file exists.
    timing_tags = ["integer", "integer_cr"]
    isfile(joinpath(RESULT_DIR, "integer_sf_dr_timing.csv")) &&
        push!(timing_tags, "integer_sf")

    return results, load_timing(timing_tags), base_stock_level, sddp_bound
end

"""
    integer_curve_specs()

Return training-curve plot specs for the integer comparison.

# Examples
```julia
curves = integer_curve_specs()
```
"""
function integer_curve_specs()
    # FixedDiscrete and ContRelax are always part of the integer benchmark.
    specs = [
        ("integer", "FixedDiscrete", :steelblue),
        ("integer_cr", "ContRelax", :royalblue),
    ]

    if isfile(joinpath(RESULT_DIR, "integer_sf_training_curve.csv"))
        # MixedGrad appears only when that expensive variant has been run.
        push!(specs, ("integer_sf", "MixedGrad", :darkorange))
    end

    return specs
end

"""
    run_relaxed_comparison() -> Nothing

Print and plot the relaxed continuous comparison.

# Examples
```julia
run_relaxed_comparison()
```
"""
function run_relaxed_comparison()
    println("\n" * "=" ^ 60)
    println("SECTION 1: Relaxed (continuous) comparison")
    println("=" ^ 60)

    # Load data, print table, and save the documentation figure.
    results, timing, base_stock_level, bound = relaxed_results()
    print_table(results, timing, bound)

    figure = make_summary_plot(
        "Relaxed (continuous) problem";
        results = results,
        base_stock_level = base_stock_level,
        sddp_tag = "relaxed",
        dr_tag = "relaxed",
        curve_specs = [("relaxed", "TS-DDR", :steelblue)],
    )

    savefig(figure, joinpath(DOCS_ASSET_DIR, "inventory_relaxed_results.png"))
    println("Saved inventory_relaxed_results.png")

    return nothing
end

"""
    run_integer_comparison() -> Nothing

Print and plot the integer MIP comparison.

# Examples
```julia
run_integer_comparison()
```
"""
function run_integer_comparison()
    println("\n" * "=" ^ 60)
    println("SECTION 2: Integer (MIP) comparison")
    println("=" ^ 60)

    # Load data, print table, and save the documentation figure.
    results, timing, base_stock_level, bound = integer_results()
    print_table(results, timing, bound)

    figure = make_summary_plot(
        "Integer (MIP) problem";
        results = results,
        base_stock_level = base_stock_level,
        sddp_tag = "integer",
        dr_tag = "integer",
        curve_specs = integer_curve_specs(),
    )

    savefig(figure, joinpath(DOCS_ASSET_DIR, "inventory_integer_results.png"))
    println("Saved inventory_integer_results.png")

    return nothing
end

"""
    main() -> Nothing

Run every inventory-result comparison.

# Examples
```julia
main()
```
"""
function main()
    # Regenerate the demand-process figure before method comparisons.
    plot_demand_process()

    # Print and plot the relaxed benchmark.
    run_relaxed_comparison()

    # Print and plot the integer benchmark.
    run_integer_comparison()

    println("\nAll assets saved to: $(relpath(DOCS_ASSET_DIR, @__DIR__))")

    return nothing
end

# Run only when invoked as a script.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
