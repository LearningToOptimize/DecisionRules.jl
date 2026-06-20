"""
Compare inventory-control benchmark results and regenerate documentation plots.

This script expects the CSV files written by:

- `train_dr_inventory.jl`;
- `evaluate_inventory.jl`; and
- `solve_sddp.jl`.

Results live in timestamped subdirectories under `results/`. Pass the run ID as
the first CLI argument, or omit it to use the most recent run:

```bash
julia --project=. compare_results.jl                 # latest run
julia --project=. compare_results.jl 20260619_231417  # specific run
```
"""

using CSV
using DataFrames
using Plots
using Printf
using Random
using Statistics
using StatsPlots

include(joinpath(@__DIR__, "build_inventory_problem.jl"))

"""
    resolve_result_dir(args) -> String

Pick the results directory from CLI args or default to the most recent run.

When the `results/` directory contains timestamped subdirectories
(e.g. `results/20260619_231417/`), the most recent one is used. If no
subdirectories exist, the flat `results/` directory itself is used for
backward compatibility with older runs.

# Arguments
- `args`: `ARGS` from the script entry point.

# Examples
```julia
result_dir = resolve_result_dir(ARGS)
```
"""
function resolve_result_dir(args)
    base = joinpath(@__DIR__, "results")

    if !isempty(args)
        dir = joinpath(base, args[1])
        isdir(dir) || error("Run directory not found: $dir")
        return dir
    end

    subdirs = filter(d -> isdir(joinpath(base, d)), readdir(base))

    if isempty(subdirs)
        return base
    end

    return joinpath(base, sort(subdirs)[end])
end

const RESULT_DIR = resolve_result_dir(ARGS)
const RESULT_BASE = joinpath(@__DIR__, "results")
println("Loading results from: $RESULT_DIR")

"""
    resolve_file(filename::AbstractString) -> String

Find a result file in `RESULT_DIR`, falling back to the base `results/`
directory. This lets run-specific TS-DDR results coexist with shared
baselines (SDDP, base-stock, random) that were generated once and live
in the parent directory.

# Arguments
- `filename::AbstractString`: file name (not a full path).

# Examples
```julia
path = resolve_file("relaxed_sddp_costs.csv")
```
"""
function resolve_file(filename::AbstractString)
    primary = joinpath(RESULT_DIR, filename)
    isfile(primary) && return primary

    fallback = joinpath(RESULT_BASE, filename)
    isfile(fallback) && return fallback

    error("Result file \"$filename\" not found in $RESULT_DIR or $RESULT_BASE")
end

"""
    resolve_file_optional(filename::AbstractString) -> Union{String, Nothing}

Like `resolve_file`, but returns `nothing` when the file does not exist in
either directory.

# Arguments
- `filename::AbstractString`: file name (not a full path).

# Examples
```julia
path = resolve_file_optional("integer_sf_training_curve.csv")
```
"""
function resolve_file_optional(filename::AbstractString)
    primary = joinpath(RESULT_DIR, filename)
    isfile(primary) && return primary

    fallback = joinpath(RESULT_BASE, filename)
    isfile(fallback) && return fallback

    return nothing
end

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
    table = CSV.read(resolve_file("$(tag)_$(method)_costs.csv"), DataFrame)

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
    filename = "$(tag)_$(method)_costs.csv"
    primary = joinpath(RESULT_DIR, filename)
    fallback = joinpath(RESULT_BASE, filename)

    path = isfile(primary) ? primary : isfile(fallback) ? fallback : nothing

    return isnothing(path) ? nothing :
        Float64.(CSV.read(path, DataFrame).operational_cost)
end

"""
    read_scalar(path::AbstractString) -> Float64

Read a scalar floating-point value from a text file.

# Arguments
- `path::AbstractString`: text file containing one numeric value.

# Examples
```julia
bound = read_scalar(resolve_file("integer_sddp_bound.txt"))
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
            path = resolve_file_optional("$(tag)_$(suffix).csv")
            !isnothing(path) && push!(rows, CSV.read(path, DataFrame))
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
        "TS-DDR (HighPenalty)" => "TS-DDR\n(HighPen)",
        "TS-DDR (LSTM)" => "TS-DDR\n(LSTM)",
        "TS-DDR (LSTM+SF)" => "TS-DDR\n(LSTM+SF)",
        "TS-DDR (trained)" => "TS-DDR",
        "TS-DDR Relaxed (LSTM)" => "TS-DDR\n(LSTM)",
        "TS-DDR Relaxed (HighPenalty)" => "TS-DDR\n(HighPen)",
        "TS-DDR Relaxed (LSTM+HP)" => "TS-DDR\n(LSTM+HP)",
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
    # Color assignments: TS-DDR variants in blues/oranges, SDDP in greens,
    # baselines in warm tones and gray.
    color_bank = [
        :steelblue, :royalblue, :darkorange, :mediumpurple,
        :coral, :teal, :darkgreen, :seagreen, :gold, :gray,
    ]

    num_methods <= length(color_bank) && return color_bank[1:num_methods]

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
function plot_sddp_learning_curve(tag::AbstractString; start_fraction::Float64 = 0.5)
    # SDDP logs may include failed or missing simulation rows.
    log = CSV.read(resolve_file("$(tag)_sddp_training_log.csv"), DataFrame)

    # Log-scale plots require strictly positive finite values.
    valid_bound_rows = filter(
        row -> !ismissing(row.bound) && isfinite(row.bound) && row.bound > 0,
        log,
    )

    # Show only the converged portion of training.
    start_iter = round(Int, start_fraction * maximum(valid_bound_rows.iteration))
    converged = filter(row -> row.iteration >= start_iter, valid_bound_rows)

    plot_handle = plot(
        converged.iteration,
        converged.bound;
        xlabel = "Iteration",
        ylabel = "Cost (log scale)",
        title = "SDDP learning curve (converged)",
        label = "LP bound",
        linewidth = 2,
        color = :darkgreen,
        legend = :right,
        yscale = :log10,
    )

    if "simulation_value" in names(converged)
        valid_sim_rows = filter(
            row -> !ismissing(row.simulation_value) &&
                isfinite(row.simulation_value) &&
                row.simulation_value > 0,
            converged,
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
        ylabel = "Out-of-sample rollout cost",
        title = "TS-DDR training curves",
        legend = :topright,
    )

    for (tag, label, color) in curve_specs
        # Optional variants should not break the plot.
        path = resolve_file_optional("$(tag)_training_curve.csv")
        isnothing(path) && continue

        curve = CSV.read(path, DataFrame)

        # Prefer the true out-of-sample rollout cost; fall back to the
        # DE training objective for data generated before the rollout
        # evaluation was added.
        if "rollout_cost" in names(curve)
            valid = dropmissing(curve, :rollout_cost)
            valid = filter(row -> isfinite(row.rollout_cost), valid)
            plot!(plot_handle, valid.batch, valid.rollout_cost;
                  label = label, linewidth = 2, color = color)
        else
            plot!(plot_handle, curve.batch, curve.loss;
                  label = label, linewidth = 2, color = color)
        end
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
    dr_paths = CSV.read(resolve_file("$(dr_tag)_dr_trajectories.csv"), DataFrame)
    base_stock_paths = CSV.read(
        resolve_file("$(baseline_tag)_basestock_trajectories.csv"),
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
        xrotation = 30,
        bottom_margin = 8Plots.mm,
        xtickfontsize = 7,
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
        size = (1200, 900),
        plot_title = problem,
        plot_titlefontsize = 12,
        margin = 6Plots.mm,
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

    # Load optional tuned-variant costs.
    lstm_costs = optional_costs("relaxed_lstm", "dr")
    hp_costs = optional_costs("relaxed_hp", "dr")
    lstm_hp_costs = optional_costs("relaxed_lstm_hp", "dr")

    # Load scalar baseline metadata.
    base_stock_level = read_scalar(resolve_file("relaxed_basestock_S_star.txt"))
    sddp_bound = read_scalar(resolve_file("relaxed_sddp_bound.txt"))

    # Build display records in table order.
    results = [
        MethodResult("TS-DDR (trained)", dr_costs),
    ]

    # Insert tuned variants after the baseline feedforward.
    !isnothing(hp_costs) &&
        push!(results, MethodResult("TS-DDR Relaxed (HighPenalty)", hp_costs))
    !isnothing(lstm_costs) &&
        push!(results, MethodResult("TS-DDR Relaxed (LSTM)", lstm_costs))
    !isnothing(lstm_hp_costs) &&
        push!(results, MethodResult("TS-DDR Relaxed (LSTM+HP)", lstm_hp_costs))

    # Append non-TS-DDR baselines.
    push!(results, MethodResult("SDDP (PAR)", sddp_costs))
    push!(results, MethodResult("Base-stock (S*=$(round(Int, base_stock_level)))", base_stock_costs))
    push!(results, MethodResult("Random (untrained)", random_costs))

    # Collect timing tags for all present variants.
    timing_tags = ["relaxed"]
    !isnothing(resolve_file_optional("relaxed_lstm_dr_timing.csv")) &&
        push!(timing_tags, "relaxed_lstm")
    !isnothing(resolve_file_optional("relaxed_hp_dr_timing.csv")) &&
        push!(timing_tags, "relaxed_hp")
    !isnothing(resolve_file_optional("relaxed_lstm_hp_dr_timing.csv")) &&
        push!(timing_tags, "relaxed_lstm_hp")

    return results, load_timing(timing_tags), base_stock_level, sddp_bound
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

    # Optional variants — load only if the result file exists.
    mixed_gradient_costs = optional_costs("integer_sf", "dr")
    hp_costs = optional_costs("integer_hp", "dr")
    lstm_costs = optional_costs("integer_lstm", "dr")
    lstm_sf_costs = optional_costs("integer_lstm_sf", "dr")

    # Load scalar baseline metadata.
    base_stock_level = read_scalar(resolve_file("integer_basestock_S_star.txt"))
    sddp_bound = read_scalar(resolve_file("integer_sddp_bound.txt"))

    # Build the method list: original TS-DDR variants first.
    results = [
        MethodResult("TS-DDR (FixedDiscrete)", fixed_discrete_costs),
        MethodResult("TS-DDR (ContRelax)", continuous_relaxation_costs),
    ]

    # Insert optional TS-DDR variants in logical order.
    !isnothing(mixed_gradient_costs) &&
        push!(results, MethodResult("TS-DDR (MixedGrad)", mixed_gradient_costs))
    !isnothing(hp_costs) &&
        push!(results, MethodResult("TS-DDR (HighPenalty)", hp_costs))
    !isnothing(lstm_costs) &&
        push!(results, MethodResult("TS-DDR (LSTM)", lstm_costs))
    !isnothing(lstm_sf_costs) &&
        push!(results, MethodResult("TS-DDR (LSTM+SF)", lstm_sf_costs))

    # Append non-TS-DDR baselines.
    push!(results, MethodResult("SDDP (MIP fwd)", sddp_mip_forward_costs))
    push!(results, MethodResult("SDDP (LP relax)", sddp_lp_relaxation_costs))
    push!(results, MethodResult("Base-stock (S*=$(round(Int, base_stock_level)))", base_stock_costs))
    push!(results, MethodResult("Random (untrained)", random_costs))

    # Collect timing tags for all present variants.
    timing_tags = ["integer", "integer_cr"]
    for tag in ["integer_sf", "integer_hp", "integer_lstm", "integer_lstm_sf"]
        !isnothing(resolve_file_optional("$(tag)_dr_timing.csv")) &&
            push!(timing_tags, tag)
    end

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

    # Optional variants appear only when their training curve exists.
    optional = [
        ("integer_sf", "MixedGrad", :darkorange),
        ("integer_hp", "HighPenalty", :mediumpurple),
        ("integer_lstm", "LSTM", :coral),
        ("integer_lstm_sf", "LSTM+SF", :teal),
    ]

    for spec in optional
        !isnothing(resolve_file_optional("$(spec[1])_training_curve.csv")) &&
            push!(specs, spec)
    end

    return specs
end

"""
    relaxed_curve_specs()

Return training-curve plot specs for the relaxed comparison.

# Examples
```julia
curves = relaxed_curve_specs()
```
"""
function relaxed_curve_specs()
    # Baseline feedforward is always present.
    specs = [("relaxed", "Feedforward", :steelblue)]

    # Optional tuned variants.
    optional = [
        ("relaxed_hp", "HighPenalty", :mediumpurple),
        ("relaxed_lstm", "LSTM", :coral),
        ("relaxed_lstm_hp", "LSTM+HP", :teal),
    ]

    for spec in optional
        !isnothing(resolve_file_optional("$(spec[1])_training_curve.csv")) &&
            push!(specs, spec)
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
        curve_specs = relaxed_curve_specs(),
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
