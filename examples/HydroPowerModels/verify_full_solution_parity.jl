#!/usr/bin/env julia

# Full-solution parity gate for the generated Bolivia stage subproblem.
#
# Replays a per-stage DECISION TRACE — incoming reservoir state, realized inflow
# and reservoir target, one row per (scenario, stage, reservoir) — through the
# serialized JuMP/MathOptFormat stage model produced by `export_subproblem_mof.jl`,
# and writes every physical primal variable of every solved stage in the shared
# long format of `hydro_solution_schema.jl`.
#
# Given a REFERENCE solution in the same format, it additionally differences the
# two variable by variable and prints the maximum absolute and relative
# disagreement per variable class.
#
# ── What this proves, and what it does not ────────────────────────────────────
#
# Fixing the trace fixes everything the policy or the cut-based value function
# decides. What is left is the stage model itself: its feasible set, its
# objective, and the serialization round trip. So a matching replay says the
# regenerated MOF IS the frozen production problem; a mismatch localizes to the
# model, not to the policy. It deliberately says nothing about whether the policy
# is good — the paired evaluation does that.
#
# Both engines are asked for the same operating point, but they are not the same
# solver: this path is Ipopt in Float64 on a JuMP model read from disk, while the
# ExaModels path is MadNLP on a GPU-resident `ExaModel`. Interior-point solvers
# stop at their own tolerance, so agreement is expected at solver tolerance, not
# at machine precision, and the report prints the achieved numbers rather than a
# pass/fail verdict computed from a threshold nobody stated.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#
#   julia --project=. verify_full_solution_parity.jl \
#       --trace=<decisions.csv> \
#       --out=<replay_solution.csv> \
#       [--reference=<engine_solution.csv>] \
#       [--tol=1e-10] [--label=tsddr_C3_scenario2]
#
# Trace CSV columns (header required):
#   scenario,stage,reservoir,state_in,target,inflow
#
# Environment:
#   DR_PARITY_FORMULATION   default "ACPPowerModel"

using DecisionRules
using JuMP, Ipopt
using JSON
using Printf
using Statistics

const HYDRO_DIR = dirname(@__FILE__)
include(joinpath(HYDRO_DIR, "load_hydropowermodels.jl"))
include(joinpath(HYDRO_DIR, "generate_canonical_case_artifacts.jl"))
include(joinpath(HYDRO_DIR, "hydro_solution_schema.jl"))
using .HydroCanonicalCase
using .HydroSolutionSchema

"""
    option(prefix, default=nothing) -> Union{Nothing,String}

Value of the first `--key=value` argument matching `prefix`.
"""
function option(prefix, default = nothing)
    index = findfirst(value -> startswith(value, prefix), ARGS)
    return isnothing(index) ? default : String(split(ARGS[index], '='; limit = 2)[2])
end

"""
    read_trace(path) -> Vector{NamedTuple}

Load the decision trace, grouped by `(scenario, stage)` in first-appearance
order, with the per-reservoir vectors ordered by reservoir index.

Every stage must carry a value for every reservoir: a stage that is missing one
would be replayed with a stale parameter, which is exactly the kind of silent
omission this gate exists to rule out.
"""
function read_trace(path::AbstractString, nhyd::Int)
    isfile(path) || error("missing decision trace: $path")
    rows = Dict{Tuple{Int,Int},Dict{Int,NTuple{3,Float64}}}()
    order = Tuple{Int,Int}[]
    open(path) do io
        header = split(strip(readline(io)), ',')
        required = ["scenario", "stage", "reservoir", "state_in", "target", "inflow"]
        header == required ||
            error("trace header must be $(join(required, ',')); got $(join(header, ','))")
        for line in eachline(io)
            isempty(strip(line)) && continue
            f = split(line, ',')
            key = (parse(Int, f[1]), parse(Int, f[2]))
            haskey(rows, key) || (rows[key] = Dict{Int,NTuple{3,Float64}}(); push!(order, key))
            reservoir = parse(Int, f[3])
            rows[key][reservoir] =
                (parse(Float64, f[4]), parse(Float64, f[5]), parse(Float64, f[6]))
        end
    end
    stages = NamedTuple[]
    for key in order
        entry = rows[key]
        length(entry) == nhyd || error(
            "trace stage $(key) has $(length(entry)) reservoirs, expected $nhyd",
        )
        push!(stages, (
            scenario = key[1], stage = key[2],
            state_in = [entry[r][1] for r in 1:nhyd],
            target = [entry[r][2] for r in 1:nhyd],
            inflow = [entry[r][3] for r in 1:nhyd],
        ))
    end
    return stages
end

"""
    residuals(model) -> (max_equality, max_inequality)

Largest absolute equality residual and largest inequality violation over every
constraint of the solved model, variable bounds included.

Reported because two models can agree on every variable and still differ in what
they enforce; a residual that is large on one side and zero on the other means
the two feasible sets are not the same set.
"""
function residuals(model)
    max_eq = 0.0
    max_ineq = 0.0
    for (F, S) in JuMP.list_of_constraint_types(model)
        for con in JuMP.all_constraints(model, F, S)
            object = JuMP.constraint_object(con)
            v = try
                JuMP.value(object.func)
            catch
                continue
            end
            set = object.set
            if set isa MOI.EqualTo
                max_eq = max(max_eq, abs(v - set.value))
            elseif set isa MOI.LessThan
                max_ineq = max(max_ineq, max(0.0, v - set.upper))
            elseif set isa MOI.GreaterThan
                max_ineq = max(max_ineq, max(0.0, set.lower - v))
            elseif set isa MOI.Interval
                max_ineq = max(max_ineq, max(0.0, v - set.upper, set.lower - v))
            end
        end
    end
    return max_eq, max_ineq
end

"""
    solve_replay!(model, tolerances) -> (ok, status, tol_used, attempts)

Solve the replayed stage, walking `tolerances` from strictest to loosest and
retrying each level once from a cold start.

Two separate reasons this is needed, both properties of the REPLAY rather than
of the model:

* the target being pinned is another solver's iterate, so it satisfies that
  solver's water balance only to its own tolerance. Requiring an equality to it
  at 1e-10 asks for more precision than the recorded number carries, and Ipopt
  correctly reports `LOCALLY_INFEASIBLE`;
* a warm iterate left over from the previous stage can strand the interior-point
  method, which is why the production evaluators also retry from a cleared
  start.

The tolerance actually achieved is recorded per stage and reported, so a stage
that needed a looser setting is visible rather than absorbed.
"""
function solve_replay!(model, tolerances)
    attempts = 0
    for tol in tolerances
        for attribute in ("tol", "constr_viol_tol", "acceptable_tol")
            JuMP.set_attribute(model, attribute, tol)
        end
        for cold in (false, true)
            if cold
                for variable in JuMP.all_variables(model)
                    JuMP.set_start_value(variable, nothing)
                end
            end
            attempts += 1
            optimize!(model)
            status = JuMP.termination_status(model)
            if status in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
                return (true, status, tol, attempts)
            end
        end
    end
    return (false, JuMP.termination_status(model), last(tolerances), attempts)
end

function main()
    case_dir = joinpath(HYDRO_DIR, "bolivia")
    formulation = get(ENV, "DR_PARITY_FORMULATION", "ACPPowerModel")
    trace_path = option("--trace=")
    out_path = option("--out=")
    reference_path = option("--reference=")
    label = option("--label=", "parity")
    tol = parse(Float64, option("--tol=", "1e-10"))
    isnothing(trace_path) && error("--trace=<decisions.csv> is required")
    isnothing(out_path) && error("--out=<replay_solution.csv> is required")

    verify_inputs(case_dir)
    verify_index_convention(case_dir, JSON.parsefile)

    # One stage model, reused for every replayed stage. Every stage of this case
    # is the same model with different parameter values — that is precisely the
    # design `build_hydropowermodels` implements — so reading 126 copies would
    # only make the gate slower and less obviously identical across stages.
    # Strictest first, then progressively looser. The strictest level is the one
    # reported when it succeeds; the ladder exists because the replayed target is
    # another solver's iterate (see `solve_replay!`), not because the model needs
    # slack.
    tolerances = [tol, 1e-8, 1e-6]
    optimizer = optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level" => 0,
        "linear_solver" => "mumps",
        "tol" => tol,
        "constr_viol_tol" => tol,
        "acceptable_tol" => tol,
        "max_iter" => 5000,
    )
    subproblems, state_params_in, state_params_out, uncertainty_samples,
        initial_state, _, hydro_meta = build_hydropowermodels(
        case_dir, formulation * ".mof.json";
        num_stages = 1, optimizer = optimizer, strict = true,
    )
    model = subproblems[1]
    nhyd = hydro_meta.nHyd

    # The inflow parameters, in reservoir order. `uncertainty_samples` holds
    # (parameter, value) pairs per scenario; the parameters are shared, so the
    # first sample is enough to recover them.
    inflow_params = [pair[1] for pair in uncertainty_samples[1][1]]
    inflow_order = [
        parse(Int, match(r"inflow\[(\d+)\]", JuMP.name(p)).captures[1])
        for p in inflow_params
    ]

    orientation = branch_orientation(case_dir, JSON.parsefile)
    variables = JuMP.all_variables(model)
    mapped = [(v, solution_class(JuMP.name(v), orientation)) for v in variables]
    mapped = [(v, c) for (v, c) in mapped if c !== nothing]

    covered = sort(unique(c[1] for (_, c) in mapped))
    @info "variable mapping" n_variables = length(variables) n_mapped = length(mapped) covered

    trace = read_trace(trace_path, nhyd)
    @info "replaying" label trace = trace_path stages = length(trace) formulation tol

    writer = SolutionWriter(out_path)
    status_rows = Tuple{Int,Int,String,Float64,Float64,Float64,Int}[]
    cumulative = Dict{Int,Float64}()
    # Costs are REPORTED over stages 1:96; the 30 look-ahead stages are simulated
    # but never priced into a published number, so both sums are printed and the
    # reported one is the number to compare against the frozen result.
    reported = Dict{Int,Float64}()
    failures = 0

    for entry in trace
        for j in 1:nhyd
            set_parameter_value(state_params_in[1][j], entry.state_in[j])
            set_parameter_value(state_params_out[1][j][1], entry.target[j])
        end
        for (k, p) in enumerate(inflow_params)
            set_parameter_value(p, entry.inflow[inflow_order[k]])
        end

        ok, status, tol_used, attempts = solve_replay!(model, tolerances)
        if !ok
            failures += 1
            @error "stage solve did not converge at any tolerance" entry.scenario entry.stage status
            push!(status_rows,
                  (entry.scenario, entry.stage, string(status), NaN, NaN, NaN, attempts))
            continue
        end

        for (v, (class, index)) in mapped
            record!(writer, entry.scenario, entry.stage, class, index, JuMP.value(v))
        end
        for j in 1:nhyd
            record!(writer, entry.scenario, entry.stage, "target", j, entry.target[j])
            record!(
                writer, entry.scenario, entry.stage, "target_multiplier", j,
                DecisionRules.pdual(state_params_out[1][j][1]),
            )
        end
        objective = JuMP.objective_value(model)
        cumulative[entry.scenario] = get(cumulative, entry.scenario, 0.0) + objective
        if entry.stage <= REPORTING_STAGES
            reported[entry.scenario] = get(reported, entry.scenario, 0.0) + objective
        end
        record_scalar!(writer, entry.scenario, entry.stage, "stage_objective", objective)
        record_scalar!(
            writer, entry.scenario, entry.stage, "cum_objective",
            cumulative[entry.scenario],
        )

        eq, ineq = residuals(model)
        push!(status_rows,
              (entry.scenario, entry.stage, string(status), eq, ineq, tol_used, attempts))
    end
    close(writer)

    status_path = replace(out_path, r"\.csv$" => "_status.csv")
    open(status_path, "w") do io
        println(io, "scenario,stage,status,max_equality_residual," *
                    "max_inequality_violation,solver_tolerance,attempts")
        for row in status_rows
            println(io, join(row, ","))
        end
    end

    println("\n", "="^96)
    println("REPLAY  $label   ($(length(trace)) stages, $formulation, Ipopt tol $tol)")
    println("="^96)
    @printf("  solved                       %d / %d\n", length(trace) - failures, length(trace))
    finite = [r for r in status_rows if isfinite(r[4])]
    if !isempty(finite)
        @printf("  max equality residual        %.3e\n", maximum(r[4] for r in finite))
        @printf("  max inequality violation     %.3e\n", maximum(r[5] for r in finite))
        for level in tolerances
            n = count(r -> r[6] == level, finite)
            n == 0 || @printf("  stages solved at tol %-8.0e %d\n", level, n)
        end
        retried = count(r -> r[7] > 1, finite)
        retried == 0 || @printf("  stages needing a retry       %d\n", retried)
    end
    for (scenario, value) in sort(collect(cumulative))
        n = count(e -> e.scenario == scenario, trace)
        @printf("  scenario %4d  stages 1:%-3d = %.6f   REPORTED 1:%d = %.6f\n",
                scenario, n, value, REPORTING_STAGES, get(reported, scenario, NaN))
    end
    println("  solution  -> $out_path")
    println("  statuses  -> $status_path")

    failures == 0 || error("$failures stage(s) failed to converge; parity gate stops here")

    if reference_path !== nothing
        gen_bus = generator_bus(case_dir, JSON.parsefile)
        replay = augment_nodal!(read_solution(out_path), gen_bus)
        reference = augment_nodal!(read_solution(reference_path), gen_bus)
        differences, missing_keys = compare_solutions(reference, replay)
        println("\n", "="^96)
        println("PARITY  reference = $reference_path")
        println("="^96)
        print(difference_table(differences))
        # One-sided classes are reported by CLASS with a count and a reason, not
        # as a list of thousands of keys. Two are structurally one-sided and are
        # named here so that anything else stands out as a genuine omission:
        #
        #   target_multiplier — the dual of the strict `reservoir_out == target`
        #     equality. It exists only where that equality exists: the strict
        #     replay always has it, an SDDP subproblem never does (its outgoing
        #     level is a decision, not a pinned parameter).
        #   min_volume_violation / min_outflow_violation — HydroPowerModels'
        #     unpriced slacks, which the ExaModels builder does not create at all
        #     (it enforces the minimums directly). Both minimums are zero in this
        #     case, so the constraint sets still coincide.
        expected_one_sided = ("target_multiplier", UNPRICED_SLACK_CLASSES...)
        if isempty(missing_keys)
            println("\n  every scalar present in both solutions (no omitted variable)")
        else
            counts = Dict{String,Int}()
            for k in missing_keys
                counts[k[3]] = get(counts, k[3], 0) + 1
            end
            println("\n  present in only one solution, by class:")
            unexpected = String[]
            for class in sort(collect(keys(counts)))
                tag = class in expected_one_sided ? "structural" : "** UNEXPECTED **"
                @printf("    %-22s %6d   %s\n", class, counts[class], tag)
                class in expected_one_sided || push!(unexpected, class)
            end
            isempty(unexpected) || error(
                "physical variable(s) present in only one solution: " *
                join(unexpected, ", ") * " — the gate requires that no physical " *
                "variable be silently omitted",
            )
        end
    end
end

(abspath(PROGRAM_FILE) == @__FILE__) && main()
