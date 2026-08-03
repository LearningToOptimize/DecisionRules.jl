# Paired SDDP simulation on the seeded paired protocol via SDDP.Historical.
#
# Scenario indices are generated from PAIRED_SCENARIO_SEED (identical to
# eval_paired_tsddr.jl's protocol), so the SDDP policy is simulated under the
# exact inflow realizations every TS-DDR evaluation uses.
#
# Usage:
#   julia --project -t auto eval_paired_sddp.jl
using MadNLP
using StableRNGs
using HydroPowerModels
using JuMP
using PowerModels
using Statistics
using SDDP: SDDP
using DelimitedFiles
using CSV, DataFrames
using JSON

const CASE = "bolivia"
const SDDP_DIR = dirname(@__FILE__)
const HYDRO_DIR = dirname(SDDP_DIR)
const CASE_DIR = joinpath(HYDRO_DIR, CASE)
const RM_STAGES = 30
const REPORT_STAGES = 96
const NUM_STAGES = REPORT_STAGES + RM_STAGES
const FORMULATION = ACPPowerModel
const FORMULATION_B = SOCWRConicPowerModel

include(joinpath(HYDRO_DIR, "hydro_solution_schema.jl"))
using .HydroSolutionSchema

# The frozen case has DETERMINISTIC demand: `0.6 x PowerModels.json` active and
# reactive load at every stage, and inflow as the only uncertainty. This is
# asserted rather than assumed — a demand file appearing in the case directory
# would silently change the stochastic program, so its absence is checked here
# and again by `generate_canonical_case_artifacts.jl --verify`.
for name in ("demand.csv", "demand_scenarios.csv", "demand_noise.csv")
    isfile(joinpath(CASE_DIR, name)) && error(
        "$name is present in $CASE_DIR. The frozen experiment has deterministic " *
        "demand and inflow-only uncertainty; a demand file means a different " *
        "stochastic program and different cuts.",
    )
end
@info "Demand model" deterministic = true uncertainty = "inflow only"

# ── Paired scenario indices (seeded protocol) ──────────────────────────────
# Identical generation to load_hydropowermodels.jl's paired_scenario_indices:
# entry [t, s] is uniform on 1:nCen from StableRNG(PAIRED_SCENARIO_SEED), so
# this script and every TS-DDR evaluation realize the same inflow at the same
# stage of the same paired scenario — with no shared data file. nCen is
# derived from the inflow data (columns ÷ hydro units), never hardcoded.
const PAIRED_SCENARIO_SEED = 20260706
# Fixed generated shape shared by ALL consumers (see load_hydropowermodels.jl:
# arrays of different shapes consume the RNG stream differently, so every
# script must generate exactly this shape and slice what it needs).
const PAIRED_NUM_STAGES = 126
const N_HYDRO = 11
@assert NUM_STAGES == PAIRED_NUM_STAGES "SDDP horizon must equal the paired protocol shape"
num_scenarios = parse(Int, get(ENV, "DR_NUM_SCENARIOS", "500"))
nCen = div(size(readdlm(joinpath(HYDRO_DIR, CASE, "inflows.csv"), ','), 2), N_HYDRO)
# ALWAYS generate the full protocol matrix (fixed shape — see the note above),
# then optionally simulate only a shard of its columns so the 500 scenarios
# can run in parallel across nodes (DR_SCENARIO_FIRST/LAST, 1-based inclusive).
all_indices = rand(StableRNG(PAIRED_SCENARIO_SEED), 1:nCen, PAIRED_NUM_STAGES, num_scenarios)
scen_first = parse(Int, get(ENV, "DR_SCENARIO_FIRST", "1"))
scen_last = parse(Int, get(ENV, "DR_SCENARIO_LAST", string(num_scenarios)))
@assert 1 <= scen_first <= scen_last <= num_scenarios
scen_range = scen_first:scen_last
println("Paired protocol: seed=$PAIRED_SCENARIO_SEED, $(PAIRED_NUM_STAGES)×$(num_scenarios), nCen=$nCen")
println("Evaluating scenarios $scen_first:$scen_last, $REPORT_STAGES reported stages (of $NUM_STAGES total)")

# ── Build SDDP model and load cuts ────────────────────────────────────────
global alldata = HydroPowerModels.parse_folder(CASE_DIR; stages=NUM_STAGES)
for data in alldata
    for load in values(data["powersystem"]["load"])
        load["pd"] *= 0.6
        load["qd"] *= 0.6
    end
    data["powersystem"]["cost_deficit"] = 6000.0 / data["powersystem"]["baseMVA"]
end
@info "Canonical MAIN demand" pd_scale=0.6 qd_scale=0.6 deficit_cost=6000.0

stage_hours = Int(get(alldata[1]["hydro"], "stage_hours", 1))  # K = 0.0036·stage_hours
params = create_param(;
    stages=NUM_STAGES,
    stage_hours=stage_hours,
    model_constructor_grid=FORMULATION,
    post_method=PowerModels.build_opf,
    # Hardened solver settings for the 500-scenario simulation: with bare
    # defaults one nonconvex ACP node failed to converge on one seeded draw
    # (SDDP aborts the whole simulation on any node failure). Larger
    # iteration budget + explicit tolerance make every node solvable; the
    # cuts themselves are unaffected (they were trained separately).
    optimizer=() -> MadNLP.Optimizer(;
        print_level=0,
        max_iter=9000,
        tol=1e-6,
    ),
)

m = hydro_thermal_operation(alldata, params)

# DR_CUTS_FILE lets the launcher point at a stable SNAPSHOT of the live cuts
# (the training job rewrites the canonical file every iteration → reading it
# directly risks a torn JSON). Defaults to the canonical inconsistent-run path.
cuts_file = get(ENV, "DR_CUTS_FILE", joinpath(
    CASE_DIR,
    string(FORMULATION),
    string(FORMULATION_B) * "-" * string(FORMULATION) * ".cuts.json",
))
SDDP.read_cuts_from_file(m.forward_graph, cuts_file)
println("Loaded cuts: $cuts_file")

# ── Build SDDP.Historical sampling scheme ──────────────────────────────────
# Each scenario is a vector of (node, noise_term) pairs: node = stage index,
# noise term = the inflow atom this protocol column realizes at that stage.
historical_scenarios = [
    [(t, all_indices[t, s]) for t in 1:NUM_STAGES]
    for s in scen_range
]
n_sim = length(scen_range)

sampling_scheme = SDDP.Historical(historical_scenarios)

# ── Simulate ───────────────────────────────────────────────────────────────
# `DR_PHYSICAL_AUDIT=1` records the PHYSICAL decision variables of every solved
# stage subproblem alongside the stock HydroPowerModels recorders. It cannot go
# through `HydroPowerModels.simulate`, which hardcodes its `custom_recorders`
# and drops `kwargs...`; so the audit path calls `SDDP.simulate` directly with a
# SUPERSET of the stock recorder dictionary and rebuilds the same result
# `Dict`. The extra recorders only READ values off models the simulation has
# already solved — no extra solve, and the recorded costs are bit-identical to
# the un-instrumented run (verified against the existing shard costs).
const PHYSICAL_AUDIT = get(ENV, "DR_PHYSICAL_AUDIT", "0") == "1"
# `DR_SOLUTION_DUMP=1` additionally records the FULL physical solution of every
# simulated stage — every named primal variable plus the nodal prices — in the
# shared long format of `hydro_solution_schema.jl`, together with the decision
# trace (incoming state, realized inflow, outgoing reservoir level).
#
# This is the only path by which per-bus physics leaves the solver. The four
# aggregate CSVs above answer "how much thermal, how much water, was any load
# shed"; they cannot answer "what was energy worth at bus 14 in week 62", which
# is a dual and exists nowhere else. The stagewise and price figures are built
# from this dump.
const SOLUTION_DUMP = get(ENV, "DR_SOLUTION_DUMP", "0") == "1"
SOLUTION_DUMP && !PHYSICAL_AUDIT &&
    error("DR_SOLUTION_DUMP=1 requires DR_PHYSICAL_AUDIT=1 (it rides on the same recorders)")
# Stages whose full solution is dumped. Defaults to the whole simulated horizon,
# not the reported window: the look-ahead stages are part of the trajectory even
# though no cost is reported from them.
const SOLUTION_DUMP_STAGES = parse(Int, get(ENV, "DR_SOLUTION_DUMP_STAGES", string(NUM_STAGES)))
println("\nSimulating $n_sim scenarios with SDDP.Historical...")
results = if !PHYSICAL_AUDIT
    HydroPowerModels.simulate(m, n_sim; sampling_scheme=sampling_scheme)
else
    println("PHYSICAL AUDIT enabled: recording per-bus deficit[b] and hydro decisions")
    sims = SDDP.simulate(
        m.forward_graph, n_sim;
        sampling_scheme=sampling_scheme,
        custom_recorders=Dict{Symbol,Function}(
            # ── stock HydroPowerModels recorders (kept identical) ──────────
            :powersystem => HydroPowerModels.build_sol_powermodels,
            :reservoirs => HydroPowerModels.build_sol_reservoirs,
            :objective => objective_value,
            # ── PHYSICAL load shedding: the per-bus active-power balance
            #    slack `deficit[b]` (pu). This — and ONLY this — is load
            #    shedding; the reservoir-target penalty bookkeeping is a
            #    different quantity entirely.
            :deficit_bus => sp -> Vector{Float64}(JuMP.value.(sp[:deficit])),
            # ── hydro physical decisions (pu-volume units of the case) ─────
            :outflow_r => sp -> Vector{Float64}(JuMP.value.(sp[:outflow])),
            :spill_r => sp -> Vector{Float64}(JuMP.value.(sp[:spill])),
            :inflow_r => sp -> Vector{Float64}(JuMP.value.(sp[:inflow])),
            # ── objective decomposition: `sp.ext[:cost]` holds the JuMP
            #    expression of each additive term of the stage objective
            #    (generation, spill, deficit, …), so the stage cost can be
            #    attributed without re-deriving it.
            :cost_gen => sp -> JuMP.value(sp.ext[:cost][:gen_cost]),
            :cost_deficit => sp -> JuMP.value(sp.ext[:cost][:deficit_cost]),
            :cost_spill => sp -> JuMP.value(sp.ext[:cost][:spill_cost]),
            # Remaining additive terms (minimal-outflow / minimal-volume
            # violation), so gen+deficit+spill+other == stage_objective and the
            # decomposition can be checked rather than assumed.
            :cost_other => sp -> sum(
                JuMP.value(e) for (k, e) in sp.ext[:cost]
                if !(k in (:gen_cost, :deficit_cost, :spill_cost));
                init=0.0,
            ),
            # ── solve status of the stage subproblem ───────────────────────
            :status => sp -> string(JuMP.termination_status(sp)),
            # ── FULL primal solution, by variable NAME ─────────────────────
            # Every named variable of the solved stage. Recorded only under
            # DR_SOLUTION_DUMP because it is one entry per variable per stage.
            # Reading them by their serialized names — the same names
            # `export_subproblem_mof.jl` writes — is what lets this solution be
            # compared, plotted or replayed elsewhere without re-deriving an
            # index convention.
            :named_solution => sp -> SOLUTION_DUMP ?
                Dict{String,Float64}(
                    JuMP.name(v) => JuMP.value(v) for v in JuMP.all_variables(sp)
                    if !isempty(JuMP.name(v))
                ) : Dict{String,Float64}(),
            # ── NODAL PRICES ───────────────────────────────────────────────
            # The dual of each bus's active-power balance is the locational
            # marginal price of energy at that bus (USD per pu per stage); the
            # reactive balance gives the price of reactive support. These are
            # the economic read-out of the dispatch — what the policy's water
            # decisions are worth to the network — and they exist only as duals,
            # so no primal recording can substitute for them.
            #
            # `lam_kcl_r` / `lam_kcl_i` are the constraint references
            # PowerModels stores per bus, and the same ones HydroPowerModels'
            # own `constraint_mod_deficit` uses to insert the load-shedding
            # variable, so the sign convention is the package's own.
            :price_active => sp -> SOLUTION_DUMP ?
                Float64[JuMP.dual(b[:lam_kcl_r])
                        for b in PowerModels.sol(sp.ext[:pm], 0, :bus)] : Float64[],
            :price_reactive => sp -> SOLUTION_DUMP ?
                Float64[JuMP.dual(b[:lam_kcl_i])
                        for b in PowerModels.sol(sp.ext[:pm], 0, :bus)] : Float64[],
        ),
    )
    Dict{Symbol,Any}(
        :simulations => sims,
        :params => m.params,
        :data => m.alldata,
    )
end

# The simulation must have realized the protocol's inflow atoms, not SDDP's own
# sampling. Checked against the independently generated index matrix.
for (si, s) in enumerate(first(scen_range, min(3, n_sim))), t in 1:min(5, REPORT_STAGES)
    recorded_ω = results[:simulations][si][t][:noise_term]
    if recorded_ω != all_indices[t, s]
        error("Mismatch at scenario $s, stage $t: got noise=$recorded_ω, expected $(all_indices[t, s])")
    end
end
println("Inflow protocol verification passed (spot-checked)")

# ── Extract results ────────────────────────────────────────────────────────
nhyd = alldata[1]["hydro"]["nHyd"]
volume_to_mw(volume; k=0.0036) = volume / k

objective_values = [
    sum(results[:simulations][i][t][:stage_objective] for t in 1:REPORT_STAGES)
    for i in 1:n_sim
]

hydro_vol = [
    mean(
        sum(
            volume_to_mw(results[:simulations][i][t][:reservoirs][:reservoir][j].out) for
            j in 1:nhyd
        ) for i in 1:n_sim
    ) for t in 1:REPORT_STAGES
]

num_gen = length(results[:simulations][1][1][:powersystem]["solution"]["gen"])
hydro_idx = HydroPowerModels.idx_hydro(results[:data][1])
thermal_gen = [
    mean(
        sum(
            results[:simulations][i][t][:powersystem]["solution"]["gen"]["$j"]["pg"] *
            results[:data][1]["powersystem"]["baseMVA"] for
            j in 1:num_gen if !(j in hydro_idx)
        ) for i in 1:n_sim
    ) for t in 1:REPORT_STAGES
]

# ── Report ─────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 60)
println("Results: Paired SDDP ($REPORT_STAGES stages, $num_scenarios scenarios)")
println("=" ^ 60)
println("  Mean cost:   $(round(mean(objective_values); digits=1))")
println("  Std:         $(round(std(objective_values); digits=1))")
println("  Min:         $(round(minimum(objective_values); digits=1))")
println("  Max:         $(round(maximum(objective_values); digits=1))")
println("  Median:      $(round(median(objective_values); digits=1))")
println("=" ^ 60)

# ── Save results ───────────────────────────────────────────────────────────
out_dir = joinpath(CASE_DIR, string(FORMULATION))

# ── PHYSICAL AUDIT OUTPUT ──────────────────────────────────────────────────
# Three CSVs per shard, written into `<out_dir>/audit/`:
#   *_deficit.csv  one row per (scenario, stage, bus) with NONZERO deficit
#                  above `AUDIT_TOL`, plus raw extrema, so both the raw and the
#                  thresholded views are recoverable;
#   *_stage.csv    one row per (scenario, stage): stage totals for shedding,
#                  thermal generation, hydro in/out/spill, cost decomposition,
#                  and solve status;
#   *_scenario.csv one row per scenario: totals + the cost that MUST reproduce
#                  the existing shard cost.
# baseMVA converts per-unit power to MW; deficit[b] is an active-power
# injection slack, so `deficit_MW = deficit_pu * baseMVA`.
const AUDIT_TOL = 1e-6                          # pu — numerical-tolerance gate
if PHYSICAL_AUDIT
    baseMVA = alldata[1]["powersystem"]["baseMVA"]
    audit_dir = joinpath(out_dir, "audit")
    isdir(audit_dir) || mkpath(audit_dir)
    tag = "sddp_$(scen_first)_$(scen_last)"

    def_rows = DataFrame(
        scenario=Int[], stage=Int[], bus=Int[],
        deficit_pu=Float64[], deficit_MW=Float64[],
    )
    stage_rows = DataFrame(
        scenario=Int[], stage=Int[],
        deficit_pu=Float64[], deficit_MW=Float64[],
        max_bus_deficit_pu=Float64[], argmax_bus=Int[], n_buses_shedding=Int[],
        thermal_MW=Float64[], hydro_MW=Float64[],
        inflow=Float64[], outflow=Float64[], spill=Float64[], storage=Float64[],
        cost_gen=Float64[], cost_deficit=Float64[], cost_spill=Float64[],
        cost_other=Float64[], stage_objective=Float64[], status=String[],
    )
    for (i, s) in enumerate(scen_range)
        for t in 1:REPORT_STAGES
            rec = results[:simulations][i][t]
            d = rec[:deficit_bus]                       # Vector{Float64}, pu
            # Raw per-bus rows, thresholded so the file stays readable; the
            # unthresholded extrema are carried in the stage row regardless.
            for (b, v) in enumerate(d)
                v > AUDIT_TOL && push!(def_rows, (s, t, b, v, v * baseMVA))
            end
            dmax, dargmax = findmax(d)
            gen = rec[:powersystem]["solution"]["gen"]
            th = sum(gen["$j"]["pg"] * baseMVA for j in 1:num_gen if !(j in hydro_idx))
            hy = sum(gen["$j"]["pg"] * baseMVA for j in 1:num_gen if j in hydro_idx)
            push!(stage_rows, (
                s, t,
                sum(d), sum(d) * baseMVA,
                dmax, dargmax, count(>(AUDIT_TOL), d),
                th, hy,
                sum(rec[:inflow_r]), sum(rec[:outflow_r]), sum(rec[:spill_r]),
                sum(rec[:reservoirs][:reservoir][j].out for j in 1:nhyd),
                rec[:cost_gen], rec[:cost_deficit], rec[:cost_spill],
                rec[:cost_other], rec[:stage_objective], rec[:status],
            ))
        end
    end

    # Per-scenario roll-up. `cost` here is the SAME sum as the shard file, so
    # equality with the existing shard CSV validates the instrumentation.
    # Per-reservoir storage / turbine outflow / spill, so a cost gap against
    # another policy can be attributed to individual reservoirs rather than only
    # to aggregate water. Mirrors the TS-DDR dump's `*_reservoir.csv` schema.
    res_rows = DataFrame(
        scenario=Int[], stage=Int[], reservoir=Int[],
        storage=Float64[], outflow=Float64[], spill=Float64[],
    )
    for (i, s) in enumerate(scen_range)
        for t in 1:REPORT_STAGES
            rec = results[:simulations][i][t]
            for r in 1:nhyd
                push!(res_rows, (
                    s, t, r,
                    rec[:reservoirs][:reservoir][r].out,
                    rec[:outflow_r][r], rec[:spill_r][r],
                ))
            end
        end
    end
    CSV.write(joinpath(audit_dir, "$(tag)_reservoir.csv"), res_rows)

    scen_rows = DataFrame(
        scenario=Int[], cost=Float64[],
        deficit_pu=Float64[], deficit_MW=Float64[],
        max_bus_stage_deficit_pu=Float64[], max_bus_stage_deficit_MW=Float64[],
        n_stages_with_deficit=Int[], n_bus_stage_with_deficit=Int[],
        cost_deficit=Float64[], cost_gen=Float64[], cost_spill=Float64[],
        all_stages_solved=Bool[],
    )
    for (i, s) in enumerate(scen_range)
        g = stage_rows[stage_rows.scenario .== s, :]
        push!(scen_rows, (
            s, objective_values[i],
            sum(g.deficit_pu), sum(g.deficit_MW),
            maximum(g.max_bus_deficit_pu), maximum(g.max_bus_deficit_pu) * baseMVA,
            count(>(AUDIT_TOL), g.deficit_pu), sum(g.n_buses_shedding),
            sum(g.cost_deficit), sum(g.cost_gen), sum(g.cost_spill),
            all(st -> st in ("LOCALLY_SOLVED", "OPTIMAL"), g.status),
        ))
    end

    CSV.write(joinpath(audit_dir, "$(tag)_deficit.csv"), def_rows)
    CSV.write(joinpath(audit_dir, "$(tag)_stage.csv"), stage_rows)
    CSV.write(joinpath(audit_dir, "$(tag)_scenario.csv"), scen_rows)

    # ── FULL PHYSICAL SOLUTION ────────────────────────────────────────────
    # Every named primal variable and both nodal prices, per stage, plus the
    # decision trace that reproduces the trajectory.
    if SOLUTION_DUMP
        verify_index_convention(CASE_DIR, JSON.parsefile)
        writer = SolutionWriter(joinpath(audit_dir, "$(tag)_solution.csv"))
        orientation = branch_orientation(CASE_DIR, JSON.parsefile)
        trace_rows = DataFrame(
            scenario=Int[], stage=Int[], reservoir=Int[],
            state_in=Float64[], target=Float64[], inflow=Float64[],
        )
        n_dump = min(SOLUTION_DUMP_STAGES, NUM_STAGES)
        for (i, s) in enumerate(scen_range)
            cumulative = 0.0
            for t in 1:n_dump
                rec = results[:simulations][i][t]
                for (name, value) in rec[:named_solution]
                    mapped = solution_class(name, orientation)
                    mapped === nothing && continue
                    record!(writer, s, t, mapped[1], mapped[2], value)
                end
                record_vector!(writer, s, t, "price_active", rec[:price_active])
                record_vector!(writer, s, t, "price_reactive", rec[:price_reactive])
                cumulative += rec[:stage_objective]
                record_scalar!(writer, s, t, "stage_objective", rec[:stage_objective])
                record_scalar!(writer, s, t, "cum_objective", cumulative)
                for r in 1:nhyd
                    state_in = rec[:reservoirs][:reservoir][r].in
                    state_out = rec[:reservoirs][:reservoir][r].out
                    # The cut policy's DECISION is the outgoing level, so it is
                    # also what a strict replay would be told to hit; recording
                    # it as `target` keeps one trace schema for both policies.
                    record!(writer, s, t, "target", r, state_out)
                    push!(trace_rows, (s, t, r, state_in, state_out, rec[:inflow_r][r]))
                end
            end
        end
        close(writer)
        CSV.write(joinpath(audit_dir, "$(tag)_trace.csv"), trace_rows)
        println("  Full solution + trace: $audit_dir/$(tag)_{solution,trace}.csv " *
                "($(n_dump) stages per scenario)")
    end

    println("\n" * "=" ^ 60)
    println("PHYSICAL LOAD-SHEDDING AUDIT  (deficit[b], tol=$AUDIT_TOL pu)")
    println("=" ^ 60)
    for r in eachrow(scen_rows)
        println("  scen $(r.scenario): cost=$(round(r.cost; digits=2)) " *
                "deficit=$(r.deficit_pu) pu ($(r.deficit_MW) MW-stage) " *
                "max_bus_stage=$(r.max_bus_stage_deficit_pu) pu " *
                "stages_shedding=$(r.n_stages_with_deficit)/$REPORT_STAGES " *
                "cost_deficit=$(r.cost_deficit) all_solved=$(r.all_stages_solved)")
    end
    println("  RAW MAX over all (scen,stage,bus): " *
            "$(maximum(stage_rows.max_bus_deficit_pu)) pu")
    println("  Audit CSVs: $audit_dir/$(tag)_{deficit,stage,scenario}.csv")
    println("=" ^ 60)
end

# Shard mode: emit only this shard's per-scenario costs (merged afterwards by
# merge_sddp_shards.jl); the full-run outputs below are skipped.
#
# The `scenario` column holds the GLOBAL protocol column id, never a
# shard-local 1..n index — that is what lets `merge_sddp_shards.jl` prove the
# shards partition the protocol, and what keeps the ids comparable to the
# TS-DDR side. `all_stages_solved` travels with the cost so an unsolved
# scenario stays visible through the merge instead of being averaged in.
if n_sim != num_scenarios
    shard_file = joinpath(out_dir, "sddp_shard_$(scen_first)_$(scen_last).csv")
    solved = if PHYSICAL_AUDIT
        [
            all(
                results[:simulations][i][t][:status] in ("LOCALLY_SOLVED", "OPTIMAL")
                for t in 1:REPORT_STAGES
            )
            for i in 1:n_sim
        ]
    else
        # Without the audit recorders the per-stage status is not read back.
        # SDDP.simulate itself aborts on a failed node, so reaching this line
        # means every stage solved; the column records that it was inferred
        # rather than observed.
        fill(true, n_sim)
    end
    CSV.write(shard_file, DataFrame(
        scenario = collect(scen_range),
        cost = objective_values,
        all_stages_solved = solved,
        status_observed = fill(PHYSICAL_AUDIT, n_sim),
    ))
    println("Shard written: $shard_file  (ids $scen_first:$scen_last, " *
            "$(count(solved))/$n_sim fully solved)")
    exit(0)
end

# Optional output tag (mirrors eval_paired_tsddr.jl): when DR_OUTPUT_TAG is
# set, every output filename gets an _<tag> suffix so re-evaluations at a
# different scenario count never overwrite existing result files. Note that a
# tagged run starts fresh tagged files; the merge-with-existing-columns logic
# only applies within the same tag.
tag_suffix = let tag = get(ENV, "DR_OUTPUT_TAG", "")
    isempty(tag) ? "" : "_$(tag)"
end
isempty(tag_suffix) || println("Output tag suffix: $tag_suffix")

const COL_NAME = "SDDP-SOC (paired)"
costs_file = joinpath(out_dir, "paired_costs$(tag_suffix).csv")
if isfile(costs_file)
    df = CSV.read(costs_file, DataFrame)
    df[!, COL_NAME] = objective_values
else
    df = DataFrame(Symbol(COL_NAME) => objective_values)
end
CSV.write(costs_file, df)
println("Updated: $costs_file")

vol_file = joinpath(out_dir, "paired_MeanVolume$(tag_suffix).csv")
if isfile(vol_file)
    df_vol = CSV.read(vol_file, DataFrame; header=true)
    df_vol[!, COL_NAME] = hydro_vol
else
    df_vol = DataFrame(Symbol(COL_NAME) => hydro_vol)
end
CSV.write(vol_file, df_vol)
println("Updated: $vol_file")

gen_file = joinpath(out_dir, "paired_MeanGeneration$(tag_suffix).csv")
if isfile(gen_file)
    df_gen = CSV.read(gen_file, DataFrame; header=true)
    df_gen[!, COL_NAME] = thermal_gen
else
    df_gen = DataFrame(Symbol(COL_NAME) => thermal_gen)
end
CSV.write(gen_file, df_gen)
println("Updated: $gen_file")

# Per-scenario paired costs for direct comparison
println("\nPer-scenario cost comparison (first 10):")
if isfile(costs_file)
    df_all = CSV.read(costs_file, DataFrame)
    if hasproperty(df_all, Symbol("TS-DDR (strict, paired)"))
        tsddr_costs = df_all[!, "TS-DDR (strict, paired)"]
        sddp_costs = df_all[!, COL_NAME]
        diffs = sddp_costs .- tsddr_costs
        println("  Scenario | SDDP      | TS-DDR    | Diff")
        for s in 1:min(10, num_scenarios)
            println("  $(lpad(s, 7)) | $(lpad(round(sddp_costs[s]; digits=1), 9)) | $(lpad(round(tsddr_costs[s]; digits=1), 9)) | $(round(diffs[s]; digits=1))")
        end
        println("\n  Mean diff (SDDP - TS-DDR): $(round(mean(diffs); digits=1))")
        println("  Std diff:                  $(round(std(diffs); digits=1))")
        println("  Paired t-test p-value:     (compute externally)")
    end
end
