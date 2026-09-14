# Merge sharded paired-evaluation outputs into one per-scenario cost table.
#
# `eval_paired_sddp.jl` writes one `sddp_shard_<lo>_<hi>.csv` per shard, keyed by
# the GLOBAL protocol column id. Concatenating those shards reproduces a full run
# EXACTLY — but only if the partition is complete and disjoint. A silently
# missing shard would change the mean without changing anything visible, so the
# partition is CHECKED here, and a violation is an error, not a warning.
#
# Nothing about the final published result is baked in: the expected scenario set
# is a parameter, and a two-scenario smoke run merges the same way a 500-scenario
# run does.
#
# Environment:
#   DR_SHARD_DIR      directory holding sddp_shard_*.csv        (default ".")
#   DR_SHARD_GLOB     filename pattern prefix                    (default "sddp_shard_")
#   DR_SCENARIO_FIRST first expected global scenario id          (default 1)
#   DR_SCENARIO_LAST  last expected global scenario id           (default 500)
#   DR_MERGE_OUT      output csv path                            (default <dir>/sddp_paired_merged.csv)
#   DR_ALLOW_PARTIAL  "true" to merge an INCOMPLETE set anyway   (default "false")
#
# `DR_ALLOW_PARTIAL` exists for inspecting a run still in flight. It renames the
# output to `*_PARTIAL.csv` and labels every printed statistic, because a mean
# over a subset of the protocol is NOT an estimate of the same quantity as a mean
# over all of it.

using CSV, DataFrames, Statistics, Printf

const DIR = get(ENV, "DR_SHARD_DIR", ".")
const PREFIX = get(ENV, "DR_SHARD_GLOB", "sddp_shard_")
const FIRST = parse(Int, get(ENV, "DR_SCENARIO_FIRST", "1"))
const LAST = parse(Int, get(ENV, "DR_SCENARIO_LAST", "500"))
const ALLOW_PARTIAL = lowercase(strip(get(ENV, "DR_ALLOW_PARTIAL", "false"))) in ("1", "true", "yes")

FIRST <= LAST || error("DR_SCENARIO_FIRST ($FIRST) must not exceed DR_SCENARIO_LAST ($LAST)")
const EXPECTED = FIRST:LAST

pattern = Regex("^" * PREFIX * raw"\d+_\d+\.csv$")
files = sort(filter(f -> occursin(pattern, f), readdir(DIR)))
isempty(files) && error("no $(PREFIX)*.csv shard files in $DIR")

# Read every shard and remember which file each row came from, so a duplicate or
# an out-of-range id can be attributed to a specific shard rather than merely
# reported to exist.
frames = DataFrame[]
for file in files
    frame = CSV.read(joinpath(DIR, file), DataFrame)
    hasproperty(frame, :scenario) ||
        error("$file has no `scenario` column; shards must carry GLOBAL scenario ids")
    hasproperty(frame, :cost) || error("$file has no `cost` column")
    frame[!, :shard_file] .= file
    push!(frames, frame)
end
df = sort!(reduce(vcat, frames; cols = :union), :scenario)

# ── Partition checks: complete, disjoint, in range ────────────────────────────
duplicates = [id for id in unique(df.scenario) if count(==(id), df.scenario) > 1]
if !isempty(duplicates)
    offenders = unique(df[in(duplicates).(df.scenario), :shard_file])
    error("duplicate scenario ids across shards: $(first(duplicates, 10)) " *
          "(shards $(offenders)). Overlapping shard ranges would double-count.")
end

out_of_range = setdiff(df.scenario, EXPECTED)
isempty(out_of_range) ||
    error("shards contain scenario ids outside $(FIRST):$(LAST): $(first(sort(out_of_range), 10))")

missing_ids = setdiff(EXPECTED, df.scenario)
if !isempty(missing_ids)
    message = "INCOMPLETE merge: $(length(missing_ids)) of $(length(EXPECTED)) " *
              "scenarios are missing (first: $(first(sort(missing_ids), 10))). " *
              "A mean over a subset is not a merge of the protocol."
    ALLOW_PARTIAL || error(message * " Set DR_ALLOW_PARTIAL=true to inspect it anyway.")
    @warn message
end

# ── Unsolved scenarios stay visible ───────────────────────────────────────────
# Shards written with the physical audit carry `all_stages_solved`. A scenario
# whose rollout did not solve every stage has a cost that is not comparable, so
# it is reported and excluded from the statistics rather than averaged in.
unsolved = if hasproperty(df, :all_stages_solved)
    df.scenario[.!coalesce.(df.all_stages_solved, false)]
else
    Int[]
end
usable = isempty(unsolved) ? df : df[.!in(unsolved).(df.scenario), :]

complete = isempty(missing_ids) && isempty(unsolved)
default_out = joinpath(DIR, complete ? "sddp_paired_merged.csv" : "sddp_paired_merged_PARTIAL.csv")
out = get(ENV, "DR_MERGE_OUT", default_out)
CSV.write(out, df)

costs = usable.cost
n = length(costs)
n > 0 || error("no usable scenarios after excluding unsolved ones")
m = mean(costs)
sd = std(costs)
se = sd / sqrt(n)
label = complete ? "COMPLETE" : "PARTIAL — NOT the protocol mean"
@printf("shards merged   : %d files, %d rows, ids %d:%d\n", length(files), nrow(df), FIRST, LAST)
@printf("coverage        : %d of %d expected  [%s]\n", nrow(df), length(EXPECTED), label)
isempty(missing_ids) || @printf("missing ids     : %s\n", string(first(sort(missing_ids), 20)))
isempty(unsolved) || @printf("unsolved ids    : %s  (excluded from statistics)\n", string(sort(unsolved)))
@printf("mean cost       : %.5f\n", m)
@printf("std dev         : %.5f\n", sd)
@printf("std error       : %.5f\n", se)
@printf("95%% CI          : [%.5f, %.5f]\n", m - 1.96se, m + 1.96se)
@printf("min / max       : %.5f / %.5f\n", minimum(costs), maximum(costs))
@printf("quartiles       : %.5f / %.5f / %.5f\n",
        quantile(costs, 0.25), median(costs), quantile(costs, 0.75))
println("merged -> $out")
