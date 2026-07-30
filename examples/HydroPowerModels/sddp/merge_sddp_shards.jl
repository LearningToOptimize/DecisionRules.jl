# Merge sharded paired-evaluation outputs into one per-scenario cost table.
#
# eval_paired_sddp.jl has always referenced this script ("merged afterwards by
# merge_sddp_shards.jl") but it was never written, so sharded runs produced
# sddp_shard_<a>_<b>.csv files with nothing to combine them. This closes that gap.
#
# The shards partition the GLOBAL protocol columns, and demand atoms are keyed by global
# column index, so concatenating them reproduces the full run exactly. That is only true
# if the partition is complete and disjoint, which is checked here rather than assumed --
# a silently missing shard would bias the mean without changing anything visible.
#
# env: DR_SHARD_DIR (directory holding sddp_shard_*.csv), DR_NUM_SCENARIOS (expected
#      total, default 500), DR_MERGE_OUT (output csv path)

using CSV, DataFrames, Statistics, Printf

const DIR = get(ENV, "DR_SHARD_DIR", ".")
const N = parse(Int, get(ENV, "DR_NUM_SCENARIOS", "500"))
const OUT = get(ENV, "DR_MERGE_OUT", joinpath(DIR, "sddp_paired_merged.csv"))

files = sort(filter(f -> occursin(r"^sddp_shard_\d+_\d+\.csv$", f), readdir(DIR)))
isempty(files) && error("no sddp_shard_*.csv in $DIR")

df = reduce(vcat, [CSV.read(joinpath(DIR, f), DataFrame) for f in files])
sort!(df, :scenario)

# Completeness / disjointness: the whole point of sharding is that the union is the full
# protocol. Report rather than silently average a partial set.
missing_cols = setdiff(1:N, df.scenario)
dupes = [s for s in unique(df.scenario) if count(==(s), df.scenario) > 1]
isempty(dupes) || error("duplicate scenarios across shards: $(first(dupes, 10))")
if !isempty(missing_cols)
    @warn "INCOMPLETE: $(length(missing_cols)) of $N scenarios missing" first_missing = first(missing_cols, 10)
end

CSV.write(OUT, df)

c = df.cost
n = length(c)
m = mean(c)
sd = std(c)
se = sd / sqrt(n)
@printf("shards merged   : %d files, %d scenarios (expected %d)\n", length(files), n, N)
@printf("mean cost       : %.1f\n", m)
@printf("std dev         : %.1f\n", sd)
@printf("std error       : %.1f\n", se)
@printf("95%% CI          : [%.1f, %.1f]\n", m - 1.96se, m + 1.96se)
@printf("min / max       : %.1f / %.1f\n", minimum(c), maximum(c))
@printf("quartiles       : %.1f / %.1f / %.1f\n",
        quantile(c, 0.25), median(c), quantile(c, 0.75))
println("merged -> $OUT")
