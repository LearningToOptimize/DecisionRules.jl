# Smoke tests for the sharded paired-evaluation merge.
#
# `eval_paired_sddp.jl` writes one CSV per shard keyed by GLOBAL protocol column
# id, and `merge_sddp_shards.jl` reassembles them. The merge is the last place a
# scenario can silently disappear before a published mean, so these tests pin
# that it REFUSES to produce a statistic from an incomplete or overlapping set,
# and that an unsolved scenario stays visible.
#
# Everything runs on synthetic shards in a temporary directory: no case, no
# solver, no final-result constants. A two-scenario configuration merges exactly
# the same way the 500-scenario one does.
#
# Usage:
#   julia --project=examples/HydroPowerModels examples/HydroPowerModels/sddp/test_merge_sddp_shards.jl

using Test
using CSV, DataFrames, Statistics

const MERGE_SCRIPT = joinpath(dirname(dirname(@__FILE__)), "sddp", "merge_sddp_shards.jl")

"""
    write_shards(dir, ranges; costs, solved) -> Nothing

Write one `sddp_shard_<lo>_<hi>.csv` per range, in the schema
`eval_paired_sddp.jl` emits.

# Arguments
- `dir::AbstractString`: destination directory.
- `ranges`: iterable of `UnitRange` global scenario id ranges.

# Keywords
- `costs`: `id -> cost` mapping.
- `solved`: `id -> Bool`, or `nothing` to omit the `all_stages_solved` column
  (the schema written before solve status was recorded).
"""
function write_shards(dir, ranges; costs, solved = nothing)
    for range in ranges
        ids = collect(range)
        frame = DataFrame(scenario = ids, cost = [costs(i) for i in ids])
        if solved !== nothing
            frame[!, :all_stages_solved] = [solved(i) for i in ids]
            frame[!, :status_observed] = trues(length(ids))
        end
        CSV.write(joinpath(dir, "sddp_shard_$(first(range))_$(last(range)).csv"), frame)
    end
    return nothing
end

"""
    run_merge(dir; kwargs...) -> (ok::Bool, output::String)

Run `merge_sddp_shards.jl` in a fresh process with the given environment, and
report whether it exited zero along with everything it printed.

A subprocess is used deliberately: the script is a top-level program and the
thing under test includes whether it FAILS, which is only observable as an exit
status.
"""
function run_merge(dir; first_id = 1, last_id = 6, allow_partial = false, out = nothing)
    environment = copy(ENV)
    environment["DR_SHARD_DIR"] = dir
    environment["DR_SCENARIO_FIRST"] = string(first_id)
    environment["DR_SCENARIO_LAST"] = string(last_id)
    environment["DR_ALLOW_PARTIAL"] = string(allow_partial)
    out === nothing || (environment["DR_MERGE_OUT"] = out)
    buffer = IOBuffer()
    command = pipeline(
        Cmd(`$(Base.julia_cmd()) --project=$(Base.active_project()) $MERGE_SCRIPT`;
            env = environment);
        stdout = buffer, stderr = buffer,
    )
    # `success` rather than `run`: a non-zero exit is an EXPECTED outcome in
    # half these tests, and `run` would raise `ProcessFailedException` on it.
    ok = success(command)
    return ok, String(take!(buffer))
end

@testset "sharded paired-evaluation merge" begin
    costs(i) = 300_000.0 + 100.0 * i

    @testset "complete, disjoint shards merge and reproduce the mean" begin
        mktempdir() do dir
            write_shards(dir, [1:2, 3:4, 5:6]; costs = costs)
            merged = joinpath(dir, "merged.csv")
            ok, output = run_merge(dir; out = merged)
            @test ok
            @test occursin("COMPLETE", output)
            frame = CSV.read(merged, DataFrame)
            # Global ids survive sharding, ascending, no gaps, no duplicates.
            @test frame.scenario == collect(1:6)
            @test frame.cost ≈ [costs(i) for i in 1:6]
            @test occursin(string(round(mean(frame.cost); digits = 5)), output)
        end
    end

    @testset "a missing shard is an ERROR, not a warning" begin
        mktempdir() do dir
            write_shards(dir, [1:2, 5:6]; costs = costs)   # 3:4 never ran
            ok, output = run_merge(dir)
            @test !ok
            @test occursin("INCOMPLETE", output)
            # The mean over the four that DID run must not be presented as a
            # merge of the protocol.
            @test !isfile(joinpath(dir, "sddp_paired_merged.csv"))
        end
    end

    @testset "an incomplete set can be inspected, but is labelled" begin
        mktempdir() do dir
            write_shards(dir, [1:2, 5:6]; costs = costs)
            ok, output = run_merge(dir; allow_partial = true)
            @test ok
            @test occursin("PARTIAL", output)
            @test isfile(joinpath(dir, "sddp_paired_merged_PARTIAL.csv"))
            @test !isfile(joinpath(dir, "sddp_paired_merged.csv"))
        end
    end

    @testset "overlapping shards are rejected" begin
        mktempdir() do dir
            write_shards(dir, [1:4, 3:6]; costs = costs)   # 3 and 4 twice
            ok, output = run_merge(dir)
            @test !ok
            @test occursin("duplicate scenario ids", output)
        end
    end

    @testset "ids outside the expected range are rejected" begin
        mktempdir() do dir
            write_shards(dir, [1:6, 7:8]; costs = costs)
            ok, output = run_merge(dir; last_id = 6)
            @test !ok
            @test occursin("outside", output)
        end
    end

    @testset "unsolved scenarios stay visible and leave the statistics" begin
        mktempdir() do dir
            write_shards(dir, [1:3, 4:6]; costs = costs, solved = i -> i != 4)
            merged = joinpath(dir, "merged.csv")
            ok, output = run_merge(dir; out = merged)
            @test ok
            @test occursin("unsolved ids", output)
            @test occursin("PARTIAL", output)     # not a complete evaluation
            frame = CSV.read(merged, DataFrame)
            # The unsolved row is KEPT in the merged table — it is evidence —
            # but excluded from the reported mean.
            @test frame.scenario == collect(1:6)
            expected = mean(costs(i) for i in [1, 2, 3, 5, 6])
            @test occursin(string(round(expected; digits = 5)), output)
        end
    end

    @testset "a two-scenario smoke configuration merges identically" begin
        mktempdir() do dir
            write_shards(dir, [1:1, 2:2]; costs = costs)
            ok, output = run_merge(dir; last_id = 2)
            @test ok
            @test occursin("COMPLETE", output)
        end
    end

    @testset "an empty shard directory is an error" begin
        mktempdir() do dir
            ok, output = run_merge(dir)
            @test !ok
            @test occursin("no sddp_shard_", output)
        end
    end
end
