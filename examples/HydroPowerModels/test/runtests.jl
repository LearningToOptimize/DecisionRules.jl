#!/usr/bin/env julia

# Regression suite for the Bolivia hydro example (JuMP / MathOptFormat engine).
#
# Every test here is a maintained regression test, not a leftover diagnostic.
# They are grouped by what a change would break:
#
#   case       the frozen case contract — input hashes, water balance, horizon,
#              cost convention, protocol, and the generated stage models
#   gradient   the reachable-policy map and its derivative, against an
#              engine-independent oracle
#   sampling   that both engines reconstruct the same inflow scenarios from the
#              same bytes
#   strict     that strict-mode stage problems build, solve and price as
#              intended, and that a reachable target is always feasible
#   merge      that the sharded evaluation merge refuses an incomplete set
#
# Usage:
#   julia --project=examples/HydroPowerModels examples/HydroPowerModels/test/runtests.jl
#   julia --project=examples/HydroPowerModels examples/HydroPowerModels/test/runtests.jl case gradient
#
# With no arguments every group runs. Naming groups runs only those, which is
# what a quick iteration wants: `case` and `merge` take seconds, `gradient` and
# `strict` take minutes because they solve.
#
# `strict` needs Ipopt and DiffOpt; `sampling` needs a DecisionRulesExa.jl
# checkout beside this one (or `DR_EXA_HPM_DIR` pointing at its
# `examples/HydroPowerModels`). A group whose prerequisites are missing is
# SKIPPED LOUDLY rather than silently passing.

using Test

const TEST_DIR = dirname(@__FILE__)
const HYDRO_DIR = dirname(TEST_DIR)

"""
    GROUPS

Test group name to the file that implements it, in the order they run: cheapest
and most fundamental first, so a broken case contract is reported before minutes
are spent solving against it.
"""
const GROUPS = [
    "case" => "test_case_manifest.jl",
    "merge" => "test_merge_sddp_shards.jl",
    "sampling" => "test_sampling_consistency.jl",
    "gradient" => "test_reachable_policy_gradient.jl",
    "strict" => "test_strict_mode.jl",
]

"""
    exa_hydro_dir() -> Union{Nothing,String}

Path to the ExaModels engine's `examples/HydroPowerModels`, from
`DR_EXA_HPM_DIR` or the side-by-side checkout layout, or `nothing` when neither
exists. Never an absolute machine path.
"""
function exa_hydro_dir()
    explicit = get(ENV, "DR_EXA_HPM_DIR", "")
    isempty(explicit) || return isdir(explicit) ? explicit : nothing
    guess = normpath(joinpath(HYDRO_DIR, "..", "..", "..", "DecisionRulesExa.jl",
                              "examples", "HydroPowerModels"))
    return isdir(guess) ? guess : nothing
end

"""
    can_run(group) -> Union{Nothing,String}

`nothing` when `group` can run here, otherwise the reason it cannot.
"""
function can_run(group)
    if group == "sampling" && exa_hydro_dir() === nothing
        return "no DecisionRulesExa.jl checkout found (set DR_EXA_HPM_DIR)"
    end
    return nothing
end

requested = isempty(ARGS) ? first.(GROUPS) : ARGS
for name in requested
    any(g -> first(g) == name, GROUPS) ||
        error("unknown test group $name; known groups: $(join(first.(GROUPS), ", "))")
end

skipped = String[]
@testset "Bolivia hydro example" begin
    for (name, file) in GROUPS
        name in requested || continue
        reason = can_run(name)
        if reason !== nothing
            @warn "SKIPPING test group" group = name reason
            push!(skipped, "$name ($reason)")
            continue
        end
        @testset "$name" begin
            # `sampling` takes the Exa directory as ARGS[1]; the others ignore
            # ARGS. Setting it for all of them keeps the include uniform.
            empty!(ARGS)
            name == "sampling" && push!(ARGS, exa_hydro_dir())
            include(joinpath(TEST_DIR, file))
        end
    end
end

isempty(skipped) || @warn "test groups skipped" skipped
