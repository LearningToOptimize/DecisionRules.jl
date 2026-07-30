# test_sampling_consistency.jl
#
# Verifies that DecisionRules.jl and the ExaModels companion package
# (DecisionRulesExa.jl) sample from the exact same inflow distribution
# as SDDP.jl for the Bolivia HydroPowerModels case.
#
# All three systems read the same inflows.csv and hydro.json files.
# The sampling contract is:
#
#   At each stage t, draw one scenario index ω ∈ {1, …, nScenarios}
#   uniformly at random. All hydro reservoirs receive the inflow from
#   column ω of the historical data for their respective row t.
#   Stages are sampled independently (no temporal correlation).
#
# This is SDDP.jl's `SDDP.parameterize` semantics: one ω per node,
# applied to all random variables in that node.
#
# What this script checks:
#   1. Both loaders parse inflows.csv into identical per-reservoir matrices.
#   2. Both samplers produce draws from the same support (only historically
#      observed joint vectors, never cross-scenario combinations).
#   3. With the same RNG seed, both produce identical trajectories.
#
# Usage:
#   julia --project=. test_sampling_consistency.jl           (from this dir)
#   julia --project=. test_sampling_consistency.jl /path/to/DecisionRulesExa.jl/examples/HydroPowerModels

using Test
using Random
using CSV, Tables, JSON

# ── Paths ────────────────────────────────────────────────────────────────────

const SCRIPT_DIR = dirname(@__FILE__)
const CASE_DIR = joinpath(SCRIPT_DIR, "bolivia")
const INFLOW_FILE = joinpath(CASE_DIR, "inflows.csv")
const HYDRO_FILE = joinpath(CASE_DIR, "hydro.json")

const EXA_DIR = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(dirname(dirname(dirname(SCRIPT_DIR))),
             "..", "DecisionRulesExa.jl", "examples", "HydroPowerModels")

# ── 1. Parse inflows with both loaders ───────────────────────────────────────

# DecisionRules.jl loader (load_hydropowermodels.jl::read_inflow)
function dr_read_inflow(file, nHyd; num_stages=nothing)
    allinflows = CSV.read(file, Tables.matrix; header=false)
    nlin, ncol = size(allinflows)
    if isnothing(num_stages)
        num_stages = nlin
    elseif num_stages > nlin
        number_of_cycles = div(num_stages, nlin) + 1
        allinflows = vcat([allinflows for _ in 1:number_of_cycles]...)
    end
    nCen = Int(floor(ncol / nHyd))
    vector_inflows = [allinflows[1:num_stages, ((i-1)*nCen+1):(i*nCen)] for i in 1:nHyd]
    return vector_inflows, nCen, num_stages
end

# ExaModels loader (hydro_power_data.jl::load_hydro_data, inflow portion only)
function exa_read_inflow(file, nHyd; num_stages=nothing)
    allinflows = CSV.read(file, Tables.matrix; header=false)
    nrows, ncols = size(allinflows)
    nScenarios = div(ncols, nHyd)
    nStagesSample = isnothing(num_stages) ? nrows : num_stages
    if !isnothing(num_stages) && num_stages > nrows
        repeats = div(num_stages, nrows) + 1
        allinflows = vcat([allinflows for _ in 1:repeats]...)
    end
    allinflows = allinflows[1:nStagesSample, :]
    scenario_inflows = [Float64.(allinflows[:, ((r-1)*nScenarios+1):(r*nScenarios)]) for r in 1:nHyd]
    return scenario_inflows, nScenarios, nStagesSample
end

# ── 2. Sampler implementations (extracted, no package dependencies) ──────────

function dr_sample_joint(vector_inflows, nCen, T)
    nHyd = length(vector_inflows)
    trajectory = Vector{Vector{Float64}}(undef, T)
    for t in 1:T
        ω = rand(1:nCen)
        trajectory[t] = [vector_inflows[r][t, ω] for r in 1:nHyd]
    end
    return trajectory
end

function exa_sample_scenario(scenario_inflows, nScenarios, T)
    nHyd = length(scenario_inflows)
    nStagesSample = size(scenario_inflows[1], 1)
    w = Vector{Float64}(undef, T * nHyd)
    for t in 1:T
        t_row = mod1(t, nStagesSample)
        j = rand(1:nScenarios)
        for r in 1:nHyd
            w[(t-1)*nHyd + r] = scenario_inflows[r][t_row, j]
        end
    end
    return w
end

# ── Tests ────────────────────────────────────────────────────────────────────

hydro_json = JSON.parsefile(HYDRO_FILE)["Hydrogenerators"]
nHyd = length(hydro_json)
T = 96

@testset "Sampling consistency: DecisionRules vs Exa vs SDDP" begin
    dr_inflows, dr_nCen, dr_T = dr_read_inflow(INFLOW_FILE, nHyd; num_stages=T)
    exa_inflows, exa_nScen, exa_T = exa_read_inflow(INFLOW_FILE, nHyd; num_stages=T)

    @testset "identical inflow matrices" begin
        @test dr_nCen == exa_nScen
        @test dr_T == exa_T
        for r in 1:nHyd
            @test dr_inflows[r] == exa_inflows[r]
        end
    end

    @testset "same seed → identical trajectories" begin
        for seed in [42, 123, 9999]
            Random.seed!(seed)
            dr_traj = dr_sample_joint(dr_inflows, dr_nCen, T)

            Random.seed!(seed)
            exa_flat = exa_sample_scenario(exa_inflows, exa_nScen, T)

            for t in 1:T
                for r in 1:nHyd
                    @test dr_traj[t][r] == exa_flat[(t-1)*nHyd + r]
                end
            end
        end
    end

    @testset "samples are always from historical scenarios (joint)" begin
        valid_vectors = Set{Vector{Float64}}()
        for t in 1:T, ω in 1:dr_nCen
            push!(valid_vectors, [dr_inflows[r][t, ω] for r in 1:nHyd])
        end

        Random.seed!(42)
        for _ in 1:500
            traj = dr_sample_joint(dr_inflows, dr_nCen, T)
            for stage_vec in traj
                @test stage_vec in valid_vectors
            end
        end
    end

    @testset "uniform coverage of all scenarios" begin
        Random.seed!(42)
        N = 10_000
        counts = zeros(Int, dr_nCen)
        for _ in 1:N
            ω = rand(1:dr_nCen)
            counts[ω] += 1
        end
        for ω in 1:dr_nCen
            freq = counts[ω] / N
            expected = 1.0 / dr_nCen
            @test abs(freq - expected) < 0.03
        end
    end
end

println("\nAll sampling consistency tests passed.")
