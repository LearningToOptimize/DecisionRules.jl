# Verification of the FROZEN case contract.
#
# The published TS-DDR-vs-SDDP comparison is a statement about one specific set
# of bytes. This test re-derives the contract from those bytes and requires the
# stored `bolivia/case_manifest.json` to agree with it, so a case edit cannot
# reach the documentation unnoticed.
#
# It also pins the individual invariants explicitly, rather than only through
# the manifest comparison, so a failure names WHAT changed rather than merely
# reporting that the manifest no longer matches.
#
# Usage:
#   julia --project=examples/HydroPowerModels examples/HydroPowerModels/test_case_manifest.jl

using Test
using JSON

const SCRIPT_DIR = dirname(dirname(@__FILE__))   # examples/HydroPowerModels
include(joinpath(SCRIPT_DIR, "generate_canonical_case_artifacts.jl"))
const CASE = HydroCanonicalCase
const CASE_DIR = joinpath(SCRIPT_DIR, "bolivia")

@testset "frozen case contract" begin

    @testset "inputs are the frozen bytes" begin
        counts = CASE.verify_inputs(CASE_DIR)
        @test CASE.sha256_file(joinpath(CASE_DIR, "hydro.json")) ==
              "b25ce1c7bafcfaf907091dcd1007949c79a79974c9a33020b2587400d756b29a"
        @test CASE.sha256_file(joinpath(CASE_DIR, "PowerModels.json")) ==
              "1ff598447957f9fc17ca570415bf5b9b5b14e1292ea3bd3163db0ad79911a782"
        @test CASE.sha256_file(joinpath(CASE_DIR, "inflows.csv")) ==
              "5afb275dff3fc879e3e93b6510b81295834faad0bcd2bd1fc070a8e3e6653c77"
        @test counts["hydro_units"] == 11
    end

    @testset "no demand uncertainty may reappear" begin
        # The frozen experiment has deterministic demand. A demand file would
        # silently change the uncertainty space, so its absence is asserted.
        for name in CASE.FORBIDDEN_CASE_FILES
            @test !isfile(joinpath(CASE_DIR, name))
        end
        @test CASE.DEMAND_IS_DETERMINISTIC
        @test CASE.ACTIVE_LOAD_FACTOR == 0.6
        @test CASE.REACTIVE_LOAD_FACTOR == 0.6
    end

    @testset "initial state is empty, not repaired to 70%" begin
        x0 = CASE.initial_state(CASE_DIR)
        @test length(x0) == 11
        # Denormal, i.e. zero relative to the smallest capacity in the case.
        @test maximum(x0) < CASE.EMPTY_VOLUME_TOL
        @test all(Float32.(x0) .== 0.0f0)
        # A 70%-of-capacity repair would put reservoir 2 near 96.6 pu.
        @test maximum(x0) < 1e-300
    end

    @testset "water balance and horizon" begin
        hydro = JSON.parsefile(joinpath(CASE_DIR, "hydro.json"))
        @test Int(hydro["stage_hours"]) == 168
        @test 0.0036 * 168 == CASE.HYDRO_CONVERSION_K == 0.6048
        @test CASE.REPORTING_STAGES == 96
        @test CASE.LOOKAHEAD_STAGES == 30
        @test CASE.TOTAL_STAGES == 126
    end

    @testset "costs and reactive balance" begin
        power = JSON.parsefile(joinpath(CASE_DIR, "PowerModels.json"))
        @test Float64(power["baseMVA"]) == 100.0
        @test Float64(power["cost_deficit"]) * 100.0 == CASE.ACTIVE_DEFICIT_COST == 6000.0
        @test CASE.REACTIVE_BALANCE == "hard"
    end

    @testset "protocol is inflow-only and reproducible by construction" begin
        @test CASE.INFLOW_PROTOCOL_SEED == 20260706
        @test CASE.PROTOCOL_STAGES == 126
        @test CASE.PROTOCOL_SCENARIOS == 500
        ncen = CASE.inflow_scenario_count(CASE_DIR)
        indices = CASE.protocol_indices(ncen)
        @test size(indices) == (126, 500)
        @test all(1 .<= indices .<= ncen)
        # Regenerating gives the identical matrix (the seed and the SHAPE both
        # matter: the same stream consumed into a different shape differs).
        @test CASE.protocol_indices(ncen) == indices
        digest = CASE.protocol_digest(CASE_DIR)
        @test digest.ncen == ncen
        @test CASE.protocol_digest(CASE_DIR).sha256 == digest.sha256
    end

    @testset "method configuration" begin
        @test CASE.TSDDR_FORMULATION == "ACPPowerModel"          # true ACP
        @test CASE.TSDDR_TARGET_MODE == "strict"
        @test CASE.TSDDR_TARGET_ACTIVATION == "stretchedsigmoid"
        @test CASE.SDDP_BACKWARD_FORMULATION == "SOCWRConicPowerModel"
        @test CASE.SDDP_FORWARD_FORMULATION == "ACPPowerModel"
    end

    @testset "generated stage models match the case" begin
        for formulation in CASE.FORMULATIONS
            summary = CASE.verify_generated_model(
                joinpath(CASE_DIR, formulation * ".mof.json"), formulation,
            )
            @test summary["objective_sense"] == "min"
            @test summary["active_deficit_terms"] == 28
            @test summary["hydro_balances"] == 11
            # The weekly water balance, in the model the JuMP workflow actually
            # loads. An export taken with stage_hours left at 1 would carry
            # 0.0036 here and model a week as an hour.
            @test summary["hydro_balance_inflow_coefficient"] == CASE.HYDRO_CONVERSION_K
            @test summary["matches_case_K"] == true
        end
    end

    @testset "stored manifest describes the committed bytes" begin
        manifest = CASE.verify(CASE_DIR)      # throws, naming every difference
        @test manifest["schema_version"] == 2
        stored = JSON.parsefile(CASE.manifest_path(CASE_DIR))
        @test stored["water_balance"]["K"] == 0.6048
        @test stored["protocol"]["seed"] == 20260706
        @test stored["initial_state"]["float32_is_exactly_zero"] == true
        @test stored["demand"]["deterministic"] == true
    end
end
