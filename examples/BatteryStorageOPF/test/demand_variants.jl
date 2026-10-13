# Focused tests of the regional demand variants (`demand_variants.toml`,
# section J of battery_portfolio.jl).
#
# Every identity the variant recipe claims is checked twice: on the MODES the
# construction produces, and on the FROZEN SUPPORT those modes become, read
# through `demand_multipliers` exactly as a stage problem reads it. A synthetic
# network with regional, out-of-region, negative and capacitive loads keeps the
# file standalone; when the frozen case793 directory sits beside the example the
# reference-identity check runs on it too.
#
#   julia --project=. test/demand_variants.jl

using Test
using SHA
using JSON
using Statistics

const DV_EXAMPLE = dirname(@__DIR__)
@isdefined(DEMAND_VARIANT_SCHEMA) || include(joinpath(DV_EXAMPLE, "battery_portfolio.jl"))

"A small network carrying only what the demand layer reads."
function dv_synthetic_network()
    loads = Dict{String,Any}()
    buses = Dict{String,Any}()
    id = 0
    # Two buses per region; regional demand descends R1 > … > R6.
    region_pd = [(2.0, 1.6), (1.5, 1.1), (1.2, 1.1), (0.9, 0.7), (0.8, 0.7), (0.9, 0.5)]
    for r in 1:6, (i, pd) in enumerate(region_pd[r])
        bus = 2r - 2 + i
        id += 1
        loads[string(id)] = Dict{String,Any}("index" => id, "load_bus" => bus, "pd" => pd,
                                             "qd" => isodd(id) ? 0.3 * pd : -0.1 * pd, "status" => 1)
    end
    # Out of every region: a negative load and a zero load.
    for (bus, pd, qd) in ((13, -0.3, 0.05), (14, 0.0, 0.0))
        id += 1
        loads[string(id)] = Dict{String,Any}("index" => id, "load_bus" => bus, "pd" => pd,
                                             "qd" => qd, "status" => 1)
    end
    for b in 1:14
        buses[string(b)] = Dict{String,Any}("index" => b, "bus_type" => b == 1 ? 3 : 1)
    end
    return Dict{String,Any}("load" => loads, "bus" => buses, "baseMVA" => 100.0), [[2r - 1, 2r] for r in 1:6]
end

@testset "regional demand variants" begin
    recipe = demand_variant_recipe()
    @test recipe["reference_manifest_digest"] == read_portfolio_manifest()["digest"]
    δ2 = (1 / 6) * 0.15^2 + (5 / 6) * 0.03^2
    amps = Float64.(recipe["means"]["amplitude_candidates"])
    @test amps == [0.15, 0.10, 0.05]
    @test_throws ErrorException variant_signs(recipe, "C", 1)

    @testset "construction identities on the modes (case793 region demand)" begin
        rec = only(filter(c -> c["case"] == "pglib_opf_case793_goc", read_portfolio_manifest()["cases"]))
        D = Float64.(rec["regions"]["demand_pu"])
        w = D ./ sum(D)
        par = variant_parameters(recipe, D)
        @test par.delta^2 ≈ δ2 atol = 1e-15
        @test abs(sum(par.G .* cis.(par.phi))) < 1e-12 * sum(par.G)
        @test sum(w .* par.u .^ 2) ≈ δ2 rtol = 1e-12
        for (a, b) in par.pairs
            @test D[a] * par.u[a] ≈ D[b] * par.u[b] rtol = 1e-14
        end
        for amp in amps
            a = variant_mean_modulation(recipe, par, amp)
            @test maximum(abs, [sum(PORTFOLIO_PROFILE .* a[p, :]) for p in 1:3]) < 1e-13
            @test maximum(abs, [sum(par.G .* a[:, t]) for t in 1:PORTFOLIO_HORIZON]) < 1e-12 * sum(D)
            for t in 1:PORTFOLIO_HORIZON
                MA = variant_modes(recipe, "A", D, t; amplitude = amp)
                MB = variant_modes(recipe, "B", D, t; amplitude = amp)
                for M in (MA, MB)
                    @test all(>(0), M)
                    @test maximum(abs, D' * M .- sum(D)) < 1e-12 * sum(D)       # total fixed per atom
                    @test vec(mean(M; dims = 2)) ≈ [1 + a[par.group[r], t] for r in 1:6] rtol = 1e-14
                    rel = (M ./ (1 .+ a[par.group, t])) .- 1
                    @test sum(w .* vec(mean(rel .^ 2; dims = 2))) ≈ δ2 rtol = 1e-12
                    @test allunique(eachcol(M))
                end
                # Same marginal per region, same atoms except the overwritten group.
                @test all(sort(MA[r, :]) == sort(MB[r, :]) for r in 1:6)
                blk = only(filter(b -> b["first"] <= t <= b["last"], recipe["variants"]["B"]["blocks"]))
                q = Int(blk["overwrite"])
                differs = [MA[:, k] != MB[:, k] for k in 1:4]
                @test count(differs) == 2
                for k in findall(differs), r in 1:6
                    par.group[r] == q || @test MA[r, k] == MB[r, k]
                end
            end
        end
        # Sign-pattern moments: A pairwise independent with s1 s2 s3 ≡ 1; B locked.
        SA = variant_signs(recipe, "A", 1)
        @test SA' * SA == 4I
        @test all(==(1), prod(SA; dims = 2))
        for blk in recipe["variants"]["B"]["blocks"]
            SB = variant_signs(recipe, "B", Int(blk["first"]))
            p, q = Int(blk["keep"]), Int(blk["overwrite"])
            C = SB' * SB
            @test C[p, q] == 4 && C[q, p] == 4
            @test count(!iszero, C) == 5
            @test sum(prod(SB; dims = 2)) == 0
            @test all(sum(SB; dims = 1) .== 0)
        end
    end

    @testset "identities on the frozen support" begin
        net, regions = dv_synthetic_network()
        meta = demand_meta(net)
        D = region_demand_pu(net, regions)
        @test issorted(D; rev = true)
        κ = 1.05
        region_of = Dict(b => r for (r, g) in enumerate(regions) for b in g)
        for variant in ("A", "B"), amp in (0.15, 0.05)
            s = portfolio_variant_support(net, regions, D, κ, recipe, variant; amplitude = amp, seed = 7)
            validate_support(s)
            par = variant_parameters(recipe, D)
            a = variant_mean_modulation(recipe, par, amp)
            @test s.source["sampler"]["sampler"] == "stage"
            for t in 1:s.horizon
                @test atom_probabilities(s, t) == fill(0.25, 4)
                h = κ * PORTFOLIO_PROFILE[t]
                M = variant_modes(recipe, variant, D, t; amplitude = amp)
                totals = Float64[]
                for k in 1:4
                    mult = demand_multipliers(s, t, k)
                    for (j, bus) in enumerate(meta.load_bus)
                        haskey(region_of, bus) || @test mult[j] == h      # out of region: profile only
                    end
                    P = [sum(meta.nominal_pd[j] * mult[j] for j in eachindex(mult)
                             if get(region_of, meta.load_bus[j], 0) == r) for r in 1:6]
                    @test P ≈ h .* D .* M[:, k] rtol = 1e-13
                    push!(totals, sum(meta.nominal_pd .* mult))
                end
                @test maximum(totals) - minimum(totals) < 1e-12 * abs(mean(totals))
                # Power factor of every load preserved.
                for k in 1:4, j in eachindex(meta.load_ids)
                    meta.nominal_pd[j] == 0 && continue
                    mult = demand_multipliers(s, t, k)
                    @test (meta.nominal_qd[j] * mult[j]) / (meta.nominal_pd[j] * mult[j]) ≈
                          meta.nominal_qd[j] / meta.nominal_pd[j] rtol = 1e-12
                end
            end
            # Deterministic: a second freeze has the same digest.
            s2 = portfolio_variant_support(net, regions, D, κ, recipe, variant; amplitude = amp, seed = 7)
            @test support_digest(s2) == support_digest(s)
        end
        sA = portfolio_variant_support(net, regions, D, κ, recipe, "A"; amplitude = 0.1, seed = 7)
        sB = portfolio_variant_support(net, regions, D, κ, recipe, "B"; amplitude = 0.1, seed = 7)
        @test support_digest(sA) != support_digest(sB)
        # Shared evaluation index matrices (the A/B pairing).
        @test scenario_index_matrix(sA, 24, 16) == scenario_index_matrix(sB, 24, 16)
    end

    ref_dir = joinpath(DV_EXAMPLE, "case", "pglib_opf_case793_goc")
    if isdir(ref_dir)
        @testset "reference support unchanged (frozen case793)" begin
            m = read_portfolio_manifest()
            rec = only(filter(c -> c["case"] == "pglib_opf_case793_goc", m["cases"]))
            case = read_battery_case(ref_dir)
            regions = [Int.(g) for g in case.demand.source["sampler"]["groups"]]
            sha = network_digest(case.network)
            @test sha == rec["network_sha256"]
            @test region_demand_pu(case.network, regions) ≈ Float64.(rec["regions"]["demand_pu"]) rtol = 1e-14
            s = portfolio_support(case.network, regions, Float64(rec["kappa_case"]);
                                  seed = portfolio_seed_for("protocol/final", sha))
            @test support_digest(s) == rec["support"]["sha256"]
        end
    end
end
