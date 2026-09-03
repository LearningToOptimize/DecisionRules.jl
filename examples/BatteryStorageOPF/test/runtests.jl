# Consolidated regression suite for the JuMP/PowerModels/SDDP battery engine.
#
# One file, grouped by the property being protected. Each group guards something
# that can silently corrupt the science rather than fail loudly: a case artifact
# that stopped being reproducible, a sampler that stopped being reproducible from
# its seed, a frozen support the two engines would read differently, a problem
# specification that stopped being PowerModels', a recourse variable that
# acquired a bound, a strict target that stopped being strict, a nodal price with
# the wrong sign, or an SDDP graph whose two passes stopped being the two
# formulations the study claims they are.
#
#   julia --project=. test/runtests.jl

using Test
using JuMP
using PowerModels
using SDDP
using Ipopt
using Clarabel
using Distributions
using Statistics
using StableRNGs
using SHA
using JSON

const EXAMPLE = dirname(@__DIR__)
include(joinpath(EXAMPLE, "battery_sddp.jl"))         # → battery_powermodels.jl → case, schema
include(joinpath(EXAMPLE, "battery_diagnostics.jl"))
include(joinpath(EXAMPLE, "battery_demand.jl"))
include(joinpath(EXAMPLE, "battery_analysis.jl"))
include(joinpath(EXAMPLE, "build_battery_case.jl"))
include(joinpath(EXAMPLE, "battery_portfolio.jl"))

const CASE_DIR = joinpath(EXAMPLE, "case", "pglib_opf_case14_ieee")
const OPT = acp_optimizer()

@testset "battery storage OPF (JuMP engine)" begin

    # The case is CONSTRUCTED, not committed: the suite builds it from the
    # public builder, which also makes the build itself part of what is tested.
    case = ensure_case(CASE_DIR)

    # ── The frozen case contract ─────────────────────────────────────────────
    @testset "case contract" begin
        # Canonical JSON must be a pure function of the value, or hashing an
        # artifact proves nothing.
        a = Dict("b" => 1, "a" => [1.0, 2.5], "c" => Dict("z" => true, "y" => nothing))
        b = Dict("c" => Dict("y" => nothing, "z" => true), "a" => [1.0, 2.5], "b" => 1)
        @test canonical_json(a) == canonical_json(b)
        @test JSON.parse(canonical_json(a))["a"] == [1.0, 2.5]
        @test_throws ErrorException canonical_json(Dict("x" => NaN))

        # Every artifact hash, the support digest and the protocol digest
        # re-verify on disk.
        @test read_battery_case(CASE_DIR; verify = true) isa BatteryCase
        for (file, want) in case.manifest["artifacts"]
            @test sha256_file(joinpath(CASE_DIR, file)) == want
        end
        @test support_digest(case.demand) == case.manifest["support"]["sha256"]
        @test protocol_digest(case.demand,
                              Int(case.manifest["protocol"]["num_stages"]),
                              Int(case.manifest["protocol"]["num_scenarios"])) ==
              case.manifest["protocol"]["sha256"]

        # Stage duration is a contract, not a default.
        @test stage_hours(case) > 0
        @test stage_hours(case) == Float64(case.manifest["stage_hours"])

        # The support covers exactly the network's loads, in sorted order.
        @test case.demand.load_ids == sort!([Int(l["index"]) for (_, l) in case.network["load"]])
        @test issorted(case.demand.load_ids) && allunique(case.demand.load_ids)
        validate_support(case.demand)

        # Stage indices outside the frozen horizon FAIL rather than wrap: a
        # cyclic reinterpretation of the horizon is how a study silently changes
        # which stage is the peak.
        @test_throws ArgumentError demand_multipliers(case.demand, horizon(case.demand) + 1, 1)
        @test_throws ArgumentError demand_multipliers(case.demand, 1, 0)

        # The demand process preserves each LOAD's power factor exactly.
        by_id = Dict(Int(l["index"]) => l for (_, l) in case.network["load"])
        for t in (1, 7, 19), k in 1:num_atoms(case.demand, t)
            m = demand_multipliers(case.demand, t, k)
            pd, qd = realized_bus_demand(case, t, k)
            expect_p = Dict{Int,Float64}(i => 0.0 for i in keys(pd))
            expect_q = Dict{Int,Float64}(i => 0.0 for i in keys(qd))
            for (j, id) in enumerate(case.demand.load_ids)
                l = by_id[id]
                expect_p[Int(l["load_bus"])] += Float64(l["pd"]) * m[j]
                expect_q[Int(l["load_bus"])] += Float64(l["qd"]) * m[j]
            end
            for i in keys(pd)
                @test pd[i] ≈ expect_p[i] atol = 1e-14
                @test qd[i] ≈ expect_q[i] atol = 1e-14
            end
        end
        for t in 1:horizon(case.demand)
            @test sum(atom_probabilities(case.demand, t)) ≈ 1
        end

        # The protocol is reproducible by construction, not stored, and every
        # index lies inside its own stage's support.
        m1 = scenario_index_matrix(case.demand, 12, 5)
        m2 = scenario_index_matrix(case.demand, 12, 5)
        @test m1 == m2
        for t in 1:12
            @test all(1 .<= m1[t, :] .<= num_atoms(case.demand, t))
        end
        @test_throws ArgumentError scenario_index_matrix(case.demand,
                                                         horizon(case.demand) + 1, 2)
    end

    # ── The authoring samplers ───────────────────────────────────────────────
    @testset "demand samplers" begin
        meta = demand_meta(case.network)
        @test meta.load_ids == case.demand.load_ids
        @test length(meta.nominal_pd) == meta.num_loads

        # A system-wide discrete law: one scalar per stage, applied to every load.
        sys = SystemMultiplier(DiscreteNonParametric([0.9, 1.1], [0.4, 0.6]))
        r = validate_sampler(sys, case.network, 6)
        @test r.num_loads == meta.num_loads
        A, p = finite_support(sys, 1, meta)
        @test size(A) == (meta.num_loads, 2)
        @test p ≈ [0.4, 0.6]
        @test all(A[:, 1] .== 0.9)

        # Independence is a particular joint law, not the representation.
        ind = IndependentMultiplier(Uniform(0.8, 1.2))
        ri = validate_sampler(ind, case.network, 4)
        @test 0.8 <= ri.multiplier_min && ri.multiplier_max <= 1.2
        one_draw = sample_multiplier_path(ind, meta, 1; seed = 5)
        @test length(unique(one_draw)) > 1        # really per-load, not broadcast

        # A regional group law is CORRELATED: loads inside a group move together.
        groups = [meta.load_ids[1:3], meta.load_ids[4:end]]
        grp = GroupMultiplier(groups, [DiscreteNonParametric([0.8, 1.2], [0.5, 0.5]),
                                       DiscreteNonParametric([1.0], [1.0])])
        v = sample_multiplier_path(grp, meta, 1; seed = 9)
        @test length(unique(v[1:3])) == 1
        @test all(v[4:end] .== 1.0)
        Ag, pg = finite_support(grp, 1, meta)
        @test size(Ag, 2) == 2
        @test sum(pg) ≈ 1
        @test_throws ArgumentError GroupMultiplier([[1, 2], [2, 3]], Normal())

        # Joint regions express what independent ones cannot: a NEGATIVE
        # correlation between regions, which is the whole locational case for
        # storage. Two regions, two modes, each stressing exactly one region.
        jr = JointRegionMultiplier(groups, [[1.2, 0.8], [0.8, 1.2]], [0.5, 0.5])
        Aj, pj = finite_support(jr, 1, meta)
        @test size(Aj, 2) == 2 && sum(pj) ≈ 1
        m1 = Aj[1, :]; m2 = Aj[4, :]
        e1 = sum(pj .* m1); e2 = sum(pj .* m2)
        cov12 = sum(pj .* (m1 .- e1) .* (m2 .- e2))
        @test cov12 < 0                       # unreachable with GroupMultiplier
        @test all(all(Aj[1:3, k] .== Aj[1, k]) for k in 1:2)   # region moves as one
        # Plain numbers throughout, so unlike the group sampler it round-trips.
        jb = sampler_from_dict(sampler_to_dict(jr))
        @test finite_support(jb, 1, meta)[1] == Aj
        @test finite_support(jb, 1, meta)[2] == pj
        @test_throws ArgumentError JointRegionMultiplier(groups, [[1.0, 1.0]], [0.6])
        @test_throws ArgumentError JointRegionMultiplier(groups, [[1.0]], [1.0])

        # Composition multiplies.
        prod = ProductMultiplier(sys, grp)
        Ap, pp = finite_support(prod, 1, meta)
        @test size(Ap, 2) == 4
        @test sum(pp) ≈ 1
        vp = sample_multiplier_path(prod, meta, 1; seed = 3)
        @test all(isfinite, vp)

        # A user callable is validated on every draw.
        bad = CallableMultiplier((rng, t, m) -> [1.0]; name = "wrong-length")
        @test_throws ErrorException sample_multiplier_path(bad, meta, 1; seed = 1)
        good = CallableMultiplier((rng, t, m) -> fill(1.0 + 0.1 * rand(rng), m.num_loads);
                                  name = "jitter")
        @test validate_sampler(good, case.network, 3).num_loads == meta.num_loads

        # Stage dependence.
        st = StageMultiplier(Dict(1 => DeterministicMultiplier(1.0), 2 => sys);
                             default = DeterministicMultiplier(0.5))
        @test all(sample_multiplier(st, StableRNG(1), 1, meta) .== 1.0)
        @test all(sample_multiplier(st, StableRNG(1), 9, meta) .== 0.5)

        # Round-trip of the samplers that can be serialized.
        fin = FiniteMultiplier([0.9, 1.0, 1.1], [0.25, 0.5, 0.25])
        back = sampler_from_dict(sampler_to_dict(fin))
        @test finite_support(back, 1, meta)[2] ≈ [0.25, 0.5, 0.25]
        @test_throws ErrorException sampler_from_dict(sampler_to_dict(good))
    end

    # ── Freezing: one support, consumed by both methods ──────────────────────
    @testset "finite-support freezing" begin
        meta = demand_meta(case.network)

        # An explicitly discrete sampler keeps its support and probabilities.
        fin = FiniteMultiplier([0.9, 1.0, 1.1], [0.25, 0.5, 0.25])
        s1 = freeze_demand_support(fin, case.network, 4; seed = 1, method = :exact,
                                   profile = 1.0)
        @test num_atoms(s1, 1) == 3
        @test atom_probabilities(s1, 1) ≈ [0.25, 0.5, 0.25]
        @test s1.atoms[1][:, 1] == fill(0.9, meta.num_loads)

        # A continuous sampler is discretized transparently and reproducibly.
        cont = SystemMultiplier(Uniform(0.9, 1.1))
        s2 = freeze_demand_support(cont, case.network, 3; seed = 7, atoms_per_stage = 5,
                                   method = :empirical, profile = 1.0)
        s3 = freeze_demand_support(cont, case.network, 3; seed = 7, atoms_per_stage = 5,
                                   method = :empirical, profile = 1.0)
        @test support_digest(s2) == support_digest(s3)
        @test num_atoms(s2, 1) == 5
        @test all(atom_probabilities(s2, 1) .≈ 1 / 5)
        s4 = freeze_demand_support(cont, case.network, 3; seed = 8, atoms_per_stage = 5,
                                   method = :empirical, profile = 1.0)
        @test support_digest(s2) != support_digest(s4)
        @test_throws ErrorException freeze_demand_support(cont, case.network, 2;
                                                          seed = 1, method = :exact)

        # Exact duplicates merge, and the probabilities they carried are summed.
        dup = FiniteMultiplier([1.0, 1.0, 2.0], [0.25, 0.25, 0.5])
        s5 = freeze_demand_support(dup, case.network, 1; seed = 1, method = :exact,
                                   profile = 1.0)
        @test num_atoms(s5, 1) == 2
        @test sort(atom_probabilities(s5, 1)) ≈ [0.5, 0.5]

        # The profile is materialized, not stored as a period, and it is
        # separable from the multiplier.
        prof = diurnal_profile(6; amplitude = 0.1, peak_hour = 3, period = 6)
        s6 = freeze_demand_support(fin, case.network, 6; seed = 1, method = :exact,
                                   profile = prof, profile_period = 6)
        @test profile_period(s6) == 6
        for t in 1:6
            @test all(s6.profile[:, t] .≈ prof[t])
            @test demand_multipliers(s6, t, 1) ≈ fill(prof[t] * 0.9, meta.num_loads)
        end

        # Stage-dependent supports survive freezing with different K per stage.
        st = StageMultiplier([FiniteMultiplier([1.0], [1.0]),
                              FiniteMultiplier([0.8, 1.2], [0.5, 0.5])])
        s7 = freeze_demand_support(st, case.network, 2; seed = 1, method = :exact,
                                   profile = 1.0)
        @test num_atoms(s7, 1) == 1
        @test num_atoms(s7, 2) == 2

        # A support round-trips through the artifacts byte-identically.
        tmp = mktempdir()
        try
            write_battery_case(tmp; name = case.name, network = case.network,
                               batteries = case.batteries, recourse = case.recourse,
                               demand = s6, source_version = "test",
                               protocol_stages = 6, protocol_scenarios = 4)
            reread = read_battery_case(tmp)
            @test support_digest(reread.demand) == support_digest(s6)
            @test reread.demand.atoms == s6.atoms
            @test reread.demand.profile == s6.profile
            # Rewriting produces identical bytes: the artifact is a pure function
            # of the value, which is the whole point of mirroring it.
            before = sha256_file(joinpath(tmp, "demand.json"))
            write_battery_case(tmp; name = case.name, network = case.network,
                               batteries = case.batteries, recourse = case.recourse,
                               demand = s6, source_version = "test",
                               protocol_stages = 6, protocol_scenarios = 4)
            @test sha256_file(joinpath(tmp, "demand.json")) == before
        finally
            rm(tmp; recursive = true, force = true)
        end
    end

    # ── Battery placement ────────────────────────────────────────────────────
    @testset "regional demand regime" begin
        net = case.network
        loads = load_buses(net)

        # Every load bus lands in exactly one region, and a bus that IS a center
        # belongs to its own region.
        centers = [loads[1], loads[end]]
        regs = nearest_bus_regions(net, centers)
        @test length(regs) == 2
        @test sort(vcat(regs...)) == sort(loads)
        @test centers[1] in regs[1] && centers[2] in regs[2]
        @test nearest_bus_regions(net, centers) == regs        # deterministic
        @test_throws ArgumentError nearest_bus_regions(net, Int[])

        # R regions cost R+2 atoms, not the 2^R of an independent product.
        s = rotating_regime_sampler(net; centers = centers, horizon = 24,
                                    min_probability = 0.0)
        meta = demand_meta(net)
        A, p = finite_support(s, 19, meta)
        @test size(A, 2) == length(centers) + 2
        @test sum(p) ≈ 1

        # The three things a policy would want to know all move with the stage.
        j1 = findfirst(==(regs[1][1]), meta.load_bus)
        j2 = findfirst(==(regs[2][1]), meta.load_bus)
        stats = map(1:24) do t
            At, pt = finite_support(s, t, meta)
            m1 = At[j1, :]; m2 = At[j2, :]
            e1 = sum(pt .* m1); e2 = sum(pt .* m2)
            v1 = sum(pt .* (m1 .- e1) .^ 2); v2 = sum(pt .* (m2 .- e2) .^ 2)
            c = sum(pt .* (m1 .- e1) .* (m2 .- e2))
            (mean1 = e1, corr = (v1 > 1e-12 && v2 > 1e-12) ? c / sqrt(v1 * v2) : 0.0)
        end
        @test maximum(x -> x.mean1, stats) - minimum(x -> x.mean1, stats) > 0.05
        # Two regions is the MINIMUM construction and gives the smallest possible
        # rotation: with R regions the stressed one is offset by 1/R of a period,
        # so the correlation swing grows with R. Measured 0.09 here against 0.60
        # on the three-region research case, hence a threshold that asserts the
        # correlation genuinely moves rather than one calibrated on the big case.
        @test maximum(x -> x.corr, stats) - minimum(x -> x.corr, stats) > 0.05

        # Pruning drops negligible modes, so quiet stages cost fewer atoms than
        # the peak — the knob that decides whether SDDP converges affordably.
        sp = rotating_regime_sampler(net; centers = centers, horizon = 24,
                                     min_probability = 0.06)
        counts = [length(finite_support(sp, t, meta)[2]) for t in 1:24]
        @test minimum(counts) < maximum(counts)
        @test all(c -> c >= 1, counts)
        @test all(sum(finite_support(sp, t, meta)[2]) ≈ 1 for t in 1:24)

        # `spread = 0` removes the rotation: every region is equally likely to be
        # the stressed one, and the process degenerates to a temporal one.
        flat = rotating_regime_sampler(net; centers = centers, horizon = 24,
                                       spread = 0.0, min_probability = 0.0)
        Af, pf = finite_support(flat, 19, meta)
        @test pf[2] ≈ pf[3]

        # The manual path: explicit region lists bypass the distance partition.
        man = rotating_regime_sampler(net; groups = [loads[1:2], loads[3:end]],
                                      horizon = 4, min_probability = 0.0)
        @test size(finite_support(man, 1, meta)[1], 2) == 4
        @test_throws ArgumentError rotating_regime_sampler(net; groups = [loads],
                                                           horizon = 4)
    end

    @testset "battery placement" begin
        net = case.network
        loads = load_buses(net)
        @test issorted(loads) && allunique(loads)

        # Explicit placement is honoured; an ineligible identifier is refused.
        b1, r1 = select_battery_buses(net, ExplicitPlacement([loads[1], loads[3]]))
        @test b1 == sort([loads[1], loads[3]])
        @test r1["strategy"] == "explicit"
        @test_throws ErrorException select_battery_buses(net, ExplicitPlacement([10_000]))
        dup = ExplicitPlacement([loads[1], loads[1]])
        @test_throws ErrorException select_battery_buses(net, dup)

        # A seeded draw is reproducible and depends on the seed.
        s1, _ = select_battery_buses(net, SampledPlacement(3; seed = 11))
        s2, _ = select_battery_buses(net, SampledPlacement(3; seed = 11))
        @test s1 == s2
        @test length(s1) == 3 && allunique(s1)
        # Different seeds give different draws. Not every PAIR of seeds must
        # differ — with 11 candidates and 3 picks a collision is ordinary — so
        # the property asserted is that the seed is a real input.
        draws = unique([select_battery_buses(net, SampledPlacement(3; seed = s))[1]
                        for s in 1:12])
        @test length(draws) > 1
        @test_throws ArgumentError select_battery_buses(net,
                                                        SampledPlacement(length(loads) + 1; seed = 1))

        # Weights are recorded, and a weight that concentrates all mass picks
        # the intended bus.
        load_at = nominal_load_at_bus(net)
        w, rw = select_battery_buses(net, SampledPlacement(1; seed = 3,
                                                           weight = b -> b == loads[2] ? 1.0 : 0.0))
        @test w == [loads[2]]
        @test rw["strategy"] == "weighted"
        @test length(rw["weights"]) == length(loads)
        @test_throws ErrorException select_battery_buses(net,
                                                         SampledPlacement(1; seed = 3,
                                                                          weight = b -> -1.0))

        # An eligibility predicate narrows the pool; a callable is validated
        # exactly like any other strategy's output.
        big = eligible_buses(net; eligible = b -> Int(b["index"]) in loads[1:2])
        @test big == sort(loads[1:2])
        cb, rc = select_battery_buses(net, CallablePlacement(
            (cands, meta) -> [first(cands)]; name = "first"))
        @test cb == [first(loads)]
        @test startswith(rc["strategy"], "callable:")
        @test_throws ErrorException select_battery_buses(net, CallablePlacement(
            (cands, meta) -> [10_000]))

        # Ratings, their validation, and a SAMPLED capacity expressed as a
        # callable closing over a distribution.
        fleet, crec = battery_fleet(net, b1; power = 0.5, energy_hours = 3.0,
                                    self_discharge = 0.99, throughput_cost = 1.0,
                                    initial_fraction = 0.25)
        @test length(fleet) == 2
        @test all(f -> f.energy_max ≈ 1.5, fleet)
        @test all(f -> f.energy_initial ≈ 0.375, fleet)
        @test [f.bus for f in fleet] == b1
        @test [f.index for f in fleet] == [1, 2]
        @test length(crec["power_pu"]) == 2
        rng = StableRNG(4)
        sampled, _ = battery_fleet(net, b1; power = _ -> rand(rng, Uniform(0.2, 0.4)),
                                   energy_hours = 2.0)
        @test all(f -> 0.2 <= f.charge_max <= 0.4, sampled)
        @test_throws ErrorException battery_fleet(net, b1; power = -1.0, energy_hours = 2.0)
        @test_throws ErrorException battery_fleet(net, b1; power = 1.0, energy_hours = 2.0,
                                                  charge_efficiency = 1.5)
    end

    # ── Battery physics, independent of any optimizer ────────────────────────
    @testset "battery dynamics" begin
        Δt = stage_hours(case)
        for b in case.batteries, e in (b.energy_min, b.energy_initial, b.energy_max)
            lo, hi = reachable_interval(b, e, Δt)
            @test b.energy_min - 1e-12 <= lo <= hi <= b.energy_max + 1e-12
            for frac in (0.0, 0.5, 1.0)
                tgt = lo + frac * (hi - lo)
                pch, pdis = dispatch_for_target(b, e, tgt, Δt)
                @test pch >= -1e-12 && pdis >= -1e-12
                @test pch <= b.charge_max + 1e-9
                @test pdis <= b.discharge_max + 1e-9
                @test min(pch, pdis) <= 1e-12          # never both at once
                @test b.self_discharge * e + b.charge_efficiency * Δt * pch -
                      (Δt / b.discharge_efficiency) * pdis ≈ tgt atol = 1e-10
            end
        end
    end

    # ── Provenance: the network formulation is PowerModels', not ours ────────
    @testset "external-model provenance" begin
        pm_acp = battery_stage_model(case, PowerModels.ACPPowerModel;
                                     optimizer = OPT, stage = 1, atom = 2, mode = :none)
        @test pm_acp isa PowerModels.ACPPowerModel
        @test typeof(pm_acp) === PowerModels.ACPPowerModel
        assert_powermodels_provenance(pm_acp, PowerModels.ACPPowerModel)
        @test_throws ErrorException assert_powermodels_provenance(pm_acp,
                                                                  PowerModels.SOCWRConicPowerModel)

        pm_soc = battery_stage_model(case, PowerModels.SOCWRConicPowerModel;
                                     stage = 1, atom = 2, mode = :none)
        @test typeof(pm_soc) === PowerModels.SOCWRConicPowerModel
        assert_powermodels_provenance(pm_soc, PowerModels.SOCWRConicPowerModel)

        # No network equation is written by hand in the problem specification or
        # in the diagnostics — AC trigonometry, the SOC-WR lifted variables and a
        # DC susceptance flow alike. Scanning for each catches a reimplementation
        # of any of the three formulations; the case builder and the residual
        # checker are out of scope because neither builds an optimization model.
        #
        # The trigonometric patterns are anchored at a non-identifier character.
        # A bare `"sin("` matches inside `occursin(`, which is an ordinary string
        # predicate and not a network equation — a false positive that would
        # eventually be silenced by deleting the check. The anchored form still
        # catches a qualified call such as `Base.sin(`, because a `.` is not an
        # identifier character.
        for f in ("battery_powermodels.jl", "battery_sddp.jl", "battery_diagnostics.jl")
            src = read(joinpath(EXAMPLE, f), String)
            code = join([l for l in split(src, '\n') if !startswith(strip(l), "#")], '\n')
            for pat in ("cos", "sin", "atan", "tan", "asin", "acos")
                @test !occursin(Regex("(?<![A-Za-z0-9_])" * pat * "\\s*\\("), code)
            end
            for pat in ("var(pm)[:w]", "var(pm)[:wr]", "var(pm)[:wi]")
                @test !occursin(pat, code)
            end
            # The DC network equation is PowerModels' too: `p_fr == -b Δθ` never
            # appears here, so a susceptance-times-angle-difference product is as
            # much of a red flag as a cosine is.
            @test !occursin(Regex("(?<![A-Za-z0-9_])calc_branch_y\\s*\\("), code)
        end
    end

    # ── Base network: the specification reduces to ordinary PowerModels OPF ──
    @testset "base network" begin
        # Find a stage/atom whose total multiplier is exactly 1, where the
        # battery layer's demand deviation is identically zero.
        t0, a0 = 0, 0
        for t in 1:horizon(case.demand), k in 1:num_atoms(case.demand, t)
            if all(demand_multipliers(case.demand, t, k) .≈ 1.0)
                t0, a0 = t, k
                break
            end
        end
        @test t0 > 0
        pm = battery_stage_model(case, PowerModels.ACPPowerModel;
                                 optimizer = OPT, stage = t0, atom = a0, mode = :none)
        JuMP.set_silent(pm.model)
        JuMP.optimize!(pm.model)
        ours = extract_stage_solution(pm)
        stock = PowerModels.solve_opf(deepcopy(case.network), PowerModels.ACPPowerModel, OPT)

        @test ours.solved
        @test stock["termination_status"] == JuMP.LOCALLY_SOLVED
        # The physical dispatch is the same to solver tolerance.
        # Active power carries the objective and is determined to 1e-6 pu.
        # Reactive power carries NO objective coefficient at all: it is pinned
        # only through the balance, and a direction the objective is flat in is
        # determined by an interior-point solve to whatever its own stopping rule
        # buys. The two sides here are two differently SCALED NLPs — ours divides
        # the objective by the cost scale, PowerModels' does not — so Ipopt's own
        # gradient-based scaling differs and the two iterates stop at slightly
        # different points of the same solution. Measured worst disagreement on
        # this case: 1.1e-6 pu = 1.1e-4 MVAr on a 100 MVA base, against 1.3e-2 pu
        # of reactive output. The cost and every active quantity agree at 1e-6,
        # which is what the comparison is for; the reactive tolerance is set to
        # the level the quantity is actually determined at, and the aggregate is
        # reported beside it so a real redistribution between generators sharing
        # a bus would still be caught.
        stock_qg_bus = Dict{Int,Float64}(i => 0.0 for i in keys(ours.qg_bus))
        for (k, g) in stock["solution"]["gen"]
            i = parse(Int, k)
            @test ours.pg[i] ≈ g["pg"] atol = 1e-6
            @test ours.qg[i] ≈ g["qg"] atol = 1e-5
            stock_qg_bus[Int(case.network["gen"][k]["gen_bus"])] += g["qg"]
        end
        for (i, q) in stock_qg_bus
            @test ours.qg_bus[i] ≈ q atol = 1e-5
        end
        for (k, b) in stock["solution"]["bus"]
            i = parse(Int, k)
            @test ours.vm[i] ≈ b["vm"] atol = 1e-6
            @test ours.va[i] ≈ b["va"] atol = 1e-6
        end
        # The generation cost is the stock objective. The two OBJECTIVES differ
        # by the recourse variables' interior-point barrier slack — an optimizer
        # parks a nonnegative variable a tolerance below its zero bound, and a
        # positively priced variable there lowers the objective by a
        # near-constant amount. That is a solver artefact, not a model
        # difference, so the comparison is made on the cost component the two
        # models actually share.
        @test ours.cost_generation ≈ stock["objective"] atol = 1e-5
        @test abs(ours.objective - ours.cost_stage) < 1e-8
    end

    # ── A generator that exists in some stages and not others ────────────────
    # The failure this guards against is silence: a schedule that is written into
    # a case, hashed, mirrored, and then ignored by the stage builder would give
    # every consumer a generator that is always available while every record says
    # otherwise.
    @testset "per-stage generator availability" begin
        # The convention is OPTIONAL and additive: the frozen case carries no
        # schedule, so nothing is scaled and the network is untouched.
        net0 = deepcopy(case.network)
        @test apply_stage_availability!(net0, 1) == 0
        @test canonical_json(net0) == canonical_json(case.network)

        # ── The construction helper rejects what it cannot model ──────────────
        src = acquire_pglib_case("pglib_opf_case14_ieee")
        host = 3
        @test_throws ArgumentError attach_scheduled_generator!(src.network, 10_000;
                                                               pmax = 1.0, cost = [1.0, 0.0])
        @test_throws ArgumentError attach_scheduled_generator!(src.network, host;
                                                               pmax = -1.0, cost = [1.0, 0.0])
        # A concave cost would make the conic stage problem a relaxation of
        # nothing, and a negative marginal cost pays the case to generate.
        @test_throws ArgumentError attach_scheduled_generator!(src.network, host;
                                                               pmax = 1.0, cost = [-1.0, 5.0, 0.0])
        @test_throws ArgumentError attach_scheduled_generator!(src.network, host;
                                                               pmax = 1.0, cost = [1.0, -5.0, 0.0])
        @test_throws ArgumentError attach_scheduled_generator!(src.network, host;
                                                               pmax = 1.0, cost = [1.0, 0.0],
                                                               availability = [1.0, -0.5])
        ngen0 = length(src.network["gen"])

        # A quadratic unit at a load bus, available in stage 1 and absent in 2.
        PMAX, C2, C1 = 0.35, 40.0, 900.0
        acq = attach_scheduled_generator!(src.network, host;
                                          pmax = PMAX, cost = [C2, C1, 0.0],
                                          availability = [1.0, 0.0], tag = "acq")
        @test length(src.network["gen"]) == ngen0 + 1
        @test !haskey(src.network["gen"], string(acq.gen + 1))
        @test src.network["gen"][string(acq.gen)][STAGE_AVAILABILITY_KEY] == [1.0, 0.0]
        # A malformed schedule is rejected while the case is still data.
        @test validate_network(src.network).scheduled_generators == 1
        bad = deepcopy(src.network)
        bad["gen"][string(acq.gen)][STAGE_AVAILABILITY_KEY] = Float64[]
        @test_throws ErrorException validate_network(bad)

        # ── Scaling is per stage, and covers reactive as well as active ───────
        two = deepcopy(src.network)
        two["gen"][string(acq.gen)]["qmin"] = -0.2
        two["gen"][string(acq.gen)]["qmax"] = 0.4
        s1 = deepcopy(two); s2 = deepcopy(two)
        @test apply_stage_availability!(s1, 1) == 1
        @test apply_stage_availability!(s2, 2) == 1
        @test s1["gen"][string(acq.gen)]["pmax"] == PMAX
        @test s1["gen"][string(acq.gen)]["qmax"] == 0.4
        @test s2["gen"][string(acq.gen)]["pmax"] == 0.0
        @test s2["gen"][string(acq.gen)]["pmin"] == 0.0
        @test s2["gen"][string(acq.gen)]["qmin"] == 0.0
        @test s2["gen"][string(acq.gen)]["qmax"] == 0.0
        # Every other generator is untouched by either stage.
        for k in keys(two["gen"])
            k == string(acq.gen) && continue
            @test s1["gen"][k]["pmax"] == two["gen"][k]["pmax"]
            @test s2["gen"][k]["pmax"] == two["gen"][k]["pmax"]
        end
        # A stage the schedule does not cover is an error, never the last entry.
        @test_throws ErrorException apply_stage_availability!(deepcopy(two), 3)

        # ── The schedule survives the frozen case, and the digest covers it ───
        fleet, crec = battery_fleet(src.network, [host];
                                    power = 0.10, energy_hours = 4.0,
                                    self_discharge = 0.999, throughput_cost = 1.0,
                                    initial_fraction = 0.5, reserve_fraction = 0.05)
        support = freeze_demand_support(FiniteMultiplier([1.0], [1.0]), src.network, 2;
                                        seed = 20260810, method = :exact,
                                        profile = 1.0, stage_hours = 1.0)
        dir = mktempdir()
        sched_case = build_case(src; dir = dir, batteries = fleet, support = support,
                                placement = Dict{String,Any}("acquisition" => acq.record),
                                protocol_stages = 2, protocol_scenarios = 4, quiet = true)
        @test sched_case.network["gen"][string(acq.gen)][STAGE_AVAILABILITY_KEY] == [1.0, 0.0]
        # Same case with a different schedule is a DIFFERENT network artifact, so
        # a run can never mistake one for the other.
        alt = acquire_pglib_case("pglib_opf_case14_ieee")
        attach_scheduled_generator!(alt.network, host; pmax = PMAX, cost = [C2, C1, 0.0],
                                    availability = [1.0, 1.0], tag = "acq")
        dir2 = mktempdir()
        alt_case = build_case(alt; dir = dir2, batteries = fleet, support = support,
                              protocol_stages = 2, protocol_scenarios = 4, quiet = true)
        @test sched_case.manifest["artifacts"]["network.json"] !=
              alt_case.manifest["artifacts"]["network.json"]

        # ── What the stage models actually contain ───────────────────────────
        bounds = Float64[]
        for t in 1:2
            pm = battery_stage_model(sched_case, PowerModels.ACPPowerModel;
                                     optimizer = OPT, stage = t, atom = 1, mode = :none)
            push!(bounds, JuMP.upper_bound(PowerModels.var(pm, :pg, acq.gen)))
        end
        @test bounds == [PMAX, 0.0]

        # And what they DISPATCH. The unit is priced above the case's own
        # marginal cost at zero output, so it earns its place only where the
        # network is short: what is asserted is the stage-2 zero, which is the
        # property the schedule exists for, and that the cost decomposition
        # still adds up on both stages.
        for t in 1:2
            s = base_feasibility(sched_case, PowerModels.ACPPowerModel; stage = t, atom = 1)
            @test s.solved
            @test abs(s.objective - s.cost_stage) < 1e-8
            if t == 2
                @test abs(s.pg[acq.gen]) <= 1e-8
            end
        end
        # The always-available twin is the null control for that zero: same unit,
        # same price, same bus, schedule [1, 1] — so a stage-2 dispatch that is
        # zero for any reason other than the schedule would be zero here too.
        avail2 = base_feasibility(alt_case, PowerModels.ACPPowerModel; stage = 2, atom = 1)
        @test avail2.solved
        @test JuMP.upper_bound(PowerModels.var(
                  battery_stage_model(alt_case, PowerModels.ACPPowerModel;
                                      optimizer = OPT, stage = 2, atom = 1, mode = :none),
                  :pg, acq.gen)) == PMAX
    end

    # ── Nodal prices: the SIGN is derived, not guessed ───────────────────────
    @testset "nodal prices" begin
        # ∂(stage cost)/∂(demand at bus i) must equal the reported active price.
        # The demand is moved through the battery layer's own deviation variable,
        # which is exactly how a demand realization enters, so the finite
        # difference measures the same derivative the price claims to be.
        base = base_feasibility(case, PowerModels.ACPPowerModel; stage = 3, atom = 2)
        @test base.solved
        @test !isempty(base.price_active)
        i0 = argmax(Dict(i => v for (i, v) in base.pd))    # a real load bus
        h = 1e-4
        vals = Float64[]
        for δ in (+h, -h)
            pm = battery_stage_model(case, PowerModels.ACPPowerModel;
                                     optimizer = OPT, stage = 3, atom = 2, mode = :none)
            bat = pm.ext[:battery]
            JuMP.fix(bat[:dpd][i0], JuMP.fix_value(bat[:dpd][i0]) + δ; force = true)
            JuMP.set_silent(pm.model)
            JuMP.optimize!(pm.model)
            push!(vals, extract_stage_solution(pm).cost_generation)
        end
        @test (vals[1] - vals[2]) / (2h) ≈ base.price_active[i0] rtol = 1e-3
        # A load bus prices energy positively: serving more demand costs more.
        @test base.price_active[i0] > 0
    end

    # ── Admissibility: every demand atom is served without recourse ──────────
    @testset "demand admissibility" begin
        worst = 0.0
        for t in 1:horizon(case.demand), a in 1:num_atoms(case.demand, t)
            b = base_feasibility(case, PowerModels.ACPPowerModel; stage = t, atom = a)
            @test b.solved
            worst = max(worst, b.worst_recourse)
        end
        # The base ACP problem is feasible at every atom with no recourse: this
        # is the admissibility precondition of the strict-target construction.
        @test worst < 1e-6
    end

    # ── Recourse structure ───────────────────────────────────────────────────
    @testset "recourse structure" begin
        ein = Dict(b.index => b.energy_initial for b in case.batteries)
        pm = battery_stage_model(case, PowerModels.ACPPowerModel;
                                 optimizer = OPT, stage = 5, atom = 2, mode = :strict,
                                 state_in = ein, target = ein)
        bat = pm.ext[:battery]
        d, s = bat[:d], bat[:s]
        for i in bat[:buses]
            @test JuMP.has_lower_bound(d[i]) && JuMP.lower_bound(d[i]) == 0.0
            @test JuMP.has_lower_bound(s[i]) && JuMP.lower_bound(s[i]) == 0.0
            @test !JuMP.has_upper_bound(d[i]) && !JuMP.has_upper_bound(s[i])
            @test !JuMP.is_fixed(d[i]) && !JuMP.is_fixed(s[i])
        end
        i0 = first(case.batteries).bus
        obj = JuMP.objective_function(pm.model)
        # The objective the SOLVER sees is the PHYSICAL stage cost, so the
        # recourse coefficients are the case's own prices, bit for bit. An exact
        # comparison is the point: anything else means a rescaling crept back in
        # between the case and the model.
        @test JuMP.coefficient(obj, d[i0]) === case.recourse.deficit
        @test JuMP.coefficient(obj, s[i0]) === case.recourse.surplus
        @test case.recourse.deficit > 0 && case.recourse.surplus > 0

        # Neither recourse variable touches the battery: not the transition,
        # not the strict target equality. There is no target slack.
        k0 = first(case.batteries).index
        for con in (bat[:transition][k0], bat[:target_con][k0], bat[:energy_in_con][k0])
            f = JuMP.constraint_object(con).func
            @test JuMP.coefficient(f, d[i0]) == 0.0
            @test JuMP.coefficient(f, s[i0]) == 0.0
        end
        # The recourse pair enters exactly one balance row, with opposite signs.
        rows = 0
        for (F, S) in JuMP.list_of_constraint_types(pm.model)
            F <: Union{JuMP.AffExpr,JuMP.QuadExpr} || continue
            for c in JuMP.all_constraints(pm.model, F, S)
                f = JuMP.constraint_object(c).func
                cd, cs = JuMP.coefficient(f, d[i0]), JuMP.coefficient(f, s[i0])
                (cd == 0 && cs == 0) && continue
                rows += 1
                @test cd * cs < 0
            end
        end
        @test rows == 1
    end

    # ── Physical objective units ─────────────────────────────────────────────
    # There is ONE unit system: the objective units of the underlying PGLib
    # case. Nothing between the frozen case and the solver rescales a cost, and
    # nothing on the way back converts one. What has to be protected is exactly
    # that: a case that records no scale, a model whose generator polynomial is
    # the case's own coefficient for coefficient, and a reported objective that
    # is the solver's own number rather than a converted one.
    @testset "physical objective units" begin
        # A case carries no objective scale, in memory or on disk, because there
        # is nothing to record: a second recorded unit is how two engines end up
        # reporting costs that differ by a constant ratio while both look right.
        @test !haskey(case.manifest, "cost_scale")
        @test !haskey(case.manifest["units"], "cost_normalization")
        tmp = mktempdir()
        try
            m = write_battery_case(tmp; name = case.name, network = case.network,
                                   batteries = case.batteries, recourse = case.recourse,
                                   demand = case.demand, source_version = "test",
                                   protocol_stages = 4, protocol_scenarios = 2)
            @test !haskey(m, "cost_scale")
            @test !haskey(m["units"], "cost_normalization")
            @test read_battery_case(tmp) isa BatteryCase
        finally
            rm(tmp; recursive = true, force = true)
        end

        ein = Dict(b.index => b.energy_initial for b in case.batteries)
        Δt = stage_hours(case)
        tgt = Dict{Int,Float64}()
        for b in case.batteries
            lo, hi = reachable_interval(b, ein[b.index], Δt)
            tgt[b.index] = lo + 0.6 * (hi - lo)
        end
        pm = battery_stage_model(case, PowerModels.ACPPowerModel;
                                 optimizer = OPT, stage = 11, atom = 2, mode = :strict,
                                 state_in = ein, target = tgt)

        # The cost polynomial PowerModels built its objective from is the case's
        # own, BIT for bit — not equal to a tolerance, which a rescale-and-undo
        # round trip would also pass. The case's network is unmoved by building
        # a model, so a stock `solve_opf` on it stays a valid external reference.
        ncoef = 0
        for (_, g) in case.network["gen"]
            i = Int(g["index"])
            phys = Float64.(collect(g["cost"]))       # a JSON-parsed Vector{Any}
            got = Float64.(collect(PowerModels.ref(pm, :gen, i)["cost"]))
            @test length(got) == length(phys)
            for (a, b) in zip(got, phys)
                @test a === b
                ncoef += 1
            end
        end
        @test ncoef > 0                               # the loop actually compared something

        JuMP.set_silent(pm.model); JuMP.optimize!(pm.model)
        sol = extract_stage_solution(pm)
        @test sol.solved
        # The reported objective is the solver's own value, unconverted.
        @test sol.objective === JuMP.objective_value(pm.model)
        # And the generation component is the PHYSICAL polynomial at the reported
        # dispatch, evaluated here from `case.network` rather than from `ref`.
        phys_gen = sum(_polynomial_cost(g, sol.pg[Int(g["index"])])
                       for (_, g) in case.network["gen"])
        @test sol.cost_generation ≈ phys_gen rtol = 1e-9
        # Recourse is priced at the case's prices, so a stage cost assembled from
        # the case's own numbers reproduces the objective the solver minimized.
        @test sol.cost_stage ≈ sol.objective rtol = 1e-7
    end

    # ── The strict formulation ───────────────────────────────────────────────
    @testset "strict targets" begin
        Δt = stage_hours(case)
        ein = Dict(b.index => b.energy_initial for b in case.batteries)
        # Dynamic reachability does NOT imply network deliverability: at the
        # demand peak, charging every battery at its rating can exceed what a
        # small network delivers, and the uncapped deficit injection supplies the
        # difference. That is the recourse doing its job, and it is why a
        # selected policy is required to leave both recourse variables at zero
        # rather than merely to exist.
        peak = argmax([sum(values(realized_bus_demand(case, t, num_atoms(case.demand, t))[1]))
                       for t in 1:24])
        for frac in (0.0, 0.45, 1.0 - 1e-3)
            tgt = Dict{Int,Float64}()
            for b in case.batteries
                lo, hi = reachable_interval(b, ein[b.index], Δt)
                tgt[b.index] = lo + frac * (hi - lo)
            end
            sol = solve_strict_stage(case, PowerModels.ACPPowerModel;
                                     stage = peak, atom = num_atoms(case.demand, peak),
                                     energy_in = ein, target = tgt, optimizer = OPT)
            @test sol.solved
            for b in case.batteries
                # The target is met EXACTLY: a strict equality, no slack, in
                # every case — including one that needs recourse.
                @test sol.energy_out[b.index] ≈ tgt[b.index] atol = 1e-9
                @test isfinite(sol.target_dual[b.index])
                @test min(sol.p_ch[b.index], sol.p_dis[b.index]) < 1e-6
            end
            # The physics of the reported solution, recomputed independently.
            r = physical_residuals(case.network, case.batteries, Δt, sol)
            @test r.branch_flow < 1e-6
            @test r.active_balance < 1e-8
            @test r.reactive_balance < 1e-8
            @test r.transition < TRANSITION_RESIDUAL_TOL
            @test r.thermal < 1e-6
            @test r.voltage < 1e-6
        end

        # The multiplier is the derivative of the solved value in the target.
        tgt = Dict{Int,Float64}()
        for b in case.batteries
            lo, hi = reachable_interval(b, ein[b.index], Δt)
            tgt[b.index] = lo + 0.45 * (hi - lo)
        end
        base = solve_strict_stage(case, PowerModels.ACPPowerModel;
                                  stage = 17, atom = 3, energy_in = ein,
                                  target = tgt, optimizer = OPT)
        k = first(case.batteries).index
        h = 1e-5
        up = copy(tgt); up[k] += h
        dn = copy(tgt); dn[k] -= h
        vp = solve_strict_stage(case, PowerModels.ACPPowerModel; stage = 17, atom = 3,
                                energy_in = ein, target = up, optimizer = OPT).objective
        vm_ = solve_strict_stage(case, PowerModels.ACPPowerModel; stage = 17, atom = 3,
                                 energy_in = ein, target = dn, optimizer = OPT).objective
        @test (vp - vm_) / (2h) ≈ base.target_dual[k] rtol = 1e-3
    end

    # ── The diagnostic toolkit ───────────────────────────────────────────────
    @testset "diagnostics" begin
        Δt = stage_hours(case)

        # B1: sampled incoming energies land inside their own bounds, and the
        # draw is reproducible from the seed.
        e1 = sample_incoming_energy(case; kind = :uniform, seed = 5, batch = 3)
        e2 = sample_incoming_energy(case; kind = :uniform, seed = 5, batch = 3)
        @test e1 == e2
        @test length(e1) == 3
        for e in e1, b in case.batteries
            @test b.energy_min <= e[b.index] <= b.energy_max
        end
        efix = sample_incoming_energy(case; kind = :fixed, level = 0.5)[1]
        @test all(b -> efix[b.index] ≈ 0.5 * b.energy_max, case.batteries)
        @test_throws ErrorException sample_incoming_energy(case; kind = :explicit,
            explicit = [Dict(b.index => 10.0 * b.energy_max for b in case.batteries)])

        # B2: the targetless probe is myopic — nothing prices what it leaves
        # behind — and its physics check out.
        p = targetless_probe(case, PowerModels.ACPPowerModel;
                             energy_in = efix, stage = 19, atom = num_atoms(case.demand, 19))
        @test p.solved
        @test worst_recourse(p) < 1e-6
        @test maximum(values(p.simultaneous)) < 1e-6
        @test p.residuals.active_balance < 1e-8
        @test p.residuals.transition < 1e-9
        @test all(b -> p.energy_out[b.index] <= efix[b.index] + 1e-6, case.batteries)
        @test !isempty(p.energy_in_dual)

        # A `:free` model with every battery pinned IS the strict stage problem.
        tgt = Dict(b.index => 0.5 * b.energy_max for b in case.batteries)
        pinned = targetless_probe(case, PowerModels.ACPPowerModel; energy_in = efix,
                                  stage = 19, atom = 3, pins = tgt)
        strict = solve_strict_stage(case, PowerModels.ACPPowerModel; stage = 19, atom = 3,
                                    energy_in = efix, target = tgt, optimizer = OPT)
        @test pinned.cost_stage ≈ strict.cost_stage atol = 1e-8
        for b in case.batteries
            @test pinned.pin_dual[b.index] ≈ strict.target_dual[b.index] atol = 1e-6
        end
        @test_throws ArgumentError battery_stage_model(case, PowerModels.ACPPowerModel;
                                                        stage = 1, atom = 1, mode = :free,
                                                        target = tgt)

        # B3: the value curve, its multipliers and the finite-difference check.
        b1 = first(case.batteries)
        lo, hi = reachable_interval(b1, efix[b1.index], Δt)
        curve = energy_value_curve(case, PowerModels.ACPPowerModel; battery = b1.index,
                                   energy_in = efix, stage = 19, atom = 3,
                                   grid = collect(range(lo, hi; length = 7)))
        @test all(curve.solved)
        @test curve.endpoint[1]
        @test curve.endpoint[end]
        smooth = [i for i in 2:6 if !curve.nonsmooth[i]]
        @test !isempty(smooth)
        @test all(i -> curve.fd_error[i] < 5e-3, smooth)
        # Convexity in the outgoing energy: the value curve's slopes increase.
        slopes = diff(curve.value) ./ diff(curve.energy)
        @test all(diff(slopes) .> -1e-6)
        # A target outside the reachable interval is reported, never solved.
        wide = energy_value_curve(case, PowerModels.ACPPowerModel; battery = b1.index,
                                  energy_in = efix, stage = 19, atom = 3,
                                  grid = [hi + 1.0])
        @test !wide.reachable[1]
        @test !wide.solved[1]

        cmp = compare_value_curves(case; battery = b1.index, energy_in = efix,
                                   stage = 19, atom = 3,
                                   grid = collect(range(lo, hi; length = 7)))
        @test length(cmp.interior) >= 1
        @test isfinite(cmp.max_abs_dlambda)

        # B4: the deterministic equivalent is a horizon solved at once, and its
        # cost is the sum of its stages'.
        proto = scenario_index_matrix(case.demand, 6, 3)
        de = deterministic_equivalent(case, proto[:, 1])
        @test de.solved
        @test de.total_cost ≈ sum(de.stage_cost) atol = 1e-8
        @test de.worst_deficit < 1e-6 && de.worst_surplus < 1e-6
        # The state really propagates across stages.
        for t in 2:de.horizon, b in case.batteries
            @test de.stages[t].energy_in[b.index] ≈ de.stages[t - 1].energy_out[b.index] atol = 1e-8
        end
        for r in de.residuals
            @test r.active_balance < 1e-8
            @test r.transition < TRANSITION_RESIDUAL_TOL
            @test r.branch_flow < 1e-6
        end
        # Perfect foresight is at most as expensive as any nonanticipative
        # decision on the SAME path: pinning the batteries to any admissible
        # trajectory cannot beat optimizing them.
        hold_cost = 0.0
        e = Dict(b.index => b.energy_initial for b in case.batteries)
        for t in 1:6
            tgt_t = Dict(b.index => b.self_discharge * e[b.index] for b in case.batteries)
            s = solve_strict_stage(case, PowerModels.ACPPowerModel; stage = t,
                                   atom = proto[t, 1], energy_in = e, target = tgt_t,
                                   optimizer = OPT)
            hold_cost += s.cost_stage
            e = copy(tgt_t)
        end
        @test de.total_cost <= hold_cost + 1e-6

        # B5: the panel keeps every requested identifier and fails closed.
        panel = perfect_foresight_panel(case, proto)
        @test panel.complete
        @test panel.num_solved == 3
        @test [r.scenario for r in panel.rows] == [1, 2, 3]
        @test panel.mean ≈ mean([r.cost for r in panel.rows])
        sub = perfect_foresight_panel(case, proto; ids = [2])
        @test sub.rows[1].scenario == 2
        @test sub.rows[1].cost ≈ panel.rows[2].cost rtol = 1e-8
        @test_throws ArgumentError perfect_foresight_panel(case, proto; ids = [99])

        pd = paired_difference([1.0, 2.0, 3.0], [1.5, 1.5, 1.5])
        @test pd.n == 3
        @test pd.mean ≈ 0.5
        @test pd.wins == 1
    end

    # ── Analysis helpers speak the shared schema ─────────────────────────────
    @testset "analysis" begin
        proto = scenario_index_matrix(case.demand, 4, 2)
        de = deterministic_equivalent(case, proto[:, 1])
        rec = SolutionRecorder()
        record_trajectory!(rec, de, case; scenario = 1)
        df = solution_frame(rec)
        @test all(in(SOLUTION_CLASSES), unique(df.class))
        @test nrow(df) > 0
        wide = pivot_class(df, "energy_out"; scenario = 1)
        @test nrow(wide) == 4
        @test Symbol(string(first(case.batteries).index)) in propertynames(wide)

        ct = stage_cost_table(de)
        @test nrow(ct) == 4
        @test ct.cumulative[end] ≈ de.total_cost atol = 1e-8
        bt = battery_table(de, case)
        @test nrow(bt) == 4 * length(case.batteries)
        @test maximum(bt.simultaneous) < 1e-6
        pt = price_table(de, case)
        @test nrow(pt) == 4 * length(case.network["bus"])

        @test competerank_desc([3.0, 1.0, 2.0]) == [1, 3, 2]
        @test competerank_desc([1.0, 1.0, 2.0]) == [2, 2, 1]

        # A round-trip through the solution file preserves every record.
        tmp = mktempdir()
        try
            path = write_solution(joinpath(tmp, "sol.csv"), rec)
            back = read_solution(path)
            @test length(back) == length(rec.rows)
            @test nrow(solution_frame(path)) == nrow(df)
        finally
            rm(tmp; recursive = true, force = true)
        end
    end

    # ── Stock SDDP over the shared specification ─────────────────────────────
    @testset "SDDP construction" begin
        trained = train_battery_sddp(case; num_stages = 2, iteration_limit = 3)
        # Both passes really are the formulations the study claims.
        assert_graph_provenance(trained.backward_graph, PowerModels.SOCWRConicPowerModel)
        assert_graph_provenance(trained.forward, PowerModels.ACPPowerModel)
        # The default arm is SOC, and it says so in every identifying field.
        @test trained.backward === :soc
        @test trained.backward_formulation === PowerModels.SOCWRConicPowerModel
        @test trained.bound_bounds_acp
        @test sddp_method_id(trained) === :sddp_soc
        # Battery energy is a genuine SDDP state.
        node = trained.forward.nodes[1]
        @test length(node.states) == length(case.batteries)
        # Stock cuts were created.
        ncuts = sum(length(n.bellman_function.global_theta.cuts) for (_, n) in trained.backward_graph.nodes)
        @test ncuts > 0
        @test isfinite(trained.bound)
        # The graph refuses a horizon the frozen support does not cover.
        @test_throws ArgumentError battery_policy_graph(case, PowerModels.ACPPowerModel;
                                                         num_stages = horizon(case.demand) + 1,
                                                         optimizer = OPT)

        ids = [b.index for b in case.batteries]
        sims = simulate_battery_sddp(trained, 3; ids = ids)
        @test length(sims) == 3
        for path in sims
            @test length(path) == 2
            # The outgoing state of stage 1 is the incoming state of stage 2.
            @test path[1][:energy_out] ≈ path[2][:energy_in] atol = 1e-8
            for t in 1:2
                @test path[t][:deficit] < 1e-5
                @test path[t][:surplus] < 1e-5
                @test isfinite(path[t][:stage_objective])
            end
        end
    end

    # ── One shared ACP bound relaxation, on every true-ACP path ─────────────
    #
    # The two solvers do not agree on what this default is — MadNLP was measured
    # resolving `1e-8` on CPU and `1e-10` on GPU when none was given — so an
    # inherited default is not a cross-engine agreement. It is therefore stated
    # explicitly at every site, from ONE constant, and this test reads the
    # attributes an optimizer actually carries rather than trusting the source.
    @testset "ACP bound-relaxation parity" begin
        @test ACP_BOUND_RELAX_FACTOR == 1e-8
        attrs = Dict{String,Any}()
        for (a, v) in acp_optimizer().params
            a isa MOI.RawOptimizerAttribute && (attrs[a.name] = v)
        end
        @test haskey(attrs, "bound_relax_factor")
        @test attrs["bound_relax_factor"] == ACP_BOUND_RELAX_FACTOR
        # It is EXPLICIT, not inherited: overriding it reaches the solver.
        other = Dict(a.name => v for (a, v) in acp_optimizer(bound_relax_factor = 2e-8).params
                     if a isa MOI.RawOptimizerAttribute)
        @test other["bound_relax_factor"] == 2e-8
        # Every true-ACP consumer in this engine goes through `acp_optimizer`:
        # the strict stage solve, the SDDP forward pass of BOTH arms, and the
        # diagnostics. The conic backward solvers are a different formulation and
        # deliberately do not carry it.
        src = read(joinpath(EXAMPLE, "battery_sddp.jl"), String)
        @test occursin("forward_optimizer = acp_optimizer()", src)
        conic = Dict(a.name => v for (a, v) in socwr_optimizer().params
                     if a isa MOI.RawOptimizerAttribute)
        @test !haskey(conic, "bound_relax_factor")
    end

    # ── The DC-approximation backward formulation ────────────────────────────
    #
    # The claim is that the DC arm is PowerModels' own `DCPPowerModel` carrying
    # this study's own battery layer on the case's own unmodified generator
    # data — not a handwritten DC network, not a repriced case, and not a second
    # SDDP implementation. Each group below establishes one clause of that, and
    # the last one establishes that the two arms cannot be mistaken for each
    # other anywhere they are written down.
    @testset "DC backward formulation" begin
        Δt = stage_hours(case)
        DCOPT = dc_optimizer()
        stage, atom = 17, 3
        ein = Dict(b.index => b.energy_initial for b in case.batteries)
        tgt = Dict{Int,Float64}()
        for b in case.batteries
            lo, hi = reachable_interval(b, ein[b.index], Δt)
            tgt[b.index] = lo + 0.45 * (hi - lo)      # interior: away from both bounds
        end

        # ── Provenance, in both directions ───────────────────────────────────
        pm_dc = battery_stage_model(case, PowerModels.DCPPowerModel;
                                    optimizer = DCOPT, stage = stage, atom = atom,
                                    mode = :strict, state_in = ein, target = tgt)
        @test typeof(pm_dc) === PowerModels.DCPPowerModel
        assert_powermodels_provenance(pm_dc, PowerModels.DCPPowerModel)
        @test_throws ErrorException assert_powermodels_provenance(pm_dc, PowerModels.ACPPowerModel)
        # A model that HAS a magnitude or a reactive variable is not the DC one,
        # and the assertion says so rather than passing on the type alone.
        pm_acp = battery_stage_model(case, PowerModels.ACPPowerModel;
                                     optimizer = OPT, stage = stage, atom = atom, mode = :none)
        @test_throws ErrorException assert_powermodels_provenance(pm_acp, PowerModels.DCPPowerModel)
        # The battery really did reach the DC nodal balance: PowerModels closes
        # it over the storage carriers that exist when it is written.
        @test !isempty(PowerModels.ref(pm_dc, :bus_storage, first(sort!(collect(PowerModels.ids(pm_dc, :bus))))))

        JuMP.set_silent(pm_dc.model)
        JuMP.optimize!(pm_dc.model)
        dc = extract_stage_solution(pm_dc)
        @test dc.solved
        # The DC approximation models no voltage magnitude and no reactive
        # quantity, and the extraction reports that as absence, not as zero.
        @test all(isnan, values(dc.vm))
        @test all(isfinite, values(dc.va))
        @test isempty(dc.qg) && isempty(dc.q_fr) && isempty(dc.q_to)
        @test isempty(dc.price_reactive)
        @test length(dc.price_active) == length(dc.va)

        # ── The battery layer, on this formulation ──────────────────────────
        @test worst_recourse(dc) < 1e-6
        for b in case.batteries
            k = b.index
            @test dc.energy_out[k] ≈ tgt[k] atol = 1e-9        # the strict equality
            @test dc.energy_in[k] ≈ ein[k] atol = 1e-9
            # The charge/discharge split the transition forces, computed from the
            # case contract rather than read back from the model.
            want_ch, want_dis = dispatch_for_target(b, ein[k], tgt[k], Δt)
            @test dc.p_ch[k] ≈ want_ch atol = 1e-8
            @test dc.p_dis[k] ≈ want_dis atol = 1e-8
            @test min(dc.p_ch[k], dc.p_dis[k]) < 1e-8
        end
        # The cost decomposes exactly into the physical components, with the
        # generator polynomial re-evaluated at the reported dispatch.
        @test dc.cost_stage ≈ dc.objective atol = 1e-6
        @test dc.cost_throughput ≈ sum(b.throughput_cost * Δt * (dc.p_ch[b.index] + dc.p_dis[b.index])
                                       for b in case.batteries)

        # ── The direct PowerModels DC oracle ────────────────────────────────
        # An ordinary `PowerModels.solve_opf` on a network with NO storage table
        # at all, whose per-bus load is the realized demand minus the battery's
        # net injection. If the battery layer does nothing but move an injection,
        # the two must agree — and the oracle shares no code with the model it is
        # checking beyond PowerModels itself.
        pd_real, qd_real = realized_bus_demand(case, stage, atom)
        pbat = Dict{Int,Float64}(i => 0.0 for i in keys(pd_real))
        for b in case.batteries
            pbat[b.bus] += dc.p_bat[b.index]
        end
        onet = deepcopy(case.network)
        apply_stage_availability!(onet, stage)
        onet["load"] = Dict{String,Any}()
        for (n, i) in enumerate(sort!(collect(keys(pd_real))))
            onet["load"][string(n)] = Dict{String,Any}(
                "index" => n, "load_bus" => i, "status" => 1,
                "pd" => pd_real[i] - pbat[i], "qd" => qd_real[i])
        end
        oracle = PowerModels.solve_opf(onet, PowerModels.DCPPowerModel, DCOPT)
        @test oracle["termination_status"] == JuMP.OPTIMAL
        # Generation, generator by generator, and the generation cost.
        for (kk, g) in oracle["solution"]["gen"]
            @test dc.pg[parse(Int, kk)] ≈ g["pg"] atol = 1e-6
        end
        @test dc.cost_generation ≈ oracle["objective"] atol = 1e-5
        # Angles are determined only up to the reference, which both models pin
        # at the same bus, so the flows are the invariant thing to difference.
        for (kk, br) in oracle["solution"]["branch"]
            @test dc.p_fr[parse(Int, kk)] ≈ br["pf"] atol = 1e-6
        end

        # ── DC balance and line flows, recomputed independently ─────────────
        # `physical_residuals` re-derives AC flows from voltage magnitudes and
        # angles, which the DC approximation does not have. The DC physics is
        # therefore recomputed here, from the reported angles and the branch
        # data, with no reference to the model that produced them:
        #     p_fr = -b (θ_f − θ_t),   p_to = −p_fr,
        #     Σ_arcs p = Σ pg + p^bat + d − s − p^d − g^s.
        inj = Dict{Int,Float64}(i => 0.0 for i in keys(dc.va))
        worst_flow = 0.0
        for (_, br) in case.network["branch"]
            Int(get(br, "br_status", 1)) == 0 && continue
            l = Int(br["index"])
            f, t = Int(br["f_bus"]), Int(br["t_bus"])
            _, bsusc = PowerModels.calc_branch_y(br)
            want_fr = -bsusc * (dc.va[f] - dc.va[t])
            worst_flow = max(worst_flow, abs(want_fr - dc.p_fr[l]),
                             abs(dc.p_to[l] + dc.p_fr[l]))     # lossless: p_to = −p_fr
            inj[f] -= dc.p_fr[l]
            inj[t] -= dc.p_to[l]
        end
        @test worst_flow < 1e-8
        for (_, gen) in case.network["gen"]
            Int(get(gen, "gen_status", 1)) == 0 && continue
            inj[Int(gen["gen_bus"])] += dc.pg[Int(gen["index"])]
        end
        for (_, sh) in get(case.network, "shunt", Dict{String,Any}())
            Int(get(sh, "status", 1)) == 0 && continue
            # The DC balance charges the conductive shunt at a flat 1.0 pu.
            inj[Int(sh["shunt_bus"])] -= Float64(get(sh, "gs", 0.0))
        end
        for b in case.batteries
            inj[b.bus] += dc.p_bat[b.index]
        end
        worst_balance = 0.0
        for i in keys(inj)
            worst_balance = max(worst_balance,
                                abs(inj[i] + dc.deficit[i] - dc.surplus[i] - dc.pd[i]))
        end
        @test worst_balance < 1e-8
        # The battery transition, from the reported controls.
        for b in case.batteries
            k = b.index
            @test abs(dc.energy_out[k] - b.self_discharge * dc.energy_in[k]
                      - b.charge_efficiency * Δt * dc.p_ch[k]
                      + (Δt / b.discharge_efficiency) * dc.p_dis[k]) < 1e-9
        end

        # ── State duals against centered finite differences ─────────────────
        # At an interior state and an interior target, under the same sign
        # convention the ACP arm is checked at: the multiplier of `e_out == t̂`
        # is ∂obj/∂t̂, and the multiplier of `e_in == e` is ∂obj/∂e.
        k = first(case.batteries).index
        h = 1e-5
        solve_dc(ei, tt) = solve_strict_stage(case, PowerModels.DCPPowerModel;
                                              stage = stage, atom = atom,
                                              energy_in = ei, target = tt,
                                              optimizer = DCOPT)
        base = solve_dc(ein, tgt)
        up = copy(tgt); up[k] += h
        dn = copy(tgt); dn[k] -= h
        @test (solve_dc(ein, up).objective - solve_dc(ein, dn).objective) / (2h) ≈
              base.target_dual[k] rtol = 1e-4
        eup = copy(ein); eup[k] += h
        edn = copy(ein); edn[k] -= h
        @test (solve_dc(eup, tgt).objective - solve_dc(edn, tgt).objective) / (2h) ≈
              base.energy_in_dual[k] rtol = 1e-4

        # ── A real three-stage SDDP run on DC backward nodes ────────────────
        dctrain = train_battery_sddp(case; backward = :dc, num_stages = 3, iteration_limit = 4)
        @test dctrain.backward === :dc
        @test dctrain.backward_formulation === PowerModels.DCPPowerModel
        @test sddp_method_id(dctrain) === :sddp_dc
        # LIVE nodes, every one of them, on both graphs.
        assert_graph_provenance(dctrain.backward_graph, PowerModels.DCPPowerModel)
        assert_graph_provenance(dctrain.forward, PowerModels.ACPPowerModel)
        for (_, n) in dctrain.backward_graph.nodes
            @test typeof(n.subproblem.ext[:pm]) === PowerModels.DCPPowerModel
        end
        for (_, n) in dctrain.forward.nodes
            @test typeof(n.subproblem.ext[:pm]) === PowerModels.ACPPowerModel
        end
        # DC backward solves produced stock cuts, and the storage state is a
        # genuine SDDP state that propagates.
        dccuts = sum(length(n.bellman_function.global_theta.cuts)
                     for (_, n) in dctrain.backward_graph.nodes)
        @test dccuts > 0
        @test isfinite(dctrain.bound)
        ids = [b.index for b in case.batteries]
        sims = simulate_battery_sddp(dctrain, 3; ids = ids)
        for path in sims
            @test length(path) == 3
            for t in 1:2
                @test path[t][:energy_out] ≈ path[t + 1][:energy_in] atol = 1e-8
            end
            for t in 1:3
                # The forward paths are true ACP and use no physical recourse.
                @test path[t][:deficit] < 1e-6
                @test path[t][:surplus] < 1e-6
                @test isfinite(path[t][:stage_objective])
            end
        end

        # ── Cut transfer preserves intercepts and coefficients exactly ──────
        # `AlternativePostIterationCallback` copies each new cut from the
        # backward graph into the ACP graph. "Copies" has to mean the same
        # numbers, or the graph being simulated is carrying a different value
        # function from the one that was trained.
        for (key, bnode) in dctrain.backward_graph.nodes
            fcuts = dctrain.forward.nodes[key].bellman_function.global_theta.cuts
            bcuts = bnode.bellman_function.global_theta.cuts
            @test length(fcuts) == length(bcuts)
            for (bc, fc) in zip(bcuts, fcuts)
                @test fc.intercept == bc.intercept          # exactly, not approximately
                @test fc.coefficients == bc.coefficients
            end
        end

        # ── SOC behaviour is untouched, and the two cannot be confused ──────
        soc = train_battery_sddp(case; backward = :soc, num_stages = 2, iteration_limit = 2)
        assert_graph_provenance(soc.backward_graph, PowerModels.SOCWRConicPowerModel)
        @test soc.backward === :soc && soc.bound_bounds_acp
        # The DC scalar is NOT called a bound on the ACP problem, anywhere.
        @test dctrain.bound_name == "DC-approximation training bound"
        @test !dctrain.bound_bounds_acp
        @test soc.bound_name == "SOC-WR relaxation bound"
        @test soc.bound_name != dctrain.bound_name
        # Serialized cuts carry the arm in the file name, and a path that does
        # not is refused rather than written.
        dir = mktempdir()
        @test basename(sddp_cut_path(dir, case, :soc)) != basename(sddp_cut_path(dir, case, :dc))
        @test "dc" in split(basename(sddp_cut_path(dir, case, :dc)), '.')
        @test "socwr" in split(basename(sddp_cut_path(dir, case, :soc)), '.')
        # Neither arm's path can be mistaken for the other's.
        @test_throws ArgumentError train_battery_sddp(case; backward = :dc, num_stages = 2,
                                                      iteration_limit = 1,
                                                      cut_path = sddp_cut_path(dir, case, :soc))
        @test_throws ArgumentError train_battery_sddp(case; backward = :soc, num_stages = 2,
                                                      iteration_limit = 1,
                                                      cut_path = sddp_cut_path(dir, case, :dc))
        @test_throws ArgumentError train_battery_sddp(case; backward = :dc, num_stages = 2,
                                                      iteration_limit = 1,
                                                      cut_path = joinpath(dir, "$(case.name).cuts.json"))
        # A case name that merely CONTAINS the tag is not the tag.
        @test_throws ArgumentError train_battery_sddp(case; backward = :dc, num_stages = 2,
                                                      iteration_limit = 1,
                                                      cut_path = joinpath(dir, "hvdcline.cuts.json"))
        written = train_battery_sddp(case; backward = :dc, num_stages = 2, iteration_limit = 1,
                                     cut_path = sddp_cut_path(dir, case, :dc))
        @test isfile(written.cut_path)
        @test occursin("dc", basename(written.cut_path))
        # An unknown backward formulation is refused by name.
        @test_throws ArgumentError train_battery_sddp(case; backward = :acp, num_stages = 2)
        @test_throws ArgumentError backward_spec(:socwr)

        # ── The cost report labels the arm, and drops the DC scalar from the
        #    recoverable ceiling ───────────────────────────────────────────────
        # This is the ONE place where the difference between the two scalars
        # changes arithmetic rather than only a label. The SOC bound lower-bounds
        # the best nonanticipative cost and may tighten the cap on what a better
        # policy could win; the DC training bound does not and may not, or the
        # report would claim a smaller prize than the one that exists.
        proto = scenario_index_matrix(case.demand, 2, 4)
        dc2 = train_battery_sddp(case; backward = :dc, num_stages = 2, iteration_limit = 2)
        for tr in (soc, dc2)
            r = cost_report(case, tr, proto; columns = 1:2, forecast = false, io = devnull)
            @test r.method === sddp_method_id(tr)
            @test r.backward === tr.backward
            @test r.bound_name == tr.bound_name
            @test r.bound_bounds_acp == tr.bound_bounds_acp
            want = tr.bound_bounds_acp ? max(r.bound, r.pf_mean) : r.pf_mean
            @test r.recoverable_ceiling ≈ (r.sddp_mean - want) / r.sddp_mean
        end
    end

    # ── The four method identifiers ─────────────────────────────────────────
    @testset "method identifiers" begin
        @test sort!(collect(keys(BATTERY_METHODS))) ==
              [:sddp_dc, :sddp_soc, :tsddr_nonlinear, :tsldr_recurrent_linear]
        for id in keys(BATTERY_METHODS)
            m = battery_method(id)
            @test m.id === id
            # The invariants are what make the four comparable, so every row
            # carries them and every row carries the SAME ones.
            @test m.horizon == 24
            @test m.protocol == "screening"
            @test m.stage_semantics == BATTERY_METHOD_INVARIANTS.stage_semantics
            @test m.recourse == BATTERY_METHOD_INVARIANTS.recourse
            @test m.cost_contract == BATTERY_METHOD_INVARIANTS.cost_contract
            @test m.comparison == BATTERY_METHOD_INVARIANTS.comparison
        end
        @test battery_method(:sddp_soc).backward === :soc
        @test battery_method(:sddp_dc).backward === :dc
        @test battery_method(:tsldr_recurrent_linear).architecture === :tsldr_recurrent_linear
        @test battery_method(:tsddr_nonlinear).engine === :exa
        @test_throws ArgumentError battery_method(:sddp)
        # This engine refuses the two it does not own, by name and with the
        # owning engine in the message, rather than failing deeper down.
        @test_throws ErrorException run_battery_method(:tsddr_nonlinear, case)
        @test_throws ErrorException run_battery_method(:tsldr_recurrent_linear, case)
        # And it really does run the two it does own, on the arm it names.
        for (id, want) in ((:sddp_soc, PowerModels.SOCWRConicPowerModel),
                           (:sddp_dc, PowerModels.DCPPowerModel))
            out = run_battery_method(id, case; num_stages = 2, iteration_limit = 1)
            @test sddp_method_id(out) === id
            @test out.backward_formulation === want
            assert_graph_provenance(out.backward_graph, want)
        end
    end

    # ── The preregistered PGLib portfolio ────────────────────────────────────
    #
    # The panel itself is frozen by the campaign, not by this suite: acquiring
    # pglib_opf_case2000_goc and running its ACP headroom calibration is hours of
    # solver time. What is tested here is every RULE the panel is a consequence
    # of, on networks small enough to run in seconds — structural rules on
    # `case118` (parsing only, no solve) and the end-to-end freeze on `case14`.
    @testset "portfolio panel" begin
        p118 = acquire_pglib_case("pglib_opf_case118_ieee")
        sha118 = network_digest(p118.network)

        @testset "hash-keyed placement" begin
            # The key rule is a SPECIFICATION: assert the literal bytes, so a
            # reader reimplementing it in another language has something to
            # check against, and so a refactor cannot quietly redraw the panel.
            @test portfolio_key("placement", "abc", 7) ==
                  "battery_storage_opf/portfolio/1\nplacement\n20260814\nabc\n7\n"
            u = portfolio_uniform("placement", sha118, 42)
            @test 0 < u < 1
            @test u == portfolio_uniform("placement", sha118, 42)
            # Different tag, different network and different id are all
            # independent draws.
            @test u != portfolio_uniform("region-label", sha118, 42)
            @test u != portfolio_uniform("placement", sha118, 43)
            @test u != portfolio_uniform("placement", repeat("0", 64), 42)

            pool = portfolio_eligible_buses(p118.network)
            load_at = nominal_load_at_bus(p118.network)
            @test issorted(pool)
            @test all(load_at[b] > 0 for b in pool)

            w = Dict(b => load_at[b] for b in pool)
            sel = weighted_selection(pool, w, 24; tag = "placement", network_sha = sha118)
            @test length(sel) == 24
            @test allunique(sel)
            @test issorted(sel)
            @test sel ⊆ pool
            # Reproducible, and independent of the order the pool is presented in.
            @test sel == weighted_selection(reverse(pool), w, 24;
                                            tag = "placement", network_sha = sha118)
            # A weight far above every other must be selected: the exponential
            # key -log(u)/w is driven to zero by a large w whatever u is.
            heavy = Dict(w); heavy[last(pool)] = 1e9
            @test last(pool) in weighted_selection(pool, heavy, 3;
                                                   tag = "placement", network_sha = sha118)
            # Independently recompute the rule and compare the whole selection.
            keys_ = sort([(-log(portfolio_uniform("placement", sha118, b)) / w[b], b)
                          for b in pool]; by = x -> (x[1], x[2]))
            @test sel == sort([b for (_, b) in keys_[1:24]])
            @test_throws ArgumentError weighted_selection(pool, w, length(pool) + 1;
                                                          tag = "placement",
                                                          network_sha = sha118)
        end

        @testset "dictionary-order invariance" begin
            # A network rebuilt through a completely different construction path
            # — canonical JSON out, JSON parser back in — is the same content in
            # a differently laid-out `Dict`. Every panel quantity must be blind
            # to that.
            round_tripped = plain(JSON.parse(canonical_json(plain(p118.network))))
            @test network_digest(round_tripped) == sha118
            @test portfolio_eligible_buses(round_tripped) ==
                  portfolio_eligible_buses(p118.network)
            b1, r1 = portfolio_placement(p118.network, sha118)
            b2, r2 = portfolio_placement(round_tripped, sha118)
            @test b1 == b2
            @test r1["digest"] == r2["digest"]
            g1 = portfolio_regions(p118.network)
            g2 = portfolio_regions(round_tripped)
            @test g1.digest == g2.digest
            @test g1.regions == g2.regions
            @test g1.corridors == g2.corridors
        end

        @testset "battery count and power budget" begin
            n_bus = length(p118.network["bus"])
            elig = length(portfolio_eligible_buses(p118.network))
            @test portfolio_battery_count(p118.network) ==
                  min(elig, clamp(round(Int, 0.20 * n_bus), 24, 240))
            @test portfolio_battery_count(p118.network) == 24   # 0.20 × 118 = 23.6 → clamped up

            buses, _ = portfolio_placement(p118.network, sha118)
            peak = 12.345
            power, budget, rec = portfolio_ratings(p118.network, buses, peak)
            @test budget == 0.10 * peak
            @test length(power) == length(buses)
            # Aggregate power equals the declared budget to numerical precision,
            # summed in the order it was accumulated in.
            total = sum(power[b] for b in sort(buses))
            @test abs(total - budget) <= 8 * eps(budget) * length(buses)
            @test all(p > 0 for p in values(power))
            # The cap binds on a heavy-tailed system: the largest RAW weight must
            # exceed three medians, and no CAPPED weight may.
            load_at = nominal_load_at_bus(p118.network)
            raw = [load_at[b] for b in sort(buses)]
            @test maximum(raw) > 3.0 * Statistics.median(raw)
            @test all(v <= rec["weight_cap_pu"] + 1e-12
                      for v in values(rec["capped_weight_pu"]))
            @test rec["weight_cap_pu"] ≈ 3.0 * Statistics.median(raw)
        end

        @testset "PTDF regions" begin
            ptdf = ptdf_matrix(p118.network)
            # The reference column of a PTDF matrix is identically zero: an
            # injection at the reference, withdrawn at the reference, moves
            # nothing.
            @test all(ptdf.M[:, ptdf.pos[ptdf.ref]] .== 0)
            @test size(ptdf.M, 1) == length(ptdf.live)
            corridors = select_corridors(p118.network, ptdf)
            @test issorted(corridors)
            @test length(corridors) <= PORTFOLIO_MAX_CORRIDORS
            @test corridors == select_corridors(p118.network, ptdf)

            g = portfolio_regions(p118.network)
            @test length(g.regions) == PORTFOLIO_REGIONS
            @test all(!isempty, g.regions)                    # every region has buses
            @test all(g.sizes .> 0)
            # ── The balance gate ────────────────────────────────────────────
            # Six regions are six LEVERS only if they carry comparable demand.
            # An unconstrained clustering does not: on the first freeze it put
            # 80.8 % of case1951_rte's demand in one region, and the six atoms
            # then move one direction rather than six.
            @test all(s -> s >= PORTFOLIO_REGION_SHARE_MIN - 1e-9, g.demand_share)
            @test all(s -> s <= PORTFOLIO_REGION_SHARE_MAX + 1e-9, g.demand_share)
            @test !g.exception                                # case118 needs no exception
            @test g.effective_regions >= PORTFOLIO_MIN_EFFECTIVE_REGIONS
            @test g.effective_regions ≈ 1 / sum(abs2, g.demand_share)
            # The balance is bought with signature dispersion, and the price is
            # reported rather than assumed. NO ordering is asserted between the
            # two: for FIXED representatives the balanced assignment is a
            # restriction of the free one and can only be worse, but the two
            # refine their representatives along different trajectories and both
            # are local searches, so the balanced one can and does land in a
            # better basin — measured at 0.98× on case118 and 0.79× on case300.
            # What must hold is that balancing did not turn the regions into an
            # arbitrary partition, which is what a dispersion blow-up would mean.
            @test isfinite(g.unconstrained_dispersion)
            @test g.dispersion <= 2.0 * g.unconstrained_dispersion
            @test g.dispersion < 1.0
            @test allunique(vcat(g.regions...))               # a partition, not a cover
            load_at = nominal_load_at_bus(p118.network)
            loadbus = sort([b for b in keys(load_at) if load_at[b] > 0])
            @test sort(vcat(g.regions...)) == loadbus         # every load bus is in one region
            # Demand conservation: the regions carry the whole nominal demand.
            @test sum(g.demand_pu) ≈ sum(load_at[b] for b in loadbus) rtol = 1e-12
            @test sum(g.demand_share) ≈ 1.0 atol = 1e-12
            @test all(g.demand_share .> 0)
            # Region IDs are canonical: labelled by descending demand.
            @test issorted(g.demand_pu; rev = true)
            @test g.digest == portfolio_regions(p118.network).digest
        end

        @testset "physical cost semantics" begin
            # The contract that keeps a barrier artifact out of a reported cost.
            # Both engines compute the headline cost with THIS code, so it is
            # tested here on synthetic solutions whose right answer is arithmetic
            # rather than a solve.
            prices = RecourseCosts(1.0e5, 2.0e5)

            # Nothing used: every element is within tolerance and projects to
            # exactly zero, so the corrected cost is generation plus throughput
            # and the whole raw penalty is the correction.
            n = 4000
            d = Dict(i => 1.0e-8 for i in 1:n)
            s = Dict(i => -1.0e-8 for i in 1:n)
            raw_pen = prices.deficit * sum(values(d)) + prices.surplus * sum(values(s))
            sol = (cost_generation = 1000.0, cost_throughput = 7.0,
                   deficit = d, surplus = s, objective = 1007.0 + raw_pen)
            c = physical_stage_cost(sol, prices)
            @test c.admissible
            @test c.deficit == 0.0
            @test c.surplus == 0.0
            @test c.corrected == 1007.0
            @test c.raw ≈ 1007.0 + raw_pen
            @test c.correction ≈ raw_pen
            @test c.worst_recourse ≈ 1.0e-8
            # Four thousand buses at 1e-8 pu is 4e-5 pu of "recourse" and, at
            # these prices, a raw penalty of several cost units. That is the leak
            # this contract exists to stop, so assert it was material and that it
            # did not reach the corrected number.
            @test abs(raw_pen) > 1.0
            @test c.corrected == sol.cost_generation + sol.cost_throughput

            # One element outside tolerance: NOT projected, and the solve is
            # inadmissible rather than silently priced.
            d2 = copy(d); d2[7] = 0.5
            sol2 = (cost_generation = 1000.0, cost_throughput = 7.0,
                    deficit = d2, surplus = s, objective = 0.0)
            c2 = physical_stage_cost(sol2, prices)
            @test !c2.admissible
            @test c2.worst_recourse ≈ 0.5
            @test c2.deficit ≈ prices.deficit * 0.5      # kept, not projected away
            @test c2.corrected ≈ 1007.0 + prices.deficit * 0.5

            # The projection is element-wise, not aggregate: many elements each
            # just OUTSIDE tolerance must be rejected, not averaged into silence.
            d3 = Dict(i => 2.0e-6 for i in 1:n)
            c3 = physical_stage_cost((cost_generation = 0.0, cost_throughput = 0.0,
                                      deficit = d3, surplus = Dict(1 => 0.0),
                                      objective = 0.0), prices)
            @test !c3.admissible
            @test c3.deficit ≈ prices.deficit * sum(values(d3))

            # And a negative excursion beyond tolerance is a violation too — a
            # recourse variable below zero is not physical.
            c4 = physical_stage_cost((cost_generation = 0.0, cost_throughput = 0.0,
                                      deficit = Dict(1 => -1.0e-3),
                                      surplus = Dict(1 => 0.0), objective = 0.0), prices)
            @test !c4.admissible
            @test c4.worst_recourse ≈ 1.0e-3

            # `project_recourse` reports the worst RAW value, before projection,
            # so an admissibility decision is never made on projected data.
            p, worst, ok = project_recourse(Dict(1 => 3.0e-7, 2 => -9.0e-7))
            @test ok && p[1] == 0.0 && p[2] == 0.0
            @test worst ≈ 9.0e-7
        end

        @testset "aggregates cancel; the element-wise contract does not" begin
            # WHY THIS FIXTURE EXISTS.
            # The SDDP simulation recorders used to return the nodal recourse
            # SUMMED over buses, and the panel was judged on that total. This is
            # the counterexample that shows a total is not a conservative
            # stand-in for the element-wise rule: it is a different test with a
            # different answer.
            prices = RecourseCosts(1.0e5, 1.0e5)
            tol = PHYSICAL_RECOURSE_TOL

            # CASE A — tolerance-scale residues of opposite sign at different
            # buses. Every element is inside tolerance and the stage is
            # admissible; the point is that the TOTAL is exactly zero, so a test
            # on the total learns nothing about any bus.
            n = 1000
            small_d = Dict(i => (isodd(i) ? 1 : -1) * 5.0e-7 for i in 1:n)
            zero_s = Dict(i => 0.0 for i in 1:n)
            # The 500 positive and 500 negative terms do not cancel to a bit-exact
            # zero in floating point, and the claim does not need them to: the
            # AGGREGATE is inside tolerance, which is all an aggregate test looks
            # at, while every individual element is inside it too.
            @test abs(sum(values(small_d))) <= tol             # the aggregate is blind
            a = physical_stage_cost((cost_generation = 10.0, cost_throughput = 1.0,
                                     deficit = small_d, surplus = zero_s,
                                     objective = 11.0), prices; tol = tol)
            @test a.admissible                                  # each |v| <= tol
            @test a.worst_recourse ≈ 5.0e-7                     # but each one is SEEN
            @test a.deficit == 0.0                              # projected element-wise
            @test a.corrected ≈ 11.0

            # CASE B — the failure the aggregate would wave through. One bus is
            # short by 1 pu and another has a surplus of 1 pu at the same price,
            # so BOTH the deficit total and the net charge are unchanged from a
            # clean stage. The stage must still be rejected, because a bus that
            # is 1 pu short is a violation whatever another bus is doing.
            big_d = Dict(1 => 1.0, 2 => 0.0)
            big_s = Dict(1 => 0.0, 2 => 1.0)
            b = physical_stage_cost((cost_generation = 10.0, cost_throughput = 1.0,
                                     deficit = big_d, surplus = big_s,
                                     objective = 11.0), prices; tol = tol)
            @test !b.admissible
            @test b.worst_recourse ≈ 1.0
            # An aggregate rule of the shape the recorders once supported —
            # `max(sum(deficit), sum(surplus)) <= tol`, with the sums clamped at
            # zero the way the old recorders clamped them — would accept CASE C
            # below, where the deficits cancel to a total of zero.
            cancel_d = Dict(1 => 1.0, 2 => -1.0)
            aggregate_verdict = max(0.0, sum(values(cancel_d))) <= tol
            @test aggregate_verdict                              # the old rule: PASS
            c = physical_stage_cost((cost_generation = 10.0, cost_throughput = 1.0,
                                     deficit = cancel_d, surplus = Dict(1 => 0.0, 2 => 0.0),
                                     objective = 11.0), prices; tol = tol)
            @test !c.admissible                                  # the contract: REJECT
            @test c.worst_recourse ≈ 1.0
            # The two rules therefore DISAGREE on a realizable stage, which is
            # what makes the aggregate insufficient rather than merely coarser.
            @test aggregate_verdict != c.admissible
        end

        @testset "SDDP path cost is element-wise" begin
            # `sddp_physical_path_cost` reads a simulation record. Feeding it a
            # synthetic record keeps the property under test — validate, project,
            # then sum — separate from any solver behaviour.
            case = built.case
            prices = case.recourse
            bus = [11, 22, 33]
            stage(dv, sv, gen) = Dict{Symbol,Any}(
                :bus_ids => bus, :deficit_by_bus => dv, :surplus_by_bus => sv,
                :cost_generation => gen, :cost_throughput => 0.0,
                :stage_objective => gen)

            # Two stages, every element inside tolerance: admissible, and the
            # corrected cost is generation alone because every element projects
            # to exactly zero.
            clean = [stage([4.0e-7, -4.0e-7, 0.0], [0.0, 0.0, 0.0], 100.0),
                     stage([0.0, 1.0e-7, -1.0e-7], [0.0, 0.0, 0.0], 200.0)]
            r = sddp_physical_path_cost(case, clean, 2)
            @test r.admissible
            @test r.cost ≈ 300.0
            @test r.worst_recourse ≈ 4.0e-7

            # One element out of tolerance in stage 2, cancelled in the total by
            # another bus. The path is REJECTED and the charge is the projected
            # one, element by element.
            dirty = [clean[1], stage([2.0, -2.0, 0.0], [0.0, 0.0, 0.0], 200.0)]
            r2 = sddp_physical_path_cost(case, dirty, 2)
            @test !r2.admissible
            @test r2.worst_recourse ≈ 2.0
            @test abs(sum(dirty[2][:deficit_by_bus])) <= PHYSICAL_RECOURSE_TOL  # invisible in aggregate
            @test r2.cost ≈ 300.0 + prices.deficit * 0.0    # +2 and −2 still sum to 0…
            # …so the COST is not what catches it. The admissibility flag is, and
            # that is exactly why the flag exists and why the panel is refused on
            # it rather than on a cost comparison.
        end

        @testset "balanced assignment" begin
            # The assignment rule on its own, against synthetic signatures whose
            # right answer is known — no network, no solver of the network, and
            # therefore no way for a network property to hide a rule defect.
            #
            # Six well-separated clusters of ten buses each, all of equal weight:
            # the balanced assignment must recover exactly those clusters,
            # because the balanced optimum and the free optimum coincide when
            # the demand is already balanced.
            k = PORTFOLIO_REGIONS
            S = zeros(60, k)
            w = fill(1.0, 60)
            for i in 1:60
                S[i, (i - 1) % k + 1] = 1.0
            end
            r = balanced_assignment(S, w, k)
            @test length(r.assign) == 60
            @test sort(unique(r.assign)) == collect(1:k)
            @test all(count(==(j), r.assign) == 10 for j in 1:k)
            @test !r.exception
            @test r.cap == PORTFOLIO_REGION_SHARE_MAX
            # Deterministic: the same input gives the same assignment, twice.
            @test balanced_assignment(S, w, k).assign == r.assign
            # And independent of the order the rows are presented in, up to the
            # relabelling that reordering induces — the PARTITION is what the
            # rule determines.
            perm = vcat(31:60, 1:30)
            rp = balanced_assignment(S[perm, :], w[perm], k)
            part(a, idx) = Set(Set(idx[findall(==(j), a)]) for j in 1:k)
            @test part(r.assign, collect(1:60)) == part(rp.assign, perm)

            # ── Balance actually binds ──────────────────────────────────────
            # One cluster carrying six times the demand of the others: the free
            # clustering would hand it a single region with 6/11 of the demand,
            # and the constraint must split it instead.
            w2 = fill(1.0, 60)
            for i in 1:60
                (i - 1) % k + 1 == 1 && (w2[i] = 6.0)
            end
            r2 = balanced_assignment(S, w2, k)
            share2 = [sum(w2[findall(==(j), r2.assign)]) / sum(w2) for j in 1:k]
            @test all(s -> s >= PORTFOLIO_REGION_SHARE_MIN - 1e-9, share2)
            @test all(s -> s <= PORTFOLIO_REGION_SHARE_MAX + 1e-9, share2)
            @test 1 / sum(abs2, share2) >= PORTFOLIO_MIN_EFFECTIVE_REGIONS

            # ── The single-heavy-bus exception ──────────────────────────────
            # A bus is indivisible. One bus carrying 40 % of the demand cannot
            # be placed anywhere without breaking a 28 % cap, so the cap rises
            # to exactly that bus's share and EXACTLY ONE region may use it.
            w3 = fill(1.0, 60)
            w3[1] = 0.40 * 59 / 0.60          # bus 1 holds 40 % of the total
            r3 = balanced_assignment(S, w3, k)
            @test r3.exception
            @test r3.cap ≈ maximum(w3) / sum(w3)
            @test r3.cap > PORTFOLIO_REGION_SHARE_MAX
            share3 = [sum(w3[findall(==(j), r3.assign)]) / sum(w3) for j in 1:k]
            @test count(s -> s > PORTFOLIO_REGION_SHARE_MAX + 1e-9, share3) == 1
            @test maximum(share3) <= r3.cap + 1e-9
            @test all(s -> s >= PORTFOLIO_REGION_SHARE_MIN - 1e-9, share3)
            # Without an exception the cap would be infeasible, and the rule
            # says so rather than silently widening every region.
            @test_throws ErrorException balanced_assignment(S, w3, k;
                                                            share_min = 0.20,
                                                            share_max = 0.21)
        end

        @testset "finite joint support" begin
            g = portfolio_regions(p118.network)
            support = portfolio_support(p118.network, g.regions, 1.0; seed = 11)
            @test support.horizon == PORTFOLIO_HORIZON == 24
            @test support.stage_hours == PORTFOLIO_STAGE_HOURS
            @test length(support.atoms) == 24
            @test length(support.probabilities) == 24
            n = length(support.load_ids)
            for t in 1:24
                @test num_atoms(support, t) == PORTFOLIO_REGIONS
                @test size(support.atoms[t]) == (n, PORTFOLIO_REGIONS)
                @test all(atom_probabilities(support, t) .≈ 1 / PORTFOLIO_REGIONS)
                # The stage's own profile value, and nothing else, is what
                # varies over stages: the six multiplier vectors are the same at
                # every stage.
                @test support.atoms[t] == support.atoms[1]
                @test all(support.profile[:, t] .≈ PORTFOLIO_PROFILE[t])
            end
            # Every load carries exactly one of the two multipliers in every
            # atom, and its support mean is one. Not `==` 1: 1.15 and 0.97 are
            # not binary fractions, so the arithmetic that is exact in ℝ leaves
            # an ulp behind in Float64.
            for t in (1, 11, 24)
                A = support.atoms[t]
                @test all(v ≈ PORTFOLIO_REGION_HIGH || v ≈ PORTFOLIO_REGION_LOW for v in A)
                μ = A * atom_probabilities(support, t)
                @test all(m -> isapprox(m, 1.0; atol = 1e-15), μ)
            end
            # Inter-region covariance is NEGATIVE: a region is high exactly when
            # the others are low, which a system-wide multiplier cannot express.
            M = portfolio_mode_matrix(PORTFOLIO_REGIONS)
            p = fill(1 / PORTFOLIO_REGIONS, PORTFOLIO_REGIONS)
            μ = M * p
            for r in 1:PORTFOLIO_REGIONS, s in 1:PORTFOLIO_REGIONS
                cov = sum(p[k] * (M[r, k] - μ[r]) * (M[s, k] - μ[s])
                          for k in 1:PORTFOLIO_REGIONS)
                r == s ? (@test cov > 0) : (@test cov < 0)
            end
        end

        @testset "headroom search" begin
            # The grid is written down, not accumulated: exactly the two
            # endpoints the constants name, descending, hitting the bottom
            # exactly rather than an ulp away from it.
            g = kappa_grid()
            @test g[1] == PORTFOLIO_KAPPA_HI
            @test g[end] == PORTFOLIO_KAPPA_LO
            @test issorted(g; rev = true)
            @test all(isapprox(g[i] - g[i + 1], PORTFOLIO_KAPPA_STEP; atol = 1e-12)
                      for i in 1:(length(g) - 1))

            # The search, isolated from what it searches on. A predicate with a
            # known threshold pins both properties that matter: the level
            # returned was ACCEPTED by the gate, and it is within one tolerance
            # of the true threshold.
            θ = 0.8137
            r = search_kappa_max(κ -> κ <= θ)
            @test r.ok
            @test r.kappa_max <= θ
            @test r.kappa_max > θ - PORTFOLIO_KAPPA_TOL
            @test r.kappa_max == search_kappa_max(κ -> κ <= θ).kappa_max
            # The same threshold costs the same number of gate evaluations on
            # any machine: the scan and the bisection are both deterministic.
            n = Ref(0); search_kappa_max(κ -> (n[] += 1; κ <= θ))
            m = Ref(0); search_kappa_max(κ -> (m[] += 1; κ <= θ))
            @test n[] == m[] > 0

            # The property the plain bisection did not have: admissibility that
            # is an INTERVAL rather than a lower set. A network that has to
            # spill at low demand and binds at high demand is admissible only in
            # the middle, and the search must find the top of that window rather
            # than call the case a data-gate failure.
            window = search_kappa_max(κ -> 0.70 <= κ <= θ)
            @test window.ok
            @test window.kappa_max <= θ
            @test window.kappa_max > θ - PORTFOLIO_KAPPA_TOL

            # A case admissible everywhere freezes at the top of the bracket; a
            # case admissible nowhere is the replacement condition.
            @test search_kappa_max(κ -> true).kappa_max == PORTFOLIO_KAPPA_HI
            bad = search_kappa_max(κ -> false)
            @test !bad.ok
            @test occursin("no level", bad.reason)
        end

        # ── The end-to-end freeze, on the smallest available network ─────────
        mktempdir() do root
            src14 = acquire_pglib_case("pglib_opf_case14_ieee")
            built = build_portfolio_case("pglib_opf_case14_ieee";
                                         dir = joinpath(root, "c14"), quiet = true)
            @test built.ok
            mpath = joinpath(root, "battery_portfolio.json")
            m = write_portfolio_manifest(mpath, [built.record]; min_cases = 1)

            @testset "manifest round-trip and tamper rejection" begin
                back = read_portfolio_manifest(mpath)
                @test back["digest"] == m["digest"]
                @test back["cases"][1]["kappa_case"] == built.record["kappa_case"]
                @test verify_portfolio_manifest(mpath; min_cases = 1)["digest"] == m["digest"]
                # Writing the same manifest twice gives the same bytes: no
                # timestamp, hostname or path leaks into it.
                second = joinpath(root, "again.json")
                write_portfolio_manifest(second, [built.record]; min_cases = 1)
                @test read(mpath) == read(second)
                # Any edit to any field is rejected by the self-digest.
                raw = JSON.parsefile(mpath)
                raw["cases"][1]["kappa_case"] = Float64(raw["cases"][1]["kappa_case"]) + 1e-9
                tampered = joinpath(root, "tampered.json")
                write_canonical_json(tampered, raw)
                @test_throws ErrorException read_portfolio_manifest(tampered)
                raw2 = JSON.parsefile(mpath)
                raw2["cases"][1]["placement"]["buses"][1] += 1
                tampered2 = joinpath(root, "tampered2.json")
                write_canonical_json(tampered2, raw2)
                @test_throws ErrorException read_portfolio_manifest(tampered2)
            end

            @testset "screening and final protocols are disjoint" begin
                c14 = built.case
                mf = c14.manifest
                @test mf["screening"]["excludes"] == "protocol"
                @test mf["screening"]["num_scenarios"] == PORTFOLIO_SCREENING_SCENARIOS
                @test mf["protocol"]["num_scenarios"] == PORTFOLIO_FINAL_SCENARIOS
                final = scenario_index_matrix(c14.demand, PORTFOLIO_HORIZON,
                                              PORTFOLIO_FINAL_SCENARIOS)
                screen = scenario_index_matrix(c14.demand, PORTFOLIO_HORIZON,
                                               PORTFOLIO_SCREENING_SCENARIOS;
                                               seed = mf["screening"]["seed"],
                                               exclude = protocol_columns(final))
                @test isempty(intersect(Set(protocol_columns(final)),
                                        Set(protocol_columns(screen))))
                @test size(final) == (24, 500)
                @test size(screen) == (24, 32)

                # The property has to hold where a collision is not merely
                # unlikely but forced: two stages of two atoms is four distinct
                # paths, and two protocols of two columns exhaust them.
                tiny = freeze_demand_support(FiniteMultiplier([0.9, 1.1], [0.5, 0.5]),
                                             src14.network, 2; seed = 5, method = :exact,
                                             profile = 1.0, stage_hours = 1.0)
                a = scenario_index_matrix(tiny, 2, 2; seed = 101)
                b = scenario_index_matrix(tiny, 2, 2; seed = 102,
                                          exclude = protocol_columns(a))
                @test isempty(intersect(Set(protocol_columns(a)), Set(protocol_columns(b))))
                # The REPAIRED protocol's own columns are distinct too; the
                # unrepaired one's are not required to be — it is an i.i.d.
                # sample and a final panel that refused to repeat a scenario
                # would not be one.
                @test length(unique(protocol_columns(b))) == 2
                @test Set(protocol_columns(b)) ⊆ Set([[i, j] for i in 1:2 for j in 1:2])
                # Asking for more disjoint columns than the support has paths is
                # a specification error, caught rather than looped on: four
                # two-stage paths cannot supply five columns disjoint from any
                # exclusion at all.
                @test_throws ArgumentError scenario_index_matrix(tiny, 2, 5;
                                                                 seed = 103,
                                                                 exclude = protocol_columns(a))
            end

            @testset "regeneration from the public command" begin
                out = joinpath(root, "regen")
                @test portfolio_main(["--case", "pglib_opf_case14_ieee",
                                      "--out", out, "--manifest", mpath]) == 0
                for f in ("network.json", "batteries.json", "demand.json")
                    @test bytes2hex(sha256(read(joinpath(out, "pglib_opf_case14_ieee", f)))) ==
                          built.record["artifacts"][f]
                end
                @test portfolio_main(["--verify", "--manifest", mpath,
                                      "--min-cases", "1"]) == 0
                @test portfolio_main(["--list", "--manifest", mpath]) == 0
                # A case the manifest does not carry is refused, not built.
                @test_throws ErrorException materialize_portfolio_case("pglib_opf_case30_ieee";
                                                                       dir = out, manifest = m)
            end

            @testset "strict-stage validation" begin
                v = validate_portfolio_case(built.case; stages = 1:2, quiet = true)
                @test v.ok
                @test v.states_ok
                @test v.reachable_ok
                @test v.hold_slack > 0        # holding is strictly inside the interval
                @test v.completed == v.attempted == 2 * PORTFOLIO_REGIONS
                @test v.worst_residual <= PORTFOLIO_RESIDUAL_TOL
                @test v.worst_recourse <= PORTFOLIO_RECOURSE_TOL
            end
        end

        @testset "cross-engine shared files" begin
            # `battery_case.jl`, `battery_solution_schema.jl` and the portfolio
            # manifest are shipped as byte-identical copies in the two engines,
            # which are independent packages that may not include each other's
            # files. `DR_BAT_PEER` names the peer example when it is checked out
            # somewhere this path cannot guess.
            peer = get(ENV, "DR_BAT_PEER",
                       normpath(joinpath(EXAMPLE, "..", "..", "..",
                                         "DecisionRulesExa.jl", "examples",
                                         "BatteryStorageOPF")))
            shared = ["battery_case.jl", "battery_solution_schema.jl"]
            isfile(portfolio_manifest_path(EXAMPLE)) && push!(shared, "battery_portfolio.json")
            if isdir(peer)
                for f in shared
                    @test isfile(joinpath(peer, f))
                    @test bytes2hex(sha256(read(joinpath(EXAMPLE, f)))) ==
                          bytes2hex(sha256(read(joinpath(peer, f))))
                end
            else
                # No peer checkout: assert at least that this side has the files
                # the mirror is asserted against.
                for f in shared
                    @test isfile(joinpath(EXAMPLE, f))
                end
            end
        end

        # The frozen panel, when it is present beside this file.
        if isfile(portfolio_manifest_path(EXAMPLE))
            @testset "the frozen panel" begin
                m = verify_portfolio_manifest()
                @test length(m["cases"]) >= PORTFOLIO_MIN_CASES
                @test m["seed"] == PORTFOLIO_SEED
                @test Float64.(m["profile"]) == PORTFOLIO_PROFILE
                @test length(m["profile"]) == 24
                for c in m["cases"]
                    @test c["counts"]["bus"] > 100
                    @test length(c["regions"]["sizes"]) == PORTFOLIO_REGIONS
                    @test c["kappa_case"] ≈ PORTFOLIO_KAPPA_MARGIN * c["kappa_max"]
                end
            end
        end
    end
end
