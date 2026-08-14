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
        # in the diagnostics. Scanning for trigonometry and for the W-space
        # lifted variables catches a reimplementation of either formulation; the
        # case builder and the residual checker are out of scope because neither
        # builds an optimization model.
        for f in ("battery_powermodels.jl", "battery_sddp.jl", "battery_diagnostics.jl")
            src = read(joinpath(EXAMPLE, f), String)
            code = join([l for l in split(src, '\n') if !startswith(strip(l), "#")], '\n')
            for pat in ("cos(", "sin(", "atan(", "tan(")
                @test !occursin(pat, code)
            end
            for pat in ("var(pm)[:w]", "var(pm)[:wr]", "var(pm)[:wi]")
                @test !occursin(pat, code)
            end
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
            @test r.transition < 1e-9
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
            @test r.transition < 1e-9
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
        assert_graph_provenance(trained.backward, PowerModels.SOCWRConicPowerModel)
        assert_graph_provenance(trained.forward, PowerModels.ACPPowerModel)
        # Battery energy is a genuine SDDP state.
        node = trained.forward.nodes[1]
        @test length(node.states) == length(case.batteries)
        # Stock cuts were created.
        ncuts = sum(length(n.bellman_function.global_theta.cuts) for (_, n) in trained.backward.nodes)
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
end
