# Focused tests of the renewable/storage variants (`renewable_variants.toml`,
# `renewable_variant.jl`).
#
# The properties guarded here are the ones whose failure is SILENT: a cost
# ordering that is not a cost ordering, a fleet resizing that quietly changes the
# fleet's size, a terminal requirement a case declares and a solver never sees,
# and a collapsed demand support that is not actually deterministic. A synthetic
# four-zone network keeps the file standalone; when a frozen case directory sits
# beside the example the round-trip check runs on it too.
#
#   julia --project=. test/renewable_variant.jl

using Test
using JuMP
using PowerModels
using TOML

const RV_EXAMPLE = dirname(@__DIR__)
@isdefined(read_battery_case) || include(joinpath(RV_EXAMPLE, "battery_case.jl"))
# `apply_stage_availability!` lives in the PowerModels layer, and the reactive
# assumption is checked against that function rather than against a restatement
# of it, so this file needs the layer loaded even though it builds no model.
@isdefined(apply_stage_availability!) || include(joinpath(RV_EXAMPLE, "battery_powermodels.jl"))
@isdefined(build_renewable_variant) || include(joinpath(RV_EXAMPLE, "renewable_variant.jl"))

"A generator dictionary with a degree-two cost, in the per-unit basis."
function rv_gen(index, bus; pmax, pmin = 0.0, qmax = 1.0, c2 = 0.0, c1 = 0.0, c0 = 0.0,
                status = 1)
    return Dict{String,Any}("index" => index, "gen_bus" => bus, "gen_status" => status,
        "pmax" => pmax, "pmin" => pmin, "qmax" => qmax, "qmin" => -qmax,
        "pg" => 0.0, "qg" => 0.0, "vg" => 1.0, "mbase" => 100.0,
        "model" => 2, "ncost" => 3, "cost" => [c2, c1, c0],
        "startup" => 0.0, "shutdown" => 0.0, "source_id" => Any["gen", index])
end

"""A four-zone synthetic network: zone 3 = L, 7 = W1, 5 = W2, 9 = S, 1 = R.

Bus 10 is the reference bus and carries two units that the threshold retires,
one of which must survive the reference-bus exception.
"""
function rv_network()
    buses = Dict{String,Any}()
    for (i, z) in [(10, 3), (11, 3), (20, 7), (21, 7), (30, 5), (40, 9), (41, 9), (50, 1)]
        buses[string(i)] = Dict{String,Any}("bus_i" => i, "index" => i, "zone" => z,
            "area" => 1, "bus_type" => i == 10 ? 3 : 1, "base_kv" => 138.0,
            "vm" => 1.0, "va" => 0.0, "vmin" => 0.9, "vmax" => 1.1)
    end
    gens = Dict{String,Any}()
    for g in [
        rv_gen(1, 10; pmax = 4.0, qmax = 2.0, c1 = 100.0),   # avc 1.0   retired but for the exception
        rv_gen(2, 10; pmax = 4.0, qmax = 2.0, c1 = 150.0),   # avc 1.5   retired
        rv_gen(3, 11; pmax = 2.0, c1 = 900.0),               # avc 9.0   kept
        rv_gen(4, 20; pmax = 3.0, c1 = 50.0),                # avc 0.5   retired
        rv_gen(5, 21; pmax = 1.0, c1 = 4000.0),              # avc 40.0  kept
        rv_gen(6, 30; pmax = 2.0, c1 = 120.0),               # avc 1.2   retired
        rv_gen(7, 40; pmax = 5.0, c1 = 60.0),                # avc 0.6   retired
        rv_gen(8, 41; pmax = 2.0, c2 = 1000.0, c1 = 0.0),    # c1 = 0 but avc 20.0: KEPT
        rv_gen(9, 50; pmax = 3.0, c1 = 3000.0),              # avc 30.0  kept
        rv_gen(99, 50; pmax = 9.9, c1 = 1.0, status = 0),    # out of service: never a candidate
    ]
        gens[string(g["index"])] = g
    end
    loads = Dict{String,Any}()
    for (j, (b, pd)) in enumerate([(11, 1.0), (21, 0.8), (30, 0.4), (41, 0.2), (50, 0.6)])
        loads[string(j)] = Dict{String,Any}("index" => j, "load_bus" => b,
            "pd" => pd, "qd" => 0.3 * pd, "status" => 1, "source_id" => Any["bus", b])
    end
    return Dict{String,Any}("baseMVA" => 100.0, "name" => "rv_synthetic",
        "per_unit" => true, "source_type" => "matpower", "source_version" => "2",
        "bus" => buses, "gen" => gens, "load" => loads,
        "branch" => Dict{String,Any}(), "shunt" => Dict{String,Any}(),
        "storage" => Dict{String,Any}(), "dcline" => Dict{String,Any}(),
        "switch" => Dict{String,Any}())
end

function rv_recipe()
    rec = renewable_variant_recipe(renewable_variants_path(RV_EXAMPLE))
    # The synthetic host has one zone per region and no zone 2/4/6/8/10/11.
    rec["regions"] = Dict{String,Any}(
        "L" => Dict{String,Any}("zones" => [3], "role" => "demand"),
        "W1" => Dict{String,Any}("zones" => [7], "role" => "overnight wind"),
        "W2" => Dict{String,Any}("zones" => [5], "role" => "evening wind"),
        "S" => Dict{String,Any}("zones" => [9], "role" => "solar"),
        "R" => Dict{String,Any}("zones" => [1], "role" => "rest"))
    rec["transition"]["keep_extra_gen_indices"] = Int[]
    return rec
end

function rv_fleet()
    # Two batteries per region so a regional factor has something to act on.
    specs = BatterySpec[]
    for (i, (bus, p, e)) in enumerate([(10, 0.4, 3.2), (11, 0.6, 4.8),    # L
                                       (20, 0.5, 4.0), (21, 0.5, 4.0),    # W1
                                       (30, 0.3, 2.4), (30, 0.1, 0.8),    # W2
                                       (40, 0.2, 1.6), (41, 0.2, 1.6),    # S
                                       (50, 0.6, 4.8), (50, 0.6, 4.8)])   # R
        push!(specs, BatterySpec(i, bus, 0.05e, e, 0.5e, p, p, 0.95, 0.95, 0.999, 5.0))
    end
    return specs
end

@testset "renewable/storage variants" begin

    @testset "the recipe is the only source of the parameters" begin
        rec = renewable_variant_recipe(renewable_variants_path(RV_EXAMPLE))
        @test rec["schema"] == RENEWABLE_VARIANT_SCHEMA
        @test haskey(rec, "regions") && haskey(rec, "transition") && haskey(rec, "renewables")
        # Every region given installed capacity must have an availability entry
        # for every template, or the case is built with a schedule nobody wrote.
        for r in keys(rec["renewables"]["installed_pu"]), tpl in rec["templates"]["names"]
            v = variant_availability(rec, r, tpl, 24)
            @test length(v) == 24
            @test all(0.0 .<= v .<= 1.0)
        end
        @test rec["templates"]["nominal"] in rec["templates"]["names"]
        # Solar is zero at night in BOTH templates, in the recipe itself.
        for tpl in rec["templates"]["names"]
            s = variant_availability(rec, "S", tpl, 24)
            @test all(s[t] == 0.0 for t in vcat(1:6, 18:24))
        end
    end

    @testset "average variable cost is not the linear coefficient" begin
        base = 100.0
        quadratic = rv_gen(1, 1; pmax = 2.0, c2 = 1000.0, c1 = 0.0)
        # avc = c2*pmax/base = 20; marginal at pmax = 2*c2*pmax/base = 40.
        @test gen_average_variable_cost(quadratic, base) ≈ 20.0
        @test gen_marginal_cost(quadratic, 2.0, base) ≈ 40.0
        # The defect this guards: c1 alone calls this unit free.
        @test gen_cost_coefficients(quadratic)[2] == 0.0
        # A purely linear unit: all three agree.
        linear = rv_gen(2, 1; pmax = 2.0, c1 = 500.0)
        @test gen_average_variable_cost(linear, base) ≈ 5.0
        @test gen_marginal_cost(linear, 2.0, base) ≈ 5.0
        @test gen_marginal_cost(linear, 0.1, base) ≈ 5.0
        # Padding: ncost 2 and ncost 1 are read without a caller branching.
        lin2 = Dict{String,Any}("cost" => [700.0, 3.0], "ncost" => 2, "pmax" => 1.0)
        @test gen_cost_coefficients(lin2) == (0.0, 700.0, 3.0)
        @test gen_average_variable_cost(lin2, base) ≈ 7.0
    end

    @testset "the transition retires what the rule names" begin
        net = rv_network(); rec = rv_recipe()
        retired = variant_retirements(net, rec)     # threshold 2.00 $/MWh
        @test retired == [2, 4, 6, 7]
        # gen 1 is at the reference bus and has the lowest avc there, so the
        # exception keeps it; gen 2 at the same bus is still retired.
        @test !(1 in retired)
        # gen 8 has c1 = 0 and avc 20: a linear-coefficient rule would retire it.
        @test !(8 in retired)
        @test gen_average_variable_cost(net["gen"]["8"], 100.0) ≈ 20.0
        # the out-of-service unit is never a candidate however cheap it is
        @test !(99 in retired)
        # and the answer does not depend on dictionary order
        shuffled = Dict{String,Any}(net)
        shuffled["gen"] = Dict{String,Any}(reverse(collect(net["gen"])))
        @test variant_retirements(shuffled, rec) == retired
    end

    @testset "the renewable build is proportional, curtailable and reactive-scaled" begin
        net = rv_network(); rec = rv_recipe()
        breg = variant_bus_region(net, rec)
        @test breg[10] == "L" && breg[20] == "W1" && breg[30] == "W2" && breg[40] == "S"
        rec["renewables"]["installed_pu"] = Dict{String,Any}("S" => 7.0, "W1" => 4.0)
        units = variant_renewable_units(net, rec, breg, "reliable", 24)
        @test sum(u["pmax"] for u in units) ≈ 11.0
        # S has two generator buses weighted 5 and 2 by original pmax.
        s_units = sort([u for u in units if u["rs_region"] == "S"]; by = u -> u["gen_bus"])
        @test [u["gen_bus"] for u in s_units] == [40, 41]
        @test s_units[1]["pmax"] ≈ 7.0 * 5 / 7 && s_units[2]["pmax"] ≈ 7.0 * 2 / 7
        for u in units
            @test u["pmin"] == 0.0                              # fully curtailable
            @test u["qmax"] ≈ rec["renewables"]["qmax_fraction"] * u["pmax"]
            @test u["qmin"] ≈ -u["qmax"]
            @test u["cost"] == [0.0, 0.0, 0.0]                  # including the constant term
            @test length(u[STAGE_AVAILABILITY_KEY]) == 24
            @test u["index"] > maximum(Int(g["index"]) for (_, g) in net["gen"])
        end
        # Availability scales P and Q together: that is what makes a night-time
        # solar unit supply no reactive power, and it is an ASSUMPTION, so it is
        # checked here against the code that implements it rather than assumed.
        probe = deepcopy(net)
        for u in units; probe["gen"][string(u["index"])] = deepcopy(u); end
        night = findfirst(==(0.0), variant_availability(rec, "S", "reliable", 24))
        night === nothing || begin
            apply_stage_availability!(probe, night)
            for u in units
                u["rs_region"] == "S" || continue
                g = probe["gen"][string(u["index"])]
                @test g["pmax"] == 0.0 && g["qmax"] == 0.0 && g["qmin"] == 0.0
            end
        end
    end

    @testset "a retired unit is deleted or becomes a condenser, as declared" begin
        net = rv_network(); rec = rv_recipe()
        retired = variant_retirements(net, rec)
        @test !isempty(retired)
        qmax_before = Dict(i => net["gen"][string(i)]["qmax"] for i in retired)

        # Condenser: no active capability, reactive untouched, no cost at all.
        cond = deepcopy(net)
        for i in retired
            g = cond["gen"][string(i)]
            g["pmin"] = 0.0; g["pmax"] = 0.0; g["cost"] = [0.0, 0.0, 0.0]
        end
        for i in retired
            g = cond["gen"][string(i)]
            @test g["pmax"] == 0.0 && g["pmin"] == 0.0
            @test g["qmax"] == qmax_before[i]          # the point of the conversion
            @test g["qmin"] == -qmax_before[i]
            @test gen_cost_coefficients(g) == (0.0, 0.0, 0.0)
            @test Int(get(g, "gen_status", 1)) == 1     # still in service, still in `ref`
        end
        # and the machine can no longer be the marginal unit at any output
        for i in retired
            @test gen_marginal_cost(cond["gen"][string(i)], 0.0, 100.0) == 0.0
        end
        # The two dispositions differ in exactly one thing: whether reactive
        # capability survives. Active capability is removed by both.
        kept_q = sum(qmax_before[i] for i in retired)
        @test kept_q > 0
    end

    @testset "the fleet is resized, not resized away" begin
        net = rv_network(); rec = rv_recipe()
        breg = variant_bus_region(net, rec)
        fleet0 = rv_fleet()
        rec["storage"]["region"] = Dict{String,Any}(
            "L"  => Dict{String,Any}("power_pu" => 1.5, "energy_puh" => 9.0),
            "W1" => Dict{String,Any}("power_pu" => 1.0, "energy_puh" => 8.0),
            "S"  => Dict{String,Any}("power_pu" => 0.8, "energy_puh" => 8.0),
            "W2" => Dict{String,Any}("power_pu" => 0.4, "energy_puh" => 3.2))
        rec["storage"]["balance"] = "R"
        fleet, report = variant_fleet(fleet0, breg, rec)

        @test length(fleet) == length(fleet0)
        @test [b.bus for b in fleet] == [b.bus for b in fleet0]
        @test [b.index for b in fleet] == [b.index for b in fleet0]
        # totals preserved to floating point, not to the decimals in the recipe
        @test sum(b.discharge_max for b in fleet) ≈ sum(b.discharge_max for b in fleet0) atol = 1e-12
        @test sum(b.energy_max for b in fleet) ≈ sum(b.energy_max for b in fleet0) atol = 1e-12
        @test sum(b.charge_max for b in fleet) ≈ sum(b.charge_max for b in fleet0) atol = 1e-12
        # and each region hits its target
        for (r, target) in rec["storage"]["region"]
            got_p = sum(b.discharge_max for b in fleet if breg[b.bus] == r)
            got_e = sum(b.energy_max for b in fleet if breg[b.bus] == r)
            @test got_p ≈ target["power_pu"] atol = 1e-12
            @test got_e ≈ target["energy_puh"] atol = 1e-12
        end
        @test report["balance_region"] == "R"
        # the reserve fraction survives the resizing
        for (a, b) in zip(fleet0, fleet)
            @test b.energy_min / b.energy_max ≈ a.energy_min / a.energy_max
        end
        # a recipe that also names the balancing region is rejected, not averaged
        bad = deepcopy(rec)
        bad["storage"]["region"]["R"] = Dict{String,Any}("power_pu" => 1.0, "energy_puh" => 1.0)
        @test_throws ErrorException variant_fleet(fleet0, breg, bad)
    end

    @testset "the boundary convention is declared, reachable and opt-in" begin
        rec = rv_recipe(); fleet0 = rv_fleet()
        fleet, report = apply_boundary_convention(fleet0, rec)
        f = rec["boundary"]["initial_fraction_of_usable"]
        for b in fleet
            @test b.energy_initial ≈ b.energy_min + f * (b.energy_max - b.energy_min)
            @test b.energy_min <= b.energy_initial <= b.energy_max
        end
        @test report["terminal_at_least_initial"]
        floors = report["terminal_energy_min"]
        @test length(floors) == length(fleet)
        for b in fleet
            @test floors[string(b.index)] ≈ b.energy_initial
        end
        # PHYSICAL REACHABILITY of the convention: holding the opening level for
        # one stage must be inside the battery's own reachable interval, or the
        # terminal requirement would be a constraint no dispatch can meet at the
        # declared initial state. Self-discharge alone pushes the level DOWN, so
        # this is a real check on efficiencies and alpha, not an identity.
        Δt = 1.0
        for b in fleet
            lo, hi = reachable_interval(b, b.energy_initial, Δt)
            @test lo <= b.energy_initial <= hi
            @test hi >= b.self_discharge * b.energy_initial   # charging can add
            # and the dispatch that holds it exists and is admissible
            p_ch, p_dis = dispatch_for_target(b, b.energy_initial, b.energy_initial, Δt)
            @test p_ch >= -1e-12 && p_dis >= -1e-12
            @test p_ch <= b.charge_max + 1e-12 && p_dis <= b.discharge_max + 1e-12
        end
        # opt-in: a recipe without [boundary] changes nothing
        plain_rec = deepcopy(rec); delete!(plain_rec, "boundary")
        untouched, empty_report = apply_boundary_convention(fleet0, plain_rec)
        @test isempty(empty_report)
        @test all(a.energy_initial == b.energy_initial for (a, b) in zip(fleet0, untouched))
    end

    @testset "the collapsed support is deterministic" begin
        loads = [1, 2, 3]
        prof = [0.8 1.0 0.9; 0.8 1.0 0.9; 0.8 1.0 0.9]
        atoms = [rand(3, 4) .+ 0.5 for _ in 1:3]
        probs = [fill(0.25, 4) for _ in 1:3]
        s = DemandSupport(1.0, 3, loads, prof, atoms, probs, 7,
                          Dict{String,Any}("method" => "whatever"))
        d = deterministic_support(s)
        @test d.horizon == 3 && d.load_ids == loads
        @test d.profile == prof                       # profile untouched
        @test d.protocol_seed == s.protocol_seed      # seed carried through
        for t in 1:3
            @test num_atoms(d, t) == 1
            @test atom_probabilities(d, t) == [1.0]
            # `demand_multipliers` is the profile TIMES the atom, which is what a
            # stage problem reads. A collapsed support must therefore return the
            # profile column exactly — not a vector of ones, which would mean the
            # profile had been thrown away along with the uncertainty.
            @test all(d.atoms[t] .== 1.0)
            @test demand_multipliers(d, t, 1) == prof[:, t]
        end
        @test d.source["method"] == "deterministic_profile"
        validate_support(d)
    end

    @testset "a case carries its terminal requirement across a write and a read" begin
        mktempdir() do dir
            net = rv_network()
            loads = sort!([Int(l["index"]) for (_, l) in net["load"]])
            n = length(loads)
            s = DemandSupport(1.0, 2, loads, repeat([0.9 1.0], n, 1),
                              [ones(n, 1), ones(n, 1)],
                              [[1.0], [1.0]], 11, Dict{String,Any}("method" => "t"))
            fleet = [BatterySpec(1, 11, 0.1, 2.0, 0.575, 0.25, 0.25, 0.95, 0.95, 0.999, 5.0)]
            floors = Dict{String,Any}("1" => 0.575)

            with_dir = joinpath(dir, "with")
            write_battery_case(with_dir; name = "t_with", network = net, batteries = fleet,
                recourse = RecourseCosts(1.0e4, 1.0e4), demand = s, source_version = "test",
                protocol_stages = 2, protocol_scenarios = 2,
                boundary = Dict{String,Any}("terminal_energy_min" => floors,
                                            "initial_fraction_of_usable" => 0.25))
            case = read_battery_case(with_dir)
            got = terminal_energy_floor(case)
            @test got !== nothing
            @test got[1] ≈ 0.575

            # and a case written WITHOUT the section declares no requirement,
            # which is what every case frozen before this field relies on.
            without_dir = joinpath(dir, "without")
            write_battery_case(without_dir; name = "t_without", network = net, batteries = fleet,
                recourse = RecourseCosts(1.0e4, 1.0e4), demand = s, source_version = "test",
                protocol_stages = 2, protocol_scenarios = 2)
            @test terminal_energy_floor(read_battery_case(without_dir)) === nothing
        end
    end
end
