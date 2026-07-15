# sddp_demand_noise.jl
#
# Stochastic demand for the SDDP baselines (run_sddp.jl / run_sddp_inconsistent.jl).
#
# Demand model: an i.i.d. per-stage MULTIPLICATIVE factor ξ_t on the active
# demand of every bus, with three equiprobable atoms
#
#     ξ_t ∈ {1 − s, 1, 1 + s},   P(ξ_t = ·) = 1/3,
#
# INDEPENDENT of the inflow noise. The spread s is read from
# `<CASE_DIR>/demand_scenarios.csv` (single line `s,<value>`, e.g. `s,0.10`);
# when that file is ABSENT the demand stays deterministic and the SDDP model is
# bit-identical to the historical one (the override below then reproduces the
# package `rainfall_noises` exactly).
#
# Mechanism: HydroPowerModels.jl builds each stage subproblem and calls
# `rainfall_noises(sp, data, params, t)` to register the SDDP noise
# (src/build_model.jl line 147). This file REPLACES that method (script-local
# method override — the package source is never edited) with one that
#
#   1. at registration time (outside the `SDDP.parameterize` do-block) adds one
#      free JuMP variable dξ_i per bus i and inserts it into the ACTIVE-power
#      KCL constraint `lam_kcl_r` with coefficient +1, then fixes dξ_i = 0;
#   2. extends the SDDP noise support from the K inflow atoms to the K × 3
#      product atoms (ω, ℓ) with probability p_inflow(ω)/3 (inflow ⫫ demand);
#   3. in the do-block fixes the inflows for ω exactly as the original method
#      and fixes dξ_i = pd_i^t · (ξ_ℓ − 1), where pd_i^t is the stage-t base
#      active demand at bus i.
#
# Sign convention (verified against PowerModels src/form/acp.jl and the WR
# forms in src/form/shared.jl): the stored KCL constraint is
#
#     Σ p_arcs == Σ pg − Σ pd − Σ gs·vm²
#
# whose JuMP-normalized form is  Σ p_arcs − Σ pg + gs·vm² == −Σ pd. The
# package's own `constraint_mod_deficit` inserts the load-serving `deficit ≥ 0`
# variable with coefficient −1 (i.e. it acts like extra GENERATION), so a
# demand INCREASE must enter with the OPPOSITE sign: coefficient +1 moves the
# effective demand to pd + dξ. Both the SOCWRConic (backward) and ACP (forward)
# formulations store `lam_kcl_r` with this same convention, so one override
# serves both policy graphs.
#
# NOTE: only the ACTIVE demand is perturbed (matching the deviation variable in
# the real-power KCL). Reactive demand qd stays at its base value — the same
# convention as the ExaModels/TS-DDR engine, where ξ_t multiplies only the
# active-demand parameter.
#
# REQUIREMENTS: `using HydroPowerModels, JuMP, PowerModels, SDDP` and
# `const CASE_DIR` must precede `include`-ing this file; the include must
# precede the `hydro_thermal_operation` call (method overrides only affect
# models built afterwards).

# Stable seed shared with DecisionRulesExa.jl's `hydro_power_data.jl`.
# Paired-evaluation column c uses StableRNG(DEMAND_NOISE_SEED + c), making its
# demand path independent of evaluation sharding and traversal order.
const DEMAND_NOISE_SEED = 20260714

"""
    load_demand_spread(path::AbstractString) -> Union{Float64, Nothing}

Read the demand-noise spread `s` from a `demand_scenarios.csv` file.

The file holds a single data line `s,<value>` (e.g. `s,0.10`) defining the
three-atom multiplicative demand distribution

```math
\\xi_t \\in \\{1 - s,\\; 1,\\; 1 + s\\}, \\qquad P = \\tfrac{1}{3} \\text{ each}.
```

# Arguments
- `path::AbstractString`: path to `demand_scenarios.csv`.

# Returns
- `Float64` spread `s ∈ [0, 1)` when the file exists.
- `nothing` when the file does not exist (deterministic demand).
"""
function load_demand_spread(path::AbstractString)
    # Missing file ⇒ deterministic demand (backwards-compatible default).
    isfile(path) || return nothing
    # Exactly one non-empty line carries the single `s,<value>` record.
    lines = [strip(line) for line in eachline(path) if !isempty(strip(line))]
    length(lines) == 1 || error(
        "demand_scenarios.csv must contain exactly one non-empty line `s,<value>`; " *
        "found $(length(lines))",
    )
    line = only(lines)
    # Split into the key token and the numeric value.
    parts = split(line, ',')
    # Enforce the exact two-field `s,<value>` format shared by both engines.
    length(parts) == 2 && strip(parts[1]) == "s" ||
        error("demand_scenarios.csv must contain a single line `s,<value>`; got `$line`")
    # Parse the spread value.
    s = parse(Float64, strip(parts[2]))
    # A spread ≥ 1 would make the low atom non-positive demand; forbid it.
    0.0 <= s < 1.0 || error("demand spread must satisfy 0 ≤ s < 1; got $s")
    return s
end

"""Return the three equiprobable demand factors `{1-s, 1, 1+s}`."""
demand_noise_atoms(spread::Real) = begin
    0.0 <= spread < 1.0 || error("demand spread must satisfy 0 ≤ s < 1; got $spread")
    [1.0 - Float64(spread), 1.0, 1.0 + Float64(spread)]
end

"""
    protocol_demand_atom_indices(T, column; seed=DEMAND_NOISE_SEED)

Return the demand-atom IDs for paired-protocol column `column`. `StableRNGs`
must be loaded by the calling evaluation script. Sampling atom IDs (rather
than floating-point values) lets SDDP.Historical replay the exact `(inflow,
demand)` product atom registered by `rainfall_noises`.
"""
function protocol_demand_atom_indices(T::Integer, column::Integer;
                                      seed::Integer=DEMAND_NOISE_SEED)
    T >= 0 || throw(ArgumentError("T must be nonnegative; got $T"))
    column >= 1 || throw(ArgumentError("column must be positive; got $column"))
    return rand(StableRNG(seed + column), 1:3, T)
end

"""Return the paired demand-factor path for protocol column `column`."""
function protocol_demand_factors(spread::Real, T::Integer, column::Integer)
    atoms = demand_noise_atoms(spread)
    return atoms[protocol_demand_atom_indices(T, column)]
end

# Spread s from the case directory (nothing ⇒ deterministic demand).
const DEMAND_SPREAD = load_demand_spread(joinpath(CASE_DIR, "demand_scenarios.csv"))
# The three equiprobable multiplicative demand atoms {1−s, 1, 1+s}.
const DEMAND_ATOMS  = DEMAND_SPREAD === nothing ? nothing :
                      demand_noise_atoms(DEMAND_SPREAD)
# Filename tag so cuts trained WITH demand noise are never mixed with cuts
# trained without it (cuts from a different stochastic program are not valid
# lower bounds for this one).
const DEMAND_TAG    = DEMAND_SPREAD === nothing ? "" : "-dnoise$(DEMAND_SPREAD)"
DEMAND_SPREAD === nothing ||
    @info "Stochastic demand ACTIVE" DEMAND_SPREAD DEMAND_ATOMS

"""
    _sddp_fix_stage_inflows!(sp, data::Dict, t::Int, ω::Int) -> Nothing

Reproduce the body of the original `HydroPowerModels.rainfall_noises` do-block
(src/constraint.jl lines 13-31) for inflow atom `ω` at (cyclic) stage row `t`:
fill missing variable primal starts with `sp.ext[:lower_bound]`, then fix

```math
\\mathrm{inflow}_i = I_i[t, \\omega], \\qquad i = 1, \\ldots, n_{\\mathrm{Hyd}}.
```

# Arguments
- `sp`: stage JuMP subproblem.
- `data::Dict`: the stage's `alldata[min(t, end)]` dictionary.
- `t::Int`: cyclic stage index (the caller already applied `cidx`).
- `ω::Int`: inflow scenario atom.
"""
function _sddp_fix_stage_inflows!(sp, data::Dict, t::Int, ω::Int)
    # Variables still lacking a primal start (verbatim original logic).
    nostart = findall(
        x -> isnothing(x),
        JuMP.MOI.get.(sp, JuMP.MOI.VariablePrimalStart(), JuMP.all_variables(sp)),
    )
    # Give each of them the stage lower bound as a start (verbatim original).
    for theta in nostart
        JuMP.MOI.set(
            sp,
            JuMP.MOI.VariablePrimalStart(),
            JuMP.all_variables(sp)[theta],
            sp.ext[:lower_bound],
        )
    end
    # Fix every reservoir's inflow variable to the atom's historical value.
    for i in 1:data["hydro"]["nHyd"]
        JuMP.fix(
            sp[:inflow][i],
            data["hydro"]["Hydrogenerators"][i]["inflow"][t, ω];
            force=true,
        )
    end
    return nothing
end

"""
    HydroPowerModels.rainfall_noises(sp, data::Dict, params::Dict, t::Int)

Script-local REPLACEMENT of the package method (same signature ⇒ overwrites
it): registers the SDDP stage noise. Without `demand_scenarios.csv` it is
functionally identical to the original (inflow-only atoms). With it, the noise
support becomes the product set

```math
(\\omega, \\ell) \\in \\{1..K\\} \\times \\{1,2,3\\}, \\qquad
P(\\omega, \\ell) = \\frac{p_{\\mathrm{inflow}}(\\omega)}{3},
```

and each realization additionally fixes the per-bus demand deviation

```math
d\\xi_i = pd_i^t\\,(\\xi_\\ell - 1), \\qquad \\xi_\\ell \\in \\{1-s, 1, 1+s\\},
```

which the +1 KCL coefficient turns into effective demand `pd_i^t · ξ_ℓ`.
See the file-top comment for the sign-convention proof.
"""
function HydroPowerModels.rainfall_noises(sp, data::Dict, params::Dict, t::Int)
    # Number of joint inflow atoms K (columns of the probability matrix).
    K = size(data["hydro"]["scenario_probabilities"], 2)
    # Stage-row inflow probabilities (cidx is idempotent here — the caller
    # already passed a cyclic t; kept verbatim from the original method).
    probs_inflow = data["hydro"]["scenario_probabilities"][
        HydroPowerModels.cidx(t, data["hydro"]["size_inflow"][1]), :,
    ]

    if DEMAND_ATOMS === nothing
        # Deterministic demand: reproduce the original method exactly.
        SDDP.parameterize(sp, collect(1:K), probs_inflow) do ω
            _sddp_fix_stage_inflows!(sp, data, t, ω)
        end
        return nothing
    end

    # ── Registration-time setup (runs ONCE per subproblem) ────────────────────
    # The PowerModels model is stored before rainfall_noises is called
    # (build_model.jl line 128 vs 147), so the KCL references exist here.
    pm = sp.ext[:pm]
    # Per-bus KCL constraint references (same accessor the package's own
    # constraint_mod_deficit uses; bus ids are 1..nb for these cases).
    buses = PowerModels.sol(pm, 0, :bus)
    nb = length(buses)
    # Stage-t base active demand per bus: sum the pd of the loads at each bus
    # (add_loads! guarantees every bus has a load entry; pd is per-unit, the
    # same unit the KCL constraint uses).
    pd_bus = zeros(nb)
    for load in values(data["powersystem"]["load"])
        pd_bus[load["load_bus"]] += load["pd"]
    end
    # One free demand-deviation variable per bus (anonymous JuMP container).
    dξ = JuMP.@variable(sp, [i in 1:nb])
    # Register it in the model dictionary for recorders/diagnostics.
    sp[:demand_dev] = dξ
    for i in 1:nb
        # +1 in the real-power KCL ⇒ dξ adds LOAD (deficit uses −1 to serve
        # load; see the sign-convention proof in the file-top comment).
        JuMP.set_normalized_coefficient(buses[i][:lam_kcl_r], dξ[i], 1)
        # Start every atom from the deterministic base demand (dξ = 0).
        JuMP.fix(dξ[i], 0.0)
    end

    # ── Product noise support: (inflow ω, demand atom ℓ) ──────────────────────
    # Independence ⇒ P(ω, ℓ) = p_inflow(ω) · (1/3).
    atoms = [(ω, ℓ) for ℓ in 1:3 for ω in 1:K]
    probs = [probs_inflow[ω] / 3.0 for ℓ in 1:3 for ω in 1:K]
    SDDP.parameterize(sp, atoms, probs) do atom
        # Destructure the product atom.
        ω, ℓ = atom
        # Original inflow fixing (primal-start fill + inflow fix).
        _sddp_fix_stage_inflows!(sp, data, t, ω)
        # Realized multiplicative demand factor for this atom.
        ξ = DEMAND_ATOMS[ℓ]
        # Fix each bus's demand deviation: effective demand = pd_bus[i] · ξ.
        for i in 1:nb
            JuMP.fix(dξ[i], pd_bus[i] * (ξ - 1.0); force=true)
        end
    end
    return nothing
end
