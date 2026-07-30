# Export a single-stage OPF subproblem from HydroPowerModels as a .mof.json file.
#
# The DecisionRules training pipeline (load_hydropowermodels.jl) reads pre-exported
# .mof.json files rather than depending on HydroPowerModels.jl at training time.
# This script builds the SDDP model, extracts one subproblem from the policy graph,
# removes the unnamed slack variable that HydroPowerModels adds, and writes the
# clean JuMP model to disk.
#
# The exported files already ship with this repository under:
#   bolivia/ACPPowerModel.mof.json
#   bolivia/SOCWRConicPowerModel.mof.json
#   bolivia/DCPPowerModel.mof.json
#   case3/ACPPowerModel.mof.json
#
# Re-run this script only if:
#   - The HydroPowerModels data (hydro.json, inflows.csv) has changed
#   - A new power-flow formulation is needed
#   - The HydroPowerModels.jl version changes the subproblem structure
#
# Requires: HydroPowerModels.jl, a compatible solver (Mosek, Gurobi, or MadNLP)
#
# Usage:
#   julia export_subproblem_mof.jl [case] [formulation]
#   julia export_subproblem_mof.jl bolivia ACPPowerModel
#   julia export_subproblem_mof.jl bolivia SOCWRConicPowerModel

using HydroPowerModels
using JuMP
# Solver only populates the subproblem models (simulate(m, 2)); the MathOptFormat
# export serializes the model STRUCTURE (variables, constraints, objective, names),
# which is solver-independent. Mosek is not available in the production depot, so
# we use the same solvers the production pipeline uses: Clarabel for the conic /
# DC formulations, MadNLP for the nonconvex polar-AC formulation.
using Clarabel
using MadNLP
include(joinpath(@__DIR__, "generate_canonical_case_artifacts.jl"))
using .HydroCanonicalCase

# ── Configuration ─────────────────────────────────────────────────────────────

case = length(ARGS) >= 1 ? ARGS[1] : "bolivia"
formulation_name = length(ARGS) >= 2 ? ARGS[2] : "ACPPowerModel"

# Map string names to PowerModels types
FORMULATIONS = Dict(
    "ACPPowerModel" => ACPPowerModel,
    "SOCWRConicPowerModel" => SOCWRConicPowerModel,
    "DCPPowerModel" => DCPPowerModel,
)

formulation = FORMULATIONS[formulation_name]

case_dir = joinpath(dirname(@__FILE__), case)
case == "bolivia" || error("Phase 2A exporter only regenerates the canonical Bolivia case")
manifest = read_manifest(case_dir)
verify_inputs(case_dir)
num_stages = Int(manifest["reporting_stages"])

@info "Exporting subproblem" case formulation num_stages

# ── Build the SDDP model ─────────────────────────────────────────────────────

alldata = HydroPowerModels.parse_folder(case_dir)

# MAIN is kept byte-exact on disk. The canonical historical convention is
# applied explicitly and symmetrically at model construction.
scale_main_loads!(alldata)
@info "Canonical demand scaling" pd_scale=ACTIVE_LOAD_FACTOR qd_scale=REACTIVE_LOAD_FACTOR

# Bake the stage duration into the exported MOF: HydroPowerModels'
# constraint_hydro_balance uses K = 0.0036·stage_hours as the inflow/outflow→volume
# coefficient. Read stage_hours from hydro.json (parsed into alldata); default 1
# (K = 0.0036) for cases predating the field. The DecisionRules loader
# (build_hydropowermodels) re-derives K from this MOF and asserts it equals
# 0.0036·stage_hours, so a stale-duration MOF is rejected fail-closed.
stage_hours = Int(get(alldata[1]["hydro"], "stage_hours", 1))
stage_hours == STAGE_HOURS || error("canonical stage_hours must be $STAGE_HOURS")
0.0036 * stage_hours == HYDRO_CONVERSION_K ||
    error("canonical hydro conversion K must be $HYDRO_CONVERSION_K")
@info "Water balance" stage_hours K_eff = 0.0036 * stage_hours

# Solver by formulation: MadNLP for nonconvex polar AC, Clarabel for conic/DC.
export_optimizer = formulation == ACPPowerModel ?
    (() -> MadNLP.Optimizer(; print_level = 0)) :
    (() -> Clarabel.Optimizer(; verbose = false))

params = create_param(;
    stages=num_stages,
    stage_hours=stage_hours,
    model_constructor_grid=formulation,
    post_method=PowerModels.build_opf,
    optimizer=export_optimizer,
)

m = hydro_thermal_operation(alldata, params)

# ── Extract and clean one subproblem ──────────────────────────────────────────

# Run a minimal simulation to populate the subproblem models. The MOF export only
# needs the subproblem STRUCTURE, which `hydro_thermal_operation` already built
# when it constructed the policy graph; the simulate solve is incidental. For the
# nonconvex polar-AC formulation a default-start MadNLP solve can fail at stressed
# points (vm=0 singularity) even though the model is well-formed, so tolerate a
# solver failure here rather than abort the export.
try
    global results = HydroPowerModels.simulate(m, 2)
catch err
    @warn "simulate() failed (structure still valid for export)" formulation exception=(err, catch_backtrace())
end

# The first stage subproblem is representative of all stages (same structure,
# different RHS values for inflows which load_hydropowermodels.jl sets via parameters)
model = m.forward_graph[1].subproblem

# HydroPowerModels adds an unnamed slack variable — remove it before export
unnamed_idx = findfirst(v -> name(v) == "", all_variables(model))
if !isnothing(unnamed_idx)
    delete(model, all_variables(model)[unnamed_idx])
end

# ── Write to disk ─────────────────────────────────────────────────────────────

outfile = joinpath(case_dir, formulation_name * ".mof.json")
JuMP.write_to_file(model, outfile)
@info "Exported subproblem to: $outfile"

# Verify the file is readable
test_model = JuMP.read_from_file(outfile; use_nlp_block=false)
nvars = length(all_variables(test_model))
ncons = length(all_constraints(test_model; include_variable_in_set_constraints=false))
deficit_vars = filter(v -> startswith(name(v), "deficit["), all_variables(test_model))
length(deficit_vars) == 28 || error("expected 28 operational active-deficit variables")
obj_fun = JuMP.objective_function(test_model)
all(
    JuMP.coefficient(obj_fun, v) == ACTIVE_DEFICIT_COST
    for v in deficit_vars
) || error("active-deficit coefficient must be $ACTIVE_DEFICIT_COST")
@info "Verification" variables = nvars constraints = ncons
