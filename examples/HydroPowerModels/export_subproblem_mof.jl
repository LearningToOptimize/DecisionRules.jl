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
using MosekTools

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
num_stages = 96

@info "Exporting subproblem" case formulation num_stages

# ── Build the SDDP model ─────────────────────────────────────────────────────

alldata = HydroPowerModels.parse_folder(case_dir)

# Scale loads to match the training setup (Bolivia uses 60% load scaling)
for load in values(alldata[1]["powersystem"]["load"])
    load["qd"] = load["qd"] * 0.6
    load["pd"] = load["pd"] * 0.6
end

params = create_param(;
    stages=num_stages,
    model_constructor_grid=formulation,
    post_method=PowerModels.build_opf,
    optimizer=Mosek.Optimizer,
)

m = hydro_thermal_operation(alldata, params)

# ── Extract and clean one subproblem ──────────────────────────────────────────

# Run a minimal simulation to populate the subproblem models
results = HydroPowerModels.simulate(m, 2)

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
@info "Verification" variables = nvars constraints = ncons
