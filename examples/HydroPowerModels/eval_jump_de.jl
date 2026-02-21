# eval_jump_de.jl
#
# Run the Bolivia hydro deterministic equivalent (ACPPowerModel, JuMP+Ipopt)
# with a constant policy (targets = TARGET_FRAC × max_volume) and a seeded
# inflow scenario.  Saves the reference results to:
#   • ./bolivia/jump_de_reference.jld2          (local copy)
#   • EXA_CASE_DIR/jump_de_reference.jld2       (ExaGPU repo, for eval_exa_de.jl)
#
# Usage (SLURM job, must have DecisionRules.jl packages loaded):
#   julia --project=../.. eval_jump_de.jl

using DecisionRules
using Random
using JLD2
using JuMP
import CUDA # if error run CUDA.set_runtime_version!(v"12.1.0")
# CUDA.set_runtime_version!(v"12.1.0")
using MadNLP 
using MadNLPGPU
using KernelAbstractions
using CUDSS_jll

const SCRIPT_DIR   = dirname(@__FILE__)
const CASE_DIR     = joinpath(SCRIPT_DIR, "bolivia")
const EXA_CASE_DIR = "/storage/home/hcoda1/9/arosemberg3/scratch/DecisionRulesExaGPU.jl/examples/HydroPowerModels/bolivia"

include(joinpath(SCRIPT_DIR, "load_hydropowermodels.jl"))

const NUM_STAGES  = 12
const SEED        = 42
const FORMULATION = "ACPPowerModel"
const TARGET_FRAC = 0.6      # constant target = TARGET_FRAC × max_volume

# ── Build subproblems ──────────────────────────────────────────────────────────

@info "Building HydroPowerModels ($FORMULATION, T=$NUM_STAGES)..."

sub, state_in, state_out, uncert, initial_state, max_volume = build_hydropowermodels(
    CASE_DIR, FORMULATION * ".mof.json";
    num_stages = NUM_STAGES,
)
nHyd = length(initial_state)
@info "  nHyd=$nHyd  num_stages=$NUM_STAGES"

# ── Build deterministic equivalent ────────────────────────────────────────────

@info "Building deterministic equivalent..."
Det_model = Model(MadNLP.Optimizer)      # for sparse problems

set_optimizer_attribute(Det_model, "array_type", CUDA.CuArray)
set_optimizer_attribute(Det_model, "linear_solver", MadNLPGPU.CUDSSSolver)
set_optimizer_attribute(Det_model, "print_level", MadNLP.ERROR)
set_optimizer_attribute(Det_model, "barrier", MadNLP.LOQOUpdate())

Det_model, uncert_de = DecisionRules.deterministic_equivalent!(
    Det_model, sub, state_in, state_out, Float64.(initial_state), uncert,
)

# ── Fix scenario (seeded random) ──────────────────────────────────────────────

Random.seed!(SEED)
base_sample  = DecisionRules.sample(uncert_de)
base_values  = [[u[2] for u in stage_u] for stage_u in base_sample]

# Vector{Vector{Tuple{VariableRef, Float64}}} — one scalar per uncertainty/stage
uncertainties_de = [
    [(stage_u[i][1], base_values[t][i]) for i in eachindex(stage_u)]
    for (t, stage_u) in enumerate(uncert_de)
]

# Flat inflow vector (stage-major): index (t-1)*nHyd + i → inflow[t, hydro i]
inflows_flat = Float64.([base_values[t][i] for t in 1:NUM_STAGES for i in 1:nHyd])

# ── Constant policy ────────────────────────────────────────────────────────────
# simulate_states expects decision_rule(vcat(uncertainty_t, prev_state)) → next_state
# Our constant policy ignores both and always returns TARGET_FRAC × max_volume.

const_target = Float64.(max_volume) .* TARGET_FRAC
const_policy = _ -> const_target

# Returns T+1 elements: [initial_state, const_target, ..., const_target]
# simulate_multistage uses states[1] for x0 and states[t+1] as target for stage t.
states_policy = DecisionRules.simulate_states(
    Float64.(initial_state), uncertainties_de, const_policy,
)

# Flat target vector (stage-major, same layout as inflows_flat)
targets_flat = Float64.([const_target[i] for t in 1:NUM_STAGES for i in 1:nHyd])

# ── Solve ──────────────────────────────────────────────────────────────────────

@info "Solving DE (seed=$SEED, constant targets @ $(Int(TARGET_FRAC*100))% max_vol)..."
obj = DecisionRules.simulate_multistage(
    Det_model, state_in, state_out, uncertainties_de, states_policy,
)
@info "  Objective: $(round(obj; digits=4))"

# ── Extract reservoir trajectory (nHyd × (T+1)) ───────────────────────────────
# After deterministic_equivalent!, state_out[t][i][2] is the DE VariableRef for
# the reservoir level at stage t, hydro i.  value() returns the solved optimum.

reservoir = zeros(Float64, nHyd, NUM_STAGES + 1)
reservoir[:, 1] = Float64.(initial_state)
for t in 1:NUM_STAGES
    reservoir[:, t+1] = Float64.([value(pair[2]) for pair in state_out[t]])
end

# ── Save ───────────────────────────────────────────────────────────────────────

function save_reference(outfile)
    jldsave(outfile;
        objective     = obj,
        reservoir     = reservoir,          # nHyd × (T+1), rows=units, cols=stages 0..T
        initial_state = Float64.(initial_state),
        inflows_flat  = inflows_flat,       # length T*nHyd, stage-major
        targets_flat  = targets_flat,       # length T*nHyd, stage-major
        max_volume    = Float64.(max_volume),
        num_stages    = NUM_STAGES,
        nHyd          = nHyd,
        seed          = SEED,
        target_frac   = TARGET_FRAC,
        formulation   = FORMULATION,
    )
    @info "Saved reference to: $outfile"
end

save_reference(joinpath(CASE_DIR, "jump_de_reference.jld2"))

if isdir(EXA_CASE_DIR)
    save_reference(joinpath(EXA_CASE_DIR, "jump_de_reference.jld2"))
else
    @warn "ExaGPU bolivia dir not found: $EXA_CASE_DIR\n" *
          "Copy bolivia/jump_de_reference.jld2 there manually before running eval_exa_de.jl"
end

# ── Print summary ──────────────────────────────────────────────────────────────

println("\nObjective: ", round(obj; digits=4))
println("\nReservoir trajectory (end-of-stage, by hydro unit):")
println("  Stage:  ", join(lpad.(0:NUM_STAGES, 9)))
for r in 1:nHyd
    vals = round.(reservoir[r, :]; digits=1)
    println("  Hydro $r: ", join(lpad.(vals, 9)))
end
