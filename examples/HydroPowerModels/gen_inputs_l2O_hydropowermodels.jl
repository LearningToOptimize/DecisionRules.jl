using DecisionRules
using L2O
using JuMP
using UUIDs

# Parameters
case_name = "case3" # bolivia, case3
formulation = "ACPPowerModel" # SOCWRConicPowerModel, DCPPowerModel, ACPPowerModel
num_stages = 48 # 96, 48
save_file = "$(case_name)-$(formulation)-h$(num_stages)"
formulation_file = formulation * ".mof.json"
batch_id = uuid1()

# Build MSP

HydroPowerModels_dir = dirname(@__FILE__)
include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))
case_dir = joinpath(HydroPowerModels_dir, case_name)

subproblems, state_params_in, state_params_out, uncertainty_samples, initial_state, max_volume = build_hydropowermodels(    
    joinpath(HydroPowerModels_dir, case_name), formulation_file; num_stages=num_stages
)

det_equivalent, uncertainty_samples = DecisionRules.deterministic_equivalent!(JuMP.Model(), subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples)

# Remove state imposing constraints

all_vars = all_variables(det_equivalent)
_deficit = all_vars[findall(x -> occursin("_deficit", name(x)), all_vars)]
state_params = vcat([[param[1] for param in params] for params in state_params_out]...)
state_vars = vcat([[param[2] for param in params] for params in state_params_out]...)
cons = JuMP.all_constraints(det_equivalent; include_variable_in_set_constraints=false)
for con in cons
    obj = JuMP.constraint_object(con)
    func = obj.func
    _vars = if !(func isa Array)
        keys(func.terms)
    else
        func
    end

    if all( [(_var in state_params) || (_var in _deficit) || (_var in state_vars) for _var in _vars])
        delete(det_equivalent, con)
    end
end

for _var in _deficit
    delete(det_equivalent, _var)
end

for _var in state_params
    delete(det_equivalent, _var)
end

# The problem iterator
num_samples = 10000
pairs = Dict{VariableRef,Vector{Float64}}()

# initial_state
for (i, hyd_in) in enumerate(state_params_in[1])
    pairs[hyd_in] = fill(initial_state[i], num_samples)
end

JuMP.write_to_file(det_equivalent, joinpath(case_dir, formulation) * "_det_equivalent.mof.json")

# inflow
recursive_merge(x::AbstractDict...) = merge(recursive_merge, x...)
inflow = vcat([[_var for _var in keys(dict)] for dict in uncertainty_samples]...)
pairs = merge(pairs, Dict(inflow .=> [Vector{Float64}(undef, num_samples) for _ in 1:length(inflow)]))
for s in 1:num_samples
    uncertainty_sample = sample(uncertainty_samples)
    uncertainty_sample = recursive_merge(uncertainty_sample...)
    for _var in inflow
        pairs[_var][s] = uncertainty_sample[_var]
    end
end

problem_iterator = ProblemIterator(pairs)

save(
    problem_iterator,
    joinpath(
        case_dir,
        case_name * "_" * formulation * "_input_" * string(batch_id),
    ),
    ArrowFile,
)
