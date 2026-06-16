################################################################
###################### Dataset Generation ######################
################################################################

using Distributed
using Random

##############
# Load Functions
##############

@everywhere l2o_path = dirname(@__DIR__)

@everywhere import Pkg

@everywhere Pkg.activate(l2o_path)

@everywhere Pkg.instantiate()

########## SCRIPT REQUIRED PACKAGES ##########

@everywhere using L2O
@everywhere using UUIDs
@everywhere import ParametricOptInterface as POI
@everywhere using JuMP
@everywhere using UUIDs
@everywhere using Arrow

## SOLVER PACKAGES ##

@everywhere using Ipopt

@everywhere filetype = ArrowFile

########## PARAMETERS ##########
model_file = joinpath(
    l2o_path, "examples/HydroPowerModels/case3/ACPPowerModel_det_equivalent.mof.json"
)
input_file = joinpath(
    l2o_path,
    "examples/HydroPowerModels/case3/case3_ACPPowerModel_input_4c4e8974-040e-11ef-1398-8195139913f4",
)
state_name = ["reservoir"; "out"]
save_path = joinpath(l2o_path, "examples/HydroPowerModels/case3/ACPPowerModel/output")
case_name = String(split(split(model_file, ".mof.")[1], "/")[end])
processed_output_files = [
    file for file in readdir(save_path; join=true) if occursin(case_name, file)
]
ids = if length(processed_output_files) == 0
    UUID[]
else
    vcat([Vector(Arrow.Table(file).id) for file in processed_output_files]...)
end
batch_size = 200

########## SOLVE ##########

problem_iterator_factory, num_batches = load(
    model_file, input_file, filetype; batch_size=batch_size, ignore_ids=ids
)

@sync @distributed for i in 1:num_batches
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    cached =
        () -> MOI.Bridges.full_bridge_optimizer(
            MOI.Utilities.CachingOptimizer(
                MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()), ipopt
            ),
            Float64,
        )
    POI_cached_optimizer() = POI.Optimizer(cached())
    batch_id = uuid1()
    problem_iterator = problem_iterator_factory(i)
    set_optimizer(problem_iterator.model, () -> POI_cached_optimizer())
    output_file = joinpath(save_path, "$(case_name)_output_$(batch_id)")
    all_vars = all_variables(problem_iterator.model)
    states = all_vars[findall(
        x -> all([occursin(part, name(x)) for part in state_name]), all_vars
    )]
    recorder = Recorder{filetype}(
        output_file;
        primal_variables=states,
        filterfn=(model) -> true,
        model=problem_iterator.model,
    )
    successfull_solves = solve_batch(problem_iterator, recorder)
    @info "Solved $(length(successfull_solves)) problems"
    L2O.compress_batch_arrow(
        save_path,
        case_name;
        keyword_all="output",
        batch_id=string(batch_id),
        keyword_any=[string(batch_id)],
    )
end
