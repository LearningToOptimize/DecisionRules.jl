using DecisionRules
using Random
using JuMP
using DiffOpt
using Ipopt, HSL_jll

HydroPowerModels_dir = dirname(@__FILE__)
include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))

# ---- Configuration ----
case_name = "bolivia"           # case3, bolivia
formulation = "ACPPowerModel" # DCPPowerModel, SOCWRConicPowerModel, ACPPowerModel
num_stages = 96
window_size = 12

ipopt_factory = () -> DiffOpt.diff_optimizer(optimizer_with_attributes(Ipopt.Optimizer,
    "print_level" => 0,
    "hsllib" => HSL_jll.libhsl_path,
    "linear_solver" => "ma27",
))


diff_optimizer = ipopt_factory

det_optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

function build_problem()
    return build_hydropowermodels(
        joinpath(HydroPowerModels_dir, case_name),
        formulation * ".mof.json";
        num_stages=num_stages,
        optimizer=diff_optimizer,
    )
end

# ---- Stage-wise simulation ----
sub_s, state_in_s, state_out_s, uncert_s, initial_state, max_volume = build_problem()
num_uncertainties = length(uncert_s[1])

# Constant policy so targets do not depend on state or uncertainty.
const_target = Float32.(max_volume) * 0.6
policy(_) = const_target

Random.seed!(1234)
base_sample = DecisionRules.sample(uncert_s)
base_values = [[u[2] for u in stage_u] for stage_u in base_sample]
uncertainties_s = [[(stage_u[i][1], base_values[t][i]) for i in eachindex(stage_u)]
                   for (t, stage_u) in enumerate(uncert_s)]

obj_stage = DecisionRules.simulate_multistage(
    sub_s, state_in_s, state_out_s, initial_state, uncertainties_s, policy,
)

states_stage = Vector{Vector{Float64}}(undef, num_stages + 1)
states_stage[1] = initial_state
for t in 1:num_stages
    states_stage[t + 1] = [value(pair[2]) for pair in state_out_s[t]]
end

# ---- Deterministic equivalent ----
sub_d, state_in_d, state_out_d, uncert_d, initial_state_d, _ = build_problem()

Det_model = JuMP.Model(det_optimizer)
Det_model, uncert_d = DecisionRules.deterministic_equivalent!(
    Det_model,
    sub_d,
    state_in_d,
    state_out_d,
    Float64.(initial_state_d),
    uncert_d,
)
uncertainties_d = [[(stage_u[i][1], base_values[t][i]) for i in eachindex(stage_u)]
                   for (t, stage_u) in enumerate(uncert_d)]

states_policy = DecisionRules.simulate_states(initial_state_d, uncertainties_d, policy)
obj_det = DecisionRules.simulate_multistage(Det_model, state_in_d, state_out_d, uncertainties_d, states_policy)

states_det = Vector{Vector{Float64}}(undef, num_stages + 1)
states_det[1] = initial_state_d
for t in 1:num_stages
    states_det[t + 1] = [value(pair[2]) for pair in state_out_d[t]]
end

# ---- Multiple shooting ----
sub_w, state_in_w, state_out_w, uncert_w, initial_state_w, _ = build_problem()

windows = DecisionRules.setup_shooting_windows(
    sub_w,
    state_in_w,
    state_out_w,
    Float64.(initial_state_w),
    uncert_w;
    window_size=window_size,
    model_factory=diff_optimizer,
)

uncertainties_w = [[(stage_u[i][1], base_values[t][i]) for i in eachindex(stage_u)]
                   for (t, stage_u) in enumerate(uncert_w)]
uncertainties_vec = [[Float32(u[2]) for u in stage_u] for stage_u in uncertainties_w]

obj_shoot = DecisionRules.simulate_multiple_shooting(
    windows,
    policy,
    Float32.(initial_state_w),
    uncertainties_w,
    uncertainties_vec,
)

states_shoot = Vector{Vector{Float64}}()
push!(states_shoot, Float64.(initial_state_w))
current_state = Float64.(initial_state_w)
for window in windows
    window_range = window.stage_range
    window_uncertainties_vec = uncertainties_vec[window_range]
    targets = DecisionRules.predict_window_targets(
        policy,
        current_state,
        window_uncertainties_vec,
    )
    DecisionRules.set_window_uncertainties!(window, uncertainties_w)
    DecisionRules.solve_window(
        window.model,
        window.state_in_params,
        window.state_out_params,
        current_state,
        targets,
    )

    for local_t in 1:length(window_range)
        push!(states_shoot, [value(pair[2]) for pair in window.state_out_params[local_t]])
    end
    current_state = states_shoot[end]
end

# ---- Diagnostics ----
function max_state_diff(a, b)
    return maximum(abs.(vcat([abs.(a[t] .- b[t]) for t in eachindex(a)]...)))
end

@assert all(all(uncertainties_s[t][i][2] == base_values[t][i] for i in eachindex(uncertainties_s[t]))
            for t in 1:num_stages)

@assert all(all(uncertainties_w[t][i][2] == base_values[t][i] for i in eachindex(uncertainties_w[t]))
            for t in 1:num_stages)

@assert all(all(uncertainties_d[t][i][2] == base_values[t][i] for i in eachindex(uncertainties_d[t]))
            for t in 1:num_stages)


println("objective(stage): ", obj_stage)
println("objective(det):   ", obj_det)
println("objective(shoot): ", obj_shoot)

println("max |stage - det| state diff:   ", max_state_diff(states_stage, states_det))
println("max |stage - shoot| state diff: ", max_state_diff(states_stage, states_shoot))
