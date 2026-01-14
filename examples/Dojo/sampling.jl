# ### Setup
# PKG_SETUP
using Dojo
using DojoEnvironments
using Random
using LinearAlgebra

# ### Parameters
rng = MersenneTwister(1)
N = 2000 # number of steps per rollout; adjust as needed for your task/horizon
M = 20 # number of rollouts for learning; adjust as needed for convergence
paramcontainer = [[0.1; 0; 1; 0; -1.5]] # result: [[0.3604389380437305, 0.15854285262309512, 0.9575825661369068, -0.325769852046206, -1.4824537456751052]]
paramstorage = [[0.1; 0; 1; 0; -1.5]]
bias = zeros(5)
distance = 0.0
explore_factor = 0.1
distancestorage = zeros(M)

# ### Path Following Parameters
# Define target path as waypoints [x, y]
# User can modify this to any curved path
path_waypoints = [
    [0.0, 0.0],
    [0.3, 0.1],
    [0.6, 0.2],
    [0.9, 0.3],
    [1.2, 0.4],
    [1.5, 0.5]
]
# using Plots
# plot(getindex.(path_waypoints, 1), getindex.(path_waypoints, 2), marker=:circle, label="Waypoints", xlabel="X", ylabel="Y", title="Target Path")
current_waypoint_idx = 1
waypoint_tolerance = 0.15  # Distance threshold to consider waypoint reached

# ### Environment
env = get_environment(:quadruped_sampling; horizon=N, timestep=0.001, joint_limits=Dict(), gravity=-9.81, contact_body=false)

# ### Controller
legmovement(k,a,b,c,offset) = a*cos(k*b*0.01*2*pi+offset)+c
Kp = [100;80;60]
Kd = [5;4;3]

function controller!(x, k)
    angle21 = legmovement(k,paramcontainer[1][2],paramcontainer[1][1],paramcontainer[1][3],0)
    angle22 = legmovement(k,paramcontainer[1][2],paramcontainer[1][1],paramcontainer[1][3],pi)
    angle31 = legmovement(k,paramcontainer[1][4],paramcontainer[1][1],paramcontainer[1][5],-pi/2)
    angle32 = legmovement(k,paramcontainer[1][4],paramcontainer[1][1],paramcontainer[1][5],pi/2)

    u = zeros(12)

    for i=1:4
         θ1 = x[12+(i-1)*6+1]
        dθ1 = x[12+(i-1)*6+2]
         θ2 = x[12+(i-1)*6+3]
        dθ2 = x[12+(i-1)*6+4]
         θ3 = x[12+(i-1)*6+5]
        dθ3 = x[12+(i-1)*6+6]

        if i == 1 || i == 4
            u[(i-1)*3+1] = Kp[1]*(0-θ1) + Kd[1]*(0-dθ1)
            u[(i-1)*3+2] = Kp[2]*(angle21-θ2) + Kd[2]*(0-dθ2)
            u[(i-1)*3+3] = Kp[3]*(angle31-θ3) + Kd[3]*(0-dθ3)
        else
            u[(i-1)*3+1] = Kp[1]*(0-θ1) + Kd[1]*(0-dθ1)
            u[(i-1)*3+2] = Kp[2]*(angle22-θ2) + Kd[2]*(0-dθ2)
            u[(i-1)*3+3] = Kp[3]*(angle32-θ3) + Kd[3]*(0-dθ3)
        end
    end

    return u
end

# ### Reset and rollout functions
function reset_state!(env)
    initialize!(env, :quadruped; body_position=[0;0;-0.43], hip_angle=0, thigh_angle=paramcontainer[1][3], calf_angle=paramcontainer[1][5]) 

    calf_state = get_body(env.mechanism, :FR_calf).state
    position = get_sdf(get_contact(env.mechanism, :FR_calf_contact), Dojo.current_position(calf_state), Dojo.current_orientation(calf_state))

    initialize!(env, :quadruped; body_position=[0;0;-position-0.43], hip_angle=0, thigh_angle=paramcontainer[1][3], calf_angle=paramcontainer[1][5]) 
    
    # Reset waypoint tracking
    global current_waypoint_idx = 1
end

function get_next_waypoint(x_pos, y_pos)
    # Find next waypoint to target based on current position
    global current_waypoint_idx
    
    if current_waypoint_idx >= length(path_waypoints)
        return path_waypoints[end]
    end
    
    current_wp = path_waypoints[current_waypoint_idx]
    dist = sqrt((x_pos - current_wp[1])^2 + (y_pos - current_wp[2])^2)
    
    # If close enough to current waypoint, move to next
    if dist < waypoint_tolerance
        current_waypoint_idx = min(current_waypoint_idx + 1, length(path_waypoints))
    end
    
    return path_waypoints[current_waypoint_idx]
end

function rollout(env; record=false)
    waypoints_reached = 0
    
    for k=1:N
        x = get_state(env)
        if x[3] < 0 || !all(isfinite.(x)) || abs(x[1]) > 1000 # upsidedown || failed || "exploding"
            println("  failed")
            return -1, waypoints_reached
        end
        
        # Get next waypoint to follow
        next_wp = get_next_waypoint(x[1], x[2])
        
        u = controller!(x, k)
        step!(env, x, u; k, record)
        
        # Track waypoints reached
        waypoints_reached = current_waypoint_idx - 1
    end
    
    return 0, waypoints_reached
end

# ### Learning routine
best_waypoints_reached = 0
path_following_score = 0.0

for i=1:M
    println("run: $i")

    if bias == zeros(5)
        paramcontainer[1] += randn!(rng, zeros(5))*explore_factor
    else
        paramcontainer[1] += randn!(rng, zeros(5))*0.002 + normalize(bias)*0.01
    end

    reset_state!(env)
    x0 = DojoEnvironments.get_state(env)[1]

    res = 0
    waypoints_reached = 0
    try
        res, waypoints_reached = rollout(env)
    catch
        res = -1
    end
    if res == -1
        println("  errored")
    end

    new_state = DojoEnvironments.get_state(env)
    
    # Compute path following score: combination of waypoints reached and distance to final waypoint
    final_waypoint = path_waypoints[end]
    dist_to_final = sqrt((new_state[1] - final_waypoint[1])^2 + (new_state[2] - final_waypoint[2])^2)
    path_score = waypoints_reached * 10.0 - dist_to_final  # Reward waypoints, penalize distance
    
    distancenew = new_state[1] - x0  # Keep original distance metric too

    if res == -1 || (waypoints_reached < best_waypoints_reached) || (waypoints_reached == best_waypoints_reached && path_score <= path_following_score)
        println("  unsuccessful")
        !all(isfinite.(new_state)) && println("  nans")
        paramcontainer[1] = paramstorage[end]
        bias = zeros(5)
        explore_factor *= 0.9
    else
        println("  successful")
        distance = distancenew
        best_waypoints_reached = waypoints_reached
        path_following_score = path_score
        push!(paramstorage,paramcontainer[1])
        bias = paramstorage[end]-paramstorage[end-1]
        explore_factor = 0.1
    end

    println("  distance: $distancenew | waypoints reached: $waypoints_reached/$(length(path_waypoints)) | score: $(round(path_score, digits=3))")
    distancestorage[i] = distance
end

# ### Controller for best parameter set
paramcontainer[1] = paramstorage[end]
function environment_controller!(environment, k)
    x = get_state(environment)

    u = controller!(x, k)
    set_input!(environment, u)
end

# ### Visualize learned behavior
reset_state!(env)
simulate!(env, environment_controller!; record=true)
vis = visualize(env)
render(vis)