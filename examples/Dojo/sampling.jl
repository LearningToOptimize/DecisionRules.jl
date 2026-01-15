# ### Setup
# PKG_SETUP
using Dojo
using DojoEnvironments
using Random
using LinearAlgebra

# ### Parameters
rng = MersenneTwister(1)
N = 2000 # number of steps per rollout; adjust as needed for your task/horizon
M = 60 # number of rollouts for learning; adjust as needed for convergence
paramcontainer = [[0.1; 0; 1; 0; -1.5]] # result: [[0.3604389380437305, 0.15854285262309512, 0.9575825661369068, -0.325769852046206, -1.4824537456751052]]
paramstorage = [[0.1; 0; 1; 0; -1.5]]
bias = zeros(5)
distance = 0.0
explore_factor = 0.1
distancestorage = zeros(M)

# ### Path Following Parameters
# Define target path as waypoints [x, y]
# User can modify this to any curved path
# line
# path_waypoints = [
#     [0.0, 0.0],
#     [0.3, 0.1],
#     [0.6, 0.2],
#     [0.9, 0.3],
#     [1.2, 0.4],
#     [1.5, 0.5]
# ]
# curve
# path_waypoints = [
#     [0.0, 0.00],
#     [0.3, 0.04],  # 0.4 * (0.3^2)
#     [0.6, 0.14],  # 0.4 * (0.6^2)
#     [0.9, 0.32],  # 0.4 * (0.9^2)
#     [1.2, 0.58],  # 0.4 * (1.2^2)
#     [1.5, 0.90]   # 0.4 * (1.5^2)
# ]
# circle
path_waypoints = [
    [0.0, 0.0],
    [0.3, 0.2],
    [0.6, 0.3],
    [0.9, 0.2],
    [1.2, 0.0],
    [1.5, -0.2]
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
    for k=1:N
        x = get_state(env)
        if x[3] < 0 || !all(isfinite.(x)) || abs(x[1]) > 1000 # upsidedown || failed || "exploding"
            println("  failed")
            return 1
        end
        
        u = controller!(x, k)
        step!(env, x, u; k, record)
    end
    
    return 0
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
    try
        res = rollout(env)
    catch
        res = 1
    end
    if res == 1
        println("  errored")
    end

    new_state = DojoEnvironments.get_state(env)
    
    # Primary metric: forward distance traveled (x-direction)
    distancenew = new_state[1] - x0
    
    # Secondary metric: how close to the path (deviation in y-direction)
    x_pos = new_state[1]
    y_pos = new_state[2]
    
    # Compute target y from waypoints: linear interpolation
    # Find where x_pos falls on the path
    target_y = 0.0
    if x_pos <= path_waypoints[1][1]
        target_y = path_waypoints[1][2]
    elseif x_pos >= path_waypoints[end][1]
        target_y = path_waypoints[end][2]
    else
        # Linear interpolation between waypoints
        for j in 1:(length(path_waypoints)-1)
            if path_waypoints[j][1] <= x_pos <= path_waypoints[j+1][1]
                # Interpolate
                x1, y1 = path_waypoints[j]
                x2, y2 = path_waypoints[j+1]
                t = (x_pos - x1) / (x2 - x1)
                target_y = y1 + t * (y2 - y1)
                break
            end
        end
    end
    
    y_deviation = abs(y_pos - target_y)
    
    # Success criterion: must make forward progress AND stay reasonably close to path
    # Only accept if y-deviation is not too large (tunable parameter)
    max_y_deviation = 0.3  # Maximum allowed deviation from path
    path_is_good = y_deviation < max_y_deviation
    
    # Modified success: forward progress + staying on path
    if res == 1 || distancenew <= distance || !path_is_good
        println("  unsuccessful")
        !all(isfinite.(new_state)) && println("  nans")
        if !path_is_good
            println("  off path (y-deviation: $(round(y_deviation, digits=3)))")
        end
        paramcontainer[1] = paramstorage[end]
        bias = zeros(5)
        explore_factor *= 0.9
    else
        println("  successful")
        distance = distancenew
        push!(paramstorage,paramcontainer[1])
        bias = paramstorage[end]-paramstorage[end-1]
        explore_factor = 0.1
    end

    println("  distance: $(round(distancenew, digits=3)) | y-target: $(round(target_y, digits=3)) | y-actual: $(round(y_pos, digits=3)) | y-dev: $(round(y_deviation, digits=3))")
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