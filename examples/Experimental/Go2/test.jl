using RigidBodyDynamics, MeshCat, MeshCatMechanisms
using Random, StaticArrays, Rotations, LinearAlgebra
using ForwardDiff; import ForwardDiff as FD   # alias for brevity

const Δt   = 0.005                # 200 Hz simulation
const N    = 300                  # 1.5 s horizon
const nd   = 12                   # DoFs (Go2 has 3×4 legs)


URDF_PATH = joinpath("urdf","go2.urdf")  # <-- change me
isfile(URDF_PATH) || error("URDF file not found: $URDF_PATH")
mech = parse_urdf(URDF_PATH)   
root = RigidBodyDynamics.root_body(mech)  # helper to fetch the root body
state       = MechanismState(mech)         # positions+velocities
feet      = map(n -> findbody(mech, n),
                ["FL_foot","FR_foot","RL_foot","RR_foot"]
)  # foot bodies

function rollout(u::AbstractVector)
    X = zeros(nd*2, N+1)            # [q; q̇] trajectory
    τ = reshape(u, nd, :)           # piecewise-constant torques
    for k in 1:N
        # set current state
        set_configuration!(state,  X[1:nd,k])
        set_velocity!(state,       X[nd+1:end,k])

        # forward dynamics: q̈ = M⁻¹(τ - C(q, q̇))  ----------------
        qdd = FDynamics(state, τ[:,k])   # alias to RBD.forward_dynamics!
        X[1:nd,   k+1] = X[1:nd,k]   + Δt*X[nd+1:end,k]
        X[nd+1:end,k+1] = X[nd+1:end,k] + Δt*qdd
    end
    return X
end

hip_body = findbody(mech, "trunk")          # Go2 torso link name
apex_k   = 180                              # ≈0.9 s

function cost(u)
    X = rollout(u)
    q_apex = X[1:nd, apex_k]
    com_z  = center_of_mass(mech, q_apex)[3]  # height of CoM

    J  = -com_z                               # higher ⇒ lower cost
    J += 1e-4*sum(abs2, u)                    # torque regularizer
    return J
end

nx = nd*2
nu = nd*(N)               # control vector length

u   = randn(nu) * 0.1      # random seed torques
lr  = 2e-2                 # learning rate
for iter in 1:350
    g  = FD.gradient(cost, u)     # ForwardDiff dual propagation
    u -= lr .* g                  # vanilla GD step
    if iter % 25 == 0
        @info "iter $iter  cost=$(cost(u))  ‖∇‖=$(norm(g))"
    end
end
X_opt = rollout(u)

vis = MechanismVisualizer(mech, URDFVisuals(urdf_path))
open(vis)                         # launches WebGL viewer

fps = Int(1/Δt)
setanimation!(vis, X_opt[1:nd,:], fps = fps)  # live replay