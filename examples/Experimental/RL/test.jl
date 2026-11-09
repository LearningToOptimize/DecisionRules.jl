import QuickPOMDPs: QuickPOMDP
import POMDPTools: ImplicitDistribution
import Distributions: Normal, Uniform
using POMDPs
using Flux
using Crux
import Crux: state_space
using POMDPs
import POMDPTools: FunctionPolicy
using Random
using Distributions

mdp = QuickPOMDP(;
    actions=[-1.0, 0.0, 1.0],
    obstype=Array{Float64,1},
    discount=0.95,
    transition=function (s, a)
        ImplicitDistribution() do rng
            x, v = s
            vp = v + a*0.001 + cos(3*x)*-0.0025 + 0.0002*randn(rng)
            vp = clamp(vp, -0.07, 0.07)
            xp = x + vp
            return (xp, vp)
        end
    end,
    observation=(a, sp) -> MvNormal([sp[1]], [0.15][:, :]),
    reward=function (s, a, sp)
        if sp[1] > 0.5
            return 100.0
        else
            return -1.0
        end
    end,
    initialstate=ImplicitDistribution(rng -> (-0.2*rand(rng), 0.0)),
    isterminal=s -> s[1] > 0.5,
    initialobs=(s) -> MvNormal([s[1]], [0.15][:, :]),
)

function Crux.state_space(mdp::QuickPOMDP; μ=0.0f0, σ=1.0f0)
    s = rand(initialstate(mdp))
    o = rand(initialobs(mdp, s))

    return state_space(o; μ=μ, σ=σ)
end

as = [actions(mdp)...]
amin = minimum(as)
amax = maximum(as)

rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))

S = state_space(mdp)

function SG()
    return SquashedGaussianPolicy(
        ContinuousNetwork(Chain(Dense(1, 64, relu), Dense(64, 64, relu), Dense(64, 1))),
        zeros(Float32, 1),
        2.0f0,
    )
end
𝒮_reinforce = REINFORCE(; π=SG(), S=S, N=100000, ΔN=2048, a_opt=(batch_size=512,))
@time π_reinforce = solve(𝒮_reinforce, mdp)
