# rl_algorithms.jl — four representative model-free RL algorithms, written directly
# in Flux 0.16 (explicit gradients) so they run on the same environment, the same
# networks and the same reward as each other:
#
#   REINFORCE  Monte-Carlo policy gradient with a learned value baseline
#   PPO        clipped surrogate, generalized advantage estimation
#   TD3        deterministic actor, twin critics, target smoothing, delayed actor
#   SAC        squashed-Gaussian actor, twin critics, automatic entropy tuning
#
# Action convention: every policy emits y ∈ [0,1]^n, which the environment maps
# into the reachable interval (hydro_rl_env.jl). On-policy methods sample a
# Gaussian pre-activation u and use y = σ(u); SAC uses y = (tanh(u)+1)/2 with the
# change-of-variables correction; TD3 emits y = σ(actor(s)) directly.
#
# Reward r_t = −c_t / REWARD_SCALE, discount GAMMA, identical for all four.

using Flux, Optimisers, Statistics, Random

const REWARD_SCALE = 1e3
const GAMMA = 0.995
const HIDDEN = (64, 64)

"Two-hidden-layer ReLU network `in → out` with linear output."
mlp(in, out) = Chain(Dense(in => HIDDEN[1], relu), Dense(HIDDEN[1] => HIDDEN[2], relu),
                     Dense(HIDDEN[2] => out))

"Diagonal Gaussian log-density of `u` (columns = samples)."
gauss_logp(u, μ, logσ) =
    sum(-0.5f0 .* ((u .- μ) ./ exp.(logσ)) .^ 2 .- logσ .- 0.9189385f0; dims = 1)

"Discounted returns-to-go of one episode's rewards."
function returns_to_go(r)
    G = similar(r); acc = 0f0
    for t in length(r):-1:1
        acc = r[t] + Float32(GAMMA) * acc; G[t] = acc
    end
    return G
end

"Polyak averaging of target parameters: θ' ← (1−τ)θ' + τθ."
soft_update!(target, src, τ) =
    Flux.fmap((a, b) -> (a isa AbstractArray ? (a .= (1 - τ) .* a .+ τ .* b) : a), target, src)

# ─────────────────────────────────────────────────────────────────────────────
# On-policy: REINFORCE and PPO share the Gaussian actor and the value critic
# ─────────────────────────────────────────────────────────────────────────────

"""
    OnPolicy

Gaussian actor `u ~ N(μ(s), diag σ²)` with state-independent log σ, value critic V(s).
`kind` is `:reinforce` or `:ppo`.
"""
mutable struct OnPolicy
    kind::Symbol
    actor::Chain
    logσ::Vector{Float32}
    critic::Chain
    opt_a::Any
    opt_c::Any
end

function OnPolicy(kind, dobs, dact; lr = 3f-4)
    actor = mlp(dobs, dact); critic = mlp(dobs, 1)
    logσ = fill(-0.5f0, dact)
    return OnPolicy(kind, actor, logσ, critic,
                    Optimisers.setup(Optimisers.Adam(lr), (actor = actor, logσ = logσ)),
                    Optimisers.setup(Optimisers.Adam(lr), critic))
end

"Exploration action: sample u, emit y = σ(u), store u and its log-probability."
function act_explore(p::OnPolicy, o, rng)
    μ = p.actor(o)
    u = μ .+ exp.(p.logσ) .* randn(rng, Float32, length(μ))
    return (sigmoid.(u), (u = u, logp = only(gauss_logp(u, μ, p.logσ))))
end
"Deterministic (evaluation) action: the mean."
act_greedy(p::OnPolicy, o) = (sigmoid.(p.actor(o)), nothing)

"""
    update!(p::OnPolicy, episodes) -> Dict

REINFORCE: advantage = return-to-go − V(s), loss −E[log π(u|s) Â].
PPO: GAE(λ=0.95) advantages, clipped ratio (ε = 0.2), 10 epochs of minibatches.
The critic regresses the returns-to-go in both.
"""
function update!(p::OnPolicy, episodes; epochs = 10, batch = 512, clip = 0.2f0, λ = 0.95f0)
    O = reduce(hcat, [t.obs for ep in episodes for t in ep])
    U = reduce(hcat, [t.extra.u for ep in episodes for t in ep])
    LP = Float32[t.extra.logp for ep in episodes for t in ep]
    G = reduce(vcat, [returns_to_go(Float32[-t.cost / REWARD_SCALE for t in ep]) for ep in episodes])
    V = vec(p.critic(O))
    if p.kind === :ppo
        # GAE over each episode with the current critic.
        A = Float32[]; off = 0
        for ep in episodes
            n = length(ep); r = Float32[-t.cost / REWARD_SCALE for t in ep]
            v = V[off+1:off+n]; adv = zeros(Float32, n); g = 0f0
            for t in n:-1:1
                vn = t < n ? v[t+1] : 0f0
                δ = r[t] + Float32(GAMMA) * vn - v[t]
                g = δ + Float32(GAMMA) * λ * g; adv[t] = g
            end
            append!(A, adv); off += n
        end
    else
        A = G .- V
    end
    A = (A .- mean(A)) ./ (std(A) + 1f-8)
    N = size(O, 2); nep = p.kind === :ppo ? epochs : 1; bs = p.kind === :ppo ? batch : N
    la = 0f0; lc = 0f0
    for _ in 1:nep, idx in Iterators.partition(randperm(N), bs)
        o, u, lp0, a, g = O[:, idx], U[:, idx], LP[idx], A[idx], G[idx]
        la, ga = Flux.withgradient((actor = p.actor, logσ = p.logσ)) do m
            lp = vec(gauss_logp(u, m.actor(o), m.logσ))
            if p.kind === :ppo
                ρ = exp.(lp .- lp0)
                -mean(min.(ρ .* a, clamp.(ρ, 1 - clip, 1 + clip) .* a))
            else
                -mean(lp .* a)
            end
        end
        Optimisers.update!(p.opt_a, (actor = p.actor, logσ = p.logσ), ga[1])
        lc, gc = Flux.withgradient(c -> mean((vec(c(o)) .- g) .^ 2), p.critic)
        Optimisers.update!(p.opt_c, p.critic, gc[1])
    end
    return Dict("loss_actor" => la, "loss_critic" => lc, "sigma_mean" => mean(exp.(p.logσ)))
end

# ─────────────────────────────────────────────────────────────────────────────
# Off-policy: TD3 and SAC share the replay buffer and the twin critics
# ─────────────────────────────────────────────────────────────────────────────

"Replay buffer of (s, y, r, s', done) as preallocated column matrices."
mutable struct Replay
    O::Matrix{Float32}; Y::Matrix{Float32}; R::Vector{Float32}; O2::Matrix{Float32}; D::Vector{Float32}
    n::Int; cap::Int; head::Int
end
Replay(dobs, dact, cap) = Replay(zeros(Float32, dobs, cap), zeros(Float32, dact, cap), zeros(Float32, cap),
                                 zeros(Float32, dobs, cap), zeros(Float32, cap), 0, cap, 0)
function push_episode!(b::Replay, ep)
    for t in ep
        b.head = b.head % b.cap + 1
        b.O[:, b.head] = t.obs; b.Y[:, b.head] = t.y; b.R[b.head] = -t.cost / REWARD_SCALE
        b.O2[:, b.head] = t.next_obs; b.D[b.head] = t.done ? 1f0 : 0f0
        b.n = min(b.n + 1, b.cap)
    end
end
function sample_batch(b::Replay, rng, k)
    i = rand(rng, 1:b.n, k)
    return b.O[:, i], b.Y[:, i], b.R[i], b.O2[:, i], b.D[i]
end

"""
    OffPolicy

`kind = :td3`: deterministic actor y = σ(π(s)); `kind = :sac`: actor outputs
[μ; log σ] of u, y = (tanh(u)+1)/2. Twin critics Q₁, Q₂(s, y) with Polyak targets.
"""
mutable struct OffPolicy
    kind::Symbol
    actor::Chain; actor_t::Chain
    q1::Chain; q2::Chain; q1_t::Chain; q2_t::Chain
    logα::Vector{Float32}
    opt_a::Any; opt_q::Any; opt_α::Any
    buffer::Replay
    updates::Int
    dact::Int
end

function OffPolicy(kind, dobs, dact; lr = 3f-4, cap = 500_000)
    actor = mlp(dobs, kind === :sac ? 2dact : dact)
    q1 = mlp(dobs + dact, 1); q2 = mlp(dobs + dact, 1)
    logα = [log(0.1f0)]
    return OffPolicy(kind, actor, deepcopy(actor), q1, q2, deepcopy(q1), deepcopy(q2), logα,
                     Optimisers.setup(Optimisers.Adam(lr), actor),
                     Optimisers.setup(Optimisers.Adam(lr), (q1 = q1, q2 = q2)),
                     Optimisers.setup(Optimisers.Adam(lr), logα),
                     Replay(dobs, dact, cap), 0, dact)
end

"SAC: sample y and its log-density under the squashed Gaussian (columns = samples)."
function sac_sample(actor, o, rng, dact)
    h = actor(o)
    μ = h[1:dact, :]; logσ = clamp.(h[dact+1:end, :], -5f0, 2f0)
    ϵ = randn(rng, Float32, size(μ))
    u = μ .+ exp.(logσ) .* ϵ
    th = tanh.(u)
    logp = gauss_logp(u, μ, logσ) .- sum(log.(0.5f0 .* (1f0 .- th .^ 2) .+ 1f-6); dims = 1)
    return (th .+ 1f0) ./ 2f0, logp
end

function act_explore(p::OffPolicy, o, rng; noise = 0.1f0)
    if p.kind === :sac
        y, _ = sac_sample(p.actor, reshape(o, :, 1), rng, p.dact)
        return (vec(y), nothing)
    end
    y = sigmoid.(p.actor(o)) .+ noise .* randn(rng, Float32, p.dact)
    return (clamp.(y, 0f0, 1f0), nothing)
end
function act_greedy(p::OffPolicy, o)
    p.kind === :sac && return ((tanh.(p.actor(o)[1:p.dact]) .+ 1f0) ./ 2f0, nothing)
    return (sigmoid.(p.actor(o)), nothing)
end

"""
    update!(p::OffPolicy, rng, n_updates) -> Dict

`n_updates` gradient steps on minibatches of 256 from the replay buffer.
TD3: target y' = clamp(σ(π'(s')) + clip(N(0, 0.2), ±0.5), 0, 1), target
r + γ(1−d) min Q'ᵢ(s', y'); actor and targets every 2nd step, τ = 0.005.
SAC: target r + γ(1−d)(min Q'ᵢ(s', y') − α log π(y'|s')); actor loss
E[α log π − min Qᵢ]; α tuned towards entropy −dim(y).
"""
function update!(p::OffPolicy, rng, n_updates; batch = 256, τ = 0.005f0)
    p.buffer.n < 10 * batch && return Dict{String,Any}()
    lq = 0f0; la = 0f0
    for _ in 1:n_updates
        o, y, r, o2, d = sample_batch(p.buffer, rng, batch)
        if p.kind === :sac
            y2, lp2 = sac_sample(p.actor, o2, rng, p.dact)
            qn = min.(p.q1_t(vcat(o2, y2)), p.q2_t(vcat(o2, y2))) .- exp(p.logα[1]) .* lp2
        else
            ϵ = clamp.(0.2f0 .* randn(rng, Float32, size(y)), -0.5f0, 0.5f0)
            y2 = clamp.(sigmoid.(p.actor_t(o2)) .+ ϵ, 0f0, 1f0)
            qn = min.(p.q1_t(vcat(o2, y2)), p.q2_t(vcat(o2, y2)))
        end
        target = reshape(r, 1, :) .+ Float32(GAMMA) .* (1f0 .- reshape(d, 1, :)) .* qn
        lq, gq = Flux.withgradient((q1 = p.q1, q2 = p.q2)) do q
            x = vcat(o, y)
            mean((q.q1(x) .- target) .^ 2) + mean((q.q2(x) .- target) .^ 2)
        end
        Optimisers.update!(p.opt_q, (q1 = p.q1, q2 = p.q2), gq[1])
        p.updates += 1
        if p.kind === :sac
            α = exp(p.logα[1])
            la, ga = Flux.withgradient(p.actor) do a
                yb, lpb = sac_sample(a, o, rng, p.dact)
                mean(α .* lpb .- min.(p.q1(vcat(o, yb)), p.q2(vcat(o, yb))))
            end
            Optimisers.update!(p.opt_a, p.actor, ga[1])
            # Entropy of the UPDATED actor, outside the actor's tape.
            lpd = Float32(mean(last(sac_sample(p.actor, o, rng, p.dact))))
            gα = Flux.gradient(lα -> -lα[1] * (lpd - p.dact), p.logα)
            Optimisers.update!(p.opt_α, p.logα, gα[1])
            soft_update!(p.q1_t, p.q1, τ); soft_update!(p.q2_t, p.q2, τ)
        elseif p.updates % 2 == 0
            la, ga = Flux.withgradient(a -> -mean(p.q1(vcat(o, sigmoid.(a(o))))), p.actor)
            Optimisers.update!(p.opt_a, p.actor, ga[1])
            soft_update!(p.actor_t, p.actor, τ); soft_update!(p.q1_t, p.q1, τ); soft_update!(p.q2_t, p.q2, τ)
        end
    end
    return Dict("loss_q" => lq, "loss_actor" => la, "alpha" => exp(p.logα[1]), "buffer" => p.buffer.n)
end

"Construct the learner for `algo ∈ (reinforce, ppo, td3, sac)`."
function make_learner(algo::AbstractString, dobs, dact)
    a = Symbol(lowercase(algo))
    a in (:reinforce, :ppo) && return OnPolicy(a, dobs, dact)
    a in (:td3, :sac) && return OffPolicy(a, dobs, dact)
    error("unknown algorithm $algo (reinforce | ppo | td3 | sac)")
end
