# train_rl.jl — train one RL baseline on the Bolivia hydro problem (or run the
# environment parity check), with episodes collected in parallel worker processes.
#
#   DR_RL_MODE      train | parity | eval                (default train)
#   DR_RL_EVAL_COLS eval: paired-protocol columns 1..N scored with the final policy (500)
#   DR_RL_ALGO      reinforce | ppo | td3 | sac          (train)
#   DR_RL_ITERS     iterations; each collects DR_RL_EPISODES episodes   (default 500)
#   DR_RL_EPISODES  episodes per iteration               (default: number of workers)
#   DR_RL_WORKERS   worker processes                     (default: CPUs − 1)
#   DR_RL_EVAL_EVERY  panel evaluation interval, iterations (default 5)
#   DR_RL_WARMUP    random-action iterations before off-policy learning (default 5)
#   DR_RL_SEED      seed                                 (default 20261001)
#   DR_RL_OUT       output directory (checkpoint, history)  (required)
#   DR_RL_TSDDR_CKPT  TS-DDR checkpoint for the parity check
#   DR_RL_PARITY_CSV  JuMP per-column reference costs of that checkpoint
#
# Every reported cost is the 96-stage cost of the GREEDY policy on the ten
# paired-protocol columns TS-DDR is selected on, from the same strict JuMP
# problem; training episodes are 126 stages (96 + the 30-stage look-ahead), as
# TS-DDR's. A run whose checkpoint exists RESUMES from it (preemptible queue).

using Distributed, Printf, Dates, JLD2, JSON, Random, Statistics

const MODE = get(ENV, "DR_RL_MODE", "train")
const NW = parse(Int, get(ENV, "DR_RL_WORKERS", string(max(1, Sys.CPU_THREADS - 1))))
addprocs(NW; exeflags = "--project=$(Base.active_project())")
const HERE = @__DIR__
@everywhere include(joinpath($HERE, "hydro_rl_env.jl"))
@everywhere include(joinpath($HERE, "rl_algorithms.jl"))
@everywhere const ENVW = build_env()

# ── Worker-side episode functions (actor VIEWS only — never the replay buffer) ──
@everywhere begin
    "Lightweight copy of what acting needs."
    actor_view(p::OnPolicy) = (kind = p.kind, actor = p.actor, logσ = p.logσ, dact = length(p.logσ))
    actor_view(p::OffPolicy) = (kind = p.kind, actor = p.actor, logσ = Float32[], dact = p.dact)

    "Exploration / greedy action of a view, matching `act_explore` / `act_greedy`."
    function view_act(v, o, rng, explore::Bool)
        if v.kind in (:reinforce, :ppo)
            μ = v.actor(o)
            explore || return (sigmoid.(μ), nothing)
            u = μ .+ exp.(v.logσ) .* randn(rng, Float32, length(μ))
            return (sigmoid.(u), (u = u, logp = only(gauss_logp(u, μ, v.logσ))))
        elseif v.kind === :sac
            explore || return ((tanh.(v.actor(o)[1:v.dact]) .+ 1f0) ./ 2f0, nothing)
            y, _ = sac_sample(v.actor, reshape(o, :, 1), rng, v.dact)
            return (vec(y), nothing)
        else
            y = sigmoid.(v.actor(o))
            explore || return (y, nothing)
            return (clamp.(y .+ 0.1f0 .* randn(rng, Float32, v.dact), 0f0, 1f0), nothing)
        end
    end

    "One 126-stage training episode on an independent inflow draw."
    function remote_episode(v, seed::Integer, mode::Symbol)
        rng = Random.Xoshiro(seed)
        sc = sample_scenario(ENVW, rng, TRAIN_STAGES)
        act = mode === :random ? ((o, x, w, t) -> (rand(rng, Float32, ENVW.nhyd), nothing)) :
              ((o, x, w, t) -> view_act(v, o, rng, mode === :explore))
        return rollout(ENVW, act, sc)
    end

    "Greedy 96-stage cost on one paired-protocol column."
    remote_panel(v, col::Integer) =
        rollout(ENVW, (o, x, w, t) -> view_act(v, o, nothing, false),
                panel_scenario(ENVW, col, REPORT_STAGES); record = false)

    "Parity: the TS-DDR checkpoint drives the environment, 96 stages, one column."
    function remote_parity(ckpt::AbstractString, col::Integer)
        pol = load_hydro_reachable_policy(ckpt, ENVW.meta, [128, 128]; combiner_layers = [256, 256])
        Flux.reset!(pol)
        act = (o, x, w, t) -> (Float64.(pol(vcat(Float32.(inflow_values(w)), Float32.(x)))), :target)
        return rollout(ENVW, act, panel_scenario(ENVW, col, REPORT_STAGES); record = false)
    end
end

"Panel evaluation: mean greedy 96-stage cost over the ten columns (NaN if any failed)."
function panel_eval(v)
    rs = pmap(c -> remote_panel(v, c), PANEL_IDS)
    nfail = count(r -> !r.ok, rs)
    return (mean = nfail == 0 ? mean(r.cost96 for r in rs) : NaN, nfail = nfail,
            costs = [r.cost96 for r in rs])
end

function run_parity()
    ckpt = ENV["DR_RL_TSDDR_CKPT"]; ref = ENV["DR_RL_PARITY_CSV"]
    rows = split.(readlines(ref)[2:end], ',')
    refcost = Dict(parse(Int, r[1]) => parse(Float64, r[2]) for r in rows)
    rs = pmap(c -> remote_parity(ckpt, c), PANEL_IDS)
    worst = 0.0
    for (c, r) in zip(PANEL_IDS, rs)
        d = r.cost96 - refcost[c]; worst = max(worst, abs(d) / refcost[c])
        @printf("[parity] column %3d  RL env %.4f  JuMP evaluator %.4f  diff %+.4f (%+.2e)  %s\n",
                c, r.cost96, refcost[c], d, d / refcost[c], r.ok ? "" : "FAILED")
    end
    m = mean(r.cost96 for r in rs); mr = mean(refcost[c] for c in PANEL_IDS)
    @printf("[parity] panel mean RL env %.4f vs evaluator %.4f  (worst column rel. diff %.2e)\n", m, mr, worst)
    return worst < 1e-5 ? 0 : 1
end

function run_train()
    algo = lowercase(ENV["DR_RL_ALGO"])
    out = ENV["DR_RL_OUT"]; mkpath(out)
    iters = parse(Int, get(ENV, "DR_RL_ITERS", "500"))
    neps = parse(Int, get(ENV, "DR_RL_EPISODES", string(NW)))
    every = parse(Int, get(ENV, "DR_RL_EVAL_EVERY", "5"))
    warm = parse(Int, get(ENV, "DR_RL_WARMUP", "5"))
    seed = parse(Int, get(ENV, "DR_RL_SEED", "20261001"))
    ckpt = joinpath(out, "$(algo).checkpoint.jld2")
    hist = joinpath(out, "$(algo).history.jsonl")
    nhyd = remotecall_fetch(() -> ENVW.nhyd, workers()[1])
    dobs = 2nhyd + 1

    # ── Resume or start ──────────────────────────────────────────────────────
    if isfile(ckpt)
        S = JLD2.load(ckpt)
        learner, it0, episodes, steps, rng, run_name, best = S["learner"], S["iter"], S["episodes"],
            S["steps"], S["rng"], S["run_name"], S["best"]
        @printf("[rl] RESUMING %s from iteration %d (%d episodes)\n", algo, it0, episodes)
    else
        Random.seed!(seed)
        learner = make_learner(algo, dobs, nhyd)
        it0, episodes, steps, rng, best = 0, 0, 0, Random.Xoshiro(seed), Inf
        run_name = "rl-$(algo)-bolivia-strict-K06048-$(Dates.format(now(), "yyyymmdd-HHMMSS"))"
    end
    lg = nothing
    if get(ENV, "DR_ENABLE_WANDB", "false") == "true"
        try
            # Loaded at run time, so the module is fetched from Main by name: a literal
            # `Wandb.` here would be resolved when this function was compiled, before
            # the package existed (that silently disabled W&B in the first runs).
            @eval Main using Wandb
            W = getglobal(Main, :Wandb)
            lg = Base.invokelatest(W.WandbLogger; project = "RL", name = run_name,
                config = Dict("algo" => algo, "workers" => NW, "episodes_per_iter" => neps,
                              "train_stages" => TRAIN_STAGES, "report_stages" => REPORT_STAGES,
                              "reward_scale" => REWARD_SCALE, "gamma" => GAMMA, "hidden" => collect(HIDDEN),
                              "panel" => PANEL_IDS, "seed" => seed))
        catch err
            @warn "W&B logger unavailable; continuing with the JSONL history" err
        end
    end
    # A failed panel column gives NaN; JSON has no NaN, so non-finite values are
    # written as null (one NaN crashed the first TD3 run).
    clean(d) = Dict(k => (v isa AbstractFloat && !isfinite(v)) ? nothing : v for (k, v) in d)
    logw(d) = (open(io -> println(io, JSON.json(clean(d))), hist, "a");
               lg === nothing || Base.invokelatest(getglobal(Main, :Wandb).log, lg,
                                                   Dict(k => v for (k, v) in d if v isa Number && isfinite(v))))

    t0 = time()
    for it in (it0 + 1):iters
        v = actor_view(learner)
        mode = (learner isa OffPolicy && it <= warm) ? :random : :explore
        seeds = [rand(rng, UInt32) for _ in 1:neps]
        eps = pmap(s -> remote_episode(v, s, mode), seeds)
        good = [e for e in eps if e.ok]
        episodes += length(eps); steps += sum(length(e.transitions) for e in good; init = 0)
        stats = Dict{String,Any}()
        if !isempty(good)
            if learner isa OffPolicy
                foreach(e -> push_episode!(learner.buffer, e.transitions), good)
                stats = update!(learner, rng, sum(length(e.transitions) for e in good))
            else
                stats = update!(learner, [e.transitions for e in good])
            end
        end
        rec = Dict{String,Any}("iter" => it, "episodes" => episodes, "env_steps" => steps,
                               "train_cost126" => isempty(good) ? NaN : mean(e.cost for e in good),
                               "train_cost96" => isempty(good) ? NaN : mean(e.cost96 for e in good),
                               "failed_episodes" => length(eps) - length(good),
                               "wall_s" => time() - t0)
        for (k, x) in stats; rec[k] = x isa Number ? Float64(x) : x; end
        if it % every == 0 || it == iters
            pe = panel_eval(actor_view(learner))
            rec["panel_cost96"] = pe.mean; rec["panel_failed"] = pe.nfail
            isfinite(pe.mean) && pe.mean < best && (best = pe.mean)
            rec["panel_best"] = best
            @printf("[rl] %s it %4d  episodes %6d  train96 %.0f  PANEL %.1f (best %.1f)  %.0f s\n",
                    algo, it, episodes, rec["train_cost96"], pe.mean, best, time() - t0)
        end
        logw(rec)
        tmp = ckpt * ".tmp"
        jldsave(tmp; learner = learner, iter = it, episodes = episodes, steps = steps, rng = rng,
                run_name = run_name, best = best)
        mv(tmp, ckpt; force = true)
        flush(stdout)
    end
    lg === nothing || Base.invokelatest(close, lg)   # WandbLogger supports Base.close
    return 0
end

"""
    run_eval() -> Int

Greedy 96-stage cost of the FINAL policy in `<DR_RL_OUT>/<algo>.checkpoint.jld2` on
every paired-protocol column 1..`DR_RL_EVAL_COLS` (default 500), the protocol the
TS-DDR and SDDP policies are compared on. The final policy is used, not the best
panel iterate, so the ten selection columns confer no advantage. Writes
`<algo>.paired_<N>.csv` (column, cost96, ok).
"""
function run_eval()
    algo = lowercase(ENV["DR_RL_ALGO"]); out = ENV["DR_RL_OUT"]
    n = parse(Int, get(ENV, "DR_RL_EVAL_COLS", "500"))
    learner = JLD2.load(joinpath(out, "$(algo).checkpoint.jld2"))["learner"]
    v = actor_view(learner)
    rs = pmap(c -> remote_panel(v, c), 1:n)
    path = joinpath(out, "$(algo).paired_$(n).csv")
    open(path, "w") do io
        println(io, "column,cost96,ok")
        for (c, r) in zip(1:n, rs); println(io, c, ",", r.cost96, ",", r.ok); end
    end
    ok = [r.cost96 for r in rs if r.ok]
    @printf("[eval] %s: %d/%d columns solved, mean %.2f, sd %.2f -> %s\n",
            algo, length(ok), n, mean(ok), std(ok), path)
    return length(ok) == n ? 0 : 1
end

exit(MODE == "parity" ? run_parity() : MODE == "eval" ? run_eval() : run_train())
