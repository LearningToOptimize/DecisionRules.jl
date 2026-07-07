# # Hydropower Scheduling
#
# This example trains target-setting decision rules for the Bolivia
# long-term hydrothermal dispatch (LTHD) problem — **TS-DDR** (deep,
# LSTM-based) and its linear counterpart **TS-LDR** — and compares them
# against an SDDP baseline with inconsistent formulations.  The strict
# (penalty-free) TS-DDR variant is trained in two independent
# implementations that we verify are numerically equivalent:
#
# 1. **Stage-wise subproblems on CPU** (this package: JuMP + DiffOpt + Ipopt),
#    solving one AC-OPF per stage in closed loop; and
# 2. **Full-horizon deterministic equivalent on GPU**
#    ([DecisionRulesExa.jl](https://github.com/LearningToOptimize/DecisionRulesExa.jl):
#    ExaModels + MadNLP/cuDSS), solving one coupled 126-stage NLP per
#    gradient sample.
#
# The Bolivia system has **28 buses**, **34 generators**, **11 hydro
# plants** (three of them in river cascades), and **AC power flow**
# constraints.  Policies are trained on a 126-stage horizon and evaluated on
# **96 monthly stages**; inflow uncertainty is sampled from **47 historical
# joint scenarios** (spatially correlated across plants, tiled cyclically
# for horizons beyond the record).
#
# ## Overview of the TS-DDR approach
#
# Classical stochastic programming (e.g., SDDP) constructs piecewise-linear
# value-function approximations.  TS-DDR takes a different route: a neural
# network policy ``\pi_\theta`` maps observations to **target states**, and a
# projection subproblem at each stage enforces physical feasibility while
# tracking those targets as closely as possible.
#
# The key insight is that the gradient of the projection subproblem with
# respect to the target parameters is available through Lagrange duality
# (or equivalently, implicit differentiation of the KKT conditions).
# This avoids differentiating through the full optimization solver.
#
# ## Problem formulation
#
# At each stage ``t``, the operator observes inflows ``w_t`` and the current
# reservoir state ``x_{t-1}``.  The policy predicts target volumes:
#
# ```math
# \hat{x}_t = \pi_\theta(w_{1:t},\, x_{t-1}).
# ```
#
# A stage subproblem projects onto the feasible set:
#
# ```math
# \begin{aligned}
# q_t(x_{t-1},\, w_t;\; \hat{x}_t)
#   \;=\;
#   \min_{x_t, u_t, \delta_t}
#   \quad &
#   c_t(x_t, u_t) + C_\delta\, \|\delta_t\| \\
# \text{s.t.}\quad
#   & x_t = x_{t-1} + w_t - \text{turbined}_t - \text{spilled}_t,
#         && \text{(reservoir balance)} \\
#   & x_t + \delta_t = \hat{x}_t,
#         && : \lambda_t \quad \text{(target constraint)} \\
#   & \text{AC-OPF}(u_t),
#         && \text{(power flow)}  \\
#   & x_t \in [0, \bar{x}],\; u_t \ge 0.
# \end{aligned}
# ```
#
# The slack variable ``\delta_t`` absorbs infeasible targets; ``\lambda_t`` is
# the dual multiplier that provides the gradient signal.
#
# ## Gradient computation: the envelope theorem
#
# By the envelope theorem, the sensitivity of the optimal value with respect
# to the target parameter is available in closed form from the multiplier of
# the target constraint.  Throughout this page we define ``\lambda_t`` as
# that sensitivity,
#
# ```math
# \lambda_t
# \;:=\;
# \frac{\partial q_t}{\partial \hat{x}_t},
# ```
#
# i.e. the constraint multiplier reported by the solver, mapped through the
# sign convention of the modeling layer (both implementations extract it this
# way, and the two extractions are verified to agree numerically — see the
# equivalence audit in the Results section).  Combined with backpropagation
# through the policy network, the full gradient of the expected cost is:
#
# ```math
# \nabla_\theta \mathbb{E}[Q]
# \;\approx\;
# \frac{1}{S} \sum_{s=1}^{S} \sum_{t=1}^{T}
#   \lambda_t^s \odot \nabla_\theta \hat{x}_t^s(\theta),
# ```
#
# where ``S`` is the number of sampled trajectories per batch and ``\odot``
# denotes elementwise multiplication.

# ## Problem setup
#
# The JuMP subproblems are built from a MOF file (exported from PowerModels.jl)
# plus hydro data (reservoir limits, inflow scenarios).  Each subproblem contains:
# - AC optimal power flow constraints
# - Reservoir balance: `vol_out = vol_in + inflow - turbined - spilled`
# - Target-slack deficit variables penalizing deviation from the policy's targets
#
# The helper `build_hydropowermodels` reads the case data, creates one JuMP model
# per stage, and parameterizes the initial volumes and inflows so they can be set
# at each training sample.

using DecisionRules
using JuMP, DiffOpt, Ipopt
using Flux
using Statistics, Random

# Load the problem builder (reads MOF + hydro JSON + inflow CSV).
#
# ```julia
# include("load_hydropowermodels.jl")
# ```

# ## Building the stage-wise subproblems
#
# Each subproblem is wrapped with `DiffOpt.diff_optimizer` so that Lagrange duals
# and implicit sensitivities are available for training.

# ```julia
# diff_optimizer = () -> DiffOpt.diff_optimizer(
#     optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "linear_solver" => "mumps")
# )
#
# subproblems, state_params_in, state_params_out, uncertainty_samples,
#     initial_state, max_volume, hydro_meta =
#     build_hydropowermodels(
#         "bolivia", "ACPPowerModel.mof.json";
#         num_stages=96,
#         optimizer=diff_optimizer,
#         penalty_l1=:auto, penalty_l2=:auto,
#     )
# ```

# ## Policy architecture
#
# The policy is a [`StateConditionedPolicy`](@ref) with two components:
#
# 1. **Encoder** — a stack of LSTM cells that processes only the uncertainty
#    (inflow) sequence, capturing temporal dependencies across stages.
# 2. **Combiner** — a feed-forward target head that merges the encoded
#    uncertainty with the previous state to produce the next target.
#
# At each stage the policy receives ``[w_t;\; x_{t-1}]`` and outputs
# target reservoir volumes ``\hat{x}_t``:
#
# ```
#  ┌─────────┐      ┌────────────────┐      ┌──────────────┐
#  │   w_t   │─────▶│  LSTM encoder  │─────▶│              │
#  └─────────┘      └────────────────┘      │ feed-forward │──▶ x̂_t
#  ┌─────────┐                              │    head      │
#  │ x_{t-1} │─────────────────────────────▶│              │
#  └─────────┘                              └──────────────┘
# ```
#
# The LSTM carries hidden state across stages, giving the policy memory of
# past inflows.  The activation is `sigmoid` (bounding outputs to ``[0,1]``,
# which is then scaled by the feasibility mapping).
#
# The `layers` argument controls recurrence over inflows only. The optional
# `combiner_layers` argument adds hidden layers to the nonrecurrent
# state-conditioned target head. This is the preferred way to make TS-DDR
# nonlinear in the current reservoir state without adding recurrence over state.
#
# !!! warning "Recurrent state must be threaded explicitly (Flux ≥ 0.16)"
#     Since Flux 0.16, calling an `LSTM` layer restarts it from
#     `initialstates` on **every call**, and `Flux.reset!` is a deprecated
#     no-op.  A per-stage policy loop written naively against this API is
#     silently *memoryless* — the "LSTM" degenerates to a per-stage MLP over
#     the current inflow.  Both packages therefore thread the hidden state
#     explicitly across stages inside their policy types (and implement a
#     real `Flux.reset!` for scenario boundaries).  The effect is large: on
#     this example, evaluating the same trained weights with and without
#     state threading changes the 100-scenario mean cost by **19%**
#     (361,056 vs 303,631).  If you build custom recurrent policies on this
#     framework, verify statefulness with the pattern used in the package
#     test suites: the same input fed twice without `reset!` must produce
#     different outputs.

# ```julia
# models = state_conditioned_policy(
#     num_uncertainties, num_hydro, num_hydro, [128, 128];
#     activation=sigmoid, encoder_type=Flux.LSTM,
#     combiner_layers=[128, 128],
# )
# ```

# ## TS-LDR: Linear Decision Rules
#
# As a baseline, we also train a **linear** policy (TS-LDR).  This uses
# `dense_multilayer_nn` with identity activation — a composition of linear
# layers equivalent to a single affine map:
#
# ```math
# \hat{x}_t = W [w_{1:t};\; x_{t-1}] + b.
# ```
#
# TS-LDR uses the same target-setting framework and training pipeline as
# TS-DDR.  The only difference is the policy class: linear maps have fewer
# parameters and cannot capture nonlinear inflow patterns, but they are a
# natural baseline from the classical LDR literature.

# ```julia
# num_inputs = DecisionRules.policy_input_dim(num_uncertainties, num_hydro)
# models = dense_multilayer_nn(num_inputs, num_hydro, [64, 64]; activation=identity)
# ```

# ## Training pipeline 1: Deterministic Equivalent
#
# The deterministic equivalent (DE) couples all 96 stages into a **single NLP**
# for each sampled trajectory.  This is the most direct formulation: the policy
# generates the full target trajectory ``\hat{x}_{1:T}`` in one forward pass,
# and a single coupled solve determines all realized states simultaneously.
#
# ### How it works
#
# ```
#  ┌──────────────────────────────────────────────────────────┐
#  │  For each sampled trajectory w_{1:T}:                    │
#  │                                                          │
#  │  1. Forward pass: x̂_{1:T} = π_θ(w_{1:T}, x_0)          │
#  │                                                          │
#  │  2. Solve coupled NLP:                                   │
#  │     min  Σ_t c_t(x_t, u_t) + C_δ Σ_t ‖δ_t‖             │
#  │     s.t. dynamics + AC-OPF for ALL stages simultaneously │
#  │          x_t + δ_t = x̂_t(θ)   ∀t  (target constraint)  │
#  │                                                          │
#  │  3. Read duals λ_t of target constraints                 │
#  │     Gradient: Σ_t λ_t ⊙ ∇_θ x̂_t(θ)                     │
#  └──────────────────────────────────────────────────────────┘
# ```
#
# ### Mathematical formulation
#
# ```math
# \begin{aligned}
# Q(w;\, \theta)
#   \;=\;
#   \min_{\{x_t, u_t, \delta_t\}_{t=1}^{T}}
#   \quad &
#   \sum_{t=1}^{T} c_t(x_t, u_t)
#   + C_\delta \sum_{t=1}^{T} \|\delta_t\| \\
# \text{s.t.}\quad
#   & x_t = T_t(w_t,\, u_t,\, x_{t-1}),
#         && t=1,\ldots,T \\
#   & x_t + \delta_t = \hat{x}_t(\theta),
#         && : \lambda_t,\quad t=1,\ldots,T \\
#   & h_t(x_t, u_t) \ge 0,
#         && t=1,\ldots,T
# \end{aligned}
# ```
#
# The gradient is exact by the envelope theorem:
#
# ```math
# \nabla_\theta Q
# \;=\;
# \sum_{t=1}^{T}
# \lambda_t \odot \nabla_\theta \hat{x}_t(\theta).
# ```
#
# **Advantages**: strongest gradient signal — full cross-stage coupling
# captures how a target at stage 3 affects costs at stage 50.
#
# **Disadvantage**: the NLP has ``96 \times (\text{AC-OPF variables})``
# decision variables; the policy generates targets without seeing realized
# states (open-loop target generation).

# ```julia
# det_equivalent, uncertainty_samples_det = DecisionRules.deterministic_equivalent!(
#     det_model, subproblems_de, state_params_in, state_params_out,
#     Float64.(initial_state), uncertainty_samples,
# )
#
# train_multistage(
#     models, initial_state, det_equivalent,
#     state_params_in, state_params_out, uncertainty_samples_det;
#     num_batches=4000, optimizer=Flux.Adam(),
#     penalty_schedule=[(1,100,0.1), (101,210,1.0), (211,300,10.0), (301,4000,30.0)],
# )
# ```

# ## Training pipeline 2: Stage-wise Decomposition (Single Shooting)
#
# Stage-wise decomposition solves one subproblem per stage sequentially.
# Unlike the DE, the policy operates in **closed loop**: after each stage
# solve, the realized state ``x_t`` (not the predicted target) is fed back
# as input to the next stage.
#
# ### How it works
#
# ```
#  ┌─────────────────────────────────────────────────────────────┐
#  │  For each sampled trajectory w_{1:T}:                       │
#  │                                                             │
#  │  x_0 = initial state                                        │
#  │  for t = 1, ..., T:                                         │
#  │      x̂_t = π_θ(w_t, x_{t-1})          ← predict target     │
#  │      solve stage-t subproblem          ← project to feasible│
#  │      x_t = realized state from solver  ← closed-loop        │
#  │      accumulate c_t + C_δ ‖δ_t‖                             │
#  │                                                             │
#  │  Gradient: chain rule through all stage solves               │
#  └─────────────────────────────────────────────────────────────┘
# ```
#
# ### Gradient chain
#
# The gradient must account for two coupled feedback paths: the realized
# state at stage ``t`` depends on the targets at all earlier stages, and the
# *policy input* at stage ``t`` includes that realized state.  Writing
# ``x_t = X_t(x_{t-1}, \hat{x}_t)`` for the realized-state map of the stage
# solve, the exact total-derivative recursion is
#
# ```math
# \frac{dQ}{d\theta}
# = \sum_{t=1}^{T} \left[
#   \frac{\partial q_t}{\partial \hat{x}_t} \frac{d \hat{x}_t}{d\theta}
#   + \frac{\partial q_t}{\partial x_{t-1}} \frac{d x_{t-1}}{d\theta}
#   \right],
# \qquad
# \frac{d x_t}{d\theta}
# = \frac{\partial X_t}{\partial x_{t-1}} \frac{d x_{t-1}}{d\theta}
# + \frac{\partial X_t}{\partial \hat{x}_t} \frac{d \hat{x}_t}{d\theta},
# ```
#
# ```math
# \frac{d \hat{x}_t}{d\theta}
# = \nabla_\theta \pi_\theta
# + \frac{\partial \pi_\theta}{\partial x_{t-1}} \frac{d x_{t-1}}{d\theta}.
# ```
#
# Reverse-mode automatic differentiation (Zygote + ChainRules `rrule`s
# defined on each stage solve) computes exactly this chain, including the
# policy-feedback term ``\partial \pi_\theta / \partial x_{t-1}``.
# The `rrule` for each stage solve reads the dual ``\lambda_t`` for the
# target constraint and uses DiffOpt's implicit differentiation for the
# state-transition sensitivities ``\partial X_t / \partial \cdot``.
#
# **Advantages**: closed-loop — the policy sees realized states, matching
# deployment semantics.  Each solve is small (single-stage AC-OPF).
#
# **Disadvantage**: gradients weaken over long horizons because the
# chain rule multiplies many Jacobians; sequential solve prevents
# parallelism.

# ```julia
# train_multistage(
#     models, initial_state, subproblems,
#     state_params_in, state_params_out, uncertainty_samples;
#     num_batches=3000, optimizer=Flux.Adam(),
#     penalty_schedule=:default_annealed,
# )
# ```

# ## Training pipeline 3: Multiple Shooting
#
# Multiple shooting partitions the ``T``-stage horizon into ``K`` windows of
# ``W`` stages each.  Within each window, a local deterministic equivalent
# couples the stages (strong gradient signal).  Between windows, the realized
# end-state is passed to the next window (closed-loop continuity).
#
# ### How it works
#
# ```
#  ┌────────────────────────────────────────────────────────────────┐
#  │  Partition T=96 stages into K=⌈96/12⌉=8 windows of W=12      │
#  │                                                                │
#  │  x_0 = initial state                                           │
#  │  for k = 1, ..., K:                                            │
#  │      stages = [(k-1)W+1, ..., kW]                              │
#  │      x̂_{stages} = π_θ(w_{stages}, x_{start_k})                │
#  │      solve window-k DE (12-stage coupled NLP)                  │
#  │      x_{end_k} = realized end-state from window solve          │
#  │      x_{start_{k+1}} = x_{end_k}                               │
#  │                                                                │
#  │  Gradient:                                                     │
#  │    Within window: duals from the coupled solve (like full DE)  │
#  │    Across windows: DiffOpt chain rule through end-states       │
#  └────────────────────────────────────────────────────────────────┘
# ```
#
# ### Gradient structure
#
# Let ``Q_k`` be the cost of window ``k``.  The total cost is
# ``Q = \sum_k Q_k``.  Within a window, the gradient is identical to the
# DE case (duals of the target constraints in the coupled model).  Across
# windows, the chain rule threads through the realized end-state:
#
# ```math
# \frac{dQ}{d\theta}
# \;=\;
# \sum_{k=1}^{K}
# \left(
#   \frac{\partial Q_k}{\partial \hat{x}_k}
#   \cdot \frac{\partial \hat{x}_k}{\partial \theta}
#   \;+\;
#   \frac{\partial Q_k}{\partial x_{\text{start}_k}}
#   \cdot \frac{d x_{\text{start}_k}}{d\theta}
# \right),
# ```
#
# where ``\frac{d x_{\text{start}_k}}{d\theta}`` involves the chain
# through all prior windows via ``x_{\text{end}_{k-1}}``.
#
# **Advantages**: balances gradient quality (12-stage coupling) with
# tractability (8 small DEs instead of one large one); inter-window
# chain provides some closed-loop signal.
#
# **Disadvantage**: window boundaries introduce gradient discontinuities;
# the full-horizon coupling is weaker than the single DE.

# ```julia
# windows = DecisionRules.setup_shooting_windows(
#     subproblems, state_params_in, state_params_out,
#     Float64.(initial_state), uncertainty_samples;
#     window_size=12,
#     model_factory=() -> DiffOpt.nonlinear_diff_model(ipopt_attrs),
# )
#
# train_multiple_shooting(
#     models, initial_state, windows, () -> uncertainty_samples;
#     num_batches=3000, optimizer=Flux.Adam(),
#     penalty_schedule=:default_annealed,
# )
# ```

# ## Training pipeline 4: Strict subproblems with reachable policy
#
# The three formulations above use a **slack penalty** ``C_\delta \|\delta_t\|``
# to handle the gap between the policy's targets and the feasible set.  While
# effective, the penalty introduces a hyperparameter and can corrupt the gradient
# signal: at high ``C_\delta``, the dual ``\lambda_t`` reflects "reduce the
# slack" rather than "improve economic dispatch."
#
# **Strict mode** eliminates the penalty entirely by enforcing a **hard equality**
# between the target and the realized state:
#
# ```math
# x_t = \hat{x}_t \quad :\lambda_t \qquad \text{(no slack, no } \delta_t \text{)}
# ```
#
# The dual ``\lambda_t`` is then the **pure shadow price**
# ``\partial q_t / \partial \hat{x}_t``: the economic value of a marginal change
# in the target, free of any penalty noise.
#
# ### Feasibility guarantee: HydroReachablePolicy
#
# Removing the slack requires that every target produced by the policy be
# **physically achievable**.  For hydro scheduling, this means the target volume
# must lie within the one-stage reachable set — the range of volumes achievable
# from the current state ``v_{r,t-1}`` by choosing turbine flow ``q_r`` and
# spillage ``s_r`` within their physical bounds.
#
# #### Per-unit reachable bounds
#
# The water balance for reservoir ``r`` at stage ``t`` is
#
# ```math
# v_{r,t} = v_{r,t-1} + K\, w_{r,t} - K\, q_{r,t} - K\, s_{r,t}
#           + \sum_{u \in \mathcal{U}_r} K\, q_{u,t}
#           + \sum_{u \in \mathcal{S}_r} K\, s_{u,t},
# ```
#
# where ``K`` is the water-balance conversion factor extracted from the model,
# ``w_{r,t}`` is the inflow,
# ``q_{r,t}`` is the turbined flow, ``s_{r,t}`` is the spillage,
# ``\mathcal{U}_r`` is the set of upstream units connected by turbine flow,
# and ``\mathcal{S}_r`` is the set connected by spillage.
#
# The reachable bounds for unit ``r`` (ignoring cascade interactions) are:
#
# ```math
# \ell_{r,t} = \max\bigl(\underline{v}_r,\;
#     v_{r,t-1} + K\, w_{r,t} - K\,\bar{q}_r - K\,\bar{s}_r
#     + K \sum_{u \in \mathcal{U}_r} \underline{q}_u\bigr),
# ```
#
# ```math
# u_{r,t} = \min\bigl(\bar{v}_r,\;
#     v_{r,t-1} + K\, w_{r,t} - K\,\underline{q}_r
#     + K \sum_{u \in \mathcal{U}_r} \bar{q}_u
#     + K \sum_{u \in \mathcal{S}_r} \bar{s}_u\bigr).
# ```
#
# These bounds assume worst-case upstream contributions (maximum turbine/spill
# capacity). The [`HydroReachablePolicy`] wraps the same LSTM uncertainty
# encoder plus feed-forward state-conditioned target head as
# [`StateConditionedPolicy`](@ref) but uses a **sigmoid** activation to bound
# the output to this reachable interval:
#
# ```math
# \hat{v}_{r,t} = \ell_{r,t} + (u_{r,t} - \ell_{r,t}) \cdot \sigma(z_{r,t}).
# ```
#
# #### Cascade-aware clamping
#
# The per-unit upper bound ``u_{r,t}`` uses worst-case upstream contributions
# (``K \bar{q}_u``, ``K \bar{s}_u``).  When an upstream unit ``u`` stores water
# (its target ``\hat{v}_{u,t}`` is high), the actual upstream release
#
# ```math
# R_u = K\, w_{u,t} + v_{u,t-1} - \hat{v}_{u,t}
# ```
#
# can be much less than the assumed maximum.  For cascaded systems, this means
# the downstream target may exceed the true reachable set, causing infeasibility
# in strict mode (no slack to absorb the gap).
#
# After computing the initial sigmoid targets for all units, the policy applies
# a **cascade clamping** step.  For each upstream→downstream connection:
#
# - **Turn + spill** connection: the full release reaches downstream,
#   so ``\text{max\_contrib} = \max(0,\, R_u)``.
# - **Turn-only** connection: only turbined flow reaches downstream,
#   so ``\text{max\_contrib} = \min(K\,\bar{q}_u,\, \max(0,\, R_u))``.
#
# The downstream target is then clamped:
#
# ```math
# \hat{v}_{d,t} \;\le\; v_{d,t-1} + K\, w_{d,t}
#   - K\,\underline{q}_d + \text{max\_contrib}.
# ```
#
# This clamping is `@non_differentiable` — gradient flows through ``\sigma``
# for unclamped targets, and is zero for clamped ones (correct projected-gradient
# signal).
#
# ### Setup
#
# Building strict subproblems requires only the `strict=true` flag:

# ```julia
# subproblems, state_params_in, state_params_out, uncertainty_samples,
#     initial_state, max_volume, hydro_meta = build_hydropowermodels(
#     case_dir, "ACPPowerModel.mof.json";
#     num_stages=126, optimizer=diff_optimizer,
#     strict=true,   # ← no deficit, hard equality targets
# )
# ```

# The reachable policy is constructed from the hydro metadata returned by
# `build_hydropowermodels`:

# ```julia
# models = hydro_reachable_policy(hydro_meta, [128, 128])
# models_with_deep_state_head = hydro_reachable_policy(
#     hydro_meta,
#     [128, 128];
#     combiner_layers=[256, 256],
# )
# ```

# Training uses the same `train_multistage` with no penalty schedule:

# ```julia
# train_multistage(
#     models, initial_state, subproblems,
#     state_params_in, state_params_out, uncertainty_samples;
#     num_batches=8000, optimizer=Flux.Adam(),
#     penalty_schedule=nothing,   # ← no penalty to tune
# )
# ```

# !!! tip "Out-of-the-box convergence"
#     Strict mode with `HydroReachablePolicy` requires **no penalty tuning**,
#     no annealing schedule, and no hyperparameter search.  The clean gradient
#     signal allows the optimizer to directly minimize operational cost.

# ## Training pipeline 5: Strict deterministic equivalent on GPU
#
# The strict formulation unlocks a second, much faster training route,
# implemented in the companion package
# [DecisionRulesExa.jl](https://github.com/LearningToOptimize/DecisionRulesExa.jl):
# the **full-horizon deterministic equivalent solved on GPU** with
# [ExaModels.jl](https://github.com/exanauts/ExaModels.jl) (SIMD-friendly
# algebraic modeling) and [MadNLP.jl](https://github.com/MadNLP/MadNLP.jl)
# with the cuDSS sparse linear solver.
#
# ### Why strict mode makes the regular DE safe
#
# A regular (non-embedded) DE is normally *open-loop*: the policy produces
# all targets ``\hat{x}_{1:T}`` before the coupled solve, seeing its own
# previous target instead of a realized state.  With an arbitrary policy this
# can render a strict DE infeasible.  The reachable policy restores safety
# **by induction**: roll targets out as ``\hat{x}_0 = x_0`` and
# ``\hat{x}_t = \pi_\theta(w_t, \hat{x}_{t-1})`` with every target inside the
# one-stage reachable set of its input state.  Stage 1 is then feasible from
# the true initial state; and if stages ``1..t`` are feasible, the strict
# equalities force ``x_t = \hat{x}_t``, so stage ``t{+}1`` starts exactly at
# the state the policy planned from — making ``\hat{x}_{t+1}`` feasible too.
# Strict feasibility *removes* the open-loop/closed-loop gap: the realized
# state path must equal the reachable target path, so training-time DE
# solves and deployment-time stage-wise rollouts traverse identical
# trajectories (we verify this numerically below).
#
# ### Reservoir volumes become parameters
#
# Because strict equalities pin every reservoir volume to its target, the
# volume trajectory is **data, not a decision**, for the inner solver.  The
# GPU formulation exploits this: the reservoir trajectory enters the NLP as
# a parameter vector, eliminating ``(T{+}1) \cdot n_{\text{hydro}}``
# variables and all slack variables from the KKT system.  The targets then
# appear only on the right-hand side of the ``T \cdot n_{\text{hydro}}``
# water-balance rows.  With ``\mu_t`` the multiplier of the stage-``t``
# water-balance row, the envelope gradient follows from the two rows each
# target touches (``+1`` on ``v_{t+1}`` in row ``t``, ``-1`` on ``v_t`` in
# row ``t{+}1``):
#
# ```math
# \frac{\partial Q}{\partial \hat{x}_t} = \mu_t - \mu_{t+1}
# \quad (t < T),
# \qquad
# \frac{\partial Q}{\partial \hat{x}_T} = \mu_T .
# ```
#
# ### What the GPU buys
#
# Each gradient sample requires one coupled 126-stage AC NLP solve.  On an
# NVIDIA H200, MadNLP + cuDSS solves it fast enough that a full 8,000-solve
# training run completes in roughly a day — and, more importantly, the
# training loss enters the SDDP-forward-cost envelope within the **first
# hours** (see the wall-clock convergence figure in the Results).  Solver
# state is reused across solves with a dual-snapshot warm-start scheme that
# prevents one failed solve from corrupting subsequent ones.
#
# ```julia
# ## In DecisionRulesExa.jl (see its examples/HydroPowerModels):
# de = build_hydro_de(power_data, hydro_data, 126;
#     formulation=:ac_polar, strict_targets=true,
#     backend=CUDABackend())
# policy = hydro_reachable_policy(hydro_data, [128, 128];
#     combiner_layers=[128, 128])
# train_tsddr(policy, x0, de, de.p_x0, de.p_target, de.p_inflow, sampler;
#     num_batches=8000, madnlp_kwargs=(tol=1e-6,))
# ```
#
# !!! note "Cross-package equivalence audit"
#     The two strict implementations are verified against each other on this
#     exact case: with identical weights, data, and scenarios, (i) the two
#     policy implementations produce bit-identical targets, (ii) the
#     stage-wise CPU rollout (Ipopt) and the strict full-horizon DE (MadNLP)
#     agree per scenario to ``10^{-9}`` relative cost, and (iii) the two
#     packages' 100-scenario evaluations of the same checkpoint agree to
#     0.001% (303,631 vs 303,635).  The audit scripts
#     (`dump_paired_policy_reference.jl`, `eval_paired_exa_strict.jl`,
#     `compare_paired_evals.jl`) ship with the packages and re-run on demand.

# ## Penalty annealing (non-strict formulations)
#
# For the non-strict formulations (DE, stage-wise, multiple shooting), the
# target penalty ``C_\delta`` controls the trade-off between following
# the policy's targets and minimizing operational cost.  DecisionRules
# supports a **penalty annealing schedule** that ramps the penalty multiplier
# during training:
#
# | Phase | Multiplier | Purpose |
# |:------|:----------:|:--------|
# | Warmup | ``0.1 \times C_\delta`` | Let the policy explore freely |
# | Nominal | ``1.0 \times C_\delta`` | Standard training |
# | Tighten | ``10.0 \times C_\delta`` | Sharpen target tracking |
# | Lock | ``30.0 \times C_\delta`` | Final precision |
#
# This is activated with `penalty_schedule=:default_annealed` or by passing
# an explicit list of `(start_iter, end_iter, multiplier)` tuples.
#
# The penalty schedule must be carefully tuned per problem.  In contrast,
# strict mode bypasses this entirely when the problem admits an always-feasible
# policy (see above).

# ## Evaluation
#
# After training, we evaluate the policy using stage-wise rollout on held-out
# scenarios.  Two modes:
# - **Target feedback** (`policy_state=:target`): the policy receives its own
#   predicted target as input, matching DE training semantics.
# - **Realized feedback** (`policy_state=:realized`): the policy receives the
#   realized state from the solver, matching deployment semantics.
#
# The **target-violation share** measures how much cost comes from the slack
# penalty rather than actual operations — it should be small (``\le 5\%``) for
# a well-trained policy.  In strict mode, the violation share is always **zero**
# by construction.
#
# ### Paired evaluation protocol
#
# All headline numbers in the Results section come from a **paired**
# protocol: a fixed 126×500 index matrix, generated deterministically from a
# documented seed (`paired_scenario_indices` in the example's
# `load_hydropowermodels.jl`, using `StableRNGs` so the stream is identical
# on every Julia version), selects the *same* joint inflow realization at
# every stage for every method.  The protocol is therefore reproducible from
# code alone — there is no scenario data artifact to distribute.  The SDDP
# policy is simulated
# with `SDDP.Historical` on those indices (`sddp/eval_paired_sddp.jl`); every
# decision-rule checkpoint is rolled out stage-wise on the identical
# trajectories (`eval_paired_tsddr.jl` here; `eval_paired_exa_strict.jl` in
# the GPU package).  Pairing removes the between-scenario variance
# (per-scenario cost std ≈ 5,600) from the *comparison*: the standard error
# of the paired mean difference is ≈ 50, roughly two orders of magnitude
# tighter than comparing unpaired means.  Data-file identity across
# implementations is enforced (byte-identical `inflows.csv`, `hydro.json`,
# `PowerModels.json`), and the stage-index-to-inflow-row mapping (cyclic
# tiling of the 47-row record) is asserted programmatically inside the
# cross-package evaluation.

# ```julia
# rollout_eval = RolloutEvaluation(
#     subproblems, state_params_in, state_params_out, initial_state, eval_scenarios;
#     stride=1, policy_state=:realized,
# )
# rollout_eval(1, models)
# println("Operational cost: ", rollout_eval.last_objective_no_deficit)
# println("Violation share:  ", rollout_eval.last_violation_share)
# ```

# ## SDDP baseline
#
# For comparison, we also train an SDDP policy using
# [SDDP.jl](https://github.com/odow/SDDP.jl) with **inconsistent
# formulations**: a convex SOC-WR relaxation for the backward pass
# (cut generation) and the nonconvex ACP formulation for the forward
# pass (simulation).  This is a pragmatic approach when the true problem
# (AC-OPF) is nonconvex — SDDP requires convexity for valid cuts, so a
# convex relaxation approximates the value function while the forward pass
# evaluates under the true physics.
#
# The learned cuts are saved to a JSON file, which can be loaded to
# simulate the policy under the ACP formulation.  On this case the SDDP
# training ran **441 iterations in ≈ 12 hours** (CPU, MadNLP subproblem
# solver), converging its lower bound to **378,207**; the forward-pass
# (ACP) simulation cost stabilizes around **380 K** on the 126-stage
# horizon.  These two numbers frame everything below: no policy can have
# expected 126-stage cost below the bound, and SDDP's own policy sits
# roughly 0.5% above it.

# ## Results
#
# We evaluate the two **strict** TS-DDR implementations — stage-wise
# subproblems (CPU) and the full-horizon GPU deterministic equivalent —
# against the SDDP baseline on the Bolivia case with AC power flow (11
# hydro plants, 47 inflow scenarios).  The non-strict formulations (DE,
# stage-wise, multiple shooting) are available through the same API but
# require penalty scheduling and careful tuning; strict mode eliminates
# this entirely, so it is the configuration we benchmark.
#
# Three comparison disciplines keep the results honest:
#
# 1. **Same scenarios** — every number below uses the paired 100-scenario
#    protocol described above.
# 2. **Same physics at evaluation** — all rollouts solve the identical
#    ACP stage problem (hard reactive balance, mof.json objective); the
#    cross-package audit pins the implementations to each other at
#    solver-tolerance level.
# 3. **Documented training recipes** — every trained artifact corresponds
#    to a from-scratch-reproducible configuration listed in the appendix
#    and in the example READMEs (no ad-hoc checkpoints).
#
# ### Understanding the metrics
#
# - **SDDP lower bound** (126 stages): the expected-cost lower bound from
#   the convex SOC-WR relaxation.  This is a *relaxation bound* — it cannot
#   be beaten by any feasible policy.  For Bolivia, it converges to
#   approximately **378 207**.
#
# - **SDDP forward-pass cost** (126 stages): the simulation cost of the
#   SDDP policy evaluated under the true AC formulation during the forward
#   pass.  This is the 126-stage operational cost of the SDDP policy.
#
# - **Simulation cost** (96 stages): the operational cost obtained by
#   rolling out a policy under AC power flow on the 500 seeded paired inflow
#   scenarios.  This is the primary metric for policy quality.  SDDP's
#   96-stage simulation cost is **303 665** (mean over the 500 paired
#   scenarios, std 5 921).
#
# During TS-DDR training, the logged loss is a *training-batch average*
# over a small number of sampled scenarios, and periodic rollout
# evaluations use small fixed held-out sets.  Neither is comparable to the
# 500-scenario paired protocol: small evaluation sets carry offsets of
# several hundred cost units, so cross-method claims are made only on the
# paired protocol.
#
# ### Training convergence against wall-clock time (126 stages)
#
# The figure below shows all 126-stage training metrics against **wall-clock
# time** (log scale), which is the axis that exposes the computational
# trade-off between the methods:
#
# ![Training convergence vs wall-clock time](../assets/hydro_training_convergence_by_time.png)
#
# - **SDDP lower bound** (SOC-WR relaxation): converges to ~378.2 K over
#   ≈ 12 hours (441 iterations).  Dashed line — no policy can beat it.
# - **SDDP forward-pass cost**: the 126-stage simulation cost of the SDDP
#   policy under the true AC formulation, ~380 K after convergence.
# - **TS-DDR strict subproblems (CPU)**: the final two-phase stage-wise
#   schedule.  Phase 1 uses single-sample gradients and training-loss
#   checkpointing; phase 2 switches to larger batches, decayed learning
#   rate, and held-out rollout checkpointing.  The wall-clock axis counts
#   both phases.
# - **TS-DDR strict DE (GPU)**: the analogous two-phase full-horizon
#   deterministic-equivalent schedule on an H200.  One coupled 126-stage
#   solve provides each gradient sample; the second phase uses the same
#   rollout-selected, lower-variance training discipline and hard reactive
#   balance used at evaluation.
#
# In cumulative wall-clock time, the plotted two-phase schedules end at
# approximately **21.3 h** for strict subproblems and **24.2 h** for strict
# DE.  The GPU DE curve reaches the lowest TS-DDR training objective, but this
# run is not a wall-clock win over SDDP.
#
# The plot is regenerated by
# `examples/HydroPowerModels/plot_hydro_strict_convergence.jl`, which parses
# the SDDP training log and pulls the policy-training histories from the
# experiment tracker.
#
# ### Two-phase TS-DDR training protocol
#
# The reported TS-DDR policies use a single, cumulative two-phase schedule
# (exact commands in the example README):
#
# 1. **Coarse training**: train from scratch with small gradient batches and
#    checkpoint on training loss.  This phase rapidly enters the SDDP cost
#    envelope but the per-batch objective remains too noisy to rank policies
#    that differ by a few hundred cost units.
# 2. **Selection phase**: continue from the selected checkpoint with larger
#    gradient batches, a warmed-up and cosine-decayed learning rate, and
#    checkpoint selection on a fixed held-out rollout objective.  This phase
#    optimizes the same quantity used for the paired evaluation below.
#
# The plotted wall-clock time is cumulative across both phases.
#
# ### 96-stage out-of-sample rollout cost (paired, 500 seeded scenarios)
#
# The primary evaluation metric is the **96-stage simulation cost** —
# total dispatch cost under AC power flow on the 500 paired inflow
# trajectories of the seeded protocol.
#
# | Method | Policy | Mean Cost | Std | Target violations | Training |
# |:-------|:------:|----------:|----:|:-----------------:|:---------|
# | SDDP (SOC-WR / ACP) | cuts | 303 665 | 5 921 | — | CPU, ≈ 12 h |
# | **TS-DDR strict DE (warm-continued)** | LSTM + reachable | 303 936 | 6 119 | 0.0% | H200 GPU, 22 h + 11 h |
# | **TS-DDR strict subproblems (two-stage)** | LSTM + reachable | 304 027 | 6 114 | 0.0% | CPU (Ipopt), ≈ 2 d + 21 h |
# | **TS-DDR strict DE (from scratch, SDDP-matched budget)** | LSTM + reachable | 304 460 | 6 052 | 0.0% | H200 GPU, 12.3 h |
#
# Because the protocol is paired, differences are measured per scenario and
# their standard errors are two orders of magnitude below the cost std:
#
# - Best TS-DDR (warm-continued GPU DE) − SDDP: **+270 ± 19**
#   (``t \\approx 14.5``); TS-DDR dispatches cheaper than SDDP on **17.4%**
#   of scenarios.
# - From-scratch GPU DE at SDDP's own ≈ 12 h training budget − SDDP:
#   **+794 ± 19** (win rate 4.2%).
#
# The verdict on this benchmark is symmetric and honest: SDDP retains a
# statistically significant but operationally tiny advantage — **0.09%**
# against the best TS-DDR policy — while TS-DDR guarantees zero target
# violations by construction, requires no penalty tuning, and reaches
# within 0.26% of SDDP from scratch in the same wall-clock budget on one
# GPU.  (On the inventory-control example, the same strict construction
# beats its SDDP baseline outright; see that example's page.)
#
# ### Cost distributions on the paired scenario set
#
# Means compress the comparison; the full per-scenario distributions show
# it.  The top panel overlays each method's cost density over the shared
# paired scenarios (kernel density estimates of the per-scenario rollout
# costs; dashed lines mark means).  The bottom panel is the statistically
# decisive view for paired data: the density of the **per-scenario paired
# difference** ``\\text{method} - \\text{SDDP}``.  Pairing removes the
# common between-scenario variance (per-scenario cost std ≈ 5.6 K vs
# paired-difference std ≈ 0.5 K), so this panel resolves differences an
# order of magnitude smaller than the raw distributions can — probability
# mass left of the zero line is exactly the fraction of scenarios where the
# method dispatches cheaper than SDDP (the win rate annotated per series).
#
# ![Paired cost distributions](../assets/hydro_paired_cost_distributions.png)
#
# The figure is regenerated by
# `examples/HydroPowerModels/plot_hydro_paired_distributions.jl` from the
# tagged outputs of the paired evaluation scripts.
#
# The key advantage of strict mode is that it requires **no penalty tuning**:
# the gradient signal from the hard-equality duals avoids soft-deficit or
# reactive-balance penalty schedules.  The two-phase protocol above uses only
# ordinary optimizer scheduling and held-out rollout checkpointing; the
# non-strict formulations require careful penalty schedules to achieve
# competitive results.
#
# ### Comparing the two strict implementations fairly
#
# The GPU DE trainer optimizes the same strict target-setting policy class,
# and the audit above shows its inner solve is numerically interchangeable
# with the CPU stage-wise solve.  The table therefore compares policies under
# the same 96-stage paired rollout, hard reactive balance, and mof.json
# operating cost.  The practical distinction is computational: the CPU path
# solves 126 small Ipopt subproblems sequentially per gradient sample, while
# the GPU path solves one coupled 126-stage MadNLP/cuDSS problem per sample.
#
# The current result is encouraging but not final for the "fastest method"
# ambition: the GPU DE schedule now reaches the best TS-DDR cost, but it has
# not yet delivered a policy that both beats SDDP and does so in less
# wall-clock time.  Closing that remaining gap requires improving the DE
# training schedule or gradient estimator, not changing the evaluation
# protocol.

# ## Appendix: experimental details
#
# Everything below is reproducible from the scripts in
# `examples/HydroPowerModels/` (this package) and the same-named folder of
# DecisionRulesExa.jl; the exact commands are in the two READMEs.
#
# ### A.1 Problem instance
#
# | Quantity | Value |
# |:---------|:------|
# | Network | Bolivia, 28 buses, 34 generators, AC polar (ACPPowerModel) |
# | Hydro plants | 11 (3 cascade links: 1 turbine-only, 2 turbine+spill) |
# | Load scaling | 0.6 × PowerModels.json loads (baked into the mof.json) |
# | Deficit cost | 6,000 per pu (= 60 $/MWh × baseMVA 100) |
# | Inflow record | 47 monthly joint scenarios, tiled cyclically beyond month 47 |
# | Training horizon | 126 stages |
# | Evaluation horizon | 96 stages, 500 seeded paired scenarios |
# | Water-balance factor | K = 0.0036 (flow → volume) |
#
# The 126/96 split is deliberate for **both** SDDP and TS-DDR: training on a
# longer horizon buffers end-of-horizon effects out of the reported window
# (SDDP additionally trains with 30 extra stages for the same reason).
#
# ### A.2 SDDP baseline
#
# | Setting | Value |
# |:--------|:------|
# | Cut generation | SOC-WR relaxation (convex), SDDP.jl |
# | Forward simulation | ACP (nonconvex, true physics) |
# | Subproblem solver | MadNLP (CPU) |
# | Iterations / wall time | 441 / ≈ 12 h |
# | Final lower bound | 378,207 (126 stages) |
# | Paired simulation | `SDDP.Historical` on the shared index matrix |
#
# ### A.3 TS-DDR strict subproblems (CPU) — phase 1
#
# | Setting | Value |
# |:--------|:------|
# | Policy | `HydroReachablePolicy`: LSTM encoder [128, 128] over inflows; linear sigmoid head over `[encoding; state]` |
# | Stage solver | Ipopt (MUMPS), wrapped in `DiffOpt.diff_optimizer` |
# | Optimizer | Adam, constant ``10^{-3}``, no gradient clipping |
# | Samples per gradient step | 1 |
# | Iteration budget | 8,000 (80 epochs × 100 batches), stalling criterion |
# | Checkpoint selection | training-batch loss |
# | Rollout evaluation | every 25 iterations, 4 fixed held-out scenarios |
# | Penalty schedule | none (strict) |
# | Seeds | 8788 (initial evaluation), 8789 (held-out scenarios) |
#
# ### A.4 TS-DDR strict subproblems (CPU) — phase 2
#
# | Setting | Value |
# |:--------|:------|
# | Initialization | selected phase-1 checkpoint (`DR_PRETRAINED_MODEL`) |
# | Samples per gradient step | 16 (`DR_NUM_TRAIN_PER_BATCH`) |
# | Learning rate | ``10^{-4} \to 10^{-5}`` cosine decay, 50-iteration linear warmup |
# | Checkpoint selection | held-out rollout objective (`DR_SAVE_METRIC=rollout`) |
# | Held-out set | 24 fixed scenarios (`DR_NUM_EVAL_SCENARIOS`) |
# | Iteration budget | 1,500 |
#
# ### A.5 TS-DDR strict DE (GPU)
#
# | Setting | Value |
# |:--------|:------|
# | Package | DecisionRulesExa.jl (ExaModels + MadNLP + cuDSS) |
# | Hardware | 1 × NVIDIA H200 |
# | Formulation | AC polar, strict targets, reservoir trajectory as parameter |
# | Policy | same `HydroReachablePolicy` family; encoder [128, 128]; head [256, 256] |
# | Phase 1 optimizer | Adam ``10^{-3}``, one sampled trajectory per gradient step |
# | Phase 2 optimizer | Adam ``10^{-4} \to 10^{-5}``, 50-iteration linear warmup, 4 sampled trajectories per gradient step |
# | Solver settings | tol ``10^{-6}``, max_iter 9,000, dual-snapshot warm starts |
# | Checkpoint selection | phase 1: training-batch loss; phase 2: held-out rollout objective |
# | Iteration budget | phase 1: 8,000; phase 2: 800 |
# | Deficit cost (training) | ``10^{5}`` (inert: deficit never activates; evaluation uses the mof.json 6,000) |
# | Reactive balance | phase 2 and evaluation use hard reactive balance (`reactive_deficit_cost = Inf`) |
# | Recurrent state | threaded explicitly across stages (see warning above) |
#
# ### A.6 Software
#
# Julia 1.10/1.12 CI matrix; JuMP + DiffOpt + Ipopt (CPU path); ExaModels +
# MadNLP + cuDSS (GPU path); Flux 0.16 (explicit recurrent-state threading);
# SDDP.jl for the baseline.  Package versions are pinned in the respective
# `Manifest.toml` files.
#
# ### A.7 Artifact-to-script map
#
# | Artifact | Produced by |
# |:---------|:------------|
# | Strict CPU training runs | `train_dr_hydropowermodels_strict.jl` (stage 1 & 2 via env recipes) |
# | Strict GPU training runs | `train_hydro_exa_strict.jl` (DecisionRulesExa.jl) |
# | SDDP cuts + log | `sddp/run_sddp.jl` |
# | Paired SDDP simulation | `sddp/eval_paired_sddp.jl` |
# | Paired CPU-policy evaluation | `eval_paired_tsddr.jl` |
# | Paired GPU-policy evaluation | `eval_paired_exa_strict.jl` (DecisionRulesExa.jl) |
# | Cross-package equivalence audit | `dump_paired_policy_reference.jl` + `compare_paired_evals.jl` |
# | Convergence figures | `plot_hydro_strict_convergence.jl` |
