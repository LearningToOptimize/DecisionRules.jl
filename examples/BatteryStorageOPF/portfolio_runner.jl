#!/usr/bin/env julia
# portfolio_runner.jl
#
# The production runner for the two methods THIS engine owns: `:sddp_soc` and
# `:sddp_dc`. The other two study methods, `:tsddr_nonlinear` and
# `:tsldr_recurrent_linear`, belong to the Exa engine
# (`DecisionRulesExa.jl/examples/BatteryStorageOPF/portfolio_runner.jl`) and are
# refused here BY NAME, exactly as `run_battery_method` refuses them.
#
# WHAT THIS FILE IS, AND WHAT IT IS NOT.
# It is a SEGMENT DRIVER: it turns "train this frozen case, with this frozen
# configuration, from global iteration `a` to global iteration `b`, and survive
# being killed at any moment" into files on disk. It builds no policy graph
# equation, writes no cut, chooses no duality handler and defines no cost — every
# one of those comes from `battery_sddp.jl`, `battery_powermodels.jl` and the two
# shared contract files, unchanged, and through SDDP.jl's own stock mechanisms.
# What it adds is identity binding, verified checkpoints, resumption, a stop
# protocol and an honest result record.
#
# ─────────────────────────────────────────────────────────────────────────────
# COMMAND CONTRACT
#
#   julia --project=<this directory> portfolio_runner.jl \
#       --case-manifest <case_manifest.json> \
#       --method        <sddp_soc|sddp_dc> \
#       --config        <frozen config .toml> \
#       --protocol      <protocol descriptor .toml> \
#       --output        <segment output directory> \
#       --resume-from   <checkpoint path, or the literal string "none">
#
# Those six flags are sufficient on their own; the command above runs with no
# scheduler and no campaign controller of any kind. Five further flags exist
# purely as conveniences for an automated caller and ALL of them default:
#
#   --run-id <string>      (default "standalone")
#   --segment <int>        (default 1)
#   --attempt <int>        (default 1)
#   --stop-file <path>     (default <output>/STOP)
#   --max-seconds <float>  (default 1e9)
#
# ─────────────────────────────────────────────────────────────────────────────
# THE FROZEN CONFIGURATION  (`--config`, TOML)
#
#   target_index      total SDDP iterations for the WHOLE run
#   segment_updates   iterations this invocation may add at most
#   checkpoint_every  iterations between checkpoints — also the CHUNK size, see
#                     "continuation is the only path" below
#   eval_every        iterations between true-ACP evaluations on the fixed panel
#   ma_window         window of the reported moving average of the forward cost
#   num_stages        horizon T
#   eval_columns      screening-protocol columns forming the fixed panel
#   seed              the run's single seed
#   max_recourse      physical admissibility tolerance, pu
#   method            OPTIONAL; when present it must equal --method
#
# ─────────────────────────────────────────────────────────────────────────────
# THE PROTOCOL DESCRIPTOR  (`--protocol`, TOML) — see `resolve_protocol`
#
#   kind          "screening" (or "sole" on a fixture case that declares only
#                 one protocol). "final" is REFUSED.
#   num_stages, num_scenarios, seed, sha256
#
# Write one for a case with:
#
#   julia --project=. -e 'include("portfolio_runner.jl");
#       write_protocol_descriptor("case/pglib_opf_case118_ieee", "screening.toml")'
#
# ─────────────────────────────────────────────────────────────────────────────
# CONTINUATION IS THE ONLY PATH
#
# This runner trains in CHUNKS of `checkpoint_every` iterations. Every chunk
# rebuilds both policy graphs from the frozen case and restores the previous
# chunk's cuts through `SDDP.read_cuts_from_file`, whether or not the process was
# ever interrupted. So the code path a resumed run takes is not "the same as"
# the uninterrupted one — it IS the uninterrupted one, and a run cut at any
# checkpoint boundary reproduces the uncut run structurally rather than by
# careful arrangement. The price is a graph rebuild per checkpoint, which is
# small beside `checkpoint_every` ACP forward passes and buys the property this
# campaign's whole recovery design rests on.
#
# Sampling is made a function of the GLOBAL iteration index the same way: each
# chunk's seed is derived from `(seed, iterations already completed)`, so the
# scenarios of iterations 4–6 are the same whether they are the tail of one run
# of six or the whole of a resumed run of three.
#
# No solver object is serialized anywhere, and no SDDP internal is parsed by
# hand: cuts leave and re-enter through SDDP.jl's own reader and writer.
#
# ─────────────────────────────────────────────────────────────────────────────
# OUTPUTS, all inside `--output`, none of them ever committed
#
#   checkpoints/ck_XXXXXXXX.<tag>.json        the checkpoint payload
#   checkpoints/ck_XXXXXXXX.<tag>.json.meta.toml   digest, indices, lineage
#   history.csv       index, train_loss, panel_value, bound, solve_ok,
#                     solve_fail, deficit                    (7 columns, fixed)
#   trajectory.csv    index, forward_cost, forward_cost_ma<w>, bound, wall_seconds
#   evaluation.csv    index, protocol, columns, acp_mean, acp_raw_mean,
#                     worst_recourse, complete, selected, best_cost
#   result.toml       the segment record
#   identity.toml     every coordinate this segment was bound to
#
# `acp_mean` is the CORRECTED physical cost under the shared cost contract,
# validated and projected PER BUS before anything is summed — the same estimand
# the Exa engine selects on. `acp_raw_mean` is the solver's raw objective sum,
# a diagnostic only.
#
# `train_loss` is SDDP's own per-iteration forward-pass cost — the ACP cost of
# ONE sampled scenario, a per-iteration sample of a random objective. It is not
# comparable with the fixed-panel evaluation, which is a mean over the same
# columns every time, and the two live in different columns for that reason.
#
# THE `bound` COLUMN IS ARM-DEPENDENT AND IS NOT ALWAYS A BOUND. For `:sddp_soc`
# it is the SOC-WR relaxation bound and it does bound the true ACP problem; for
# `:sddp_dc` it is the DC-approximation training bound and it bounds NOTHING
# about ACP in either direction. `bound_name` and `bound_bounds_acp` travel in
# the result and in every checkpoint so no consumer has to infer which one it is
# holding.

using TOML
using SHA
using Dates
using Printf
using Random
using Statistics
using JSON
using SDDP

# The certified implementation. Everything scientific comes from here; this file
# adds no second copy of any of it. Its `PROGRAM_FILE` guard keeps the include
# from launching the construction smoke.
include(joinpath(@__DIR__, "battery_sddp.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

"""
Result-record schema. Must match the number the campaign controller verifies;
a runner speaking a different one is rejected rather than half-read.
"""
const RUNNER_RESULT_SCHEMA = 1

"""
Segment-checkpoint schema.

The same tag the Exa runner writes, because the two produce the same KIND of
object — a self-contained continuation point for one segmented run — even though
their payloads are a policy and a cut set.
"""
const RUNNER_CHECKPOINT_SCHEMA = "battery_storage_opf/segment/1"

"The two methods this engine owns, and the engine that owns the other two."
const RUNNER_METHODS = (:sddp_soc, :sddp_dc)

# ─────────────────────────────────────────────────────────────────────────────
# Small self-contained primitives
#
# Deliberately reimplemented here rather than shared with any caller: this file
# must run standalone, from a public checkout, with no orchestration package on
# the load path. `sha256_file` is the one exception — it already exists in
# `battery_case.jl`, which this file includes, and defining a second one would
# leave two digest functions that could drift apart.
# ─────────────────────────────────────────────────────────────────────────────

"UTC timestamp in the one format every record in this campaign uses."
utcnow() = Dates.format(now(UTC), dateformat"yyyy-mm-dd\THH:MM:SS\Z")

"SHA-256 of a byte buffer or a string, as lowercase hex."
sha256_hex(data::Vector{UInt8}) = bytes2hex(sha256(data))
sha256_hex(s::AbstractString) = bytes2hex(sha256(codeunits(String(s))))

"""
    atomic_write(path, data) -> String

Write `data` to a sibling temporary file, flush it, `fsync` it, `rename(2)` it
onto `path`, then read it back and re-hash it. Returns the digest.

# Notes
A file half-written when the node dies is never visible under its final name,
which is the whole reason a controller may trust any file it finds. The read-back
is not paranoia about `rename`: it catches a full filesystem and a silently
truncated write, both of which this project has seen.
"""
function atomic_write(path::AbstractString, data::Vector{UInt8})
    mkpath(dirname(abspath(path)))
    tmp = string(path, ".tmp.", getpid(), ".", time_ns())
    open(tmp, "w") do io
        write(io, data)
        flush(io)
        try
            ccall(:fsync, Cint, (Cint,), fd(io))
        catch
            # fsync is a durability optimisation here, not a correctness one:
            # the rename is what makes the file atomic. A filesystem that
            # refuses it must not take the run down.
        end
    end
    mv(tmp, path; force = true)
    got = sha256_file(path)
    want = sha256_hex(data)
    got == want || error("atomic_write verification failed for $path")
    return got
end
atomic_write(p::AbstractString, s::AbstractString) =
    atomic_write(p, Vector{UInt8}(codeunits(String(s))))

"Serialize a dictionary to TOML and write it atomically. Returns the digest."
function write_toml_atomic(path::AbstractString, d::AbstractDict)
    buf = IOBuffer()
    TOML.print(buf, d; sorted = true)
    return atomic_write(path, take!(buf))
end

"""
    runner_code_digest(dir) -> String

SHA-256 over the sorted `(relative path, file digest)` list of every `.jl` file
beside this runner.

# Notes
Recorded in the result so a number can be tied to the exact code that produced
it even when the checkout was dirty — which, during a study, it usually is. The
git commit is recorded too, and the two answer different questions: the commit
says which revision was checked out, the digest says what was actually run.
"""
function runner_code_digest(dir::AbstractString)
    rows = String[]
    for (root, _, files) in walkdir(dir)
        occursin("/.git", root) && continue
        for f in files
            endswith(f, ".jl") || continue
            full = joinpath(root, f)
            push!(rows, string(relpath(full, dir), " ", sha256_file(full)))
        end
    end
    sort!(rows)
    return sha256_hex(join(rows, "\n"))
end

"The git commit of `dir`, or `\"none\"` outside a repository."
function git_commit(dir::AbstractString)
    try
        return strip(read(`git -C $dir rev-parse HEAD`, String))
    catch
        return "none"
    end
end

"""
    parse_args(args) -> Dict{String,String}

`--key value` / `--flag` parser.

# Notes
Unknown flags are KEPT rather than rejected, so an automated caller may pass
extras; but no flag outside the documented six is ever REQUIRED, which is what
keeps the six-flag command a complete command.
"""
function parse_args(args)
    d = Dict{String,String}()
    i = 1
    while i <= length(args)
        if startswith(args[i], "--")
            k = args[i][3:end]
            if i < length(args) && !startswith(args[i+1], "--")
                d[k] = args[i+1]
                i += 2
            else
                d[k] = "true"
                i += 1
            end
        else
            i += 1
        end
    end
    return d
end

"Read an integer vector from a TOML value that may be a list or a single number."
_int_list(v) = v isa AbstractVector ? [Int(x) for x in v] : [Int(v)]

"""
    chunk_seed(seed, completed) -> Int

The seed of the chunk that starts at global iteration `completed`.

``s_k = \\mathrm{hash}(seed, k) \\bmod 2^{31}``

# Notes
Deriving the seed from the GLOBAL index is what makes the sampling of an
iteration independent of the segmentation: iterations 4–6 face the same
scenarios whether they are the tail of one call or the whole of a resumed one.
Seeding once per run and letting the stream carry across chunks would not — a
resumed process starts with a fresh global RNG no matter what.
"""
chunk_seed(seed::Integer, completed::Integer) =
    Int(hash((Int(seed), Int(completed))) % 0x7fffffff)

# ─────────────────────────────────────────────────────────────────────────────
# The protocol descriptor
# ─────────────────────────────────────────────────────────────────────────────

"""
    write_protocol_descriptor(case_dir, out_path; kind=:screening) -> String

Write the protocol descriptor a run is launched against, and return its path.

# Notes
Generated from the frozen case itself: the `kind`, the shape and the digest are
copied out of the case manifest, so a descriptor cannot describe a protocol the
case does not declare. `:final` is refused here as well as at load time — a
descriptor naming the fresh panel should not exist in the first place.
"""
function write_protocol_descriptor(case_dir::AbstractString, out_path::AbstractString;
                                   kind::Symbol = :screening)
    kind === :final && error("refusing to write a descriptor for the FINAL protocol")
    case = read_battery_case(case_dir)
    scr = get(case.manifest, "screening", nothing)
    block, resolved = scr === nothing ? (case.manifest["protocol"], "sole") : (scr, "screening")
    kind === :screening || String(kind) == resolved ||
        error("case $(case.name) declares a $resolved protocol, not a $kind one")
    write_toml_atomic(out_path, Dict{String,Any}(
        "kind"          => resolved,
        "case"          => case.name,
        "num_stages"    => Int(block["num_stages"]),
        "num_scenarios" => Int(block["num_scenarios"]),
        "seed"          => Int(block["seed"]),
        "sha256"        => String(block["sha256"]),
        "written_utc"   => utcnow(),
    ))
    return out_path
end

"""
    resolve_protocol(case, descriptor_path) -> (matrix, kind, declared)

Regenerate the evaluation protocol and bind it to the descriptor, fail-closed.

# Returns
`(matrix, kind, declared)` — the `(stages × scenarios)` atom-index matrix, the
kind [`evaluation_protocol`](@ref) actually produced, and the descriptor as read.

# Notes
FOUR refusals, in this order, and every one of them happens before a single
scenario outcome is computed:

 1. a descriptor whose `kind` is `"final"` — training may never be selected on
    the fresh panel, and the refusal must not depend on noticing it later;
 2. a descriptor whose kind disagrees with what the case declares;
 3. a shape that disagrees with the regenerated matrix;
 4. a digest that disagrees with the regenerated protocol's.

The last one is the load-bearing check. `evaluation_protocol` already re-derives
the screening protocol from the frozen support and re-verifies it against the
manifest; the descriptor adds the statement that THIS RUN was launched against
that protocol and not another, which is the part a result file can be audited on
afterwards.
"""
function resolve_protocol(case::BatteryCase, descriptor_path::AbstractString)
    isfile(descriptor_path) || error("no protocol descriptor at $descriptor_path")
    d = TOML.parsefile(descriptor_path)
    declared = String(get(d, "kind", ""))
    declared == "final" && error(
        "protocol descriptor $descriptor_path declares the FINAL protocol; " *
        "training may only be selected on the screening protocol")
    declared in ("screening", "sole") || error(
        "protocol descriptor $descriptor_path declares kind $(repr(declared)); " *
        "expected \"screening\" (or \"sole\" on a fixture case)")

    matrix, kind = evaluation_protocol(case)
    String(kind) == declared || error(
        "protocol descriptor declares $declared but the case regenerates a $kind protocol")

    block = kind === :sole ? case.manifest["protocol"] : case.manifest["screening"]
    Int(get(d, "num_stages", -1)) == Int(block["num_stages"]) ||
        error("protocol descriptor stage count does not match the case")
    Int(get(d, "num_scenarios", -1)) == Int(block["num_scenarios"]) ||
        error("protocol descriptor scenario count does not match the case")
    String(get(d, "sha256", "")) == String(block["sha256"]) ||
        error("protocol descriptor digest does not match the case's $kind protocol")
    size(matrix, 2) == Int(block["num_scenarios"]) ||
        error("regenerated protocol has $(size(matrix, 2)) columns, not $(block["num_scenarios"])")
    return matrix, kind, d
end

# ─────────────────────────────────────────────────────────────────────────────
# Identity
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_identity(; manifest_path, case, method, config_path, conf,
                   protocol_path, protocol_kind, num_stages, backward) -> Dict

Every coordinate that makes two runs scientifically different, plus one digest
over all of them.

# The coordinates
| field | why it is here |
|---|---|
| `case_manifest_sha256` | the manifest FILE, byte for byte |
| `case_content_sha256` | the case CONTENT: the manifest's own artifact digests, so an edited manifest pointing at the same artifacts is still caught, and so is the reverse |
| `method` | which of the study's four this is |
| `config_sha256` | the frozen configuration file |
| `protocol_sha256`, `protocol_kind` | which panel selection may look at |
| `horizon` | the stage count actually trained |
| `seed` | the run's single seed |
| `formulation` | the backward PowerModels type, spelled out |
| `acp_bound_relax_factor` | the common true-ACP setting both engines state explicitly |

# Notes
`identity_sha256` is a digest of the canonical `key=value` rendering of the
others. It is written into every checkpoint and re-derived on resume; a
mismatch on ANY coordinate refuses the resume rather than continuing a run whose
meaning changed underneath it.
"""
function run_identity(; manifest_path, case, method, config_path, conf,
                        protocol_path, protocol_kind, num_stages, backward)
    spec = backward_spec(backward)
    arts = case.manifest["artifacts"]
    content = join([string(k, "=", arts[k]) for k in sort!(collect(keys(arts)))], ";")

    id = Dict{String,Any}(
        "case"                   => case.name,
        "case_manifest"          => abspath(manifest_path),
        "case_manifest_sha256"   => sha256_file(manifest_path),
        "case_content_sha256"    => sha256_hex(content),
        "method"                 => String(method),
        "config_sha256"          => sha256_file(config_path),
        "protocol_sha256"        => sha256_file(protocol_path),
        "protocol_kind"          => String(protocol_kind),
        "horizon"                => Int(num_stages),
        "seed"                   => Int(conf["seed"]),
        "formulation"            => string(spec.formulation),
        "forward_formulation"    => string(FORWARD_FORMULATION),
        "engine"                 => "jump",
        "acp_bound_relax_factor" => ACP_BOUND_RELAX_FACTOR,
        "checkpoint_schema"      => RUNNER_CHECKPOINT_SCHEMA,
    )
    id["identity_sha256"] = sha256_hex(join(
        [string(k, "=", id[k]) for k in sort!(collect(keys(id)))], "\n"))
    return id
end

"""
    assert_identity(want, got, whence)

Refuse a continuation whose identity differs from this segment's, naming the
first field that differs.

# Notes
Reporting the FIELD matters. "identity mismatch" sends a reader to diff two
hashes; "seed 1 vs 2" ends the investigation.
"""
function assert_identity(want::AbstractDict, got::AbstractDict, whence::AbstractString)
    for k in sort!(collect(keys(want)))
        k == "identity_sha256" && continue
        haskey(got, k) || error("$whence is missing the identity field `$k`")
        got[k] == want[k] || error(
            "$whence identity mismatch on `$k`: checkpoint has $(repr(got[k])), " *
            "this segment has $(repr(want[k]))")
    end
    String(get(got, "identity_sha256", "")) == String(want["identity_sha256"]) ||
        error("$whence identity digest mismatch")
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoints
# ─────────────────────────────────────────────────────────────────────────────

"""
    checkpoint_paths(output, index, tag) -> (payload, sidecar)

`checkpoints/ck_XXXXXXXX.<tag>.json` and its `.meta.toml`.

# Notes
Zero-padded so a listing sorts in run order, and TAGGED so a SOC checkpoint and
a DC checkpoint of the same case at the same iteration cannot be confused by
anything as weak as a `readdir` — the same rule [`sddp_cut_path`](@ref) enforces
on cut files, for the same reason.
"""
function checkpoint_paths(output::AbstractString, index::Integer, tag::AbstractString)
    dir = joinpath(output, "checkpoints")
    name = @sprintf("ck_%08d.%s.json", index, tag)
    return joinpath(dir, name), joinpath(dir, name * ".meta.toml")
end

"""
    write_segment_checkpoint(output, index, cuts, state, ident; kind) -> (path, sha)

Write one self-contained, verified, monotonically numbered checkpoint.

# What it preserves
All cuts, as SDDP.jl's own `write_cuts_to_file` serialized them, embedded
verbatim; the completed iteration count; the sampling position (the seed and the
count the next chunk's seed is derived from); the convergence and evaluation
histories; and the best admissible true-ACP forward result with the iteration
that produced it.

# Notes
ORDER MATTERS, and it is the same order the controller assumes. The payload is
written atomically and hashed FIRST; only then is the sidecar written naming
that digest. A crash between the two leaves a payload with no sidecar, which is
ignored; a crash before either leaves nothing.

The cuts are EMBEDDED rather than referenced. A checkpoint that pointed at a
sibling cut file would be two files the controller hashes as one, and the pair
could be separated by any of the ways this filesystem loses things.
"""
function write_segment_checkpoint(output::AbstractString, index::Integer,
                                  cuts, state::NamedTuple, ident::AbstractDict;
                                  kind::AbstractString = "periodic")
    path, meta_path = checkpoint_paths(output, index, state.tag)
    body = Dict{String,Any}(
        "segment_schema"       => RUNNER_CHECKPOINT_SCHEMA,
        "method"               => ident["method"],
        "backward"             => String(state.backward),
        "tag"                  => state.tag,
        "global_index"         => Int(index),
        "iterations_completed" => Int(index),
        "seed"                 => Int(state.seed),
        "next_chunk_seed"      => chunk_seed(state.seed, index),
        "bound"                => state.bound,
        "bound_name"           => state.bound_name,
        "bound_bounds_acp"     => state.bound_bounds_acp,
        # `null` when no complete evaluation has been admissible yet. JSON has no
        # infinity, so the sentinel is an absent value and not a magic number.
        "best_cost"            => isfinite(state.best_cost) ? state.best_cost : nothing,
        "best_index"           => Int(state.best_index),
        "trajectory_checksum"  => state.checksum,
        "convergence"          => state.convergence,
        "evaluations"          => state.evaluations,
        "identity"             => Dict{String,Any}(ident),
        "written_utc"          => utcnow(),
        "cuts"                 => cuts,
    )
    sha = atomic_write(path, JSON.json(body))

    write_toml_atomic(meta_path, Dict{String,Any}(
        "file"          => basename(path),
        "sha256"        => sha,
        "parent_sha256" => state.parent_sha,
        "global_index"  => Int(index),
        "index_from"    => Int(state.index_from),
        "run_id"        => state.run_id,
        "segment"       => Int(state.segment),
        "attempt"       => Int(state.attempt),
        "kind"          => kind,
        "written_utc"   => utcnow(),
    ))
    return path, sha
end

"""
    load_segment_checkpoint(path) -> Dict

Read a parent checkpoint, refusing it unless its sidecar digest matches the
payload on disk and it carries this file's segment schema.

# Notes
The controller verifies checkpoints too. This check is not redundant with it: a
worker resuming from a file nobody re-hashed since the scan would be trusting a
window it cannot see into, and the two ends enforce the rule independently so
neither has to assume the other ran.
"""
function load_segment_checkpoint(path::AbstractString)
    isfile(path) || error("--resume-from does not exist: $path")
    meta_path = path * ".meta.toml"
    if isfile(meta_path)
        side = TOML.parsefile(meta_path)
        got = sha256_file(path)
        got == String(get(side, "sha256", "")) || error(
            "parent checkpoint digest mismatch: $path has $got, its sidecar claims " *
            "$(get(side, "sha256", "missing"))")
    end
    ck = JSON.parsefile(path)
    String(get(ck, "segment_schema", "")) == RUNNER_CHECKPOINT_SCHEMA || error(
        "parent checkpoint $path carries segment schema " *
        "$(repr(get(ck, "segment_schema", missing))) but this runner writes " *
        "$RUNNER_CHECKPOINT_SCHEMA")
    haskey(ck, "identity") || error(
        "parent checkpoint $path carries no identity record; it was not written " *
        "by this runner and cannot be continued")
    return ck
end

"""
    materialize_cuts(dir, cuts, tag) -> String

Write an embedded cut set back out as the standalone JSON file
`SDDP.read_cuts_from_file` expects, and return its path.

# Notes
The file name carries the arm's tag, so `train_battery_sddp`'s own refusal —
which rejects a cut path that does not name the formulation — applies to a
resume exactly as it applies to a write. Round-tripping through SDDP's own
format, rather than reconstructing cut objects, is what keeps this file free of
any knowledge of how a cut is represented.
"""
function materialize_cuts(dir::AbstractString, cuts, tag::AbstractString)
    mkpath(dir)
    path = joinpath(dir, "resume.$tag.cuts.json")
    atomic_write(path, JSON.json(cuts))
    return path
end

# ─────────────────────────────────────────────────────────────────────────────
# The segment
# ─────────────────────────────────────────────────────────────────────────────

"""
    moving_average(v, w) -> Vector{Float64}

Trailing moving average of window `w`, defined from the first sample:

``\\mathrm{ma}_i = \\frac{1}{\\min(i,w)} \\sum_{j=\\max(1,i-w+1)}^{i} v_j``

# Notes
SDDP's per-iteration forward cost is one sample of a random objective on one
sampled scenario. It is a different estimand from the fixed-panel evaluation and
is never compared with it; only its moving average is legible, so both are
written and the raw column is kept beside it.
"""
function moving_average(v::AbstractVector, w::Integer)
    n = length(v)
    out = zeros(Float64, n)
    s = 0.0
    for i in 1:n
        s += v[i]
        i > w && (s -= v[i-w])
        out[i] = s / min(i, w)
    end
    return out
end

"""
    evaluate_acp_panel(case, trained, matrix, columns; max_recourse) -> NamedTuple

The true-ACP forward cost of the current policy on the fixed panel, under the
shared physical cost contract.

# Returns
`(mean_cost, costs, worst_recourse, complete, raw_mean)`.

# Notes
SDDP's own `Historical` replay of the panel's columns through the ACP graph
([`simulate_battery_sddp_on`](@ref)) — the same simulation
`train_battery_sddp`'s periodic callback and `cost_report` use — scored by
[`sddp_physical_path_cost`](@ref), which puts every stage through
`physical_stage_cost`, the byte-identical contract the Exa engine also selects
on.

**Element by element, and in this order: validate, project, then sum.** The
recorders store the per-bus recourse raw; the contract projects each individual
value within `PHYSICAL_RECOURSE_TOL` to exactly zero, refuses the stage if any
individual value is outside it, and only then are the charges summed. An
aggregate test is not a weaker version of this — it is a different and wrong
test, because tolerance-scale residues of opposite sign at different buses
cancel in a total.

`raw_mean` is the sum of the solver's raw stage objectives, carried as a
diagnostic. It is never the headline and never the selection metric: the raw
objective carries the barrier residue of whatever `bound_relax_factor` the solve
ran at, and selecting on it would select on a solver setting.
"""
function evaluate_acp_panel(case::BatteryCase, trained, matrix::AbstractMatrix{<:Integer},
                            columns::AbstractVector{<:Integer}; max_recourse::Real = 1e-6)
    ids = [b.index for b in case.batteries]
    T = trained.num_stages
    sims = simulate_battery_sddp_on(trained, matrix; ids = ids, columns = columns)
    phys = [sddp_physical_path_cost(case, s, T; tol = max_recourse) for s in sims]
    costs = [p.cost for p in phys]
    raw = [sddp_path_cost(case, s, T) for s in sims]
    worst = maximum(p.worst_recourse for p in phys; init = 0.0)
    complete = length(costs) == length(columns) && all(isfinite, costs) &&
               all(p.admissible for p in phys)
    return (mean_cost = isempty(costs) ? NaN : mean(costs), costs = costs,
            worst_recourse = worst, complete = complete,
            raw_mean = isempty(raw) ? NaN : mean(raw))
end

"""
    run_segment(a) -> Int

Drive one segment: bind identity, resume or start, train to the segment's stop
index or until asked to stop, and write a verified checkpoint and an honest
result. Returns a process exit code.

# The loop, one chunk
Rebuild both policy graphs from the frozen case, restore the previous chunk's
cuts, run `checkpoint_every` stock SDDP iterations against the seed derived from
the global index, write out the cuts, and checkpoint. The per-iteration bound
and forward cost come from SDDP's own training log, so recording the trajectory
costs no extra solve.

# Stopping
The stop file is polled after every COMPLETE chunk — i.e. after a checkpoint
exists for the work just done. On a stop request an honest `preempted` result is
written and the process exits 0. `complete` is reported only when the configured
target index was actually reached.
"""
function run_segment(a::AbstractDict)
    t_start = time()

    # ---- the six scientific flags -----------------------------------------
    manifest_path = abspath(a["case-manifest"])
    method        = Symbol(a["method"])
    config_path   = abspath(a["config"])
    protocol_path = abspath(a["protocol"])
    output        = abspath(a["output"])
    resume        = get(a, "resume-from", "none")
    resume = (resume == "none" || isempty(resume)) ? "none" : abspath(resume)

    # ---- controller conveniences, every one defaulted ----------------------
    run_id    = get(a, "run-id", "standalone")
    segment   = parse(Int, get(a, "segment", "1"))
    attempt   = parse(Int, get(a, "attempt", "1"))
    stop_file = abspath(get(a, "stop-file", joinpath(output, "STOP")))
    max_secs  = parse(Float64, get(a, "max-seconds", "1e9"))

    # ---- method ownership, before anything is loaded -----------------------
    if !(method in RUNNER_METHODS)
        haskey(BATTERY_METHODS, method) || error(
            "unknown method :$method; the study's methods are " *
            "$(sort!(collect(keys(BATTERY_METHODS))))")
        error("method :$method runs on the $(battery_method(method).engine) engine " *
              "(DecisionRulesExa.jl/examples/BatteryStorageOPF/portfolio_runner.jl), " *
              "not on this one")
    end
    backward = battery_method(method).backward::Symbol
    spec = backward_spec(backward)

    mkpath(joinpath(output, "checkpoints"))
    conf = TOML.parsefile(config_path)
    haskey(conf, "method") && String(conf["method"]) != String(method) && error(
        "the frozen config names method $(conf["method"]) but --method is $method")

    target_index    = Int(get(conf, "target_index", 200))
    segment_updates = Int(get(conf, "segment_updates", 50))
    ckpt_every      = Int(get(conf, "checkpoint_every", 10))
    eval_every      = Int(get(conf, "eval_every", 20))
    ma_window       = Int(get(conf, "ma_window", 25))
    num_stages      = Int(get(conf, "num_stages", 24))
    eval_columns    = _int_list(get(conf, "eval_columns", [1, 2, 3, 4]))
    seed            = Int(get(conf, "seed", 20260804))
    max_recourse    = Float64(get(conf, "max_recourse", 1e-6))

    # ---- the frozen case, its protocol, and the identity -------------------
    case_dir = dirname(manifest_path)
    basename(manifest_path) == "case_manifest.json" || error(
        "--case-manifest must name a case_manifest.json, got $(basename(manifest_path))")
    case = read_battery_case(case_dir)

    eval_matrix, protocol_kind, _ = resolve_protocol(case, protocol_path)
    size(eval_matrix, 1) >= num_stages || error(
        "the $protocol_kind protocol covers $(size(eval_matrix, 1)) stages but " *
        "training asks for $num_stages")
    maximum(eval_columns) <= size(eval_matrix, 2) || error(
        "panel column $(maximum(eval_columns)) is outside the $protocol_kind " *
        "protocol's $(size(eval_matrix, 2)) columns")

    ident = run_identity(; manifest_path = manifest_path, case = case, method = method,
                           config_path = config_path, conf = conf,
                           protocol_path = protocol_path, protocol_kind = protocol_kind,
                           num_stages = num_stages, backward = backward)
    write_toml_atomic(joinpath(output, "identity.toml"), ident)

    @printf("segment %s seg%d att%d · method %s (%s backward) · case %s\n",
            run_id, segment, attempt, method, spec.formulation, case.name)
    @printf("  panel: %s protocol, %d of %d columns · identity %s\n",
            protocol_kind, length(eval_columns), size(eval_matrix, 2),
            first(ident["identity_sha256"], 16))

    # SDDP.jl writes its training log into the working directory. Run from the
    # attempt directory so it lands with the rest of this segment's evidence
    # instead of in whatever directory the caller happened to be in; every path
    # this function touches was made absolute above.
    cd(output)

    # ---- resume ------------------------------------------------------------
    index_from = 0
    parent_sha = "none"
    convergence = Any[]
    evaluations = Any[]
    best_cost = Inf
    best_index = 0
    checksum = 0.0
    cur_cuts = nothing            # path of the cut file the next chunk resumes from

    if resume != "none"
        ck = load_segment_checkpoint(resume)
        assert_identity(ident, Dict{String,Any}(ck["identity"]), "parent checkpoint")
        index_from  = Int(ck["global_index"])
        checksum    = Float64(ck["trajectory_checksum"])
        best_cost   = ck["best_cost"] === nothing ? Inf : Float64(ck["best_cost"])
        best_index  = Int(ck["best_index"])
        convergence = collect(ck["convergence"])
        evaluations = collect(ck["evaluations"])
        cur_cuts    = materialize_cuts(joinpath(output, "checkpoints"), ck["cuts"], spec.tag)
        parent_sha  = sha256_file(resume)
        @printf("  resumed from iteration %d · checksum %.10e · best %.6f\n",
                index_from, checksum, best_cost)
    end

    stop_target = min(target_index, index_from + segment_updates)
    index_from < stop_target || error(
        "nothing to do: resumed at $index_from with stop target $stop_target")

    # ---- local artifacts ---------------------------------------------------
    hist_io = open(joinpath(output, "history.csv"), "w")
    println(hist_io, "index,train_loss,panel_value,bound,solve_ok,solve_fail,deficit")
    flush(hist_io)

    traj_rows = Tuple{Int,Float64,Float64,Float64}[]
    forward_costs = Float64[]
    new_evals = Any[]
    # SDDP counts its own subproblem solves in `total_solves`, which restarts at
    # zero on every freshly built graph — so it is accumulated across chunks
    # rather than read as a running total.
    solve_ok = 0
    solve_fail = 0
    worst_recourse_seen = 0.0
    # TIME ACCOUNTING, matching the Exa runner: setup, training (the SDDP
    # iterations) and evaluation (the true-ACP screening panel) are measured
    # separately, because the campaign's budget is on ACTIVE TRAINING and queue
    # time and depot construction are the controller's to report, not this
    # process's.
    setup_seconds = time() - t_start
    training_seconds = 0.0
    evaluation_seconds = 0.0

    idx = index_from
    last_ck_path = resume == "none" ? "" : resume
    last_ck_sha = parent_sha
    bound = NaN
    reason = "segment_updates_reached"

    while idx < stop_target
        n = min(ckpt_every, stop_target - idx)
        s_k = chunk_seed(seed, idx)

        _t_train = time()
        trained = train_battery_sddp(case;
                                     backward = backward,
                                     num_stages = num_stages,
                                     iteration_limit = n,
                                     seed = s_k,
                                     print_level = 0,
                                     resume_cuts = cur_cuts,
                                     cut_path = joinpath(output, "checkpoints",
                                                         "chunk.$(spec.tag).cuts.json"))
        training_seconds += time() - _t_train
        bound = trained.bound

        # SDDP's own per-iteration log: the bound and the forward-pass cost of
        # that iteration's sampled scenario, at no extra solve. `iteration` is
        # local to the call, so it is offset onto the global index here.
        log = trained.backward_graph.most_recent_training_results.log
        chunk_solves = 0
        for entry in log
            gi = idx + Int(entry.iteration)
            fc = Float64(entry.simulation_value)
            checksum += fc
            push!(forward_costs, fc)
            push!(traj_rows, (gi, fc, Float64(entry.bound), time() - t_start))
            push!(convergence, Dict{String,Any}(
                "iteration" => gi, "bound" => Float64(entry.bound),
                "forward_cost" => fc, "wall_seconds" => time() - t_start))
            chunk_solves = max(chunk_solves, Int(entry.total_solves))
            entry.serious_numerical_issue && (solve_fail += 1)
        end
        solve_ok += chunk_solves
        idx += n

        # ---- fixed-panel evaluation and checkpoint selection ---------------
        panel_value = NaN
        selected = false
        if eval_every > 0 && (idx % eval_every == 0 || idx == stop_target)
            _t_eval = time()
            ev = evaluate_acp_panel(case, trained, eval_matrix, eval_columns;
                                    max_recourse = max_recourse)
            evaluation_seconds += time() - _t_eval
            panel_value = ev.mean_cost
            worst_recourse_seen = max(worst_recourse_seen, ev.worst_recourse)
            selected = ev.complete && ev.mean_cost < best_cost
            selected && (best_cost = ev.mean_cost; best_index = idx)
            row = Dict{String,Any}(
                "index" => idx, "protocol" => String(protocol_kind),
                "acp_mean" => ev.mean_cost, "acp_raw_mean" => ev.raw_mean,
                "worst_recourse" => ev.worst_recourse,
                "complete" => ev.complete, "selected" => selected,
                "best_cost" => best_cost)
            push!(evaluations, row)
            push!(new_evals, row)
            @printf("  iteration %d · %s panel: true-ACP mean %16.6f  worst recourse %.3e  complete %s%s\n",
                    idx, protocol_kind, ev.mean_cost, ev.worst_recourse, ev.complete,
                    selected ? "  [selected]" : "")
        end

        # One history row per ITERATION, so the controller's continuous curve has
        # a point per unit of global index. The panel value and the deficit are
        # attached to the last iteration of the chunk they were measured after.
        for (k, r) in enumerate(traj_rows[end-length(log)+1:end])
            last = k == length(log)
            @printf(hist_io, "%d,%.10f,%s,%.10f,%d,%d,%.10e\n", r[1], r[2],
                    (last && !isnan(panel_value)) ? @sprintf("%.10f", panel_value) : "NaN",
                    r[3], solve_ok, solve_fail, worst_recourse_seen)
        end
        flush(hist_io)

        @printf("chunk done · iterations %d→%d · %s %.6f%s\n",
                idx - n, idx, trained.bound_name, trained.bound,
                trained.bound_bounds_acp ? "" : "  (NOT a bound on the ACP problem)")

        # ---- checkpoint ----------------------------------------------------
        cuts = JSON.parsefile(joinpath(output, "checkpoints",
                                       "chunk.$(spec.tag).cuts.json"))
        state = (tag = spec.tag, backward = backward, seed = seed, bound = trained.bound,
                 bound_name = trained.bound_name,
                 bound_bounds_acp = trained.bound_bounds_acp,
                 best_cost = best_cost, best_index = best_index, checksum = checksum,
                 convergence = convergence, evaluations = evaluations,
                 parent_sha = parent_sha, index_from = index_from,
                 run_id = run_id, segment = segment, attempt = attempt)
        last_ck_path, last_ck_sha = write_segment_checkpoint(
            output, idx, cuts, state, ident;
            kind = idx == stop_target ? "final" : "periodic")
        cur_cuts = materialize_cuts(joinpath(output, "checkpoints"), cuts, spec.tag)

        # ---- graceful stop, between complete chunks ------------------------
        if isfile(stop_file) || (time() - t_start) > max_secs
            reason = isfile(stop_file) ? "signal_stop" : "max_seconds"
            @info "stopping early" reason index = idx
            break
        end
    end
    close(hist_io)

    # ---- the richer local artifacts ----------------------------------------
    ma = moving_average(forward_costs, ma_window)
    open(joinpath(output, "trajectory.csv"), "w") do io
        println(io, "index,forward_cost,forward_cost_ma$(ma_window),bound,wall_seconds")
        for (k, r) in enumerate(traj_rows)
            @printf(io, "%d,%.10f,%.10f,%.10f,%.3f\n", r[1], r[2], ma[k], r[3], r[4])
        end
    end
    open(joinpath(output, "evaluation.csv"), "w") do io
        println(io, "index,protocol,columns,acp_mean,acp_raw_mean,worst_recourse,complete,selected,best_cost")
        for e in new_evals
            @printf(io, "%d,%s,%s,%.10f,%.10f,%.6e,%s,%s,%.10f\n", e["index"], e["protocol"],
                    join(eval_columns, " "), e["acp_mean"], e["acp_raw_mean"],
                    e["worst_recourse"], e["complete"], e["selected"], e["best_cost"])
        end
    end

    # ---- the segment result -------------------------------------------------
    reached = idx >= stop_target
    reached && idx >= target_index && (reason = "target_reached")
    isempty(last_ck_path) && error(
        "the segment produced no checkpoint; refusing to write a result that " *
        "claims progress it cannot evidence")

    here = @__DIR__
    projdir = dirname(something(Base.active_project(), joinpath(here, "Project.toml")))
    result = Dict{String,Any}(
        "schema"               => RUNNER_RESULT_SCHEMA,
        "run_id"               => run_id,
        "segment"              => segment,
        "attempt"              => attempt,
        "method"               => String(method),
        "status"               => reached ? "complete" : "preempted",
        # `status` is the SEGMENT's verdict, and the controller depends on that:
        # a segment that finished its planned updates must be acceptable, or a
        # multi-segment run could never make progress. Whether the RUN is
        # finished is a different question and gets its own field — a reader
        # must never infer "the run reached its target" from "the segment
        # completed". `termination_reason` separates them too:
        # `target_reached` only when the configured target index was reached.
        "run_complete"         => idx >= target_index,
        "termination_reason"   => reason,
        "command"              => join(vcat(["julia", "--project=" * projdir, @__FILE__],
                                            ARGS), " "),
        "julia_version"        => string(VERSION),
        "project_toml_sha256"  => sha256_file(joinpath(projdir, "Project.toml")),
        "manifest_toml_sha256" => isfile(joinpath(projdir, "Manifest.toml")) ?
                                  sha256_file(joinpath(projdir, "Manifest.toml")) : "none",
        "code_commit"          => git_commit(here),
        "code_digest"          => runner_code_digest(here),
        "case_manifest"        => manifest_path,
        "case_digest"          => sha256_file(manifest_path),
        "config_digest"        => sha256_file(config_path),
        "protocol_digest"      => sha256_file(protocol_path),
        "protocol_kind"        => String(protocol_kind),
        "support_digest"       => String(case.manifest["support"]["sha256"]),
        "identity_digest"      => ident["identity_sha256"],
        "parent_checkpoint"    => resume,
        "parent_sha256"        => parent_sha,
        "child_checkpoint"     => last_ck_path,
        "child_sha256"         => last_ck_sha,
        "index_from"           => index_from,
        "index_to"             => idx,
        "updates_completed"    => idx - index_from,
        "target_index"         => target_index,
        "wall_seconds"         => round(time() - t_start, digits = 3),
        "setup_seconds"        => round(setup_seconds, digits = 3),
        "training_seconds"     => round(training_seconds, digits = 3),
        "evaluation_seconds"   => round(evaluation_seconds, digits = 3),
        "gpu_seconds"          => 0.0,
        "solve_total"          => solve_ok + solve_fail,
        "solve_optimal"        => solve_ok,
        "solve_failed"         => solve_fail,
        "physical_deficit"     => worst_recourse_seen,
        "physical_surplus"     => 0.0,
        # The scalar and, beside it, what it is allowed to be called and whether
        # it bounds anything about the true ACP problem. The DC arm's does not,
        # and it must never acquire the word on the strength of a column name.
        "sddp_bound"           => bound,
        "bound_name"           => spec.bound_name,
        "bound_bounds_acp"     => spec.bounds_acp,
        "backward_formulation" => string(spec.formulation),
        "best_panel_cost"      => best_cost == Inf ? NaN : best_cost,
        "best_panel_index"     => best_index,
        "trajectory_checksum"  => checksum,
        "history_rows"         => length(traj_rows),
        "history_sha256"       => sha256_file(joinpath(output, "history.csv")),
        "slurm_job_id"         => get(ENV, "SLURM_JOB_ID", ""),
        "slurm_array_id"       => get(ENV, "SLURM_ARRAY_JOB_ID", ""),
        "node"                 => gethostname(),
        "started_utc"          => Dates.format(unix2datetime(t_start),
                                               dateformat"yyyy-mm-dd\THH:MM:SS\Z"),
        "finished_utc"         => utcnow(),
        "eval_indices"         => [e["index"] for e in new_evals],
        "eval_values"          => [e["acp_mean"] for e in new_evals],
    )
    write_toml_atomic(joinpath(output, "result.toml"), result)
    @printf("segment done · %s · iteration %d→%d · checksum %.10e · best %.6f\n",
            result["status"], index_from, idx, checksum, best_cost)
    return 0
end

"""
    main(args=ARGS) -> Int

Check that the six scientific flags are present, then run one segment.
"""
function main(args = ARGS)
    a = parse_args(args)
    for k in ("case-manifest", "method", "config", "protocol", "output")
        haskey(a, k) || error("--$k is required; see the header of $(@__FILE__)")
    end
    return run_segment(a)
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
