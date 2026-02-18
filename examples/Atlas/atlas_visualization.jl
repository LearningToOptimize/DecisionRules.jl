# Visualization utilities for the Atlas model
# This file contains MeshCat-related functions for visualizing the Atlas robot.
# Import this file only when visualization is needed.

using MeshCat
using MeshCatMechanisms
using GeometryBasics: Point, Vec, HyperSphere
using RigidBodyDynamics: findbody, findjoint, successor, frame_after, default_frame, MechanismState, relative_transform, translation
using CoordinateTransformations: Translation
using Colors
using LinearAlgebra: norm, cross

const URDFPATH = joinpath(@__DIR__, "urdf", "atlas_all.urdf")

function init_visualizer(model::Atlas, vis::Visualizer)
    delete!(vis)
    meshes_path = joinpath(@__DIR__, "urdf")
    mvis = MechanismVisualizer(model.mech, URDFVisuals(URDFPATH, package_path=[meshes_path]), vis)
    return mvis
end

function visualize!(model::Atlas, mvis::MechanismVisualizer, q)
    set_configuration!(mvis, q[1:model.nq])
end

function animate!(model::Atlas, mvis::MechanismVisualizer, qs; Δt=0.001)
    anim = MeshCat.Animation(mvis.visualizer; fps=convert(Int, floor(1.0 / Δt)))
    for (t, q) in enumerate(qs)
        MeshCat.atframe(anim, t) do 
            set_configuration!(mvis, q[1:model.nq])
        end
    end
    MeshCat.setanimation!(mvis, anim)

    return anim
end

"""
    animate_with_perturbation_cause!(model, mvis, qs, perturbations; kwargs...)

Animate Atlas and overlay an illustrative perturbation-cause arrow in MeshCat.
The arrow points in perturbation direction and its length scales with magnitude.

Arguments:
- `qs`: state trajectory (length `T`)
- `perturbations`: per-stage perturbations (typically length `T-1`)

Keyword arguments:
- `Δt`: frame step in seconds
- `arrow_scale`: converts perturbation magnitude to arrow length
- `min_arrow_length`: minimum visible arrow length when active
- `show_threshold`: hide arrow when `abs(perturbation) <= show_threshold`
- `linger_seconds`: how long the cause arrow remains visible after each perturbation
- `perturbation_state_index`: state index (in `x`) where perturbation is injected;
  if it maps to a velocity state, the arrow is attached to that joint/body.
- `arrow_base`: local-frame base offset from the selected anchor frame
- `impact_distance`: local x-distance from anchor to initial contact point (keeps marker outside robot)
- `retreat_distance`: local x-distance the marker retreats after impact
- `shaft_radius`: thickness of the arrow shaft
"""
function animate_with_perturbation_cause!(
    model::Atlas,
    mvis::MechanismVisualizer,
    qs,
    perturbations;
    Δt=0.001,
    arrow_scale=1.0,
    min_arrow_length=0.12,
    show_threshold=1e-6,
    linger_seconds=0.35,
    perturbation_state_index=nothing,
    arrow_base=Point(0.0, 0.0, 0.12),
    impact_distance=0.18,
    retreat_distance=0.35,
    shaft_radius=0.03,
)
    vis = mvis.visualizer
    if isnothing(perturbation_state_index)
        perturbation_state_index = model.nq + 5
    end

    velocity_idx = perturbation_state_index - model.nq
    anchor_body = nothing
    anchor_origin = Point(0.0, 0.0, 0.0)
    perturbation_dir_local = Vec(0.0, 1.0, 0.0)
    anchor_description = ""

    if 1 <= velocity_idx <= length(model.joint_names)
        joint_name = model.joint_names[velocity_idx]
        joint = findjoint(model.mech, joint_name)
        anchor_body = successor(joint, model.mech)
        state0 = MechanismState(model.mech)
        joint_in_body = translation(relative_transform(state0, default_frame(anchor_body), frame_after(joint)))
        anchor_origin = Point(joint_in_body[1], joint_in_body[2], joint_in_body[3])
        joint_type = getfield(joint, :joint_type)
        if hasfield(typeof(joint_type), :axis)
            axis = collect(getfield(joint_type, :axis))
            axis_norm = norm(axis)
            if axis_norm > 1e-8
                axis ./= axis_norm
                # Build a direction orthogonal to the joint axis so the effect reads as a lateral collision.
                dir = cross(axis, [0.0, 0.0, 1.0])
                if norm(dir) < 1e-8
                    dir = cross(axis, [1.0, 0.0, 0.0])
                end
                if norm(dir) > 1e-8
                    dir ./= norm(dir)
                    perturbation_dir_local = Vec(dir[1], dir[2], dir[3])
                end
            end
        end
        anchor_description =
            "joint=$(joint_name), body=$(getfield(anchor_body, :name)), dir_local=$(collect(perturbation_dir_local))"
    else
        anchor_body = findbody(model.mech, "pelvis")
        anchor_description =
            "fallback=pelvis (state index $perturbation_state_index not mapped to velocity DOF), dir_local=$(collect(perturbation_dir_local))"
    end

    cause_arrow_parent = mvis[anchor_body]
    cause_arrow = ArrowVisualizer(cause_arrow_parent[:perturbation_cause_arrow])
    setobject!(
        cause_arrow;
        shaft_material=MeshLambertMaterial(color=colorant"red"),
        head_material=MeshLambertMaterial(color=colorant"yellow"),
    )
    cause_impactor = cause_arrow_parent[:perturbation_cause_impactor]
    setobject!(
        cause_impactor,
        HyperSphere(Point(0.0, 0.0, 0.0), 0.055),
        MeshLambertMaterial(color=colorant"orange"),
    )
    linger_frames = max(1, round(Int, linger_seconds / Δt))
    head_radius = 2.2 * shaft_radius
    head_length = 2.8 * shaft_radius

    anim = MeshCat.Animation(vis; fps=convert(Int, floor(1.0 / Δt)))
    last_event_frame = 0
    last_event_value = 0.0
    last_event_sign = 1.0
    event_count = count(p -> abs(p) > show_threshold, perturbations)
    max_abs_pert = isempty(perturbations) ? 0.0 : maximum(abs.(perturbations))
    println(
        "Perturbation-cause overlay: events=$event_count, max_abs=$(round(max_abs_pert, digits=6)), " *
        "perturb_state_idx=$perturbation_state_index, anchor={$anchor_description}, " *
        "impact_distance=$impact_distance, retreat_distance=$retreat_distance"
    )
    for (frame, q) in enumerate(qs)
        MeshCat.atframe(anim, frame) do
            set_configuration!(mvis, q[1:model.nq])

            p = frame <= length(perturbations) ? perturbations[frame] : 0.0
            if abs(p) > show_threshold
                last_event_frame = frame
                last_event_value = p
                last_event_sign = sign(p) == 0 ? 1.0 : sign(p)
            end

            frames_since_event = frame - last_event_frame
            if last_event_frame > 0 && frames_since_event <= linger_frames
                progress = frames_since_event / linger_frames
                decay = 1.0 - progress
                outward_dir = Vec(
                    last_event_sign * perturbation_dir_local[1],
                    last_event_sign * perturbation_dir_local[2],
                    last_event_sign * perturbation_dir_local[3],
                )

                # Contact happens just outside the body, then marker backs away.
                contact_point = Point(
                    anchor_origin[1] + arrow_base[1] + outward_dir[1] * impact_distance,
                    anchor_origin[2] + arrow_base[2] + outward_dir[2] * impact_distance,
                    anchor_origin[3] + arrow_base[3] + outward_dir[3] * impact_distance,
                )
                impactor_point = Point(
                    contact_point[1] + outward_dir[1] * retreat_distance * progress,
                    contact_point[2] + outward_dir[2] * retreat_distance * progress,
                    contact_point[3] + outward_dir[3] * retreat_distance * progress,
                )
                settransform!(
                    cause_impactor,
                    Translation(impactor_point[1], impactor_point[2], impactor_point[3]),
                )

                effective_p = last_event_value * decay
                arrow_length = max(min_arrow_length * decay, abs(effective_p) * arrow_scale)
                # Arrow points from impactor toward robot (collision cause direction).
                direction = Vec(
                    -outward_dir[1] * arrow_length,
                    -outward_dir[2] * arrow_length,
                    -outward_dir[3] * arrow_length,
                )
                settransform!(
                    cause_arrow,
                    impactor_point,
                    direction;
                    shaft_radius=shaft_radius,
                    max_head_radius=head_radius,
                    max_head_length=head_length,
                )
            else
                # "Hide" by shrinking to zero length (more robust than animating visibility).
                settransform!(
                    cause_arrow,
                    anchor_origin,
                    Vec(0.0, 0.0, 0.0);
                    shaft_radius=shaft_radius,
                    max_head_radius=head_radius,
                    max_head_length=head_length,
                )
                # Keep impactor out of view when there is no active perturbation event.
                settransform!(cause_impactor, Translation(1000.0, 1000.0, 1000.0))
            end
        end
    end
    MeshCat.setanimation!(mvis, anim)
    return anim
end
