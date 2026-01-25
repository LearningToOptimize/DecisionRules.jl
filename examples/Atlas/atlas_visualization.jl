# Visualization utilities for the Atlas model
# This file contains MeshCat-related functions for visualizing the Atlas robot.
# Import this file only when visualization is needed.

using MeshCat
using MeshCatMechanisms

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
