using JuMP
using Ipopt, HSL_jll
import Base: sincos, promote_type
import RigidBodyDynamics: dynamics_solve!

sincos(x::NonlinearExpr) = (sin(x), cos(x))
promote_type(::Type{<:NonlinearExpr}, ::Type{<:NonlinearExpr}) = NonlinearExpr
promote_type(::Type{<:NonlinearExpr}, ::Type{<:Real}) = NonlinearExpr
NonlinearExpr(x::Int64) = convert(NonlinearExpr, x * AffExpr(1.0))  # Dummy conversion for NonlinearExpr
Base.:*(x::NonlinearExpr, y::UniformScaling{Bool}) = convert(NonlinearExpr, x * AffExpr(y.λ))  # Dummy conversion for UniformScaling

"""
    function dynamics_solve!(v̇::AbstractVector, λ::AbstractVector,
        G::Matrix{NonlinearExpr}, r::Vector{NonlinearExpr}
    )

Creates JuMP Constraints equivalent to the linear system.
"""
function RigidBodyDynamics.dynamics_solve!(_v̇::AbstractVector, λ::AbstractVector,
    G::Matrix{NonlinearExpr}, _c::AbstractVector, k::AbstractVector, nv::Int, nl::Int, τ::AbstractVector
)
    println("Solving dynamics with nl = $nl, nv = $nv")
    println("c: ", _c)
    println("k: ", k)
    println("τ: ", τ)
    println("v̇: ", _v̇)
    println("Λ: ", λ)
    c = parent(_c)
    r = [τ - c; -k]
    v̇ = parent(_v̇)
    model = owner_model(r[1])
    
    if nl == 0
        aux_v̇ = @variable(model, [1:nv])
        con = @constraint(model, G * aux_v̇ .== r)
        println("Constraint: ", con)
        v̇ .= aux_v̇
    else
        @constraint(model, G * [v̇; λ] .== r)
    end
    nothing
end
