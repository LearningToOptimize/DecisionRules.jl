# Parameter Duals for General JuMP Models
# This module computes the dual values (sensitivities) of parameters in JuMP models.
# Unlike POI which only works with convex problems, this works with any solved model
# by examining constraints and the objective function.

# Note: JuMP and MOI are already imported in the parent module

"""
    compute_parameter_dual(model::JuMP.Model, param::JuMP.VariableRef)

Compute the dual value (sensitivity) of a parameter in a solved JuMP model.

The parameter dual represents ∂(objective)/∂(parameter_value) and is computed by:
1. Finding all constraints where the parameter appears
2. For each constraint, computing: -coefficient * constraint_dual
3. For the objective, adding the coefficient (or negative for maximization)
4. Summing all contributions

This works for any solved model, not just convex ones, as long as dual values are available.

# Arguments
- `model`: A solved JuMP model
- `param`: A parameter variable (created with `@variable(model, p in MOI.Parameter(value))`)

# Returns
- The dual value (sensitivity) of the parameter

# Example
```julia
model = Model(HiGHS.Optimizer)
@variable(model, x >= 0)
@variable(model, p in MOI.Parameter(1.0))
@constraint(model, con, x >= 2 * p)
@objective(model, Min, 3 * x + p)
optimize!(model)
dual_p = compute_parameter_dual(model, p)  # Should be -2 * dual(con) + 1
```
"""
function compute_parameter_dual(model::JuMP.Model, param::JuMP.VariableRef)
    if !JuMP.is_parameter(param)
        error("Variable is not a parameter")
    end
    
    dual_value = 0.0
    
    # Get contributions from constraints
    dual_value += _get_dual_from_constraints(model, param)
    
    # Get contribution from objective
    dual_value += _get_dual_from_objective(model, param)
    
    return dual_value
end

"""
    _get_dual_from_constraints(model::JuMP.Model, param::JuMP.VariableRef)

Compute the dual contribution from all constraints containing the parameter.
"""
function _get_dual_from_constraints(model::JuMP.Model, param::JuMP.VariableRef)
    dual_contribution = 0.0
    
    # Iterate over all constraint types in the model
    for (F, S) in JuMP.list_of_constraint_types(model)
        # Handle scalar affine constraints
        if F <: JuMP.GenericAffExpr
            dual_contribution += _get_dual_from_affine_constraints(model, param, S)
        # Handle scalar quadratic constraints
        elseif F <: JuMP.GenericQuadExpr
            dual_contribution += _get_dual_from_quadratic_constraints(model, param, S)
        # Handle vector affine constraints (conic)
        elseif F <: Vector{<:JuMP.GenericAffExpr}
            dual_contribution += _get_dual_from_vector_affine_constraints(model, param, F, S)
        end
    end
    
    return dual_contribution
end

"""
    _get_dual_from_affine_constraints(model::JuMP.Model, param::JuMP.VariableRef, S)

Get dual contribution from scalar affine constraints of set type S.
"""
function _get_dual_from_affine_constraints(model::JuMP.Model, param::JuMP.VariableRef, S)
    dual_contribution = 0.0
    
    for con in JuMP.all_constraints(model, JuMP.AffExpr, S)
        con_obj = JuMP.constraint_object(con)
        func = con_obj.func
        
        # Check if parameter appears in this constraint
        coef = _get_parameter_coefficient(func, param)
        if !iszero(coef)
            try
                con_dual = JuMP.dual(con)
                # The dual contribution is -coefficient * constraint_dual
                # This follows from the Lagrangian: L = f(x) + λ*(g(x) - p*coef - b)
                # ∂L/∂p = -λ * coef
                dual_contribution -= coef * con_dual
            catch
                # If dual is not available, skip this constraint
            end
        end
    end
    
    return dual_contribution
end

"""
    _get_dual_from_quadratic_constraints(model::JuMP.Model, param::JuMP.VariableRef, S)

Get dual contribution from quadratic constraints of set type S.
Handles both affine terms (parameter appears linearly) and quadratic terms 
(parameter appears in products p*v or p*p).
"""
function _get_dual_from_quadratic_constraints(model::JuMP.Model, param::JuMP.VariableRef, S)
    dual_contribution = 0.0
    
    for con in JuMP.all_constraints(model, JuMP.QuadExpr, S)
        con_obj = JuMP.constraint_object(con)
        func = con_obj.func
        
        # Get coefficient from affine part
        coef = _get_parameter_coefficient_from_affine(func, param)
        
        # Get coefficient from quadratic part (p*v or p*p terms)
        quad_coef = _get_parameter_coefficient_from_quadratic(func, param)
        
        total_coef = coef + quad_coef
        
        if !iszero(total_coef)
            try
                con_dual = JuMP.dual(con)
                dual_contribution -= total_coef * con_dual
            catch
                # If dual is not available, skip this constraint
            end
        end
    end
    
    return dual_contribution
end

"""
    _get_dual_from_vector_affine_constraints(model::JuMP.Model, param::JuMP.VariableRef, F, S)

Get dual contribution from vector affine constraints (like conic constraints).
"""
function _get_dual_from_vector_affine_constraints(model::JuMP.Model, param::JuMP.VariableRef, F, S)
    dual_contribution = 0.0
    
    for con in JuMP.all_constraints(model, F, S)
        con_obj = JuMP.constraint_object(con)
        func = con_obj.func  # Vector of AffExpr
        
        try
            con_dual = JuMP.dual(con)  # Vector of duals
            
            for (i, expr) in enumerate(func)
                coef = _get_parameter_coefficient(expr, param)
                if !iszero(coef)
                    dual_contribution -= coef * con_dual[i]
                end
            end
        catch
            # If dual is not available, skip this constraint
        end
    end
    
    return dual_contribution
end

"""
    _get_dual_from_objective(model::JuMP.Model, param::JuMP.VariableRef)

Get the dual contribution from the objective function.
If parameter appears in the objective with coefficient c, contribution is:
- +c for minimization
- -c for maximization
"""
function _get_dual_from_objective(model::JuMP.Model, param::JuMP.VariableRef)
    obj = JuMP.objective_function(model)
    sense = JuMP.objective_sense(model)
    
    coef = _get_objective_parameter_coefficient(obj, param)
    
    if sense == MOI.MIN_SENSE
        return coef
    elseif sense == MOI.MAX_SENSE
        return -coef
    else
        return 0.0  # FEASIBILITY_SENSE or unknown
    end
end

"""
    _get_parameter_coefficient(expr::JuMP.GenericAffExpr, param::JuMP.VariableRef)

Get the coefficient of a parameter in an affine expression.
"""
function _get_parameter_coefficient(expr::JuMP.GenericAffExpr, param::JuMP.VariableRef)
    # AffExpr has terms as a dictionary from variables to coefficients
    return get(expr.terms, param, 0.0)
end

function _get_parameter_coefficient(expr::JuMP.GenericQuadExpr, param::JuMP.VariableRef)
    return _get_parameter_coefficient_from_affine(expr, param) + 
           _get_parameter_coefficient_from_quadratic(expr, param)
end

"""
    _get_parameter_coefficient_from_affine(expr::JuMP.GenericQuadExpr, param::JuMP.VariableRef)

Get the coefficient of a parameter from the affine part of a quadratic expression.
"""
function _get_parameter_coefficient_from_affine(expr::JuMP.GenericQuadExpr, param::JuMP.VariableRef)
    # The affine part is expr.aff
    return get(expr.aff.terms, param, 0.0)
end

"""
    _get_parameter_coefficient_from_quadratic(expr::JuMP.GenericQuadExpr, param::JuMP.VariableRef)

Get the effective coefficient of a parameter from quadratic terms.
For terms like coef * p * v, the effective coefficient is coef * value(v).
For terms like coef * p * p, the effective coefficient is 2 * coef * value(p).
"""
function _get_parameter_coefficient_from_quadratic(expr::JuMP.GenericQuadExpr, param::JuMP.VariableRef)
    effective_coef = 0.0
    model = JuMP.owner_model(param)
    
    # QuadExpr has terms as OrderedDict from UnorderedPair to coefficient
    for (pair, coef) in expr.terms
        v1, v2 = pair.a, pair.b
        
        if v1 == param && v2 == param
            # p * p term: ∂(coef * p^2)/∂p = 2 * coef * p
            effective_coef += 2 * coef * JuMP.parameter_value(param)
        elseif v1 == param
            # p * v term: ∂(coef * p * v)/∂p = coef * v
            if JuMP.is_parameter(v2)
                effective_coef += coef * JuMP.parameter_value(v2)
            else
                effective_coef += coef * JuMP.value(v2)
            end
        elseif v2 == param
            # v * p term: ∂(coef * v * p)/∂p = coef * v
            if JuMP.is_parameter(v1)
                effective_coef += coef * JuMP.parameter_value(v1)
            else
                effective_coef += coef * JuMP.value(v1)
            end
        end
    end
    
    return effective_coef
end

"""
    _get_objective_parameter_coefficient(obj, param::JuMP.VariableRef)

Get the coefficient of a parameter in the objective function.
"""
function _get_objective_parameter_coefficient(obj::JuMP.GenericAffExpr, param::JuMP.VariableRef)
    return get(obj.terms, param, 0.0)
end

function _get_objective_parameter_coefficient(obj::JuMP.GenericQuadExpr, param::JuMP.VariableRef)
    return _get_parameter_coefficient_from_affine(obj, param) + 
           _get_parameter_coefficient_from_quadratic(obj, param)
end

function _get_objective_parameter_coefficient(obj::JuMP.VariableRef, param::JuMP.VariableRef)
    return obj == param ? 1.0 : 0.0
end

function _get_objective_parameter_coefficient(obj::Real, param::JuMP.VariableRef)
    return 0.0
end

# Fallback for other objective types
function _get_objective_parameter_coefficient(obj, param::JuMP.VariableRef)
    @warn "Unsupported objective type $(typeof(obj)) for parameter dual computation"
    return 0.0
end
