# taken liberally from ECOS.jl
# differences from ECOS.jl's implementation:
#    no support for exponential cone 
#    final target is a dense matrix, not a sparse matrix

using MathOptInterface
using Compat.SparseArrays


const MOI = MathOptInterface
const CI = MOI.ConstraintIndex
const VI = MOI.VariableIndex

const MOIU = MOI.Utilities


const SF = Union{MOI.SingleVariable, MOI.ScalarAffineFunction{Float64}, MOI.VectorOfVariables, MOI.VectorAffineFunction{Float64}}
const SS = Union{MOI.EqualTo{Float64}, MOI.GreaterThan{Float64}, MOI.LessThan{Float64}, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives, MOI.SecondOrderCone}


# Used to build the data with allocate-load during `copy_to`.
# When `optimize!` is called, a the data is used to build `ECOSMatrix`
# and the `ModelData` struct is discarded
mutable struct ModelData
    m::Int # Number of rows/constraints
    n::Int # Number of cols/variables
    IA::Vector{Int} # List of conic rows
    JA::Vector{Int} # List of conic cols
    VA::Vector{Float64} # List of conic coefficients
    b::Vector{Float64} # List of conic coefficients
    IG::Vector{Int} # List of equality rows
    JG::Vector{Int} # List of equality cols
    VG::Vector{Float64} # List of equality coefficients
    h::Vector{Float64} # List of equality coefficients
    objconstant::Float64 # The objective is min c'x + objconstant
    c::Vector{Float64}
end

# This is tied to ECOS's internal representation
mutable struct ConeData
    f::Int # number of linear equality constraints
    l::Int # length of LP cone
    q::Int # length of SOC cone
    qa::Vector{Int} # array of second-order cone constraints
    # The following four field store model information needed to compute `ConstraintPrimal` and `ConstraintDual`
    eqsetconstant::Dict{Int, Float64}   # For the constant of EqualTo
    eqnrows::Dict{Int, Int}             # The number of rows of Zeros
    ineqsetconstant::Dict{Int, Float64} # For the constant of LessThan and GreaterThan
    ineqnrows::Dict{Int, Int}           # The number of rows of each vector sets except Zeros
    function ConeData()
        new(0, 0, 0, Int[],
            Dict{Int, Float64}(),
            Dict{Int, UnitRange{Int}}(),
            Dict{Int, Float64}(),
            Dict{Int, UnitRange{Int}}())
    end
end

mutable struct Optimizer <: MOI.AbstractOptimizer
    cone::ConeData
    maxsense::Bool
    data::Union{Nothing, ModelData} # only non-Nothing between MOI.copy_to and MOI.optimize!
    sol::Union{Nothing, State}
    options
    function Optimizer(; kwargs...)
        new(ConeData(), false, nothing, nothing, kwargs)
    end
end

MOI.get(::Optimizer, ::MOI.SolverName) = "SOCP.jl"
MOI.supports(::Optimizer, ::MOI.Silent) = true
function MOI.is_empty(optimizer::Optimizer)
    !optimizer.maxsense && optimizer.data === nothing
end
function MOI.empty!(optimizer::Optimizer)
    optimizer.maxsense = false
    optimizer.data = nothing # It should already be nothing except if an error is thrown inside copy_to
    optimizer.sol = nothing
end
function MOI.supports(::Optimizer,
                      ::Union{MOI.ObjectiveSense,
                              MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}})
    return true
end

function MOI.supports_constraint(::Optimizer,
                                 ::Type{MOI.VectorAffineFunction{Float64}},
                                 ::Type{<:Union{MOI.Zeros, MOI.Nonnegatives,
                                                MOI.SecondOrderCone}})
    return true
end

MOIU.supports_allocate_load(::Optimizer, copy_names::Bool) = !copy_names

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike; kws...)
    return MOIU.automatic_copy_to(dest, src; kws...)
end
# Computes cone dimensions
constroffset(cone::ConeData, ci::CI{<:MOI.AbstractFunction, MOI.Zeros}) = ci.value
function _allocate_constraint(cone::ConeData, f, s::MOI.Zeros)
    ci = cone.f
    cone.f += MOI.dimension(s)
    ci
end
constroffset(cone::ConeData, ci::CI{<:MOI.AbstractFunction, MOI.Nonnegatives}) = ci.value
function _allocate_constraint(cone::ConeData, f, s::MOI.Nonnegatives)
    ci = cone.l
    cone.l += MOI.dimension(s)
    ci
end
constroffset(cone::ConeData, ci::CI{<:MOI.AbstractFunction, <:MOI.SecondOrderCone}) = cone.l + ci.value
function _allocate_constraint(cone::ConeData, f, s::MOI.SecondOrderCone)
    push!(cone.qa, s.dimension)
    ci = cone.q
    cone.q += MOI.dimension(s)
    ci
end
constroffset(optimizer::Optimizer, ci::CI) = constroffset(optimizer.cone, ci::CI)
function MOIU.allocate_constraint(optimizer::Optimizer, f::F, s::S) where {F <: MOI.AbstractFunction, S <: MOI.AbstractSet}
    CI{F, S}(_allocate_constraint(optimizer.cone, f, s))
end

# Build constraint matrix
output_index(t::MOI.VectorAffineTerm) = t.output_index
variable_index_value(t::MOI.ScalarAffineTerm) = t.variable_index.value
variable_index_value(t::MOI.VectorAffineTerm) = variable_index_value(t.scalar_term)
coefficient(t::MOI.ScalarAffineTerm) = t.coefficient
coefficient(t::MOI.VectorAffineTerm) = coefficient(t.scalar_term)
constrrows(s::MOI.AbstractVectorSet) = 1:MOI.dimension(s)
constrrows(optimizer::Optimizer, ci::CI{<:MOI.AbstractVectorFunction, MOI.Zeros}) = 1:optimizer.cone.eqnrows[constroffset(optimizer, ci)]
constrrows(optimizer::Optimizer, ci::CI{<:MOI.AbstractVectorFunction, <:MOI.AbstractVectorSet}) = 1:optimizer.cone.ineqnrows[constroffset(optimizer, ci)]
matrix(data::ModelData, s::MOI.Zeros) = data.b, data.IA, data.JA, data.VA
matrix(data::ModelData, s::Union{MOI.Nonnegatives, MOI.SecondOrderCone}) = data.h, data.IG, data.JG, data.VG
matrix(optimizer::Optimizer, s) = matrix(optimizer.data, s)
# ECOS orders differently than MOI the second and third dimension of the exponential cone
orderval(val, s) = val
orderidx(idx, s) = idx
expmap(i) = (1, 3, 2)[i]
function MOIU.load_constraint(optimizer::Optimizer, ci::MOI.ConstraintIndex, f::MOI.VectorAffineFunction, s::MOI.AbstractVectorSet)
    func = MOIU.canonical(f)
    I = Int[output_index(term) for term in func.terms]
    J = Int[variable_index_value(term) for term in func.terms]
    V = Float64[-coefficient(term) for term in func.terms]
    offset = constroffset(optimizer, ci)
    rows = constrrows(s)
    if s isa MOI.Zeros
        optimizer.cone.eqnrows[offset] = length(rows)
    else
        optimizer.cone.ineqnrows[offset] = length(rows)
    end
    i = offset .+ rows
    # The ECOS format is b - Ax ∈ cone
    # so minus=false for b and minus=true for A
    b, Is, Js, Vs = matrix(optimizer, s)
    b[i] .= orderval(f.constants, s)
    append!(Is, offset .+ orderidx(I, s))
    append!(Js, J)
    append!(Vs, V)
end

function MOIU.allocate_variables(optimizer::Optimizer, nvars::Integer)
    optimizer.cone = ConeData()
    VI.(1:nvars)
end

function MOIU.load_variables(optimizer::Optimizer, nvars::Integer)
    cone = optimizer.cone
    m = cone.l + cone.q
    IA = Int[]
    JA = Int[]
    VA = Float64[]
    b = zeros(cone.f)
    IG = Int[]
    JG = Int[]
    VG = Float64[]
    h = zeros(m)
    c = zeros(nvars)
    optimizer.data = ModelData(m, nvars, IA, JA, VA, b, IG, JG, VG, h, 0., c)
end

function MOIU.allocate(optimizer::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    optimizer.maxsense = sense == MOI.MAX_SENSE
end
function MOIU.allocate(::Optimizer, ::MOI.ObjectiveFunction,
                       ::MOI.ScalarAffineFunction{Float64})
end

function MOIU.load(::Optimizer, ::MOI.ObjectiveSense, ::MOI.OptimizationSense)
end
function MOIU.load(optimizer::Optimizer, ::MOI.ObjectiveFunction,
                   f::MOI.ScalarAffineFunction)
    c0 = Vector(sparsevec(variable_index_value.(f.terms), coefficient.(f.terms),
                          optimizer.data.n))
    optimizer.data.objconstant = f.constant
    optimizer.data.c = optimizer.maxsense ? -c0 : c0
    return nothing
end


function MOI.optimize!(instance::Optimizer)
    if instance.data === nothing
        # optimize! has already been called and no new model has been copied
        return
    end
    cone = instance.cone
    m = instance.data.m
    n = instance.data.n
    A = Array{Float64,2}(sparse(instance.data.IA, instance.data.JA, instance.data.VA, cone.f, n))
    b = instance.data.b
    G = Array{Float64,2}(sparse(instance.data.IG, instance.data.JG, instance.data.VG, m, n))
    h = instance.data.h
    objconstant = instance.data.objconstant
    c = instance.data.c

    cones = Socp.Cone[Socp.POC(0,cone.l)]
    offs = cone.l
    for q in cone.qa
    	push!(cones, Socp.SOC(offs, q))
    	offs += q
    end
    prob = Socp.Problem(c, A, b, G, h, cones)
    res = solve_socp(prob, Socp.SolverState(prob))
    instance.sol = res
end

#=
using MathOptInterface
const MOI = MathOptInterface
model = MOI.instantiate(() -> Socp.Optimizer(maxit=10000); with_bridge_type=Float64)
x = MOI.add_variable(model)
y = MOI.add_variable(model)

MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([2.0,-5.0], [x,y]), 0.0))
MOI.add_constraint(model, MOI.SingleVariable(x), MOI.Interval(100.0, 200.0))
MOI.add_constraint(model, MOI.SingleVariable(y), MOI.Interval(80.0, 170.0))
MOI.add_constraint(model, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,1.0], [x,y]), 0.0), MOI.GreaterThan(200.0))
=#

const reorderval = orderval
function MOI.get(instance::Optimizer, ::MOI.VariablePrimal, vi::VI)
    instance.sol.x[vi.value]
end
MOI.get(instance::Optimizer, a::MOI.VariablePrimal, vi::Vector{VI}) = MOI.get.(instance, Ref(a), vi)

# setconstant: Retrieve set constant stored in `ConeData` during `copy_to`
setconstant(instance::Optimizer, offset, ::CI{<:MOI.AbstractFunction, <:MOI.EqualTo}) = instance.cone.eqsetconstant[offset]
setconstant(instance::Optimizer, offset, ::CI) = instance.cone.ineqsetconstant[offset]
_unshift(instance::Optimizer, offset, value, ::CI) = value
_unshift(instance::Optimizer, offset, value, ci::CI{<:MOI.AbstractScalarFunction, <:MOI.AbstractScalarSet}) = value + setconstant(instance, offset, ci)
function MOI.get(instance::Optimizer, ::MOI.ConstraintPrimal, ci::CI{<:MOI.AbstractFunction, MOI.Zeros})
    rows = constrrows(instance, ci)
    zeros(length(rows))
end
function MOI.get(instance::Optimizer, ::MOI.ConstraintPrimal, ci::CI{<:MOI.AbstractFunction, <:MOI.EqualTo})
    offset = constroffset(instance, ci)
    setconstant(instance, offset, ci)
end
function MOI.get(instance::Optimizer, ::MOI.ConstraintPrimal, ci::CI{<:MOI.AbstractFunction, S}) where S <: MOI.AbstractSet
    offset = constroffset(instance, ci)
    rows = constrrows(instance, ci)
    _unshift(instance, offset, scalecoef(rows, reorderval(instance.sol.s[offset .+ rows], S), false, S), ci)
end

_dual(instance, ci::CI) = instance.sol.z
function MOI.get(instance::Optimizer, ::MOI.ConstraintDual, ci::CI{<:MOI.AbstractFunction, S}) where S <: MOI.AbstractSet
    offset = constroffset(instance, ci)
    rows = constrrows(instance, ci)
    scalecoef(rows, reorderval(_dual(instance, ci)[offset .+ rows], S), false, S)
end

MOI.get(instance::Optimizer, ::MOI.ResultCount) = 1
