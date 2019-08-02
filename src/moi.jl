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
        new(ConeData(), false, nothing,  kwargs)
    end
end

MOI.get(::Optimizer, ::MOI.SolverName) = "SOCP.jl"


function MOI.is_empty(instance::Optimizer)
    !instance.maxsense && instance.data === nothing
end

function MOI.empty!(instance::Optimizer)
    instance.maxsense = false
    instance.data = nothing # It should already be nothing except if an error is thrown inside copy_to
    instance.sol = nothing
end


MOIU.supports_allocate_load(::Optimizer, copy_names::Bool) = !copy_names

function MOI.supports(::Optimizer,
                      ::Union{MOI.ObjectiveSense,
                              MOI.ObjectiveFunction{MOI.SingleVariable},
                              MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}})
    return true
end

MOI.supports_constraint(::Optimizer, ::Type{<:SF}, ::Type{<:SS}) = true

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike; kws...)
    return MOIU.automatic_copy_to(dest, src; kws...)
end

const ZeroCones = Union{MOI.EqualTo, MOI.Zeros}
const LPCones = Union{MOI.GreaterThan, MOI.LessThan, MOI.Nonnegatives, MOI.Nonpositives}

# Computes cone dimensions
constroffset(cone::ConeData, ci::CI{<:MOI.AbstractFunction, <:ZeroCones}) = ci.value
function _allocate_constraint(cone::ConeData, f, s::ZeroCones)
    ci = cone.f
    cone.f += MOI.dimension(s)
    ci
end
constroffset(cone::ConeData, ci::CI{<:MOI.AbstractFunction, <:LPCones}) = ci.value
function _allocate_constraint(cone::ConeData, f, s::LPCones)
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
constroffset(instance::Optimizer, ci::CI) = constroffset(instance.cone, ci::CI)
function MOIU.allocate_constraint(instance::Optimizer, f::F, s::S) where {F <: MOI.AbstractFunction, S <: MOI.AbstractSet}
    CI{F, S}(_allocate_constraint(instance.cone, f, s))
end

# Build constraint matrix
scalecoef(rows, coef, minus, s) = minus ? -coef : coef
scalecoef(rows, coef, minus, s::Union{MOI.LessThan, Type{<:MOI.LessThan}, MOI.Nonpositives, Type{MOI.Nonpositives}}) = minus ? coef : -coef
output_index(t::MOI.VectorAffineTerm) = t.output_index
variable_index_value(t::MOI.ScalarAffineTerm) = t.variable_index.value
variable_index_value(t::MOI.VectorAffineTerm) = variable_index_value(t.scalar_term)
coefficient(t::MOI.ScalarAffineTerm) = t.coefficient
coefficient(t::MOI.VectorAffineTerm) = coefficient(t.scalar_term)
constrrows(::MOI.AbstractScalarSet) = 1
constrrows(s::MOI.AbstractVectorSet) = 1:MOI.dimension(s)
constrrows(instance::Optimizer, ci::CI{<:MOI.AbstractScalarFunction, <:MOI.AbstractScalarSet}) = 1
constrrows(instance::Optimizer, ci::CI{<:MOI.AbstractVectorFunction, MOI.Zeros}) = 1:instance.cone.eqnrows[constroffset(instance, ci)]
constrrows(instance::Optimizer, ci::CI{<:MOI.AbstractVectorFunction, <:MOI.AbstractVectorSet}) = 1:instance.cone.ineqnrows[constroffset(instance, ci)]
matrix(data::ModelData, s::ZeroCones) = data.b, data.IA, data.JA, data.VA
matrix(data::ModelData, s::Union{LPCones, MOI.SecondOrderCone, MOI.ExponentialCone}) = data.h, data.IG, data.JG, data.VG
matrix(instance::Optimizer, s) = matrix(instance.data, s)
MOIU.load_constraint(instance::Optimizer, ci, f::MOI.SingleVariable, s) = MOIU.load_constraint(instance, ci, MOI.ScalarAffineFunction{Float64}(f), s)

function MOIU.load_constraint(instance::Optimizer, ci, f::MOI.ScalarAffineFunction, s::MOI.AbstractScalarSet)
    a = sparsevec(variable_index_value.(f.terms), coefficient.(f.terms))
    # sparsevec combines duplicates with + but does not remove zeros created so we call dropzeros!
    dropzeros!(a)
    offset = constroffset(instance, ci)
    row = constrrows(s)
    i = offset + row
    # The ECOS format is b - Ax ∈ cone
    # so minus=false for b and minus=true for A
    setconstant = MOIU.getconstant(s)
    if s isa MOI.EqualTo
        instance.cone.eqsetconstant[offset] = setconstant
    else
        instance.cone.ineqsetconstant[offset] = setconstant
    end
    constant = f.constant - setconstant
    b, I, J, V = matrix(instance, s)
    b[i] = scalecoef(row, constant, false, s)
    append!(I, fill(i, length(a.nzind)))
    append!(J, a.nzind)
    append!(V, scalecoef(row, a.nzval, true, s))
end

MOIU.load_constraint(instance::Optimizer, ci, f::MOI.VectorOfVariables, s) = MOIU.load_constraint(instance, ci, MOI.VectorAffineFunction{Float64}(f), s)
orderval(val, s) = val
orderidx(idx, s) = idx
function MOIU.load_constraint(instance::Optimizer, ci, f::MOI.VectorAffineFunction, s::MOI.AbstractVectorSet)
    A = sparse(output_index.(f.terms), variable_index_value.(f.terms), coefficient.(f.terms))
    # sparse combines duplicates with + but does not remove zeros created so we call dropzeros!
    dropzeros!(A)
    I, J, V = findnz(A)
    offset = constroffset(instance, ci)
    rows = constrrows(s)
    if s isa MOI.Zeros
        instance.cone.eqnrows[offset] = length(rows)
    else
        instance.cone.ineqnrows[offset] = length(rows)
    end
    i = offset .+ rows
    # The ECOS format is b - Ax ∈ cone
    # so minus=false for b and minus=true for A
    b, Is, Js, Vs = matrix(instance, s)
    b[i] .= scalecoef(rows, orderval(f.constants, s), false, s)
    append!(Is, offset .+ orderidx(I, s))
    append!(Js, J)
    append!(Vs, scalecoef(I, V, true, s))
end

function MOIU.allocate_variables(instance::Optimizer, nvars::Integer)
    instance.cone = ConeData()
    VI.(1:nvars)
end

function MOIU.load_variables(instance::Optimizer, nvars::Integer)
    cone = instance.cone
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
    instance.data = ModelData(m, nvars, IA, JA, VA, b, IG, JG, VG, h, 0., c)
end

function MOIU.allocate(instance::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    instance.maxsense = sense == MOI.MAX_SENSE
end
function MOIU.allocate(::Optimizer, ::MOI.ObjectiveFunction,
                       ::MOI.Union{MOI.SingleVariable,
                                   MOI.ScalarAffineFunction{Float64}})
end

function MOIU.load(::Optimizer, ::MOI.ObjectiveSense, ::MOI.OptimizationSense)
end
function MOIU.load(optimizer::Optimizer, ::MOI.ObjectiveFunction,
                   f::MOI.SingleVariable)
    MOIU.load(optimizer,
              MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
              MOI.ScalarAffineFunction{Float64}(f))
end
function MOIU.load(instance::Optimizer, ::MOI.ObjectiveFunction,
                   f::MOI.ScalarAffineFunction)
    c0 = Vector(sparsevec(variable_index_value.(f.terms), coefficient.(f.terms),
                          instance.data.n))
    instance.data.objconstant = f.constant
    instance.data.c = instance.maxsense ? -c0 : c0
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
    res = solve_socp(prob)
    instance.sol = res
end

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

_dual(instance, ci::CI{<:MOI.AbstractFunction, <:ZeroCones}) = instance.sol.y
_dual(instance, ci::CI) = instance.sol.z
function MOI.get(instance::Optimizer, ::MOI.ConstraintDual, ci::CI{<:MOI.AbstractFunction, S}) where S <: MOI.AbstractSet
    offset = constroffset(instance, ci)
    rows = constrrows(instance, ci)
    scalecoef(rows, reorderval(_dual(instance, ci)[offset .+ rows], S), false, S)
end

MOI.get(instance::Optimizer, ::MOI.ResultCount) = 1
