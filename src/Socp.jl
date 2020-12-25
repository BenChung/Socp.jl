module Socp
using LinearAlgebra
using SparseArrays
using StaticArrays
using Unrolled
using SuiteSparse

abstract type Cone{D} end
struct POC{D} <: Cone{D} # positive orthant cone
	offs::Int
	POC(offs,dim) = new{dim}(offs)
end
struct SOC{D} <: Cone{D} # second order cone
	offs::Int
	SOC(offs,dim) = new{dim}(offs)
end
@generated conedim(::POC{d}) where d = d 
@generated conedim(::SOC{d}) where d = d

struct Problem{C <: Tuple{Vararg{Cone}}, n, m, k, sing}
	#minimize c' x
	c::Vector{Float64} # dim n

	# s.t. A x = b
	A::SparseMatrixCSC{Float64, Int} # dim m x n 
	b::Vector{Float64} # dim m

	# s.t. G x + s = h
	G::SparseMatrixCSC{Float64, Int} # dim k x n
	h::Vector{Float64} # dim k

	# wrt cones
	cones::C

	#dimensions
	n::Int
	m::Int
	k::Int

	Problem(c, A, b, G, h, cones::C) where C <: Tuple{Vararg{Cone}} = begin
	    n = length(c)
	    m = size(A)[1]
	    @assert length(b) == m
	    @assert size(A)[2] == n 
	    @assert size(G)[2] == n
	    k = size(G)[1]
	    @assert length(h) == k

	    sing = false
	    try 
	    	cholesky(G' * G)
		catch e
			if isa(e, PosDefException) || isa(e, SingularException) || isa(e, ArgumentError)
				sing = true
			end
		end
	    #todo: check that each variable appears in a type of cone only once
	    return new{C, n, m, k, sing}(c, A, b, G, h, cones, n, m, k)
	end
end

struct State
	x::Vector{Float64}
	y::Vector{Float64}
	z::Vector{Float64}
	s::Vector{Float64}
	State(prob::Problem, x, y, z, s) = begin
		#todo: initalization proc
		@assert length(x) == prob.n 
		@assert length(y) == prob.m
		@assert length(z) == prob.k
		@assert length(s) == prob.k
		return new(x,y,z,s)
	end
end

include("cholutils.jl")
include("sqrscalings.jl")
include("scalings.jl")
include("vectors.jl")
include("mats.jl")
include("solver.jl")
include("moi.jl")

end # module
