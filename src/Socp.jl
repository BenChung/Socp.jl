module Socp
using LinearAlgebra
abstract type Cone end
struct POC <: Cone # positive orthant cone
	offs::Int
	dim::Int
end

struct SOC <: Cone # second order cone
	offs::Int
	dim::Int
end

struct Problem 
	#minimize c' x
	c::Vector{Float64} # dim n

	# s.t. A x = b
	A::Matrix{Float64} # dim m x n 
	b::Vector{Float64} # dim m

	# s.t. G x + s = h
	G::Matrix{Float64} # dim k x n
	h::Vector{Float64} # dim k

	# wrt cones
	cones::Vector{Cone}

	#dimensions
	n::Int
	m::Int
	k::Int

	Problem(c, A, b, G, h, cones) = begin
	    n = length(c)
	    m = size(A)[1]
	    @assert length(b) == m
	    @assert size(A)[2] == n 
	    @assert size(G)[2] == n
	    k = size(G)[1]
	    @assert length(h) == k
	    #todo: check that each variable appears in a type of cone only once
	    return new(c, A, b, G, h, cones, n, m, k)
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

include("scalings.jl")
include("vectors.jl")
include("mats.jl")
include("solver.jl")

end # module
