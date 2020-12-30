mutable struct DenseSolver <: KKTSolver{Scaling}
	k0::Vector{Float64}
	k1::Vector{Float64}
	k2::Vector{Float64}
	m0::Vector{Float64}
	n0::Vector{Float64}
	n1::Vector{Float64}
	GWiWi::Matrix{Float64}
	GWiWiG::Matrix{Float64}
	Li::Matrix{Float64}
	AtLi::Matrix{Float64}
	AtLiA::Matrix{Float64}
	GWiWiGherm::Hermitian{Float64, Array{Float64, 2}}
	AtLiAherm::Hermitian{Float64, Array{Float64, 2}}
	GWiWiGfact::Union{Nothing, Cholesky{Float64, Array{Float64, 2}}}
	AtLiAfact::Union{Nothing, Cholesky{Float64, Array{Float64, 2}}}
	eyetgt::Matrix{Float64}
	AA::Matrix{Float64}
	function DenseSolver(pr::Problem)
		k0, k1, k2 = zeros(pr.k), zeros(pr.k), zeros(pr.k)
		m0 = zeros(pr.m)
		n0, n1 = zeros(pr.n), zeros(pr.n)
		GWiWi = zeros(pr.n, pr.k)
		GWiWiG = zeros(pr.n, pr.n)
		GWiWiGherm = Hermitian(GWiWiG)
		Li = zeros(pr.n, pr.n)
		AtLi = zeros(pr.m, pr.n)
		AtLiA = zeros(pr.m, pr.m)
		AtLiAherm = Hermitian(AtLiA)
		eyetgt = Matrix{Float64}(I,pr.n, pr.n)
		AA = zeros(pr.n, pr.n)
		mul!(AA, pr.A', pr.A)
		return new(k0, k1, k2, m0, n0, n1, 
			GWiWi, GWiWiG, Li, AtLi, AtLiA, 
			GWiWiGherm, AtLiAherm, 
			nothing, nothing,
			eyetgt, AA)
	end
end

function setup_iter(ss::DenseSolver, pr::Problem{C,n,m,k,sing}, s::State, scaling::Scaling) where {C, n, m, k, sing}
	mul!(ss.GWiWi, pr.G', scaling.iWiW) #GWiWi = G'*W^-1 * W^-T 
	mul!(ss.GWiWiG, ss.GWiWi, pr.G) #ss.GWiWiG = G'*W^-1*W^-T*G 
	if sing # ss.GWiWiG will be not pos def if it wasn't before
		ss.GWiWiG .+= ss.AA # we know that aa's been populated by now
	end
	ss.GWiWiGfact = cholesky!(ss.GWiWiGherm)
	ldiv!(ss.Li, ss.GWiWiGfact, et) #iR = L^-T L^-1
	mul!(ss.ALi, pr.A, ss.Li)
	mul!(ss.AtLiA, ss.ALi, At)
	ss.AtLiAfact = cholesky!(ss.AtLiAherm)
end

function solve_kkt(ss::DenseSolver, pr::Problem{C,n,m,k,sing}, s::State, scaling::Scaling, 
				   dx::Vector{Float64}, dy::Vector{Float64}, dz::Vector{Float64}, ds::Vector{Float64}, 
				   cx::Vector{Float64}, cy::Vector{Float64}, cz::Vector{Float64}, cs::Vector{Float64}) where {C, n, m, k, sing}
	W,iW,l = scaling.W,scaling.iW,scaling.l
	et = ss.eyetgt
	aa = ss.AA

	iprod!(pr.cones, ss.k0, l, ds)
	scale!(pr.cones, scaling, ss.k0, ss.k1)
	bx = dx
	by = dy
	ss.k2 .= dz .- ss.k1
	mul!(ss.n0, ss.GWiWi, ss.k2)
	ss.n0 .+= bx
	At = pr.A'
	if ss.issng
		ss.n0 .+= At*by
	end

	mul!(ss.m0, ss.AtLi, ss.n0)
	ss.m0 .-= by
	ldiv!(cy, ss.AtLiAfact, ss.m0)
	if ss.issng
		ss.m0 .= by .- cy
	else
		ss.m0 .= .-cy
	end
	mul!(ss.n1, At, ss.m0)
	ss.n0 .+= ss.n1
	mul!(cx, ss.Li, ss.n0)
	mul!(ss.k1, pr.G, cx)
	ss.k1 .-= ss.k2
	mul!(cz, scaling.iWiW, ss.k1)
	scale!(pr.cones, scaling, cz, ss.k1)
	ss.k0 .-= ss.k1
	scale!(pr.cones, scaling, ss.k0, cs)
end