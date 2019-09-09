function max_step(cones::Vector{Cone}, x)
	maxim = typemin(Float64)
	for cone in cones 
		val = max_step(cone, x)
		if val > maxim 
			maxim = val
		end
	end
	return maxim
end

function max_step(cone::POC, x)
	minim = typemax(Float64)
	for i=cti(cone, 1):cti(cone,cone.dim)
		if x[i] < minim 
			minim = x[i]
		end
	end
	return -minim
end

function max_step(cone::SOC, x)
	sqnrm = 0.0
	for i=cti(cone, 2):cti(cone,cone.dim)
		sqnrm += x[i]^2
	end
	return sqrt(sqnrm) - x[cti(cone, 1)]
end

function compute_step(cones, l, ds, dz)
	mxs = scmax(cones, l, ds) # max_step(cones, scale2(cones, l, ds))
	mxz = scmax(cones, l, dz) # max_step(cones, scale2(cones, l, dz))
	t = max(mxs,mxz,0.0)
	if t == 0.0
		step = 1.0
	else 
		step = min(1.0, 1.0/t)
	end
	return step
end
function scale2(cones::Vector{Cone}, l, x)
	return vcat((scale2(c,l,x) for c in cones)...)
end

function scale2(c::POC, l, x)
	rng = cti(c, 1):cti(c, c.dim)
	return x[rng]/l[rng]
end

function scale2(c::SOC, li, xi)
	ln = @view li[cti(c,1):cti(c,c.dim)]
	x = @view xi[cti(c,1):cti(c,c.dim)]
	a = 1/sqrt(ln[1]^2 - sum(ln[2:end].^2))
	l = ln .* a
	r1 = l[1]*x[1] - dot(l[2:end], x[2:end])
	cst = (r1 + x[1])/(l[1] + 1)
	r2 = (x[2:end] - cst * l[2:end])
	return a.*[r1; r2]
end

function scmax(cones::Vector{Cone}, l, x)
	mxv = typemin(Float64)
	for cone in cones
		val = scmax(cone, l, x)
		if val > mxv
			mxv = val
		end
	end
	return mxv
end

function scmax(c::POC, l, x)
	mxv = typemin(Float64)
	for i=cti(c, 1):cti(c, c.dim)
		val = -x[i]/l[i]
		if val > mxv
			mxv = val
		end
	end
	return mxv
end

function scmax(c::SOC, li, xi)
	# a = 1/sqrt(ln[1]^2 - sum(ln[2:end].^2))
	i1 = cti(c,1)
	ai = li[i1]^2
	for ii = cti(c,2):cti(c,c.dim)
		ai -= li[ii]^2
	end
	a = 1/sqrt(ai)

	# r1 = a*ln[1]*x[1] - dot(a*ln[2:end], x[2:end])
	r1 = a*li[i1]*xi[i1]
	for ii = cti(c,2):cti(c,c.dim)
		r1 -= a*li[ii]*xi[ii]
	end

	# r2 = a .* (x[2:end] - cst * a*ln[2:end])
	cst = (r1 + xi[i1])/(a*li[i1] + 1)
	r2s = 0.0
	for ii = cti(c,2):cti(c,c.dim)
		r2s += (a * (xi[ii] - cst * a * li[ii]))^2
	end
	return sqrt(r2s) - a * r1 # norm(r2) - a*r1
end

const t = 0.9
const maxiters = 1000

function line_search(cones, z, dz, s, ds)
	h = 1.0
	iter = 0
	cdz, cds = similar(z), similar(s)
	cdz .= dz
	cds .= ds
	while (!cgt(cones, z, cdz) || !cgt(cones, s, cds)) && iter < maxiters
		rmul!(cdz, t)
		rmul!(cds, t)
		h *= t
		iter += 1
	end
	return h
end

function line_search_scaled(cones, scaling, dz, ds)	
	h = 1.0
	iter = 0
	W,iW,l = scaling.W,scaling.iW,scaling.l
	sds, sdz = iW' * ds/0.99, W * dz/0.99
	while (!cgt(cones, l, sds) || !cgt(cones, l, sdz)) && iter < maxiters
		rmul!(sds, t)
		rmul!(sdz, t)
		h *= t
		iter += 1
	end
	return h
end

mutable struct KKTState
	ipr::Vector{Float64}
	temp1::Vector{Float64}
	temp2::Vector{Float64}
	temp3::Vector{Float64}
	prf1::Matrix{Float64}
	prf2::Vector{Float64}
	prf3::Matrix{Float64}
	prf4::Vector{Float64}
	prf5::Matrix{Float64}
	iR::Matrix{Float64}
	siR::Matrix{Float64}
	hp3::Hermitian
	hp5::Hermitian
	eyetgt::Matrix{Float64}
	iL::Matrix{Float64}
	issng::Bool
	AA::Matrix{Float64}
	function KKTState(pr::Problem)
		temp3 = zeros(pr.m)
		prf3 = zeros(pr.n, pr.n)
		prf5 = zeros(pr.m, pr.m)
		AA = zeros(pr.n, pr.n)
		return new(zeros(pr.k), zeros(pr.k), zeros(pr.k), temp3, zeros(pr.n, pr.k), zeros(pr.n), 
			prf3, zeros(pr.n), prf5, zeros(pr.n, pr.n), zeros(pr.m, pr.n), Hermitian(prf3), Hermitian(prf5), 
			Matrix{Float64}(I,pr.n, pr.n), zeros(pr.n, pr.n), false, AA)
	end
end

function solve_kkt(pr::Problem, s::State, scaling::Scaling, 
				   dx::Vector{Float64}, dy::Vector{Float64}, dz::Vector{Float64}, ds::Vector{Float64}, 
				   cx::Vector{Float64}, cy::Vector{Float64}, cz::Vector{Float64}, cs::Vector{Float64}, mehrotra::Bool,
		ss::KKTState)
	n,m,k = pr.n,pr.m,pr.k
	W,iW,l = scaling.W,scaling.iW,scaling.l


	ipr = ss.ipr
	temp1 = ss.temp1
	temp2 = ss.temp2
	temp3 = ss.temp3
	prf1 = ss.prf1
	prf2 = ss.prf2
	prf3 = ss.prf3
	prf4 = ss.prf4
	prf5 = ss.prf5
	iR = ss.iR
	siR = ss.siR
	prf0 = scaling.iWiW
	et = ss.eyetgt
	iL = ss.iL
	aa = ss.AA

	if mehrotra 
		mul!(prf1, pr.G', prf0) #prf1 = G'*W^-1 * W^-T 
		mul!(prf3, prf1, pr.G) #prf3 = G'*W^-1*W^-T*G 
		if ss.issng # prf3 will be not pos def if it wasn't before
			prf3 .+= aa # we know that aa's been populated by now
			L = cholesky!(ss.hp3)
		else
			try
				L = cholesky!(ss.hp3)
			catch e 
				if isa(e, PosDefException) || isa(e, SingularException)
					mul!(aa, pr.A', pr.A)
					prf3 .+= aa
					L = cholesky!(ss.hp3)
					ss.issng = true
				else
					rethrow(e)
				end
			end
		end
		ldiv!(iR, L, et) #iR = L^-T L^-1
		mul!(siR, pr.A, iR)
	end

	iprod!(pr.cones, ipr, l, ds)
	scale!(pr.cones, scaling, ipr, temp1)
	bx = dx
	by = dy
	temp2 .= dz .- temp1
	mul!(prf2, prf1, temp2)
	prf2 .+= bx
	At = pr.A'
	if ss.issng
		prf2 .+= At*by
	end

	mul!(prf5, siR, At)
	yL = cholesky!(ss.hp5)

	mul!(temp3, siR, prf2)
	temp3 .-= by
	ldiv!(cy, yL, temp3)
	if ss.issng
		temp3 .= by .- cy
	else
		temp3 .= .-cy
	end
	mul!(prf4, At, temp3)
	prf2 .+= prf4
	mul!(cx, iR, prf2)
	mul!(temp1, pr.G, cx)
	temp1 .-= temp2
	mul!(cz, prf0, temp1)
	scale!(pr.cones, scaling, cz, temp1)
	ipr .-= temp1
	scale!(pr.cones, scaling, ipr, cs)
end
