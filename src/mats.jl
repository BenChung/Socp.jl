function max_step(cones::Vector{Cone}, x)
	return maximum(max_step(cone, x) for cone in cones)
end

function max_step(cone::POC, x)
	return -minimum(x[cti(cone, 1):cti(cone,cone.dim)])
end

function max_step(cone::SOC, x)
	return norm(x[cti(cone, 2):cti(cone,cone.dim)]) - x[cti(cone, 1)]
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

struct SolveState
	ipr::Vector{Float64}
	temp1::Vector{Float64}
	temp2::Vector{Float64}
	prf1::Matrix{Float64}
	prf2::Vector{Float64}
	prf3::Matrix{Float64}
	prf4::Vector{Float64}
	iR::Matrix{Float64}
	siR::Matrix{Float64}
	SolveState(pr::Problem) = 
		new(zeros(pr.k), zeros(pr.k), zeros(pr.k), zeros(pr.n, pr.k), zeros(pr.n), 
			zeros(pr.n, pr.n), zeros(pr.n), zeros(pr.n, pr.n), zeros(pr.m, pr.n))
end

function solve_kkt(pr::Problem, s::State, scaling, dx, dy, dz, ds, cx, cy, cz, cs;
		ss::SolveState = SolveState(pr))
	n,m,k = pr.n,pr.m,pr.k
	W,iW,l = scaling.W,scaling.iW,scaling.l


	ipr = ss.ipr
	temp1 = ss.temp1
	temp2 = ss.temp2
	prf1 = ss.prf1
	prf2 = ss.prf2
	prf3 = ss.prf3
	prf4 = ss.prf4
	iR = ss.iR
	siR = ss.siR

	iprod!(pr.cones, ipr, l, ds)
	mul!(temp1, W', ipr)
	bx = dx
	by = dy
	temp2 .= dz .- temp1

	prf0 = scaling.iWiW
	mul!(prf1, pr.G', prf0)
	mul!(prf3, prf1, pr.G)
	L = cholesky(Hermitian(prf3)).L
	iL = inv(L)
	mul!(iR, iL', iL)
	mul!(siR, pr.A, iR)
	mul!(prf2, prf1, temp2)
	prf2 .+= bx

	At = pr.A'
	cy .= (siR*At)\(siR*prf2 - by)
	mul!(prf4, At, cy)
	prf2 .-= prf4
	mul!(cx, iR, prf2)
	mul!(temp1, pr.G, cx)
	temp1 .-= temp2
	mul!(cz, prf0, temp1)
	mul!(cs, W', (ipr - W*cz))

	return cx,cy,cz,cs
end
