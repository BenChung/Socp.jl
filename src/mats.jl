@unroll function max_step(cones::Tuple{Vararg{Cone}}, x)
	maxim = typemin(Float64)
	@unroll for cone in cones 
		val = max_step(cone, x)
		if val > maxim 
			maxim = val
		end
	end
	return maxim
end

function max_step(cone::POC{dim}, x) where dim
	minim = typemax(Float64)
	for i=cti(cone, 1):cti(cone,dim)
		if x[i] < minim 
			minim = x[i]
		end
	end
	return -minim
end

function max_step(cone::SOC{dim}, x) where dim
	sqnrm = 0.0
	for i=cti(cone, 2):cti(cone,dim)
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

@unroll function scmax(cones::Tuple{Vararg{Cone}}, l, x)
	mxv = typemin(Float64)
	@unroll for cone in cones
		val = scmax(cone, l, x)
		if val > mxv
			mxv = val
		end
	end
	return mxv
end

function scmax(c::POC{dim}, l, x) where dim
	mxv = typemin(Float64)
	for i=cti(c, 1):cti(c, dim)
		val = -x[i]/l[i]
		if val > mxv
			mxv = val
		end
	end
	return mxv
end

function scmax(c::SOC{dim}, li, xi) where dim
	# a = 1/sqrt(ln[1]^2 - sum(ln[2:end].^2))
	i1 = cti(c,1)
	ai = li[i1]^2
	for ii = cti(c,2):cti(c,dim)
		ai -= li[ii]^2
	end
	a = 1/sqrt(ai)

	# r1 = a*ln[1]*x[1] - dot(a*ln[2:end], x[2:end])
	r1 = a*li[i1]*xi[i1]
	for ii = cti(c,2):cti(c,dim)
		r1 -= a*li[ii]*xi[ii]
	end

	# r2 = a .* (x[2:end] - cst * a*ln[2:end])
	cst = (r1 + xi[i1])/(a*li[i1] + 1)
	r2s = 0.0
	for ii = cti(c,2):cti(c,dim)
		r2s += (a * (xi[ii] - cst * a * li[ii]))^2
	end
	return sqrt(r2s) - a * r1 # norm(r2) - a*r1
end
