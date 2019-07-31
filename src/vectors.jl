@inline function cti(cone::Cone, i)
	return i + cone.offs
end

#e vector generation

function make_e(cone::POC)
	return ones(cone.dim)
end

function make_e(cone::SOC)
	return [1; zeros(cone.dim-1)]
end

function make_e(cones::Vector{Cone})
	return reduce(vcat, make_e(c) for c in cones)
end

# vector products

function vprod(cone::POC, u, v)
	res = zeros(cone.dim+cone.offs)
	vprod!(cone, res, u, v)
	return res
end

function vprod(cone::SOC, u, v)
	res = zeros(cone.dim+cone.offs)
	vprod!(cone, res, u, v)
	return res
end

function vprod(cones::Vector{Cone}, u, v)
	tvpres = zeros(cones[end].offs + cones[end].dim)
	vprod!(cones, tvpres, u, v)
	return tvpres
end

function vprod!(cone::POC, t, u, v)
	for i in cti(cone, 1):cti(cone, cone.dim)
		t[i] = u[i]*v[i]
	end
end

function vprod!(cone::SOC, t, u, v)
	i1 = cti(cone, 1)
	t[i1] = 0.0
	for i = cti(cone, 1):cti(cone, cone.dim)
		t[i1] += u[i] * v[i]
	end
	iu = u[i1]
	iv = v[i1]
	for i in cti(cone, 2):cti(cone, cone.dim)
		t[i] = iu * v[i] + iv * u[i]
	end 
end

function vprod!(cones::Vector{Cone}, t, u, v)
	for cone in cones 
		vprod!(cone, t, u, v)
	end
end

# inverse vector products

function iprod(cone::Cone, lam::Vector{Float64}, v::Vector{Float64})
	tres = zeros(cone.dim+cone.offs)
	iprod!(cone, tres, lam, v)
	return tres
end

function iprod(cones::Vector{Cone}, u, v)
	tres = zeros(cones[end].dim+cones[end].offs)
	for c in cones 
		iprod!(c, tres, u, v)
	end
	return tres
end

function iprod!(cone::POC, t::Vector{Float64}, lam::Vector{Float64}, v::Vector{Float64}) 
	for i=cti(cone, 1):cti(cone, cone.dim)
		t[i] = v[i]/lam[i]
	end
end

function iprod!(cone::SOC, t, lam::Vector{Float64}, v::Vector{Float64})
	i1 = cti(cone, 1)
	l1 = lam[i1] 
	a = l1^2 
	for i=cti(cone, 2):cti(cone, cone.dim)
		a -= lam[i] * lam[i]
	end
	len = cone.dim
	for i=i1:cti(cone, cone.dim)
		t[i] = 0
	end
	t[i1] += v[i1] * l1/a
	for j=cti(cone, 2):cti(cone, len)
		t[i1] -= v[j] * lam[j]/a
	end
	for i=cti(cone, 2):cti(cone, len)
		t[i] -= v[i1] * lam[i]/a
		for j=cti(cone, 2):cti(cone, len)
			t[i] += v[j] * ((i == j ? a : 0.0) + lam[i]*lam[j])/(l1*a)
		end
	end
end

function iprod!(cones::Vector{Cone}, t, u, v)
	for cone in cones 
		iprod!(cone, t, u, v)
	end
end


# per-cone gt defns

function cgt(c::POC, x, dx) 
	for i in cti(c, 1):cti(c, c.dim)
		if x[i] + dx[i] < 0
			return false
		end
	end
	return true
end

function cgt(c::SOC, x, dx)
	tot = 0.0
	for i in cti(c, 2):cti(c, c.dim)
		val = x[i] + dx[i]
		tot += val * val
	end
	return sqrt(tot) <= x[cti(c, 1)] + dx[cti(c, 1)]
end

function cgt(cs::Vector{Cone}, x, dx)
	for c in cs 
		if !cgt(c, x, dx) 
			return false
		end
	end
	return true
end

# degree

function deg(c::POC) 
	return c.dim
end

function deg(c::SOC)
	return 1
end

function deg(cs::Vector{Cone})
	return sum(deg.(cs))
end