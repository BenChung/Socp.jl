@inline function cti(cone::Cone, i)
	return i + cone.offs
end

#e vector generation

function make_e!(cones::Union{Vector{Cone}, Vector{Cone{T}} where T}, r)
	for c in cones 
		make_e!(c, r)
	end
end

function make_e!(cone::POC{dim}, r) where dim
	for i=cti(cone, 1):cti(cone, dim)
		r[i] = 1.0
	end
end

function make_e!(cone::SOC{dim}, r) where dim
	r[cti(cone,1)] = 1.0 
	for i=cti(cone, 2):cti(cone, dim)
		r[i] = 0.0
	end
end

function make_e(cone::Union{SOC,POC})
	res = zeros(cone.offs + conedim(cone))
	make_e!(cone, res)
	return @view res[cti(cone, 1):cti(cone, conedim(cone))]
end

function make_e(cones::Union{Vector{Cone},Vector{Cone{d}} where d})
	tvpres = zeros(cones[end].offs + conedim(cones[end]))
	make_e!(cones, tvpres)
	return tvpres
end

# vector products

function vprod(cone::POC{dim}, u, v) where dim
	res = zeros(dim+cone.offs)
	vprod!(cone, res, u, v)
	return res
end

function vprod(cone::SOC{dim}, u, v) where dim
	res = zeros(dim+cone.offs)
	vprod!(cone, res, u, v)
	return res
end

function vprod(cones::Union{Vector{Cone},Vector{Cone{d}} where d}, u, v)
	tvpres = zeros(cones[end].offs + conedim(cones[end]))
	vprod!(cones, tvpres, u, v)
	return tvpres
end

function vprod!(cone::POC{dim}, t, u, v) where dim
	for i in cti(cone, 1):cti(cone, dim)
		t[i] = u[i]*v[i]
	end
end

function vprod!(cone::SOC{dim}, t, u, v) where dim
	i1 = cti(cone, 1)
	t[i1] = 0.0
	for i = cti(cone, 1):cti(cone, dim)
		t[i1] += u[i] * v[i]
	end
	iu = u[i1]
	iv = v[i1]
	for i in cti(cone, 2):cti(cone, dim)
		t[i] = iu * v[i] + iv * u[i]
	end 
end

function vprod!(cones::Union{Vector{Cone},Vector{Cone{d}} where d}, t, u, v)
	for cone in cones 
		vprod!(cone, t, u, v)
	end
end

# inverse vector products

function iprod(cone::Cone{dim}, lam::Vector{Float64}, v::Vector{Float64}) where dim
	tres = zeros(dim+cone.offs)
	iprod!(cone, tres, lam, v)
	return tres
end

function iprod(cones::Union{Vector{Cone},Vector{Cone{d}} where d}, u, v)
	tres = zeros(conedim(cones[end])+cones[end].offs)
	for c in cones 
		iprod!(c, tres, u, v)
	end
	return tres
end

function iprod!(cone::POC{dim}, t::Vector{Float64}, lam::Vector{Float64}, v::Vector{Float64}) where dim
	for i=cti(cone, 1):cti(cone, dim)
		t[i] = v[i]/lam[i]
	end
end

function iprod!(cone::SOC{dim}, t, lam::Vector{Float64}, v::Vector{Float64}) where dim
	i1 = cti(cone, 1)
	l1 = lam[i1] 
	a = l1^2 
	for i=cti(cone, 2):cti(cone, dim)
		a -= lam[i] * lam[i]
	end
	for i=i1:cti(cone, dim)
		t[i] = 0
	end
	t[i1] += v[i1] * l1/a
	for j=cti(cone, 2):cti(cone, dim)
		t[i1] -= v[j] * lam[j]/a
	end
	for i=cti(cone, 2):cti(cone, dim)
		t[i] -= v[i1] * lam[i]/a
		for j=cti(cone, 2):cti(cone, dim)
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

function cgt(c::POC{dim}, x, dx) where dim
	for i in cti(c, 1):cti(c, dim)
		if x[i] + dx[i] < 0
			return false
		end
	end
	return true
end

function cgt(c::SOC{dim}, x, dx) where dim
	tot = 0.0
	for i in cti(c, 2):cti(c, dim)
		val = x[i] + dx[i]
		tot += val * val
	end
	return sqrt(tot) <= x[cti(c, 1)] + dx[cti(c, 1)]
end

function cgt(cs::Union{Vector{Cone},Vector{Cone{d}} where d}, x, dx)
	for c in cs 
		if !cgt(c, x, dx) 
			return false
		end
	end
	return true
end

# degree

function deg(c::POC{d}) where d 
	return d
end

function deg(c::SOC)
	return 1
end

function deg(cs::Union{Vector{Cone},Vector{Cone{d}} where d})
	dg = 0
	for c in cs 
		dg += deg(c)
	end
	return dg
end