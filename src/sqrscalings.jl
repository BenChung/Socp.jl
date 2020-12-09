struct SqrScaling{dim}
	iWiW::Diagonal{Float64}
	l::Vector{Float64}

	us :: Vector{Vector{Float64}}
	vs :: Vector{Vector{Float64}}
	#per cone
	mu::Vector{Float64}	
	wbs::Vector{Float64} # sqrt(s/z) for POC, wb for SOC
	function SqrScaling(p::Problem) 
		return SqrScaling(Diagonal(zeros(p.k)), zeros(p.k), p.cones, p.n)
	end
	function SqrScaling(iWiW,l,cones, id) 
		psz = cones[end].offs + conedim(cones[end])
		return new{id}(iWiW, l, 
			[zeros(psz) for cone in cones],
			[zeros(psz) for cone in cones],
			zeros(length(cones)), zeros(psz))
	end
end

function compute_sqscaling(cone::POC{dim}, cind, scaling::SqrScaling, s, z) where dim
	for i = 1:dim
		ii = cti(cone, i)
		scaling.iWiW[ii,ii] = z[ii] / s[ii]
		scaling.l[ii] = sqrt(s[ii] * z[ii])
		scaling.wbs[ii] = sqrt(s[ii] / z[ii])
	end
end

function modify_factors!(cone::POC{dim}, cind, factor::SuiteSparse.CHOLMOD.Factor, 
						 scaling::SqrScaling, g::SparseMatrixCSC) where dim
	# we don't need to do anything here
end


function compute_sqscaling(c::SOC{dim}, cind, scaling::SqrScaling, s, z) where dim
	function cone_prod(vect)
		bsum = vect[1] * vect[1]
		for i=2:dim
			bsum -= vect[i] * vect[i]
		end
		return bsum
	end
	sbk = MVector{dim}(@view s[c.offs+1:c.offs+dim])
	zbk = MVector{dim}(@view z[c.offs+1:c.offs+dim])
	sprod, zprod = cone_prod(sbk), cone_prod(zbk)
	sbk .*= 1/sqrt(sprod)
	zbk .*= 1/sqrt(zprod)

	nsum = sum(zbk .* sbk)
	gamma = sqrt((1 + nsum)/2)

	# compute scaling point
	wb = vcat(sbk[1] + zbk[1], (@view sbk[2:end]) .- (@view zbk[2:end])) ./ (2*gamma)

	mu = sqrt(sqrt(sprod/zprod))
	scaling.mu[cind] = mu

	# compute scaling matrix 
	inusq = 1/sqrt(sprod/zprod)
	inu = 1/sqrt(sqrt(sprod/zprod))
	wb0 = wb[1]
	wb1 = @view wb[2:end]
	wb1sq = sum(wb1 .* wb1)
	cv = -(1 + wb0 + wb1sq/(1+wb0))
	d = 1 + 2/(1+wb0) + wb1sq/((1+wb0)*(1+wb0))
	a = (wb0*wb0 + wb1sq - cv*cv*wb1sq/(1+d*wb1sq))/2
	u0 = sqrt(wb0*wb0 + wb1sq - a)
	u1 = cv/u0
	u = inu .* vcat(u0, u1 .* wb1)
	v1 = sqrt(cv*cv/(u0*u0) - d)
	v = inu .* vcat(0, v1 .* wb1)

	scaling.iWiW[cti(c, 1), cti(c, 1)] = a*inusq
	for i=2:dim
		scaling.iWiW[cti(c, i),cti(c, i)] = inusq
	end
	scaling.us[cind][c.offs+1:c.offs+dim] .= u
	scaling.vs[cind][c.offs+1:c.offs+dim] .= v

	# compute scaling variable
	ziv,siv = zbk[1], sbk[1]
	tmv1 = sqrt(sqrt(sprod)*sqrt(zprod))
	mult = tmv1/(ziv + siv + 2*gamma)
	for i=2:dim
		scaling.l[cti(c, i)] = (sbk[i]*(gamma + ziv) + zbk[i]*(gamma + siv))*mult
	end
	(@view scaling.wbs[cti(c,1):cti(c,conedim(c))]) .= wb
	scaling.l[cti(c,1)] = gamma*tmv1
end

function modify_factors!(cone::SOC{dim}, cind, factor::SuiteSparse.CHOLMOD.Factor, 
						 scaling::SqrScaling{odim}, g::SparseMatrixCSC) where {dim, odim}
	temp = zeros(MVector{odim})
	mul!(temp, g', scaling.us[cind])
	SuiteSparse.CHOLMOD.lowrankupdate!(factor, temp)
	mul!(temp, g', scaling.vs[cind])
	SuiteSparse.CHOLMOD.lowrankdowndate!(factor, temp)
end

@unroll function compute_sqscaling(cones::Tuple{Vararg{Cone}}, scaling::SqrScaling, s, z)
	# assume that cones is in the order POC ... POC SOC ... SOC
	i=1
	@unroll for cone in cones
		compute_sqscaling(cone, i, scaling, s, z)
		i += 1
	end
	return scaling
end 

@unroll function modify_factors!(cones::Tuple{Vararg{Cone}}, factor::SuiteSparse.CHOLMOD.Factor, 
						 scaling::SqrScaling, g::SparseMatrixCSC) where dim
	i=1
	@unroll for cone in cones
		modify_factors!(cone, i, factor, scaling, g)
		i += 1
	end
end

@unroll function compute_full_scaling(cones::Tuple{Vararg{Cone}}, scaling::SqrScaling)
	i=1
	oiWiW = zeros(MMatrix{length(scaling.l), length(scaling.l)})
	@unroll for cone in cones
		compute_full_scaling(cone, i, scaling, oiWiW)
		i += 1
	end
	return oiWiW
end

function compute_full_scaling(cone::SOC{dim}, cind, scaling::SqrScaling, output) where dim 
	output[cone.offs+1:cone.offs+dim, cone.offs+1:cone.offs+dim] .= scaling.iWiW[cone.offs+1:cone.offs+dim, cone.offs+1:cone.offs+dim]
	BLAS.gemm!('N', 'T', 1.0, scaling.us[cind], scaling.us[cind], 1.0, output)
	BLAS.gemm!('N', 'T', -1.0, scaling.vs[cind], scaling.vs[cind], 1.0, output)
end

function compute_full_scaling(cone::POC{dim}, cind, scaling::SqrScaling, output) where dim
	output[cone.offs+1:cone.offs+dim, cone.offs+1:cone.offs+dim] .= scaling.iWiW[cone.offs+1:cone.offs+dim, cone.offs+1:cone.offs+dim]
end