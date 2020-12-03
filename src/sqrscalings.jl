struct SqrScaling 
	iWiW::Matrix{Float64}
	l::Vector{Float64}

	#per cone
	mu::Vector{Float64}	
	wbs::Vector{Float64} # sqrt(s/z) for POC, wb for SOC
	function SqrScaling(p::Problem) 
		return SqrScaling(zeros(p.k,p.k), zeros(p.k), p.cones)
	end
	function SqrScaling(iWiW,l,cones) 
		return new(iWiW, l, zeros(length(cones)), zeros(cones[end].offs + conedim(cones[end])))
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
	iWiW = zeros(MMatrix{dim,dim})
	wb0 = wb[1]
	wb1 = @view wb[2:end]
	wb1sq = sum(wb1 .* wb1)
	cv = -(1 + wb0 + wb1sq/(1+wb0))
	d = 1 + 2/(1+wb0) + wb1sq/((1+wb0)*(1+wb0))
	a = (wb0*wb0 + wb1sq - cv*cv*wb1sq/(1+d*wb1sq))/2
	u0 = sqrt(wb0*wb0 + wb1sq - a)
	u1 = cv/u0
	u = vcat(u0, u1 .* wb1)
	v1 = sqrt(cv*cv/(u0*u0) - d)
	v = vcat(0, v1 .* wb1)

	iWiW[1, 1] = a
	for i=2:dim
		iWiW[i,i] = 1
	end
	inusq = 1/sqrt(sprod/zprod)
	BLAS.gemm!('N', 'T', inusq, u, u, inusq, iWiW)
	BLAS.gemm!('N', 'T', -inusq, v, v, 1.0, iWiW)
	(@view scaling.iWiW[c.offs+1:c.offs+dim, c.offs+1:c.offs+dim]) .= iWiW

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
@unroll function compute_sqscaling(cones::Tuple{Vararg{Cone}}, scaling::SqrScaling, s, z)
	# assume that cones is in the order POC ... POC SOC ... SOC
	i=1
	@unroll for cone in cones
		compute_sqscaling(cone, i, scaling, s, z)
		i += 1
	end
	return scaling
end 