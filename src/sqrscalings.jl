function allocate_nzs(mat, bv)
	ptm = sparse(mat)
	ptm.nzval .= 1.0 # set all the nonzeros to 1
	res = sparse(ptm * bv)
	return res
end

struct SqrScaling{dim}
	iWiW::Diagonal{Float64, Vector{Float64}}
	iW::Diagonal{Float64, Vector{Float64}}
	l::Vector{Float64}

	us :: Vector{Vector{Float64}}
	vs :: Vector{Vector{Float64}}
	#per cone
	mu::Vector{Float64}	
	wbs::Vector{Float64} # sqrt(s/z) for POC, wb for SOC

	# workspaces for scaling functions
	ws_1::Vector{Vector{Float64}}
	ws_2::Vector{Vector{Float64}}
	ws_3::Vector{Vector{Float64}}
	ws_4::Vector{Vector{Float64}}
	ws_5::Vector{SparseMatrixCSC{Float64, Int}}
	ws_6::Vector{SuiteSparse.CHOLMOD.Sparse{Float64}}
	function SqrScaling(p::Problem) 
		return SqrScaling(Diagonal(zeros(p.k)), Diagonal(zeros(p.k)), zeros(p.k), p.cones, p.n, p.G)
	end
	function SqrScaling(iWiW, iW,l,cones, id, G) 
		psz = cones[end].offs + conedim(cones[end])
		ws5 = [allocate_nzs(G', collect(if cone.offs <= i && i <= cone.offs+conedim(cone) 1.0 else 0.0 end for i in 1:psz)) for cone in cones]
		ws6 = [SuiteSparse.CHOLMOD.Sparse(exv) for exv in ws5]
		return new{id}(iWiW, iW, l, 
			[zeros(psz) for cone in cones],
			[zeros(psz) for cone in cones],
			zeros(length(cones)), zeros(psz),
			[alloc_ws1(cone, psz) for cone in cones],
			[alloc_ws2(cone, psz) for cone in cones],
			[zeros(id) for cone in cones],
			[zeros(conedim(cone)) for cone in cones],
			ws5, ws6)
	end
end

function alloc_ws1(cone::POC{dim}, total) where dim return zeros(0) end 
function alloc_ws2(cone::POC{dim}, total) where dim return zeros(0) end 
function alloc_ws1(cone::SOC{dim}, total) where dim return zeros(total) end 
function alloc_ws2(cone::SOC{dim}, total) where dim return zeros(total) end 

function compute_sqscaling(cone::POC{dim}, cind, scaling::SqrScaling, s, z) where dim
	for i = 1:dim
		ii = cti(cone, i)
		scaling.iWiW[ii,ii] = z[ii] / s[ii]
		scaling.iW[ii,ii] = sqrt(z[ii] / s[ii])
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
	sbk = scaling.ws_1[cind]
	zbk = scaling.ws_2[cind]
	sbk[1:dim] .= @view s[c.offs+1:c.offs+dim]
	zbk[1:dim] .= @view z[c.offs+1:c.offs+dim]
	sprod, zprod = cone_prod(sbk), cone_prod(zbk)
	sbk .*= 1/sqrt(sprod)
	zbk .*= 1/sqrt(zprod)

	nsum = 0.0
	for i=1:dim 
		nsum += zbk[i] * sbk[i]
	end
	gamma = sqrt((1 + nsum)/2)

	# compute scaling point
	wb = scaling.ws_4[cind]
	wb[1] = (sbk[1] + zbk[1]) / (2*gamma)
	for i=2:dim
		wb[i] = (sbk[i] - zbk[i]) / (2*gamma)
	end

	mu = sqrt(sqrt(sprod/zprod))
	scaling.mu[cind] = mu

	# compute scaling matrix 
	inusq = 1/sqrt(sprod/zprod)
	inu = 1/sqrt(sqrt(sprod/zprod))
	wb0 = wb[1]
	wb1 = @view wb[2:dim]
	wb1sq = 0.0 #wb1sq = sum(wb1 .* wb1)
	for i=1:dim-1
		wb1sq += wb1[i]*wb1[i]
	end
	cv = -(1 + wb0 + wb1sq/(1+wb0))
	d = 1 + 2/(1+wb0) + wb1sq/((1+wb0)*(1+wb0))
	a = (wb0*wb0 + wb1sq - cv*cv*wb1sq/(1+d*wb1sq))/2
	u0 = sqrt(wb0*wb0 + wb1sq - a)
	u1 = cv/u0
	v1 = sqrt(cv*cv/(u0*u0) - d)

	scaling.iWiW.diag[cti(c, 1)] = a*inusq
	scaling.iW.diag[cti(c, 1)] = sqrt(abs(a*inusq))
	for i=2:dim
		scaling.iWiW.diag[cti(c, i)] = inusq
		scaling.iW.diag[cti(c, i)] = sqrt(inusq)
	end
	scu, scv = scaling.us[cind], scaling.vs[cind]
	# u = inu .* vcat(u0, u1 .* wb1); v = inu .* vcat(0, v1 .* wb1)
	scu[c.offs+1] = inu * u0
	scv[c.offs+1] = 0.0
	for i=2:dim
		wbv = inu * wb1[i-1]
		scu[c.offs+i] = u1 * wbv
		scv[c.offs+i] = v1 * wbv
	end

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

# this removes every nonzero in the sparse vector dest first, then re-fills it. This retains the # nonzeros
function clear_reset!(dest::SparseMatrixCSC, src::Vector)
	nnz = 0
	cnz = 1
	dest.nzval .= 0.0
	# this is all 1d so we don't have to touch colptr
	for i=1:length(src)
		if src[i] != 0.0
			if cnz > length(dest.nzval)
				throw(ArgumentError("too many nonzeros in source; limit $(length(dest.nzval))"))
			end
			dest.rowval[cnz] = i
			dest.nzval[cnz] = src[i]
			cnz += 1
			nnz += 1
		end
	end
end

function modify_factors!(cone::SOC{dim}, cind, factor::SuiteSparse.CHOLMOD.Factor, 
						 scaling::SqrScaling{odim}, g::SparseMatrixCSC) where {dim, odim}
	perm = SuiteSparse.CHOLMOD.get_perm(factor) # this is allocating but... meh. The permutation might be updated.

	temp = scaling.ws_3[cind]
	mul!(temp, g', scaling.us[cind])
	permute!(temp, perm)
	clear_reset!(scaling.ws_5[cind], temp)
	copyto!(scaling.ws_6[cind], scaling.ws_5[cind])
	SuiteSparse.CHOLMOD.lowrankupdowndate!(factor, scaling.ws_6[cind], Cint(1))
	mul!(temp, g', scaling.vs[cind])
	permute!(temp, perm)
	clear_reset!(scaling.ws_5[cind], temp)
	copyto!(scaling.ws_6[cind], scaling.ws_5[cind])
	SuiteSparse.CHOLMOD.lowrankupdowndate!(factor, scaling.ws_6[cind], Cint(0))
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