mutable struct SparseSolver{dim} <: KKTSolver{SqrScaling{dim}}
	k0::Vector{Float64}
	k1::Vector{Float64}
	k2::Vector{Float64}

	n0::Vector{Float64}
	n1::Vector{Float64}

	m0::Vector{Float64}
	AA::SparseMatrixCSC{Float64, Int}
	Gt::SparseMatrixCSC{Float64, Int}
	GiW::SuiteSparse.CHOLMOD.Sparse{Float64}
	AtS::SuiteSparse.CHOLMOD.Sparse{Float64}
	Gint::SparseMatrixCSC{Float64, Int}
	Gfact::SuiteSparse.CHOLMOD.Factor{Float64}
	Afact::SuiteSparse.CHOLMOD.Factor{Float64}

	ws1::LdivWorkspace{Float64}
	dv1::SuiteSparse.CHOLMOD.Dense{Float64}
	dv2::SuiteSparse.CHOLMOD.Dense{Float64}

	ws2::LdivWorkspace{Float64}
	dv3::SuiteSparse.CHOLMOD.Dense{Float64}
	dv4::SuiteSparse.CHOLMOD.Dense{Float64}
	function SparseSolver(pr::Problem{C,n,m,k,sing}) where {C, n, m, k, sing}
		k0, k1, k2 = zeros(k), zeros(k), zeros(k)
		n0, n1 = zeros(n), zeros(n)
		m0 = zeros(m)

		ws1 = LdivWorkspace(Float64)
		dv1, dv2 = SuiteSparse.CHOLMOD.Dense(zeros(pr.n)), SuiteSparse.CHOLMOD.Dense(zeros(pr.n))
		ws2 = LdivWorkspace(Float64)
		dv3, dv4 = SuiteSparse.CHOLMOD.Dense(zeros(pr.m)), SuiteSparse.CHOLMOD.Dense(zeros(pr.m))

		AA = pr.A'*pr.A
		if !sing 
			Gt = sparse(pr.G')
			desired_stype = 0
		else
			Gt = sparse(pr.G'*pr.G + pr.A'*pr.A)
			desired_stype = -1
		end 
		cm = SuiteSparse.CHOLMOD.defaults(SuiteSparse.CHOLMOD.common_struct[Threads.threadid()])
		GtS = SuiteSparse.CHOLMOD.Sparse(Gt, desired_stype)
		AtCSC = sparse(pr.A')
		AtS = SuiteSparse.CHOLMOD.Sparse(AtCSC)
		#note that we're setting up both Gfact and Afact to be unsymmetric, so CHOLMOD will factorize AA'
		Gfact = SuiteSparse.CHOLMOD.analyze(GtS, cm)
		cholesky!(Gfact, GtS) # get the sparsity pattern right
		examplemat = SuiteSparse.CHOLMOD.Sparse(sparse((Gfact.UP\AtCSC)'), 0)
		Afact = SuiteSparse.CHOLMOD.analyze(examplemat, cm) # factorize W^t W, where L W = A or W = L^-1 A; thus W^t = A^t L^-t
		Gint = spzeros(pr.k, pr.n)
		return new{n}(k0, k1, k2, n0, n1, m0,
			AA, Gt, GtS, AtS, Gint, Gfact, Afact, 
			ws1, dv1, dv2,
			ws2, dv3, dv4)
	end
end

function setup_iter(ss::SparseSolver, pr::Problem{C,n,m,k,sing}, s::State, scaling::SqrScaling) where {C, n, m, k, sing}
	if !sing
		copyto!(ss.GiW, ss.Gt)
		lmul!(scaling.iW, ss.GiW)
		cholesky!(ss.Gfact, ss.GiW)
		# do the low rank updates
		modify_factors!(pr.cones, ss.Gfact, scaling, pr.G)
	else
		Gint = ss.Gint
		copyto!(Gint, pr.G)
		lmul!(scaling.iWiW, Gint)
		i1 = pr.G' * Gint :: SparseMatrixCSC{Float64, Int}
		i1 .+= ss.AA
		cholesky!(ss.Gfact, i1)
		# do the low rank updates
		modify_factors!(pr.cones, ss.Gfact, scaling, pr.G)
	end	
	# convert the factorization to an LLt one (is LDLt if we had to modify the factor)
	SuiteSparse.CHOLMOD.change_factor!(ss.Gfact, true, false, false, false)
	# solve L C^t = A^t giving C^t = L^-1 A^t, then transpose to get C = A L^-T
	i2 = SuiteSparse.CHOLMOD.spsolve(SuiteSparse.CHOLMOD.CHOLMOD_P, ss.Gfact, ss.AtS)
	Ct = SuiteSparse.CHOLMOD.spsolve(SuiteSparse.CHOLMOD.CHOLMOD_L, ss.Gfact, i2)
	Ctt = SuiteSparse.CHOLMOD.transpose_(Ct, 1)
	cholesky!(ss.Afact, Ctt)
end

function solve_kkt(ss::SparseSolver, pr::Problem{C,n,m,k,sing}, s::State, scaling::SqrScaling, 
				   dx::Vector{Float64}, dy::Vector{Float64}, dz::Vector{Float64}, ds::Vector{Float64}, 
				   cx::Vector{Float64}, cy::Vector{Float64}, cz::Vector{Float64}, cs::Vector{Float64}) where {C, n, m, k, sing}
	iprod!(pr.cones, ss.k0, scaling.l, ds)
	scale!(pr.cones, scaling, ss.k0, ss.k1)
	bx = dx
	by = dy
	ss.k2 .= dz .- ss.k1
	iscale!(pr.cones, scaling, ss.k2, ss.k1)
	iscale!(pr.cones, scaling, ss.k1, ss.k1)
	mul!(ss.n0, pr.G', ss.k1) #ss.n0 = G'*W^-1 * W^-T*(dz - ss.k1)
	ss.n0 .+= bx
	At = pr.A' :: Adjoint{Float64, SparseMatrixCSC{Float64, Int}}
	if sing
		ss.n0 .+= At*by
	end

	# ss.m0 = A L-TL-1 ss.n0
	copyto!(ss.dv1, ss.n0)
	div!(ss.Gfact, ss.dv2, ss.dv1, ss.ws1)
	copyto!(ss.n1, ss.dv2)
	mul!(ss.m0, pr.A, ss.n1)
	ss.m0 .-= by
	copyto!(ss.dv3, ss.m0)
	div!(ss.Afact, ss.dv4, ss.dv3, ss.ws2)
	copyto!(cy, ss.dv4)
	if sing
		ss.m0 .= by .- cy
	else
		ss.m0 .= .-cy
	end
	mul!(ss.n1, At, ss.m0)
	ss.n0 .+= ss.n1
	# cx = L-TL-1 ss.n0
	copyto!(ss.dv1, ss.n0)
	div!(ss.Gfact, ss.dv2, ss.dv1, ss.ws1)
	copyto!(cx, ss.dv2)
	mul!(ss.k1, pr.G, cx)
	ss.k1 .-= ss.k2
	iscale!(pr.cones, scaling, ss.k1, cz)
	iscale!(pr.cones, scaling, cz, cz)
	scale!(pr.cones, scaling, cz, ss.k1)
	ss.k0 .-= ss.k1
	scale!(pr.cones, scaling, ss.k0, cs)
end
