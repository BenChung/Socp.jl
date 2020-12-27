struct SolverState{cdim, Solver, Scaling}
	scaling::Scaling
	solver::Solver
	initm::Matrix{Float64}
	initv::Vector{Float64}
	idel::Vector{Float64}
	dx::Vector{Float64}
	dy::Vector{Float64}
	dz::Vector{Float64}
	ds::Vector{Float64}
	nt1::Vector{Float64}
	nt2::Vector{Float64}
	mt1::Vector{Float64}
	kt1::Vector{Float64}
	kt2::Vector{Float64}
	kt3::Vector{Float64}
	rx::Vector{Float64}
	ry::Vector{Float64}
	rz::Vector{Float64}
	rs::Vector{Float64}

	function SolverState(pr::Problem, solver::K) where K <: KKTSolver{S} where S <: AbstractScaling
		scaling = S(pr)
		n = pr.n
		m = pr.m
		k = pr.k
		
	    initm = zeros(n+m+k, n+m+k)
	    initv = zeros(n+m+k)
		idel = zeros(k)
		dx,dy,dz,ds = zeros(n),zeros(m),zeros(k),zeros(k)
		nt1,nt2 = zeros(n),zeros(n)
		mt1 = zeros(m)
		kt1,kt2,kt3 = zeros(k),zeros(k),zeros(k)
		rx,ry,rz,rs = zeros(n), zeros(m), zeros(k), zeros(k)
		return new{pr.n, K, S}(scaling, solver, initm, initv, idel, dx, dy, dz, ds, nt1, nt2, mt1, kt1, kt2, kt3, rx, ry, rz, rs)
	end
end

function solve_socp(prob::Problem{C,n,m,k,sing}, ss::SolverState{dim, K, S}) where {C,K,S,n,m,k,sing,dim}
	cones = prob.cones
	
	initm, initv, idel = ss.initm, ss.initv, ss.idel
	dx,dy,dz,ds = ss.dx,ss.dy,ss.dz,ss.ds
	nt1,nt2,mt1,kt1,kt2,kt3 = ss.nt1,ss.nt2,ss.mt1,ss.kt1,ss.kt2,ss.kt3
	rx,ry,rz,rs = ss.rx,ss.ry,ss.rz,ss.rs
	println("$n $m $k")
	#=
    open("A.txt", "w") do f
    	println(f, prob.A)
    end
    open("G.txt", "w") do f
    	println(f, prob.G)
    end
    open("c.txt", "w") do f
    	println(f, prob.c)
    end 
    open("b.txt", "w") do f
    	println(f, prob.b)
    end
    open("h.txt", "w") do f
    	println(f, prob.h)
    end
    open("cones.txt", "w") do f
    	println(f, prob.cones)
    end
	=# 
	r1 = hcat(spzeros(n,n), prob.A', prob.G')
	r2 = hcat(prob.A, spzeros(m,m), spzeros(m, k))
	r3 = hcat(prob.G, spzeros(k,m), sparse(-I, k, k))
	idmat = vcat(r1, r2, r3)
    initv[1:n] .= (-).(prob.c)
    initv[n+1:n+m] .= prob.b 
    initv[n+m+1:n+m+k] .= prob.h
    #=
    open("initm.txt", "w") do f
    	println(f, initm)
    end
    open("initv.txt", "w") do f
    	println(f, initv)
    end
    =#

	initials::Vector{Float64} = idmat\initv

	make_e!(cones, idel)
	iz::Vector{Float64} = initials[n+m+1:n+m+k]
	alphp = max_step(cones, -iz)
	alphd = max_step(cones, iz)

	if abs(alphp) < 1e-10
		inits = -iz
	else
		inits = -iz + (1+alphp)*idel
	end

	if abs(alphd) < 1e-10
		initz = iz
	else
		initz = iz + (1+alphd)*idel
	end

#	println("$initials $inits $initz")
	state = State(prob, initials[1:n], initials[n+1:n+m], initz, inits)
	for i=1:40
		scaling = compute_scaling(cones, ss.scaling, state.s, state.z)
		l = scaling.l
		# solve affine direction
		# dx = prob.A'*state.y .+ prob.G'*state.z .+ prob.c
		mul!(nt1, prob.A', state.y)
		mul!(nt2, prob.G', state.z)
		dx .= nt1 .+ nt2 .+ prob.c
		# dy = prob.A*state.x .- prob.b
		mul!(mt1, prob.A, state.x)
		dy .= mt1 .- prob.b
		# dz = prob.G*state.x .+ state.s .- prob.h
		mul!(kt1, prob.G, state.x)
		dz .= kt1 .+ state.s .- prob.h

		vprod!(cones, ds, l, l)

		if (norm(dx) + norm(dy) + dot(state.z,state.s)) < 1e-5
			break
		end
		rmul!(dx, -1.0); rmul!(dy, -1.0); rmul!(dz, -1.0); rmul!(ds, -1.0)
		ss.solver(prob, state, scaling, dx, dy, dz, ds, rx, ry, rz, rs)
		scale!(prob.cones, scaling, rz, kt3)
		iscale!(prob.cones, scaling, rs, kt2)
		t = compute_step(cones, l, kt3, kt2)

		rho = 1-t-t^2 * dot(kt2, kt3)/dot(l,l)
		sig = max(0, min(1, rho))^3
		mu = dot(l,l)/deg(cones)

		scfact = 1.0-sig
		vprod!(cones, kt1, kt2, kt3)
		mul!(kt2, sig*mu, idel)
		ds .+= kt2 .- kt1
		rmul!(dx, scfact); rmul!(dy, scfact); rmul!(dz, scfact)
		ss.solver(prob, state, scaling, dx, dy, dz, ds, rx, ry, rz, rs)

		scale!(prob.cones, scaling, rz, kt3)
		iscale!(prob.cones, scaling, rs, kt2)
		step = compute_step(cones, l, kt3, kt2) 
		step *= 0.99
		state.x .+= rx .* step
		state.y .+= ry .* step
		state.z .+= rz .* step
		state.s .+= rs .* step
    end
    return state
end