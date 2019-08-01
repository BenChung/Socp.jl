function solve_socp(prob::Problem)
	n = prob.n
	m = prob.m
	k = prob.k
	cones = prob.cones
	initials::Vector{Float64} = [zeros(n, n) prob.A' prob.G'; 
	             prob.A zeros(m, m) zeros(m, k); 
	             prob.G zeros(k, m) -Matrix{Float64}(I, k, k)]\[-prob.c;prob.b;prob.h]
	idel = make_e(cones)
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
	scaling = Scaling(prob)
	ss = SolveState(prob)

	ds = zeros(length(initz))
	ic1 = zeros(length(initz))
	zdz = zeros(length(state.s))
	rx,ry,rz,rs = zeros(n), zeros(m), zeros(k), zeros(k)
	for i=1:40
		scaling = compute_scaling(cones, scaling, state.s, state.z)
		W,iW,l = scaling.W, scaling.iW, scaling.l
		# solve affine direction
		dx = prob.A'*state.y .+ prob.G'*state.z .+ prob.c
		dy = prob.A*state.x .- prob.b
		dz = prob.G*state.x .+ state.s .- prob.h
		vprod!(cones, ds, l, l)
		# println("s $(state.x) $(state.y) $(state.z) $(state.s)")
		#println("d $dx $dy $dz $ds")

		if (norm(dx) + norm(dy) + dot(state.z,state.s)) < 1e-5 && 
			cgt(cones, state.s, zdz) &&
			cgt(cones, state.z, zdz)
			break
		end
		rmul!(dx, -1.0); rmul!(dy, -1.0); rmul!(dz, -1.0)
		solve_kkt(prob, state, scaling, dx, dy, dz, -ds, rx,ry,rz,rs, ss=ss)
		t = compute_step(cones,l,W*rz, iW'*rs)

		rho = 1-t-t^2 * dot(iW'*rs, W*rz)/dot(l,l)
		sig = max(0, min(1, rho))^3
		mu = dot(l,l)/deg(cones)

		scfact = 1.0-sig
		vprod!(cones, ic1, iW'*rs, W*rz)
		comb_s = -ds .+ sig*mu*idel .- ic1
		rmul!(dx, scfact); rmul!(dy, scfact); rmul!(dz, scfact)
		solve_kkt(prob, state, scaling, dx, dy, dz, comb_s, rx,ry,rz,rs, ss=ss)

		step = compute_step(cones, l, W*rz, iW'*rs) 
		step *= 0.99
		sup = line_search_scaled(cones, scaling, rz, rs)
		println("$(state.x)")
		state.x .+= rx .* step
		state.y .+= ry .* step
		state.z .+= rz .* step
		state.s .+= rs .* step
    end
    return state
end