using Socp: Problem, State, Cone, POC, SOC, vprod, iprod, make_e, KKTState
using Socp: compute_scaling, compute_sqscaling, solve_kkt, cgt
using Socp: deg, max_step, solve_socp, Scaling, SqrScaling, scale!, iscale!, SolverState, modify_factors!, compute_full_scaling
using Test
using LinearAlgebra
using SparseArrays
using SuiteSparse
using SuiteSparse.CHOLMOD

@testset "Vector operations" begin
    tv1 = Float64[1,1,1,1,2,3]
    tv2 = Float64[1,1,1,1,5,6]
    tcone = (POC(0,3), SOC(3,3))
    tid = make_e(tcone)
    @test vprod(tcone, tv1, tv1) == [1,1,1,14,4,6]
    @test vprod(tcone, tv1, tv2) == [1,1,1,29,7,9]
    @test norm(vprod(tcone, tv1, iprod(tcone, tv1, tv2)) .- tv2) < 0.0001
    @test vprod(tcone, tid, tv1) == tv1
    @test vprod(tcone, tid, tv2) == tv2
    @test cgt(POC(0,3), [1,2,3], [0,0,0])
    @test !cgt(POC(0,3), [-1,2,3], [0,0,0])
    @test cgt(SOC(0,3), [3,2,2], [0,0,0])
    @test !cgt(SOC(0,3), [2,2,2], [0,0,0])

    @test max_step(POC(0,3), [1,2,3]) == -1
    @test max_step(SOC(0,3), [1,2,3]) == sqrt(2^2 + 3^2)-1.0
    @test max_step((POC(0,3), SOC(0,3)), [1,2,3]) == sqrt(2^2 + 3^2)-1.0
end

@testset "Nesterov-Todd Scalings" begin
    tv1 = Float64[1,1,1,9,2,3]
    tv2 = Float64[1,1,1,22,5,6]
    tcone = (POC(0,3), SOC(3,3))
    s = Scaling(zeros(6,6),zeros(6,6),zeros(6,6),zeros(6),tcone)
    compute_scaling(tcone, s, tv1, tv2)
    sca,isca,pt = s.W,s.iW,s.l
    tvp1 = isca' * tv1 
    tvp2 = sca * tv2
    @test sum(isca*sca .- Matrix{Float64}(I, size(sca)...)) < 0.001
    @test norm(isca' * tv1 .- sca * tv2) < 0.001	
    @test norm(isca' * tv1 .- pt) < 0.001
    op,op2 = similar(tv2),similar(tv2)
    scale!(tcone, s, tv2, op)
    @test norm(sca*tv2 .- op) < 0.001
    iscale!(tcone, s, op, op2)
    println(tv2 .- op2)
    @test norm(tv2 .- op2) < 0.001
end

@testset "Squared NT Scalings" begin 
    tv1 = Float64[1,1,1,9,2,3]
    tv2 = Float64[1,1,1,22,5,6]
    tcone = (POC(0,3), SOC(3,3))
    s = Scaling(zeros(6,6),zeros(6,6),zeros(6,6),zeros(6),tcone)
    compute_scaling(tcone, s, tv1, tv2)
    s2 = SqrScaling(Diagonal(zeros(6)), Diagonal(zeros(6)), zeros(6), tcone, 3, sparse(I, 6, 6))
    compute_sqscaling(tcone, s2, tv1, tv2)
    @test norm(s.l .- s2.l) < 0.0001
    @test norm(s.wbs .- s2.wbs) < 0.0001
    @test norm(s.mu .- s2.mu) < 0.0001
    #@test norm(norm(s.iWiW .- s2.iWiW)) < 0.01 # this doesn't hold because we've factored out the low rank terms

	G = [0 0 1.0; 0 0 -1; 0 -1 0; -1 0 0]
	cones = (POC(0,1), SOC(1,3))
    u=[3.414213562373095, 2.414213562373095, 1.0, 1.0]
    v=[1.414213562373095, 2.414213562373095, -1.0, -1.0]
    s = Scaling(zeros(4,4),zeros(4,4),zeros(4,4),zeros(4),cones)
    compute_scaling(cones, s, u, v)
    s2 = SqrScaling(Diagonal(zeros(4)), Diagonal(zeros(4)), zeros(4), cones, 3, G)
    compute_sqscaling(cones, s2, u, v)
    @test norm(s.l .- s2.l) < 0.0001
    @test norm(s.wbs .- s2.wbs) < 0.0001
    @test norm(s.mu .- s2.mu) < 0.0001
    f1 = cholesky(Sparse(sparse(G' * s.iWiW * G)))
    f2 = cholesky(Sparse(sparse(G' * s2.iWiW * G)))
    modify_factors!(cones, f2, s2, sparse(G))
    @test norm(norm(f2\Matrix(I, 3, 3) .- f1\Matrix(I, 3, 3))) < 0.01

    fs = compute_full_scaling(cones, s2)
    @test norm(norm(fs .- s.iWiW)) < 0.01

    u=[2.5571536140045033, 2.594521365784194, 1.6419234525608073, 1.6419234525608069] 
    v=[0.42421356237309515, 1.424213562373095, -1.0, -0.9999999999999997]
    s = Scaling(zeros(4,4),zeros(4,4),zeros(4,4),zeros(4),cones)
    compute_scaling(cones, s, u, v)
    s2 = SqrScaling(Diagonal(zeros(4)), Diagonal(zeros(4)), zeros(4), cones, 3, G)
    compute_sqscaling(cones, s2, u, v)
    fs = compute_full_scaling(cones, s2)
    @test norm(norm(fs .- s.iWiW)) < 0.01
    @test norm(norm(s2.iW*s2.iW .- s2.iWiW)) < 0.01

    #@test norm(norm(s.iWiW .- s2.iWiW)) < 0.01
end

@testset "KKT reference solution" begin

	A = Array{Float64,2}(UndefInitializer(), 0, 3)
	b = Array{Float64,1}(UndefInitializer(), 0)

	c = [-1.0,-1.0,1.0]
	G = [0 0 1.0; 0 0 -1; 0 -1 0; -1 0 0]
	h = [5.0,0.0,0.0,0.0]
	cones = (POC(0,1), SOC(1,3))
	x=[3.5093936289670493, 3.5093936289670546, 4.984484369850202]
	y=Float64[] 
	z=[0.4416936891219317, 1.4416936891219376, -1.0000000000000004, -0.9999999999999993] 
	s=[0.025571536140045037, 4.994540275840449, 3.5093936289670546, 3.5093936289670484] 
	dx=[6.661338147750939e-16, -4.440892098500626e-16, 5.995204332975845e-15] 
	dy=Float64[] 
	dz=[-0.010055905990247638, -0.01005590599024675, -0.0, 8.881784197001252e-16] 
	ds=[-0.011294786134211294, -0.18180993781041443, -0.06493037168608469, -0.06493037168608427] 
	cxr=[0.02479569536244916, 0.02479569536250157, 0.013918468357446631] 
	cyr=Float64[] 
	czr=[-0.02758755986830461, -0.02758755986827908, 1.3092496892856528e-14, 3.9336377471191126e-14] 
	csr=[-0.02397437434769427, 0.003862562367179831, 0.024795695362486953, 0.02479569536243503] 
	cx,cy,cz,cs = zeros(size(cxr)), zeros(size(cyr)), zeros(size(czr)), zeros(size(csr))
	mehrotra=true
	prob = Problem(c, A, b, G, h, cones)
    s2 = SqrScaling(Diagonal(zeros(4)), Diagonal(zeros(4)), zeros(4), cones, 3, G)
    compute_sqscaling(cones, s2, s, z)
    solve_kkt(prob, State(prob, x,y,z,s), s2, dx, dy, dz, ds, cx, cy, cz, cs, true, KKTState(prob))
    @test norm(cx .- cxr) < 0.001
    @test norm(cy .- cyr) < 0.001
    @test norm(cz .- czr) < 0.001
    @test norm(cs .- csr) < 0.001
end

@testset "SOC programming 1" begin
	c = [-1.0,-1.0,1.0]

	A = Array{Float64,2}(UndefInitializer(), 0, 3)
	b = Array{Float64,1}(UndefInitializer(), 0)

	G = [0 0 1.0; 0 0 -1; 0 -1 0; -1 0 0]
	h = [5.0,0.0,0.0,0.0]
	cones = (POC(0,1), SOC(1,3))
	prob = Problem(c, A, b, G, h, cones)
	ss = SolverState(prob)
	soln = solve_socp(prob, ss)
	println(soln.x)
	@test norm(soln.x .- [3.53553, 3.53553, 5.0]) < 0.001
end

@testset "SOC programming 2" begin
	c = [-2.0, 1.0, 5.0]

	A = Array{Float64,2}(UndefInitializer(), 0, 3)
	b = Array{Float64,1}(UndefInitializer(), 0)

	G = [12.0 6.0 -5.0;
		 13.0 -3.0 -5.0;
		 12.0 -12.0 6.0;
		 3.0 -6.0 10.0;
		 3.0 -6.0 -2.0;
		 -1.0 -9.0 -2.0;
		 1.0 19.0 -3.0]
	h = [-12.0, -3.0, -2.0, 27.0, 0.0, 3.0, -42.0]
	cones = (SOC(0,3), SOC(3,4))
	prob = Problem(c, A, b, G, h, cones)
	ss = SolverState(prob)
	soln = solve_socp(prob, ss)
	@test norm(soln.x .- [-5.01467, -5.7669, -8.52176]) < 0.001
end

@testset "SOC programming 3" begin
	c = [-2.0, 1.0, 5.0]

	A = [1.0 0.0 0.0] # Array{Float64,2}(UndefInitializer(), 0, 3)
	b = [-3.0] # Array{Float64,1}(UndefInitializer(), 0)

	G = [12.0 6.0 -5.0;
		 13.0 -3.0 -5.0;
		 12.0 -12.0 6.0;
		 3.0 -6.0 10.0;
		 3.0 -6.0 -2.0;
		 -1.0 -9.0 -2.0;
		 1.0 19.0 -3.0]
	h = [-12.0, -3.0, -2.0, 27.0, 0.0, 3.0, -42.0]
	cones = (SOC(0,3), SOC(3,4))
	prob = Problem(c, A, b, G, h, cones)
	ss = SolverState(prob)
	soln = solve_socp(prob, ss)
	@test norm(soln.x .- [-3.0, -4.82569, -6.64011]) < 0.001
end

# problem: mass 1 travelling in +x direction with velocity 1 starting from position 0
# objective: min applied force for velocity of 0 at position 0
# state: velocity, position
# velocity_0 = 1, position_0 = 0
# velocity_i = force_i-1 + velocity_i-1
# position_i = velocity_i-1 + position_i-1
# velocity_n = 0, position_0 = 0
# |force_i| <= tot 
# n vel, n pos, n-1 force, total
# x = [vel..., pos..., force...,tot] length 3n

@testset "Linear optimal control" begin
	n = 50;
	c = [zeros(3n-1); 1.0] # min tot ;
	vel = 1:n; pos = n+1:2*n; force = 2*n+1:3*n-1;
	A = zeros(2*n+2, 3*n);
	b = zeros(2*n+2);
	A[vel[1],vel[1]] = 1.0 # vel[1] = 1.0;
	b[vel[1]] = 1.0;
	A[pos[1],pos[1]] = 1.0 # pos[1] = 0.0;
	b[pos[1]] = 0.0;
	for stp=2:n;
		A[vel[stp], vel[stp]] = -1.0 # vel[stp] - vel[stp-1] - force[stp-1] = 0.0;
		A[vel[stp], vel[stp-1]] = 1.0;
		A[vel[stp], force[stp-1]] = 1.0;
		b[vel[stp]] = 0.0;

		A[pos[stp], pos[stp]] = -1.0;
		A[pos[stp], pos[stp-1]] = 1.0;
		A[pos[stp], vel[stp-1]] = 1.0;
		b[pos[stp]] = 0.0;
	end;
	A[2*n+1, vel[n]] = 1.0 # vel[n] = 0.0;
	b[2*n+1] = 0.0;
	A[2*n+2, pos[n]] = 1.0;
	b[2*n+2] = 0.0;

	# G = [0 0 0 -1.0; 0 0 -I 0];
	G = zeros(n, 3*n);
	G[1,3*n] = -1.0;
	for i = 1:n-1;
		G[i+1,2*n+i] = -1.0;
	end	;
	h = zeros(n);
	cones = (SOC(0,n),);
	prob = Problem(c, A, b, G, h, cones);
	ss = SolverState(prob);
	soln = solve_socp(prob, ss);
	println(soln.x[force])
	println("Performance evaluation")
	@time for i=1:100 soln = solve_socp(prob, ss) end
end
