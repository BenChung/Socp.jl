using Socp: Problem, State, Cone, POC, SOC, vprod, iprod, make_e
using Socp: compute_scaling, solve_kkt, cgt, line_search, line_search_scaled
using Socp: deg, max_step, solve_socp, Scaling
using Test
using LinearAlgebra

@testset "Vector operations" begin
    tv1 = Float64[1,1,1,1,2,3]
    tv2 = Float64[1,1,1,1,5,6]
    tcone = [POC(0,3), SOC(3,3)]
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
    @test max_step([POC(0,3), SOC(0,3)], [1,2,3]) == sqrt(2^2 + 3^2)-1.0
end

@testset "Nesterov-Todd Scalings" begin
    tv1 = Float64[1,1,1,9,2,3]
    tv2 = Float64[1,1,1,22,5,6]
    tcone = [POC(0,3), SOC(3,3)]
    tid = make_e(tcone)
    s = Scaling(zeros(6,6),zeros(6,6),zeros(6,6),zeros(6))
    compute_scaling(tcone,s, tv1, tv2)
    sca,isca,pt = s.W,s.iW,s.l
    tvp1 = isca' * tv1 
    tvp2 = sca * tv2
    @test sum(isca*sca .- Matrix{Float64}(I, size(sca)...)) < 0.001
    @test norm(isca' * tv1 .- sca * tv2) < 0.001	
    @test norm(isca' * tv1 .- pt) < 0.001
end

@testset "SOC programming 1" begin
	c = [-1.0,-1.0,1.0]

	A = Array{Float64,2}(UndefInitializer(), 0, 3)
	b = Array{Float64,1}(UndefInitializer(), 0)

	G = [0 0 1.0; 0 0 -1; 0 -1 0; -1 0 0]
	h = [5.0,0.0,0.0,0.0]
	cones = Cone[POC(0,1), SOC(1,3)]
	prob = Problem(c, A, b, G, h, cones)
	soln = solve_socp(prob)
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
	cones = Cone[SOC(0,3), SOC(3,4)]
	prob = Problem(c, A, b, G, h, cones)
	soln = solve_socp(prob)
	@test norm(soln.x .- [-5.01467, -5.7669, -8.52176]) < 0.001
end
