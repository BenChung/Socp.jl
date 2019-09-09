	struct Scaling 
		W::Matrix{Float64}
		iW::Matrix{Float64}
		iWiW::Matrix{Float64}
		l::Vector{Float64}

		#per cone
		mu::Vector{Float64}
		wbs::Vector{Float64} # sqrt(s/z) for POC, wb for SOC
		zik::Vector{Float64} # storage for compute_scaling
		sik::Vector{Float64}
		function Scaling(p::Problem) 
			return Scaling(zeros(p.k, p.k), zeros(p.k,p.k), zeros(p.k,p.k), zeros(p.k), p.cones)
		end
		function Scaling(W,iW,iWiW,l,cones) 
			md = maximum(getfield.(cones, :dim))
			zik,sik = zeros(md),zeros(md)
			return new(W, iW, iWiW, l, zeros(length(cones)), zeros(cones[end].offs + cones[end].dim), zik, sik)
		end
	end

	function compute_scaling(cone::POC, cind, scaling::Scaling, s, z)
		for i = 1:cone.dim
			ii = cti(cone, i)
			scaling.W[ii,ii] = sqrt(s[ii] / z[ii])
			scaling.iW[ii,ii] = sqrt(z[ii] / s[ii])
			scaling.l[ii] = sqrt(s[ii] * z[ii])
			scaling.wbs[ii] = sqrt(s[ii] / z[ii])
		end
	end

	function compute_scaling(c::SOC, cind, scaling::Scaling, s, z)
		sik,zik = scaling.sik,scaling.zik
		for ii = 1:c.dim
			i = cti(c, ii)
			sik[ii] = s[i]
			zik[ii] = z[i]
		end
		onrmz = zik[1]^2 
		onrms = sik[1]^2 
		wl = c.dim
		for i=2:wl
			onrmz -= zik[i] * zik[i]
			onrms -= sik[i] * sik[i]
		end
		nrmz = sqrt(onrmz)
		nrms = sqrt(onrms)
		rmul!(zik, 1/nrmz)
		rmul!(sik, 1/nrms)
		zbk = zik
		sbk = sik

		nsum = 0.0
		for i = 1:wl
			nsum += zbk[i] * sbk[i]
		end
		gamma = sqrt((1 + nsum)/2)

		wb = @view scaling.wbs[cti(c,1):cti(c,c.dim)]
		wb[1] = sbk[1] + zbk[1]
		for i = 2:wl 
			wb[i] = sbk[i] - zbk[i]
		end
		rmul!(wb, 1/(2*gamma))

		bl = length(wb)-1
		denom = wb[1] + 1
		mu = sqrt(nrms/nrmz)
		scaling.mu[cind] = mu
		for i=1:bl, j=1:bl
			cellv = ((i==j) ? 1.0 : 0.0) + wb[i+1]*wb[j+1]/denom
			ii,ji = cti(c, i), cti(c, j)
			scaling.W[ii+1,ji+1] = cellv*mu
			scaling.iW[ii+1,ji+1] = cellv/mu
		end

		ic1 = cti(c,1)
		for i=1:wl
			scaling.W[ic1,cti(c, i)] = wb[i]*mu
		end
		scaling.iW[ic1,ic1] = wb[1]/mu

		for i=2:wl
			ii = cti(c, i)
			scaling.W[ii,ic1] = wb[i]*mu
			scaling.iW[ic1,ii] = -wb[i]/mu
			scaling.iW[ii,ic1] = -wb[i]/mu
		end

		ziv,siv = zbk[1], sbk[1]
		tmv1 = sqrt(nrms*nrmz)
		mult = tmv1/(ziv + siv + 2*gamma)
		rmul!(sbk, gamma + ziv)
		rmul!(zbk, gamma + siv)
		for i=2:wl
			scaling.l[cti(c, i)] = (sbk[i] + zbk[i])*mult
		end
		scaling.l[ic1] = gamma*tmv1
	end

	function compute_scaling(cones::Vector{Cone}, scaling::Scaling, s, z)
		# assume that cones is in the order POC ... POC SOC ... SOC
		odim = 0
		for cone in cones
			odim += cone.dim
		end
		for i=1:length(cones)
			cone = cones[i]
			compute_scaling(cone, i, scaling, s, z)
		end
		mul!(scaling.iWiW, scaling.iW, scaling.iW')
		return scaling
	end 

	function scale!(cone::POC, cind, scl, s, op)
		wb = scl.wbs
		for i=cti(cone,1):cti(cone, cone.dim)
			op[i] = wb[i]*s[i]
		end
	end

	function iscale!(cone::POC, cind, scl, s, op)
		wb = scl.wbs
		for i=cti(cone,1):cti(cone, cone.dim)
			op[i] = 1/wb[i]*s[i]
		end
	end

	function scale!(cone::SOC, cind, scl, s, op)
		wb = scl.wbs
		mu = scl.mu[cind]
		del = 0.0
		for i=cti(cone, 2):cti(cone, cone.dim)
			del += wb[i]*s[i]
		end

		ind1 = cti(cone, 1)
		op[ind1] = mu * (wb[ind1]*s[ind1] + del)
		cst = (s[ind1] + del/(1+wb[ind1]))
		for i=cti(cone, 2):cti(cone, cone.dim)
			op[i] = mu * (s[i] + cst*wb[i])
		end
	end

	function iscale!(cone::SOC, cind, scl, s, op)
		wb = scl.wbs
		mu = scl.mu[cind]
		del = 0.0
		for i=cti(cone, 2):cti(cone, cone.dim)
			del += wb[i]*s[i]
		end

		ind1 = cti(cone, 1)
		op[ind1] = 1/mu * (wb[ind1]*s[ind1] - del)
		cst = (-s[ind1] + del/(1+wb[ind1]))
		for i=cti(cone, 2):cti(cone, cone.dim)
			op[i] = 1/mu * (s[i] + cst*wb[i])
		end
	end

	# computes W s
	function scale!(cones::Vector{Cone}, scl::Scaling, s, op)
		for i=1:length(cones)
			scale!(cones[i], i, scl, s, op)
		end
	end
	# computes W^-1 s
	function iscale!(cones::Vector{Cone}, scl::Scaling, s, op)
		for i=1:length(cones)
			iscale!(cones[i], i, scl, s, op)
		end
	end

	# update NT scalings

	function update_scaling!(cone::POC, scaling, s, z)
		mat,imat,sv = scaling
	end

	function update_scaling!(cone::SOC, scaling, s, z)
	end

	function update_scaling!(cones::Vector{Cone}, scaling, s, z)

	end
