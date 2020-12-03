struct SqrScaling 
	iWiW::Matrix{Float64}
	l::Vector{Float64}

	#per cone
	mu::Vector{Float64}
	wbs::Vector{Float64} # sqrt(s/z) for POC, wb for SOC
	zik::Vector{Float64} # storage for compute_scaling
	sik::Vector{Float64}
	function Scaling(p::Problem) 
		return Scaling(zeros(p.k,p.k), zeros(p.k), p.cones)
	end
	function Scaling(iWiW,l,cones) 
		md = maximum(conedim.(cones))
		zik,sik = zeros(md),zeros(md)
		return new(iWiW, l, zeros(length(cones)), zeros(cones[end].offs + conedim(cones[end])), zik, sik)
	end
end

function compute_scaling(cone::POC, cind, scaling::SqrScaling, s, z)
	for i = 1:cone.dim
		ii = cti(cone, i)
		scaling.iWiW[ii,ii] = z[ii] / s[ii]
		scaling.l[ii] = sqrt(s[ii] * z[ii])
		scaling.wbs[ii] = sqrt(s[ii] / z[ii])
	end
end


function compute_scaling(c::SOC, cind, scaling::SqrScaling, s, z)
	function cone_prod(vect,len)
		bsum = vect[1] * vect[1]
		for i=2:len
			bsum -= vect[i] * vect[i]
		end
		return bsum
	end
	sik,zik = scaling.sik,scaling.zik
	wl = c.dim
	for ii = 1:wl
		i = cti(c, ii)
		sik[ii] = s[i]
		zik[ii] = z[i]
	end
	nrmz = sqrt(cone_prod(zik, wl))
	nrms = sqrt(cone_prod(sik, wl))
	rmul!(zik, 1/nrmz)
	rmul!(sik, 1/nrms)
	zbk = zik
	sbk = sik

	nsum = 0.0
	for i = 1:wl
		nsum += zbk[i] * sbk[i]
	end
	gamma = sqrt((1 + nsum)/2)

	# compute scaling point
	wb = @view scaling.wbs[cti(c,1):cti(c,c.dim)]
	wb[1] = sbk[1] + zbk[1]
	for i = 2:wl 
		wb[i] = sbk[i] - zbk[i]
	end
	rmul!(wb, 1/(2*gamma))

	# compute scaling variable
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