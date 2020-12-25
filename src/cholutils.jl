# note that this function assumes that the number of nonzeros, number of cols, and number of rows is already the same
function Base.copyto!(dest::SuiteSparse.CHOLMOD.Sparse, src::SparseMatrixCSC) 
	crep = unsafe_load(dest.ptr)
	ht,wd = size(src)
	nnz = length(src.nzval)
	@assert crep.nzmax == nnz "$(crep.nzmax) neq $(nnz)"
	@assert crep.nrow == src.m
	@assert crep.ncol == src.n
	for i=1:src.n+1
		unsafe_store!(crep.p, (@inbounds src.colptr[i])-1, i)
	end
	for i=1:nnz
		unsafe_store!(crep.i, (@inbounds src.rowval[i])-1, i)
		unsafe_store!(crep.x, (@inbounds src.nzval[i]), i)
	end
end

function Base.copyto!(dest::SuiteSparse.CHOLMOD.Dense{T}, src::AbstractArray{T, 1}) where {T}
	uref = unsafe_load(dest.ptr)
	@assert uref.nrow == length(src) "$(uref.nrow) neq $(length(src))"
	for i=1:length(src)
		unsafe_store!(uref.x, src[i], i)
	end
end

function LinearAlgebra.lmul!(l::Diagonal, r::SuiteSparse.CHOLMOD.Sparse)
	crep = unsafe_load(r.ptr)
	diag = l.diag
	for i=1:crep.ncol
		colstart = unsafe_load(crep.p, i)+1
		colend = unsafe_load(crep.p, i+1)
		for j=colstart:colend
			rowidx = unsafe_load(crep.i, j)
			cellv = unsafe_load(crep.x, j)
			unsafe_store!(crep.x, cellv*diag[i], j)
		end
	end
end

struct LdivWorkspace{T<:SuiteSparse.CHOLMOD.VTypes}
	Xref::Ref{Ptr{I}} where I
	Y::Ref{Ptr{I}} where I
	E::Ref{Ptr{I}} where I
	LdivWorkspace(::Type{T}) where T = new{T}(Ref(C_NULL), Ref(C_NULL), Ref(C_NULL))
end

function div!(F::SuiteSparse.CHOLMOD.Factor{Tv}, x::SuiteSparse.CHOLMOD.Dense{Tv}, B::SuiteSparse.CHOLMOD.Dense{Tv}, W::LdivWorkspace{Tv}) where Tv<:SuiteSparse.CHOLMOD.VTypes
    if size(F,1) != size(B,1)
        throw(DimensionMismatch("LHS and RHS should have the same number of rows. " *
            "LHS has $(size(F,1)) rows, but RHS has $(size(B,1)) rows."))
    end
    if !issuccess(F)
        s = unsafe_load(pointer(F))
        if s.is_ll == 1
            throw(LinearAlgebra.PosDefException(s.minor))
        else
            throw(LinearAlgebra.ZeroPivotException(s.minor))
        end
    end
    W.Xref[] = Base.unsafe_convert(Ptr{SuiteSparse.CHOLMOD.C_Dense{Tv}}, x)
    res = ccall((SuiteSparse.CHOLMOD.@cholmod_name("solve2"),:libcholmod), Cint,
            (	# input
            	Cint, Ptr{SuiteSparse.CHOLMOD.C_Factor{Tv}}, Ptr{SuiteSparse.CHOLMOD.C_Dense{Tv}}, Ptr{Nothing},
            	# output
            	Ref{Ptr{Nothing}}, Ptr{Nothing},
            	# workspace
            	Ref{Ptr{Nothing}}, Ref{Ptr{Nothing}},
            	# common
            	Ptr{UInt8}),
                SuiteSparse.CHOLMOD.CHOLMOD_A, F, B, C_NULL,
                W.Xref, C_NULL,
                W.Y, W.E,
                SuiteSparse.CHOLMOD.common_struct[Threads.threadid()])
    return res
end

function change_xtype!(F::SuiteSparse.CHOLMOD.Sparse{T}, xtype::Int32) where T
	return ccall((SuiteSparse.CHOLMOD.@cholmod_name("sparse_xtype"), :libcholmod), Cint,
		(Cint, Ptr{SuiteSparse.CHOLMOD.C_Sparse{T}}, Ptr{UInt8}),
		xtype, F, SuiteSparse.CHOLMOD.common_struct[Threads.threadid()])
end