using LinearAlgebra


struct CenteredDifference{N, T}
	dr::NTuple{N, T}
	coeffs::NTuple{N, SVector{3, T}}
end

#accuracy order two
function CenteredDifference(derivative_order::Int, Δr::NTuple{N,T}) where {N,T}
	@assert derivative_order in (1,2)     #to make sure order derivative is 1 or 2

	n = derivative_order
	v = if n == 1
		SVector{3,T}(-0.5,0,0.5)    #we converted the SVector to type T
	elseif n == 2
		SVector{3,T}(1,-2,1)
	end
	coeffs = Tuple(v / Δrᵢ^n for Δrᵢ in Δr)
	return CenteredDifference(Δr, coeffs)
end

function CenteredDifference{N}(derivative_order::Int, Δr::T) where {N, T <: Real}
	dr = ntuple(i -> Δr, Val(N))
	return CenteredDifference(derivative_order, dr)
end


abstract type BoundaryCondition end

struct DirichletBC{N, T <: Real} <: BoundaryCondition
	c_l::NTuple{N, T}
	c_h::NTuple{N, T}
end

function DirichletBC{N}(c::T) where {N, T}
	c_l = ntuple(i -> c, Val(N))
	return DirichletBC(c_l, c_l)    #or c_h????
end

struct NeumannBC{N, T <: Real} <: BoundaryCondition
	c_l::NTuple{N, T}
	c_h::NTuple{N, T}
end

#if BCs are the same at each end
function NeumannBC{N}(c::T) where {N, T}
	c_l = ntuple(i -> c, Val(N))
	return NeumannBC(c_l, c_l)
end

struct PeriodicBC <: BoundaryCondition
end


function Base.:*(Op::CenteredDifference{2}, A::AbstractMatrix)
	B = fill!(similar(A) , zero(eltype(A)))
	C = CartesianIndices(A)
	coeffs = Op.coeffs
	# While not on the edges
	for I in C[2:(end - 1), 2:(end - 1)]
		B[I] = laplacian(A, I, coeffs)
	end
	return B
end

function LinearAlgebra.mul!(B, Op::CenteredDifference{2}, A::AbstractMatrix)
	C = CartesianIndices(A)
	coeffs = Op.coeffs
	# While not on the edges
	for I in C[2:(end - 1), 2:(end - 1)]
		B[I] = laplacian(A, I, coeffs)
	end
	return B
end

function laplacian(A, I, coeffs::NTuple{2})
	Ix = CartesianIndex(0, 1)
	Iy = CartesianIndex(1, 0)

	Cx, Cy = coeffs

	return (
		Cx[1] * A[I - Ix] + Cx[2] * A[I] + Cx[3] * A[I + Ix] +
		Cy[1] * A[I - Iy] + Cy[2] * A[I] + Cy[3] * A[I + Iy]
	)
end

function build_caches(op, bc::BoundaryCondition, u0)
	sz = size(u0) .+ 2
	extended = zeros(sz)
	result = copy(extended)

	return extended, result
end

function build_caches(op, bc::DirichletBC{2}, u0) #do we need op?
	m, n = size(u0) .+ 2

	extended = zeros(m, n)
	result = copy(extended)

	CI_l1 = CartesianIndices((2:m-1, 1:1))  #1st axis and low-value
	extended[CI_l1] .= bc.c_l[1]

	CI_h1 = CartesianIndices((2:m-1, n:n))  #1st axis and high-value
	extended[CI_h1] .= bc.c_h[1]

	CI_l2 = CartesianIndices((1:1, 2:n-1))  #2nd axis and low-value
	extended[CI_l2] .= bc.c_l[2]

	Cl_h2 = CartesianIndices((m:m, 2:n-1))  #2nd axis and high-value
	extended[Cl_h2] .= bc.c_h[2]

	return extended, result
end

function apply_BCs(op, u, bc::DirichletBC{2}, extended)
	m, n = size(u) .+ 2
	extended[2:end-1, 2:end-1] .= u
	return extended
end

function apply_BCs(op, u, bc::NeumannBC{2}, extended)
	m, n = size(u) .+ 2

	CI_l1 = CartesianIndices((2:m-1, 1:1))  #1st axis and low-value of axis
	extended[CI_l1] .= u[:, 2] .+ 2*bc.c_l[1]*op.dr[1]

	CI_h1 = CartesianIndices((2:m-1, n:n))  #1st axis and high-value of axis
	extended[CI_h1] .= u[:, end-1]  .- 2*bc.c_h[1]*op.dr[1]

	CI_l2 = CartesianIndices((1:1, 2:n-1))  #2nd axis and low-value of axis
	extended[CI_l2] .= u[2:2, :]   .+ 2*bc.c_l[2]*op.dr[2]

	CI_h2 = CartesianIndices((m:m, 2:n-1))  #2nd axis and high-value of axis
	extended[CI_h2] .= u[end-1:end-1, :] .- 2*bc.c_h[2]*op.dr[2]

	extended[2:end-1, 2:end-1] .= u
	return extended
end

function apply_BCs(op, u, bc::PeriodicBC, extended)
	m, n = size(u) .+ 2

	CI_l1 = CartesianIndices((2:m-1, 1:1))  #1st axis and low-value of axis
	extended[CI_l1] .= u[:, end]

	CI_h1 = CartesianIndices((2:m-1, n:n))  #1st axis and high-value of axis
	extended[CI_h1] .= u[:, 1]

	CI_l2 = CartesianIndices((1:1, 2:n-1))  #2nd axis and low-value of axis
	extended[CI_l2] .= u[end:end, :]

	CI_h2 = CartesianIndices((m:m, 2:n-1))  #2nd axis and high-value of axis
	extended[CI_h2] .= u[1:1, :]

	extended[2:end-1, 2:end-1] .= u
	return extended
end
