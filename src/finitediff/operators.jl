# using LinearAlgebra: mul!
# using StaticArrays: SVector


struct CenteredDifference{N, T, Tag}
	dr::NTuple{N, T}
	coeffs::NTuple{N, SVector{3, T}}
end

#accuracy order two
function CenteredDifference{Tag}(Δr::NTuple{N, T}) where {N, T, Tag}
	# @assert derivative_order in (1,2)     #to make sure order derivative is 1 or 2
	
	n = derivative_order(Tag)
	v = if n == 1 
		SVector{3,T}(-0.5,0,0.5)    #we converted the SVector to type T
	elseif n == 2
		SVector{3,T}(1,-2,1)
	end
	coeffs = Tuple(v / Δrᵢ^n for Δrᵢ in Δr)
	return CenteredDifference{N, T, Tag}(Δr, coeffs)
end

function CenteredDifference{N, Tag}(Δr::T) where {N, T, Tag}
	dr = ntuple(i -> Δr, Val(N))
	return CenteredDifference{Tag}(dr)
end

abstract type OperatorTag end

struct Laplacian <: OperatorTag end

struct Divergence <: OperatorTag end

struct Gradient <: OperatorTag end

derivative_order(::Type{Laplacian}) = 2

derivative_order(::Type{Divergence}) = 1

derivative_order(::Type{Gradient}) = 1

concretize(::Type{Laplacian}) = laplacian

concretize(::Type{Divergence}) = divergence

concretize(::Type{Gradient}) = gradient


abstract type BoundaryCondition end

abstract type AxisBC <: BoundaryCondition end

abstract type Axis end

struct XAxis <: Axis end

struct YAxis <: Axis end

struct ZAxis <: Axis end

struct DirichletBC{N, T <: Real} <: BoundaryCondition
    c_l::NTuple{N, T}
    c_h::NTuple{N, T}
end

function DirichletBC{N}(c::T) where {N, T}
    c_ = ntuple(i -> c, Val(N))
    return DirichletBC(c_, c_)
end

struct NeumannBC{N, T <: Real} <: BoundaryCondition
    c_l::NTuple{N, T}
    c_h::NTuple{N, T}
end

# If BCs are the same at each end
function NeumannBC{N}(c::T) where {N, T}
    c_ = ntuple(i -> c, Val(N))
    return NeumannBC(c_, c_)
end

struct PeriodicBC <: BoundaryCondition
end



struct BoxBC{N, T <: NTuple{N, AxisBC}} <: BoundaryCondition
	bcs::T
end

BoxBC(bcs::AxisBC...) = BoxBC(bcs)

struct DirichletAxisBC{A <: Axis, T <: Real} <: AxisBC
	c_l::T
	c_h::T

	function DirichletAxisBC{A}(c_l::T, c_h::T) where {A <: Axis, T}
		return new{A, T}(c_l, c_h)
	end
end
	
DirichletAxisBC{A}(c) where {A <: Axis} = DirichletAxisBC{A}(c, c)


struct NeumannAxisBC{A <: Axis, T <: Real} <: AxisBC
	c_l::T
	c_h::T

function NeumannAxisBC{A}(c_l::T, c_h::T) where {A <: Axis, T}
	return new{A, T}(c_l, c_h)
	end
end
	
NeumannAxisBC{A}(c) where {A <: Axis} = NeumannAxisBC{A}(c, c)


struct RobinAxisBC{A <: Axis, T <: Real} <: AxisBC
	a_l::T
	b_l::T
	c_l::T
	a_h::T
	b_h::T
	c_h::T

	function RobinAxisBC{A}(a_l::T, b_l::T, c_l::T, a_h::T, b_h::T, c_h::T) where {A <: Axis, T}
		return new{A, T}(a_l, b_l, c_l, a_h, b_h, c_h)
	end
end


struct PeriodicAxisBC{A <: Axis, T <: Real} <: AxisBC
    c_l::T
    c_h::T

    function PeriodicAxisBC{A}(c_l::T, c_h::T) where {A <: Axis, T}
    	return new{A, T}(c_l, c_h)
    end
end
	
PeriodicAxisBC{A}(c) where {A <: Axis} = PeriodicAxisBC{A}(c, c)
	

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

function gradient(A, I, coeffs::NTuple{2})
	Iy = CartesianIndex(0, 1)
	Ix = CartesianIndex(1, 0)

	Cx, Cy = coeffs
	∇_x = Cx[1] * A[I - Ix] + Cx[2] * A[I] + Cx[3] * A[I + Ix] 
	∇_y = Cy[1] * A[I - Iy] + Cy[2] * A[I] + Cy[3] * A[I + Iy]
	
	return SVector(∇_x , ∇_y)
end

function gradient(A, I, coeffs::NTuple{3})
	Iz = CartesianIndex(0, 0, 1)
	Iy = CartesianIndex(0, 1, 0)
	Ix = CartesianIndex(1, 0, 0)

	Cx, Cy, Cz = coeffs
	∇_x = Cx[1] * A[I - Ix] + Cx[2] * A[I] + Cx[3] * A[I + Ix] 
	∇_y = Cy[1] * A[I - Iy] + Cy[2] * A[I] + Cy[3] * A[I + Iy]
	∇_z = Cz[1] * A[I - Iz] + Cz[2] * A[I] + Cz[3] * A[I + Iz]
	
	return SVector(∇_x, ∇_y, ∇_z)
end

function divergence(A, I, coeffs::NTuple{2})
	Iy = CartesianIndex(0, 1)
	Ix = CartesianIndex(1, 0)

	Cx, Cy = coeffs
	div = ( Cx[1] * A[I - Ix][1] + Cx[2] * A[I][1] + Cx[3] * A[I + Ix][1] 
	+ Cy[1] * A[I - Iy][2] + Cy[2] * A[I][2] + Cy[3] * A[I + Iy][2] )
	
	return div
end

function divergence(A, I, coeffs::NTuple{3})
	Iz = CartesianIndex(0, 0, 1)
	Iy = CartesianIndex(0, 1, 0)
	Ix = CartesianIndex(1, 0, 0)

	Cx, Cy, Cz = coeffs
	div = ( Cx[1] * A[I - Ix][1] + Cx[2] * A[I][1] + Cx[3] * A[I + Ix][1] 
	+ Cy[1] * A[I - Iy][2] + Cy[2] * A[I][2] + Cy[3] * A[I + Iy][2]
	+ Cz[1] * A[I - Iz][3] + Cz[2] * A[I][3] + Cz[3] * A[I + Iz][3] )
	
	return div
end

function laplacian(A, I, coeffs::NTuple{2})
    Iy = CartesianIndex(0, 1)
    Ix = CartesianIndex(1, 0)

    Cx, Cy = coeffs

    return (
        Cx[1] * A[I - Ix] + Cx[2] * A[I] + Cx[3] * A[I + Ix] +
        Cy[1] * A[I - Iy] + Cy[2] * A[I] + Cy[3] * A[I + Iy]
    )
end

function laplacian(A, I, coeffs::NTuple{3})
	Ix = CartesianIndex(0,0,1)
	Iy = CartesianIndex(0,1,0)
	Iz = CartesianIndex(1,0,0)

	Cx, Cy, Cz = coeffs
	
	return (
		Cx[1] * A[I - Ix] + Cx[2] * A[I] + Cx[3] * A[I + Ix] +
		Cy[1] * A[I - Iy] + Cy[2] * A[I] + Cy[3] * A[I + Iy] +
		Cz[1] * A[I - Iz] + Cz[2] * A[I] + Cz[3] * A[I + Iz]
	)
end

function laplacian(A, I, coeffs::NTuple{3})
	Iz = CartesianIndex(0,0,1)
	Iy = CartesianIndex(0,1,0)
	Ix = CartesianIndex(1,0,0)

	Cx, Cy, Cz = coeffs
	
	return (
		Cx[1] * A[I - Ix] + Cx[2] * A[I] + Cx[3] * A[I + Ix] +
		Cy[1] * A[I - Iy] + Cy[2] * A[I] + Cy[3] * A[I + Iy] +
		Cz[1] * A[I - Iz] + Cz[2] * A[I] + Cz[3] * A[I + Iz]
	)
end

# The next applies to Laplacian and Divergence
auxiliar_caches(::Type{E}, op) where{E} = E 
auxiliar_caches(::Type{E}, ::CenteredDifference{N, T, Gradient}) where {N, T, E} = SVector{N, E}


function build_caches(op::CenteredDifference{N, T, Tag}, box::BoxBC, u0) where {N, T, Tag} #do we need op?, yes we do! 
	sz = size(u0) .+ 2
	
	E = eltype(u0)
	extended = zeros(E, sz)
	result = zeros(auxiliar_caches(E, op) , sz)

	for bc in box.bcs
		initialize(op::CenteredDifference{N, T, Tag}, bc, extended)
	end

	
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
