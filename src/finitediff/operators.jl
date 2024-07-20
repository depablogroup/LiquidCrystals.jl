# using LinearAlgebra: mul!
# using StaticArrays: SVector


struct CenteredDifference{N, T}
    dr::NTuple{N, T}
    coeffs::NTuple{N, SVector{3, T}}
end

# Accuracy order: two
function CenteredDifference(derivative_order::Int, Δr::NTuple{N, T}) where {N, T}
    @assert derivative_order in (1, 2)  # make sure order derivative is 1 or 2
    n = derivative_order
    v = if n == 1
        SVector{3, T}(-1 // 2, 0, 1 // 2)  # Use the type T of Δr
    elseif n == 2
        SVector{3, T}(1, -2, 1)
    end
    coeffs = Tuple(v / Δrᵢ^n for Δrᵢ in Δr)
    return CenteredDifference(Δr, coeffs)
end

function CenteredDifference{N}(derivative_order::Int, Δr::T) where {N, T <: Real}
    dr = ntuple(i -> Δr, Val(N))
    return CenteredDifference(derivative_order, dr)
end


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

function LinearAlgebra.mul!(B::AbstractArray, Op::CenteredDifference{N}, A::AbstractArray) where {N}
	sz = size(A) .- 1
	CI = CartesianIndices(UnitRange.(2, sz))
	#C = CartesianIndices(A)
	coeffs = Op.coeffs
	# While not on the edges
	#for I in C[2:(end - 1), 2:(end - 1)]
	for I in CI
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

function apply_BCs(op, u, box::BoxBC, extended)
	sz = size(extended) .- 1
	CI = CartesianIndices(UnitRange.(2, sz))
	extended[CI] .= u

	for bc in box.bcs
		apply_BCs(op, u, bc, extended)
	end

	return extended
end

apply_BCs(op, u, bc::DirichletAxisBC, extended) = extended

function build_caches(op, box::BoxBC, u0) #do we need op?
	sz = size(u0) .+ 2
	
	extended = zeros(sz)
	result = copy(extended)

	for bc in box.bcs
		initialize(op, bc, extended)
	end

	
	return extended, result

end

function initialize(op, bc::AxisBC, extended)

	return extended

end

function get_dr(::Type{XAxis})
	return 1
end

function get_dr(::Type{YAxis})
	return 2
end

function get_dr(::Type{ZAxis})
	return 3
end

function initialize(op, bc::DirichletAxisBC{A}, extended) where {A <: Axis}
	sz = size(extended)
	lo, hi = ghost_ranges(sz, A) 
	
	CI_lo = CartesianIndices(lo)       #low-value of x-axis
	extended[CI_lo] .= bc.c_l

	CI_hi = CartesianIndices(hi)      #high-value of x-axis
	extended[CI_hi] .= bc.c_h

	return extended

end

function apply_BCs(op, u, bc::NeumannAxisBC{A}, extended)  where {A <: Axis}

	axis=get_dr(A)
	sz = size(extended)
	lo_g, hi_g = ghost_ranges(sz, A) 

	CI_lb, CI_hb = boundary_ranges(sz, bc)


	
	CI_lg = CartesianIndices(lo_g)  #low-value of axis
	extended[CI_lg] .= u[CI_lb] .+ 2*bc.c_l*op.dr[axis]  

	CI_hg = CartesianIndices(hi_g)  #high-value of axis
	extended[CI_hg] .= u[CI_hb]  .- 2*bc.c_h*op.dr[axis]  


	return extended

end

function apply_BCs(op, u, bc::PeriodicAxisBC{A}, extended) where {A <: Axis}

	sz = size(extended)
	lo_g, hi_g = ghost_ranges(sz, A) 

	CI_lb, CI_hb = boundary_ranges(sz, bc)
	
	CI_lg = CartesianIndices(lo_g)  #low-value of x-axis
	extended[CI_lg] .= u[CI_lb]   

	CI_hg = CartesianIndices(hi_g)  #high-value of x-axis
	extended[CI_hg] .= u[CI_hb]  


	return extended

end

function apply_BCs(op, u, bc::RobinAxisBC{A}, extended) where {A <: Axis}

	axis=get_dr(A)
	sz = size(extended)
	lo_g, hi_g = ghost_ranges(sz, A) 

	CI_lb, CI_hb = boundary_ranges(sz, bc)


	
	CI_lg = CartesianIndices(lo_g)  #low-value of axis
	extended[CI_lg] .=  (bc.c_l .+ (bc.b_l / (2*op.dr[axis]))  * u[CI_lb]) / (bc.a_l + (bc.b_l / (2* op.dr[axis] ) ) ) 

	CI_hg = CartesianIndices(hi_g)  #high-value of axis
	extended[CI_hg] .= (bc.c_h .- (bc.b_h / (2*op.dr[axis])) * u[CI_hb]) / (bc.a_h - (bc.b_h / (2* op.dr[axis] ) )) 


	return extended

end

function ghost_ranges(sz::NTuple{2},::Type{XAxis})
	m, n = sz
	return (2:m-1, 1:1), (2:m-1, n:n)
end

function ghost_ranges(sz::NTuple{2},::Type{YAxis})
	m, n = sz
	return (1:1, 2:n-1), (m:m, 2:n-1)
end

function ghost_ranges(sz::NTuple{3},::Type{XAxis})
	l, m, n = sz
	return (2:l-1, 2:m-1, 1:1), (2:l-1, 2:m-1, n:n)
end

function ghost_ranges(sz::NTuple{3},::Type{YAxis})
	l, m, n = sz
	return (2:l-1, 1:1, 2:n-1), (2:l-1, m:m, 2:n-1)
end

function ghost_ranges(sz::NTuple{3},::Type{ZAxis})
	l, m, n = sz
	return (1:1, 2:m-1, 2:n-1), (l:l, 2:m-1, 2:n-1)
end

const DerivativeAxisBC{T} = Union{NeumannAxisBC{T}, RobinAxisBC{T}}

function boundary_ranges(sz::NTuple{2}, ::DerivativeAxisBC{XAxis})
	m, n = sz
	#extract two as it is the extended matrix size
	lo = CartesianIndices((1:m-2, 2:2))    
	hi = CartesianIndices((1:m-2, (n-2-1):(n-2-1)))

	return lo, hi
end

function boundary_ranges(sz::NTuple{2}, ::DerivativeAxisBC{YAxis})
	m, n = sz
	#extract two as it is the extended matrix size
	lo = CartesianIndices((2:2, 1:n-2))    
	hi = CartesianIndices((m-2-1:m-2-1, 1:n-2))

	return lo, hi
end

function boundary_ranges(sz::NTuple{3}, ::DerivativeAxisBC{XAxis})
	l, m, n = sz
	#extract two as it is the extended matrix size
	lo = CartesianIndices((1:l-2,1:m-2, 2:2))    
	hi = CartesianIndices((1:l-2,1:m-2, (n-2-1):(n-2-1)))

	return lo, hi
end

function boundary_ranges(sz::NTuple{3}, ::DerivativeAxisBC{YAxis})
	l, m, n = sz
	#extract two as it is the extended matrix size
	lo = CartesianIndices((1:l-2,2:2, 1:n-2))    
	hi = CartesianIndices((1:l-2,m-2-1:m-2-1,1:n-2))

	return lo, hi
end

function boundary_ranges(sz::NTuple{3}, ::DerivativeAxisBC{ZAxis})
	l, m, n = sz
	#extract two as it is the extended matrix size
	lo = CartesianIndices((2:2,1:m-2, 1:n-2))    
	hi = CartesianIndices((l-2-1:l-2-1,1:m-2, 1:n-2))

	return lo, hi
end

function boundary_ranges(sz::NTuple{2}, ::PeriodicAxisBC{XAxis})
	m, n = sz

	#extract two as it is the extended matrix size
	lo = CartesianIndices((1:m-2, n-2:n-2))    
	hi = CartesianIndices((1:m-2, 1:1))

	return lo, hi
end

function boundary_ranges(sz::NTuple{2}, ::PeriodicAxisBC{YAxis})
	m, n = sz

	#extract two as it is the extended matrix size
	lo = CartesianIndices((m-2:m-2, 1:n-2))    
	hi = CartesianIndices((1:1, 1:n-2))

	return lo, hi
end

function boundary_ranges(sz::NTuple{3}, ::PeriodicAxisBC{XAxis})
	l, m, n = sz

	#extract two as it is the extended matrix size
	lo = CartesianIndices((1:l-2,1:m-2, n-2:n-2))    
	hi = CartesianIndices((1:l-2,1:m-2, 1:1))

	return lo, hi
end

function boundary_ranges(sz::NTuple{3}, ::PeriodicAxisBC{YAxis})
	l, m, n = sz

	#extract two as it is the extended matrix size
	lo = CartesianIndices((1:l-2,m-2:m-2, 1:n-2))    
	hi = CartesianIndices((1:l-2,1:1, 1:n-2))

	return lo, hi
end

function boundary_ranges(sz::NTuple{3}, ::PeriodicAxisBC{ZAxis})
	l, m, n = sz

	#extract two as it is the extended matrix size
	lo = CartesianIndices((l-2:l-2,1:m-2, 1:n-2))    
	hi = CartesianIndices((1:1,    1:m-2, 1:n-2))

	return lo, hi
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
