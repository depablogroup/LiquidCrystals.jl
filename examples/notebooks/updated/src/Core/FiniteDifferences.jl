#src/Core/FunctionalDerivatives.jl
module FiniteDifferences

using LinearAlgebra
using LiquidCrystals
using ..Utils
using StaticArrays

export CenteredDifference, ForwardDifference, BackwardDifference
export Gradient, Laplacian, Divergence, Hessian
export build_caches
export ghost_ranges, boundary_ranges, surface_ranges_extended
export AnchoringAxisBC
export apply_BCs, initialize, build_caches
export mul!
export DirichletAxisBC, AnchoringAxisBC

abstract type OperatorTag end

struct Laplacian <: OperatorTag end
struct Divergence <: OperatorTag end
struct Gradient <: OperatorTag end
struct Hessian <: OperatorTag end
struct Derivative_x <: OperatorTag end
struct Derivative_y <: OperatorTag end
	
derivative_order(::Type{Laplacian}) = 2
derivative_order(::Type{Divergence}) = 1
derivative_order(::Type{Gradient}) = 1
derivative_order(::Type{Hessian}) = 2
derivative_order(::Type{Derivative_x}) = 1
derivative_order(::Type{Derivative_y}) = 1
	
concretize(::Type{Laplacian}) = laplacian
concretize(::Type{Divergence}) = divergence
concretize(::Type{Gradient}) = gradient
concretize(::Type{Derivative_x}) = derivative_x
concretize(::Type{Derivative_y}) = derivative_y
concretize(::Type{Hessian}) = hessian


# ----------------------------------------------------- Forward Difference -------------------------------------------------------------- #
struct ForwardDifference{N, M, T, Tag}
    dr::NTuple{N, T}
    coeffs::NTuple{N, SVector{M, T}}
	offsets::NTuple{N, SVector{M, Int}}
end

# Second-order accurate forward difference coefficients
function ForwardDifference{Tag}(Δr::NTuple{N, T}) where {N, T, Tag}
    n = derivative_order(Tag)
	if n == 1
        # Second-order accurate forward difference coefficients for first derivative
		m = 3
        v = SVector{m,T}(-3/2, 2.0, -1/2)
		offsets = ntuple(_ -> SVector{3, Int}(0, 1, 2), N)
    elseif n == 2
        # Second-order accurate forward difference coefficients for second derivative
		m = 4
        v = SVector{m,T}(2.0, -5.0, 4.0, -1.0)
		offsets = ntuple(_ -> SVector{4, Int}(0, 1, 2, 3), N)
    end
    coeffs = Tuple(v / Δrᵢ^n for Δrᵢ in Δr)
    return ForwardDifference{N, m, T, Tag}(Δr, coeffs, offsets)
end

function ForwardDifference{N, Tag}(Δr::T) where {N, T, Tag}
    dr = ntuple(i -> Δr, Val(N))
    return ForwardDifference{Tag}(dr)
end

# ----------------------------------------------------- Backward Difference ------------------------------------------------------------- #
struct BackwardDifference{N, M, T, Tag}
    dr::NTuple{N, T}
    coeffs::NTuple{N, SVector{M, T}}
	offsets::NTuple{N, SVector{M, Int}}
end

# Second-order accurate backward difference coefficients
function BackwardDifference{Tag}(Δr::NTuple{N, T}) where {N, T, Tag}
    n = derivative_order(Tag)
    if n == 1
        # Second-order accurate backward difference coefficients for first derivative
		m = 3
        v = SVector{m,T}(1/2, -2.0, 3/2)
		offsets = ntuple(_ -> SVector{3, Int}(-2, -1, 0), N)
    elseif n == 2
        # Second-order accurate backward difference coefficients for second derivative
		m = 4
        v = SVector{m,T}(-1.0, 4.0, -5.0, 2.0)
		offsets = nyuple(_ -> SVector{4, Int}(-3, -2, -1, 0), N)
    end
    coeffs = Tuple(v / Δrᵢ^n for Δrᵢ in Δr)
    return BackwardDifference{N, m, T, Tag}(Δr, coeffs, offsets)
end

function BackwardDifference{N, Tag}(Δr::T) where {N, T, Tag}
    dr = ntuple(i -> Δr, Val(N))
    return BackwardDifference{Tag}(dr)
end

# ----------------------------------------------------- Centered Difference ---------------------------------------------------------------- #
struct CenteredDifference{N, T, Tag, C, D}
	dr::NTuple{N, T}
	coeffs::C
	offsets::D
end

function CenteredDifference{Hessian}(Δr::NTuple{N, T}) where {N, T}
	    
	coeffs_Δ = Tuple(SVector{3, T}(1.0, -2.0, 1.0) / (Δrᵢ^2) for Δrᵢ in Δr)
	offsets_Δ = ntuple(_ -> SVector{3, Int}(-1, 0, 1), N)
	
	mixed_terms = Vector{SVector{4, T}}()
	for i in 1:N
	    for j in i+1:N
	        push!(mixed_terms, SVector{4, T}(0.25, -0.25, -0.25, 0.25) / (Δr[i] * Δr[j]))
	    end
	end
	offsets_mixed = ntuple(_ -> (SVector{2, Int}(-1, -1), SVector{2, Int}(1, -1), SVector{2, Int}(-1, 1), SVector{2, Int}(1, 1)), N)

	coeffs = (coeffs_Δ, mixed_terms)
	offsets = (offsets_Δ, offsets_mixed)
	
	return CenteredDifference{N, T, Hessian, typeof(coeffs), typeof(offsets)}(Δr, coeffs, offsets)
end

function CenteredDifference{Tag}(Δr::NTuple{N, T}) where {N, T, Tag}
	    
	n = derivative_order(Tag)
	if n == 1
	    v = SVector{3, T}(-0.5, 0.0, 0.5)
		offsets = ntuple(_ -> SVector{3, Int}(-1, 0, 1), N)
	elseif n == 2
	    v = SVector{3, T}(1.0, -2.0, 1.0)
		offsets = ntuple(_ -> SVector{3, Int}(-1, 0, 1), N)
	end

	coeffs = Tuple(v / Δrᵢ^n for Δrᵢ in Δr)
	return CenteredDifference{N, T, Tag, typeof(coeffs), typeof(offsets)}(Δr, coeffs, offsets)
end
	
function CenteredDifference{N, Tag}(Δr::T) where {N, T, Tag}
	dr = ntuple(i -> Δr, Val(N))
	return CenteredDifference{Tag}(dr)
end


# ------------------------------------------------- Operations -----------------------------------------------------------#
function gradient(A, I::CartesianIndex{3}, coeffs::NTuple{3}, offsets::NTuple{3})
	Iz = CartesianIndex(0, 0, 1)
	Iy = CartesianIndex(0, 1, 0)
	Ix = CartesianIndex(1, 0, 0)
	
	Cx, Cy, Cz = coeffs
	Ox, Oy, Oz = offsets
	∇_x = sum(c * A[I + o * Ix] for (c, o) in zip(Cx, Ox))
	∇_y = sum(c * A[I + o * Iy] for (c, o) in zip(Cy, Oy))
	∇_z = sum(c * A[I + o * Iz] for (c, o) in zip(Cz, Oz))
	    
	return SVector(∇_x, ∇_y, ∇_z)
end

function derivative(A, I::CartesianIndex{3}, Ij::CartesianIndex{3}, Cj, Oj)
	∇_j = sum(c * A[I + o * Ij] for (c, o) in zip(Cj, Oj))
	return ∇_j
end

function laplacian(A, I::CartesianIndex{3}, coeffs::NTuple{3}, offsets::NTuple{3})
	Iz = CartesianIndex(0, 0, 1)
	Iy = CartesianIndex(0, 1, 0)
	Ix = CartesianIndex(1, 0, 0)
	
	Cxx, Cyy, Czz = coeffs
	Oxx, Oyy, Ozz = offsets

	∂x² = sum(c * A[I + o * Ix] for (c, o) in zip(Cxx, Oxx))
	∂y² = sum(c * A[I + o * Iy] for (c, o) in zip(Cyy, Oyy))
	∂z² = sum(c * A[I + o * Iz] for (c, o) in zip(Czz, Ozz))
		
	return sum([∂x², ∂y², ∂z²]) 
end

function hessian(A, I, coeffs::Tuple{NTuple{3, SVector{3, T}}, Vector{SVector{4, T}}}, offsets::Tuple{NTuple{3, SVector{3, Int}}, NTuple{3, NTuple{4, SVector{2, Int}}}}) where T

	Iz = CartesianIndex(0, 0, 1)
	Iy = CartesianIndex(0, 1, 0)
	Ix = CartesianIndex(1, 0, 0)
	
	coeffs_pure, coeffs_mixed = coeffs
	offsets_pure, offsets_mixed = offsets
	Cxx, Cyy, Czz = coeffs_pure
	Cxy, Cyz, Czx = coeffs_mixed
	Oxx, Oyy, Ozz = offsets_pure
	Oxy, Oyz, Ozx = offsets_mixed

	∂x² = sum(c * A[I + o * Ix] for (c, o) in zip(Cxx, Oxx))
	∂y² = sum(c * A[I + o * Iy] for (c, o) in zip(Cyy, Oyy))
	∂z² = sum(c * A[I + o * Iz] for (c, o) in zip(Czz, Ozz))
	∂x∂y = sum(c * A[I + o[1] * Ix + o[2] * Iy] for (c, o) in zip(Cxy, Oxy))
	∂y∂x = ∂x∂y
	∂y∂z = sum(c * A[I + o[1] * Iy + o[2] * Iz] for (c, o) in zip(Cyz, Oyz))
	∂z∂y = ∂y∂z
	∂z∂x = sum(c * A[I + o[1] * Iz + o[2] * Ix] for (c, o) in zip(Czx, Ozx))
	∂x∂z = ∂z∂x
		
	return @SMatrix [∂x² ∂x∂y ∂x∂z; ∂y∂x ∂y² ∂y∂z; ∂z∂x ∂z∂y ∂z²]
end

# ---------------------------------------------------------- Box Setup ------------------------------------------------------------------#
auxiliar_caches(::Type{E}, op) where{E} = E 
auxiliar_caches(::Type{E}, ::CenteredDifference{N, T, Gradient}) where {N, T, E} = SVector{N, E}
auxiliar_caches(::Type{E}, ::ForwardDifference{N, T, Gradient}) where {N, T, E} = SVector{N, E}
auxiliar_caches(::Type{E}, ::BackwardDifference{N, T, Gradient}) where {N, T, E} = SVector{N, E}
auxiliar_caches(::Type{E}, ::CenteredDifference{N, T, Hessian}) where {N, T, E} = SMatrix{N, N, E}

# -------------------------------------------------------- Custom mul! operator ---------------------------------------------------------#

function LinearAlgebra.mul!(B::AbstractArray, Op::Main.FiniteDifferences.CenteredDifference{N, T, Tag}, A::AbstractArray) where {N, T, Tag}
	
	op = concretize(Tag)
	coeffs = Op.coeffs
	offsets = Op.offsets
	
	sz = size(A) .- 1
	CIs = CartesianIndices(UnitRange.(2, sz))
				
	for I in CIs
		B[I] = op(A, I, coeffs, offsets)
	end
			
	return B
end
#----------------------------------------------------------------------------------------------------------------------------------------------------#

begin

	function build_caches(op::CenteredDifference{N, T, Tag}, box::LiquidCrystals.BoxBC, u0) where {N, T, Tag} #do we need op?, yes we do! 
		sz = size(u0) .+ 2
		
		E = eltype(u0)
		extended = zeros(E, sz)
		result = zeros(auxiliar_caches(E, op) , sz)
	
		for bc in box.bcs
			initialize(op::CenteredDifference{N, T, Tag}, bc, extended)
		end
		
		return extended, result
	
	end

	function build_caches(op::ForwardDifference{N, T, Tag}, box::LiquidCrystals.BoxBC, u0) where {N, T, Tag} #do we need op?, yes we do! 
		sz = size(u0) .+ 2
		
		E = eltype(u0)
		extended = zeros(E, sz)
		result = zeros(auxiliar_caches(E, op) , sz)
	
		for bc in box.bcs
			initialize(op::ForwardDifference{N, T, Tag}, bc, extended)
		end
		
		return extended, result
	
	end

	function build_caches(op::BackwardDifference{N, T, Tag}, box::LiquidCrystals.BoxBC, u0) where {N, T, Tag} #do we need op?, yes we do! 
		sz = size(u0) .+ 2
		
		E = eltype(u0)
		extended = zeros(E, sz)
		result = zeros(auxiliar_caches(E, op) , sz)
	
		for bc in box.bcs
			initialize(op::BackwardDifference{N, T, Tag}, bc, extended)
		end
		
		return extended, result
	
	end

	function build_caches(op::CenteredDifference{N, T, Hessian}, box::LiquidCrystals.BoxBC, u0) where {N, T} #do we need op?, yes we do! 
		sz = size(u0) .+ 2
		
		E = eltype(u0)
		extended = zeros(E, sz)
		result = zeros(auxiliar_caches(E, op), sz)
	
		for bc in box.bcs
			initialize(op::CenteredDifference{N, T, Hessian}, bc, extended)
		end
		
		return extended, result
	
	end
		
end

function initialize(op, bc::LiquidCrystals.AxisBC, extended)

	return extended

end

struct AnchoringAxisBC{A <: LiquidCrystals.Axis} <: LiquidCrystals.AxisBC	

	function AnchoringAxisBC{A}() where {A <: LiquidCrystals.ZAxis}
		return new{A}()
	end

	function AnchoringAxisBC{A}() where {A <: LiquidCrystals.YAxis}
		return new{A}()
	end

	function AnchoringAxisBC{A}() where {A <: LiquidCrystals.XAxis}
		return new{A}()
	end
end


struct DirichletAxisBC{A <: LiquidCrystals.Axis, T} <: LiquidCrystals.AxisBC
	Q_l::T
	Q_h::T

	function DirichletAxisBC{A}(Q_l::T, Q_h::T) where {A <: LiquidCrystals.Axis, T}
		return new{A, T}(Q_l, Q_h)
	end
end

begin 
	function initialize(op, bc::DirichletAxisBC{A}, extended) where {A <: LiquidCrystals.Axis}

		sz = size(extended)
		
		CI_lb, CI_hb = surface_ranges_extended(sz, A) 
	
		extended[CI_lb] .= fill(bc.Q_h, size(CI_lb)) 
		extended[CI_hb] .= fill(bc.Q_l, size(CI_hb)) 
		
		return extended
	end


	function initialize(op, bc::DirichletAxisBC{A}, extended::Array{SVector{3, Float64}, 3}) where {A <: LiquidCrystals.Axis}

		sz = size(extended)
		
		CI_lb, CI_hb = surface_ranges_extended(sz, A) 
	
		extended[CI_lb] .= fill(SVector(0.0,0.0,0.0), size(CI_lb)) 
		extended[CI_hb] .= fill(SVector(0.0,0.0,0.0), size(CI_hb)) 
		
		return extended
	end

end

begin

	function ghost_ranges(sz::NTuple{3},::Type{LiquidCrystals.YAxis})
		m, n, p = sz
		return CartesianIndices((2:m-1, 1:1, 2:p-1)), CartesianIndices((2:m-1, n:n, 2:p-1))
	end
		
	function ghost_ranges(sz::NTuple{3},::Type{LiquidCrystals.XAxis})
		m, n, p = sz
		return CartesianIndices((m:m, 2:n-1, 2:p-1)), CartesianIndices((1:1, 2:n-1, 2:p-1))
	end
		
	function ghost_ranges(sz::NTuple{3},::Type{LiquidCrystals.ZAxis})
		m, n, p = sz
		return CartesianIndices((2:m-1, 2:n-1, p:p)), CartesianIndices((2:m-1, 2:n-1, 1:1))
	end
		
	function boundary_ranges(sz::NTuple{3}, ::Type{LiquidCrystals.YAxis})
		m, n, p = sz
		
		#extract two as it is the extended matrix size
		lo = CartesianIndices((1:m-2, 1:1, 1:p-2))    
		hi = CartesianIndices((1:m-2, n-2:n-2, 1:p-2))
		
		return lo, hi
	end
		
	function boundary_ranges(sz::NTuple{3}, ::Type{LiquidCrystals.XAxis})
		m, n, p = sz
		
		#extract two as it is the extended matrix size
		lo = CartesianIndices((m-2:m-2, 1:n-2, 1:p-2))
		hi = CartesianIndices((1:1, 1:n-2, 1:p-2))
		
		return lo, hi
	end
		
		
	function boundary_ranges(sz::NTuple{3}, ::Type{LiquidCrystals.ZAxis})
		m, n, p = sz
		
		#extract two as it is the extended matrix size
		lo = CartesianIndices((1:m-2, 1:n-2, p-2:p-2))
		hi = CartesianIndices((1:m-2, 1:n-2, 1:1))
		
		return lo, hi
	end

	function surface_ranges_extended(sz::NTuple{3}, ::Type{LiquidCrystals.YAxis})
		m, n, p = sz
		
		#extract two as it is the extended matrix size
		lo = CartesianIndices((2:m-1, 2:2, 2:p-1))    
		hi = CartesianIndices((2:m-1, n-1:n-1, 2:p-1))
		
		return lo, hi
	end
		
	function surface_ranges_extended(sz::NTuple{3}, ::Type{LiquidCrystals.XAxis})
		m, n, p = sz
			
		#extract two as it is the extended matrix size
		lo = CartesianIndices((m-1:m-1, 2:n-1, 2:p-1))
		hi = CartesianIndices((2:2, 2:n-1, 2:p-1))
		
		return lo, hi
	end
		
		
	function surface_ranges_extended(sz::NTuple{3}, ::Type{LiquidCrystals.ZAxis})
		m, n, p = sz
		
		#extract two as it is the extended matrix size
		lo = CartesianIndices((2:m-1, 2:n-1, p-1:p-1))
		hi = CartesianIndices((2:m-1, 2:n-1, 2:2))
		
		return lo, hi
	end

end

begin

	function apply_BCs(op, u, bc::LiquidCrystals.PeriodicAxisBC{A}, extended) where {A <: LiquidCrystals.Axis}
	
		sz = size(extended)
		
		CI_lg, CI_hg = ghost_ranges(sz, A) 
		CI_lb, CI_hb = boundary_ranges(sz, A)

		extended[CI_lg] .= u[CI_hb]
		extended[CI_hg] .= u[CI_lb]  
	
		return extended
	
	end

	function apply_BCs(op, u, bc::AnchoringAxisBC{A}, extended) where {A <: LiquidCrystals.Axis}

		sz = size(extended)

		CI_lg, CI_hg = ghost_ranges(sz, A) 
		CI_lb, CI_hb = boundary_ranges(sz, A)

		extended[CI_lg] .= u[CI_lb]
		extended[CI_hg] .= u[CI_hb]
		
		return extended
	end

	function apply_BCs(op, u, bc::DirichletAxisBC{A}, extended) where {A <: LiquidCrystals.Axis}

		sz = size(extended)
	
		CI_lb, CI_hb = surface_ranges_extended(sz, A)
		
		extended[CI_lb] .= fill(bc.Q_h, size(CI_lb)) 
		extended[CI_hb] .= fill(bc.Q_l, size(CI_hb)) 
			
		return extended
	end
		
	function apply_BCs(op, u, box::LiquidCrystals.BoxBC, extended)
	
		sz = size(extended) .- 1
		CI = CartesianIndices(UnitRange.(2, sz))
		extended[CI] .= u
	
		for bc in box.bcs
			apply_BCs(op, u, bc, extended)
		end
	
		return extended
	end

end





end

# function LinearAlgebra.mul!(B::AbstractArray, Op::CenteredDifference{N, T, Hessian}, A::AbstractArray, bulkCIs::Vector{CartesianIndex{3}}) where {N, T}

# 	op = concretize(Hessian)
# 	coeffs = Op.coeffs
#     offsets = Op.offsets
			
# 	for I in bulkCIs
# 		B[I] = op(A, I, coeffs, offsets)
# 	end
# 	# B[CI] .= op.((A,), CI, (coeffs,))
		
# 	return B
# end

# function LinearAlgebra.mul!(B::AbstractArray, Op::CenteredDifference{N, T, Tag}, A::AbstractArray, bulkCIs::Vector{CartesianIndex{3}}) where {N, T, Tag}

# 	op = concretize(Tag)
# 	coeffs = Op.coeffs
#     offsets = Op.offsets
			
# 	for I in bulkCIs
# 		B[I] = op(A, I, coeffs, offsets)
# 	end
# 	# B[CI] .= op.((A,), CI, (coeffs,))
		
# 	return B
# end


# function LinearAlgebra.mul!(
# 	B::AbstractArray,
# 	Op::Union{CenteredDifference{N, T, Tag}, UpwindDifference{N, T, Tag}},
# 	A::AbstractArray,
# 	indices::Union{Vector{CartesianIndex{3}}, Vector{SurfacePoint{3, Float64}}}
# ) where {N, T, Tag}
	    
# 	if indices isa Vector{CartesianIndex{3}}
	        
# 	    @assert Op isa CenteredDifference "For CartesianIndex{3} indices, Op must be a CenteredDifference."
# 	    op = concretize(Tag)
# 	    coeffs = Op.coeffs
#         offsets = Op.offsets
	        
# 	    for I in indices
# 	        B[I] = op(A, I, coeffs, offsets)
# 	    end
	        
# 	elseif indices isa Vector{SurfacePoint{3, Float64}}
	        
# 	    @assert Op isa UpwindDifference{N, T, Tag} "For SurfacePoint{3, Float64} indices, Op must be a ForwardDifference or BackwardDifference."
	        
# 	    function compute_gradient_component(A, I, normal_component, forward_coeff, backward_coeff, Ij)
# 	        Cj = normal_component > 0 ? forward_coeff :
# 	            (normal_component < 0 ? backward_coeff : SVector(0.0, -0.5, 0.0, 0.5, 0.0))
#             Oj = 
# 	        return derivative(A, I, Ij, Cj, Oj)
# 	    end
	
# 	    Iz = CartesianIndex(0, 0, 1)
# 	    Iy = CartesianIndex(0, 1, 0)
# 	    Ix = CartesianIndex(1, 0, 0)
	
# 	    for point in indices
# 	        ∇_x = compute_gradient_component(A, point.index, point.normal[1], Op.FD_coeffs[1], Op.BD_coeffs[1], Ix)
# 	        ∇_y = compute_gradient_component(A, point.index, point.normal[2], Op.FD_coeffs[2], Op.BD_coeffs[2], Iy)
# 	        ∇_z = compute_gradient_component(A, point.index, point.normal[3], Op.FD_coeffs[3], Op.BD_coeffs[3], Iz)
	            
# 	        B[point.index] = SVector(∇_x, ∇_y, ∇_z)
# 	    end
# 	else
# 	    error("Unsupported type for indices: $(typeof(indices))")
# 	end
	
# 	return B
# end