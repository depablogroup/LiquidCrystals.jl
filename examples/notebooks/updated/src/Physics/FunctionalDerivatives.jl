#src/Physics/FunctionalDerivatives.jl
module FunctionalDerivatives

using StaticArrays
using TensorOperations
using LinearAlgebra

using ..Utils
using LiquidCrystals

export compute_δfLdG_δQ_bulk, compute_δfelas_δQ_bulk, compute_δfdiel_δQ_bulk, compute_δflex_δQ_bulk
export compute_νp_∂felas_∂∇Q, compute_νp_∂flex_∂∇Q, compute_∂fRP_∂Q

q_eq = q_steady_state[]

#-----------------------------Landau de Gennes Free Energy-------------------------------------------------------#
"""
    compute_δfLdG_δQ_bulk(Q, constants, bulkCIs)

Compute Landau-de Gennes free energy derivative δf/δQ for bulk liquid crystal points.

# Arguments
- `Q`: Array of Q-tensors (QLocal or SVector)
- `constants`: Must contain A₀ (coefficient) and U (reduced temperature)
- `bulkCIs`: Cartesian indices of bulk points to compute

# Math
δf/δQ_{mn} = A₀(1-U/3)Q_{mn} - A₀U(Q²_{mn} - ¹/₃Tr(Q²)δ_{mn}) + A₀UQ³_{mn}

# Returns
Array of δfLdG/δQ values at each bulk point.
"""
function compute_δfLdG_δQ_bulk(Q::AbstractArray{T}, constants::NamedTuple, bulkCIs::Vector{CartesianIndex{3}}) where {T <: LiquidCrystals.QLocal}
		
	δfLdG_δQ_grid = fill(LiquidCrystals.QLocal(SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)), size(Q))

	A₀ = constants.A₀
	U = constants.U

	dim = 3
		
	@inbounds for ci in bulkCIs
		Qᵢ = Q[ci]
		Qᵢ² = Qᵢ * Qᵢ
		@tensor trQᵢ² = Qᵢ²[k,k] 
		Qᵢ³ = trQᵢ² * Qᵢ

		δfLdG_δQᵢ = A₀ .* (1 - (U/3)) .* Qᵢ .- A₀ .* U .* (Qᵢ² .- (1/dim) .* trQᵢ² .* I(dim)) .+ A₀ .* U .* Qᵢ³
			
	    δfLdG_δQ_grid[ci] = LiquidCrystals.QLocal(δfLdG_δQᵢ)
	end
	
	return δfLdG_δQ_grid
end

function compute_δfLdG_δQ_bulk(Q::AbstractArray{T}, constants::NamedTuple, bulkCIs::Vector{CartesianIndex{3}}) where {T <: SVector}
    return compute_δfLdG_δQ_bulk(reinterpret(LiquidCrystals.qtype(T), Q), constants, bulkCIs)
end

#-------------------------------Elastic Free Energy-------------------------------------------------------------#

"""
    compute_δfelas_δQ_bulk(Q, ∇Q, ∇∇Q, constants, bulkCIs)

Compute elastic free energy derivative δf/δQ using full gradient terms.

# Arguments
- `Q`: Q-tensor field
- `∇Q`, `∇∇Q`: First and second derivatives of Q
- `constants`: Must contain L₁, L₂, L₃, L₄ elastic constants
- `bulkCIs`: Bulk point indices

# Math
δf/δQ_{mn} = -L₁∂ₖ∂ₖQ_{mn} - L₂∂ₙ∂ₖQ_{mk} 
            + L₃(¹/₂(∂ₖQ_{ml})(∂ₖQ_{nl}) - ∂ₗQ_{kl}∂ₖQ_{mn} - Q_{kl}∂ₗ∂ₖQ_{mn})
            - L₄∂ₖ∂ₙQ_{mk}

# Returns
Array of δfElas/δQ values at each bulk point.
"""
function compute_δfelas_δQ_bulk(Q::AbstractArray{T},
                           	   ∇Q::SubArray{<:SVector{3, <:SVector{6, Float64}}, 3},
                           	   ∇∇Q::SubArray{<:SMatrix{3, 3, <:SVector{6, Float64}}, 3},
                           	   constants::NamedTuple, bulkCIs::Vector{CartesianIndex{3}}) where {T <: LiquidCrystals.QLocal}

	δfelas_δQ_grid = fill(LiquidCrystals.QLocal(SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)), size(Q))

	L₁ = constants.L₁
	L₂ = constants.L₂
	L₃ = constants.L₃
	L₄ = constants.L₄

	dim = 3
		
	@inbounds for ci in bulkCIs

		Qᵢ = Q[ci] 
		∇Qᵢ = reshape_and_reinterpret(∇Q[ci])
		∇∇Qᵢ = reshape_and_reinterpret(∇∇Q[ci])

		@tensor δfelas_δQmn_t1[m,n] := -L₁ * ∇∇Qᵢ[k,k,m,n]
		@tensor δfelas_δQmn_t2[m,n] := -L₂ * ∇∇Qᵢ[n,k,m,k]
		@tensor δfelas_δQmn_t3[m,n] := L₃ * (0.5 * ∇Qᵢ[m,k,l] * ∇Qᵢ[n,k,l] - ∇Qᵢ[l,k,l] * ∇Qᵢ[k,m,n] - ∇∇Qᵢ[l,k,m,n] * Qᵢ[k,l]) 
		@tensor δfelas_δQmn_t4[m,n] := -L₄ * ∇∇Qᵢ[k,n,m,k]

		δfelas_δQmn = δfelas_δQmn_t1 + δfelas_δQmn_t2 + δfelas_δQmn_t3 + δfelas_δQmn_t4

		@tensor δfelas_δQll = δfelas_δQmn[k,k]

		δfelas_δQ_grid[ci] = convert_to_QLocal(δfelas_δQmn .- (1/dim) .* δfelas_δQll .* I(dim))
	end

	return δfelas_δQ_grid
end

"""
    compute_δfelas_δQ_bulk(Q, ΔQ, constants, bulkCIs)

Compute elastic free energy derivative δf/δQ for one-constant approximation

# Arguments
- `Q`: Q-tensor field
- `ΔQ`: Laplacian of Q
- `constants`: Must contain L elastic constant
- `bulkCIs`: Bulk point indices

# Math
δf/δQ_{mn} = -L₁∂ₖ∂ₖQ_{mn}

# Returns
Array of δfElas/δQ values at each bulk point.
"""
function compute_δfelas_δQ_bulk(Q::AbstractArray{T},
                           	   ΔQ::SubArray{<:SVector{6, Float64}, 3},
                           	   constants::NamedTuple, bulkCIs::Vector{CartesianIndex{3}}) where {T <: LiquidCrystals.QLocal}

	δfelas_δQ_grid = fill(SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), size(Q))

	L = constants.L

	dim = 3
		
	@inbounds for ci in bulkCIs

		Qᵢ = Q[ci] 
		ΔQᵢ = ΔQ[ci]

		δfelas_δQ_grid[ci] = -L .* ΔQᵢ
	end

	return δfelas_δQ_grid
end

"""
    compute_νp_∂felas_∂∇Q(Q, ∇Q, constants, surfaceCIs)

Compute elastic surface term ν·(∂f/∂∇Q) for boundary conditions. 

# Arguments
- `Q`: Q-tensor field
- `∇Q`: Gradient of Q
- `constants`: Elastic constants (either L₁-L₄ or just L)
- `surfaceCIs`: Surface point indices with normals

# Returns
Array of δfElas/δQ values at each surface point
"""
function compute_νp_∂felas_∂∇Q(Q::AbstractArray{T},
                           	   ∇Q::SubArray{<:SVector{3, <:SVector{6, Float64}}, 3},
                           	   constants::NamedTuple, surfaceCIs::Vector{<:SurfacePoint}) where {T <: LiquidCrystals.QLocal}
	
	νp_∂felas_∂∇Q_grid = fill(LiquidCrystals.QLocal(SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)), size(Q))

	
	dim = 3
		
	try 
		L₁ = constants.L₁
		L₂ = constants.L₂
		L₃ = constants.L₃
		L₄ = constants.L₄

		@inbounds for ci in surfaceCIs

			Qᵢ = Q[ci.index] 
			∇Qᵢ = reshape_and_reinterpret(∇Q[ci.index])
	
			@tensor ∂felas_∂∇Q_t1[m,n,p] := L₁ * ∇Qᵢ[p,m,n]
			@tensor ∂felas_∂∇Q_t2[m,n,p] := L₂ * ∇Qᵢ[k,m,k] * I(dim)[n,p]
			@tensor ∂felas_∂∇Q_t3[m,n,p] := L₃ * ∇Qᵢ[k,m,n] * Qᵢ[k,p]
			@tensor ∂felas_∂∇Q_t4[m,n,p] := L₄ * ∇Qᵢ[n,m,p] 
				
			∂felas_∂∇Q = ∂felas_∂∇Q_t1 + ∂felas_∂∇Q_t2 + ∂felas_∂∇Q_t3 + ∂felas_∂∇Q_t4
	
			@tensor νp_∂felas_∂∇Q[m,n] := ∂felas_∂∇Q[m,n,p] * ci.normal[p]
			@tensor trace = νp_∂felas_∂∇Q[k,k]
	
			νp_∂felas_∂∇Q_grid[ci.index] = convert_to_QLocal(νp_∂felas_∂∇Q .- (1/dim) .* trace .* I(dim))
		end
	catch e
		L = constants.L

		@inbounds for ci in surfaceCIs

			Qᵢ = Q[ci.index] 
			∇Qᵢ = reshape_and_reinterpret(∇Q[ci.index])
	
			@tensor ∂felas_∂∇Q[m,n,p] := L * ∇Qᵢ[p,m,n]
			@tensor νp_∂felas_∂∇Q[m,n] := ∂felas_∂∇Q[m,n,p] * ci.normal[p]
			@tensor trace = νp_∂felas_∂∇Q[k,k]
	
			νp_∂felas_∂∇Q_grid[ci.index] = convert_to_QLocal(νp_∂felas_∂∇Q .- (1/dim) .* trace .* I(dim))
		end
	end
		
	return νp_∂felas_∂∇Q_grid
end

function compute_δfelas_δQ_bulk(Q::AbstractArray{T},
                           	   ∇Q::SubArray{<:SVector{3, <:SVector{6, Float64}}, 3},
                           	   ∇∇Q::SubArray{<:SMatrix{3, 3, <:SVector{6, Float64}}, 3},
                           	   constants::NamedTuple, bulkCIs::Vector{CartesianIndex{3}}) where {T <: SVector}
	return compute_δfelas_δQ_bulk(reinterpret(LiquidCrystals.qtype(T), Q), ∇Q, ∇∇Q, constants, bulkCIs)
end

function compute_δfelas_δQ_bulk(Q::AbstractArray{T},
                           	   ΔQ::SubArray{<:SVector{6, Float64}, 3},
                           	   constants::NamedTuple, bulkCIs::Vector{CartesianIndex{3}}) where {T <: SVector}
	return compute_δfelas_δQ_bulk(reinterpret(LiquidCrystals.qtype(T), Q), ΔQ, constants, bulkCIs)
end

function compute_νp_∂felas_∂∇Q(Q::AbstractArray{T},
                           	   ∇Q::SubArray{<:SVector{3, <:SVector{6, Float64}}, 3},
                           	   constants::NamedTuple, surfaceCIs::Vector{<:SurfacePoint}) where {T <: SVector}
	return compute_νp_∂felas_∂∇Q(reinterpret(LiquidCrystals.qtype(T), Q), ∇Q, constants, surfaceCIs)
end

#-------------------------------------------------Dielectric Free Energy-----------------------------------------------------#

"""
    compute_δfdiel_δQ_bulk(Q, E, constants, bulkCIs)

Compute dielectric free energy derivative δf/δQ from electric field coupling.

# Arguments
- Q::Array{QLocal} : Order parameter
- E::Array{SVector{3}} : Electric field
- constants::NamedTuple : Must contain ϵₐ (Dielectric Anisotropy)
- bulkCIs::Vector{CartesianIndex} : Bulk points

# Math
δf/δQ_{mn} = -¹/₂ϵₐEₘEₙ

# Returns
Array of δfDiel/δQ values at each bulk point, traceless (QLocal format)
"""	
function compute_δfdiel_δQ_bulk(Q::AbstractArray{T}, E::Array{SVector{3, Float64}}, constants::NamedTuple, bulkCIs::Vector{CartesianIndex{3}})  where {T <: LiquidCrystals.QLocal}
	
	ϵₐ = constants.ϵₐ
	
	δfdiel_δQ_grid = fill(LiquidCrystals.QLocal(SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)), size(Q))

	dim = 3

	@inbounds for ci in bulkCIs

		Eᵢ = E[ci]

		@tensor δfdiel_δQmn[m,n] := -0.5 * ϵₐ * Eᵢ[m] * Eᵢ[n]
		@tensor trace = δfdiel_δQmn[k,k]

		δfdiel_δQ_grid[ci] = convert_to_QLocal(δfdiel_δQmn .- (1/dim) .* trace .* I(dim))
	end
		
	return δfdiel_δQ_grid
end

function compute_δfdiel_δQ_bulk(Q::AbstractArray{T}, E::Array{SVector{3, Float64}}, constants::NamedTuple, bulkCIs::Vector{CartesianIndex{3}})  where {T <: SVector}
	compute_δfdiel_δQ_bulk(reinterpret(LiquidCrystals.qtype(T), Q), E, constants, bulkCIs)
end


#----------------------------------Flexoelectric Free Energy-------------------------------------------------------------------#


"""
    compute_δflex_δQ_bulk(Q, ∇Q, E, ∇E, constants, bulkCIs)

Compute flexoelectric free energy derivative δf/δQ for bulk points.

# Arguments
- `Q::Array{QLocal}`: Order parameter
- `∇Q::Array{SVector{3,SVector{6}}}`: Q gradients
- `E::Array{SVector{3}}`: Electric field
- `∇E::Array{SVector{3,SVector{3}}}`: E gradients
- `constants::NamedTuple`: Must contain ζ₁,ζ₂
- `bulkCIs::Vector{CartesianIndex}`: Bulk points

# Math
δf/δQ_{mn} = -ζ₁∂ₙEₘ + ζ₂(Eₘ∂ₖQₙₖ - Eₖ∂ₙQₖₘ - Qₖₘ∂ₙEₖ)

# Return
Array{QLocal} : δfFlex/δQ at bulk points
"""
function compute_δflex_δQ_bulk(Q::AbstractArray{T}, ∇Q::SubArray{<:SVector{3, <:SVector{6, Float64}}, 3},
							  E::Array{SVector{3, Float64}}, ∇E::SubArray{<:SVector{3, <:SVector{3, Float64}}, 3},
							  constants::NamedTuple, bulkCIs::Vector{CartesianIndex{3}}) where {T <: LiquidCrystals.QLocal}

	# params
	ζ₁ = constants.ζ₁
	ζ₂ = constants.ζ₂

	δflex_δQ_grid = fill(LiquidCrystals.QLocal(SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)), size(Q))

	dim = 3 

	@inbounds for ci in bulkCIs

		Qᵢ = Q[ci] 
		∇Qᵢ = reshape_and_reinterpret(∇Q[ci])
		Eᵢ = E[ci]
		∇Eᵢ = reshape_and_reinterpret(∇E[ci])

		@tensor δflex_δQmn_t1[m,n] := -ζ₁ * ∇Eᵢ[n,m]
		@tensor δflex_δQmn_t2[m,n] := ζ₂ * (Eᵢ[m] * ∇Qᵢ[k,n,k] - Eᵢ[k] * ∇Qᵢ[n,k,m] - Qᵢ[k,m] * ∇Eᵢ[n,k])

		δflex_δQmn = δflex_δQmn_t1 + δflex_δQmn_t2

		@tensor trace = δflex_δQmn[k,k]

		δflex_δQ_grid[ci] = convert_to_QLocal(δflex_δQmn .- (1/dim) .* trace .* I(dim))
	end
		
	return δflex_δQ_grid

end


"""
    compute_νp_∂flex_∂∇Q(Q, ∇Q, E, constants, surfaceCIs)

Compute flexoelectric surface term ν·(∂f/∂∇Q) for boundary conditions.

# Arguments
- `Q::Array{QLocal}`: Order parameter field (rank-2 tensor components)
- `∇Q::Array{SVector{3,SVector{6}}}`: Gradient of Q-tensor field
- `E::Array{SVector{3}}`: Electric field vector
- `constants::NamedTuple`: Must contain:
    - `ζ₁`: flexoelectric coefficient: (e₁+e₃)/q²
    - `ζ₂`: flexoelectric coefficient: -(e₁+4e₃)/3q
	(where e₁ is the splay flexoelectric coefficient and e₃ is the bend flexoelectric coefficient)
- `surfaceCIs::Vector{SurfacePoint}`: Surface points containing:
    - `index`: CartesianIndex of grid point
    - `normal`: Surface normal vector

# Mathematics
Computes the surface term in Einstein notation:
νₖ(∂f/∂(∂ₖQₘₙ)) = ζ₁Eₘδₙₖνₖ + ζ₂EₗQₗₘδₙₖνₖ

with traceless projection:
ν·(∂f/∂∇Q)ₘₙ - ⅓Tr[ν·(∂f/∂∇Q)]δₘₙ

# Returns
- `Array{QLocal}`: Surface term values at each specified point,
  stored as traceless Q-tensors (SVector{6,Float64} format)
"""
function compute_νp_∂flex_∂∇Q(Q::AbstractArray{T}, ∇Q::SubArray{<:SVector{3, <:SVector{6, Float64}}, 3},
							  E::Array{SVector{3, Float64}},
							  constants::NamedTuple, surfaceCIs::Vector{<:SurfacePoint}) where {T <: LiquidCrystals.QLocal}
	# params
	ζ₁ = constants.ζ₁
	ζ₂ = constants.ζ₂

	νp_∂flex_∂∇Q_grid = fill(LiquidCrystals.QLocal(SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)), size(Q))

	dim = 3

	@inbounds for ci in surfaceCIs
		Qᵢ = Q[ci.index] 
		∇Qᵢ = reshape_and_reinterpret(∇Q[ci.index])
		Eᵢ = E[ci.index]

		@tensor ∂flex_∂∇Q_t1[m,n,p] := ζ₁ * Eᵢ[m] * I(dim)[n,p]
		@tensor ∂flex_∂∇Q_t2[m,n,p] := ζ₂ * Eᵢ[k] * Qᵢ[k,m] * I(dim)[n,p]

		∂flex_∂∇Q = ∂flex_∂∇Q_t1 + ∂flex_∂∇Q_t2

		@tensor νp_∂flex_∂∇Q[m,n] := ∂flex_∂∇Q[m,n,p] * ci.normal[p]
		@tensor trace = νp_∂flex_∂∇Q[k,k]

		νp_∂flex_∂∇Q_grid[ci.index] = convert_to_QLocal(νp_∂flex_∂∇Q .- (1/dim) .* trace .* I(dim))
	end

	return νp_∂flex_∂∇Q_grid
end

function compute_δflex_δQ_bulk(Q::AbstractArray{T}, ∇Q::SubArray{<:SVector{3, <:SVector{6, Float64}}, 3},
							  E::Array{SVector{3, Float64}}, ∇E::SubArray{<:SVector{3, <:SVector{3, Float64}}, 3},
							  constants::NamedTuple, bulkCIs::Vector{CartesianIndex{3}}) where {T <: SVector}
	compute_δflex_δQ_bulk(reinterpret(LiquidCrystals.qtype(T), Q), ∇Q, E, ∇E, constants, bulkCIs)
end

function compute_νp_∂flex_∂∇Q(Q::AbstractArray{T}, ∇Q::SubArray{<:SVector{3, <:SVector{6, Float64}}, 3},
							  E::Array{SVector{3, Float64}},
							  constants::NamedTuple, surfaceCIs::Vector{<:SurfacePoint}) where {T <: SVector}
	compute_νp_∂flex_∂∇Q(reinterpret(LiquidCrystals.qtype(T), Q), ∇Q, E, constants, surfaceCIs)
end

#-----------------------------------------Rapini-Papoular---------------------------------------------------------------#

"""
    compute_∂fRP_∂Q(Q, surfaceCIs, params)

Compute the Rapini-Popoular surface anchoring derivative ∂f/∂Q for colloid/liquid crystal interfaces.

# Arguments
- `Q::Array{QLocal}`: Bulk Q-tensor field
- `surfaceCIs::Vector{SurfacePoint}`: Surface points with:
  - `.index`: CartesianIndex of grid point
  - `.normal`: Surface normal vector
- `params::NamedTuple`: Must contain either:
  - `anchoring.strength` (scalar) and `anchoring.direction` (vector) OR
  - Direct `strength` and `direction` overrides

# Physics (Rapini-Popoular Formulation)
Computes the surface anchoring energy gradient:
∂f/∂Q = W * (Q - Q₀)
where:
- W: Anchoring strength (params.strength)
- Q₀: Preferred surface-aligned Q-tensor:
  Q₀ = q_eq * (n⊗n - ¹/₃I) 
  with n = params.direction or surface normal

# Returns
- `Array{QLocal}`: ∂fRP/∂Q at surface points (zero in bulk)
"""
function compute_∂fRP_∂Q(Q::AbstractArray{T}, surfaceCIs::Vector{<:SurfacePoint}, params::NamedTuple) where {T <: LiquidCrystals.QLocal}
	
	∂fParticleBC_∂Q_grid = fill(LiquidCrystals.QLocal(SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)), size(Q))

	strength = get(params, :strength, nothing)
	if strength==nothing
		strength = params.anchoring.strength
	end
		
	dim = 3
	
	for ci in surfaceCIs
		index = SVector(Tuple(ci.index)...) 
		n = get(params, :direction, nothing) == nothing ? ci.normal : params.direction
		nn = n * n'
		@tensor trace = nn[k,k]
	
		Q0 = LiquidCrystals.QLocal(q_eq .* (nn - (1/dim) .* trace .* I(dim)))

		∂fParticleBC_∂Q_grid[ci.index] = LiquidCrystals.QLocal(strength .* (Q[ci.index] .- Q0))		# Rapini-Popoular type
	end
		
	return ∂fParticleBC_∂Q_grid
end
	
function compute_∂fRP_∂Q(Q::AbstractArray{T}, surfaceCIs::Vector{<:SurfacePoint}, params::NamedTuple) where {T <: SVector}
	compute_∂fRP_∂Q(reinterpret(LiquidCrystals.qtype(T), Q), surfaceCIs, params)
end

end


# begin

# 	function fournier_galatola_energy(Q::AbstractArray{T}, box::LiquidCrystals.BoxBC, CIs::Vector{CartesianIndex{3}}) where {T <: LiquidCrystals.QLocal}
	
# 		energy_bc = 0.0 
# 		for bc in box.bcs
# 			if bc isa PlanarAxisBC
# 				energy_bc += fournier_galatola_energy(Q, bc, CIs)
# 			end
# 		end

# 		return energy_bc
# 	end
	
# 	function fournier_galatola_energy(Q::AbstractArray{T}, bc::PlanarAxisBC{A}, CIs::Vector{CartesianIndex{3}}) where {T <: LiquidCrystals.QLocal, A <: LiquidCrystals.Axis}

# 	    dim = 3
# 		energy = 0.0
		
# 		for idx in CIs
# 			Qᵢ = Q[idx]
# 			proj = I(dim) - (bc.n * bc.n')
# 			Q̄ᵢ = Qᵢ + (1/dim) .* bc.S_eq .* I(dim)
# 			@tensor Q̄ᵢ_perp[m,n] := proj[m,k] * Q̄ᵢ[k,l] * proj[l,n]
# 			@tensor t1 = 0.5 * bc.W_l * (Q̄ᵢ[k,l] - Q̄ᵢ_perp[k,l]) * (Q̄ᵢ[l,k] - Q̄ᵢ_perp[l,k])
# 			@tensor t2 = Q̄ᵢ[k,l] * Q̄ᵢ[k,l]

# 			energy += t1 + 0.25 * bc.W_l * (t2 - bc.S_eq^2)^2		
# 		end

# 		return energy
# 	end

# 	function fournier_galatola_energy(Q::AbstractArray{T}, box::LiquidCrystals.BoxBC, CIs::Vector{CartesianIndex{3}}) where {T <: SVector}
# 		fournier_galatola_energy(reinterpret(LiquidCrystals.qtype(T), Q), box, CIs)
# 	end

# end

# # ╔═╡ e9e92f3b-f464-4389-b094-c225dea7ca45
# md"""
# $$\frac{\delta f_{\text{FG}}}{\delta Q_{mn}}=W\left(\bar{Q}_{kl}-\bar{Q}_{\perp,kl}\right)\left(\delta_{km}\delta_{ln}-p_{km}p_{ln}\right)+W\left(\bar{Q}_{kl}\bar{Q}_{kl}-S_{\text{eq}}^2\right)\bar{Q}_{mn}$$
# """

# # ╔═╡ 3f0d6aae-6ece-4956-861b-44fdbc1fcc9b
# begin

# 	function compute_δfFG_δQ(Q::AbstractArray{T}, bc::PlanarAxisBC{A}, CIs::Vector{CartesianIndex{3}}) where {T <: LiquidCrystals.QLocal, A <: LiquidCrystals.Axis}

# 		dim = 3

# 		δfFG_δQ_grid = fill(LiquidCrystals.QLocal(SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)), size(Q))

# 		for idx in CIs
# 			Qᵢ = Q[idx]
# 			proj = I(dim) - (bc.n * bc.n')
# 			Q̄ᵢ = Qᵢ + (1/dim) .* bc.S_eq .* I(dim)
# 			@tensor Q̄ᵢ_perp[m,n] := proj[m,k] * Q̄ᵢ[k,l] * proj[l,n]
# 			@tensor t1[m,n] := 0.5 * bc.W_l * (Q̄ᵢ[k,l] - Q̄ᵢ_perp[k,l]) * (I(dim)[l,m] * I(dim)[k,n] - proj[l,m] * proj[k,n])
# 			@tensor t2[m,n] := 0.5 * bc.W_l * (Q̄ᵢ[l,k] - Q̄ᵢ_perp[l,k]) * (I(dim)[k,m] * I(dim)[l,n] - proj[k,m] * proj[l,n])
# 			@tensor t3 = Q̄ᵢ[k,l] * Q̄ᵢ[k,l]
			
# 			δfFG_δQmn = t1 + t2 + bc.W_l .* (t3 - bc.S_eq^2) .* Q̄ᵢ
# 			@tensor δfFG_δQll = δfFG_δQmn[l,l]
# 			δfFG_δQmn = δfFG_δQmn - (1/dim) .* δfFG_δQll .* I(dim)

# 			δfFG_δQ_grid[idx] = LiquidCrystals.QLocal(δfFG_δQmn)
# 		end

# 		return δfFG_δQ_grid
# 	end

# 	function compute_δfFG_δQ(Q::AbstractArray{T}, bc::PlanarAxisBC{A}, CIs::Vector{CartesianIndex{3}}) where {T <: SVector, A <: LiquidCrystals.Axis}
# 		compute_δfFG_δQ(reinterpret(LiquidCrystals.qtype(T), Q), bc, CIs)
# 	end
	
# end