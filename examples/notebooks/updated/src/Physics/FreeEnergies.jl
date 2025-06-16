#src/Physics/FreeEnergies.jl
module FreeEnergies

using StaticArrays
using LiquidCrystals
using TensorOperations
using ..Utils

export compute_FRP_heatmap, compute_FLdG_heatmap

#----------------------------------------------Rapini-Papoular------------------------------------------------------------------#
function compute_FRP_heatmap(Q::AbstractArray{T}, params::NamedTuple, dS::Real, CIs::Vector{<:SurfacePoint}) where {T <: LiquidCrystals.QLocal}

	strength = get(params, :strength, nothing)
	if strength==nothing
		strength = params.anchoring.strength
	end

	FRP_heatmap = fill(0.0, size(Q))

	dim = 3
	@inbounds for ci in CIs

		Qᵢ = Q[ci.index]

		n = get(params, :direction, nothing) == nothing ? ci.normal : params.direction
		nn = n * n'
		@tensor trace = nn[k,k]
	
		Q0 = LiquidCrystals.QLocal(q_eq .* (nn - (1/dim) .* trace .* I(dim)))

		@tensor e = 0.5 * strength * (Qᵢ[i,j] - Q0[i,j]) * (Qᵢ[j,i] - Q0[j,i]) 
		FRP_heatmap[ci] = e
	end

	return FRP_heatmap .* dS
end

function compute_FRP_heatmap(Q::AbstractArray{T}, params::NamedTuple, dS::Real, CIs::Vector{<:SurfacePoint}) where {T <: SVector}
	compute_FRP(reinterpret(LiquidCrystals.qtype(T), Q), params, dS, CIs)
end

function compute_FRP_average(Q::AbstractArray{T}, params::NamedTuple, dS::Real, CIs::Vector{<:SurfacePoint}) where {T <: LiquidCrystals.QLocal}

	strength = get(params, :strength, nothing)
	if strength==nothing
		strength = params.anchoring.strength
	end

	dim = 3
	energy = 0.0
	@inbounds for ci in CIs

		Qᵢ = Q[ci.index]

		n = get(params, :direction, nothing) == nothing ? ci.normal : params.direction
		nn = n * n'
		@tensor trace = nn[k,k]
	
		Q0 = LiquidCrystals.QLocal(q_eq .* (nn - (1/dim) .* trace .* I(dim)))

		@tensor e = 0.5 * strength * (Qᵢ[i,j] - Q0[i,j]) * (Qᵢ[j,i] - Q0[j,i]) 
		energy += e
	end

	return energy * dS
end

function compute_FRP_average(Q::AbstractArray{T}, params::NamedTuple, dS::Real, CIs::Vector{<:SurfacePoint}) where {T <: SVector}
	compute_FRP(reinterpret(LiquidCrystals.qtype(T), Q), params, dS, CIs)
end


#----------------------------------------Landau de Gennes Free Energy (Bulk)--------------------------------------------------------#
function compute_FLdG_heatmap(Q::AbstractArray{T}, model::PhysicsModel, CIs::Union{Vector{CartesianIndex{3}},Vector{<:SurfacePoint}}) where {T <: LiquidCrystals.QLocal}

	A₀ = model.parameters.LandaudeGennes.A₀
	U = model.parameters.LandaudeGennes.U

	dV = model.parameters.dx * model.parameters.dy * model.parameters.dz

	FLdG_heatmap = fill(0.0, size(Q))
	
	@inbounds for ci in CIs

		if ci isa CartesianIndex{3}
			index = ci  
		elseif ci isa SurfacePoint
			index = ci.index  
		else
			error("Unsupported type for ci: $(typeof(ci))")
		end
		
		Qᵢ = Q[index]
		Qᵢ² = Qᵢ * Qᵢ
		Qᵢ³ = Qᵢ² * Qᵢ
		@tensor trQ² = Qᵢ²[k,k] 
		@tensor trQ³ = Qᵢ³[k,k]

		FLdG_heatmap[ci] = A₀ * (1 - (U/3)) * 0.5 * trQ² - (A₀ * U/3) * trQ³ + (A₀ * U/4) * (trQ²)^2
	end
	
	return FLdG_heatmap .* dV
end

	
function compute_FLdG_heatmap(Q::AbstractArray{T}, model::PhysicsModel, CIs::Union{Vector{CartesianIndex{3}},Vector{<:SurfacePoint}}) where {T <: SVector}
	return compute_FLdG_heatmap(reinterpret(LiquidCrystals.qtype(T), Q), model, CIs)
end

function compute_FLdG_average(Q::AbstractArray{T}, constants::NamedTuple, dV::Real, CIs::Union{Vector{CartesianIndex{3}},Vector{<:SurfacePoint}}) where {T <: LiquidCrystals.QLocal}

	A₀ = constants.A₀
	U = constants.U
	
	energy = 0.0
	@inbounds for ci in CIs

		if ci isa CartesianIndex{3}
			index = ci  
		elseif ci isa SurfacePoint
			index = ci.index  
		else
			error("Unsupported type for ci: $(typeof(ci))")
		end
		
		Qᵢ = Q[index]
		Qᵢ² = Qᵢ * Qᵢ
		Qᵢ³ = Qᵢ² * Qᵢ
		@tensor trQ² = Qᵢ²[k,k] 
		@tensor trQ³ = Qᵢ³[k,k]

		energy += A₀ * (1 - (U/3)) * 0.5 * trQ² - (A₀ * U/3) * trQ³ + (A₀ * U/4) * (trQ²)^2
	end
	
	return energy * dV
end

	
function compute_FLdG_average(Q::AbstractArray{T}, constants::NamedTuple, dV::Real, CIs::Union{Vector{CartesianIndex{3}},Vector{<:SurfacePoint}}) where {T <: SVector}
	return compute_FLdG(reinterpret(LiquidCrystals.qtype(T), Q), constants, dV, CIs)
end


#------------------------------------------Elastic Free Energy---------------------------------------------------------------------------#
function compute_Felas(Q::AbstractArray{T},
					∇Q::AbstractArray{SVector{3, SVector{6, Float64}}, 3},
					constants::NamedTuple, dV::Real, CIs::Union{Vector{CartesianIndex{3}},Vector{<:SurfacePoint}}) where {T <: LiquidCrystals.QLocal}

	energy = 0.0
	if constants isa NamedTuple{(:L,)}  
        L = constants.L
			
		@inbounds for ci in CIs

			if ci isa CartesianIndex{3}
				index = ci  
			elseif ci isa SurfacePoint{3, Float64}
				index = ci.index  
			else
				error("Unsupported type for ci: $(typeof(ci))")
			end
				
			∇Qᵢ = reshape_and_reinterpret(∇Q[index])
	
			@tensor begin
		        energy += 0.5 * L * ∇Qᵢ[k,i,j] * ∇Qᵢ[k,i,j]
		    end
	
		end
    elseif constants isa NamedTuple{(:L₁, :L₂, :L₃, :L₄)}
		L₁ = constants.L₁
		L₂ = constants.L₂
		L₃ = constants.L₃
		L₄ = constants.L₄

		@inbounds for ci in CIs

			if ci isa CartesianIndex{3}
				index = ci  
			elseif ci isa SurfacePoint{3, Float64}
				index = ci.index  
			else
				error("Unsupported type for ci: $(typeof(ci))")
			end
				
			Qᵢ = Q[index] 
			∇Qᵢ = reshape_and_reinterpret(∇Q[index])
	
			@tensor begin
		        t1 = 0.5 * L₁ * ∇Qᵢ[k,i,j] * ∇Qᵢ[k,i,j]
		        t2 = 0.5 * L₂ * ∇Qᵢ[k,j,k] * ∇Qᵢ[l,j,l]
		        t3 = 0.5 * L₃ * Qᵢ[i,j] * ∇Qᵢ[k,i,l] * ∇Qᵢ[k,j,l]
		        t4 = 0.5 * L₄ * ∇Qᵢ[j,i,k] * ∇Qᵢ[i,j,k]
		        energy += t1 + t2 + t3 + t4
		    end
	
		end
	end

	return energy * dV
		
end


function compute_Felas(Q::AbstractArray{T},
						∇Q::AbstractArray{SVector{3, SVector{6, Float64}}, 3},
                        constants::NamedTuple, dV::Real, CIs::Union{Vector{CartesianIndex{3}},Vector{<:SurfacePoint}}) where {T <: SVector}
	return compute_Felas(reinterpret(LiquidCrystals.qtype(T), Q), ∇Q, constants, dV, CIs)
end 


#-----------------------------------------------------Dielectric Free Energy--------------------------------------------------------#
function compute_Fdiel(Q::AbstractArray{T}, E::Array{SVector{3, Float64}}, constants::NamedTuple, dV::Real, CIs::Union{Vector{CartesianIndex{3}},Vector{<:SurfacePoint}})  where {T <: LiquidCrystals.QLocal}
	
	ϵₐ = constants.ϵₐ
	
	energy = 0.0
	@inbounds for ci in CIs

		if ci isa CartesianIndex{3}
			index = ci  
		elseif ci isa SurfacePoint{3, Float64}
			index = ci.index  
		else
			error("Unsupported type for ci: $(typeof(ci))")
		end

		Eᵢ = E[index]
		Qᵢ = Q[index]

		@tensor t1 = -0.5 * ϵₐ * Eᵢ[i] * Qᵢ[i,j] * Eᵢ[j]
	
		energy += t1
	end
	
	return energy * dV
	 	
end

function compute_Fdiel(Q::AbstractArray{T}, E::Array{SVector{3, Float64}}, constants::NamedTuple, dV::Real, CIs::Union{Vector{CartesianIndex{3}},Vector{<:SurfacePoint}}) where {T <: SVector}
		compute_Fdiel(reinterpret(LiquidCrystals.qtype(T), Q), E, constants, dV, CIs)
end


#-----------------------------------------------------Flexoelectric Free Energy------------------------------------------------------------------#
function compute_Fflex(Q::AbstractArray{T},
	                    ∇Q::AbstractArray{SVector{3, SVector{6, Float64}}, 3},
						E::Array{SVector{3, Float64}},
						constants::NamedTuple, dV::Real, CIs::Union{Vector{CartesianIndex{3}},Vector{<:SurfacePoint}}) where {T <: LiquidCrystals.QLocal}
	
	ζ₁ = constants.ζ₁
	ζ₂ = constants.ζ₂

	energy = 0.0
	@inbounds for ci in CIs

		if ci isa CartesianIndex{3}
			index = ci  
		elseif ci isa SurfacePoint{3, Float64}
			index = ci.index  
		else
			error("Unsupported type for ci: $(typeof(ci))")
		end
			
		Qᵢ = Q[index]
		∇Qᵢ = reshape_and_reinterpret(∇Q[index])
		Eᵢ = E[index]
	
		@tensor t1 = ζ₁ * ∇Qᵢ[j,i,j] * Eᵢ[i]
		@tensor t2 = ζ₂ * Eᵢ[i] * Qᵢ[i,j] * ∇Qᵢ[k,j,k]
	
		energy += t1 + t2
	end
	
	return energy * dV
	 	
end

function compute_Fflex(Q::AbstractArray{T},
	                    ∇Q::AbstractArray{SVector{3, SVector{6, Float64}}, 3},
						E::Array{SVector{3, Float64}},
						constants::NamedTuple, dV::Real, CIs::Union{Vector{CartesianIndex{3}},Vector{<:SurfacePoint}}) where {T <: SVector}
	compute_Fflex(reinterpret(LiquidCrystals.qtype(T), Q), ∇Q, E, constants, dV, CIs)
end


end