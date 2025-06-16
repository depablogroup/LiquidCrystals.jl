#src/Physics/ElectricFields.jl
module ElectricFields

using StaticArrays
using LiquidCrystals
using LinearAlgebra

export E_AC, E_DC
export fill_box_with_constant_electric_field

function fill_box_with_constant_electric_field(::Type{D}, shape, EVector::SVector{3, Float64}) where {D <: LiquidCrystals.DirectorDimensionality}
    @assert length(shape) == 3
    Nx, Ny, Nz = shape
    
    E = fill(EVector, Nx, Ny, Nz)
    
    return E
end

function E_AC(t, model; tstart=0, noise::SVector{3, Float64}=SVector(0.0, 0.0, 0.0))

	electricField = model.parameters.external_conditions.electric_field
	@assert electricField.type == :AC

	E₀ = electricField.amplitude
	E₀Base = electricField.base_strength
	fᶜ = electricField.frequency 
	
	ωᶜ = 2π * fᶜ
	E = E₀ .* sin(ωᶜ * t) .+ E₀Base

	if t > tstart
		if electricField.randnoise
			return E .+ noise .* @SVector [rand(), rand(), rand()]
		else
			return E 
		end
	else
		return @SVector [0.0, 0.0, 0.0]
	end
end

function E_DC(t, model; tstart=0, noise::SVector{3, Float64}=SVector(0.0, 0.0, 0.0))

	electricField = model.parameters.external_conditions.electric_field
	@assert electricField.type == :DC

	E₀Base = electricField.base_strength

    if t > tstart
        if electricField.randnoise
			return E₀Base .+ noise .* @SVector [0.0, rand(), rand()]
		else
			return E₀Base
		end
    else
        return @SVector [0.0, 0.0, 0.0]
    end
end


end