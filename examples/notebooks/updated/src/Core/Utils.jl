#src/Core/Utils.jl
module Utils

using TOML
using StaticArrays
using LiquidCrystals

export DEFAULT_ANCHORING
export load_params, reshape_and_reinterpret, convert_to_QLocal
export PhysicsModel, SurfacePoint
export save_∇Q, adjust_trace!, raise_error!, traceless_condition

struct PhysicsModel{P <: NamedTuple, F <: Dict{Symbol, Bool}}
	parameters::P
	flags::F
end

const MODEL = Ref{PhysicsModel}()
const INDICES = Ref{NamedTuple}()
export MODEL
export INDICES

const q_steady_state = Ref{Float64}()
export q_steady_state

struct SurfacePoint{N, T}
    index::CartesianIndex{N}
    normal::SVector{N, T}
end

SurfacePoint(index::CartesianIndex{N}, normal::SVector{N, T}) where {N, T} = SurfacePoint{N, T}(index, normal)

function PhysicsModel(; parameters, flags)
	return PhysicsModel(parameters, flags)
end

const DEFAULT_ANCHORING = (
	strength = 0.0,
	type = :planar,
	direction = nothing
)

"""
    load_params(filename="params.toml")

Load parameters from a TOML file and return a `PhysicsModel` configuration.
Handles elasticity simplification (L = L₁ if L₂/L₃/L₄ = 0).
"""
function load_params(filename="params.toml")
    params = TOML.parsefile(filename)

	println(params)
    
    # Helper to parse wall anchoring (returns DEFAULT if not specified)
    function get_anchoring(wall)
        if haskey(params["wall_anchoring"], wall)
            anchoring = params["wall_anchoring"][wall]
            return (
                strength = anchoring["strength"],
                type = Symbol(anchoring["type"]),
                direction = SVector{3}(anchoring["direction"])
            )
        else
            return DEFAULT_ANCHORING
        end
    end

	elasticity_params = if params["L2"] == 0.0 && params["L3"] == 0.0 && params["L4"] == 0.0
        (L = params["L1"],)  # one-constant approximation
    else
        (L₁=params["L1"], L₂=params["L2"], L₃=params["L3"], L₄=params["L4"])  
    end

    model = PhysicsModel(
        parameters = (
            box = (Nx=params["Nx"], Ny=params["Ny"], Nz=params["Nz"]),
            dx = params["dx"],
            dy = params["dy"],
            dz = params["dz"],
            dt = params["dt"],
			total_time = params["tf"],
            Γ = params["Gamma"],
            LandaudeGennes = (A₀=params["A"], U=params["U"]),
            elasticity = elasticity_params,
            flexoelectricity = (ζ₁=params["zeta1"], ζ₂=params["zeta2"]),
            dielectricity = (ϵₐ=params["diel_anisotropy"],),
            wall_anchoring = (
                left = get_anchoring("left"),
                right = get_anchoring("right"),
                bottom = get_anchoring("bottom"),
                top = get_anchoring("top"),
                back = get_anchoring("back"),
                front = get_anchoring("front"),
            ),
            external_conditions = (
                electric_field = (
                    amplitude = SVector{3}(params["external_conditions"]["electric_field"]["amplitude"]),
                    base_strength = SVector{3}(params["external_conditions"]["electric_field"]["base_strength"]),
                    frequency = params["external_conditions"]["electric_field"]["frequency"],
                    type = Symbol(params["external_conditions"]["electric_field"]["type"]),
                    randnoise = params["external_conditions"]["electric_field"]["randnoise"],
                ),
            ),
            particle = (
                exists = params["particle"]["exists"],
                anchoring = (
                    strength = params["particle"]["anchoring"]["strength"],
                    type = Symbol(params["particle"]["anchoring"]["type"]),
                ),
                center = CartesianIndex(
                    params["particle"]["center"]["x"],
                    params["particle"]["center"]["y"],
                    params["particle"]["center"]["z"],
                ),
                radius = params["particle"]["radius"],
            ),
        ),
        flags = Dict(
            :LandaudeGennes => params["flags"]["LandaudeGennes"],
            :elasticity => params["flags"]["elasticity"],
            :anchoring => params["flags"]["anchoring"],
            :flexoelectricity => params["flags"]["flexoelectricity"],
            :dielectricity => params["flags"]["dielectricity"],
        )
    )
    return model
end


# Reshape and reinterpret

"""
Convert an `SVector` structure representing gradients or second-order derivatives of a tensor field into a standard multidimensional array of `Float64` values.
	
# Methods
	
- `reshape_and_reinterpret(∇Q_SVector::SVector{3, SVector{6, Float64}})`: 
	Converts a 3D gradient tensor `∇Q_SVector` represented by an `SVector{3, SVector{6, Float64}}` to a 3×3×3 array of `Float64` values, where each element corresponds to a component of the gradient tensor `∂Qᵢⱼ/∂rₘ`.
	
- `reshape_and_reinterpret(∇Q_SVector::SVector{2, SVector{3, Float64}})`: 
	Converts a 2D gradient tensor `∇Q_SVector` represented by an `SVector{2, SVector{3, Float64}}` to a 2×2×2 array of `Float64` values, where each element corresponds to a component of the gradient tensor `∂Qᵢⱼ/∂rₘ`.
	
- `reshape_and_reinterpret(∇∇Q_SVector::SVector{3, SVector{3, SVector{6, Float64}}})`: 
	Converts a 3D double gradient tensor `∇∇Q_SVector` represented by an `SVector{3, SVector{3, SVector{6, Float64}}}` to a 3×3×3×3 array of `Float64` values, where each element corresponds to a component of the second-order derivative tensor `∂²Qᵢⱼ/∂rₘ∂rₙ`.
	
- `reshape_and_reinterpret(∇∇Q_SVector::SVector{2, SVector{2, SVector{3, Float64}}})`: 
	Converts a 2D double gradient tensor `∇∇Q_SVector` represented by an `SVector{2, SVector{2, SVector{3, Float64}}}` to a 2×2×2×2 array of `Float64` values, where each element corresponds to a component of the second-order derivative tensor `∂²Qᵢⱼ/∂rₘ∂rₙ`.
	
# Returns
- A multidimensional array of `Float64` values with dimensions corresponding to the input tensor structure, containing the components of the tensor derivatives.
	
# Example
```
julia> ∇Q_SVector = SVector(SVector(1.0, 2.0, 3.0), SVector(4.0, 5.0, 6.0), SVector(7.0, 8.0, 9.0))
julia> ∇Q = reshape_and_reinterpret(∇Q_SVector)
		
julia> ∇∇Q_SVector = SVector(SVector(SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), SVector(7.0, 8.0, 9.0, 10.0, 11.0, 12.0), SVector(13.0, 14.0, 15.0, 16.0, 17.0, 18.0)),
		                    SVector(SVector(19.0, 20.0, 21.0, 22.0, 23.0, 24.0), SVector(25.0, 26.0, 27.0, 28.0, 29.0, 30.0), SVector(31.0, 32.0, 33.0, 34.0, 35.0, 36.0)),
		                    SVector(SVector(37.0, 38.0, 39.0, 40.0, 41.0, 42.0), SVector(43.0, 44.0, 45.0, 46.0, 47.0, 48.0), SVector(49.0, 50.0, 51.0, 52.0, 53.0, 54.0)))
		
julia> ∇∇Q = reshape_and_reinterpret(∇∇Q_SVector)
```
"""
function reshape_and_reinterpret(∇E_SVector::SVector{3, SVector{3, Float64}})

	dim = 3

	∇E = Array{Float64, 2}(undef, dim, dim)

	for m in 1:dim
		for i in 1:dim
			∇E[m, i] = ∇E_SVector[m][i]
			# Here, ∇E[m, i] represents ∂Eᵢ/∂rₘ
		end
	end

	return ∇E

end

function reshape_and_reinterpret(∇E_SVector::SVector{2, SVector{2, Float64}})

	dim = 2

	∇E = Array{Float64, 2}(undef, dim, dim)

	for m in 1:dim
		for i in 1:dim
			∇E[m, i] = ∇E_SVector[m][i]
			# Here, ∇E[m, i] represents ∂Eᵢ/∂rₘ
		end
	end

	return ∇E

end
	
function reshape_and_reinterpret(∇Q_SVector::SVector{3, SVector{6, Float64}})

	dim = 3
		
	∇Q = Array{Float64, 3}(undef, dim, dim, dim)

	for m in 1:dim
		for i in 1:dim
			for j in 1:dim
				∇Q[m, i, j] = LiquidCrystals.QLocal(∇Q_SVector[m])[i,j]
				# Here, ∇Q[m, i, j] represents ∂Qᵢⱼ/∂rₘ
			end
		end
	end
	
	return ∇Q
end

function reshape_and_reinterpret(∇Q_SVector::SVector{2, SVector{3, Float64}})

	dim = 2
		
	∇Q = Array{Float64, 3}(undef, dim, dim, dim)

	for m in 1:dim
		for i in 1:dim
			for j in 1:dim
				∇Q[m, i, j] = LiquidCrystals.QLocal(∇Q_SVector[m])[i,j]
				# Here, ∇Q[m, i, j] represents ∂Qᵢⱼ/∂rₘ
			end
		end
	end
	
	return ∇Q
end

function reshape_and_reinterpret(∇∇Q_SVector::SMatrix{3, 3, SVector{6, Float64}, 9})

	# For 3D, on each grid point, it is a 3×3 tensor of second-order derivatives where each element is a SVector of size 6 which represents a 3×3 Q-tensor

	dim = 3
		
	∇∇Q = Array{Float64, 4}(undef, dim, dim, dim, dim)
		            
	for m in 1:dim
		for n in 1:dim
			for i in 1:dim
				for j in 1:dim
					∇∇Q[m, n, i, j] = LiquidCrystals.QLocal(∇∇Q_SVector[m,n])[i,j]
					# Here, ∇∇Q[m, n, i, j] represents ∂²Qᵢⱼ/∂rₘ∂rₙ
				end
			end
		end
	end
		
	return ∇∇Q
end


function reshape_and_reinterpret(∇∇Q_SVector::SMatrix{2, 2, SVector{3, Float64}, 4})

	dim = 2
		
	∇∇Q = Array{Float64, 4}(undef, dim, dim, dim, dim)
		            
	for m in 1:dim
		for n in 1:dim
			for i in 1:dim
				for j in 1:dim
					∇∇Q[m, n, i, j] = LiquidCrystals.QLocal(∇∇Q_SVector[m,n])[i,j]
					# Here, ∇∇Q[m, n, p, q] represents ∂²Q[p,q]/∂rₘ∂rₙ
				end
			end
		end
	end
		
	return ∇∇Q
end


function convert_to_QLocal(mat::Matrix{Float64})
	
	if size(mat) == (2,2)
		svec = SVector(mat[1,1], mat[2,1], mat[2,2])
		return LiquidCrystals.QLocal(svec)
	end

	if size(mat) == (3,3)
		svec = SVector(mat[1,1], mat[1,2], mat[1,3], mat[2,2], mat[2,3], mat[3,3])
		return LiquidCrystals.QLocal(svec)
	end

end

function check_NaN_element!(A::Array{SVector{3, SVector{6, Float64}}, 3}, t)
    
    for (i, element) in enumerate(A)
        for (j, subelement) in enumerate(element)
            if any(isnan, subelement)
                error("Simulation stopped: Found NaN vector at index ($i, $j) at time $t")
            end
        end
    end
end

function save_∇Q(u, t, integrator)
    ∇Q = copy(integrator.p.current_∇Q)
    return ∇Q
end

function traceless_condition(Q, t, ODEProblem, get_bool_matrix = false)
	boolArr = abs.(getindex.(Q, 1) .+ getindex.(Q, 4) .+ getindex.(Q, 6)) .> 1e-8
	if get_bool_matrix
		return boolArr
	else 
		return any(boolArr)
	end
end 


function adjust_trace!(ODEProblem)
	Q = ODEProblem.u
	boolArr = traceless_condition(Q, ODEProblem.t, ODEProblem, true)
	Q[boolArr] .= SVector{6, Float64}.(getindex.(Q[boolArr], 1), 
	                                	 getindex.(Q[boolArr], 2),
										   getindex.(Q[boolArr], 3),
											 getindex.(Q[boolArr], 4), 
	                                		   getindex.(Q[boolArr], 5),
										 		 -(getindex.(Q[boolArr], 1) .+ getindex.(Q[boolArr], 4)))
end

function raise_error!(ODEProblem)
	println("Traceless condition not met! Simulation terminated 😧")
	terminate!(ODEProblem)
end



end