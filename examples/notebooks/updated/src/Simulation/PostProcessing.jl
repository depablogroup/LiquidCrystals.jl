#src/Simulation/PostProcessing.jl
module PostProcessing

using StaticArrays

using LiquidCrystals
using ..Utils
using ..FreeEnergies

export space
export paramovie, writeFrame, write_model_to_file

model = MODEL[]
indices = INDICES[]

Nx = model.parameters.box.Nx
Ny = model.parameters.box.Ny
Nz = model.parameters.box.Nz

function readvtk(filename, ::Type{D}, U, shape) where {D <: LiquidCrystals.DirectorDimensionality}
    # Initialize grid to store director vectors

	N = prod(shape)
	ndim = length(shape)
	
    Q = Vector{LiquidCrystals.qtype(D)}(undef, N)
	S = LiquidCrystals.nematic_order_param(D, U)
    n = LiquidCrystals.mtype(D)(undef)

    open(filename, "r") do file
        lines = readlines(file)

        # Find the index where director vectors begin
        vector_start = findfirst(x -> occursin(r"VECTORS Director", x), lines) + 1

        # Parse the director vectors from the file
		if ndim == 2
            Nx, Ny = shape
            for j in 1:Ny
                for i in 1:Nx
                    index = (j - 1) * Nx + (i - 1)
                    line_index = vector_start + index - 1
                    vector_values = parse.(Float64, split(lines[line_index]))[1:2]
					n = SVector{2, Float64}(vector_values...)
                    Q[index] = LiquidCrystals.alignment_tensor(S, n)
                end
            end
        elseif ndim == 3
            Nx, Ny, Nz = shape
            for k in 1:Nz
                for j in 1:Ny
                    for i in 1:Nx
                        index = (k - 1) * (Nx * Ny) + (j - 1) * Nx + (i - 1)
                        line_index = vector_start + index
                        vector_values = parse.(Float64, split(lines[line_index]))
						n = SVector{3, Float64}(vector_values...)
                        Q[index + 1] = LiquidCrystals.alignment_tensor(S, n)
                    end
                end
            end
        else
            error("Unsupported dimensionality: $ndim")
        end
    end

    return Q
end

function writevtk(filename, space, Ss, n̂s)
    open(filename, "w") do file
        write(file, "# vtk DataFile Version 3.0 \nvtk output\nASCII\nDATASET UNSTRUCTURED_GRID \n")
        write(file, "POINTS $(length(Ss)) float\n")
        write(file, space)

        write(file, "\n POINT_DATA $(length(Ss)) \n")
        write(file, "SCALARS q float\nLOOKUP_TABLE default\n")

        write(file, join(Ss, "\n"))

        write(file, "\n VECTORS Director float\n")

        delim = eltype(n̂s) <: SVector{2} ? " 0\n" : " \n"
        write(file, replace(join(n̂s, delim), r"[\[,\]]" => ""))
        write(file, delim)
    end
end

function writevtk(filename, space, Ss, n̂s, FLdG)
    open(filename, "w") do file
        write(file, "# vtk DataFile Version 3.0 \nvtk output\nASCII\nDATASET UNSTRUCTURED_GRID \n")
        write(file, "POINTS $(length(Ss)) float\n")
        write(file, space)

        write(file, "\n POINT_DATA $(length(Ss)) \n")
        write(file, "SCALARS q float\nLOOKUP_TABLE default\n")
		
		write(file, join(Ss, "\n"))

		write(file, "\n POINT_DATA $(length(FLdG)) \n")
        write(file, "SCALARS FLdG float\nLOOKUP_TABLE default\n")

		write(file, join(FLdG, "\n"))

        write(file, "\n VECTORS Director float\n")

        delim = eltype(n̂s) <: SVector{2} ? " 0\n" : " \n"
        write(file, replace(join(n̂s, delim), r"[\[,\]]" => ""))
        write(file, delim)
    end
end


function s_and_directors(A::AbstractArray{T}) where {T <: SVector}
	# If we get `SVector`s, first reinterpret as `QLocal`s.
	return s_and_directors(reinterpret(reshape, LiquidCrystals.qtype(T), A))
end
	
function s_and_directors(Qs::AbstractArray{QT}) where {QT <: LiquidCrystals.QLocal}
	d = size(QT, 1)  # dimensionality of the directors
	c = d // (d - 1)
	
	n = length(Qs)
	NT = LiquidCrystals.ntype(QT)
	Ss = Vector{eltype(NT)}(undef, n)
	n̂s = Vector{NT}(undef, n)
	
	@inbounds @simd for i in eachindex(Qs)
	    λ, n̂ = LiquidCrystals.max_eigen(Qs[i])
	    Ss[i] = c * λ
	    n̂s[i] = n̂
	end
	
	return Ss, n̂s
end

function FLdG_paraview(FLdG::AbstractArray{Base.Float64}) 
	
	len = length(FLdG)
	fldg = Vector{Base.Float64}(undef, len)
	
	@inbounds @simd for i in eachindex(FLdG)
	    fldg[i] = FLdG[i]
	end
	
	return fldg
end

function Felas_energy_on_grid(Q::AbstractArray{T}, ∇Q::SubArray{<:SVector{3, <:SVector{6, Float64}}, 3}, constants::Dict) where {T <: LiquidCrystals.QLocal}
	NT = LiquidCrystals.ntype(T)
	n = length(Q)
	Felas = Vector{eltype(NT)}(undef, n)

	dV = dx * dy * dz
	
	@inbounds for i in eachindex(Q)
		Felas[i] = compute_Felasᵢ(Q[i], ∇Q[i], dV, constants)
	end
	
	return Felas
end


function writeFrame(Q, space, path)
    Ss, n̂s = s_and_directors(Q)
    writevtk(path, space, Ss, n̂s)
end

space = replace(
    join(Iterators.product(1:Nx, 1:Ny, 1:Nz), " \n"), r"[(,)]" => ""
) * " \n"

function paramovie(sol, space, nₜ, path)
	mkpath(path)
	for i in 1:nₜ
	    name = joinpath(path, "frame_$i.vtk")
	    Ss, n̂s = s_and_directors(sol[i])
		if model.flags[:LandaudeGennes]
			FLdG = FLdG_paraview(compute_FLdG_heatmap(sol[i], model, indices.bulk))
			writevtk(name, space, Ss, n̂s, FLdG)
		else
	    	writevtk(name, space, Ss, n̂s)
		end
	end
end

function write_model_to_file(model::PhysicsModel, filename::String; total_time::Float64=0.0)
	open(filename, "w") do file
	    # Write the model structure
	    println(file, "model = PhysicsModel(")
	    write_parameters(file, model.parameters, 4)
	    write_flags(file, model.flags, 4)
	    println(file, ")")
	
	    # Write additional information
	    println(file, "\n# Additional Information")
	    println(file, "total_time = ", total_time)
	        
	    # Add number of AC cycles if electric field type is :AC
	    if model.parameters.external_conditions.electric_field.type == :AC
	        frequency = model.parameters.external_conditions.electric_field.frequency
	        num_cycles = total_time * frequency
	        println(file, "number_of_AC_cycles = ", num_cycles)
	    end
	end
end
	
function write_parameters(file, parameters, indent_level)
	indent = " " ^ indent_level
	println(file, indent * "parameters = (")
	for field in propertynames(parameters)
	    value = getproperty(parameters, field)
	    if typeof(value) <: NamedTuple
	        println(file, indent * "    ", field, " = (")
	        write_namedtuple(file, value, indent_level + 8)
	        println(file, indent * "    ),")
	    else
	        println(file, indent * "    ", field, " = ", value, ",")
	    end
	end
	println(file, indent * ")")
end
	
function write_namedtuple(file, nt::NamedTuple, indent_level)
	indent = " " ^ indent_level
	for field in propertynames(nt)
	    value = getproperty(nt, field)
	    if typeof(value) <: NamedTuple
	        println(file, indent, field, " = (")
	        write_namedtuple(file, value, indent_level + 4)
	        println(file, indent, "),")
	    else
	        println(file, indent, field, " = ", value, ",")
	    end
	end
end
	
function write_flags(file, flags, indent_level)
	indent = " " ^ indent_level
	println(file, indent * "flags = Dict(")
	for (key, value) in flags
	    println(file, indent * "    :", key, " => ", value, ",")
	end
	println(file, indent * ")")
end


end