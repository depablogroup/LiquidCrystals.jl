#src/Simulation/Initialization.jl
module Initialization

using StaticArrays
using LinearAlgebra
using LiquidCrystals
using ..Utils
using ..FiniteDifferences

export insert_spherical_particle
export generate_initial_config, generate_aligned_initial_config
export get_BoxBC, get_indices_and_init_state, get_Qinit
export get_wall_and_particle_params

abstract type DirectorDimensionality end
struct TwoD <: DirectorDimensionality end
struct ThreeD <: DirectorDimensionality end

const EMPTY_SURFACE = Vector{SurfacePoint{3, Float64}}()

model = MODEL[]

Nx = model.parameters.box.Nx
Ny = model.parameters.box.Ny
Nz = model.parameters.box.Nz

const WALL_CONFIG = (
		left = (
	        normal = SVector(1.0, 0.0, 0.0),
	        range = (i=1, j=1:Ny, k=1:Nz),
	        is_anchored = false
	    ),
	    right = (
	        normal = SVector(-1.0, 0.0, 0.0),
	        range = (i=Nx, j=1:Ny, k=1:Nz),
	        is_anchored = false
	    ),
	    bottom = (
	        normal = SVector(0.0, 1.0, 0.0),
	        range = (i=1:Nx, j=1, k=1:Nz),
	        is_anchored = false  # Default state
	    ),
	    top = (
	        normal = SVector(0.0, -1.0, 0.0),
	        range = (i=1:Nx, j=Ny, k=1:Nz),
	        is_anchored = false
	    ),
	    back = (
	        normal = SVector(0.0, 0.0, 1.0),
	        range = (i=1:Nx, j=1:Ny, k=1),
	        is_anchored = false
	    ),
	    front = (
	        normal = SVector(0.0, 0.0, -1.0),
	        range = (i=1:Nx, j=1:Ny, k=Nz),
	        is_anchored = false
	    )
	)

function generate_surface_points(wall)
	return [
	    SurfacePoint(CartesianIndex(i, j, k), wall.normal)
	    for i in wall.range.i, j in wall.range.j, k in wall.range.k
	] |> vec  
end

function generate_initial_config(::Type{D}, U, shape) where {D <: LiquidCrystals.DirectorDimensionality}
	    
	N = prod(shape)
	S = LiquidCrystals.nematic_order_param(D, U)
	
	Q = Vector{LiquidCrystals.qtype(D)}(undef, N)
	n = LiquidCrystals.mtype(D)(undef)
	
	for i in eachindex(Q)
	    n .= rand.() .- 1 // 2
	    n .= n ./ norm(n)
	    Q[i] = LiquidCrystals.alignment_tensor(S, n)
	end
	
	return Q
end

function generate_initial_config(::Type{D}, U, shape, bc_box::LiquidCrystals.BoxBC) where {D <: LiquidCrystals.DirectorDimensionality}
	    
	N = prod(shape)
	Nx, Ny, Nz = shape
	S = LiquidCrystals.nematic_order_param(D, U)
	
	Q = Vector{LiquidCrystals.qtype(D)}(undef, N)
	n = LiquidCrystals.mtype(D)(undef)
	
	for i in eachindex(Q)
	    n .= rand.() .- 1 // 2
	    n .= n ./ norm(n)
	    Q[i] = LiquidCrystals.alignment_tensor(S, n)
	end
	
	Q_3D = reshape(Q, shape)
	
	for bc in bc_box.bcs
		if bc isa DirichletAxisBC{LiquidCrystals.XAxis}
		    Q_3D[1, :, :] .= fill(LiquidCrystals.QLocal(bc.Q_l), (Ny, Nz)) 
		    Q_3D[end, :, :] .= fill(LiquidCrystals.QLocal(bc.Q_h), (Ny, Nz))
		end
			
		if bc isa DirichletAxisBC{LiquidCrystals.YAxis}
		    Q_3D[:, 1, :] .= fill(LiquidCrystals.QLocal(bc.Q_l), (Nx, Nz))
		    Q_3D[:, end, :] .= fill(LiquidCrystals.QLocal(bc.Q_h), (Nx, Nz))
		end
		
		if bc isa DirichletAxisBC{LiquidCrystals.ZAxis}
		    Q_3D[:, :, 1] .= fill(LiquidCrystals.QLocal(bc.Q_l), (Nx, Ny))
		    Q_3D[:, :, end] .= fill(LiquidCrystals.QLocal(bc.Q_h), (Nx, Ny))
	    end 
	end
	
	return vec(Q_3D)
end

function generate_aligned_initial_config(::Type{D}, U, shape, align::Type{A}) where {D <: LiquidCrystals.DirectorDimensionality, A <: LiquidCrystals.Axis}
	
	N = prod(shape)
	S = LiquidCrystals.nematic_order_param(D, U)
	
	# Allocate memory for the Q field
	Q = Vector{LiquidCrystals.qtype(D)}(undef, N)
	
	# Write the Q field at each point of the grid
	for i in eachindex(Q)
	    n = define_director(D, A)
	    Q[i] = LiquidCrystals.alignment_tensor(S, n)
	end
	
	return Q
end

function define_director(::Type{D}, ::Type{A}, noise_level = 1e-3) where {D <: LiquidCrystals.DirectorDimensionality, A <: LiquidCrystals.Axis}
		
	axis_map_3D = Dict(LiquidCrystals.XAxis => 1, LiquidCrystals.YAxis => 2, LiquidCrystals.ZAxis => 3)
	axis_map_2D = Dict(LiquidCrystals.XAxis => 1, LiquidCrystals.YAxis => 2)

	n = LiquidCrystals.mtype(D)(undef)
	
	if D <: LiquidCrystals.ThreeD
	    index = axis_map_3D[A]
	elseif D <: LiquidCrystals.TwoD
	    index = axis_map_2D[A]
	else
	    error("Dimensionality $D is not implemented!")
	end
	
	n .= randn(eltype(n)) * noise_level
		
	n[index] = 1.0
	n .= n / norm(n)

	return n
end

function spherical_particle_CIs(shape, center::CartesianIndex{3}, radius::Real; tol = 0.5)
	
	@assert length(shape) == 3
	Nx, Ny, Nz = shape
	
	patch_CIs = CartesianIndex[]
	surfacePoints = SurfacePoint[]

	center = SVector{3, Float64}(Tuple(center)...)
	x0 = center[1]
	y0 = center[2]
	z0 = center[3]
	
	for i in Int(x0-radius):Int(x0+radius)
	    for j in Int(y0-radius):Int(y0+radius)
	        for k in Int(z0-2*radius):Int(z0+radius)
				if i>0 && j>0 && k>0
		            point = SVector{3, Float64}(i, j, k)
					normal = point - center
					normal = normal / norm(normal)
		            dist = norm(point - center)
		            if dist < radius
		                push!(patch_CIs, CartesianIndex(i, j, k))
		            end
					if (dist >= radius) && (dist < radius + tol)  
		                push!(surfacePoints, SurfacePoint(CartesianIndex(i, j, k), normal))
					end
				end
	        end
	    end
	end
	
	return patch_CIs, surfacePoints
end

function cubical_particle_CIs(shape, center::CartesianIndex{3}, d::Int64)

	@assert length(shape) == 3

	function getSurfaceIndices(normal::CartesianIndex{3})
		centerPoint = center .+ d .* normal
		return CartesianIndices(ntuple(i -> 
		    abs(normal[i]) == 1 ? (centerPoint[i]:centerPoint[i]) : (centerPoint[i] - (d-1)):(centerPoint[i] + (d-1)), 3
		)) 
	end

	function getBulkIndices()
		return CartesianIndices(ntuple(i ->
			(center[i] - (d-1)):(center[i] + (d-1)), 3
		))
	end

	interiorPoints = CartesianIndex[]
	surfacePoints = SurfacePoint[]

	for idx in getBulkIndices()
		push!(interiorPoints, idx)
	end
		
	interiorPoints = filter(ci -> all(c -> c > 0, Tuple(ci)), interiorPoints)

	normals = [
		CartesianIndex(1, 0, 0),   # +x face
		CartesianIndex(-1, 0, 0),  # -x face
		CartesianIndex(0, 1, 0),   # +y face
		CartesianIndex(0, -1, 0),  # -y face
		CartesianIndex(0, 0, 1),   # +z face
		CartesianIndex(0, 0, -1)   # -z face
	]

	for normal in normals
		filteredSurfaceIdxs = [ci for ci in getSurfaceIndices(normal) if all(x -> x > 1, Tuple(ci))]
		for idx in filteredSurfaceIdxs
			push!(surfacePoints, SurfacePoint(idx, SVector{3, Float64}(Tuple(normal)...)))
		end
	end

	return interiorPoints, surfacePoints
end

function insert_spherical_particle(center::CartesianIndex{3}, radius::Real, shape)
	
	particleCIs, particle_surfaceCIs = spherical_particle_CIs(shape, center, radius)
		
	return particleCIs, particle_surfaceCIs
end

function get_indices_and_init_state(model::PhysicsModel, Q_00::AbstractArray{SVector{6, Float64}})

	#Define system state (initial)
	system_state = (
	    walls = merge(WALL_CONFIG, (
	        (wall_name => 
	            let
					wall_config = getproperty(WALL_CONFIG, wall_name)
	                is_anchored = getproperty(model.parameters.wall_anchoring, wall_name).strength != 0
	                surface_points = is_anchored ? generate_surface_points(wall_config) : SurfacePoint[]
	                merge(wall_config, (; is_anchored, surface_points))
	            end
	        ) for wall_name in propertynames(model.parameters.wall_anchoring)
	    )),
		particles = (exists=model.parameters.particle.exists, surface_points=SurfacePoint[], bulkCIs=CartesianIndex[]),
	)

	wall_points = [wall.surface_points for wall in system_state.walls if wall.is_anchored]
	if !isempty(wall_points)
		surface_points = reduce(vcat, wall_points)
	else
		surface_points = Vector{SurfacePoint{3, Float64}}()
	end

	bulk_ranges = (
	    x = setdiff(1:Nx, vcat([wall.range.i for wall in system_state.walls if wall.is_anchored && :i ∈ propertynames(wall.range) && (wall == system_state.walls.left || wall == system_state.walls.right)]...)),  
	    y = setdiff(1:Ny, vcat([wall.range.j for wall in system_state.walls if wall.is_anchored && :j ∈ propertynames(wall.range) && (wall == system_state.walls.top || wall == system_state.walls.bottom)]...)),  
	    z = setdiff(1:Nz, vcat([wall.range.k for wall in system_state.walls if wall.is_anchored && :k ∈ propertynames(wall.range) && (wall == system_state.walls.front || wall == system_state.walls.back)]...))  
	)

	# Define bulk indices
	bulkCIs = [CartesianIndex(i, j, k) for i in bulk_ranges.x, j in bulk_ranges.y, k in bulk_ranges.z]

	if model.parameters.particle.exists 
		particle_BC_strength = model.parameters.particle.anchoring.strength
		particle_center = model.parameters.particle.center 
		particle_radius = model.parameters.particle.radius

		particleCIs, particle_surfaceCIs = insert_spherical_particle(particle_center, particle_radius, (Nx, Ny, Nz))
		
	    system_state = (
	        walls = system_state.walls, 
	        particles = (
	            exists = true,
	            surface_points = particle_surfaceCIs,
	            bulkCIs = particleCIs
	        )
	    )
	    bulkCIs = filter(bp -> !(bp in Set(system_state.particles.bulkCIs)), bulkCIs)
		bulkCIs = filter(bp -> !(bp in Set([sp.index for sp in system_state.particles.surface_points])), bulkCIs)
		
		"""
			NOTE: Here, if there exists surface points on the particle that are on the anchored wall, they get the treatment similar to that of the walls! 
		"""

		anchored_wall_sp_indices = Set{CartesianIndex{3}}()
		for wall in system_state.walls
		    if wall.is_anchored
		        union!(anchored_wall_sp_indices, [sp.index for sp in wall.surface_points])
		    end
		end

		filtered_particle_surfaceCIs = filter(
		    sp -> !(sp.index in anchored_wall_sp_indices),
		    system_state.particles.surface_points
		)
		
		system_state = merge(system_state, (
		    particles = merge(system_state.particles, (
		        surface_points = filtered_particle_surfaceCIs,
		    )),
		))

		updated_walls = map(system_state.walls) do wall
			if wall.is_anchored
				filtered_surface_points = filter(sp -> !(sp.index in Set(vcat(system_state.particles.bulkCIs, [sp.index for sp in system_state.particles.surface_points]))), wall.surface_points)
				return merge(wall, (surface_points = filtered_surface_points,))
			else
				return wall
			end
		end
		system_state = merge(system_state, (walls = updated_walls,))

		upt_wall_points = [wall.surface_points for wall in system_state.walls if wall.is_anchored]
		full_surface = vcat(vcat(upt_wall_points...), filtered_particle_surfaceCIs)

		indices = (
		    bulk = bulkCIs|>vec,
		    walls = system_state.walls,
		    particles = (
        		surface = system_state.particles.exists ? system_state.particles.surface_points : EMPTY_SURFACE,
        		bulk = system_state.particles.exists ? system_state.particles.bulkCIs : []
    		),
			full_surface = full_surface
		)
	else
		indices = (
		    bulk = bulkCIs|>vec,
		    walls = system_state.walls,
		    particles = (
        		surface = system_state.particles.exists ? system_state.particles.surface_points : EMPTY_SURFACE,
        		bulk = system_state.particles.exists ? system_state.particles.bulkCIs : []
    		),
			full_surface = surface_points
		)
	end

	

	# Q_00_modified = copy(Q_00)
	# if model.parameters.particle.exists
	# 	if model.parameters.particle.anchoring.strength == Inf
	# 		for ci in indices.particles.surface
	# 			n = ci.normal
	# 			Q_00_modified[ci.index] = LiquidCrystals.QLocal(q_eq .* (n *n' - (1/3) .* I(3))).data
	# 		end
	# 		indices = (
	# 			bulk = indices.bulk,
	# 			walls = indices.walls,
	# 			particles = (
	# 				surface = [],
	# 				bulk = indices.particles.bulk
	# 			)
	# 		)
	# 	end
	# end

	return indices, Q_00

end


function get_wall_and_particle_params(model::PhysicsModel)
	wall_params = model.parameters.wall_anchoring
	if model.parameters.particle.exists
		particle_params = model.parameters.particle
	else
		particle_params = nothing
	end

	return wall_params, particle_params
end

function get_Qinit(Q₀)
	ST = LiquidCrystals.stype(eltype(Q₀))
	QT = eltype(Q₀)
	Q_0 = reinterpret(reshape, ST, Q₀)

	return reshape(Q_0, Nx, Ny, Nz)
end

function get_BoxBC(model::PhysicsModel)

	bcs = []
	wallAnchoring = model.parameters.wall_anchoring
	if wallAnchoring.left.strength == Inf && wallAnchoring.right.strength == Inf
		n_lo = wallAnchoring.left.direction
		n_hi = wallAnchoring.right.direction
		Qdirichlet_lo_x = LiquidCrystals.QLocal(q_eq .* (n_lo * n_lo' .- (1/3) .* I(3))).data
		Qdirichlet_hi_x = LiquidCrystals.QLocal(q_eq .* (n_hi * n_hi' .- (1/3) .* I(3))).data
		push!(bcs, DirichletAxisBC{LiquidCrystals.XAxis}(Qdirichlet_lo_x, Qdirichlet_hi_x))
	elseif wallAnchoring.left.strength != Inf && wallAnchoring.right.strength != Inf && wallAnchoring.left.strength != 0.0 && wallAnchoring.right.strength != 0.0
		push!(bcs, AnchoringAxisBC{LiquidCrystals.XAxis}())
	elseif wallAnchoring.left.strength != Inf && wallAnchoring.right.strength != Inf && wallAnchoring.left.strength == 0.0 && wallAnchoring.right.strength == 0.0
		push!(bcs, LiquidCrystals.PeriodicAxisBC{LiquidCrystals.XAxis}(0.0, 0.0))
	else
		error("Hetero-AnchoringAxisBC along an axis has not yet been implemented!!")
	end

	if wallAnchoring.top.strength == Inf && wallAnchoring.bottom.strength == Inf
		n_lo = wallAnchoring.top.direction
		n_hi = wallAnchoring.bottom.direction
		Qdirichlet_lo_x = LiquidCrystals.QLocal(q_eq .* (n_lo * n_lo' .- (1/3) .* I(3))).data
		Qdirichlet_hi_x = LiquidCrystals.QLocal(q_eq .* (n_hi * n_hi' .- (1/3) .* I(3))).data
		push!(bcs, DirichletAxisBC{LiquidCrystals.YAxis}(Qdirichlet_lo_x, Qdirichlet_hi_x))
	elseif wallAnchoring.top.strength != Inf && wallAnchoring.bottom.strength != Inf && wallAnchoring.top.strength != 0.0 && wallAnchoring.bottom.strength != 0.0
		push!(bcs, AnchoringAxisBC{LiquidCrystals.YAxis}())
	elseif wallAnchoring.top.strength != Inf && wallAnchoring.bottom.strength != Inf && wallAnchoring.top.strength == 0.0 && wallAnchoring.bottom.strength == 0.0
		push!(bcs, LiquidCrystals.PeriodicAxisBC{LiquidCrystals.YAxis}(0.0, 0.0))
	else
		error("Hetero-AnchoringAxisBC along an axis has not yet been implemented!!")
	end

	if wallAnchoring.front.strength == Inf && wallAnchoring.back.strength == Inf
		n_lo = wallAnchoring.back.direction
		n_hi = wallAnchoring.front.direction
		Qdirichlet_lo_x = LiquidCrystals.QLocal(q_eq .* (n_lo * n_lo' .- (1/3) .* I(3))).data
		Qdirichlet_hi_x = LiquidCrystals.QLocal(q_eq .* (n_hi * n_hi' .- (1/3) .* I(3))).data
		push!(bcs, DirichletAxisBC{LiquidCrystals.ZAxis}(Qdirichlet_lo_x, Qdirichlet_hi_x))
	elseif wallAnchoring.front.strength != Inf && wallAnchoring.back.strength != Inf && wallAnchoring.front.strength != 0.0 && wallAnchoring.back.strength != 0.0
		push!(bcs, AnchoringAxisBC{LiquidCrystals.ZAxis}())
	elseif wallAnchoring.front.strength != Inf && wallAnchoring.back.strength != Inf && wallAnchoring.front.strength == 0.0 && wallAnchoring.back.strength == 0.0
		push!(bcs, LiquidCrystals.PeriodicAxisBC{LiquidCrystals.ZAxis}(0.0, 0.0))
	else
		error("Hetero-AnchoringAxisBC along an axis has not yet been implemented!!")
	end

	return LiquidCrystals.BoxBC(bcs...)
end

end

#----------------------------Module Ends-----------------------------------------------------------------#
#------------------------- Additional Functions -----------------------------------------------------------#

# function triangle_indices(A::CartesianIndex{2}, B::CartesianIndex{2}, C::CartesianIndex{2})
# 	x1, y1 = A.I
# 	x2, y2 = B.I
# 	x3, y3 = C.I
	
# 	xmin, xmax = extrema((x1, x2, x3))
# 	ymin, ymax = extrema((y1, y2, y3))
	
# 	denominator = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
# 	denominator == 0 && error("Degenerate triangle")
	
# 	indices = CartesianIndex{2}[]
# 	for x in xmin:xmax, y in ymin:ymax
# 	    dx, dy = x - x1, y - y1
# 	    s_num = dx * (y3 - y1) - dy * (x3 - x1)
# 	    t_num = dy * (x2 - x1) - dx * (y2 - y1)
	
# 	    if denominator > 0
# 	        (s_num ≥ 0) && (t_num ≥ 0) && (s_num + t_num ≤ denominator) && push!(indices, CartesianIndex(x, y))
# 	    else
# 	        (s_num ≤ 0) && (t_num ≤ 0) && (s_num + t_num ≥ denominator) && push!(indices, CartesianIndex(x, y))
# 	    end
# 	end
# 	return indices
# end

# function triangular_prism_indices(base_vertices::NTuple{3, CartesianIndex{3}}, height_range::UnitRange{Int})
# 	# Extract 2D triangle (ignoring z)
# 	# The Z-axis is the direction of the height
# 	A_xy = CartesianIndex(base_vertices[1].I[1:2]...)
# 	B_xy = CartesianIndex(base_vertices[2].I[1:2]...)
# 	C_xy = CartesianIndex(base_vertices[3].I[1:2]...)
# 	triangle_2d = triangle_indices(A_xy, B_xy, C_xy)
	    
# 	zmin, zmax = extrema(height_range)
# 	prism_indices = CartesianIndex{3}[]
# 	for z in zmin:zmax, idx in triangle_2d
# 	    push!(prism_indices, CartesianIndex(idx.I..., z))
# 	end
# 	return prism_indices
# end
	
# function add_custom_region!(::Type{D}, Qbox, indices, ypoint::Int64) where {D <: LiquidCrystals.DirectorDimensionality}
# 	for idx in indices
# 		n = LiquidCrystals.mtype(D)(undef)
# 		pointing_dir = CartesianIndex(Int(Nx/2), ypoint, idx[3]) .- idx
# 		if norm(Tuple(pointing_dir)) != 0.0
# 			n .= SVector{3, Float64}(Tuple(pointing_dir)) / norm(Tuple(pointing_dir))
# 			nn = n * n'
# 			@tensor trace = nn[k,k]
# 			Q = q_eq .* (nn - (1/3) .* trace .* I(3))
# 			Qbox[idx] = LiquidCrystals.QLocal(Q).data
# 		end
# 	end
# end
	
#-------------------------------------------------------------------------------------------------------------------------------

