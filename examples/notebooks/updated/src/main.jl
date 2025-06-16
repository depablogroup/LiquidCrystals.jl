using GLMakie
using LinearAlgebra
using OrdinaryDiffEq
using StaticArrays
using DifferentialEquations: DiscreteCallback
using JLD2
using TensorOperations

import Pkg as JLPkg
JLPkg.develop(path = "D:\\Codes\\LiquidCrystals")
using LiquidCrystals
using DiffEqCallbacks
using TOML

include("./Core/Utils.jl")
using ..Utils

model = load_params("params.toml")
MODEL[] = model

include("./Core/FiniteDifferences.jl")
include("./Physics/FunctionalDerivatives.jl")
include("./Physics/ElectricFields.jl")
include("./Physics/FreeEnergies.jl")

using ..FiniteDifferences
using ..FunctionalDerivatives
using ..ElectricFields
using ..FreeEnergies

const ∇CD = FiniteDifferences.CenteredDifference{3, Gradient}(model.parameters.dx)
const ∇FD = FiniteDifferences.ForwardDifference{3, Gradient}(model.parameters.dx)
const ∇BD = FiniteDifferences.BackwardDifference{3, Gradient}(model.parameters.dx)
const Δ = FiniteDifferences.CenteredDifference{3, Laplacian}(model.parameters.dx)
const div = FiniteDifferences.CenteredDifference{3, Divergence}(model.parameters.dx)
const ∇∇ = FiniteDifferences.CenteredDifference{3, Hessian}(model.parameters.dx)

include("./Simulation/Initialization.jl")
using ..Initialization

q_eq = LiquidCrystals.nematic_order_param(LiquidCrystals.ThreeD, model.parameters.LandaudeGennes.U)
q_steady_state[] = q_eq

begin 
	function bulkFuncDerivative(Q, ∇Q, ∇∇Q::SubArray{<:SMatrix{3, 3, <:SVector{6, Float64}}, 3}, E, ∇E, p)
		model, indices = p
	
		QLdG = reinterpret(eltype(Q), compute_δfLdG_δQ_bulk(Q, model.parameters.LandaudeGennes, indices))
		
		QElas = reinterpret(eltype(Q), compute_δfelas_δQ_bulk(Q, ∇Q, ∇∇Q, model.parameters.elasticity, indices))
		
		δF_δQ = QLdG .+ QElas
	
		if model.flags[:flexoelectricity]
		    QFlex = reinterpret(eltype(Q), compute_δflex_δQ_bulk(Q, ∇Q, E, ∇E, model.parameters.flexoelectricity, indices))
		    δF_δQ .+= QFlex
		end
	
		if model.flags[:dielectricity]
		    QDiel = reinterpret(eltype(Q), compute_δfdiel_δQ_bulk(Q, E, model.parameters.dielectricity, indices))
		    δF_δQ .+= QDiel
		end
		
		return δF_δQ
	end

	function bulkFuncDerivative(Q, ∇Q, ΔQ::SubArray{<:SVector{6, Float64}, 3}, E, ∇E, p)
		model, indices = p
	
		QLdG = reinterpret(eltype(Q), compute_δfLdG_δQ_bulk(Q, model.parameters.LandaudeGennes, indices))
		
		QElas = reinterpret(eltype(Q), compute_δfelas_δQ_bulk(Q, ΔQ, model.parameters.elasticity, indices))
		
		δF_δQ = QLdG .+ QElas
	
		if model.flags[:flexoelectricity]
		    QFlex = reinterpret(eltype(Q), compute_δflex_δQ_bulk(Q, ∇Q, E, ∇E, model.parameters.flexoelectricity, indices))
		    δF_δQ .+= QFlex
		end
	
		if model.flags[:dielectricity]
		    QDiel = reinterpret(eltype(Q), compute_δfdiel_δQ_bulk(Q, E, model.parameters.dielectricity, indices))
		    δF_δQ .+= QDiel
		end
		
		return δF_δQ
	end
	
end

function surfaceFuncDerivative(Q, ∇Q, E, p)
	model, indices, wall_params, particle_params = p

	QParticleBC = fill(SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), size(Q))
	if model.parameters.particle.exists
		QParticleBC += reinterpret(eltype(Q), compute_∂fRP_∂Q(Q, indices.particles.surface, particle_params))
	end

	QElas = reinterpret(eltype(Q), compute_νp_∂felas_∂∇Q(Q, ∇Q, model.parameters.elasticity, indices.full_surface))
	Qνp_∂f∂GradQ = QElas

	if model.flags[:flexoelectricity]
		QFlex = reinterpret(eltype(Q), compute_νp_∂flex_∂∇Q(Q, ∇Q, E, model.parameters.flexoelectricity, indices.full_surface))
		Qνp_∂f∂GradQ .+= QFlex
	end
		
	QWallBC = fill(SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), size(Q))
	for (wall, wallParam) in zip(indices.walls, wall_params)
		if wall.is_anchored
			QWallBC = QWallBC .+ reinterpret(eltype(Q), compute_∂fRP_∂Q(Q, wall.surface_points, wallParam))
		end
	end

	QSurface = QWallBC .+ QParticleBC

	δF_δQ = QSurface .- Qνp_∂f∂GradQ
	
	return δF_δQ 
end

begin
	function ginzburgLandauHessian!(dQ, Q, p, t)
		box, Eₜ, P_∇E, E_∇Q, P_∇Q, E_∇∇Q, P_∇∇Q, model, indices, wall_params, particle_params = p.bc_box, p.E0, p.P_∇E, p.E_∇Q, p.P_∇Q, p.E_∇∇Q, p.P_∇∇Q, p.model, p.indices, p.wall_params, p.particle_params

		E_∇Q = apply_BCs(∇CD, Q, box, E_∇Q)
		mul!(P_∇Q, ∇CD, E_∇Q)
		∇Q = @view P_∇Q[2:end-1, 2:end-1, 2:end-1]
		
		p.current_∇Q .= ∇Q

		E_∇∇Q = apply_BCs(∇∇, Q, box, E_∇∇Q)
		mul!(P_∇∇Q, ∇∇, E_∇∇Q)
		∇∇Q = @view P_∇∇Q[2:end-1, 2:end-1, 2:end-1]

		Eₜ .= fill(E_AC(t, model), size(Eₜ))
		∇E = @view P_∇E[2:end-1, 2:end-1, 2:end-1]

		p_bulk = (model, indices.bulk)
		p_surface = (model, indices, wall_params, particle_params)

		δF_δQ = bulkFuncDerivative(Q, ∇Q, ∇∇Q, Eₜ, ∇E, p_bulk)
		δF_δQ_surface = surfaceFuncDerivative(Q, ∇Q, Eₜ, p_surface)

		dQ .= .- model.parameters.Γ .* (δF_δQ .+ δF_δQ_surface)
		
		@info "time: $t"

	end

	function ginzburgLandauLaplacian!(dQ, Q, p, t)
		box, Eₜ, P_∇E, E_∇Q, P_∇Q, E_ΔQ, P_ΔQ, model, indices, wall_params, particle_params = p.bc_box, p.E0, p.P_∇E, p.E_∇Q, p.P_∇Q, p.E_ΔQ, p.P_ΔQ, p.model, p.indices, p.wall_params, p.particle_params

		E_∇Q = apply_BCs(∇CD, Q, box, E_∇Q)
		mul!(P_∇Q, ∇CD, E_∇Q)
		∇Q = @view P_∇Q[2:end-1, 2:end-1, 2:end-1]
		
		p.current_∇Q .= ∇Q

		E_ΔQ = apply_BCs(Δ, Q, bc_box, E_ΔQ)
		mul!(P_ΔQ, Δ, E_ΔQ)
		ΔQ = @view P_ΔQ[2:end-1, 2:end-1, 2:end-1]

		Eₜ .= fill(E_AC(t, model), size(Eₜ))
		∇E = @view P_∇E[2:end-1, 2:end-1, 2:end-1]

		p_bulk = (model, indices.bulk)
		p_surface = (model, indices, wall_params, particle_params)

		δF_δQ = bulkFuncDerivative(Q, ∇Q, ΔQ, Eₜ, ∇E, p_bulk)
		δF_δQ_surface = surfaceFuncDerivative(Q, ∇Q, Eₜ, p_surface)

		dQ .= .- model.parameters.Γ .* (δF_δQ .+ δF_δQ_surface)
		
		@info "time: $t"

	end

end

function get_caches(model::PhysicsModel, bc_box, Qinit; Einit=nothing)
	E_ΔQ, P_ΔQ = build_caches(Δ, bc_box, Qinit)
	E_∇Q, P_∇Q = build_caches(∇CD, bc_box, Qinit)
	E_∇∇Q, P_∇∇Q = build_caches(∇∇, bc_box, Qinit)

	if !isnothing(Einit)
		E_∇E, P_∇E = build_caches(∇CD, bc_box, Einit)
		E_∇E .= fill(model.parameters.external_conditions.electric_field.base_strength, size(E_∇E))
		mul!(P_∇E, ∇CD, E_∇E)

		return E_ΔQ, P_ΔQ, E_∇Q, P_∇Q, E_∇∇Q, P_∇∇Q, E_∇E, P_∇E
	end

	return E_ΔQ, P_ΔQ, E_∇Q, P_∇Q, E_∇∇Q, P_∇∇Q, nothing, nothing
end

Qdirichlet_lo = q_eq .* SVector(-1/3,0.0,0.0,-1/3,0.0,2/3)
Qdirichlet_hi = q_eq .* SVector(-1/3,0.0,0.0,2/3,0.0,-1/3)

bc_box = get_BoxBC(model)

# Q₀ = generate_aligned_initial_config(LiquidCrystals.ThreeD, 
# 									model.parameters.LandaudeGennes.U, 
# 									(model.parameters.box.Nx, model.parameters.box.Ny, model.parameters.box.Nz), 
# 									LiquidCrystals.XAxis)
# Q₀ = readvtk("D:/Projects/testing/solitons-Noe/q-data-16/initialization/frame_100.vtk", LiquidCrystals.ThreeD, model.parameters.LandaudeGennes.U, (Nx,Ny,Nz))
Q₀ = Initialization.generate_initial_config(LiquidCrystals.ThreeD, 
							model.parameters.LandaudeGennes.U, 
							(model.parameters.box.Nx, model.parameters.box.Ny, model.parameters.box.Nz), 
							bc_box)

indices, Q_00 = get_indices_and_init_state(model, get_Qinit(Q₀))
INDICES[] = indices

include("./Simulation/PostProcessing.jl")
using ..PostProcessing

E0 = ElectricFields.fill_box_with_constant_electric_field(LiquidCrystals.ThreeD, 
										(model.parameters.box.Nx, model.parameters.box.Ny, model.parameters.box.Nz), 
										model.parameters.external_conditions.electric_field.base_strength)
	
E_ΔQ, P_ΔQ, E_∇Q, P_∇Q, E_∇∇Q, P_∇∇Q, E_∇E, P_∇E = get_caches(model, bc_box, Q_00, Einit=E0)


path = "D:/Projects/testing/solitons-Noe/VS-Code/test2/"
if !isdir(path)
   	mkpath(path)
end

begin

	Ncycles = 1
	dV = model.parameters.dx * model.parameters.dy * model.parameters.dz
	dS = model.parameters.dx * model.parameters.dz

	wall_params, particle_params = get_wall_and_particle_params(model)

	Nframes = 100
	saveat_times = 0:(model.parameters.total_time/Nframes):model.parameters.total_time

	
end 

function run_simulation(Q_00, model, indices, Nframes, saveat_times, path)

	write_model_to_file(model, joinpath(path,"model.out"); total_time=model.parameters.total_time)

	current_∇Q = zeros(SVector{3, SVector{6, Float64}}, size(Q_00))
	gradQ = SavedValues(Float64, typeof(current_∇Q))
	saveeverystep(u, t, integrator) = true
		
	cb1 = DiscreteCallback((Q, t, ODEProblem) -> traceless_condition(Q, t, ODEProblem, false), (ODEProblem) -> raise_error!(ODEProblem); save_positions=(false, false))
	cb2 = SavingCallback(save_∇Q, gradQ; saveat=saveat_times)
	cb_set = CallbackSet(cb1, cb2)
		
	if model.parameters.elasticity isa NamedTuple{(:L,)}
		params = (
			bc_box = bc_box,
			E0 = E0,
			P_ΔQ = P_ΔQ,
			E_ΔQ = E_ΔQ,
			P_∇E = P_∇E,
			E_∇Q = E_∇Q,
			P_∇Q = P_∇Q,
			current_∇Q = current_∇Q,
			model = model,
			indices = indices,
			wall_params = wall_params,
			particle_params = particle_params
		)
		prob = ODEProblem(ginzburgLandauLaplacian!, Q_00, (0.0, model.parameters.total_time), params)
	elseif model.parameters.elasticity isa NamedTuple{(:L₁, :L₂, :L₃, :L₄)}
		params = (
			bc_box = bc_box,
			E0 = E0,
			P_∇E = P_∇E,
			E_∇Q = E_∇Q,
			P_∇Q = P_∇Q,
			E_∇∇Q = E_∇∇Q,
			P_∇∇Q = P_∇∇Q,
			current_∇Q = current_∇Q,
			model = model,
			indices = indices,
			wall_params = wall_params,
			particle_params = particle_params
		)
		prob = ODEProblem(ginzburgLandauHessian!, Q_00, (0.0, model.parameters.total_time), params)
	end
		
	solQ = solve(prob, Euler(), adaptive=false, dt = model.parameters.dt, saveat = saveat_times, tstops = saveat_times, save_start = true, callback = cb_set)
	# solQ = solve(prob, CarpenterKennedy2N54(williamson_condition=true), dt = model.parameters.dt, saveat = saveat_times, save_start = true, callback = cb_set)
	# solQ = solve(prob, Tsit5(), dt = dt, saveat = saveat_times, save_start = true, callback = cb_set)

	paramovie(solQ, space, Nframes, path)

	return solQ, gradQ
end

writeFrame(Q_00, space, joinpath(path, "init.vtk"))
solQ, gradQ = @time run_simulation(Q_00, model, indices, Nframes, saveat_times, path)
@save joinpath(path, "results.jld2") solQ gradQ


# begin
# 	bulk_energies = Dict("fLdG" => Float64[], "fElas" => Float64[], "fFlex" => Float64[], "fDiel" => Float64[])
# 	wall_energies = Dict("fLdG" => Float64[], "fElas" => Float64[], "fFlex" => Float64[], "fDiel" => Float64[], "fAnchor" => Float64[])
# 	particle_surface_energies = Dict("fLdG" => Float64[], "fElas" => Float64[], "fFlex" => Float64[], "fDiel" => Float64[], "fAnchor" => Float64[])
		
# 	read_energy_data!(bulk_energies, wall_energies, particle_surface_energies, filename=joinpath(path, "energy_data.csv"))
# end

# # ╔═╡ f3ddcb4b-ea30-4c4a-8d5c-49abd58dcf65
# md"""
# # Wall Energy Components
# """

# # ╔═╡ b7386f22-e4a1-422d-bf1c-38045e6a0c64
# plot_wall_energies(wall_energies, saveat_times, path, model)

# # ╔═╡ a06f7c9e-e769-4c86-b913-0d7a538e3d4e
# md"""
# # Bulk Energy Components
# """

# # ╔═╡ 4cd39c28-53b3-406a-86d6-d8780deecb73
# plot_bulk_energies(bulk_energies, saveat_times, path, model)

# # ╔═╡ 00284ba7-6ce2-4a7a-896b-a607e01c52c5
# md"""
# # Particle Surface Energy Components
# """

# # ╔═╡ 320918df-2c72-41ad-abb2-1721e8778e23
# if @isdefined(particle_surfaceCIs) && @isdefined(particleCIs)
# 	 plot_particle_surface_energies(particle_surface_energies, saveat_times, path, model)
# end

# # ╔═╡ 35278b84-3411-4dbc-962e-8701a220ec4a
# bulk_energies


# begin
# # Time points
# t_values = 0:model.parameters.dt:(Ncycles * 200 * model.parameters.dt) # Time range (from 0 to 5000 steps)

# # Evaluate the function for each time point
# E_values = [E_AC(t) for t in t_values]

# # Extract components
# Ex = [E[1] for E in E_values]
# Ey = [E[2] for E in E_values]
# Ez = [E[3] for E in E_values]

# # Define colors for components
# colors = (:red, :blue, :green)

# # Create the plot
# figure = Figure(resolution=(800, 600))

# ax = Axis(figure[1, 1], title="AC Electric Field Components over Time",
#           xlabel="Time (t)", ylabel="Field", titlealign=:center)

# lines!(ax, t_values, Ex, color=colors[1], label="Eₓ(t)")
# lines!(ax, t_values, Ey, color=colors[2], label="Eᵧ(t)")
# lines!(ax, t_values, Ez, color=colors[3], label="E𝓏(t)")

# axislegend(ax, position=:rt)
# figure

# save(joinpath(path,"electric-field.png"), figure; size=(800, 500), dpi=600)

# end

