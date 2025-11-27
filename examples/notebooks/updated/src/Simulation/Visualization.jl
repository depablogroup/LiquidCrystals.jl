#src/Simulation/Visualization.jl
module Visualization

 #--------------------------------------------------------------Plotting Tools------------------------------------------------------------------------------------#
function plot_particle_surface_energies(particle_surface_energies::Dict, saveat_times, path::String, model::PhysicsModel)

	particle_surface_energies["fTotal"] = particle_surface_energies["fAnchor"] .+ particle_surface_energies["fElas"]
	if model.flags[:flexoelectricity]
	    particle_surface_energies["fTotal"] += particle_surface_energies["fFlex"]
	end
	
	particle_surface_rising_zero = Dict(key => arr[1:4:end-1] for (key, arr) in particle_surface_energies)
	particle_surface_pos_peak   = Dict(key => arr[2:4:end-1] for (key, arr) in particle_surface_energies)
	particle_surface_falling_zero = Dict(key => arr[3:4:end-1] for (key, arr) in particle_surface_energies)
	particle_surface_neg_peak   = Dict(key => arr[4:4:end-1] for (key, arr) in particle_surface_energies)

	time_downsampled = saveat_times[1:4:end-1]

	fig = Figure(size=(1600, 1200))
	axes = []
		
	# Define the four phases to plot
	phases = [
	    ("Rising Zero Crossing (0°)", particle_surface_rising_zero),
	    ("Positive Peak (90°)", particle_surface_pos_peak),
	    ("Falling Zero Crossing (180°)", particle_surface_falling_zero),
	    ("Negative Peak (270°)", particle_surface_neg_peak)
	]

	energies = ["fElas", "fFlex", "fAnchor", "fTotal"]

	energy_ylims = Dict(energy => extrema(particle_surface_energies[energy]) for energy in energies)

	for (i, (phase_name, phase_data)) in enumerate(phases)
	    for (j, energy) in enumerate(energies)
	        # Skip if energy is not applicable (e.g., Flex/Diel disabled)
	        if (energy == "fFlex" && !model.flags[:flexoelectricity])
	            continue
	        end
	
	        ax = Axis(fig[i, j],
	            title="$(phase_name) - $(energy)",
	            xlabel="Time",
	            ylabel="Energy",
	            xgridvisible=true,
	            ygridvisible=true
	        )

			xlims!(ax, extrema(time_downsampled))
            ylims!(ax, energy_ylims[energy])
	
	        # Plot the downsampled data
	        if energy == "fTotal"
	            ΔfTotal = phase_data["fAnchor"] + phase_data["fElas"]
	            if model.flags[:flexoelectricity]
	                ΔfTotal += phase_data["fFlex"]
	            end
	            lines!(ax, time_downsampled, ΔfTotal, linewidth=2, color=:darkgoldenrod)
	        else
	            lines!(ax, time_downsampled, phase_data[energy], linewidth=2, color=:blue)
	        end
	
	        push!(axes, ax)
	    end
	end
	
	save(joinpath(path,"particle_surface_energies.png"), fig; size=(1600, 1200), dpi=600)
	
	fig
end

function plot_wall_energies(wall_energies::Dict, saveat_times, path::String, model::PhysicsModel)

	wall_energies["fTotal"] = wall_energies["fAnchor"] .+ wall_energies["fElas"]
	if model.flags[:flexoelectricity]
	    wall_energies["fTotal"] += wall_energies["fFlex"]
	end

	wall_rising_zero = Dict(key => arr[1:4:end-1] for (key, arr) in wall_energies)
	wall_pos_peak   = Dict(key => arr[2:4:end-1] for (key, arr) in wall_energies)
	wall_falling_zero = Dict(key => arr[3:4:end-1] for (key, arr) in wall_energies)
	wall_neg_peak   = Dict(key => arr[4:4:end-1] for (key, arr) in wall_energies)

	time_downsampled = saveat_times[1:4:end-1]

	fig = Figure(size=(1600, 1200))
	axes = []
		
	# Define the four phases to plot
	phases = [
	    ("Rising Zero Crossing (0°)", wall_rising_zero),
	    ("Positive Peak (90°)", wall_pos_peak),
	    ("Falling Zero Crossing (180°)", wall_falling_zero),
	    ("Negative Peak (270°)", wall_neg_peak)
	]

	energies = ["fElas", "fFlex", "fAnchor", "fTotal"]

	energy_ylims = Dict(energy => extrema(wall_energies[energy]) for energy in energies)

	for (i, (phase_name, phase_data)) in enumerate(phases)
	    for (j, energy) in enumerate(energies)
	        # Skip if energy is not applicable (e.g., Flex/Diel disabled)
	        if (energy == "fFlex" && !model.flags[:flexoelectricity])
	             continue
	        end
	
	        ax = Axis(fig[i, j],
	            title="$(phase_name) - $(energy)",
	            xlabel="Time",
	            ylabel="Energy",
	            xgridvisible=true,
	            ygridvisible=true
	        )

			xlims!(ax, extrema(time_downsampled))
            ylims!(ax, energy_ylims[energy])
	
	        # Plot the downsampled data
	        if energy == "fTotal"
	            ΔfTotal = phase_data["fAnchor"] + phase_data["fElas"]
	            if model.flags[:flexoelectricity]
	                ΔfTotal += phase_data["fFlex"]
	            end
	            lines!(ax, time_downsampled, ΔfTotal, linewidth=2, color=:darkgoldenrod)
	        else
	            lines!(ax, time_downsampled, phase_data[energy], linewidth=2, color=:blue)
	        end
	
	        push!(axes, ax)
	    end
	end
	
	save(joinpath(path,"wall_energies.png"), fig; size=(1600, 1200), dpi=600)
	
	fig
end

function plot_bulk_energies(bulk_energies::Dict, saveat_times, path::String, model::PhysicsModel)

	bulk_energies["fTotal"] = bulk_energies["fLdG"] .+ bulk_energies["fElas"]
	if model.flags[:flexoelectricity]
	    bulk_energies["fTotal"] += bulk_energies["fFlex"]
	end
	if model.flags[:dielectricity]
	    bulk_energies["fTotal"] += bulk_energies["fDiel"]
	end 

	bulk_rising_zero = Dict(key => arr[1:4:end-1] for (key, arr) in bulk_energies)
	bulk_pos_peak   = Dict(key => arr[2:4:end-1] for (key, arr) in bulk_energies)
	bulk_falling_zero = Dict(key => arr[3:4:end-1] for (key, arr) in bulk_energies)
	bulk_neg_peak   = Dict(key => arr[4:4:end-1] for (key, arr) in bulk_energies)
	
	time_downsampled = saveat_times[1:4:end-1]
	
	fig = Figure(size=(1600, 1200))
	axes = []
	
	# Define the four phases to plot
	phases = [
	    ("Rising Zero Crossing (0°)", bulk_rising_zero),
	    ("Positive Peak (90°)", bulk_pos_peak),
	    ("Falling Zero Crossing (180°)", bulk_falling_zero),
	    ("Negative Peak (270°)", bulk_neg_peak)
    ]

	energies = ["fLdG", "fElas", "fFlex", "fDiel", "fTotal"]

	energy_ylims = Dict(energy => extrema(bulk_energies[energy]) for energy in energies)

	for (i, (phase_name, phase_data)) in enumerate(phases)
	    for (j, energy) in enumerate(energies)
	        # Skip if energy is not applicable (e.g., Flex/Diel disabled)
	        if (energy == "fFlex" && !model.flags[:flexoelectricity]) ||
	            (energy == "fDiel" && !model.flags[:dielectricity])
	            continue
	        end
	
	        ax = Axis(fig[i, j],
	            title="$(phase_name) - $(energy)",
	            xlabel="Time",
	            ylabel="Energy",
	            xgridvisible=true,
	            ygridvisible=true
	        )

			xlims!(ax, extrema(time_downsampled))
            ylims!(ax, energy_ylims[energy])
	
	        # Plot the downsampled data
	        if energy == "fTotal"
	            ΔfTotal = phase_data["fLdG"] + phase_data["fElas"]
	            if model.flags[:flexoelectricity]
	                ΔfTotal += phase_data["fFlex"]
	            end
	            if model.flags[:dielectricity]
	                ΔfTotal += phase_data["fDiel"]
	            end
	            lines!(ax, time_downsampled, ΔfTotal, linewidth=2, color=:darkgoldenrod)
	        else
	            lines!(ax, time_downsampled, phase_data[energy], linewidth=2, color=:blue)
	        end
	
	        push!(axes, ax)
	    end
	end
	    
	save(joinpath(path, "bulk_energies.png"), fig; size=(1600, 1200), dpi=600)
	    
	return fig
end


function read_energy_data!(bulk_energies, wall_energies, particle_surface_energies; filename=joinpath(path, "energy_data.csv"))
    open(filename, "r") do io
        # Read and parse the header
        header = readline(io)
        columns = split(header, ',')

        
        # Create a mapping from column index to (target_dict, key)
        column_map = []
        for (i, col) in enumerate(columns)
            i == 1 && continue  # Skip the "time" column
            parts = split(col, '_')
            if length(parts) != 2
                error("Invalid column name: $col")
            end
            dict_prefix, key = parts
            
            # Map the prefix to the corresponding dictionary
            target_dict = if dict_prefix == "bulk"
                bulk_energies
            elseif dict_prefix == "wall"
                wall_energies
            elseif dict_prefix == "particle"
                particle_surface_energies
            else
                error("Unknown dictionary prefix: $dict_prefix")
            end
            
            # Ensure the key exists in the target dictionary
            if !haskey(target_dict, key)
                target_dict[key] = Float64[]  # Initialize if missing
            end
            
            push!(column_map, (target_dict, key, i))
        end

        
        # Process each data line
        for line in eachline(io)
            values = split(line, ',')			
            for (dict, key, idx) in column_map
                val = parse(Float64, values[idx])
                push!(dict[key], val)
            end
        end
    end
end

function get_energies(t::Real, Q::AbstractArray{SVector{6, Float64}}, ∇Q::AbstractArray{SVector{3, SVector{6, Float64}}}, model::PhysicsModel, indices::NamedTuple, energy_file)
	wall_points = [wall.surface_points for wall in system_state.walls if wall.is_anchored]
	if !isempty(wall_points)
		wall_indices = reduce(vcat, wall_points)
	else
		wall_indices = Vector{SurfacePoint{3, Float64}}()
	end

	Eₜ = fill(E_AC(t), (model.parameters.box.Nx,model.parameters.box.Ny,model.parameters.box.Nz))
		
	if model.parameters.particle.exists
		     
		fElas_particle = compute_Felas(Q, ∇Q, model.parameters.elasticity, dV, system_state.particles.surface_points)
		    
		fAnchor_particle = compute_FRP(Q, particle_params, dV, system_state.particles.surface_points)
		
		fElas_wall = compute_Felas(Q, ∇Q, model.parameters.elasticity, dV, wall_indices)

		fFlex_particle = 0
		fFlex_wall = 0
			
		if model.flags[:flexoelectricity]
			fFlex_particle += compute_Fflex(Q, ∇Q, Eₜ, model.parameters.flexoelectricity, dV, system_state.particles.surface_points)
			fFlex_wall += compute_Fflex(Q, ∇Q, Eₜ, model.parameters.flexoelectricity, dV, wall_indices)
		end
		
		fAnchor_wall = 0
		for (wall, wallParam) in zip(indices.walls, wall_params)
			if wall.is_anchored
		        fAnchor_wall += compute_FRP(Q, wallParam, dV, wall.surface_points)
			end
		end
	else
		            
		fElas_particle = 0.0
		fFlex_particle = 0.0
		fAnchor_particle = 0.0
		
		fElas_wall = compute_Felas(Q, ∇Q, model.parameters.elasticity, dV, wall_indices)
		fAnchor_wall = 0
		
		for (wall, wallParam) in zip(indices.walls, wall_params)
			if wall.is_anchored
		        fAnchor_wall += compute_FRP(Q, wallParam, dV, wall.surface_points)
			end
		end
			
			
		if model.flags[:flexoelectricity]
			fFlex_wall = compute_Fflex(Q, ∇Q, Eₜ, model.parameters.flexoelectricity, dV, wall_indices)
		else
			fFlex_wall = 0
		end
					
	end
		
	fLdG_bulk = compute_FLdG(Q, model.parameters.LandaudeGennes, dV, indices.bulk)
	fElas_bulk = compute_Felas(Q, ∇Q, model.parameters.elasticity, dV, indices.bulk)
		
	if model.flags[:flexoelectricity]
		fFlex_bulk = compute_Fflex(Q, ∇Q, Eₜ, model.parameters.flexoelectricity, dV, indices.bulk)
	else
		fFlex_bulk = 0
	end
		
	if model.flags[:dielectricity]
		fDiel_bulk = compute_Fdiel(Q, Eₜ, model.parameters.dielectricity, dV, indices.bulk)
	else
		fDiel_bulk = 0
	end
		
	data_line = "$t,$fLdG_bulk,$fElas_bulk,$fDiel_bulk,$fFlex_bulk,$fElas_wall,$fFlex_wall,$fAnchor_wall,$fElas_particle,$fFlex_particle,$fAnchor_particle\n"
	write(energy_file, data_line)
end

end