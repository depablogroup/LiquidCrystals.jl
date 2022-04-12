### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# ╔═╡ d4df4278-b056-11ec-18db-097556b36509
begin

using LinearAlgebra
using StaticArrays
using OrdinaryDiffEq
using RecursiveArrayTools
using Plots
using PyPlot

end

# ╔═╡ 002d026a-c09f-4421-b800-3a7e7e14851d
using LiquidCrystals

# ╔═╡ 176ad4d0-8140-4000-9dc2-424a5cd14b94
let
	Pack = Base.require(
		Base.PkgId(Base.UUID(0x44cfe95a1eb252eab672e2afdf69b78f), "Pkg")
	)
	Pack.develop(path="/home/jonathan/.julia/dev/LiquidCrystals")
end

# ╔═╡ 6589a256-4190-48c8-829d-c937280fc2a3
U=3.0

# ╔═╡ 23c72901-ef1a-410d-ac08-399a9121b5ac
Nx=Ny=75

# ╔═╡ 8e42d469-152f-46c9-ba5b-83d1fd198232
dims=(Nx,Ny,1)

# ╔═╡ 2e4434f4-3207-4b35-b75d-739becc00bc9
#compared to the general one, I changed the definition of n and the entries q3 and q5 in the matrix
function generate_initial_config_2D(dims, U)
	N = prod(dims)
	S = 1 // 4 + 3 // 4 * sqrt(1 - 8 / (3 * U))

	# Allocate memory for the Q field
	Q = Vector{QLocal{Float64}}(undef, N)
	# MVector below comes from the StaticArrays package
	n = MVector{2, Float64}(undef)  #MVector{3, Float64}(undef)

	# Write the Q field at each point of the grid
	for i in eachindex(Q)
		# generate a random unit vector
		n .= rand.() .- 1 // 2
		n .= n ./ norm(n)

		q₁ = S * (n[1] * n[1] - 1 // 3)
		q₂ = S * (n[1] * n[2])
		q₃ = 0#S * (n[1] * n[3])
		q₄ = S * (n[2] * n[2] - 1 // 3)
		q₅ = 0#S * (n[2] * n[3])
		q₆ = S * ( - 1 // 3)

		Q[i] = QLocal(q₁, q₂, q₃, q₄, q₅, q₆)
	end

	return Q
end

# ╔═╡ c0499600-0b28-450e-985d-cd79377513ed
Q₀ = generate_initial_config_2D(dims, U)

# ╔═╡ e8f25b28-a0f3-40fb-a1fe-bae65edcc055
S = 1 // 4 + 3 // 4 * sqrt(1 - 8 / (3 * U))

# ╔═╡ 29e8b617-6fa1-4d7b-9d65-89dd1c5427e3
Q_0=reinterpret(reshape, SVector{6, Float64}, Q₀)

# ╔═╡ f23433e9-f6e5-4207-a5dd-97c8d57b5411
Q_00=reshape(Q_0, Nx,Ny)

# ╔═╡ 30fc9e12-ff00-4640-9c48-fd67cf8d7bfa
Q_trial = generate_initial_config((Nx+2,Ny+2,1), U)

# ╔═╡ 891eebd3-3de7-42e1-8287-ac7301ac74f2
Q_trial1=reinterpret(reshape, SVector{6, Float64},similar(Q_trial))

# ╔═╡ 7af3c9ac-ed01-4f14-9718-ec4dc24898c7
E_Q=reshape(Q_trial1, Nx+2,Ny+2)

# ╔═╡ 9cc3585d-31d5-432f-8423-d3bbfe058321
P_Q=similar(E_Q)

# ╔═╡ 19340a4c-1d6d-4835-99b5-2509372d167a
dx= 1.0

# ╔═╡ 1650c98c-f31c-44c9-8eac-eada9b467f08


# ╔═╡ e32759c6-4ef6-4043-b4ff-41f2a6d07008


# ╔═╡ ba87ac48-d968-47a8-8ca1-452c3583809e


# ╔═╡ a46f4be1-3973-498e-903c-0dbe4bc589ff
Op=CenteredDifference{2}(2, dx)

# ╔═╡ e9c7ec35-8150-4d41-9290-484ed3940f2a
bc=PeriodicBC()

# ╔═╡ 35f9157d-0cf1-4f9b-b929-48f46c4e94dc
P_Q[2:end-1, 2:end-1]

# ╔═╡ b99b418a-59df-405b-9d43-6d682ce179c9
A=1

# ╔═╡ 6acc864b-4cbe-4a22-acfa-36e7af020378
Q_2=-0.1*reinterpret(reshape, SVector{6, Float64},volterra(A,U,reinterpret(reshape, QLocal{Float64}, Q_00))  )  + 0.1*P_Q[2:end-1, 2:end-1]

# ╔═╡ 2ad7facf-482a-4400-a9ea-e047d5474b08
volterra(A,U,reinterpret(reshape, QLocal{Float64}, Q_2))

# ╔═╡ 20360839-195c-4a26-b3eb-75d372f5a5b5
params = (Op,bc, E_Q, P_Q)

# ╔═╡ 29bd6b89-b959-41b4-898f-7a9b6f3eb587
t_f=5000

# ╔═╡ 9ebb114e-d209-4c15-9b04-bccbe6d6ce7e
function Frank_Oseen(K, Q::AbstractArray{T}) where {T}
	∇²Q = similar(Q)

	# Allocate one vector for partial computations
	#q = MVector{6, eltype(T)}(undef)
	
	@inbounds for (i, Qᵢ) in enumerate(Q)
		#Qᵢ² = Qᵢ * Qᵢ
		# Extract the vector representations to work with them here
	#	vᵢ = Qᵢ.data
	#	vᵢ² = Qᵢ².data
		q = # (1 - U / 3) .* vᵢ .- U .* (vᵢ² .- tr(Qᵢ²) .* (vᵢ .+ δ₃))
		# `q` above is a MVector so we need to convert back to QLocal
		∇²Q[i] = QLocal(q)
	end
	
	return ∇²Q
end

# ╔═╡ d365ff2e-e06f-4ddc-9f95-c73af806400d
function trial_Q!(dQ, Q, p, t)
	Op, bc, E_Q, P_Q = p
	
	E_Q= apply_BCs(Op, Q, bc, E_Q)

	mul!(P_Q, Op, E_Q) #Laplacian operator
	
	
	 dQ .=  -0.1.*reinterpret(reshape, SVector{6, Float64},volterra(A,U,reinterpret(reshape, QLocal{Float64}, Q))  ) +0.1*@view(P_Q[2:end-1, 2:end-1]) 
	
	#hLdG= B[1] + 1
end 

# ╔═╡ 38f7a102-e733-49b5-a76a-36ea562e222c
prob = ODEProblem(trial_Q!, Q_00, (0.0, t_f), params)

# ╔═╡ ee00acfc-ce27-4176-ab66-1866905d5ebf
sol= solve(prob,Tsit5(),progress=true,saveat=(t_f/100),save_start=false)

# ╔═╡ 95734e5d-213a-4cdc-8cab-88b74b7368c3


# ╔═╡ c27f5407-5068-4a0b-801b-65dd83975d20
AA=reinterpret(reshape, QLocal{Float64}, sol)

# ╔═╡ 80877ca1-b27f-4cfb-9a0b-a4ce5124620d
function get_max_eigen(M)
	vals, vecs = eigen(M)
	return vals[end], vecs[:, end]
end

# ╔═╡ 8bcaef99-4ce0-4a59-b82c-331d76b7c7b3


# ╔═╡ 42840e73-6703-4015-98f6-f4b40fce1cac
function get_s_and_directors(A)
	# d = ndims(A)
	α = 1.5  # d // (d - 1)
	Qs = reinterpret(reshape, QLocal{Float64}, A)
	n = length(Qs)
	Ss = Vector{Float64}(undef, n)
	n̂s = Vector{SVector{3, Float64}}(undef, n)
	
	@inbounds @simd for i in eachindex(Qs)
		s, n̂ = get_max_eigen(Qs[i])
		Ss[i] = α * s
		n̂s[i] = n̂
	end
	
	return Ss,n̂s
end

# ╔═╡ 5145d969-b5fe-4a07-8ea5-8f5c81514a8c


# ╔═╡ 425f1710-f681-4a45-b11a-881c699da543
function writevtk(filename, space, Ss, n̂s)
	open(filename, "w") do file
		write(file, "# vtk DataFile Version 3.0 \nvtk output\nASCII\nDATASET UNSTRUCTURED_GRID \n")
		write(file, "POINTS $(length(Ss)) float\n")
		write(file, space)
	

		write(file, "\n POINT_DATA $(length(Ss)) \n")
		write(file, "SCALARS S float\nLOOKUP_TABLE default\n")
		
		write(file, join(Ss, "\n"))


		write(file, "\n VECTORS Director float\n")
		
		#for i in eachindex(n_array[1,:,:])
			#write(file, n_array[1,i])
		
		write(file, replace(join(n̂s, "\n"), r"[\[,\]]" => ""))
	end
end

# ╔═╡ dd41ef33-a9c6-48aa-ae54-2918e0385db8
space = replace(join(Iterators.product(1:Nx, 1:Ny), " 0 \n"), r"[(,)]" => "") * " 0 \n"

# ╔═╡ 653286e4-516f-439a-9b8a-9ed18af73406
function paramovie2(sol, space, u₀, nₜ)
	name = "frame_0.vtk"
	Ss, n̂s = get_s_and_directors(u₀)
	writevtk(name, space, Ss, n̂s)
	
	for i in 1:nₜ
		name = "frame_$i.vtk"
		Ss, n̂s = get_s_and_directors(sol[i])
		writevtk(name, space, Ss, n̂s)
	end
end

# ╔═╡ 6f47f468-9235-44d7-9def-93d5b60e9d5d
paramovie2(sol, space, Q_00, 100)

# ╔═╡ dcd08501-2585-4193-9068-1e2643c6d7de


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LiquidCrystals = "90861fa5-6d0e-476f-90c5-b56067c52d58"
OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[compat]
LiquidCrystals = "~0.1.0"
OrdinaryDiffEq = "~6.7.1"
Plots = "~1.27.3"
PyPlot = "~2.10.0"
RecursiveArrayTools = "~2.25.1"
StaticArrays = "~1.4.3"
"""

# ╔═╡ Cell order:
# ╠═d4df4278-b056-11ec-18db-097556b36509
# ╠═176ad4d0-8140-4000-9dc2-424a5cd14b94
# ╠═002d026a-c09f-4421-b800-3a7e7e14851d
# ╠═6589a256-4190-48c8-829d-c937280fc2a3
# ╠═23c72901-ef1a-410d-ac08-399a9121b5ac
# ╠═8e42d469-152f-46c9-ba5b-83d1fd198232
# ╠═2e4434f4-3207-4b35-b75d-739becc00bc9
# ╠═c0499600-0b28-450e-985d-cd79377513ed
# ╠═e8f25b28-a0f3-40fb-a1fe-bae65edcc055
# ╠═29e8b617-6fa1-4d7b-9d65-89dd1c5427e3
# ╠═f23433e9-f6e5-4207-a5dd-97c8d57b5411
# ╠═30fc9e12-ff00-4640-9c48-fd67cf8d7bfa
# ╠═891eebd3-3de7-42e1-8287-ac7301ac74f2
# ╠═7af3c9ac-ed01-4f14-9718-ec4dc24898c7
# ╠═9cc3585d-31d5-432f-8423-d3bbfe058321
# ╠═19340a4c-1d6d-4835-99b5-2509372d167a
# ╠═1650c98c-f31c-44c9-8eac-eada9b467f08
# ╠═e32759c6-4ef6-4043-b4ff-41f2a6d07008
# ╠═ba87ac48-d968-47a8-8ca1-452c3583809e
# ╠═a46f4be1-3973-498e-903c-0dbe4bc589ff
# ╠═e9c7ec35-8150-4d41-9290-484ed3940f2a
# ╠═35f9157d-0cf1-4f9b-b929-48f46c4e94dc
# ╠═6acc864b-4cbe-4a22-acfa-36e7af020378
# ╠═b99b418a-59df-405b-9d43-6d682ce179c9
# ╠═2ad7facf-482a-4400-a9ea-e047d5474b08
# ╠═20360839-195c-4a26-b3eb-75d372f5a5b5
# ╠═29bd6b89-b959-41b4-898f-7a9b6f3eb587
# ╠═9ebb114e-d209-4c15-9b04-bccbe6d6ce7e
# ╠═d365ff2e-e06f-4ddc-9f95-c73af806400d
# ╠═38f7a102-e733-49b5-a76a-36ea562e222c
# ╠═ee00acfc-ce27-4176-ab66-1866905d5ebf
# ╠═95734e5d-213a-4cdc-8cab-88b74b7368c3
# ╠═c27f5407-5068-4a0b-801b-65dd83975d20
# ╠═80877ca1-b27f-4cfb-9a0b-a4ce5124620d
# ╠═8bcaef99-4ce0-4a59-b82c-331d76b7c7b3
# ╠═42840e73-6703-4015-98f6-f4b40fce1cac
# ╠═5145d969-b5fe-4a07-8ea5-8f5c81514a8c
# ╠═425f1710-f681-4a45-b11a-881c699da543
# ╠═dd41ef33-a9c6-48aa-ae54-2918e0385db8
# ╠═653286e4-516f-439a-9b8a-9ed18af73406
# ╠═6f47f468-9235-44d7-9def-93d5b60e9d5d
# ╠═dcd08501-2585-4193-9068-1e2643c6d7de
# ╟─00000000-0000-0000-0000-000000000001
