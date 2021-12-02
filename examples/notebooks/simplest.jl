### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ 76bc042c-40c0-11ec-1f62-1fece22a67b5
using Pkg

# ╔═╡ f02876aa-a5f4-4417-83d9-7c24ed90643f
pkg"activate ."

# ╔═╡ 2d85c3b1-401b-4719-9f19-3db2d8ff09c2
using DifferentialEquations

# ╔═╡ 7a54874b-62d0-43ef-9abd-3882c9e3e3c1
using LinearAlgebra  # imports norm

# ╔═╡ 5a851a55-6004-4752-9c3d-c13dbce5f0c6
using StaticArrays  # imports MVector

# ╔═╡ ff39156e-0fc4-477d-a6ce-68358a2c1c6d
using BenchmarkTools  # imports @belapsed

# ╔═╡ 038a20bf-6100-4798-be56-fe2f34defefc
md"""
Setup the environment and load libraries
"""

# ╔═╡ a1d6582f-2416-4b80-ad9f-e3fc78d94d35
md"""
# Initialization
"""

# ╔═╡ f994ee31-f226-460c-a2f6-a27545aec18e
function generate_initial_config(N, U)
	nx, ny, nz = N
	S = 1 // 4 + 3 // 4 * sqrt(1 - 8 / (3 * U))

	Q = Matrix{Float64}(undef, (6, prod(N)))
	ns = rand(Float64, (3, prod(N))) .- 1 // 2
	ns .= ns ./ (norm.(eachcol(ns)))'

	for (i, n) in enumerate(eachcol(ns))
		Q[1, i] = S * (n[1] * n[1] - 1 // 3)
		Q[2, i] = S * (n[1] * n[2])
		Q[3, i] = S * (n[1] * n[3])
		Q[4, i] = S * (n[2] * n[2] - 1 // 3)
		Q[5, i] = S * (n[2] * n[3])
		Q[6, i] = S * (n[3] * n[3] - 1 // 3)
	end

	return Q
end

# ╔═╡ 17b6628c-09df-4cb7-8b0a-512de2db9d68
function generate_initial_config_optimized(dims, U)
	N = prod(dims)
	S = 1 // 4 + 3 // 4 * sqrt(1 - 8 / (3 * U))

	# Allocate memory for the Q field
	Q = Matrix{Float64}(undef, (6, N))
	# MVector below comes from the StaticArrays package
	n = MVector{3, Float64}(undef)

	# Write the Q field at each point of the grid
	for q in eachcol(Q)
		# generate a random unit vector
		n .= rand.() .- 1 // 2
		n .= n ./ norm(n)

		q[1] = S * (n[1] * n[1] - 1 // 3)
		q[2] = S * (n[1] * n[2])
		q[3] = S * (n[1] * n[3])
		q[4] = S * (n[2] * n[2] - 1 // 3)
		q[5] = S * (n[2] * n[3])
		q[6] = S * (n[3] * n[3] - 1 // 3)
	end

	return Q
end

# ╔═╡ 72c9f4c0-cee6-4df0-9d74-46bb68c810de
md"Dimensions of the grid"

# ╔═╡ cb4da596-09c3-4089-bee3-bdc6158f8888
dims = (10, 10, 10)

# ╔═╡ 9cbfff8b-2155-4c1f-b6e3-7df41abc836f
U = 3.0

# ╔═╡ 9bfb745c-c3e0-4d17-92c4-6d3a25f76401
generate_initial_config(dims, U)

# ╔═╡ 1cbe26b5-33d8-4c83-be78-d182a162bccf
Q₀ = generate_initial_config_optimized(dims, U)

# ╔═╡ c906f201-a445-4122-be5f-48f5d9f149f7
md"""
Measure the time it takes to run each function
"""

# ╔═╡ 240b46ee-f8bd-4d72-9cfe-246d4053993e
@belapsed generate_initial_config(dims, U)  # time measured in seconds

# ╔═╡ eb6c5dd7-ea00-4a69-ba61-2ebdf6c4edad
@belapsed generate_initial_config_optimized(dims, U)  # time measured in seconds

# ╔═╡ 6f141d37-e16e-4a13-8bbb-dcd1bf756cba
md"""
# Relaxation
"""

# ╔═╡ afab883b-50d7-4049-9cc7-a58d20bbeee2
function trace_q2(Q)
	return Q[1] * Q[1] + Q[4] * Q[4] + Q[6] * Q[6] + 2 * (Q[2] * Q[2] + Q[3] * Q[3] + Q[5] * Q[5])
end

# ╔═╡ ef1ad5dc-0384-4bd8-9c6d-52740dcbe95a
function trace_q3(Q)
	return (
		Q[1] * Q[1] * Q[1] + Q[4] * Q[4] * Q[4] +Q[6] * Q[6] * Q[6] +
		6 * Q[2] * Q[3] * Q[5] +
		3 * Q[1] * (Q[2] * Q[2] + Q[3] * Q[3]) +
		3 * Q[4] * (Q[2] * Q[2] + Q[5] * Q[5]) +
		3 * Q[6] * (Q[5] * Q[5] + Q[3] * Q[3])
	)
end

# ╔═╡ e50cb03b-578c-4109-9df1-c38e40f69445
function free_energy(A, U, Q, V)
	fLdG = zero(eltype(Q))
	for q in eachcol(Q)
		trQ² = trace_q2(q)
		trQ³ = trace_q3(q)
		fLdG += (1 - U / 3) * trQ² / 2 - U / 3 * trQ³ + U / 4 * trQ²^2
	end
	return A * V * fLdG
end

# ╔═╡ 2c1b9826-e079-4e1b-8974-a3f897f101b6
A = 1e6

# ╔═╡ 1c9669ea-6e05-4581-8d76-d38a5c37baf4
V = 1

# ╔═╡ ef10a07f-9dee-40c7-99b1-237408b795bc
free_energy(A, U, Q₀, V)

# ╔═╡ d151eb83-e396-4c95-9785-839bc1dfd1d3
function volterra(A, U, Q)
	third = 1 // 3
	QLdG = similar(Q)
	QQ = MVector{6, eltype(Q)}(undef)
	δ = SVector(1, 0, 0, 1, 0, 1)
	for (i, Qin) in enumerate(eachcol(Q))
		QQ[1] = Qin[1]*Qin[1]+Qin[2]*Qin[2]+Qin[3]*Qin[3]
		QQ[2] = Qin[1]*Qin[2]+Qin[2]*Qin[4]+Qin[3]*Qin[5]
		QQ[3] = Qin[1]*Qin[3]+Qin[2]*Qin[5]+Qin[3]*Qin[6]
		QQ[4] = Qin[2]*Qin[2]+Qin[4]*Qin[4]+Qin[5]*Qin[5]
		QQ[5] = Qin[2]*Qin[3]+Qin[4]*Qin[5]+Qin[5]*Qin[6]
		QQ[6] = Qin[3]*Qin[3]+Qin[5]*Qin[5]+Qin[6]*Qin[6]
		trQQ = trace_q2(Qin)
		QLdG[:, i] .= (1-U*third) .* Qin .- U .* (QQ .- trQQ .* (Qin .+ δ .* third))
	end
	return QLdG
end

# ╔═╡ a0d8d4fe-d88c-4644-8ece-062294c95e0f
volterra(A, U, Q₀)

# ╔═╡ f335cc3e-4a07-49b7-9b7c-4a6351480b68
function f(q, p, t)
	# Not yet implemented
	return nothing
end

# ╔═╡ f5a20983-d02c-429c-abcc-72295972fd1e



# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[compat]
BenchmarkTools = "~1.2.0"
DifferentialEquations = "~6.20.0"
StaticArrays = "~1.2.13"
"""

# ╔═╡ Cell order:
# ╟─038a20bf-6100-4798-be56-fe2f34defefc
# ╠═76bc042c-40c0-11ec-1f62-1fece22a67b5
# ╠═f02876aa-a5f4-4417-83d9-7c24ed90643f
# ╠═2d85c3b1-401b-4719-9f19-3db2d8ff09c2
# ╠═7a54874b-62d0-43ef-9abd-3882c9e3e3c1
# ╠═5a851a55-6004-4752-9c3d-c13dbce5f0c6
# ╠═ff39156e-0fc4-477d-a6ce-68358a2c1c6d
# ╟─a1d6582f-2416-4b80-ad9f-e3fc78d94d35
# ╠═f994ee31-f226-460c-a2f6-a27545aec18e
# ╠═17b6628c-09df-4cb7-8b0a-512de2db9d68
# ╟─72c9f4c0-cee6-4df0-9d74-46bb68c810de
# ╠═cb4da596-09c3-4089-bee3-bdc6158f8888
# ╠═9cbfff8b-2155-4c1f-b6e3-7df41abc836f
# ╠═9bfb745c-c3e0-4d17-92c4-6d3a25f76401
# ╠═1cbe26b5-33d8-4c83-be78-d182a162bccf
# ╟─c906f201-a445-4122-be5f-48f5d9f149f7
# ╠═240b46ee-f8bd-4d72-9cfe-246d4053993e
# ╠═eb6c5dd7-ea00-4a69-ba61-2ebdf6c4edad
# ╟─6f141d37-e16e-4a13-8bbb-dcd1bf756cba
# ╠═afab883b-50d7-4049-9cc7-a58d20bbeee2
# ╠═ef1ad5dc-0384-4bd8-9c6d-52740dcbe95a
# ╠═e50cb03b-578c-4109-9df1-c38e40f69445
# ╠═2c1b9826-e079-4e1b-8974-a3f897f101b6
# ╠═1c9669ea-6e05-4581-8d76-d38a5c37baf4
# ╠═ef10a07f-9dee-40c7-99b1-237408b795bc
# ╠═d151eb83-e396-4c95-9785-839bc1dfd1d3
# ╠═a0d8d4fe-d88c-4644-8ece-062294c95e0f
# ╠═f335cc3e-4a07-49b7-9b7c-4a6351480b68
# ╠═f5a20983-d02c-429c-abcc-72295972fd1e
# ╟─00000000-0000-0000-0000-000000000001
