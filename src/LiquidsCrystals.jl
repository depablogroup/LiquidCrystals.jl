module LiquidsCrystals


export CenteredDifference, DirichletBC, NeumannBC, PeriodicBC, build_caches, apply_BCs
export QLocal, tr_sq_cb
export generate_initial_config, free_energy, volterra


include("qtensors.jl")
include("finitediffops.jl")


function generate_initial_config(dims, U)
	N = prod(dims)
	S = 1 // 4 + 3 // 4 * sqrt(1 - 8 / (3 * U))

	# Allocate memory for the Q field
	Q = Vector{QLocal{Float64}}(undef, N)
	# MVector below comes from the StaticArrays package
	n = MVector{3, Float64}(undef)

	# Write the Q field at each point of the grid
	for i in eachindex(Q)
		# generate a random unit vector
		n .= rand.() .- 1 // 2
		n .= n ./ norm(n)

		q₁ = S * (n[1] * n[1] - 1 // 3)
		q₂ = S * (n[1] * n[2])
		q₃ = S * (n[1] * n[3])
		q₄ = S * (n[2] * n[2] - 1 // 3)
		q₅ = S * (n[2] * n[3])
		q₆ = S * (n[3] * n[3] - 1 // 3)

		Q[i] = QLocal(q₁, q₂, q₃, q₄, q₅, q₆)
	end

	return Q
end

function free_energy(A, U, Q, V)
	fLdG = zero(typeof(U))

	@simd for q in Q
		trQ², trQ³ = tr_sq_cb(q)
		fLdG += (1 - U / 3) * trQ² / 2 - U / 3 * trQ³ + U / 4 * trQ²^2
	end

	return A * V * fLdG
end

function volterra(A, U, Q::AbstractArray{T}) where {T}
	QLdG = similar(Q)
	# The following is equivalent to the lower triangular part
	# of one third of the unit matrix
	δ₃ = SVector(1, 0, 0, 1, 0, 1) / 3
	# Allocate one vector for partial computations
	#q = MVector{6, eltype(T)}(undef)

	@inbounds for (i, Qᵢ) in enumerate(Q)
		Qᵢ² = Qᵢ * Qᵢ
		# Extract the vector representations to work with them here
		vᵢ = Qᵢ.data
		vᵢ² = Qᵢ².data
		q = (1 - U / 3) .* vᵢ .- U .* (vᵢ² .- tr(Qᵢ²) .* (vᵢ .+ δ₃))
		# `q` above is a MVector so we need to convert back to QLocal
		QLdG[i] = QLocal(A * q)
	end

	return QLdG
end


end  # module
