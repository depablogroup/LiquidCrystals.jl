module LiquidCrystals


using LinearAlgebra
using StaticArrays


export QLocal, QLocal2, QLocal3
export CenteredDifference, DirichletBC, NeumannBC, PeriodicBC
export TwoD, generate_initial_config


include("qtensors.jl")
include("initialization.jl")
include("outputs.jl")
include("finitediff/operators.jl")


function free_energy(A, U, Q, V)
    fLdG = zero(typeof(U))

    @simd for q in Q
        trQ², trQ³ = tr_sq_cb(q)
        fLdG += (1 - U / 3) * trQ² / 2 - U / 3 * trQ³ + U / 4 * trQ²^2
    end

    return A * V * fLdG
end

function volterra(A, U, Q::AbstractArray{T}) where {T <: QLocal}
    QLdG = similar(Q)
    # The following is equivalent to the lower triangular part
    # of one third of the unit matrix
    δ₃ = SVector(1, 0, 0, 1, 0, 1) // 3
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

function volterra(A, U, Q::AbstractArray{T}) where {T <: SVector}
    return volterra(A, U, reinterpret(qtype(T), Q))
end


end  # module LiquidCrystals
