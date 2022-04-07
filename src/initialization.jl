# using StaticArrays: MVector, StaticVector


abstract type DirectorDimensionality end
struct TwoD <: DirectorDimensionality end
struct ThreeD <: DirectorDimensionality end

qtype(::TwoD) = QLocal{2, Float64, 3}
qtype(::ThreeD) = QLocal{3, Float64, 6}

mtype(::TwoD) = MVector{2, Float64}
mtype(::ThreeD) = MVector{3, Float64}


"""
    generate_initial_config(U, shape)

Given a value of `U` and a shape, returns a vector of `QLocal`s of size `prod(shape)`
initialized from randomly oriented directors.
"""
generate_initial_config(U, shape) = generate_initial_config(U, shape, ThreeD())

function generate_initial_config(U, shape, D::DirectorDimensionality)
    N = prod(shape)
    S = 1 // 4 + 3 // 4 * sqrt(1 - 8 / (3 * U))

    # Allocate memory for the Q field
    Q = Vector{qtype(D)}(undef, N)
    n = mtype(D)(undef)

    # Write the Q field at each point of the grid
    for i in eachindex(Q)
        # Generate a random unit vector
        n .= rand.() .- 1 // 2
        n .= n ./ norm(n)
        Q[i] = alignment_tensor(n)
    end

    return Q
end

"""
    alignment_tensor(n::StaticVector)

Given a director `n::StaticVector{N}` where `N in (2, 3)`, returns the corresponding
value of the alignment tensor. It assumes `norm(n) == 1`.
"""
function alignment_tensor(n::StaticVector{2})
    q₁ = S * (n[1] * n[1] - 1 // 2)
    q₂ = S * (n[1] * n[2])
    return QLocal(q₁, q₂)
end

function alignment_tensor(n::StaticVector{3})
    q₁ = S * (n[1] * n[1] - 1 // 3)
    q₂ = S * (n[1] * n[2])
    q₃ = S * (n[1] * n[3])
    q₄ = S * (n[2] * n[2] - 1 // 3)
    q₅ = S * (n[2] * n[3])
    q₆ = S * (n[3] * n[3] - 1 // 3)
    return QLocal(q₁, q₂, q₃, q₄, q₅, q₆)
end
