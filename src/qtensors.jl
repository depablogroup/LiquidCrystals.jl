using StaticArrays


"""
Map from the linear indices (that is, how they are expected in memory)
to the internal indices of our 6-element vector.
"""
const Q_LINEAR_INDICES = (1, 2, 3, 2, 4, 5, 3, 5, 6)


@doc raw"""
Encodes the local value of the $\bar{Q}$ tensor field at a given point.

    QLocal(q₁, q₂, q₃, q₄, q₅, q₆)

Provides a convenience constructor for `QLocal` from the
six lower triangular elements of the matrix.
"""
struct QLocal{T} <: StaticMatrix{3, 3, T}
    data::SVector{6, T}

    function QLocal(data::SVector{6, T}) where {T}
        return new{T}(data)
    end

    function QLocal(data::MVector{6})
        return QLocal(SVector(data))
    end

    function QLocal(q₁, q₂, q₃, q₄, q₅, q₆)
        return QLocal(SVector(q₁, q₂, q₃, q₄, q₅, q₆))
    end

    function QLocal{T}(data::Tuple{Tuple{NTuple{9, T}}}) where {T}
        q = data[1][1]
        return QLocal(SVector(q[1], q[2], q[3], q[5], q[6], q[9]))
    end
end

function Base.getindex(q::QLocal, i::Int)
    return q.data[Q_LINEAR_INDICES[i]]
end

function Base.zero(::QLocal{T}) where {T}
    return QLocal(zero(SVector{6, T}))
end

function Base.:*(Q₁::QLocal, Q₂::QLocal)
    v₁ = Q₁.data
    v₂ = Q₂.data
    @inbounds begin
        q₁ = v₁[1] * v₂[1] + v₁[2] * v₂[2] + v₁[3] * v₂[3]
        q₂ = v₁[1] * v₂[2] + v₁[2] * v₂[4] + v₁[3] * v₂[5]
        q₃ = v₁[1] * v₂[3] + v₁[2] * v₂[5] + v₁[3] * v₂[6]
        q₄ = v₁[2] * v₂[2] + v₁[4] * v₂[4] + v₁[5] * v₂[5]
        q₅ = v₁[2] * v₂[3] + v₁[4] * v₂[5] + v₁[5] * v₂[6]
        q₆ = v₁[3] * v₂[3] + v₁[5] * v₂[5] + v₁[6] * v₂[6]
    end
    return QLocal(q₁, q₂, q₃, q₄, q₅, q₆)
end

"""    tr_sq(::QLocal)

Trace of the square of a QLocal tensor
"""
function tr_sq(q::QLocal)
    v = q.data
    return  (v[1] * v[1] + v[4] * v[4] + v[6] * v[6]) +
        2 * (v[2] * v[2] + v[3] * v[3] + v[5] * v[5])
end

"""    tr_sq(::QLocal)

Trace of the cube of a QLocal tensor
"""
function tr_cb(q::QLocal)
    v = q.data
    return (
        v[1] * v[1] * v[1] +
        v[4] * v[4] * v[4] +
        v[6] * v[6] * v[6] +
        6 * v[2] * v[3] * v[5] +
        3 * v[1] * (v[2] * v[2] + v[3] * v[3]) +
        3 * v[4] * (v[2] * v[2] + v[5] * v[5]) +
        3 * v[6] * (v[5] * v[5] + v[3] * v[3])
    )
end

"""    tr_sq_cb(::QLocal)

Simultaneously computes the traces of the square and the cube
of a QLocal tensor.
"""
function tr_sq_cb(q::QLocal)
    v = q.data

    @inbounds begin
        v₁ = v[1]
        v₂ = v[2]
        v₃ = v[3]
        v₄ = v[4]
        v₅ = v[5]
        v₆ = v[6]
    end

    v₁² = v₁ * v₁
    v₂² = v₂ * v₂
    v₃² = v₃ * v₃
    v₄² = v₄ * v₄
    v₅² = v₅ * v₅
    v₆² = v₆ * v₆

    trv² = v₁² + v₄² + v₆² + 2 * (v₂² + v₃² + v₅²)
    trv³ = (
        v₁ * v₁² + v₄ * v₄² + v₆ * v₆²
        + 6 * v₂ * v₃ * v₅
        + 3 * (v₁ * (v₂² + v₃²) + v₄ * (v₂² + v₅²) + v₆ * (v₃² + v₅²))
    )

    return trv², trv³
end
