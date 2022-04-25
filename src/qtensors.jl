# using StaticArrays: MVector, SVector


"""
Map from the linear indices (that is, how they are expected in memory)
to the internal indices of the 3-element vector of `QLocal2{T}`.
"""
const Q2_LINEAR_INDICES = (1, 2, 2, 3)

"""
Map from the linear indices (that is, how they are expected in memory)
to the internal indices of the 6-element vector of `QLocal3{T}`.
"""
const Q3_LINEAR_INDICES = (1, 2, 3, 2, 4, 5, 3, 5, 6)


@doc raw"""
Encodes the local value of the $\bar{Q}$ tensor field.
"""
struct QLocal{N, T, M} <: StaticMatrix{N, N, T}
    data::SVector{M, T}

    QLocal{3, T, 6}(data::SVector{6, T}) where {T} = new{3, T, 6}(data)
    QLocal{2, T, 3}(data::SVector{3, T}) where {T} = new{2, T, 3}(data)
end

# Aliases for two and three dimensions
const QLocal3{T} = QLocal{3, T, 6}
const QLocal2{T} = QLocal{2, T, 3}

# Constructors for `QLocal`
QLocal(data::SVector) = qtype(data)(data)
QLocal(data::MVector) = QLocal(SVector(data))

"""
    QLocal(q₁, q₂, q₃)

Provides a convenience constructor for `QLocal2` from the
three lower triangular elements of the matrix.
"""
QLocal(q₁, q₂, q₃) = QLocal(SVector(q₁, q₂, q₃))
QLocal(q₁, q₂) = QLocal(SVector(q₁, q₂, -q₁))

"""
    QLocal(q₁, q₂, q₃, q₄, q₅, q₆)

Provides a convenience constructor for `QLocal3` from the
six lower triangular elements of the matrix.
"""
QLocal(q₁, q₂, q₃, q₄, q₅, q₆) = QLocal(SVector(q₁, q₂, q₃, q₄, q₅, q₆))

@inline QLocal(t::Tuple) = QLocal(t...)
QLocal(t::NTuple{4, Any}) = @inbounds QLocal(t[1], t[2], t[4])
QLocal(t::NTuple{9, Any}) = @inbounds QLocal(t[1], t[2], t[3], t[5], t[6], t[9])


# Methods for QLocal

"""
Maps the type of an `SVector` to the appropriate `QLocal` container type.
"""
function qtype end
qtype(::Type{SVector{3, T}}) where {T} = QLocal2{T}
qtype(::Type{SVector{6, T}}) where {T} = QLocal3{T}
qtype(::S) where {S <: SVector} = qtype(S)

"""
Maps the type of a `QLocal` to the type of its stored `SVector`.
"""
function stype end
stype(::Type{QLocal2{T}}) where {T} = SVector{3, T}
stype(::Type{QLocal3{T}}) where {T} = SVector{6, T}
stype(::Q) where {Q <: QLocal} = stype(Q)

function Base.getindex(q::QLocal2, i::Int)
    return q.data[Q2_LINEAR_INDICES[i]]
end

function Base.getindex(q::QLocal3, i::Int)
    return q.data[Q3_LINEAR_INDICES[i]]
end

Base.zero(q::QLocal) = QLocal(zero(q.data))

function Base.:*(Q₁::QLocal2, Q₂::QLocal2)
    v₁ = Q₁.data
    v₂ = Q₂.data
    @inbounds begin
        q₁ = v₁[1] * v₂[1] + v₁[2] * v₂[2]
        q₂ = v₁[2] * v₂[1] + v₁[3] * v₂[2]
        q₃ = v₁[2] * v₂[2] + v₁[3] * v₂[3]
    end
    return QLocal(q₁, q₂, q₃)
end

function Base.:*(Q₁::QLocal3, Q₂::QLocal3)
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
function tr_sq end

function tr_sq(q::QLocal2)
    v = q.data
    return @inbounds 2 * (v[1]^2 + v[2]^2)
end

function tr_sq(q::QLocal3)
    v = q.data
    return @inbounds (v[1]^2 + v[4]^2 + v[6]^2) + 2 * (v[2]^2 + v[3]^2 + v[5]^2)
end

"""    tr_sq(::QLocal)

Trace of the cube of a QLocal tensor
"""
function tr_cb end

tr_cb(::QLocal2{T}) where {T} = zero(T)

function tr_cb(q::QLocal3)
    v = q.data
    return @inbounds (
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
function tr_sq_cb end

function tr_sq_cb(q::QLocal2{T}) where {T}
    # For two dimensional tensors both values are independent
    return tr_sq(q), tr_cb(q)
end

function tr_sq_cb(q::QLocal3)
    v = q.data

    @inbounds begin
        v₁, v₂, v₃, v₄, v₅, v₆ = v[1], v[2], v[3], v[4], v[5], v[6]
    end

    v₁² = v₁ * v₁
    v₂² = v₂ * v₂
    v₃² = v₃ * v₃
    v₄² = v₄ * v₄
    v₅² = v₅ * v₅
    v₆² = v₆ * v₆

    trq² = v₁² + v₄² + v₆² + 2 * (v₂² + v₃² + v₅²)
    trq³ = (
        v₁ * v₁² + v₄ * v₄² + v₆ * v₆²
        + 6 * v₂ * v₃ * v₅
        + 3 * (v₁ * (v₂² + v₃²) + v₄ * (v₂² + v₅²) + v₆ * (v₃² + v₅²))
    )

    return trq², trq³
end
