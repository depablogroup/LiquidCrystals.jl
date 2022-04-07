"""
    max_eigen(M)

Returns the value of the maximum eigenvalue of `M` and its correspoding eigenvector.
"""
function max_eigen(M)
    vals, vecs = eigen(M)
    # The results of `eigen` are ordered from lower to higher eigenvalue,
    # so we take the last one.
    return vals[end], vecs[:, end]
end

"""
    s_and_directors(Qs)

Returns the vector of nematic order parameters and the vector of directors
for each local value of the discretization `Qs` of the Q-field.
"""
function s_and_directors(Qs::AbstractArray{QT}) where {QT <: QLocal}
    d = size(QT, 1)  # dimensionality of the directors
    c = d // (d - 1)

    n = length(Qs)
    ST = stype(QT)  # `QLocal` internal `SVector` type
    Ss = Vector{eltype(ST)}(undef, n)
    n̂s = Vector{ST}(undef, n)

    @inbounds @simd for i in eachindex(Qs)
        λ, n̂ = max_eigen(Qs[i])
        Ss[i] = c * λ
        n̂s[i] = n̂
    end

    return Ss, n̂s
end

function s_and_directors(A::AbstractArray{T}) where {T <: SVector}
    # If we get `SVector`s, first reinterpret as `QLocal`s.
    return s_and_directors(reinterpret(reshape, qtype(T), A))
end
