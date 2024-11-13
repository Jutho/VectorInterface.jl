"""
    MinimalVec{M,V<:AbstractVector}
    MinimalVec{M}(vec::V) where {M,V<:AbstractVector}

Wraps a vector of type `V<:AbstractVector` in such a way that the wrapper only supports the
minimal interface put forward by VectorInterface.jl. The type parameter `M` can take the
values `true` or `false` and determines whether the vector behaves as a mutable vector that
supports in-place operations (`M == true`) or whether it behaves as an immutable or
static vector (`M == false`).

This wrapper can be used to test whether an algorithm is implemented using only the minimal
interface of VectorInterface.jl, without relying on other methods that would for example
be available for `AbstractVector` or `AbstractArray`.

To unwrap the contents of a `v::MinimalVec` instance, the field access `v.vec` can be used.

See also [`MinimalMVec`](@ref) and [`MinimalSVec`](@ref) for convenience constructors.
"""
struct MinimalVec{M,V<:AbstractVector}
    vec::V
    function MinimalVec{M,V}(vec::V) where {M,V}
        M isa Bool || throw(ArgumentError("first type parameter must be `true` or `false`"))
        return new{M,V}(vec)
    end
    MinimalVec{M}(vec::V) where {M,V} = MinimalVec{M,V}(vec)
end
"""
    const MinimalMVec = MinimalVec{true}
    MinimalMVec(v::AbstractVector)

Type alias for `MinimalVec{true}`, representing a vector wrapper that implements the minimal
interface of VectorInterface.jl, including in-place operations (!-methods).

See also [`MinimalVec`](@ref) and [`MinimalSVec`](@ref).
"""
const MinimalMVec{V} = MinimalVec{true,V}
"""
    const MinimalSVec = MinimalVec{false}
    MinimalSVec(v::AbstractVector)

Type alias for `MinimalVec{false}`, representing a vector wrapper that implements the
minimal interface of VectorInterface.jl, excluding in-place operations (!-methods).

See also [`MinimalVec`](@ref) and [`MinimalMVec`](@ref).
"""
const MinimalSVec{V} = MinimalVec{false,V}

MinimalMVec(v::AbstractVector) = MinimalVec{true}(v)
MinimalSVec(v::AbstractVector) = MinimalVec{false}(v)

_ismutable(::Type{MinimalVec{M,V}}) where {V,M} = M
_ismutable(v::MinimalVec) = _ismutable(typeof(v))

scalartype(::Type{<:MinimalVec{M,V}}) where {M,V} = scalartype(V)

function zerovector(v::MinimalVec, S::Type{<:Number})
    return MinimalVec{_ismutable(v)}(zerovector(v.vec, S))
end
function zerovector!(v::MinimalMVec{V}) where {V}
    zerovector!(v.vec)
    return v
end
zerovector!!(v::MinimalVec) = _ismutable(v) ? zerovector!(v) : zerovector(v)

function scale(v::MinimalVec, α::Number)
    return MinimalVec{_ismutable(v)}(scale(v.vec, α))
end
function scale!(v::MinimalMVec{V}, α::Number) where {V}
    scale!(v.vec, α)
    return v
end
function scale!!(v::MinimalVec, α::Number)
    if _ismutable(v)
        w = scale!!(v.vec, α)
        return w === v.vec ? v : MinimalMVec(w)
    else
        return scale(v, α)
    end
end
function scale!(w::MinimalMVec{V}, v::MinimalMVec{W}, α::Number) where {V,W}
    scale!(w.vec, v.vec, α)
    return w
end
function scale!!(w::MinimalVec, v::MinimalVec, α::Number)
    if _ismutable(w)
        wvec = scale!!(w.vec, v.vec, α)
        return wvec === w.vec ? w : MinimalMVec(wvec)
    else
        return scale(v, α * one(scalartype(w)))
    end
end

function add(y::MinimalVec, x::MinimalVec, α::Number, β::Number)
    return MinimalVec{_ismutable(y)}(add(y.vec, x.vec, α, β))
end
function add!(y::MinimalMVec{W}, x::MinimalMVec{V}, α::Number, β::Number) where {W,V}
    add!(y.vec, x.vec, α, β)
    return y
end
function add!!(y::MinimalVec, x::MinimalVec, α::Number, β::Number)
    if _ismutable(y)
        yvec = add!!(y.vec, x.vec, α, β)
        return yvec === y.vec ? y : MinimalMVec(yvec)
    else
        return add(y, x, α, β)
    end
end

inner(x::MinimalVec, y::MinimalVec) = inner(x.vec, y.vec)
LinearAlgebra.norm(x::MinimalVec) = LinearAlgebra.norm(x.vec)
