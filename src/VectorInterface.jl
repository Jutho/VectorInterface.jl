module VectorInterface

import LinearAlgebra
import LinearAlgebra: norm
import LinearAlgebra: BlasFloat
import Base: promote_type, promote_op

export scalartype
export zerovector, zerovector!, zerovector!!
export scale, scale!, scale!!
export add, add!, add!
export inner, norm

"""
    scalartype(x)

Returns the type of scalar over which the vector-like object `x` behaves as a vector, e.g.
the type of scalars with which `x` could be scaled in-place.
"""
scalartype(x) = scalartype(typeof(x))
scalartype(::Type{T}) where {T<:Number} = T
scalartype(::Type{A}) where {T, A<:AbstractArray{T}} = scalartype(T)
function scalartype(T::Type)
    error("No scalar type is defined for type $T")
end

## zerovector
"""
    zerovector(x, [S::Type{<:Number} = scalartype(x)])

Returns a zero vector in the vector space of `x`. Optionally, a modified scalar type `S` for
the resulting zero vector can be specified.

Also see: [`zerovector!`](@ref), [`zerovector!!`](@ref)
"""
zerovector(x) = zerovector(x, scalartype(x))
zerovector(x::Number, ::Type{S}) where {S<:Number} = zero(S)
zerovector(x::AbstractArray, ::Type{S}) where {S<:Number} = zerovector.(x, S)

"""
    zerovector!(x, [S::Type{<:Number} = scalartype(x)])

Modifies `x` in-place to become the zero vector. Optionally, a modified scalar type `S` can
be specified, but this can only work if `x` is a container of a non-concrete type.

Also see: [`zerovector`](@ref), [`zerovector!!`](@ref)
"""
function zerovector!(x::AbstractArray) where {S<:Number}
    try
        x .= zerovector!!.(x, S)
        return x
    catch
        error("Cannot make a zero vector with scalar type $S in-place in an object of type $(typeof(x))")
    end
end

"""
    zerovector!!(x, [S::Type{<:Number} = scalartype(x)])

Construct a zero vector in the vector space of `x`, thereby trying to overwrite and thus
recycle `x` when possible. Optionally, a modified scalar type `S` for the resulting zero
vector can be specified.

Also see: [`zerovector`](@ref), [`zerovector!`](@ref)
"""
zerovector!!(x::Number, ::Type{S} = scalartype(x)) where {S<:Number} = zero(S)
function zerovector!!(x::AbstractArray, ::Type{S} = scalartype(x)) where {S<:Number}
    try
        x .= zerovector!!.(x, S)
        return x
    catch
        return zerovector!!.(x, S)
    end
end

"""
    scale(x, α::Number)

Computes the new vector-like object obtained from scaling `x` with the scalar `α`.

For unknown types, `scale(x, α) falls back to `x * α`.

Also see: [`scale!`](@ref) and [`scale!!`](@ref)
"""
scale(x, α) = x * α
scale(x::AbstractArray, α) = scale.(x, α)

"""
    scale!(x, α::Number) -> x
    scale!(y, x, α::Number) -> y

Rescale `x` with the scalar coefficient `α`, thereby overwrite and thus recylcing the
contents of `x` (in the first form) or `y` (in the second form). This is only possible if
`x`, respectively `y` is mutable, and if the scalar types involed are compatible and can
be promoted and converted.

Also see: [`scale`](@ref) and [`scale!!`](@ref)
"""
function scale!(x::AbstractArray{T}, α::Number) where {T<:BlasFloat}
    LinearAlgebra.rmul!(x, convert(T, α))
    return x
end
function scale!(x::AbstractArray, α::Number)
    x .= scale!!.(x, α)
    return x
end
function scale!(y::AbstractArray, x::AbstractArray, α::Number)
    y .= scale!!.(y, x, α)
    return x
end

"""
    scale!!(x, α)
    scale!!(y, x, α)

Rescale `x` with the scalar coefficient `α`, thereby trying to overwrite and thus recylce
the contens of `x` (in the first form) or `y` (in the second form). When not possible
(because of type or size incompatibilities), a new object will be created to store the
result.

Also see: [`scale`](@ref) and [`scale!`](@ref)
"""
scale!!(x::Number, α::Number) = scale(x, α)
function scale!!(x::AbstractArray{T}, α::T) where {T<:BlasFloat}
    LinearAlgera.rmul!(x, α)
    return x
end
function scale!!(x::AbstractArray{T}, α::T) where {T<:Number}
    x .*= α
    return x
end
function scale!!(x::AbstractArray{T}, α::Number) where {T<:Number}
    if promote_type(T, typeof(α)) <: T
        return scale!!(x, convert(T, α))
    else
        return scale!!.(x, α)
    end
end
function scale!!(x::AbstractArray, α::Number)
    if promote_op(scale, eltype(x), typeof(α)) <: eltype(x)
        x .= scale!!.(x, α)
    else
        return scale!!.(x, α)
    end
end

scale!!(y::Number, x::Number, α::Number) = scale(x, α)
function scale!!(y::AbstractArray, x::AbstractArray, α::Number)
    try
        y .= scale!!.(y, x, α)
        return y
    catch
        return scale!!.(y, x, α)
    end
end

struct _One
end
const _one = _One()
Base.convert(T::Type{<:Number}, ::_One) = one(T)
Base.promote_rule(::Type{_One}, ::Type{T}) where {T<:Number} = T
const ONumber = Union{_One, Number}


"""
    add(y, [β::Number = 1], x, [α::Number = 1])

Add `y` and `x`, or more generally construct the linear combination `y * β + x * α`.

For unknown types, `add(y, x)` is implemented as `y + x`, and `add(y, β, x, α)` falls back
to `add(scale(y, β), scale(x, α))`.

See also: [`add!`](@ref) and [`add!!`](@ref)
"""
add(y, x) = x + y
add(y, x, α::Number) = add(y, scale(x, α))
add(y, x, α::_One) = add(y, x)
add(y, β::Number, x, α::Number) = add(scale(y, β), scale(x, α))
add(y, β::_One, x, α::Number) = add(y, scale(x, α))
add(y, β::_One, x, α::_One) = add(y, x)

add(y::Number, x::Number, α::Number) = muladd(x, α, y)
add(y::Number, β::_One, x::Number, α::Number) = muladd(x, α, y)
add(y::Number, β::Number, x::Number, α::_One) = muladd(y, β, x)
add(y::Number, β::Number, x::Number, α::Number) = muladd(x, α, β*y)

add(y::AbstractArray, x::AbstractArray, α::ONumber = _one) = add(y, _one, x, α)
function add(y::AbstractArray, β::ONumber, x::AbstractArray, α::ONumber)
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("Output axes $ay differ from input axes $ax"))
    return add.(y, β, x, α)
end

"""
    add!(y, [β::Number = 1], x, [α::Number = 1])

Add `y` and `x`, or more generally construct the linear combination `y * β + x * α`, storing
the result in `y`. This will error in case of incompatible scalar types or incommensurate sizes.

See also: [`add`](@ref) and [`add!!`](@ref)
"""
function add!(y::AbstractArray{T}, x::AbstractArray{T}, α::ONumber = _one) where {T<:BlasFloat}
    LinearAlgebra.axpy!(convert(T, α), x, y)
    return y
end
function add!(y::AbstractArray{T}, β::_One, x::AbstractArray{T}, α::ONumber) where {T<:BlasFloat}
    LinearAlgebra.axpy!(convert(T, α), x, y)
    return y
end
function add!(y::AbstractArray{T}, β::Number, x::AbstractArray{T}, α::ONumber) where {T<:BlasFloat}
    LinearAlgebra.axpby!(convert(T, α), x, convert(T, β), y)
    return y
end
add!(y::AbstractArray, x::AbstractArray, α::ONumber = _one) = add!(y, _one, x, α)
function add!(y::AbstractArray, β::ONumber, x::AbstractArray, α::ONumber)
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("Output axes $ay differ from input axes $ax"))
    y .= add!!.(y, β, x, α) # might error
    return y
end

"""
    add!!(y, [β::Number = 1], x, [α::Number = 1])

Add `y` and `x`, or more generally construct the linear combination `y * β + x * α`, thereby
trying to store the result in `y`. A new object will be created when this fails due to
incompatible scalar types or incommensurate sizes.

See also: [`add`](@ref) and [`add!`](@ref)
"""
add!!(y::Number, x::Number) = add(y, x)
add!!(y::Number, x::Number, α::ONumber) = add(y, x, α)
add!!(y::Number, β::ONumber, x::Number, α::ONumber) = add(y, β, x, α)

function add!!(y::AbstractArray{T}, x::AbstractArray{T}, α::ONumber = _one) where {T<:BlasFloat}
    if promote_type(T, typeof(α)) <: T
        y = add!(y, x, convert(T, α))
        return y
    else
        ax = axes(x)
        ay = axes(y)
        ax == ay || throw(DimensionMismatch("Output axes $ay differ from input axes $ax"))
        return add!!.(y, β, x, α)
    end
end
function add!!(y::AbstractArray{T}, β::ONumber, x::AbstractArray{T}, α::ONumber) where {T<:BlasFloat}
    if promote_type(T, typeof(α)) <: T && promote_type(T, typeof(β)) <: T
        y = add!(y, β, x, α)
        return y
    else
        ax = axes(x)
        ay = axes(y)
        ax == ay || throw(DimensionMismatch("Output axes $ay differ from input axes $ax"))
        return add!!.(y, β, x, α)
    end
end

add!!(y::AbstractArray, x::AbstractArray, α::ONumber = _one) = add!(y, _one, x, α)
function add!!(y::AbstractArray, β::ONumber, x::AbstractArray, α::ONumber)
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("Output axes $ay differ from input axes $ax"))
    try
        y .= add!!.(y, β, x, α)
        return y
    catch
        return add!!.(y, β, x, α)
    end
end

# Inner product
inner(x::Number, y::Number) = conj(x)*y
inner(x::AbstractArray{<:Number}, y::AbstractArray{<:Number}) = LinearAlgebra.dot(x, y)
function inner(x::AbstractArray, y::AbstractArray)
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("Non-matching axes $ax and $ay"))
    s = zero(scalartype(x)) * zero(scalartype(y))
    for I in eachindex(x)
        s += oftype(s, inner(x[I], y[I]))
    end
    return s
end

include("upfordebate.jl")

end
