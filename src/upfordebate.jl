# Supporting Tuples as vectors (as long as they have a homogeneous scalar type)?

scalartype(::Type{Tuple{}}) = error("No scalar type is defined for empty tuple")
scalartype(::Type{Tuple{T}}) where {T} = scalartype(T)
scalartype(::Type{NTuple{N,T}}) where {N,T} = scalartype(T)
function scalartype(::Type{TT}) where {TT<:Tuple}
    S = scalartype(Base.tuple_type_head(TT))
    S2 = scalartype(Base.tuple_type_tail(TT))
    S == S2 || error("No scalar type is defined for type $TT")
    return S
end

function zerovector(x::Tuple, ::Type{S}) where {S<:Number}
    y = zerovector.(x, S)
    return y
end
function zerovector!!(x::Tuple{Any}, ::Type{S} = scalartype(x)) where {S<:Number}
    return (zerovector!!(x[1], S),)
end
function zerovector!!(x::Tuple, ::Type{S} = scalartype(x)) where {S<:Number}
    return (zerovector!!(x[1], S), zerovector!!(Base.tail(x), S)...)
end

scale(x::Tuple, α) = scale.(x, α)
scale!!(x::Tuple, α::Number) = scale!!.(x, α)
function scale!!(y::Tuple, x::Tuple, α::Number)
    if length(y) == length(x)
        return scale!!.(y, x, α)
    else
        return scale(x, α)
    end
end

add(y::Tuple, x::Tuple, α::ONumber = _one) = add(y, _one, x, α)
function add(y::Tuple, β::ONumber, x::Tuple, α::ONumber)
    lx = length(x)
    ly = length(y)
    lx == ly || throw(DimensionMismatch("Output tuple length $ly differs from input tuple length $lx"))
    return add.(y, β, x, α)
end

add!!(y::Tuple, x::Tuple, α::ONumber = _one) = add!!(y, _one, x, α)
function add!!(y::Tuple, β::ONumber, x::Tuple, α::ONumber)
    lx = length(x)
    ly = length(y)
    lx == ly || throw(DimensionMismatch("Output tuple length $ly differs from input tuple length $lx"))
    return add!!.(y, β, x, α)
end

function inner(x::Tuple, y::Tuple)
    lx = length(x)
    ly = length(y)
    lx == ly || throw(DimensionMismatch("Non-matching tuple lengths $lx and $ly"))
    s = zero(scalartype(x)) * zero(scalartype(y))
    for i = 1:lx
        s += oftype(s, inner(x[i], y[i]))
    end
    return s
end

# general fallbacks ?
scale!(x, α::Number) = (LinearAlgebra.rmul!(x, α); return x)
scale!(y, x, α::Number) = (LinearAlgebra.mul!(y, x, α); return y)

scale!!(x, α::Number) = scale(x, α)
scale!!(y, x, α) = scale(x, α)

add!(y, x, α::Number = true) = (LinearAlgebra.axpy!(α, x, y); return y)
add!(y, β::Number, x, α::Number) = (LinearAlgebra.axpby!(α, x, β, y); return y)

add!!(y, x) = add(y, x)
add!!(y, x, α::ONumber) = add(y, x, α)
add!!(y, β::ONumber, x, α::ONumber) = add(y, β, x, α)

inner(x, y) = LinearAlgebra.dot(x, y)
# impose stricter `inner` behaviour for mixed Base type arguments?
inner(x::Union{Number, Tuple, AbstractArray}, y::Union{Number, Tuple, AbstractArray}) =
    error("No inner product between vector of type $(typeof(x)) and of type $(typeof(y))")
