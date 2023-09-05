# Vector interface implementation for subtypes of `AbstractArray`
##################################################################
const BLASVector{T<:BlasFloat} = Union{DenseArray{T},StridedVector{T}}

# scalartype
#------------
scalartype(::Type{A}) where {T,A<:AbstractArray{T}} = scalartype(T) # recursive

# zerovector & zerovector!!
#---------------------------
function zerovector(x::AbstractArray, ::Type{S}) where {S<:Number}
    return zerovector.(x, S)
end

function zerovector!(x::AbstractArray{<:Number})
    return fill!(x, zero(scalartype(x)))
end
function zerovector!(x::AbstractArray)
    T = eltype(x)
    for I in eachindex(x)
        if isbitstype(T) || isassigned(x, I)
            x[I] = zerovector!!(x[I])
        else
            x[I] = zero(eltype(x))
        end
    end
    return x
end

zerovector!!(x::AbstractArray) = zerovector!(x)

# scale, scale! & scale!!
#-------------------------
scale(x::AbstractArray, α::Number) = scale.(x, (α,))

function scale!(x::AbstractArray, α::Number)
    (α === _one) && return x
    x .= scale!!.(x, (α,))
    return x
end
function scale!(y::AbstractArray, x::AbstractArray, α::Number)
    y .= scale!!.(y, x, (α,))
    return y
end

function scale!!(x::AbstractArray, α::Number)
    (α === _one) && return x
    if Base.promote_op(scale, scalartype(x), typeof(α)) <: scalartype(x)
        return scale!(x, α)
    else
        return scale!!.(x, (α,))
    end
end
function scale!!(y::AbstractArray, x::AbstractArray, α::Number)
    if Base.promote_op(scale, scalartype(x), typeof(α)) <: scalartype(y)
        return scale!(y, x, α)
    else
        return scale!!.(y, x, (α,))
    end
end

# add, add! & add!!
#-------------------
function add(y::AbstractArray, x::AbstractArray, α::Number=_one, β::Number=_one)
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("Output axes $ay differ from input axes $ax"))
    return add.(y, x, (α,), (β,))
end

# Special case: simple numerical arrays with BLAS-compatible floating point type
function add!(y::BLASVector{T}, x::BLASVector{T},
              α::Number=_one, β::Number=_one) where {T<:BlasFloat}
    if β === _one
        LinearAlgebra.axpy!(convert(T, α), x, y)
    else
        LinearAlgebra.axpby!(convert(T, α), x, convert(T, β), y)
    end
    return y
end
# General case:
function add!(y::AbstractArray, x::AbstractArray, α::Number=_one, β::Number=_one)
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("Output axes $ay differ from input axes $ax"))
    y .= add!!.(y, x, (α,), (β,)) # might error
    return y
end

function add!!(y::AbstractArray, x::AbstractArray, α::Number=_one, β::Number=_one)
    if Base.promote_op(add, scalartype(y), scalartype(x), scalartype(α), scalartype(β)) <:
       scalartype(y)
        return add!(y, x, α, β)
    else
        ax = axes(x)
        ay = axes(y)
        ax == ay || throw(DimensionMismatch("Output axes $ay differ from input axes $ax"))
        return add!!.(y, x, (α,), (β,))
    end
end

# inner
#-------
function inner(x::BLASVector{T}, y::BLASVector{T}) where {T<:BlasFloat}
    return LinearAlgebra.dot(x, y)
end
function inner(x::AbstractArray, y::AbstractArray)
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("Non-matching axes $ax and $ay"))
    T = Base.promote_op(inner, scalartype(x), scalartype(y))
    s::T = zero(T)
    for I in eachindex(x)
        s += inner(x[I], y[I])
    end
    return s
end
