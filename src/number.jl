# Vector interface implementation for subtypes of `Number`
###########################################################

# scalartype
#------------
scalartype(::Type{T}) where {T<:Number} = T

# zerovector & zerovector!!
#---------------------------
zerovector(x::Number, ::Type{S} = scalartype(x)) where {S<:Number} = zero(S)
zerovector!!(x::Number, ::Type{S} = scalartype(x)) where {S<:Number} = zero(S)

# scale & scale!!
#-----------------
# note: the following is required to make scale(NaN, 0) = 0
scale(x::Number, α::Number) = ifelse(iszero(α), zero(x) * α, x * α)
scale!!(x::Number, α::Number) = scale(x, α)
scale!!(y::Number, x::Number, α::Number) = scale(x, α)

# add & add!!
#-------------
# note: the following are required to make zero coefficients kill NaN values in x or y
add(y::Number, x::Number) = y + x
add(y::Number, x::Number, α::_One) = add(y, x)
function add(y::Number, x::Number, α::Number)
    return ifelse(iszero(α), muladd(zero(x), α, y), muladd(x, α, y))
end
add(y::Number, x::Number, α::_One, β::_One) = add(y, x)
function add(y::Number, x::Number, α::Number, β::_One)
    return ifelse(iszero(α), muladd(zero(x), α, y), muladd(x, α, y))
end
function add(y::Number, x::Number, α::_One, β::Number)
    return ifelse(iszero(β), muladd(zero(y), β, x), muladd(y, β, x))
end
add(y::Number, x::Number, α::Number, β::Number) = add(scale(y, β), x, α)

add!!(y::Number, x::Number) = add(y, x)
add!!(y::Number, x::Number, α::ONumber) = add(y, x, α)
add!!(y::Number, x::Number, α::ONumber, β::ONumber) = add(y, x, α, β)

# inner
#-------
inner(x::Number, y::Number) = conj(x)*y
