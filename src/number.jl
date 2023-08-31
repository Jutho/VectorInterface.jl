# Vector interface implementation for subtypes of `Number`
###########################################################

# scalartype
#------------
scalartype(::Type{T}) where {T<:Number} = T

# zerovector & zerovector!!
#---------------------------
@inline zerovector(x::Number, ::Type{S}) where {S<:Number} = zero(S)
@inline zerovector!!(x::Number) = zero(x)

# scale & scale!!
#-----------------
@inline scale(x::Number, ::_One) = x
@inline scale(x::Number, α::Number) = (iszero(α) ? zero(x) * α : x * α) # note: required to make scale(NaN, 0) = 0
@inline scale!!(x::Number, α::Number) = scale(x, α)
@inline scale!!(y::Number, x::Number, α::Number) = scale(x, α)

# add & add!!
#-------------
@inline add(y::Number, x::Number) = y + x
@inline add(y::Number, x::Number, α::_One) = add(y, x)
@inline function add(y::Number, x::Number, α::Number)
    return iszero(α) ? muladd(zero(x), α, y) : muladd(x, α, y) # required to make zero coefficients kill NaN values in x or y
end
@inline add(y::Number, x::Number, α::Number, β::Number) = add(scale(y, β), x, α)

@inline add!!(y::Number, x::Number) = add(y, x)
@inline add!!(y::Number, x::Number, α::Number) = add(y, x, α)
@inline add!!(y::Number, x::Number, α::Number, β::Number) = add(y, x, α, β)

# inner
#-------
@inline inner(x::Number, y::Number) = conj(x) * y
