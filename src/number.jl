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
scale(x::Number, α::Number) = x * α
scale!!(x::Number, α::Number) = scale(x, α)
scale!!(y::Number, x::Number, α::Number) = scale(x, α)

# add & add!!
#-------------
add(y::Number, x::Number) = y + x
add(y::Number, x::Number, α::_One) = add(y, x)
add(y::Number, x::Number, α::Number) = muladd(x, α, y)
add(y::Number, x::Number, α::_One, β::_One) = add(y, x)
add(y::Number, x::Number, α::Number, β::_One) = muladd(x, α, y)
add(y::Number, x::Number, α::_One, β::Number) = muladd(y, β, x)
add(y::Number, x::Number, α::Number, β::Number) = muladd(x, α, β*y)

add!!(y::Number, x::Number) = add(y, x)
add!!(y::Number, x::Number, α::ONumber) = add(y, x, α)
add!!(y::Number, x::Number, α::ONumber, β::ONumber) = add(y, x, α, β)

# inner
#-------
inner(x::Number, y::Number) = conj(x)*y
