# Vector interface implementation: general fallbacks?
#######################################################
# Are these desirable; should they exist?
# They might lead to plenty of invalidations?
# They do not enforce any compatibility between the different vector types

# scalartype
#------------
function scalartype(T::Type)
    error("No scalar type is defined for type $T")
end
# should this try to use `eltype` instead? e.g. scalartype(T) = scalartype(eltype(T))

# zerovector & zerovector!!
#---------------------------
zerovector(x) = zero(x)
zerovector(x, ::Type{S}) where {S<:Number} = zero(x)*zero(S)

zerovector!!(x) = zerovector(x)
zerovector!!(x, ::Type{S}) where {S<:Number} = zerovector(x, S)

# scale, scale! & scale!!
#-------------------------
scale(x, α::Number) = x * α

scale!!(x, α::Number) = scale(x, α)
scale!!(y, x, α::Number) = scale(x, α)

scale!(x, α::Number) = (LinearAlgebra.rmul!(x, α); return x)
scale!(y, x, α::Number) = (LinearAlgebra.mul!(y, x, α); return y)

# add, add! & add!!
#-------------------
add(y, x) = y + x
add(y, x, α::Number) = add(y, scale(x, α))
add(y, x, α::_One) = add(y, x)
add(y, x, α::_One, β::_One) = add(y, x)
add(y, x, α::Number, β::_One) = add(y, scale(x, α))
add(y, x, α::Number, β::Number) = add(scale(y, β), scale(x, α))

add!!(y, x) = add(y, x)
add!!(y, x, α::Number) = add(y, x, α)
add!!(y, x, α::Number, β::Number) = add(y, x, α, β)

add!(y, x, α::Number = true) = (LinearAlgebra.axpy!(α, x, y); return y)
add!(y, x, α::Number, β::Number) = (LinearAlgebra.axpby!(α, x, β, y); return y)

# inner
#-------
inner(x, y) = LinearAlgebra.dot(x, y)
# impose stricter `inner` behaviour for mixed Base type arguments?
inner(x::Union{Number, Tuple, AbstractArray}, y::Union{Number, Tuple, AbstractArray}) =
    error("No inner product between vector of type $(typeof(x)) and of type $(typeof(y))")
