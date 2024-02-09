module VectorInterfaceMultivariatePolynomialsExt

import VectorInterface
using MultivariatePolynomials

# union of all abstract types that can be crudely supported?
const PolyTypes = Union{<:AbstractPolynomialLike, <:AbstractTermLike, <:AbstractMonomialLike}

# not clear if this is really the true `scalartype` we want
VectorInterface.scalartype(T::Type{<:PolyTypes}) = T

function VectorInterface.add!!(w::PolyTypes, v::PolyTypes, α::Number, β::Number)
    return w * β + v * α
end

VectorInterface.scale!!(v::PolyTypes, α::Number) = v * α

end