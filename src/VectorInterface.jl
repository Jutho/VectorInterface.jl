module VectorInterface

using LinearAlgebra: LinearAlgebra, norm, BlasFloat
using Base: promote_type

# General interface: export
export scalartype
export zerovector, zerovector!, zerovector!!
export scale, scale!, scale!!
export add, add!, add!!
export inner, norm

include("interface.jl")

# Auxiliary type for representing constant 1 in vector addition / linear combinations
struct _One
end
const _one = _One()
Base.convert(T::Type{<:Number}, ::_One) = one(T)
Base.promote_rule(::Type{_One}, ::Type{T}) where {T<:Number} = T

const ONumber = Union{_One,Number}

# Specific implementations for Base types / type hierarchies
include("number.jl")
include("abstractarray.jl")
include("tuple.jl")
include("namedtuple.jl")

# General fallback implementation: comes with warning and some overhead
include("fallbacks.jl")

end
