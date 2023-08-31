module VectorInterface

using LinearAlgebra: LinearAlgebra, norm, BlasFloat
using Static
using Base: promote_type

# General interface: export
export scalartype
export zerovector, zerovector!, zerovector!!
export scale, scale!, scale!!
export add, add!, add!!
export inner, norm

include("interface.jl")

# Auxiliary type for representing constant 1 in vector addition / linear combinations
const _one = static(1)
const _One = typeof(_one)
const _zero = static(0)

# Specific implementations for Base types / type hierarchies
include("number.jl")
include("abstractarray.jl")
include("tuple.jl")
include("namedtuple.jl")

# General fallback implementation: comes with warning and some overhead
include("fallbacks.jl")

end
