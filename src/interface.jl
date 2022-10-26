# Vector Interface
#------------------
# each of the following methods need to be implemented by custom types

"""
    scalartype(x)

Returns the type of scalar over which the vector-like object `x` behaves as a vector, e.g.
the type of scalars with which `x` could be scaled in-place.
"""
function scalartype end
scalartype(x) = scalartype(typeof(x))
# implementation should be in type domain

## zerovector
"""
    zerovector(x, [S::Type{<:Number} = scalartype(x)])

Returns a zero vector in the vector space of `x`. Optionally, a modified scalar type `S` for
the resulting zero vector can be specified.

Also see: [`zerovector!`](@ref), [`zerovector!!`](@ref)
"""
function zerovector end

"""
    zerovector!(x, [S::Type{<:Number} = scalartype(x)])

Modifies `x` in-place to become the zero vector. Optionally, a modified scalar type `S` can
be specified, but this can only work if `x` is a container of a non-concrete type.

Also see: [`zerovector`](@ref), [`zerovector!!`](@ref)
"""
function zerovector! end

"""
    zerovector!!(x, [S::Type{<:Number} = scalartype(x)])

Construct a zero vector in the vector space of `x`, thereby trying to overwrite and thus
recycle `x` when possible. Optionally, a modified scalar type `S` for the resulting zero
vector can be specified.

Also see: [`zerovector`](@ref), [`zerovector!`](@ref)
"""
function zerovector!! end

"""
    scale(x, α::Number)

Computes the new vector-like object obtained from scaling `x` with the scalar `α`.

For unknown types, `scale(x, α) falls back to `x * α`.

Also see: [`scale!`](@ref) and [`scale!!`](@ref)
"""
function scale end

"""
    scale!(x, α::Number) -> x
    scale!(y, x, α::Number) -> y

Rescale `x` with the scalar coefficient `α`, thereby overwrite and thus recylcing the
contents of `x` (in the first form) or `y` (in the second form). This is only possible if
`x`, respectively `y` is mutable, and if the scalar types involed are compatible and can
be promoted and converted.

Also see: [`scale`](@ref) and [`scale!!`](@ref)
"""
function scale! end

"""
    scale!!(x, α)
    scale!!(y, x, α)

Rescale `x` with the scalar coefficient `α`, thereby trying to overwrite and thus recylce
the contens of `x` (in the first form) or `y` (in the second form). When not possible
(because of type or size incompatibilities), a new object will be created to store the
result.

Also see: [`scale`](@ref) and [`scale!`](@ref)
"""
function scale!! end

"""
    add(y, x, [α::Number = 1, β::Number = 1])

Add `y` and `x`, or more generally construct the linear combination `y * β + x * α`.

For unknown types, `add(y, x)` is implemented as `y + x`, and `add(y, β, x, α)` falls back
to `add(scale(y, β), scale(x, α))`.

See also: [`add!`](@ref) and [`add!!`](@ref)
"""
function add end

"""
    add!(y, x, [α::Number = 1, β::Number = 1])

Add `y` and `x`, or more generally construct the linear combination `y * β + x * α`, storing
the result in `y`. This will error in case of incompatible scalar types or incommensurate sizes.

See also: [`add`](@ref) and [`add!!`](@ref)
"""
function add! end

"""
    add!!(y, x, [α::Number = 1, β::Number = 1])

Add `y` and `x`, or more generally construct the linear combination `y * β + x * α`, thereby
trying to store the result in `y`. A new object will be created when this fails due to
incompatible scalar types or incommensurate sizes.

See also: [`add`](@ref) and [`add!`](@ref)
"""
function add!! end

# Inner product
"""
    inner(x, y)

Compute the inner product between `x` and `y`.

For unknown types, `inner(x, y)` falls back to `LinearAlgebra.dot(x, y)`.
"""
function inner end
