"""
    struct One end

Singleton type for representing a hard-coded constant 1 in vector addition / linear
combinations.
"""
struct One <: Number end

"""
    struct Zero end
    
Singleton type for representing a hard-coded constant 0 in vector addition / linear
combinations.
"""
struct Zero <: Number end

# Base Arithmetic
# ---------------

Base.:(-)(::One) = -1
Base.:(-)(::Zero) = Zero()
Base.:(+)(::Zero, x::Number) = x
Base.:(+)(x::Number, ::Zero) = x
Base.:(+)(::Zero, ::Zero) = Zero()

Base.:(*)(::One, x::Number) = x
Base.:(*)(::Zero, ::Number) = Zero()
Base.:(*)(x::Number, ::One) = x
Base.:(*)(::Number, ::Zero) = Zero()
Base.:(*)(::Zero, ::Zero) = Zero()
Base.:(*)(::One, ::One) = One()
Base.:(*)(::Zero, ::One) = Zero()
Base.:(*)(::One, ::Zero) = Zero()
Base.inv(::One) = One()

Base.conj(::One) = One()
Base.conj(::Zero) = Zero()

Base.one(::Type{<:Union{One,Zero}}) = One()
Base.one(x::Union{One,Zero}) = one(typeof(x))
Base.zero(::Type{<:Union{One,Zero}}) = Zero()
Base.zero(x::Union{One,Zero}) = zero(typeof(x))

Base.:(==)(::One, ::One) = true
Base.:(==)(::Zero, ::Zero) = true
Base.:(==)(::One, ::Zero) = false
Base.:(==)(::Zero, ::One) = false

# Promotion
# ---------
Base.promote_rule(::Type{<:Union{One,Zero}}, ::Type{T}) where {T<:Number} = T
Base.convert(::Type{T}, ::One) where {T<:Number} = one(T)
(T::Type{<:Number})(::One) = one(T)
Base.convert(::Type{T}, ::Zero) where {T<:Number} = zero(T)
(T::Type{<:Number})(::Zero) = zero(T)
