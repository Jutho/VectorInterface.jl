module VectorInterfaceChainRulesCoreExt

using VectorInterface
using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent, StructuralTangent, rrule,
    backing, unthunk

# scale
# -----
function ChainRulesCore.rrule(::typeof(scale), x, α::Number)
    function scale_pullback(Δx_)
        Δx = unthunk(Δx_)
        return NoTangent(), scale(Δx, conj(α)), inner(x, Δx)
    end
    return scale(x, α), scale_pullback
end

ChainRulesCore.rrule(::typeof(scale!!), x, α::Number) = rrule(scale, x, α)

function ChainRulesCore.rrule(::typeof(scale!!), y, x, α::Number)
    function scale_pullback(Δy_)
        Δy = unthunk(Δy_)
        return NoTangent(), ZeroTangent(), scale(Δy, conj(α)), inner(x, Δy)
    end
    return scale!!(y, x, α), scale_pullback
end

# add
# ---
function ChainRulesCore.rrule(::typeof(add), y, x, α::Number, β::Number)
    z = add(y, x, α, β)
    function add_pullback(Δz_)
        Δz = unthunk(Δz_)
        return NoTangent(),
            scale(Δz, conj(β)), scale(Δz, conj(α)),
            inner(x, Δz), inner(y, Δz)
    end
    return z, add_pullback
end

ChainRulesCore.rrule(::typeof(add!!), y, x, α::Number, β::Number) = rrule(add, y, x, α, β)

# inner
# -----
function ChainRulesCore.rrule(::typeof(inner), x, y)
    function inner_pullback(Δn_)
        Δn = unthunk(Δn_)
        return NoTangent(), scale(y, conj(Δn)), scale(x, Δn)
    end
    return inner(x, y), inner_pullback
end

# Tangent support
# ---------------
VectorInterface.scale(x::StructuralTangent, α::Number) = map(Base.Fix2(scale, α), x)

function VectorInterface.add(
        y::StructuralTangent{P}, x::StructuralTangent{P}, α::Number, β::Number
    ) where {P}
    return ChainRulesCore.add!!(scale(y, β), scale(x, α))
end

function VectorInterface.inner(y::StructuralTangent{P}, x::P) where {P}
    return inner(backing(y), backing(x))
end
function VectorInterface.inner(y::P, x::StructuralTangent{P}) where {P}
    return inner(backing(y), backing(x))
end

end
