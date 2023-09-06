# Vector interface implementation: general fallbacks?
#######################################################
# Are these desirable; should they exist?
# They might lead to plenty of invalidations?
# They do not enforce any compatibility between the different vector types

@noinline function _warn_message(fun, type)
    return string("The function `", fun,
                  "` is not implemented for (values of) type `", type, "`;\n",
                  "this fallback will disappear in future versions of VectorInterface.jl")
end
@noinline function _error_message(fun, type)
    return string("No fallback for applying `", fun, "` to (values of) type `", type,
                  "` could be determined")
end

# scalartype
#------------
function scalartype(T::Type)
    @warn _warn_message(scalartype, T) maxlog = 1
    if applicable(eltype, T)
        return scalartype(eltype(T))
    else
        throw(ArgumentError(_error_message(scalartype, T)))
    end
end
# should this try to use `eltype` instead? e.g. scalartype(T) = scalartype(eltype(T))

# zerovector & zerovector!!
#---------------------------
function zerovector(x, ::Type{S}) where {S<:Number}
    T = Tuple{typeof(x),S}
    @warn _warn_message(zerovector, T) maxlog = 1
    if applicable(similar, x, S)
        return zerovector!(similar(x, S))
    else
        throw(ArgumentError(_error_message(zerovector, T)))
    end
end
function zerovector!(x)
    T = typeof(x)
    @warn _warn_message(zerovector!, T) maxlog = 1
    if applicable(LinearAlgebra.rmul!, x, false)
        return LinearAlgebra.rmul!(x, false)
    else
        throw(ArgumentError(_error_message(zerovector!, T)))
    end
end
function zerovector!!(x)
    T = typeof(x)
    @warn _warn_message(zerovector!!, T) maxlog = 1
    if applicable(LinearAlgebra.rmul!, x, false)
        return LinearAlgebra.rmul!(x, false)
    elseif applicable(zero, x)
        return zero(x)
    elseif applicable(similar, x) && applicable(LinearAlgebra.rmul!, x, false)
        return LinearAlgebra.rmul!(similar(x), false)
    else
        throw(ArgumentError(_error_message(zerovector!!, T)))
    end
end

# scale, scale! & scale!!
#-------------------------
function scale(x, α::Number)
    (α === One()) && return x
    T = Tuple{typeof(x),typeof(α)}
    @warn _warn_message(scale, T) maxlog = 1
    if applicable(*, x, α)
        return x * α
    elseif applicable(*, α, x)
        return α * x
    else
        throw(ArgumentError(_error_message(scale, T)))
    end
end

function scale!(x, α::Number)
    (α === One()) && return x
    T = Tuple{typeof(x),typeof(α)}
    @warn _warn_message(scale!, T) maxlog = 1
    if applicable(LinearAlgebra.rmul!, x, α)
        return LinearAlgebra.rmul!(x, α)
    else
        throw(ArgumentError(_error_message(scale!, T)))
    end
end

function scale!!(x, α::Number)
    (α === One()) && return x
    T = Tuple{typeof(x),typeof(α)}
    @warn _warn_message(scale!!, T) maxlog = 1
    if applicable(LinearAlgebra.rmul!, x, α) && promote_scale(x, α) <: scalartype(x)
        return LinearAlgebra.rmul!(x, α)
    elseif applicable(*, x, α)
        return x * α
    else
        throw(ArgumentError(_error_message(scale!!, T)))
    end
end

function scale!(y, x, α::Number)
    T = Tuple{typeof(y),typeof(x),typeof(α)}
    @warn _warn_message(scale!, T) maxlog = 1
    if applicable(LinearAlgebra.mul!, y, x, α)
        return LinearAlgebra.mul!(y, x, α)
    else
        throw(ArgumentError(_error_message(scale!, T)))
    end
end

function scale!!(y, x, α::Number)
    T = Tuple{typeof(y),typeof(x),typeof(α)}
    @warn _warn_message(scale!!, T) maxlog = 1
    if applicable(LinearAlgebra.mul!, y, x, α) && promote_scale(x, α) <: scalartype(y)
        return LinearAlgebra.mul!(y, x, α)
    elseif applicable(*, x, α)
        return x * α
    else
        throw(ArgumentError(_error_message(scale!!, T)))
    end
end

# add, add! & add!!
#-------------------
function add(y, x)
    T = Tuple{typeof(y),typeof(x)}
    @warn _warn_message(add, T) maxlog = 1
    if applicable(+, y, x)
        return y + x
    else
        throw(ArgumentError(_error_message(add, T)))
    end
end
add(y, x, α::Number) = (α === One()) ? add(y, x) : add(y, scale(x, α))
function add(y, x, α::Number, β::Number)
    yb = (β === One()) ? y : scale(y, β)
    xa = (α === One()) ? x : scale(x, α)
    return add(yb, xa)
end

function add!(y, x, α::Number=One())
    T = Tuple{typeof(y),typeof(x),typeof(α)}
    @warn _warn_message(add!, T) maxlog = 1
    α′ = (α === One()) ? true : α
    if applicable(LinearAlgebra.axpy!, α′, x, y)
        return LinearAlgebra.axpy!(α′, x, y)
    else
        throw(ArgumentError(_error_message(add!, T)))
    end
end

function add!(y, x, α::Number, β::Number)
    if β === One()
        return add!(y, x, α)
    else
        T = Tuple{typeof(y),typeof(x),typeof(α),typeof(β)}
        @warn _warn_message(add!, T) maxlog = 1
        α′ = (α === One()) ? true : α
        if applicable(LinearAlgebra.axpby!, α′, x, β, y)
            return LinearAlgebra.axpby!(α′, x, β, y)
        else
            throw(ArgumentError(_error_message(add!, T)))
        end
    end
end

function add!!(y, x)
    T = Tuple{typeof(y),typeof(x)}
    @warn _warn_message(add!!, T) maxlog = 1
    if applicable(LinearAlgebra.axpy!, true, x, y) && promote_add(y, x) <: scalartype(y)
        return LinearAlgebra.axpy!(true, x, y)
    elseif applicable(+, y, x)
        return y + x
    else
        throw(ArgumentError(_error_message(add!!, T)))
    end
end

function add!!(y, x, α::Number)
    if α === One()
        return add!!(y, x)
    else
        if applicable(LinearAlgebra.axpy!, α, x, y) && promote_add(y, x, α) <: scalartype(y)
            T = Tuple{typeof(y),typeof(x),typeof(α)}
            @warn _warn_message(add!!, T) maxlog = 1
            return LinearAlgebra.axpy!(α, x, y)
        else
            return add!!(y, scale(x, α))
        end
    end
end

function add!!(y, x, α::Number, β::Number)
    if β === One()
        return add!!(y, x, α)
    else
        α′ = (α === One()) ? true : α
        if applicable(LinearAlgebra.axpby!, α′, x, β, y) &&
           promote_add(y, x, α, β) <: scalartype(y)
            T = Tuple{typeof(y),typeof(x),typeof(α),typeof(β)}
            @warn _warn_message(add!!, T) maxlog = 1
            return LinearAlgebra.axpby!(α′, x, β, y)
        else
            return add!!(scale!!(y, β), scale(x, α))
        end
    end
end

# inner
#-------
function inner(x, y)
    T = Tuple{typeof(x),typeof(y)}
    @warn _warn_message(inner, T) maxlog = 1
    if applicable(LinearAlgebra.dot, x, y)
        return LinearAlgebra.dot(x, y)
    else
        throw(ArgumentError(_error_message(inner, T)))
    end
end
