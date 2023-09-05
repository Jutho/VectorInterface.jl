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
    (α === _one) && return x
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
    (α === _one) && return x
    T = Tuple{typeof(x),typeof(α)}
    @warn _warn_message(scale!, T) maxlog = 1
    if applicable(LinearAlgebra.rmul!, x, α)
        return LinearAlgebra.rmul!(x, α)
    else
        throw(ArgumentError(_error_message(scale!, T)))
    end
end

function scale!!(x, α::Number)
    (α === _one) && return x
    T = Tuple{typeof(x),typeof(α)}
    @warn _warn_message(scale!!, T) maxlog = 1
    Tx = scalartype(x)
    if applicable(LinearAlgebra.rmul!, x, α) &&
       Base.promote_op(scale, Tx, scalartype(α)) <: Tx
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
    Tx = scalartype(x)
    Ty = scalartype(y)
    if applicable(LinearAlgebra.mul!, y, x, α) &&
       Base.promote_op(scale, Tx, scalartype(α)) <: Ty
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
add(y, x, α::Number) = (α === _one) ? add(y, x) : add(y, scale(x, α))
function add(y, x, α::Number, β::Number)
    yb = (β === _one) ? y : scale(y, β)
    xa = (α === _one) ? x : scale(x, α)
    return add(yb, xa)
end

function add!(y, x, α::Number=_one)
    T = Tuple{typeof(y),typeof(x),typeof(α)}
    @warn _warn_message(add!, T) maxlog = 1
    if applicable(LinearAlgebra.axpy!, α, x, y)
        return LinearAlgebra.axpy!(α, x, y)
    else
        throw(ArgumentError(_error_message(add!, T)))
    end
end

function add!(y, x, α::Number, β::Number)
    if β === _one
        return add!(y, x, α)
    else
        T = Tuple{typeof(y),typeof(x),typeof(α),typeof(β)}
        @warn _warn_message(add!, T) maxlog = 1
        if applicable(LinearAlgebra.axpby!, α, x, β, y)
            return LinearAlgebra.axpby!(α, x, β, y)
        else
            throw(ArgumentError(_error_message(add!, T)))
        end
    end
end

function add!!(y, x)
    T = Tuple{typeof(y),typeof(x)}
    @warn _warn_message(add!!, T) maxlog = 1
    Tx = scalartype(x)
    Ty = scalartype(y)
    if applicable(LinearAlgebra.axpy!, true, x, y) && Base.promote_op(add, Tx, Ty) <: Ty
        return LinearAlgebra.axpy!(true, x, y)
    elseif applicable(+, y, x)
        return y + x
    else
        throw(ArgumentError(_error_message(add!!, T)))
    end
end

function add!!(y, x, α::Number)
    if α === _one
        return add!!(y, x)
    else
        Tx = scalartype(x)
        Ty = scalartype(y)
        if applicable(LinearAlgebra.axpy!, α, x, y) &&
           Base.promote_op(add, Ty, Tx, scalartype(α)) <: Ty
            T = Tuple{typeof(y),typeof(x),typeof(α)}
            @warn _warn_message(add!!, T) maxlog = 1
            return LinearAlgebra.axpy!(α, x, y)
        else
            return add!!(y, scale(x, α))
        end
    end
end

function add!!(y, x, α::Number, β::Number)
    if β === _one
        return add!!(y, x, α)
    else
        α′ = (α === _one) ? true : α
        Tx = scalartype(x)
        Ty = scalartype(y)
        if applicable(LinearAlgebra.axpby!, α′, x, β, y) &&
           Base.promote_op(add, Ty, Tx, scalartype(α), scalartype(β)) <: Ty
            T = Tuple{typeof(y),typeof(x),typeof(α′),typeof(β)}
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
