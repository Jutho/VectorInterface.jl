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
function scale(x, ??::ONumber)
    if ?? isa _One
        return x
    else
        T = Tuple{typeof(x),typeof(??)}
        @warn _warn_message(scale, T) maxlog = 1
        if applicable(*, x, ??)
            return x * ??
        elseif applicable(*, ??, x)
            return ?? * x
        else
            throw(ArgumentError(_error_message(scale, T)))
        end
    end
end

function scale!(x, ??::ONumber)
    if ?? isa _One
        return x
    else
        T = Tuple{typeof(x),typeof(??)}
        @warn _warn_message(scale!, T) maxlog = 1
        if applicable(LinearAlgebra.rmul!, x, ??)
            return LinearAlgebra.rmul!(x, ??)
        else
            throw(ArgumentError(_error_message(scale!, T)))
        end
    end
end

function scale!!(x, ??::ONumber)
    (?? isa _One) && return x
    T = Tuple{typeof(x),typeof(??)}
    @warn _warn_message(scale!!, T) maxlog = 1
    Tx = scalartype(x)
    if applicable(LinearAlgebra.rmul!, x, ??) && promote_type(Tx, typeof(??)) <: Tx
        return LinearAlgebra.rmul!(x, ??)
    elseif applicable(*, x, ??)
        return x * ??
    else
        throw(ArgumentError(_error_message(scale!!, T)))
    end
end

function scale!(y, x, ??::ONumber)
    if ?? isa _One
        return x
    else
        T = Tuple{typeof(y),typeof(x),typeof(??)}
        @warn _warn_message(scale!, T) maxlog = 1
        if applicable(LinearAlgebra.mul!, y, x, ??)
            return LinearAlgebra.mul!(y, x, ??)
        else
            throw(ArgumentError(_error_message(scale!, T)))
        end
    end
end

function scale!!(y, x, ??::ONumber)
    if ?? isa _One
        return x
    else
        T = Tuple{typeof(y),typeof(x),typeof(??)}
        @warn _warn_message(scale!!, T) maxlog = 1
        Tx = scalartype(x)
        Ty = scalartype(y)
        if applicable(LinearAlgebra.mul!, y, x, ??) && promote_type(Ty, Tx, typeof(??)) <: Ty
            return LinearAlgebra.mul!(y, x, ??)
        elseif applicable(*, x, ??)
            return x * ??
        else
            throw(ArgumentError(_error_message(scale!!, T)))
        end
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
add(y, x, ??::ONumber) = (?? isa _One) ? add(y, x) : add(y, scale(x, ??))
function add(y, x, ??::ONumber, ??::ONumber)
    yb = (?? isa _One) ? y : scale(y, ??)
    xa = (?? isa _One) ? x : scale(x, ??)
    return add(yb, xa)
end

function add!(y, x, ??::ONumber=_one)
    ????? = (?? isa _One) ? true : ??
    T = Tuple{typeof(y),typeof(x),typeof(?????)}
    @warn _warn_message(add!, T) maxlog = 1
    if applicable(LinearAlgebra.axpy!, ?????, x, y)
        return LinearAlgebra.axpy!(?????, x, y)
    else
        throw(ArgumentError(_error_message(add!, T)))
    end
end

function add!(y, x, ??::ONumber, ??::ONumber)
    if ?? isa _One
        return add!(y, x, ??)
    else
        ????? = (?? isa _One) ? true : ??
        T = Tuple{typeof(y),typeof(x),typeof(?????),typeof(??)}
        @warn _warn_message(add!, T) maxlog = 1
        if applicable(LinearAlgebra.axpby!, ?????, x, ??, y)
            return LinearAlgebra.axpby!(?????, x, ??, y)
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
    if applicable(LinearAlgebra.axpy!, true, x, y) && promote_type(Tx, Ty) <: Ty
        return LinearAlgebra.axpy!(true, x, y)
    elseif applicable(+, y, x)
        return y + x
    else
        throw(ArgumentError(_error_message(add!!, T)))
    end
end

function add!!(y, x, ??::ONumber)
    if ?? isa _One
        return add!!(y, x)
    else
        Tx = scalartype(x)
        Ty = scalartype(y)
        if applicable(LinearAlgebra.axpy!, ??, x, y) && promote_type(typeof(??), Tx, Ty) <: Ty
            T = Tuple{typeof(y),typeof(x),typeof(??)}
            @warn _warn_message(add!!, T) maxlog = 1
            return LinearAlgebra.axpy!(??, x, y)
        else
            return add!!(y, scale(x, ??))
        end
    end
end

function add!!(y, x, ??::ONumber, ??::ONumber)
    if ?? isa _One
        return add!!(y, x, ??)
    else
        ????? = (?? isa _One) ? true : ??
        Tx = scalartype(x)
        Ty = scalartype(y)
        if applicable(LinearAlgebra.axpby!, ?????, x, ??, y) &&
           promote_type(typeof(??), Tx, typeof(??), Ty) <: Ty
            T = Tuple{typeof(y),typeof(x),typeof(?????),typeof(??)}
            @warn _warn_message(add!!, T) maxlog = 1
            return LinearAlgebra.axpby!(?????, x, ??, y)
        else
            return add!!(scale!!(y, ??), scale(x, ??))
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
