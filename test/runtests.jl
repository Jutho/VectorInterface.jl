using VectorInterface
using Test
using TestExtras

deepcollect(x) = vcat(map(deepcollect, x)...)
deepcollect(x::Number) = x


x = (collect(zip(randn(2,2), rand(2,2))), (randn(), randn(3), randn(2,2)'), randn(), (view(randn(4,4), 1:2, [1,3,4]),))
y = (collect(zip(randn(2,2), rand(2,2))), (randn(), randn(3), randn(2,2)'), randn(), (view(randn(4,4), 1:2, [1,3,4]),))

@testset "scalartype" begin
    s = @constinferred scalartype(x)
    @test s == Float64
end

@testset "zerovector" begin
    z = @constinferred zerovector(x)
    @test all(iszero, deepcollect(z))
    @test all(deepcollect(z) .=== zero(scalartype(x)))
    z1 = @constinferred zerovector!!(deepcopy(x))
    @test all(deepcollect(z1) .=== zero(scalartype(x)))

    z2 = @constinferred zerovector(x, ComplexF64)
    @test all(deepcollect(z2) .=== zero(ComplexF64))
    z3 = @constinferred zerovector!!(deepcopy(x), ComplexF64)
    @test all(deepcollect(z3) .=== zero(ComplexF64))

    xy = [deepcopy(x), deepcopy(y)]
    z4 = @constinferred zerovector!(xy)
    @test all(deepcollect(z4) .=== zero(scalartype(xy)))
end

@testset "scale" begin
    α = randn()
    z = @constinferred scale(x, α)
    @test all(deepcollect(z) .== α .* deepcollect(x))
    z2 = @constinferred scale!!(deepcopy(x), α)
    @test all(deepcollect(z2) .== α .* deepcollect(x))

    xy = [deepcopy(x), deepcopy(y)]
    z3 = @constinferred scale!(deepcopy(xy), α)
    @test all(deepcollect(z3) .== α .* deepcollect(xy))

    α = randn(ComplexF64)
    z4 = @constinferred scale(x, α)
    @test all(deepcollect(z4) .== α .* deepcollect(x))
    z5 = @constinferred scale!!(deepcopy(x), α)
    @test all(deepcollect(z5) .== α .* deepcollect(x))
end

@testset "add" begin
    α, β = randn(2)
    z = @constinferred add(y, x)
    @test all(deepcollect(z) .== deepcollect(x) .+ deepcollect(y))
    z = @constinferred add(y, x, α)
    @test all(deepcollect(z) .== muladd.(deepcollect(x), α, deepcollect(y)))
    z = @constinferred add(y, x, α, β)
    @test all(deepcollect(z) .== muladd.(deepcollect(x), α, deepcollect(y) .* β))

    α, β = randn(2)
    z2 = @constinferred add!!(deepcopy(y), deepcopy(x))
    @test deepcollect(z2) ≈ (deepcollect(x) .+ deepcollect(y))
    z2 = @constinferred add!!(deepcopy(y), deepcopy(x), α)
    @test deepcollect(z2) ≈ (muladd.(deepcollect(x), α, deepcollect(y)))
    z2 = @constinferred add!!(deepcopy(y), deepcopy(x), α, β)
    @test deepcollect(z2) ≈ (muladd.(deepcollect(x), α, deepcollect(y) .* β))

    α, β = randn(2)
    xy = [deepcopy(x), deepcopy(y)]
    yx = [deepcopy(y), deepcopy(x)]
    z3 = @constinferred add!(deepcopy(xy), deepcopy(yx))
    @test deepcollect(z3) ≈ (deepcollect(xy) .+ deepcollect(yx))
    z3 = @constinferred add!(deepcopy(xy), deepcopy(yx), α)
    @test deepcollect(z3) ≈ (muladd.(deepcollect(yx), α, deepcollect(xy)))
    z3 = @constinferred add!(deepcopy(xy), deepcopy(yx), α, β)
    @test deepcollect(z3) ≈ (muladd.(deepcollect(yx), α, deepcollect(xy) .* β))

    α, β = randn(ComplexF64, 2)
    z4 = @constinferred add(y, x, α)
    @test all(deepcollect(z4) .== muladd.(deepcollect(x), α, deepcollect(y)))
    z4 = @constinferred add(y, x, α, β)
    @test all(deepcollect(z4) .== muladd.(deepcollect(x), α, deepcollect(y) .* β))

    α, β = randn(ComplexF64, 2)
    z5 = @constinferred add!!(deepcopy(y), deepcopy(x), α)
    @test deepcollect(z5) ≈ (muladd.(deepcollect(x), α, deepcollect(y)))
    z5 = @constinferred add!!(deepcopy(y), deepcopy(x), α, β)
    @test deepcollect(z5) ≈ (muladd.(deepcollect(x), α, deepcollect(y) .* β))
end

@testset "inner" begin
    s = @constinferred inner(x, y)
    @test s ≈ inner(deepcollect(x), deepcollect(y))

    α, β = randn(ComplexF64, 2)
    s2 = @constinferred inner(scale(x, α), scale(y, β))
    @test s2 ≈ inner(α * deepcollect(x), β * deepcollect(y))
end
