using VectorInterface
using Test

println("Testing with simple numerical array")
println("===================================")
include("simple.jl")

println("Testing with complicated composite object")
println("=========================================")
include("complicated.jl")

println("Quality control test with Aqua.jl")
println("=================================")
module AquaVectorInterface
    using VectorInterface
    using Aqua
    Aqua.test_all(VectorInterface)
end
