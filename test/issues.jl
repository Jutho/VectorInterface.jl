# came up here: https://github.com/Jutho/TensorOperations.jl/issues/154

module Issues
using VectorInterface
using Test
using TestExtras

@test @constinferred zerovector(fill(0.5), Float32) isa Array{Float32,0}
@test @constinferred scale(fill(0.5), 0.3) isa Array{Float64,0}
@test @constinferred add(fill(0.5), fill(0.8), 0.3f0, 0.25im) isa Array{ComplexF64,0}

end