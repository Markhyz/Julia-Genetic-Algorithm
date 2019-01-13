push!(LOAD_PATH, ".")

module Fitness

include("utility.jl")

using Individual
using RealIndividual

abstract type AbstractFitness{N} end

@define Fitness begin
  direction::NTuple{N, Int64} where N
end

function getSize(fit::AbstractFitness{N}) where N
  return N
end

function getDirection(fit::AbstractFitness)
  return fit.direction
end

end