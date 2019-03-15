push!(LOAD_PATH, ".")

module Fitness

include("utility.jl")

using Chromosome
using RealChromosome

FitnessType = NTuple{N, Float64} where N

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