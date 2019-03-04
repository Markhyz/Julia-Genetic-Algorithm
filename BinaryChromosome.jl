push!(LOAD_PATH, ".")

module BinaryChromosome

include("utility.jl")

using IntegerChromosome

abstract type AbstractBinaryChromosome <: IntegerChromosome.AbstractIntegerChromosome{Int64} end

GeneType = Int64

@define BinaryChromosome begin
  BinaryChromosome.IntegerChromosome.@IntegerChromosome
end

struct BinaryChromosomeType <: AbstractBinaryChromosome
  @BinaryChromosome

  function BinaryChromosomeType(x...)
    args = build(x...)
    new(args[1], args[2], args[3])
  end
end

function build(ng::Integer)
  return IntegerChromosome.build(ng, fill((0, 1), ng))
end

function build(ind::AbstractBinaryChromosome)
  return IntegerChromosome.build(ind)
end

end