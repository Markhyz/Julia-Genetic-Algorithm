push!(LOAD_PATH, ".")

module BinaryIndividual

include("utility.jl")

using IntegerIndividual

abstract type AbstractBinaryIndividual <: IntegerIndividual.AbstractIntegerIndividual{Int64} end

GeneType = Int64

@define BinaryIndividual begin
  BinaryIndividual.IntegerIndividual.@IntegerIndividual
end

struct BinaryIndividualType <: AbstractBinaryIndividual
  @BinaryIndividual

  function BinaryIndividualType(x...)
    args = build(x...)
    new(args[1], args[2], args[3])
  end
end

function build(ng::Integer)
  return IntegerIndividual.build(ng, fill((0, 1), ng))
end

function build(ind::AbstractBinaryIndividual)
  return IntegerIndividual.build(ind)
end

end