push!(LOAD_PATH, ".")

module IntegerIndividual

include("utility.jl")

using Individual
using Parallel

abstract type AbstractIntegerIndividual{GeneType <: Integer} <: Individual.AbstractIndividual{GeneType} end

@define IntegerIndividual begin
  IntegerIndividual.Individual.@Individual
end

struct IntegerIndividualType{GeneType} <: AbstractIntegerIndividual{GeneType}
  @IntegerIndividual

  function IntegerIndividualType{GeneType}(x...) where {GeneType}
    args = build(x...)
    new(args[1], args[2], args[3])
  end
end

function build(x...)
  return Individual.build(x...)
end

function Individual.generateRandom!(this::AbstractIntegerIndividual)
  for i in eachindex(this)
    gene_lb, gene_ub = this.bounds[i]
    this[i] = Parallel.threadRand(gene_lb : gene_ub)
  end
end

end
      