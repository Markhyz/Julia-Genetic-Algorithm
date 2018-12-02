push!(LOAD_PATH, ".")

module RealIndividual

include("utility.jl")

using Individual
using Parallel

abstract type AbstractRealIndividual{GeneType <: Real} <: Individual.AbstractIndividual{GeneType} end

@define RealIndividual begin
  RealIndividual.Individual.@Individual
end

struct RealIndividualType{GeneType} <: AbstractRealIndividual{GeneType}
  @RealIndividual

  function RealIndividualType{GeneType}(x...) where {GeneType}
    args = build(x...)
    new(args[1], args[2], args[3])
  end
end

function build(x...)
  return Individual.build(x...)
end

function Individual.generateRandom!(this::AbstractRealIndividual)
  for i in eachindex(this)
    gene_lb, gene_ub = this.bounds[i]
    this[i] = gene_lb + Parallel.threadRand() * (gene_ub - gene_lb)
  end
end

end