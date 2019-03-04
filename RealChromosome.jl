push!(LOAD_PATH, ".")

module RealChromosome

include("utility.jl")

using Chromosome
using Parallel

abstract type AbstractRealChromosome{GeneType <: Real} <: Chromosome.AbstractChromosome{GeneType} end

@define RealChromosome begin
  RealChromosome.Chromosome.@Chromosome
end

struct RealChromosomeType{GeneType} <: AbstractRealChromosome{GeneType}
  @RealChromosome

  function RealChromosomeType{GeneType}(x...) where {GeneType}
    args = build(x...)
    new(args[1], args[2], args[3])
  end
end

function build(x...)
  return Chromosome.build(x...)
end

function Chromosome.generateRandom!(this::AbstractRealChromosome)
  for i in eachindex(this)
    gene_lb, gene_ub = this.bounds[i]
    this[i] = gene_lb + Parallel.threadRand() * (gene_ub - gene_lb)
  end
end

end