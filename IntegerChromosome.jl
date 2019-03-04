push!(LOAD_PATH, ".")

module IntegerChromosome

include("utility.jl")

using Chromosome
using Parallel

abstract type AbstractIntegerChromosome{GeneType <: Integer} <: Chromosome.AbstractChromosome{GeneType} end

@define IntegerChromosome begin
  IntegerChromosome.Chromosome.@Chromosome
end

struct IntegerChromosomeType{GeneType} <: AbstractIntegerChromosome{GeneType}
  @IntegerChromosome

  function IntegerChromosomeType{GeneType}(x...) where {GeneType}
    args = build(x...)
    new(args[1], args[2], args[3])
  end
end

function build(x...)
  return Chromosome.build(x...)
end

function Chromosome.generateRandom!(this::AbstractIntegerChromosome)
  for i in eachindex(this)
    gene_lb, gene_ub = this.bounds[i]
    this[i] = Parallel.threadRand(gene_lb : gene_ub)
  end
end

end
      