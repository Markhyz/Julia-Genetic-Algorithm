push!(LOAD_PATH, ".")

module CardinalityChromosome

include("utility.jl")

using Chromosome
using Parallel

abstract type AbstractCardinalityChromosome{GeneType <: Tuple{Integer, Float64}} <: Chromosome.AbstractChromosome{GeneType} end

@define CardinalityChromosome begin
  CardinalityChromosome.Chromosome.@Chromosome
end

struct CardinalityChromosomeType{GeneType} <: AbstractCardinalityChromosome{GeneType}
  @CardinalityChromosome
  n::Integer

  function CardinalityChromosomeType{GeneType}(x...) where {GeneType}
    args = build(x...)
    new(args[1], args[2], args[3], args[4])
  end
end

function build(n::Integer, x...)
  return (Chromosome.build(x...)..., n)
end

function getCardinality(this::AbstractCardinalityChromosome)
  return this.n
end

function Chromosome.generateRandom!(this::AbstractCardinalityChromosome)
  k = Chromosome.getNumGenes(this)
  order = Parallel.threadShuffle(Array(1:this.n))

  rem = 1.0
  for idx = 1 : (k - 1)
    this[idx][1] = order[idx]

    val = Parallel.threadRand() * rem
    this[idx][2] = val
    rem = rem - val
  end
  this[k] = (order[k], rem)
end

end
      