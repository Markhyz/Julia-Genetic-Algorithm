push!(LOAD_PATH, ".")

module CardinalityChromosome

include("utility.jl")

using Chromosome
using Parallel

abstract type AbstractCardinalityChromosome <: Chromosome.AbstractChromosome{Tuple{Int64, Float64}} end

GeneType = Tuple{Int64, Float64}

@define CardinalityChromosome begin
  CardinalityChromosome.Chromosome.@Chromosome
end

struct CardinalityChromosomeType <: AbstractCardinalityChromosome
  @CardinalityChromosome
  numAssets::Integer

  function CardinalityChromosomeType(x...)
    args = build(x...)
    new(args[1], args[2], args[3], args[4])
  end
end

function build(numAssets::Integer, x...)
  return (Chromosome.build(x...)..., numAssets)
end

function getCardinality(this::AbstractCardinalityChromosome)
  return Chromosome.getNumGenes(this)
end

function getNumAssets(this::AbstractCardinalityChromosome)
  return this.numAssets
end

function Chromosome.generateRandom!(this::AbstractCardinalityChromosome)
  card = getCardinality(this)
  order = Parallel.threadShuffle(Array(1:this.numAssets))

  rem = 1.0
  for idx = 1 : (card - 1)
    val = Parallel.threadRand() * rem
    this[idx] = (order[idx], val)
    rem = rem - val
  end
  this[card] = (order[card], rem)
end

end
      