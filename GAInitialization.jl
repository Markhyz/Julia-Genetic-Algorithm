push!(LOAD_PATH, ".")

module GAInitialization

using Population
using Parallel
using Individual

abstract type InitRandom end
abstract type InitRandomOne end

function initializePopulation!(pop::Population.AbstractPopulation, pop_size::Integer, ::Type{InitRandom})
  Population.populateRandom!(pop, pop_size)
  Population.evalFitness!(pop)
end

function initializePopulation!(pop::Population.AbstractPopulation{IndType}, pop_size::Integer, ::Type{InitRandomOne}) where {IndType}
  for i = 1:pop_size
    new_ind = IndType(Population.getIndArgs(pop)...)
    order = Parallel.threadShuffle(Array(1:Individual.getNumGenes(new_ind)))
    rem = 1.0
    for gene in @view order[1:end-1]
      val = Parallel.threadRand() * rem
      new_ind[gene] = val
      rem = rem - val
    end
    new_ind[order[end]] = rem
    if abs(sum(new_ind) - 1.0) > 1e-9
      throw("$(sum(new_ind)) NANI")
    end
    Population.insertIndividual!(pop, new_ind)
  end
  Population.evalFitness!(pop)
end

end