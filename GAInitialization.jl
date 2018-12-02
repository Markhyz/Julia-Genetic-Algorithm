push!(LOAD_PATH, ".")

module GAInitialization

using Population

abstract type InitRandom end

function initializePopulation!(pop::Population.AbstractPopulation, pop_size::Integer, ::Type{InitRandom})
  Population.populateRandom!(pop, pop_size)
  Population.evalFitness!(pop)
end

end