push!(LOAD_PATH, ".")

module GASelection

using Debug
using Population
using Random
using Parallel

abstract type Tournament end

function selectParents(individuals::Vector{<: Population.GeneticAlgorithmFit{IndT}},  
  										 parent_num::Integer, ::Type{Tournament}, tour_size::Integer) where {IndT}
  parents = Vector{Tuple{IndT, IndT}}(undef, parent_num)
  choices = [(individuals[i], i) for i = eachindex(individuals)]

  Debug.ga_debug && println("----- Tournament Start -----\n")

  for i = 1 : parent_num
    tournament = Parallel.threadShuffle(choices)[1:tour_size]
    p1 = maximum(tournament)[2]

    if Debug.ga_debug
      println("Selection ", i)
      println("First tournament: ", [(b, ind[2:end]) for (ind, b) in tournament], " -> winner ", p1, "\n")
    end

    tournament = eltype(choices)[]
    for (ind, i) in Parallel.threadShuffle(choices)
      i == p1 || push!(tournament, (ind, i))
      length(tournament) < tour_size || break
    end
    p2 = maximum(tournament)[2]
    
    Debug.ga_debug && println("Second tournament: ", [(b, ind[2:end]) for (ind, b) in tournament], " -> winner ", p2, "\n")
    
    for (ind, b) in tournament
      (ind <= individuals[p2]) || error("LOL")
    end

    parents[i] = (individuals[p1][1], individuals[p2][1])
  end

  Debug.ga_debug && println("\n----- Tournament End -----\n")

  return parents
end

end