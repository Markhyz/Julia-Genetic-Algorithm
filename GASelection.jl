push!(LOAD_PATH, ".")

module GASelection

using Debug
using Population
using Individual
using Random

abstract type Tournament end

function selectParents(individuals::Vector{Population.IndFitType{IndType}},  parent_num::Integer, ::Type{Tournament}, tour_size::Integer) where {IndType <: Individual.AbstractIndividual}
  parents = Vector{Tuple{IndType, IndType}}(undef, parent_num)
  choices = [(individuals[i], i) for i = eachindex(individuals)]

  Debug.ga_debug && println("----- Tournament Start -----\n")

  for i = 1 : parent_num
    tournament = shuffle(choices)[1:tour_size]
    p1 = maximum(tournament)[2]

    if Debug.ga_debug
      println("Selection ", i)
      println("First tournament: ", [(b, a) for ((x, a), b) in tournament], " -> winner ", p1, "\n")
    end

    tournament = eltype(choices)[]
    for (ind, i) in shuffle(choices)
      i == p1 || push!(tournament, (ind, i))
      length(tournament) < tour_size || break
    end
    p2 = maximum(tournament)[2]
    for ((x, a), b) in tournament
      (a <= individuals[p2][2]) || error("LOL")
    end

    Debug.ga_debug && println("Second tournament: ", [(b, a) for ((x, a), b) in tournament], " -> winner ", p2, "\n")

    parents[i] = (individuals[p1][1], individuals[p2][1])
  end

  Debug.ga_debug && println("\n----- Tournament End -----\n")

  return parents
end

end