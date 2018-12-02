push!(LOAD_PATH, ".")

module GACrossover

using Debug
using Individual
using RealIndividual
using Parallel

abstract type PointCrossover end
abstract type AritmeticCrossover end

function crossover(p1::IndType, p2::IndType, ::Type{PointCrossover}, cr::Float64) where {IndType <: Individual.AbstractIndividual}
  gene_num = Individual.getNumGenes(p1)
  c1, c2 = deepcopy(p1), deepcopy(p2)
  
  Debug.ga_debug && println("----- Point Crossover -----\n")

  pr = Parallel.threadRand()

  Debug.ga_debug && println("Test: ", pr, " < ", cr, "\n")

  if pr < cr

    if Debug.ga_debug
      println("Parent 1: ", c1[:])
      println("Parent 2: ", c2[:], "\n")
    end

    point = Parallel.threadRand(1:(gene_num-1))

    Debug.ga_debug && println("Cut point: ", point, "\n")

    cuts = [1:point, (point + 1):gene_num]
    chosen_cut = Parallel.threadRand(cuts)
    c1[chosen_cut], c2[chosen_cut] = c2[chosen_cut], c1[chosen_cut]  

    if Debug.ga_debug
      println("Child 1: ", c1[:])
      println("Child 2: ", c2[:], "\n")
    end

  end

  Debug.ga_debug && println("----- Point Crossover End -----\n")

  return (c1, c2)
end

function crossover(p1::IndType, p2::IndType, ::Type{AritmeticCrossover}, cr::Float64) where {IndType <: RealIndividual.AbstractRealIndividual}
  c1, c2 = deepcopy(p1), deepcopy(p2)
  
  pr = Parallel.threadRand()
  if pr < cr
    α = Parallel.threadRand()
    c1[:] = α * c1 + (1 - α) * c2
    c2[:] = α * c2 + (1 - α) * c1
  end
  return (c1, c2) 
end

end