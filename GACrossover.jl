push!(LOAD_PATH, ".")

module GACrossover

using Debug
using Chromosome
using CardinalityChromosome
using RealChromosome
using BinaryChromosome
using Parallel

abstract type PointCrossover end
abstract type AritmeticCrossover end
abstract type UniformPOCrossover end
abstract type CardinalityPOCrossover end

function crossover(p1::ChromoT, p2::ChromoT, ::Type{PointCrossover}, cr::Float64) where {ChromoT <: Chromosome.AbstractChromosome}
  gene_num = Chromosome.getNumGenes(p1)
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

function crossover(p1::ChromoT, p2::ChromoT, ::Type{AritmeticCrossover}, cr::Float64) where {ChromoT <: RealChromosome.AbstractRealChromosome}
  c1, c2 = deepcopy(p1), deepcopy(p2)
  
  pr = Parallel.threadRand()
  if pr < cr
    α = Parallel.threadRand()
    c1[:] = α * c1 + (1 - α) * c2
    c2[:] = α * c2 + (1 - α) * c1
  end
  return (c1, c2) 
end

function crossover(p1::Tuple{BinaryChromosome.BinaryChromosomeType, RealChromosome.RealChromosomeType}, 
                   p2::Tuple{BinaryChromosome.BinaryChromosomeType, RealChromosome.RealChromosomeType},
                   ::Type{UniformPOCrossover}, cr::Float64)
  child = deepcopy(p1)
  
  pr = Parallel.threadRand()
  if pr < cr
    for i in eachindex(p1)
      α = Parallel.threadRand()
      child[1][i], child[2][i] = α > 0.5 ? (p1[1][i], p1[2][i]) : (p2[1][i], p2[2][i])
    end
  end
  return (child,)
end

function crossover(p1::CardinalityChromosome.CardinalityChromosomeType, 
                   p2::CardinalityChromosome.CardinalityChromosomeType,
                   ::Type{CardinalityPOCrossover}, cr::Float64)
  gene_num = Chromosome.getNumGenes(p1)
  child = deepcopy(p1)
  
  Debug.ga_debug && println("----- Cardinality Special Crossover -----\n")

  pr = Parallel.threadRand()

  Debug.ga_debug && println("Test: ", pr, " < ", cr, "\n")

  if pr < cr 

    if Debug.ga_debug
      println("Parent 1: ", p1[:])
      println("Parent 2: ", p2[:], "\n")
    end

    used = zeros(Bool, CardinalityChromosome.getNumAssets(p1))
    bounds = fill((0.0, 0.0), CardinalityChromosome.getNumAssets(p1))
    assets = Int64[]

    for (asset, weight) in p1
      push!(assets, asset)
      bounds[asset] = (0.0, weight)
    end

    for (asset, weight) in p2
      push!(assets, asset)
      weight2 = bounds[asset][2]
      bounds[asset] = weight < weight2 ? (weight, weight2) : (weight2, weight)
    end

    assets = Parallel.threadShuffle(assets)
    cur_gene = 1

    for asset in assets
      if !used[asset]
        used[asset] = true

        lb, ub = bounds[asset]
        x = Parallel.threadRand()
        weight = lb + x * (ub - lb) 
        
        child[cur_gene] = (asset, weight)
        cur_gene += 1
        if cur_gene > gene_num
          break
        end
      end
    end

    total_weight = sum(gene -> gene[2], child)
    for idx in eachindex(child)
      asset, weight = child[idx]
      child[idx] = (asset, weight / total_weight)
    end

    ### Assert if child is valid ###
    total_weight = sum(gene -> gene[2], child)
    @assert abs(total_weight - 1.0) <= 1e-9

    used = zeros(Bool, CardinalityChromosome.getNumAssets(p1))
    for (asset, weight) in child
      @assert !used[asset]
      used[asset] = true
    end
    ### End of assertion ###

    if Debug.ga_debug
      println("Child: ", child[:])
    end

  end
  return (child,)
end

end