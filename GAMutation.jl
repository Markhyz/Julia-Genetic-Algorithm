push!(LOAD_PATH, ".")

module GAMutation

using Debug
using Chromosome
using BinaryChromosome
using RealChromosome
using Parallel
using Random
using Distributions

abstract type BitFlipMutation end
abstract type UniformMutation end
abstract type SwapMutation end
abstract type GaussMutation end
abstract type BitFlipPOMutation end
abstract type CardinalityPOMutation end

function mutate!(ind::BinaryChromosome.AbstractBinaryChromosome, ::Type{BitFlipMutation},  mr::Float64)

  Debug.ga_debug && println("----- Bit Flip -----\n")
  Debug.ga_debug && println("Before mutation: ", ind[:])

  for i in eachindex(ind)
    pr = Parallel.threadRand()
    if pr < mr
      ind[i] = xor(ind[i], 1)
    end
  end

  Debug.ga_debug && println("After mutation: ", ind[:], "\n")
  Debug.ga_debug && println("----- Bit Flip End -----\n")

  return ind
end

function mutate!(ind::RealChromosome.AbstractRealChromosome, ::Type{UniformMutation}, mr::Float64, lb::Float64, ub::Float64)

  Debug.ga_debug && println("----- Uniform -----\n")
  Debug.ga_debug && println("Before mutation: ", ind[:])

  for i in eachindex(ind)
    pr = Parallel.threadRand()
    if pr < mr
      Δ = lb + Parallel.threadRand() * (ub - lb)
      ind[i] += Parallel.threadRand([1, -1]) * Δ
      ind[i] = max(lb, min(ind[i], ub))
    end
  end

  Debug.ga_debug && println("After mutation: ", ind[:], "\n")
  Debug.ga_debug && println("----- Uniform End -----\n")

  return ind
end

function mutate!(ind::Chromosome.AbstractChromosome, ::Type{SwapMutation},  mr::Float64)

  Debug.ga_debug && println("----- Swap -----\n")
  Debug.ga_debug && println("Before mutation: ", ind[:])

  for i in eachindex(ind)
    pr = Parallel.threadRand()
    if pr < mr
      idx = Parallel.threadRand(1:(Chromosome.getNumGenes(ind) - 1)) + i
      if idx > Chromosome.getNumGenes(ind)
        idx = idx - Chromosome.getNumGenes(ind)
      end
      ind[i], ind[idx] = ind[idx], ind[i]
    end
  end

  Debug.ga_debug && println("After mutation: ", ind[:], "\n")
  Debug.ga_debug && println("----- Swap End -----\n")

  return ind
end

function mutate!(ind::Chromosome.AbstractChromosome, ::Type{GaussMutation},  mr::Float64, σ::Float64)

  Debug.ga_debug && println("----- Gauss -----\n")
  Debug.ga_debug && println("Before mutation: ", ind[:])

  normal_dist = Normal(0.0, σ)

  for i in eachindex(ind)
    pr = Parallel.threadRand()
    if pr < mr
      Δ, = rand(normal_dist, 1)
      ind[i] = max(min(ind[i] + Δ, 1.0), 0.0)
    end
  end

  Debug.ga_debug && println("After mutation: ", ind[:], "\n")
  Debug.ga_debug && println("----- Gauss End -----\n")

  return ind
end

function mutate!(ind::Tuple{BinaryChromosome.AbstractBinaryChromosome, RealChromosome.AbstractRealChromosome}, ::Type{BitFlipPOMutation},  mr::Float64)

  Debug.ga_debug && println("----- Bit Flip -----\n")
  Debug.ga_debug && println("Before mutation: ", ind[:])

  for i in eachindex(ind[1])
    pr = Parallel.threadRand()
    if pr < mr
      ind[i] = xor(ind[1][i], 1)
      if ind[1][i] == 1
        ind[2][i] = Parallel.threadRand()
      end
    end
  end

  Debug.ga_debug && println("After mutation: ", ind[:], "\n")
  Debug.ga_debug && println("----- Bit Flip End -----\n")

  return ind
end

function mutate!(ind::CardinalityChromosome.AbstractCardinalityChromosome, ::Type{CardinalityPOMutation},  mr::Float64)

  Debug.ga_debug && println("----- Cardinality -----\n")
  Debug.ga_debug && println("Before mutation: ", ind[:])

  for i in eachindex(ind[1])
    pr = Parallel.threadRand()
    if pr < mr
      ind[i] = xor(ind[1][i], 1)
      if ind[1][i] == 1
        ind[2][i] = Parallel.threadRand()
      end
    end
  end

  Debug.ga_debug && println("After mutation: ", ind[:], "\n")
  Debug.ga_debug && println("----- Cardinality End -----\n")

  return ind
end

end