push!(LOAD_PATH, ".")

module GAMutation

using Debug
using Individual
using BinaryIndividual
using RealIndividual
using Parallel

abstract type BitFlipMutation end
abstract type UniformMutation end
abstract type SwapMutation end

function mutate!(ind::BinaryIndividual.AbstractBinaryIndividual, ::Type{BitFlipMutation},  mr::Float64)

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

function mutate!(ind::RealIndividual.AbstractRealIndividual, ::Type{UniformMutation}, mr::Float64, lb::Float64, ub::Float64)

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

function mutate!(ind::Individual.AbstractIndividual, ::Type{SwapMutation},  mr::Float64)

  Debug.ga_debug && println("----- Swap -----\n")
  Debug.ga_debug && println("Before mutation: ", ind[:])

  for i in eachindex(ind)
    pr = Parallel.threadRand()
    if pr < mr
      idx = Parallel.threadRand(1:(Individual.getNumGenes(ind) - 1)) + i
      if idx > Individual.getNumGenes(ind)
        idx = idx - Individual.getNumGenes(ind)
      end
      ind[i], ind[idx] = ind[idx], ind[i]
    end
  end

  Debug.ga_debug && println("After mutation: ", ind[:], "\n")
  Debug.ga_debug && println("----- Swap End -----\n")

  return ind
end

end