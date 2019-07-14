push!(LOAD_PATH, ".")

module GACrossover

using Debug
using Chromosome
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
  return child 
end

function crossover(p1::CardinalityChromosome.CardinalityChromosomeType, 
                   p2::CardinalityChromosome.CardinalityChromosomeType,
                   ::Type{CardinalityPOCrossover}, cr::Float64)
  gene_num = Chromosome.getNumGenes(p1)
  c1, c2 = deepcopy(p1), deepcopy(p2)
  
  Debug.ga_debug && println("----- Cardinality Special Crossover -----\n")

  pr = Parallel.threadRand()

  Debug.ga_debug && println("Test: ", pr, " < ", cr, "\n")

  if pr < cr 

    if Debug.ga_debug
      println("Parent 1: ", p1[:])
      println("Parent 2: ", p2[:], "\n")
    end

    bounds = Dict{Int64, Tuple{Float64, Float64}}()
    for i in eachindex(p1)
      gene, value = p1[i]
      if haskey(bounds, gene)
        v = bounds[gene][2]
        bounds[gene] = (min(v, value), max(v, value))
      else
        bounds[gene] = (0.0, value)
      end
    end

    k1 = Parallel.threadRand(1:gene_num)
    k2 = (k1 + Parallel.threadRand(1:(gene_num - 1)) % (gene_num + 1))
    x, y = k1, k2
    if k2 < k1
      x = k2 + 1
      y = k1
    end

    if Debug.ga_debug
      println("Cut points: $x $y\n")
    end

    function pmx(x, y, z, a, b)
      gene_map = Dict{Int64, Int64}()
      already_used = Set{Int64}()

      for i in eachindex(p1)
        gene_map[x[i][1]] = y[i][1]
      end

      for i = a : b
        z[i] = y[i]
        push!(already_used, z[i])
      end

      function setNotUsed(x, s, m)
        return in(x, s) ? setNotUsed(m[x], s, m) : x
      end

      for i = 1 : (a - 1)
        gene = setNotUsed(x[i], already_used, gene_map)
        z[i] = gene
        push!(already_used, gene)
      end
      for i = (b + 1) : length(x)
        gene = setNotUsed(x[i], already_used, gene_map)
        z[i] = gene
        push!(already_used, gene)
      end
    end

    pmx(p1, p2, c1, x, y)
    pmx(p2, p1, c2, x, y)

    function newWeight(x, a, b, bounds)
      for i = a : b
        lb, ub = bounds[x[i][1]]
        k = Parallel.threadRand()
        x[i][2] = lb + (ub - lb) * ((k - lb) / (ub - lb))
      end

      err = 1.0 - sum(getindex.(x, 2))
      δ = abs(err)
      if err < 0.0
        for gene in @view x[end:-1:2]
          lb, ub = bounds[gene[1]]
          Δ = min(Parallel.threadRand() * δ, gene[2] - lb)
          gene[2] = gene[2] - Δ
          δ = δ - Δ
          if δ < 1e-9
            break
          end
        end
        gene[1] = gene[1] - δ
      else
        for gene in @view x[1:end-1]
          lb, ub = bounds[gene[1]]
          Δ = min(Parallel.threadRand() * δ, ub - gene[2])
          gene[2] = gene[2] + Δ
          δ = δ - Δ
          if δ < 1e-9
            break
          end
        end
        gene[2] = gene[end] + δ
      end
    end

    newWeight(c1, x, y, bounds)
    newWeight(c2, x, y, bounds)

    try
      @assert abs(sum(getindex.(x, 2)) - 1.0) < 1e-9
    catch err
      println(sum(getindex.(x, 2)))
      throw(err)
    end

    if Debug.ga_debug
      println("Child 1: ", c1[:])
      println("Child 2: ", c2[:], "\n")
    end

  end
  return (c1, c2)
end

end