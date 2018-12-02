push!(LOAD_PATH, ".")

module GeneticAlgorithm

using Population
using Individual
using BinaryIndividual
using IntegerIndividual
using RealIndividual
using Fitness
using GAInitialization
using GASelection
using GACrossover
using GAMutation
using Debug
using Statistics
using Plots

include("utility.jl")

abstract type AbstractGeneticAlgorithm end

mutable struct GeneticAlgorithmType{N}
  pop::Population.PopulationType{IndType, N} where {IndType <: Individual.AbstractIndividual}
  pop_size::Integer
  best_solution::Vector{Population.IndFitType{IndType, N}} where {IndType <: Individual.AbstractIndividual}
  elite_size::Integer 
  init_args::NTuple{M, Any} where {M}
  sel_args::NTuple{M, Any} where {M}
  cross_args::NTuple{M, Any} where {M}
  mut_args::NTuple{M, Any} where {M}
  function GeneticAlgorithmType{N}(x...; 
                        init_args::NTuple{N1, Any} = (GAInitialization.InitRandom,),
                        sel_args::NTuple{N2, Any} = (GASelection.Tournament, 2),
                        cross_args::NTuple{N3, Any} = (GACrossover.PointCrossover, 0.95),
                        mut_args::NTuple{N4, Any} = (GAMutation.BitFlipMutation, 0.01)) where {N, N1, N2, N3, N4}
    args = build(N, x..., init_args, sel_args, cross_args, mut_args)
    new(args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8])
  end
end

function build(N::Integer, ind::IndType, fit::FitType, ps::Integer, es::Integer, 
               i_a::NTuple{N1, Any}, s_a::NTuple{N2, Any}, 
               c_a::NTuple{N3, Any}, m_a::NTuple{N4, Any}) where {IndType, FitType, N1, N2, N3, N4}
  ps > 1 || error("GA: Population size must be greater than 1")
  return Population.PopulationType{IndType, N}(ind, fit), ps, Vector{Population.IndFitType{IndType, N}}(), es, i_a, s_a, c_a, m_a
end

function getBestSolution(this::GeneticAlgorithmType)
  return this.best_solution
end

function evolve!(this::GeneticAlgorithmType{1}, num_it::Integer, log::Integer = 0)
  best_fitness = zeros(Float64, num_it)
  mean_fitness = zeros(Float64, num_it)
  if Debug.ga_plot
    gr()
    plt = plot(zeros(0, 2), zeros(0, 2), xlims=(1, num_it), label=["Best" "Mean"], xlabel="Generation", ylabel="Fitness", legend=:bottomright, show=true)
  end

  # Create initial population
  GAInitialization.initializePopulation!(this.pop, this.pop_size, this.init_args...)

  # Main loop
  @time for it = 1 : num_it
    if Debug.ga_debug
      println("----- Generation ", it, " -----\n")
      show(this.pop)
    end

    pop_ind = Population.getBaseInd(this.pop)
    pop_fit = Population.getFitFunction(this.pop)
    new_pop = typeof(this.pop)(pop_ind, pop_fit)
    
    # Population sorting
    cur_individuals = collect(this.pop) 
    sort!(cur_individuals; rev = true)

    # Selection
    child_num = this.pop_size - this.elite_size
    parents_group = GASelection.selectParents(cur_individuals, child_num, this.sel_args...)

    # Crossover
    childs = typeof(Population.getBaseInd(this.pop))[]
    for parents in parents_group
      res_ch = GACrossover.crossover(parents..., this.cross_args...)
      push!(childs, res_ch...)
      length(childs) < child_num || break
    end
    childs = childs[1:child_num]

    # Mutation
    for child in childs
      Population.insertIndividual!(new_pop, GAMutation.mutate!(child, this.mut_args...))
    end

    # Elitism
    for (ind, fit) in @view cur_individuals[1:this.elite_size]
      Population.insertIndividual!(new_pop, ind, fit)
    end

    # New generation
    Population.evalFitness!(new_pop)
    this.pop = new_pop
    cur_best_solution = maximum(this.pop)
    if isempty(this.best_solution) || cur_best_solution > this.best_solution[1]
      this.best_solution = [deepcopy(cur_best_solution)]
    end
    if log > 0 && (it - 1) % log == 0
      println(it, " -> Fitness: (", cur_best_solution[2], ", ", this.best_solution[1][2], ")")
    end

    # Statistics
    best_fitness[it] = cur_best_solution[2][1]
    mean_fitness[it] = mean([fit[1] for (ind, fit) in this.pop])
    if Debug.ga_plot
      push!(plt, [it], [best_fitness[it], mean_fitness[it]])
      gui()
    end
  end
  return best_fitness, mean_fitness
end

function evolve!(this::GeneticAlgorithmType{N}, num_it::Integer, log::Integer = 0; fun = tuple(ones(N)...)) where {N}
  gr()

  # Create initial population
  GAInitialization.initializePopulation!(this.pop, this.pop_size, this.init_args...)

  # Main loop
  @time for it = 1 : num_it
    if Debug.ga_debug
      println("----- Generation ", it, " -----\n")
      show(this.pop)
    end

    pop_ind = Population.getBaseInd(this.pop)
    pop_fit = Population.getFitFunction(this.pop)
    new_pop = typeof(this.pop)(pop_ind, pop_fit)

    # Population sorting
    cur_individuals = [(this.pop[i], i) for i in eachindex(this.pop)]
    sorted_individuals = Array{Int64, 1}[]
    while !isempty(cur_individuals)
      dominated = fill(false, length(cur_individuals))
      frontier = Int64[]
      for i in eachindex(cur_individuals)
        for j in eachindex(cur_individuals)
          if cur_individuals[j][1] > cur_individuals[i][1]
            dominated[i] = true
            break
          end
        end
        if !dominated[i]
          push!(frontier, cur_individuals[i][2])
        end
      end
      cur_individuals = cur_individuals[filter(x -> dominated[x], eachindex(cur_individuals))]
      push!(sorted_individuals, frontier)
    end
    cur_individuals = [(this.pop[ind][1], (Float64(-i),))
                       for i in eachindex(sorted_individuals) for ind in sorted_individuals[i]]
    println(typeof(cur_individuals))
    # Selection
    child_num = this.pop_size - this.elite_size
    parents_group = GASelection.selectParents(cur_individuals, child_num, this.sel_args...)

    # Crossover
    childs = typeof(Population.getBaseInd(this.pop))[]
    for parents in parents_group
      res_ch = GACrossover.crossover(parents..., this.cross_args...)
      push!(childs, res_ch...)
      length(childs) < child_num || break
    end
    childs = childs[1:child_num]

    # Mutation
    for child in childs
      Population.insertIndividual!(new_pop, GAMutation.mutate!(child, this.mut_args...))
    end

    # Elitism
    for ind in @view vcat(sorted_individuals...)[1:this.elite_size]
      Population.insertIndividual!(new_pop, this.pop[ind]...)
    end

    # New generation
    Population.evalFitness!(new_pop)
    this.pop = new_pop
    cur_best_solution = typeof(this.best_solution)()
    for ind in this.pop
      dominated = false
      for ind2 in this.pop
        if ind2 > ind
          dominated = true
          break
        end
      end
      if !dominated
        push!(cur_best_solution, ind)
      end
    end
    this.best_solution = dominationIntersection(this.best_solution, cur_best_solution) 
    if log > 0 && (it - 1) % log == 0
      println(it, " -> Fitness: (", length(this.best_solution), ")")
    end

    # Statistics
    scatter([x .* fun for x in getindex.(this.best_solution, 2)])
  end
  savefig("fitness.png")
end

function dominationIntersection(x::Array{Population.IndFitType{IndType, N}}, y::Array{Population.IndFitType{IndType, N}}) where {IndType, N}
  x_dominated = fill(false, length(x))
  y_dominated = fill(false, length(y))

  for indx in eachindex(x), indy in eachindex(y)
    if x[indx] > y[indy]
      y_dominated[indy] = true
    elseif y[indy] > x[indx]
      x_dominated[indx] = true
    end
  end

  x_non_dominated = x[filter(x -> !x_dominated[x], eachindex(x))]
  y_non_dominated = y[filter(x -> !y_dominated[x], eachindex(y))]
  res = [x_non_dominated..., y_non_dominated...]
  return res[1:min(length(res), 100)]
end

end