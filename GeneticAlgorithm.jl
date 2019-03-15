push!(LOAD_PATH, ".")

module GeneticAlgorithm

using Population
using Chromosome
using BinaryChromosome
using IntegerChromosome
using RealChromosome
using Fitness
using GAInitialization
using GASelection
using GACrossover
using GAMutation
using Debug
using Statistics
using Plots

include("utility.jl")

mutable struct GeneticAlgorithmType
  pop::Population.PopulationType
  pop_size::Integer
  best_solution::Vector{Population.StandardFit{<: Population.IndividualType}}
  elite_size::Integer 
  init_args::Tuple
  sel_args::Tuple
  cross_args::Tuple
  mut_args::Tuple
  function GeneticAlgorithmType(x...; 
                        init_args::Tuple = (GAInitialization.InitRandom,),
                        sel_args::Tuple = (GASelection.Tournament, 2),
                        cross_args::Tuple = (GACrossover.PointCrossover, 0.95),
                        mut_args::Tuple = (GAMutation.BitFlipMutation, 0.01))
    args = build(x..., init_args, sel_args, cross_args, mut_args)
    new(args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8])
  end
end

function build(ind_args::Tuple, fit::Fitness.AbstractFitness, ps::Integer, es::Integer, 
               i_a::Tuple, s_a::Tuple, c_a::Tuple, m_a::Tuple)
  ps > 1 || error("GA: Population size must be greater than 1")
  pop_type = Tuple{getindex.(ind_args, 1)...}
  return Population.PopulationType{pop_type}(tuple([v[2:end] for v in ind_args]...), fit), ps, Vector{Population.StandardFit{pop_type}}(), es, i_a, s_a, c_a, m_a
end

function getBestSolution(this::GeneticAlgorithmType)
  return this.best_solution
end

function initialize(this::GeneticAlgorithmType)
  Population.clear(this.pop)
  pop_type = typeof(this.pop).parameters[1]
  chromo_types = pop_type.parameters
  num_chromo = length(chromo_types)
  for i = 1 : this.pop_size
    new_ind = Vector{Chromosome.AbstractChromosome}(undef, num_chromo)
    for j = 1 : num_chromo
      new_ind[j] = chromo_types[j](Population.getIndArgs(this.pop)[j]...)
      GAInitialization.initializeChromosome!(new_ind[j], this.init_args[j]...)
    end
    Population.insertIndividual!(this.pop, tuple(new_ind...))
  end
  Population.evalFitness!(this.pop)
end

function newGeneration(pop::PopT) where {PopT}
  ind_args = Population.getIndArgs(pop)
  pop_fit = Population.getFitFunction(pop)
  new_pop = PopT(ind_args, pop_fit)
  return new_pop
end

function selection(this::GeneticAlgorithmType, cur_individuals::Vector{<: Population.GeneticAlgorithmFit{IndT}}) where {IndT}
  parents_group = GASelection.selectParents(cur_individuals, this.pop_size, this.sel_args...)
  return parents_group
end

function crossover(this::GeneticAlgorithmType, parents_group::Vector{Tuple{IndT, IndT}}) where {IndT}
  child_type = typeof(this.pop).parameters[1]
  childs = child_type[]
  chromo_num = length(child_type.parameters)
  for parents in parents_group
    res_ch = []
    for i = 1 : chromo_num
      ch_chromo = GACrossover.crossover(getindex.(parents, i)..., this.cross_args[i]...)
      push!(res_ch, ch_chromo)
    end
    push!(childs, [tuple(getindex.(res_ch, i)...) for i = 1:chromo_num]...)
    length(childs) < this.pop_size || break
  end
  childs = childs[1:this.pop_size]
  return childs
end

function mutation(this::GeneticAlgorithmType, new_pop::Population.AbstractPopulation, childs::Vector{IndT}) where {IndT}
  child_type = typeof(this.pop).parameters[1]
  chromo_num = length(child_type.parameters)
  for child in childs
    mut_child = Array{Chromosome.AbstractChromosome}(undef, chromo_num)
    for i = 1:chromo_num
      mut_child[i] = GAMutation.mutate!(child[i], this.mut_args[i]...)
    end
    Population.insertIndividual!(new_pop, tuple(mut_child...))
  end
end

function evalNewGen(this::GeneticAlgorithmType, new_pop::Population.AbstractPopulation)
  Population.evalFitness!(new_pop)
  this.pop[:] = new_pop[:]
end

### Single Objective GA ###

function evolveSO!(this::GeneticAlgorithmType, num_it::Integer, log::Integer = 0)
  best_fitness = zeros(Float64, num_it)
  mean_fitness = zeros(Float64, num_it)
  if Debug.ga_plot
    gr()
    plt = plot(zeros(0, 2), zeros(0, 2), xlims=(1, num_it), label=["Best" "Mean"], xlabel="Generation", ylabel="Fitness", legend=:bottomright, show=true)
  end

  # Create initial population
	initialize(this)

  # Main loop
  @time for it = 1 : num_it
    if Debug.ga_debug
      println("----- Generation ", it, " -----\n")
      show(this.pop)
    end

    # Create clean population
		new_pop = newGeneration(this.pop)

    # Population sorting
    cur_individuals = collect(this.pop) 
    sort!(cur_individuals; rev = true)

    # Selection
    parents_group = selection(this, cur_individuals)

    # Crossover
    childs = crossover(this, parents_group)

    # Mutation
   	mutation(this, new_pop, childs)

    # Elitism
    for ind = 1:this.elite_size
      new_pop[ind] = cur_individuals[ind]
    end

    # New generation evaluation
    evalNewGen(this, new_pop)

    # Best solution
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

### Multi Objective GA ###

function dominationFrontier(this::GeneticAlgorithmType)
  frontier = typeof(this.best_solution)()
  for ind in this.pop
    dominated = false
    for ind2 in this.pop
      if ind2 > ind
        dominated = true
        break
      end
    end
    if !dominated
      push!(frontier, ind)
    end
  end
  return frontier
end

function dominationIntersection(x::Array{Population.StandardFit{IndT}}, y::Array{Population.StandardFit{IndT}}, max_size::Integer) where {IndT}
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
  return res[1:min(length(res), max_size)]
end

function evolveMO!(this::GeneticAlgorithmType, num_it::Integer, log::Integer = 0)
  # Create initial population
 	initialize(this)

  # Main loop
  @time for it = 1 : num_it
    if Debug.ga_debug
      println("----- Generation ", it, " -----\n")
      show(this.pop)
    end

    # Create clean population
    new_pop = newGeneration(this.pop)

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
    cur_individuals = [(this.pop[ind][1], i, 0.0)
                       for i in eachindex(sorted_individuals) for ind in sorted_individuals[i]]       
   
    # Selection
    parents_group = selection(this, cur_individuals)

    # Crossover
    childs = crossover(this, parents_group)

    # Mutation
    mutation(this, new_pop, childs)

    # Elitism
    elite = vcat(sorted_individuals...)
    for ind = 1:this.elite_size
      new_pop[ind] = this.pop[elite[ind]]
    end

    # New generation evaluation
    evalNewGen(this, new_pop)

    # Best solution
    cur_best_solution = dominationFrontier(this)
    this.best_solution = dominationIntersection(this.best_solution, cur_best_solution, Population.getPopSize(this.pop)) 
    if log > 0 && (it - 1) % log == 0
      println(it, " -> Fitness: (", length(this.best_solution), ")")
    end
  end
  return this.best_solution
end

### NSGA II ###

function crowdDistance!(diver::Vector{Float64}, front::Vector{Int64}, individuals::Vector{<: Population.StandardFit}, this::GeneticAlgorithmType)
  for k = 1:Fitness.getSize(Population.getFitFunction(this.pop))
    k_fit_order = [(individuals[ind][2][k], ind) for ind in front]
    sort!(k_fit_order)
    diver[k_fit_order[1][2]] = Inf
    k_min = k_fit_order[1][1]
    diver[k_fit_order[end][2]] = Inf
    k_max = k_fit_order[end][1]
    for idx = 2:(length(k_fit_order) - 1)
			ind = k_fit_order[idx][2]
      diver[ind] = diver[ind] + (k_fit_order[idx + 1][1] - k_fit_order[idx - 1][1]) / (k_max - k_min)
    end
  end
end

function nonDominatedSorting(this::GeneticAlgorithmType, individuals::Vector{<: Population.StandardFit})
  sorted_individuals = Array{Int64, 1}[]
  domination = fill(Int64[], length(individuals))
  dominated = zeros(length(individuals))
  diversity = zeros(Float64, length(individuals))
  frontier = Int64[]

  dominate = function (i, j)
    if individuals[i] > individuals[j]
      dominated[j] = dominated[j] + 1
      push!(domination[i], j)
    end
  end
  for i in eachindex(individuals)
    for j = (i + 1):length(individuals)
    	dominate(i, j)
      dominate(j, i)
    end
    if dominated[i] == 0
      push!(frontier, i)
    end
  end

  new_pop_size = 0
  while !isempty(frontier) && new_pop_size + length(frontier) <= this.pop_size
    new_frontier = Int64[]
    crowdDistance!(diversity, frontier, individuals, this)
    new_pop_size = new_pop_size + length(frontier)
    for ind in frontier
      for i in domination[ind]
        dominated[i] = dominated[i] - 1
        if dominated[i] == 0
          push!(new_frontier, i)
        end
      end
    end
    push!(sorted_individuals, frontier)
    frontier = new_frontier
  end
  if new_pop_size < this.pop_size
    crowdDistance!(diversity, frontier, individuals, this)
		remain = this.pop_size - new_pop_size
		cur_ind = [(diversity[ind], ind) for ind in frontier]
		sort!(cur_ind; rev = true)
		push!(sorted_individuals, [ind for (diver, ind) in @view cur_ind[1:remain]])
	end
  
  pop_diver = [diversity[ind] for front in sorted_individuals for ind in front]
  return sorted_individuals, pop_diver
end

# Main algorithm

function evolveNSGA2!(this::GeneticAlgorithmType, num_it::Integer, log::Integer = 0)
  # Create initial population
 	initialize(this)

  # Initial population sorting
  sorted_individuals, pop_diversity = nonDominatedSorting(this, this.pop[:])

  # Main loop
  @time for it = 1 : num_it
    if Debug.ga_debug
      println("----- Generation ", it, " -----\n")
      show(this.pop)
    end
    
    # Create clean population
    new_pop = newGeneration(this.pop)

    # Selection
    cur_individuals = [(this.pop[ind][1], i, pop_diversity[ind])
                       for i in eachindex(sorted_individuals) for ind in sorted_individuals[i]]         
    parents_group = selection(this, cur_individuals)

    # Crossover
    childs = crossover(this, parents_group)

    # Mutation
    mutation(this, new_pop, childs)

    # New generation evaluation
    Population.evalFitness!(new_pop)
    old_new_pop = vcat(this.pop[:], new_pop[:])
    sorted_individuals, pop_diversity = nonDominatedSorting(this, old_new_pop)
    idx = 1
    for front in eachindex(sorted_individuals), ind in eachindex(sorted_individuals[front])
      this.pop[idx] = old_new_pop[sorted_individuals[front][ind]]
      sorted_individuals[front][ind] = idx
      idx = idx + 1
    end
    
    # Best solution
    this.best_solution = [this.pop[ind] for ind in sorted_individuals[1]]
    if log > 0 && (it - 1) % log == 0
      println(it, " -> Fitness: (", length(this.best_solution), ")")
    end
  end
  return this.best_solution
end

# NSGA-II PO

function evolveNSGA2PO!(this::GeneticAlgorithmType, num_it::Integer, log::Integer = 0)
  # Create initial population
 	initialize(this)

  # Initial population sorting
  sorted_individuals, pop_diversity = nonDominatedSorting(this, this.pop[:])

  # Main loop
  @time for it = 1 : num_it
    if Debug.ga_debug
      println("----- Generation ", it, " -----\n")
      show(this.pop)
    end
    
    # Create clean population
    new_pop = newGeneration(this.pop)

    # Selection
    cur_individuals = [(this.pop[ind][1], i, pop_diversity[ind])
                       for i in eachindex(sorted_individuals) for ind in sorted_individuals[i]]         
    parents_group = selection(this, cur_individuals)

    # Crossover
    childs = crossover(this, parents_group)

    # Mutation
    mutation(this, new_pop, childs)

    # New generation evaluation
    Population.evalFitness!(new_pop)
    old_new_pop = vcat(this.pop[:], new_pop[:])
    sorted_individuals, pop_diversity = nonDominatedSorting(this, old_new_pop)
    idx = 1
    for front in eachindex(sorted_individuals), ind in eachindex(sorted_individuals[front])
      this.pop[idx] = old_new_pop[sorted_individuals[front][ind]]
      sorted_individuals[front][ind] = idx
      idx = idx + 1
    end
    
    # Best solution
    this.best_solution = [this.pop[ind] for ind in sorted_individuals[1]]
    if log > 0 && (it - 1) % log == 0
      println(it, " -> Fitness: (", length(this.best_solution), ")")
    end
  end
  return this.best_solution
end

end