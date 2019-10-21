push!(LOAD_PATH, ".")

module GeneticAlgorithm

using Population
using Chromosome
using BinaryChromosome
using IntegerChromosome
using RealChromosome
using CardinalityChromosome
using Fitness
using GAInitialization
using GASelection
using GACrossover
using GAMutation
using Debug
using Statistics
using Plots
using LinearAlgebra
using DataStructures

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
    num_ch = -1
    for i = 1 : chromo_num
      ch_chromo = GACrossover.crossover(getindex.(parents, i)..., this.cross_args[i]...)
      num_ch = length(ch_chromo)
      push!(res_ch, ch_chromo)
    end
    res_ch = [tuple(getindex.(res_ch, i)...) for i = 1 : num_ch]
    push!(childs, res_ch...)
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
    new_pop, t1 = @timed newGeneration(this.pop)

    # Selection
    cur_individuals = [(this.pop[ind][1], i, pop_diversity[ind])
                       for i in eachindex(sorted_individuals) for ind in sorted_individuals[i]]         
    parents_group, t2 =  @timed selection(this, cur_individuals)

    # Crossover
    childs, t3 = @timed crossover(this, parents_group)

    # Mutation
    _, t4 = @timed mutation(this, new_pop, childs)

    # New generation evaluation
    _, t5 = @timed Population.evalFitness!(new_pop)
    old_new_pop = vcat(this.pop[:], new_pop[:])
    (sorted_individuals, pop_diversity), t6 = @timed nonDominatedSorting(this, old_new_pop)
    idx = 1
    for front in eachindex(sorted_individuals), ind in eachindex(sorted_individuals[front])
      this.pop[idx] = old_new_pop[sorted_individuals[front][ind]]
      sorted_individuals[front][ind] = idx
      idx = idx + 1
    end
    
    # Best solution
    this.best_solution = [this.pop[ind] for ind in sorted_individuals[1]]
    if log > 0 && (it - 1) % log == 0
      println(it, " -> Fitness: (", length(this.best_solution), ") ", t1, " ", t2, " ", t3, " ", t4, " ", t5, " ", t6)
    end
  end
  return this.best_solution
end

##### NSGA-II PO #####

function repairPO!(this::GeneticAlgorithmType, pop::Population.AbstractPopulation)
  k = Population.getFitFunction(pop).k
  for i = 1 : this.pop_size
    cur_ind = pop[i][1]
    gene_num = Chromosome.getNumGenes(cur_ind[1])
    for j = 1 : gene_num
      if cur_ind[2][j] < 1e-9
        cur_ind[2][j] = cur_ind[2][j] + 1e-9
      end
    end
    cur_k = sum(cur_ind[1])
    cur_w = sort([ (cur_ind[2][i], i) for i in eachindex(cur_ind[2]) ])
    
    j = 1
    while cur_k > k && j <= gene_num
      w, ind = cur_w[j]
      if cur_ind[1][ind] == 1
        cur_ind[1][ind] = 0
        cur_k = cur_k - 1
      end
      j = j + 1
    end
  
    j = gene_num
    while cur_k < k && j > 0
      w, ind = cur_w[j]
      if cur_ind[1][ind] == 0
        cur_ind[1][ind] = 1
        cur_k = cur_k + 1
      end
      j = j - 1
    end

    total_w = 0.0
    for j in eachindex(cur_ind[1])
      if cur_ind[1][j] == 1
        total_w = total_w + cur_ind[2][j]
      end
    end

    @assert total_w > 0

    for j in eachindex(cur_ind[1])
      if cur_ind[1][j] == 1
        cur_ind[2][j] = cur_ind[2][j] / total_w
      end
    end
    
    try
      @assert sum(cur_ind[1]) == k

      w_t = 0
      for j in eachindex(cur_ind[1])
        if cur_ind[1][j] == 1
          w_t = w_t + cur_ind[2][j]
        end
      end

      @assert abs(w_t - 1.0) < 1e-9
    catch err
      println(cur_ind[1][:], " ", cur_ind[2][:])
      println(sum(cur_ind[1]), ' ', sum(cur_ind[2]))
      throw(err)
    end
  end
end

function initializeNSGA2PO!(this::GeneticAlgorithmType)
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

  repairPO!(this, this.pop)
  
  Population.evalFitness!(this.pop)
end

function crossoverNSGA2PO(this::GeneticAlgorithmType, parents_group::Vector{Tuple{IndT, IndT}}) where {IndT}
  child_type = typeof(this.pop).parameters[1]
  childs = child_type[]
  for parents in parents_group
    res_ch = GACrossover.crossover(parents..., this.cross_args[1]...)
    res_ch2 = GACrossover.crossover(parents..., this.cross_args[1]...)
    push!(childs, res_ch, res_ch2)
    length(childs) < this.pop_size || break
  end
  childs = childs[1:this.pop_size]
  return childs
end

function evolveNSGA2PO!(this::GeneticAlgorithmType, num_it::Integer, log::Integer = 0)
  # Create initial population
 	initializeNSGA2PO!(this)

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
    childs = crossoverNSGA2PO(this, parents_group)

    # Mutation
    mutation(this, new_pop, childs)

    # New generation evaluation
    
    repairPO!(this, new_pop)
    
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

##### NSGA-III #####

function generateReferencePoints(obj_num::Integer, p::Integer)
  reference_points = Vector{Float64}[]
  point = zeros(obj_num)
  δ = 1.0 / p
  ε = 1e-12
  
  function stepIteration(obj_index::Integer, remaining::Float64)
    if obj_index == obj_num
      point[obj_index] = remaining
      push!(reference_points, [point...])
    elseif remaining < ε
      point[obj_index] = 0
      stepIteration(obj_index + 1, remaining)
    else
      aux = 0.0
      while remaining - aux > -ε
        point[obj_index] = aux
        stepIteration(obj_index + 1, remaining - aux)
        aux = aux + δ
      end
    end
  end

  stepIteration(1, 1.0)

  return reference_points
end

function pointLineDistance(line_point::Vector{Float64}, point::Vector{Float64})
  scale = (point ⋅ line_point) / (line_point ⋅ line_point)
  scaled_point = point - scale * line_point
  distance = 0.0
  for value in scaled_point
    distance = distance + value ^ 2
  end
  return sqrt(distance)
end

function referencePointNiching!(sorted_individuals::Vector{Vector{Int64}}, last_frontier::Vector{Int64}, 
                                individuals::Vector{<: Population.StandardFit}, reference_points::Vector{Vector{Float64}},
                                old_extreme_solutions::Vector{Vector{Float64}}, remaining_size::Integer)
  pop_size = length(individuals)
  new_population = vcat(sorted_individuals...)
  unset_pop = vcat(new_population, last_frontier) 

  min_fitness = [-[individuals[ind_idx][2]...] for ind_idx in unset_pop]
  
  fit_size = length(min_fitness[1])
  ideal_point = fill(1e9, fit_size)
  worst_point = fill(-1e9, fit_size)
  for fit in min_fitness
    for (idx, obj_value) in enumerate(fit)
      if obj_value < ideal_point[idx]
        ideal_point[idx] = obj_value
      end
      if obj_value > worst_point[idx]
        worst_point[idx] = obj_value
      end
    end
  end
 
  possible_extreme_solutions = vcat(min_fitness, old_extreme_solutions)
  extreme_solutions = []
  for obj_idx = 1 : fit_size
    extreme_idx = -1
    extreme_value = -1e9
    for (idx, fit) in enumerate(possible_extreme_solutions)
      if fit[obj_idx] > extreme_value
        extreme_idx = idx
        extreme_value = fit[obj_idx]
      end
    end
    push!(extreme_solutions, possible_extreme_solutions[extreme_idx])
  end
  for (index, solution) in enumerate(extreme_solutions)
    if index <= length(old_extreme_solutions)
      old_extreme_solutions[index] = solution
    else
      push!(old_extreme_solutions, solution)
    end
  end
  extreme_solutions = [extreme_solution - ideal_point for extreme_solution in extreme_solutions]

  failSolve = false
  x = ones(length(extreme_solutions))
  nadir_point = []
  try
    plane = extreme_solutions \ x
    intercepts = [1 / k for k in plane]
    if any(x -> x <= 1e-6, intercepts)
      throw()
    end
    nadir_point = ideal_point + intercepts
  catch
    failSolve = true
  end

  if failSolve
    nadir_point = worst_point
  end

  normalized_fitness = []
  for fit in min_fitness
    push!(normalized_fitness, [(fit[idx] - ideal_point[idx]) / nadir_point[idx] for idx in eachindex(fit)])
  end

  niche_dist = zeros(pop_size)
  niche_point = zeros(Int64, pop_size)
  for idx in eachindex(unset_pop)
    fit = normalized_fitness[idx]
    closest_dist = 1e9
    closest_point = -1
    for (point_idx, ref_point) in enumerate(reference_points)
      dist = pointLineDistance(ref_point, fit)
      if dist < closest_dist
        closest_dist = dist
        closest_point = point_idx
      end
    end
    niche_dist[unset_pop[idx]] = closest_dist
    niche_point[unset_pop[idx]] = closest_point
  end
  niche_count = zeros(Int64, length(reference_points))
  for ind in new_population
    selected_point = niche_point[ind]
    niche_count[selected_point] = niche_count[selected_point] + 1
  end
  reference_points_queue = [PriorityQueue{Int64, Float64}() for i in eachindex(reference_points)]
  for ind in last_frontier
    selected_point = niche_point[ind]
    enqueue!(reference_points_queue[selected_point], ind, niche_dist[ind])
  end

  point_queue = PriorityQueue{Int64, Int64}()
  for (idx, count) in enumerate(niche_count)
    enqueue!(point_queue, idx, count)
  end
  selected_individuals = []
  while length(selected_individuals) < remaining_size
    point = dequeue!(point_queue)
    if length(reference_points_queue[point]) > 0
      selected_ind = dequeue!(reference_points_queue[point])
      push!(selected_individuals, selected_ind)
      niche_count[point] = niche_count[point] + 1
      enqueue!(point_queue, point, niche_count[point])
    end
  end
  return selected_individuals
end

function adjustedNonDominatedSorting(this::GeneticAlgorithmType, individuals::Vector{<: Population.StandardFit}, 
                                     reference_points::Vector{Vector{Float64}}, extreme_solutions::Vector{Vector{Float64}})
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
    selected_individuals = referencePointNiching!(sorted_individuals, frontier, individuals, reference_points, extreme_solutions, this.pop_size - new_pop_size)
    push!(sorted_individuals, selected_individuals)
	end
  
  pop_diver = [0.0 for front in sorted_individuals for ind in front]
  return sorted_individuals, pop_diver
end

function evolveNSGA3!(this::GeneticAlgorithmType, ref_p::Integer, num_it::Integer, log::Integer = 0)
  num_obj = Fitness.getSize(Population.getFitFunction(this.pop))
  reference_points = generateReferencePoints(num_obj, ref_p)
  extreme_solutions = Vector{Float64}[]

  # Create initial population
 	initialize(this)

  # Initial population sorting
  sorted_individuals, pop_diversity = adjustedNonDominatedSorting(this, this.pop[:], reference_points, extreme_solutions)

  # Main loop
  @time for it = 1 : num_it
    if Debug.ga_debug
      println("----- Generation ", it, " -----\n")
      show(this.pop)
    end
    
    # Create clean population
    new_pop, t1 = @timed newGeneration(this.pop)

    # Selection
    cur_individuals = [(this.pop[ind][1], i, pop_diversity[ind])
                       for i in eachindex(sorted_individuals) for ind in sorted_individuals[i]]         
    parents_group, t2 =  @timed selection(this, cur_individuals)

    # Crossover
    childs, t3 = @timed crossover(this, parents_group)

    # Mutation
    _, t4 = @timed mutation(this, new_pop, childs)

    # New generation evaluation
    _, t5 = @timed Population.evalFitness!(new_pop)
    old_new_pop = vcat(this.pop[:], new_pop[:])
    (sorted_individuals, pop_diversity), t6 = @timed adjustedNonDominatedSorting(this, old_new_pop, reference_points, extreme_solutions)
    idx = 1
    for front in eachindex(sorted_individuals), ind in eachindex(sorted_individuals[front])
      this.pop[idx] = old_new_pop[sorted_individuals[front][ind]]
      sorted_individuals[front][ind] = idx
      idx = idx + 1
    end
    
    # Best solution
    this.best_solution = [this.pop[ind] for ind in sorted_individuals[1]]
    if log > 0 && (it - 1) % log == 0
      println(it, " -> Fitness: (", length(this.best_solution), ") ", t1, " ", t2, " ", t3, " ", t4, " ", t5, " ", t6)
    end
  end
  return this.best_solution
end

end