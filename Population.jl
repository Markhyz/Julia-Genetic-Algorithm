push!(LOAD_PATH, ".")

module Population

include("utility.jl")  

using Chromosome
using Fitness

IndividualType = NTuple{N, Chromosome.AbstractChromosome} where N

StandardFit{IndT <: IndividualType} = Tuple{IndT, Fitness.FitnessType}
NSGAFit{IndT <: IndividualType} = Tuple{IndT, Int64, Float64}

GeneticAlgorithmFit{IndT <: IndividualType} = Union{StandardFit{IndT}, NSGAFit{IndT}}

abstract type AbstractPopulation{IndT <: IndividualType} <: AbstractArray{StandardFit{IndT}, 1} end

struct PopulationType{IndT} <: AbstractPopulation{IndT}
  ind_args::Tuple
  pop::Vector{IndT}
  fitness::Fitness.AbstractFitness{N} where N
  pop_fit::Vector{Fitness.FitnessType}
  fit_refresh::Vector{Bool}
  function PopulationType{IndT}(x...) where {IndT}
    args = build(IndT, x...)
    new(args[1], args[2], args[3], args[4], args[5])
  end
end

function build(ind_type::Type, ind_args::Tuple, f::Fitness.AbstractFitness{N}) where N
  return ind_args, Vector{ind_type}(), f, Vector{Fitness.FitnessType}(), Vector{Bool}()
end

function clear(this::AbstractPopulation{IndT}) where {IndT}
  empty!(this.pop)
  empty!(this.pop_fit)
  empty!(this.fit_refresh)
end

function getPopSize(this::AbstractPopulation)
  return length(this.pop)   
end

function insertIndividual!(this::AbstractPopulation{IndT}, ind::IndT, fit::Union{Fitness.FitnessType, Nothing} = nothing) where {IndT}
  if fit == nothing
    fit = tuple(fill(NaN, Fitness.getSize(this.fitness))...)
  end
  push!(this.pop, ind)
  push!(this.pop_fit, fit)
  push!(this.fit_refresh, fit[1] === NaN)
end

function getIndividual(this::AbstractPopulation, pos::Int64)
  return (this.pop[pos], this.pop_fit[pos])
end

function refreshIndividual!(this::AbstractPopulation, pos::Int64)
  if this.fit_refresh[pos]
    this.fit_refresh[pos] = false
    this.pop_fit[pos] = this.fitness(this.pop[pos])
  end
end

function evalFitness!(this::AbstractPopulation)
  for i = 1 : length(this.pop)
    refreshIndividual!(this, i)
  end
end

function getIndArgs(this::AbstractPopulation)
  return this.ind_args
end

function getFitFunction(this::AbstractPopulation)
  return this.fitness
end

function Base.show(io::IO, this::AbstractPopulation)
  println("Population ", getPopSize(this), "\n")
  for i = 1 : getPopSize(this)
    println("Chromosome ", i, ": [")
    for chromosome in this.pop[i]
      println(chromosome[:])
    end
    println(" ] -> ", this.pop_fit[i])
  end
  print("\n")
end

function Base.size(this::AbstractPopulation)
  return (getPopSize(this), )
end

function Base.similar(this::AbstractPopulation, ::Type{StandardFit}, sz::Int64)
  return Vector{StandardFit}(undef, sz)
end

function Base.IndexStyle(::Type{<:AbstractPopulation})
  return IndexLinear()
end

function Base.getindex(this::AbstractPopulation, pos::Int64)
  return (this.pop[pos], this.pop_fit[pos])
end

function Base.setindex!(this::AbstractPopulation, value::StandardFit, pos::Int64)
  this.pop[pos], this.pop_fit[pos] = value
  this.fit_refresh[pos] = false
end

function Base.firstindex(this::AbstractPopulation)
  return 1
end

function Base.lastindex(this::AbstractPopulation)
  return getPopSize(this)
end

function Base.:<(x::StandardFit{IndT}, y::StandardFit{IndT}) where {IndT}
  fit_size = length(x[2])
  fit_size > 1 || return x[2] < y[2]
  x[2] != y[2] || return false
  for i in eachindex(x[2])
    x[2][i] <= y[2][i] || return false
  end
  return true
end

function Base.isless(x::StandardFit, y::StandardFit)
  return x < y
end

function Base. ==(x::StandardFit{IndT}, y::StandardFit{IndT}) where {IndT}
  return x[2] == y[2]
end

function Base.isequal(x::StandardFit, y::StandardFit)
  return x == y
end

function Base.:<(x::NSGAFit{IndT}, y::NSGAFit{IndT}) where {IndT}
  x[2] != y[2] || return x[3] < y[3]
  return x[2] > y[2]
end

function Base.isless(x::NSGAFit, y::NSGAFit)
  return x < y
end

function Base. ==(x::NSGAFit{IndT}, y::NSGAFit{IndT}) where {IndT}
  return x[2:3] ==  y[2:3]
end

function Base.isequal(x::NSGAFit, y::NSGAFit)
  return x == y
end

function Base.maximum(this::AbstractPopulation)
  res = this[1]
  for ind in @view this[2:end]
    res = max(ind, res)
  end
  return res
end

end