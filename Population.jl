push!(LOAD_PATH, ".")

module Population

include("utility.jl")  

using Individual
using Fitness

IndFitType{IndType <: Individual.AbstractIndividual} = Tuple{IndType, NTuple{N, Float64} where N}
CrowdFitType{IndType <: Individual.AbstractIndividual} = Tuple{IndType, Int64, Float64}

abstract type AbstractPopulation{IndType <: Individual.AbstractIndividual} <: AbstractArray{Tuple{IndType, NTuple{N, Float64} where N}, 1} end

struct PopulationType{IndType} <: AbstractPopulation{IndType}
  ind_args::Tuple
  pop::Vector{IndType}
  fitness::Fitness.AbstractFitness{N} where N
  pop_fit::Vector{NTuple{N, Float64} where N}
  fit_refresh::Vector{Bool}
  function PopulationType{IndType}(x...) where {IndType}
    args = build(IndType, x...)
    new(args[1], args[2], args[3], args[4], args[5])
  end
end

function build(ind_type::Type, ind_args::Tuple, f::Fitness.AbstractFitness{N}) where N
  return ind_args, Vector{ind_type}(), f, Vector{NTuple{N, Float64}}(), Vector{Bool}()
end

function clear(this::AbstractPopulation{IndType}) where {IndType}
  empty!(this.pop)
  empty!(this.pop_fit)
  empty!(this.fit_refresh)
end

function getPopSize(this::AbstractPopulation)
  return length(this.pop)   
end

function insertIndividual!(this::AbstractPopulation{IndType}, ind::IndType, fit::NTuple{N, Float64} = (NaN,)) where {IndType, N}
  if N != Fitness.getSize(this.fitness)
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

function populateRandom!(this::AbstractPopulation{IndType}, num_ind::Int64) where {IndType}
  for i = 1 : num_ind
    new_ind = IndType(this.ind_args...)
    Individual.generateRandom!(new_ind)
    insertIndividual!(this, new_ind)
  end
end

function Base.show(io::IO, this::AbstractPopulation)
  println("Population ", getPopSize(this), "\n")
  for i = 1 : getPopSize(this)
    print("Individual ", i, ": [")
    for gene in this.pop[i]
      print(" ", gene)
    end
    println(" ] -> ", this.pop_fit[i])
  end
  print("\n")
end

function Base.size(this::AbstractPopulation)
  return (getPopSize(this), )
end

function Base.similar(this::AbstractPopulation, ::Type{IndFitType}, sz::Int64)
  return Vector{IndFitType}(undef, sz)
end

function Base.IndexStyle(::Type{<:AbstractPopulation})
  return IndexLinear()
end

function Base.getindex(this::AbstractPopulation, pos::Int64)
  return (this.pop[pos], this.pop_fit[pos])
end

function Base.setindex!(this::AbstractPopulation, value::IndFitType, pos::Int64)
  this.pop[pos], this.pop_fit[pos] = value
  this.fit_refresh[pos] = false
end

function Base.firstindex(this::AbstractPopulation)
  return 1
end

function Base.lastindex(this::AbstractPopulation)
  return getPopSize(this)
end

function Base.:<(x::IndFitType{IndType}, y::IndFitType{IndType}) where {IndType}
  fit_size = length(x[2])
  fit_size > 1 || return x[2] < y[2]
  x[2] != y[2] || return false
  for i in eachindex(x[2])
    x[2][i] <= y[2][i] || return false
  end
  return true
end

function Base.isless(x::IndFitType, y::IndFitType)
  return x < y
end

function Base. ==(x::IndFitType{IndType}, y::IndFitType{IndType}) where {IndType}
  return x[2] == y[2]
end

function Base.isequal(x::IndFitType, y::IndFitType)
  return x == y
end

function Base.:<(x::CrowdFitType{IndType}, y::CrowdFitType{IndType}) where {IndType}
  x[2] != y[2] || return x[3] < y[3]
  return x[2] > y[2]
end

function Base.isless(x::CrowdFitType, y::CrowdFitType)
  return x < y
end

function Base. ==(x::CrowdFitType{IndType}, y::CrowdFitType{IndType}) where {IndType}
  return x[2:3] ==  y[2:3]
end

function Base.isequal(x::CrowdFitType, y::CrowdFitType)
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