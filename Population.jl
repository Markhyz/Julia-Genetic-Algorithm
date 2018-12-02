push!(LOAD_PATH, ".")

module Population

include("utility.jl")  

using Individual
using Fitness

IndFitType{IndType, N} = Tuple{IndType, NTuple{N, Float64}} where {IndType <: Individual.AbstractIndividual, N}

abstract type AbstractPopulation{IndType <: Individual.AbstractIndividual, N} <: AbstractArray{Tuple{IndType, NTuple{N, Float64}} , 1} end

mutable struct PopulationType{IndType, N} <: AbstractPopulation{IndType, N}
  base_ind::IndType
  pop::Vector{IndType}
  fitness::FitType where {FitType <: Fitness.AbstractFitness}
  pop_fit::Vector{NTuple{N, Float64}}
  fit_refresh::Vector{Bool}
  function PopulationType{IndType, N}(x...) where {IndType, N}
    args = build(N, x...)
    new(args[1], args[2], args[3], args[4], args[5])
  end
end

function build(N::Int64, base::IndType, f::FitType) where {IndType, FitType}
  return base, Vector{IndType}(), f, Vector{NTuple{N, Float64}}(), Vector{Bool}()
end

function getPopSize(this::AbstractPopulation)
  return length(this.pop)   
end

function insertIndividual!(this::AbstractPopulation{IndType, N}, ind::IndType, fit::NTuple{N, Float64} = tuple(fill(NaN, N)...)) where {IndType, N}
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

function getBaseInd(this::AbstractPopulation)
  return this.base_ind
end

function getFitFunction(this::AbstractPopulation)
  return this.fitness
end

function populateRandom!(this::AbstractPopulation{IndType}, num_ind::Int64) where {IndType}
  for i = 1 : num_ind
    new_ind = IndType(this.base_ind)
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
end

function Base.firstindex(this::AbstractPopulation)
  return 1
end

function Base.lastindex(this::AbstractPopulation)
  return getPopSize(this)
end

function Base.:<(x::IndFitType{IndType, N}, y::IndFitType{IndType, N}) where {IndType, N}
  N > 1 || return x[2][1] < y[2][1]
  x[2] != y[2] || return false
  for i in eachindex(x[2])
    x[2][i] <= y[2][i] || return false
  end
  return true
end

function Base.isless(x::IndFitType, y::IndFitType)
  return x < y
end

function Base.maximum(this::AbstractPopulation)
  res = this[1]
  for ind in @view this[2:end]
    res = max(ind, res)
  end
  return res
end

end