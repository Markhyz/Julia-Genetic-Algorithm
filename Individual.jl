push!(LOAD_PATH, ".")

module Individual

include("utility.jl")

abstract type AbstractIndividual{GeneType} <: AbstractArray{GeneType, 1} end

@define Individual begin
  chromosome::Vector{GeneType}
  num_genes::Integer
  bounds::Vector{Tuple{GeneType, GeneType}}  
end

function build(ng::Integer, b::Vector{Tuple{GeneType, GeneType}}) where {GeneType}
  return Vector{GeneType}(undef, ng), ng, b
end

function build(ind::AbstractIndividual)
  return deepcopy(ind.chromosome), deepcopy(ind.num_genes), deepcopy(ind.bounds)
end

function generateRandom!() end

function getGeneValue(this::AbstractIndividual, pos::Integer)
  return this.chromosome[pos]
end

function getNumGenes(this::AbstractIndividual)
  return this.num_genes
end

function getGeneBounds(this::AbstractIndividual, pos::Integer)
    return this.bounds[pos]
end

function setGeneValue!(this::AbstractIndividual{GeneType}, pos::Integer, value::GeneType) where {GeneType}
    return this.chromosome[pos] = value
end

function setGeneBounds!(this::AbstractIndividual{GeneType}, pos::Integer, bound::Vector{Tuple{GeneType, GeneType}}) where {GeneType}
    return this.bounds[pos] = bound
end

function Base.show(io::IO, this::AbstractIndividual)
  println(io, "$(this.num_genes) genes.")
  println(io, "Chromosome: ", this.chromosome)
  println(io, "Bounds: ", this.bounds)
end

function Base.size(this::AbstractIndividual)
  return (getNumGenes(this), )
end

function Base.similar(this::AbstractIndividual, ::Type{GeneType}, sz::Integer) where {GeneType}
  return Vector{GeneType}(undef, sz)
end

function Base.IndexStyle(::Type{<:AbstractIndividual})
  return IndexLinear()
end

function Base.getindex(this::AbstractIndividual, pos::Integer)
  return this.chromosome[pos]
end

function Base.setindex!(this::AbstractIndividual{GeneType}, value::GeneType, pos::Integer) where {GeneType}
  return this.chromosome[pos] = value
end

function Base.firstindex(this::AbstractIndividual)
  return 1
end

function Base.lastindex(this::AbstractIndividual)
  return getNumGenes(this)
end

end