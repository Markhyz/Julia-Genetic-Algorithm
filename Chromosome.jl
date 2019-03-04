push!(LOAD_PATH, ".")

module Chromosome

include("utility.jl")

abstract type AbstractChromosome{GeneType} <: AbstractArray{GeneType, 1} end

@define Chromosome begin
  chromosome::Vector{GeneType}
  num_genes::Integer
  bounds::Vector{Tuple{GeneType, GeneType}}  
end

function build(ng::Integer, b::Vector{Tuple{GeneType, GeneType}}) where {GeneType}
  return Vector{GeneType}(undef, ng), ng, b
end

function build(ind::AbstractChromosome)
  return deepcopy(ind.chromosome), deepcopy(ind.num_genes), deepcopy(ind.bounds)
end

function generateRandom!() end

function getGeneValue(this::AbstractChromosome, pos::Integer)
  return this.chromosome[pos]
end

function getNumGenes(this::AbstractChromosome)
  return this.num_genes
end

function getGeneBounds(this::AbstractChromosome, pos::Integer)
    return this.bounds[pos]
end

function setGeneValue!(this::AbstractChromosome{GeneType}, pos::Integer, value::GeneType) where {GeneType}
    return this.chromosome[pos] = value
end

function setGeneBounds!(this::AbstractChromosome{GeneType}, pos::Integer, bound::Vector{Tuple{GeneType, GeneType}}) where {GeneType}
    return this.bounds[pos] = bound
end

function Base.show(io::IO, this::AbstractChromosome)
  println(io, "$(this.num_genes) genes.")
  println(io, "Chromosome: ", this.chromosome)
  println(io, "Bounds: ", this.bounds)
end

function Base.size(this::AbstractChromosome)
  return (getNumGenes(this), )
end

function Base.similar(this::AbstractChromosome, ::Type{GeneType}, sz::Integer) where {GeneType}
  return Vector{GeneType}(undef, sz)
end

function Base.IndexStyle(::Type{<:AbstractChromosome})
  return IndexLinear()
end

function Base.getindex(this::AbstractChromosome, pos::Integer)
  return this.chromosome[pos]
end

function Base.setindex!(this::AbstractChromosome{GeneType}, value::GeneType, pos::Integer) where {GeneType}
  return this.chromosome[pos] = value
end

function Base.firstindex(this::AbstractChromosome)
  return 1
end

function Base.lastindex(this::AbstractChromosome)
  return getNumGenes(this)
end

end