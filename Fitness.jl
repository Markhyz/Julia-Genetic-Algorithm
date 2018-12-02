push!(LOAD_PATH, ".")

module Fitness

include("utility.jl")

using Individual
using RealIndividual

abstract type AbstractFitness end

struct NullFit <: AbstractFitness end
function (::NullFit)()
  return (0.0, )
end

struct TestFit <: AbstractFitness end
function (::TestFit)(x::Individual.AbstractIndividual)
  return (convert(Float64, sum(x)),)
end

struct AckleyFit <: AbstractFitness end
function (::AckleyFit)(x::RealIndividual.AbstractRealIndividual)
  a, b, c, d = 20.0, 0.2, 2π, Individual.getNumGenes(x)
  s1, s2 = foldl(((x1, x2), y) -> (x1 + y ^ 2, x2 + cos(c * y)), x; init = (0, 0))
  res = -(-a * exp(-b * sqrt((1 / d) * s1)) - exp((1 / d) * s2) + a + exp(1.0))
  return (res,)
end

struct SchFit <: AbstractFitness end
function (::SchFit)(x::RealIndividual.AbstractRealIndividual)
  res1 = x[1] ^ 2
  res2 = (x[1] - 2) ^ 2
  return (-res1, -res2)
end

struct KurFit <: AbstractFitness end
function (::KurFit)(x::RealIndividual.AbstractRealIndividual)
  res1 = 0.0
  res2 = 0.0
  for i = 1:(length(x) - 1)
    res1 += -10 * exp(-0.2 * sqrt(x[i] ^ 2 + x[i + 1] ^ 2)) 
  end
  for gene in x
    res2 += abs(gene) ^ 0.8 + 5 * sin(gene ^ 3)
  end 
  return (-res1, -res2)
end

struct Zdt2Fit <: AbstractFitness end
function (::Zdt2Fit)(x::Individual.AbstractIndividual)
  res1 = x[1]
  k = 1 + 9 * sum(@view x[2:end]) / (Individual.getNumGenes(x) - 1)
  res2 = k * (1 - (x[1] / k) ^ 2)
  return (-res1, -res2)
end

struct ViennetFit <: AbstractFitness end
function (::ViennetFit)(v::Individual.AbstractIndividual)
  x, y = v[1], v[2]
  res1 = 0.5 * (x ^ 2 + y ^ 2) + sin(x ^ 2 + y ^ 2)
  res2 = ((3 * x - 2 * y + 4) ^ 2) / 8 + ((x - y + 1) ^ 2) / 27 + 15
  res3 = 1 / (x ^ 2 + y ^ 2 + 1) - 1.1 * exp(-(x ^ 2 + y ^ 2))
  return (-res1, -res2, -res3)
end

struct Zdt3Fit <: AbstractFitness end
function (::Zdt3Fit)(x::Individual.AbstractIndividual)
  res1 = x[1]
  g = 1 + 9 * sum(@view x[2:end]) / (Individual.getNumGenes(x) - 1)
  h = 1 - sqrt(x[1] / g) - (x[1] / g) * sin(10 * π * x[1])
  res2 = g * h
  return (-res1, -res2)
end

end