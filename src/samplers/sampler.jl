#################### Sampler ####################

const samplerfxargs = [(:model, Mamba.AbstractModel), (:block, Integer)]


#################### Types and Constructors ####################

type NullFunction end

type SamplingBlock{M<:AbstractModel} <: AbstractSamplingBlock
  model::M
  index::Int
  transform::Bool

  SamplingBlock{M}(model::M, index::Integer=0, transform::Bool=false) where M<: AbstractModel =
    new(model, index, transform)
end


Sampler(param::Symbol, args...) = Sampler([param], args...)

function Sampler(params::Vector{Symbol}, f::Function, tune::Any=Dict())
  Sampler(params, modelfx(samplerfxargs, f), tune, Symbol[])
end

# function SamplerVariate{T<:SamplerTune, U<:Real}(x::AbstractVector{U}, tune::T)
#   SamplerVariate{T}(x, tune)
# end
type WithIter end

function SamplerVariate{VS,T}(block::SamplingBlock{M}, pargs...; kargs...) where M where VS where T<:SamplerTune
  m = block.model
  SamplerVariate(
    #{vstype(M),T}
    WithIter, unlist(block), m.samplers[block.index], pargs...;
                 iter=m.iter, kargs...)
end

function SamplerVariate{VS<:AbstractVariateVals,T<:SamplerTune}(wi::Type{WithIter}, x::VS,
                                                 s::Sampler{T}, pargs...;
                                                 iter::Integer=1, kargs...)
  if iter == 1
    v = SamplerVariate{VS,T}(x, pargs...; kargs...)
    s.tune = v.tune
  else
    v = SamplerVariate{VS,T}(x, s.tune)
  end
  v
end


#################### Base Methods ####################

function Base.show(io::IO, s::Sampler)
  print(io, "An object of type \"$(summary(s))\"\n")
  print(io, "Sampling Block Nodes:\n")
  show(io, s.params)
  print(io, "\n\n")
  show(io, "text/plain", first(code_typed(s.eval)))
  println(io)
end

function Base.showall(io::IO, s::Sampler)
  show(io, s)
  print(io, "\nTuning Parameters:\n")
  show(io, s.tune)
  print(io, "\n\nTarget Nodes:\n")
  show(io, s.targets)
end


#################### Variate Validators ####################

validate(v::SamplerVariate) = v

function validatebinary(v::SamplerVariate)
  all(insupport(Bernoulli, v)) ||
    throw(ArgumentError("variate is not a binary vector"))
  v
end

function validatesimplex(v::SamplerVariate)
  isprobvec(v) || throw(ArgumentError("variate is not a probability vector"))
  v
end


#################### sample! Generics ####################

function sample!(v::SamplerVariate, density::Nullable; args...)
  isnull(density) && error("must specify a target density in $(typeof(v))",
                           " constructor or sample! method")
  sample!(v, get(density); args...)
end


#################### Simulation Methods ####################

function gradlogpdf!{T<:Real}(block::SamplingBlock, x::AbstractArray{T},
                              dtype::Symbol=:forward)
  gradlogpdf!(block.model, x, block.index, block.transform, dtype=dtype)
end

function logpdf!{T<:Real}(block::SamplingBlock, x::AbstractArray{T})
  logpdf!(block.model, x, block.index, block.transform)
end

function logpdfgrad!{T<:Real}(block::SamplingBlock, x::AbstractVector{T},
                              dtype::Symbol)
  grad = gradlogpdf!(block, x, dtype)
  logf = logpdf!(block, x)
  (logf, ifelse.(isfinite.(grad), grad, 0.0))
end

function unlist(block::SamplingBlock)
  unlist(block.model, block.index, block.transform)
end

function relist{T<:Real}(block::SamplingBlock, x::AbstractArray{T})
  relist(block.model, x, block.index, block.transform)
end

function relist{V<:AbstractArray{T} where T<:Real}(block::SamplingBlock, x::Variate{V})
  relist(block.model, x.value, block.index, block.transform)
end


#################### Auxiliary Functions ####################

asvec(x::Union{Number, Symbol}) = [x]
asvec(x::Vector) = x
