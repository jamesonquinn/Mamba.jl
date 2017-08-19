#################### Discrete Gibbs Sampler ####################

#################### Types and Constructors ####################

const DGSUnivariateDistribution =
          Union{Bernoulli, Binomial, Categorical, DiscreteUniform,
                Hypergeometric, NoncentralHypergeometric}


const DSForm = Union{Function, Vector{Float64}}

type DSTune{F<:DSForm} <: SamplerTune
  mass::Nullable{F}
  support::Matrix{Real}

  DSTune{F}() where F<:DSForm = new()

  DSTune{F}(x::Vector, support::AbstractMatrix) where F<:DSForm =
    new(Nullable{Function}(), support)

  DSTune{F}(x::Vector, support::AbstractMatrix, mass::Function) where
    F<:DSForm = new(Nullable{F}(mass), support)
end


const DGSVariate = FlatSamplerVariate{DSTune{Function}}
const DiscreteVariate = FlatSamplerVariate{DSTune{Vector{Float64}}}
const EitherDGSVariate = Union{DGSVariate, DiscreteVariate}

validate(v::DGSVariate) = validate(v, v.tune.support)

function validate(v::DiscreteVariate)
  validate(v, v.tune.support)
  validate(v, v.tune.support, v.tune.mass)
end

function validate(v::EitherDGSVariate, support::Matrix)
  n = length(v.value)
  size(support, 1) == n ||
    throw(ArgumentError("size(support, 1) differs from variate length $n"))
  v
end

validate(v::DiscreteVariate, support::Matrix, mass::Maybe{Vector{Float64}}) =
  isnull(mass) ? v : validate(v, support, get(mass))

function validate(v::EitherDGSVariate, support::Matrix, mass::Vector{Float64})
  n = length(mass)
  size(support, 2) == n ||
    throw(ArgumentError("size(support, 2) differs from mass length $n"))
  v
end


#################### Sampler Constructor ####################
"""
    DGS(params::ElementOrVector{Symbol},
              overSupport=Maybe{Vector{Int}}(), overIndices=Maybe{Vector{Int}}()
              getTargetIndex::Maybe{Function})

# Arguments
- `params`: the name(s) of the node(s) to resample
- `overSupport`: the support to use, if not full support.
    For instance, when splitting a group in reversible jump, this would be the indices of the two new groups.
- `overIndices`: which indices of the node(s) to resample, if not all
- `getTargetIndex`: when resampling, which target values to calculate the logpdf for, if not all.
"""
function DGS(params::ElementOrVector{Symbol},
          overSupport::Maybe{Vector{Int}}=nothing, overIndices::Maybe{Vector{Int}}=nothing,
          getTargetIndex::Maybe{Function}=nothing;
          returnLogp = false)

  params = asvec(params)
  samplerfx = function(model::AbstractModel, block::Integer, proposal::Maybe{(DictVariateVals{SVT} where SVT)}=nothing; kargs...)
    #A function that goes in the Sampler object; when called by sample! (line #?),
    #it loops over blocks, and for each block, calls DGS_sub!, which loops over indices
    s = model.samplers[block]
    local node, x
    logptot = 0
    for key in params
      node = model[key]
      if isnull(proposal)
        x = unlist(node)
      else
        x = proposal[key]
      end

      sim = function(i::Integer, d::DGSUnivariateDistribution, mass::Function)
        v = DGSVariate([x[i]], !isnull(overSupport) ? overSupport : support(d)')
        logp = sample!(v, mass; kargs...)
        x[i] = v[1]
        relist!(model, x, key)
        logp
      end

      mass = function(d::DGSUnivariateDistribution, v::AbstractVector,
                      i::Integer)
        x[i] = value = v[1]
        relist!(model, x, key)
        targetIndex = isnull(getTargetIndex) ? () : (get(getTargetIndex)(i),)
        exp(logpdf(d, value) + logpdf(model, node.targets, targetIndex...))
      end

      indices = isnull(overIndices) ? () : (get(overIndices),)
      logptot += DGS_sub!(node.distr, sim, mass, indices...)
    end
    if returnLogp
      return logptot
    end
    nothing
  end
  Sampler(params, samplerfx, DSTune{Function}())
end


function DGS_sub!(d::UnivariateDistribution, sim::Function, mass::Function)
  sim(1, d, v -> mass(d, v, 1))
end

function DGS_sub!(D::Array{UnivariateDistribution}, sim::Function,
                  mass::Function)
  DGS_sub!(D,sim,mass,1:length(D))
end

function DGS_sub!(D::Array{UnivariateDistribution}, sim::Function,
                  mass::Function, indices)
  logp = 0.
  for i in indices
    d = D[i]
    logp += sim(i, d, v -> mass(d, v, i))
  end
  logp
end

function DGS_sub!(d, sim::Function, mass::Function)
  throw(ArgumentError("unsupported distribution structure $(typeof(d))"))
end


#################### Sampling Functions ####################

sample!{F<:DSForm}(v::SamplerVariate{DSTune{F}}) = sample!(v, v.tune.mass)

"""
Sample just one value; store it in v; return log probability that it would be so sampled.
"""
function sample!(v::DGSVariate, mass::Function; donothing::Bool=false)
  tune = v.tune
  n = size(tune.support, 2)
  probs = Vector{Float64}(n)
  psum = 0.0
  for i in 1:n
    value = mass(tune.support[:, i])
    probs[i] = value
    psum += value
  end
  if psum > 0
    probs /= psum
  else
    probs[:] = 1 / n
  end
  if donothing
    r = index(tune.support, v[1])
  else
    r = rand(Categorical(probs)) #TODO: wouldn't "sample" be faster here?
  end
  v[:] = tune.support[:, r]
  log(probs[r])
end


"""
Sample just one value; store it in v; return log probability that it would be so sampled.

Input is a probability vector.
"""
function sample!(v::DiscreteVariate, mass::AbstractVector{Float64})
  validate(v, v.tune.support, mass)
  if !isprobvec(mass)
    mass /= sum(mass)
  end
  d = Categorical(mass)
  r = rand(d)
  v[:] = v.tune.support[:, r]
  logpdf(d,r)
end
