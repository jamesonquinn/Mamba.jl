#################### Discrete Gibbs Sampler ####################

#################### Types and Constructors ####################

const DGSUnivariateDistribution =
          Union{Bernoulli, Binomial, Categorical, DiscreteUniform,
                Hypergeometric, NoncentralHypergeometric,
                DirichletPInt}


const DSForm = Union{Function, Vector{Float64}}

type DSTune{F<:DSForm} <: SamplerTune
  mass::Nullable{F}
  support::Matrix{Int64} #qqqq: is this typing too strict? Do we have to make it parametric?

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
  isnull(mass) ? v : validate(v, support, mass) #TODO: check, should this Maybe be Nullable (with a get(mass))?

function validate(v::EitherDGSVariate, support::Matrix, mass::Vector{Float64})
  n = length(mass)
  size(support, 2) == n ||
    throw(ArgumentError("size(support, 2) differs from mass length $n"))
  v
end






################### Sampler Constructor ####################
"""
    DGS(params::ElementOrVector{Symbol})

    if returnLogp is true, returns a function whose arguments include:
              overSupport=Maybe{Vector{Int}}(), overIndices=Maybe{Vector{Int}}()
              getTargetIndex::Maybe{Function})

# Arguments
- `params`: the name(s) of the node(s) to resample
- `overSupport`: the support to use, if not full support.
    For instance, when splitting a group in reversible jump, this would be the indices of the two new groups.
- `overIndices`: which indices of the node(s) to resample, if not all
- `getTargetIndex`: when resampling, which target values to calculate the logpdf for, if not all.
"""
function DGS(params::ElementOrVector{Symbol}, returnLogp = false)
  params = asvec(params)
  println("creating DGS for $(params)")
  samplerfx = function(model::AbstractModel, block::Integer,
    proposal::Maybe{(DictVariateVals{SVT} where SVT)}=nothing,
              overSupport::Maybe{Vector{Int}}=nothing, supportWeights::Maybe{AbstractVector{Float64}}=nothing,
              overIndices::Maybe{Vector{Int}}=nothing,
              getTargetIndex::Maybe{Function}=nothing; kargs...)
    #A function that goes in the Sampler object; when called by sample! (line #?),
    #it loops over blocks, and for each block, calls DGS_sub!, which loops over indices
    local node, x, xAsInt #TODO: more local vars

    if !isnull(supportWeights)
      w = fill(0.,length(supportWeights))
      tot = sum([supportWeights[s] for s in overSupport])
      #tot = sum([get(()->0.,supportWeights,s) for s in overSupport])
      for s in overSupport
        w[s] = supportWeights[s] / tot
      end
    end

    logptot = 0
    println("using DGS at block $(block) for $(params)")
    for key in params
      node = model[key]
      if isnull(proposal)
        x = unlist(node)
      else
        x = proposal[key]
      end
      xAsInt = [Int(val) for val in x]
      sup = emptysup = range(1,0)'

      sim = function(i::Integer, d::DGSUnivariateDistribution, mass::Function)
        v = DGSVariate([x[i]], !isnull(overSupport) ? overSupport' : (
                                    !isa(d,DirichletPInt) ? support(d)' : (
                                        sup==emptysup ? sup=(1:Int(max(x.value...)))' : sup
                                    )))
        logp = sample!(v, mass; kargs...)
        x[i] = v[1]
        xAsInt[i] = Int(v[1])
        relist!(model, x, key)
        logp
      end

      mass = function(d::DGSUnivariateDistribution, v::AbstractVector,
                      i::Integer)

        value = v[1]

        x[i] = myvaltype(x)(value)
        xAsInt[i] = Int(value)
        relist!(model, x, key)
        targetIndex = isnull(getTargetIndex) ? () : (getTargetIndex(i),) #optional parameter idiom, yuck
        if isa(d,DirichletPInt)
          if isnull(supportWeights)
            counts = countmap(x.value)
            groups = collect(keys(counts))
            ln = length(groups)
            mx = Int64(max(groups...))
            if ln < mx
              sort!(groups)
              for j in 1:ln
                if j < groups[j]
                  break
                end
              end
              maybenewi = [j]
            else
              maybenewi = []
            end
            d = DirichletPIntMarginal(d,xAsInt,i,maybenewi...)
          else
            d = Categorical(w)
          end
        end

        if isnull(targetIndex)
          return exp(logpdf(d, value) + logpdf(model, node.targets))
        else
          return exp(logpdf(d, value) + logpdf(model, node.targets; index=targetIndex))
        end
      end

      indices = isnull(overIndices) ? () : (overIndices,) #this is how you make arguments optional when passing them on... ugly idiom
      logptot += DGS_sub!(node.distr, sim, mass, indices...)
    end
    logptot
  end
  if returnLogp
    return samplerfx #used directly as a substep for reversible jump
  else
    function sfx(args...)
      samplerfx(args...)
      nothing
    end
    return Sampler(params, sfx, DSTune{Function}())
  end
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

function DGS_sub!(D::DirichletPInt, sim::Function,
                  mass::Function, indices::Union{Vector{Int}, Void}=nothing)
  logp = 0.
  if indices==nothing
    indices = 1:D.len
  end
  for i in indices
    logp += sim(i, D, v -> mass(D, v, i))
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
