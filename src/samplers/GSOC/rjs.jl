#################### Reversible Jump Sampler ####################

# Richardson and Green, 1997 https://people.maths.bris.ac.uk/~mapjg/papers/RichardsonGreenRSSB.pdf
#################### Types and Constructors ####################

const RJSDistribution = DirichletPInt

const RJForm = Union{Function, Vector{Float64}}

type RJTune{F<:DSForm} <: SamplerTune
  mass::Nullable{F}
  ncomps::Int

  RJTune{F}() where F<:DSForm = new()

  RJTune{F}(x::Vector, ncomps::Int) where F<:DSForm =
    new(Nullable{Vector{Float64}}(), support)
end


const RJSVariate = DictSamplerVariate{DSTune{Vector{Float64}}}

validate(v::RJSVariate) = validate(v, v.tune.support)

#
# function validate{F<:DSForm}(v::SamplerVariate{DSTune{F}}, support::Matrix)
#   n = length(v)
#   size(support, 1) == n ||
#     throw(ArgumentError("size(support, 1) differs from variate length $n"))
#   v
# end
#
# validate(v::DiscreteVariate, support::Matrix, mass::Nullable{Vector{Float64}}) =
#   isnull(mass) ? v : validate(v, support, get(mass))
#
# function validate(v::DiscreteVariate, support::Matrix, mass::Vector{Float64})
#   n = length(mass)
#   size(support, 2) == n ||
#     throw(ArgumentError("size(support, 2) differs from mass length $n"))
#   v
# end


#################### Sampler Constructor ####################

function proposeSplitParams!(conditionalDist::Distribution,
        k::Int, #index of existing group
        j::Int, #index of new proposed group
        dirichparam::Float64,
        params::DictVariateVals{SVT}, condParamIndices::Vector{Tuple},
        weights::AbstractVector{Float64},
        tune::Maybe{RJTune}=nothing,
        hyperCondParamDist::Maybe{UnivariateDistribution}=nothing,
        hyperparams::Maybe{Vector}=nothing) where SVT

  throw(ArgumentError("unsupported distribution structure $(typeof(conditionalDist))"))
end

function proposeSplitParams!(conditionalDist::Normal,
        k::Int, #index of existing group
        j::Int, #index of new proposed group
        dirichparam::Float64,
        params::DictVariateVals{SVT}, condParamIndices::Vector{Tuple},
        weights::AbstractVector{Float64},
        tune::Maybe{RJTune}=nothing,
        hyperCondParamDist::Maybe{UnivariateDistribution}=nothing,
        hyperparams::Maybe{Vector}=nothing,
        u2dist=Beta(2,2), u3dist=Beta(1,1)) where SVT

  #=If you were going to follow Richardson and Green 1997 fully, you'd
  keep track of the restriction that the means are ordered, and only adjacent
  means are merged; so splits that don't preserve those orderings would be rejected.
  This could be done using params and paramIndices. But we're not gonna do that — too
  much programming.=#

  u1 = rand(Beta(dirichparam,dirichparam)) #R&G used 2,2 but this choice simplifies A
  u2 = rand(u2dist)
  u3 = rand(u3dist)
  w0 = params[weightIndex...,k]
  mu0 = params[condParamIndices[1]...,k]
  sig0 = params[condParamIndices[2]...,k]
  w1 = w0 * u1
  w2 = w0 - w1
  sqrtu1odds = sqrt(u1/(1.-u1))
  mu1 = mu0 - u2*sig0/sqrtu1odds
  mu2 = mu0 + u2*sig0*sqrtu1odds
  sig1 = sqrt(u3*(1.-u2^2)*sig0/u1)
  sig2 = sqrt((1.-u3)*(1.-u2^2)*sig0/(1.-u1))
  params[weightIndex...,k] = w1
  params[condParamIndices[1]...,k] = mu1
  params[condParamIndices[2]...,k] = sig1
  params[weightIndex...,j] = w2
  params[condParamIndices[1]...,j] = mu2
  params[condParamIndices[2]...,j] = sig2
  logAfac= (log(w0*abs(mu1-mu2)*sig1*sig2/(u2*(1-u2^2)*u3*(1-u3)*sig0)) #jacobian
    -logpdf(u2dist,u2)-logpdf(u3dist,u3)) #density; because of our choice of dist, u1 cancels out
  #A = Afac * (posterior ratio) / P(allocation)
  logAfac
end

function proposeMergeParams!(conditionalDist::Distribution,
        k::Int, #index of target group
        j::Int, #index of disappearing group
        dirichparam::Float64,
        params::DictVariateVals{SVT}, condParamIndices::Vector{Tuple},
        weights::AbstractVector{Float64},
        tune::Maybe{RJTune}=nothing,
        hyperCondParamDist::Maybe{UnivariateDistribution}=nothing,
        hyperparams::Maybe{Vector}=nothing) where SVT

  throw(ArgumentError("unsupported distribution structure $(typeof(conditionalDist))"))
end

function proposeMergeParams!(conditionalDist::Normal,
        k::Int, #index of target group
        j::Int, #index of disappearing group
        dirichparam::Float64,
        params::DictVariateVals{SVT}, condParamIndices::Vector{Tuple},
        weights::AbstractVector{Float64},
        tune::Maybe{RJTune}=nothing,
        hyperCondParamDist::Maybe{UnivariateDistribution}=nothing,
        hyperparams::Maybe{Vector}=nothing) where SVT

  w1 = params[weightIndex...,j]
  mu1 = params[condParamIndices[1]...,j]
  sig1 = params[condParamIndices[2]...,j]
  w2 = params[weightIndex...,k]
  mu2 = params[condParamIndices[1]...,k]
  sig2 = params[condParamIndices[2]...,k]
  w0 = w1+w2
  params[weightIndex...,k] = w0
  mu0 = (mu1*w1 + mu2*w2)/w0
  params[condParamIndices[1]...,k] = mu0
  sig0 = sqrt(mu1*(sig1^2 + (mu1-mu0)^2) + mu2*(sig2^2 + (mu2-mu0)^2))
  params[condParamIndices[2]...,k] = sig0
  u1 = (mu0-mu1)/(mu2-mu1)
  u2 = (mu2-mu0)/sig0*sqrt(w2/w1)
  logAfac= (log(w0*abs(mu1-mu2)*sig1*sig2/(u2*(1-u2^2)*u3*(1-u3)*sig0)) #jacobian
    -logpdf(u2dist,u2)-logpdf(u3dist,u3)) #density; because of our choice of dist for u1, it cancels out
  #P(accept) = min(1,1/Afac * (posterior ratio) * P(allocation)
  logAfac #remember, use reciprocal of Afac for merges
end

function RJS(param::Symbol, splitprob=0.5) #TODO: allow use on hierarchical stuff; param::Tuple??

  samplerfx = function(model::Model, block::Integer)

    #local node, x #TODO: does anything need to be local here?
    node = model[param]
    cur = unlist(model)
    proposal = copy(cur)
    logA = log((1. - splitprob) / splitprob) #base acceptance for split; use reciprocal for merge
    logA -= logpdf(model,block)

    #counts of groups
    counts = countmap(cur[param])
    groups = keys(counts)
    ln = length(groups)
    mx = max(groups)

    #latent weights of groups.
    #This is just a Gibbs sample from the distribution of the latent weights, which
    # depends only on the counts and the dirichlet process parameter.
    rawweights = Vector{Float64}(ln)
    sum = 0.
    for i in 1:ln
      sum += rawweights[i] = rand(Gamma(counts[groups[i]] + alpha))
    end
    w = ProbabilityWeights([get(rawweights,i,0.)/sum for i in 1:mx])

    splitIndicator = rand()<splitprob
    if splitIndicator

      #choose group to split
      k = sample(w)

      #where to put new group
      if ln == mx #no gaps
        j = mx + 1 #index of new proposed group, if needed
      else
        sort!(groups)
        for j in 1:ln
          if j < groups[j]
            break
          end
        end
      end


      #propose individual splits

      for target in node.targets
        tnode = model[target]
        musigfrom = [s for s in tnode.sources if s!=param]
        logA += proposeSplitParams!(tnode.distr,
                k, #index of existing group
                j, #index of new proposed group
                params(node.distr)[1],
                proposal, musigfrom,
                w)#TODO: tune
        assigner = DGS(target,
              [k,j],
              [i for i in 1:length[cur[param]] if cur[param,i]==k],
              i -> i)
        logA += assigner.eval(model,block,proposal)

      end

    else
      #merge

      #choose group to merge to (by weight)
      k = sample(w)

      #choose group to merge from (uniform random)
      j = rand(groups)

      #propose individual splits

      for target in node.targets
        tnode = model[target]
        logA += proposeMergeParams!(tnode.distr,
                k, #index of existing group
                j, #index of new proposed group
                params(node.distr)[1],
                proposal, musigfrom,
                w) #TODO: tune

        assigner = DGS(target,
              [k,j],
              [i for i in 1:length[cur[param]] if cur[param,i] in [j,k]],
              i -> i;
              )
        logA += assigner.eval(model,block,proposal)
      end

      relist!(model, proposal)
      logA += logpdf(model,block)

      if !splitIndicator
        logA = -logA
      end

      acceptIndicator = rand() < exp(logA)
      if !acceptIndicator
        relist!(model, cur)
      end

      acceptIndicator
    end
    nothing
  end
  Sampler([param], samplerfx, DSTune{Function}())
end


#################### Sampling Functions ####################
#
# sample!{F<:DSForm}(v::SamplerVariate{DSTune{F}}) = sample!(v, v.tune.mass)
#
#
# function sample!(v::RJSVariate, mass::Function)
#   tune = v.tune
#   n = size(tune.support, 2)
#   probs = Vector{Float64}(n)
#   psum = 0.0
#   for i in 1:n
#     value = mass(tune.support[:, i])
#     probs[i] = value
#     psum += value
#   end
#   if psum > 0
#     probs /= psum
#   else
#     probs[:] = 1 / n
#   end
#   v[:] = tune.support[:, rand(Categorical(probs))]
#   v
# end
#
#
# function sample!(v::DiscreteVariate, mass::Vector{Float64})
#   validate(v, v.tune.support, mass)
#   v[:] = v.tune.support[:, rand(Categorical(mass))]
#   v
# end
