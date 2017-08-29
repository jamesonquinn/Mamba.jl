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


#################### Sampler Constructor ####################

function proposeSplitParams!(conditionalDist::Distribution,
        k::Int, #index of existing group
        j::Int, #index of new proposed group
        dirichparam::Float64,
        params::DictVariateVals{SVT}, condParamIndices::Vector{Tup},
        weights::AbstractVector{Float64},
        hyperParamDists::Maybe{Vector{UnivariateDistribution}}=nothing,
        tune::Maybe{RJTune}=nothing) where SVT where Tup<:Tuple

  throw(ArgumentError("unsupported distribution structure $(typeof(conditionalDist))"))
end

function proposeSplitParams!(conditionalDist::Normal,
        k::Int, #index of existing group
        j::Int, #index of new proposed group
        dirichparam::Float64,
        params::DictVariateVals{SVT}, condParamIndices::Vector{Tup},
        weights::AbstractVector{Float64},
        hyperParamDists::Maybe{Vector{UnivariateDistribution}}=nothing,
        tune::Maybe{RJTune}=nothing,
        u2dist=Beta(2,2), u3dist=Beta(1,1)) where SVT where Tup<:Tuple

  #=If you were going to follow Richardson and Green 1997 fully, you'd
  keep track of the restriction that the means are ordered, and only adjacent
  means are merged; so splits that don't preserve those orderings would be rejected.
  This could be done using params and paramIndices. But we're not gonna do that — too
  much programming.=#

  u1 = rand(Beta(dirichparam,dirichparam)) #R&G used 2,2 but this choice simplifies A
  u2 = rand(u2dist)
  u3 = rand(u3dist)
  w0 = weights[k]
  mu0 = params[condParamIndices[1]...,k]
  sig0 = params[condParamIndices[2]...,k]
  weights[k] = w1 = w0 * u1
  weights[j] = w2 = w0 - w1
  sqrtu1odds = sqrt(u1/(1.-u1))
  mu1 = mu0 - u2*sig0/sqrtu1odds
  mu2 = mu0 + u2*sig0*sqrtu1odds
  sig1 = sqrt(u3*(1.-u2^2)*sig0/u1)
  sig2 = sqrt((1.-u3)*(1.-u2^2)*sig0/(1.-u1))
  params[condParamIndices[1]...,k] = mu1
  params[condParamIndices[2]...,k] = sig1
  params[condParamIndices[1]...,j] = mu2
  params[condParamIndices[2]...,j] = sig2
  logAfac= (log(w0*abs(mu1-mu2)*sig1*sig2/(u2*(1-u2^2)*u3*(1-u3)*sig0)) #jacobian
    -logpdf(u2dist,u2)-logpdf(u3dist,u3)) #density; because of our choice of dist, u1 cancels out
    #the factors due to prior probability of the parameters are part of the posterior ratio.
        #+logpdf(hyperParamDists[1],mu1)+logpdf(hyperParamDists[1],mu2)-logpdf(hyperParamDists[1],mu0)
        #+logpdf(hyperParamDists[2],sig1)+logpdf(hyperParamDists[2],sig2)-logpdf(hyperParamDists[2],sig0))
  #A = Afac * (posterior ratio) / P(allocation)
  logAfac
end

function proposeMergeParams!(conditionalDist::Distribution,
        k::Int, #index of target group
        j::Int, #index of disappearing group
        dirichparam::Float64,
        params::DictVariateVals{SVT}, condParamIndices::Vector{Tup},
        weights::AbstractVector{Float64},
        hyperParamDists::Maybe{Vector{UnivariateDistribution}}=nothing,
        tune::Maybe{RJTune}=nothing) where SVT where Tup<:Tuple

  throw(ArgumentError("unsupported distribution structure $(typeof(conditionalDist))"))
end

function proposeMergeParams!(conditionalDist::Normal,
        k::Int, #index of target group
        j::Int, #index of disappearing group
        dirichparam::Float64,
        params::DictVariateVals{SVT}, condParamIndices::Vector{Tup},
        weights::AbstractVector{Float64},
        hyperParamDists::Maybe{Vector{UnivariateDistribution}}=nothing,
        tune::Maybe{RJTune}=nothing,
        u2dist=Beta(2,2), u3dist=Beta(1,1)) where SVT where Tup<:Tuple

  w1 = weights[j]
  mu1 = params[condParamIndices[1]...,j]
  sig1 = params[condParamIndices[2]...,j]
  w2 = weights[k]
  mu2 = params[condParamIndices[1]...,k]
  sig2 = params[condParamIndices[2]...,k]

  weights[k] = w0 = w1+w2
  weights[j] = 0

  mu0 = (mu1*w1 + mu2*w2)/w0
  params[condParamIndices[1]...,k] = mu0


  sig0 = sqrt((w1*(sig1^2 + (mu1-mu0)^2) + w2*(sig2^2 + (mu2-mu0)^2))/w0)
  params[condParamIndices[2]...,k] = sig0

  u1 = w1/w0 #should always be positive if w1 and w2 both are.
  evOverEvve = (sig1^2*w1 + sig2^2*w2)/(sig0^2*w0)
  u2 = sqrt(1-evOverEvve) #u2 governs how much of the variance is external to both components combined — the V(E) part of E(V)+V(E)
  u3 = ((sig1/sig0)^2)*u1/evOverEvve #u3 governs how the internal variance is broken down between the two
  ((u3 < 1) && (u3 > 0)) || throw(ErrorException("u3 not in (0,1)"))
  logAfac= (log(w0*abs(mu1-mu2)*sig1*sig2/(u2*(1-u2^2)*u3*(1-u3)*sig0)) #jacobian
    -logpdf(u2dist,u2)-logpdf(u3dist,u3) #density; because of our choice of dist for u1, it cancels out
    )
    #the factors due to prior probability of the parameters are part of the posterior ratio.
        #+logpdf(hyperParamDists[1],mu1)+logpdf(hyperParamDists[1],mu2)-logpdf(hyperParamDists[1],mu0)
        #+logpdf(hyperParamDists[2],sig1)+logpdf(hyperParamDists[2],sig2)-logpdf(hyperParamDists[2],sig0))
  #P(accept) = min(1,1/Afac * (posterior ratio) * P(allocation)
  logAfac #remember, use reciprocal of Afac for merges
end

# function hyperdists(model::AbstractModel,target::Symbol)
#   tnode = model[target]
#   [s.distr for s in tnode.sources[1:end-2]]
# end

function RJS(dpparam::Symbol, themodel::AbstractModel, groupNumParam::Symbol, splitprob=0.5) #TODO: allow use on hierarchical stuff; dpparam::Tuple??

  #figure out which target is the "bottom";
  #e.g., the Normally-distributed one, not the mean or sd thereof.

  thenode = themodel[dpparam]
  targets=Set{Symbol}(thenode.targets)
  allSourceParamsOfTargets=Set{Symbol}()
  for target in targets
    tnode = themodel[target]
    union!(allSourceParamsOfTargets,Set{Symbol}(tnode.sources))
  end
  setdiff!(targets,allSourceParamsOfTargets)
  sourceParamsOfTargets=setdiff(Set{Symbol}(thenode.targets),targets)
  allRelevantParams=collect(chain([dpparam],targets,sourceParamsOfTargets))

  assigner! = DGS(dpparam, true) #must be declared outside function below or you get world problems
  checkAssign = (args...)->assigner!(args...; donothing = true)

  samplerfx = function(model::AbstractModel, block::Integer)

    #local node, x #TODO: does anything need to be local here?
    node = model[dpparam]
    (alpha,) = params(node.distr)
    cur = unlist(model)
    proposal = copy(cur)
    logA = log((1. - splitprob) / splitprob) #base acceptance for split; use reciprocal for merge
    #logA -= logpdf(model,block) #double-counting

    #hyperParamDists
    #hyperParamDistses = [hyperdists(model,target) for target in node.targets]

    #counts of groups
    counts = countmap(cur[dpparam].value)
    groups = [Int64(g) for g in keys(counts)]
    ln = length(groups)
    mx = Int64(max(groups...))

    #latent weights of groups.
    #This is just a Gibbs sample from the distribution of the latent weights, which
    # depends only on the counts and the dirichlet process parameter.
    newweights = fill(0.,mx+1) #leave room to grow
    sumweights = 0.
    for i in 1:ln
      sumweights += newweights[groups[i]] = rand(Gamma(counts[groups[i]]))
    end
    oldweights = ProbabilityWeights(newweights/sumweights)

    splitIndicator = rand()<splitprob

    if splitIndicator

      #choose group to split
      k = sample(oldweights)

      #where to put new group
      if ln == mx #no gaps
        j = mx + 1 #index of new proposed group, if needed
        for aparam in sourceParamsOfTargets #all params
          proposal[aparam,j] = NaN #make room
          proposal[groupNumParam,1] = Float64(j)
          cur[groupNumParam,1] = Float64(j)
          #relist!(model,cur) #needed so that the distr below works #TODO: fix this; for now, just short-circuiting
          model[groupNumParam].value = j #update!(model[groupNumParam],model)
          aparmnode = update!(model[aparam],model)
          cur[aparam,j] = rand(aparmnode.distr[j]) #thus, there must always be 1 extra copy of the distribution
          #cur[aparam,j] = rand(model[aparam].distr[1]) #TODO: fix; "1" is evil hack, though valid for iid.
        end
      else
        sort!(groups)
        for j in 1:ln
          if j < groups[j]
            break
          end
        end
      end

      #propose individual splits — parameters for each group
      for target in targets
        tnode = model[target]
        musigfrom = tnode.sources[1:end-2]
        logA += proposeSplitParams!(tnode.distr[k],
                k, #index of existing group
                j, #index of new proposed group
                params(node.distr)[1],
                proposal, [(aparam,) for aparam in musigfrom],
                newweights
                )#TODO: tune
      end

      itemsToAssign = [i for i in 1:length(cur[dpparam].value) if cur[dpparam,i]==k]

      relist!(model, cur) #parameter space may have expanded
      update!(model, allRelevantParams)
      logA -= checkAssign(model,block,cur,
          [k], oldweights,
          itemsToAssign,
          i -> i)
      #account for priors/hyperparameters on params
      for aparam in sourceParamsOfTargets
        logA -= logpdf(model, [aparam])
      end

      relist!(model, proposal)
      update!(model, allRelevantParams)
      #assign elems to newly-specified groups and adjust acceptance prob
      logA += assigner!(model,block,proposal,
          [k,j], newweights,
          itemsToAssign,
          i -> i)

      #account for priors/hyperparameters on params
      for aparam in sourceParamsOfTargets
        logA += logpdf(model, [aparam])
      end

    else
      #merge

      #choose group to merge to (by weight)
      k = sample(oldweights)

      #choose group to merge from (uniform random)
      mergeables = setdiff(Set(groups),[k])
      if length(mergeables) > 0
        j = rand(mergeables)

        #propose individual merged params

        for target in targets
          tnode = model[target]
          musigfrom = tnode.sources[1:end-2]
          logA += proposeMergeParams!(tnode.distr[k],
                  k, #index of remaining group
                  j, #index of disappearing group
                  params(node.distr)[1],
                  proposal, [(aparam,) for aparam in musigfrom],
                  newweights) #TODO: tune
        end

        #move elems to newly-specified group and adjust acceptance prob
        itemsToAssign = [i for i in 1:length(cur[dpparam].value) if cur[dpparam,i] in [j,k]]
        for i in itemsToAssign
          proposal[dpparam,i] = Float64(k)
        end

        relist!(model, cur) #redundant??
        update!(model, allRelevantParams)
        logA += checkAssign(model,block,cur,
            [k,j], oldweights,
            itemsToAssign,
            i -> i)
        for aparam in sourceParamsOfTargets
          #account for priors/hyperparameters on params
          logA -= logpdf(model, [aparam])
        end


        relist!(model, proposal)
        update!(model, allRelevantParams)
        logA -= checkAssign(model,block,proposal,
              [k], newweights,
              itemsToAssign,
              i -> i)
        for aparam in sourceParamsOfTargets
          #account for priors/hyperparameters on params
          logA += logpdf(model, [aparam])
        end

        #relist!(model, proposal)
        #logA += logpdf(model,block) #double-counting, after above

        logA = -logA #change from split acceptance prob to merge acceptance prob
      end
    end
    acceptIndicator = rand() < exp(logA)
    if !acceptIndicator
      relist!(model, cur)
    end

    if true #TODO: replace with debug conditional
      if length(counts) > 1
        println("qqqq2 $(acceptIndicator ? :YES : :NOT) $(splitIndicator ? :splitting : :merging) $(k) $(j) $([(v,c) for (v,c) in counts])")
      else
        println("not merging; 1 group")
      end
    end
    nothing
  end
  Sampler([dpparam], samplerfx, DSTune{Function}())
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
