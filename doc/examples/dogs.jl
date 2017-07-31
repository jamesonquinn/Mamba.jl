using Mamba
using Optim


## Data
dogs = Dict{Symbol, Any}(
  :Y =>
    [0 0 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1
     0 0 0 0 0 1 1 0 1 1 0 0 1 1 0 1 0 1 1 1 1 1 1 1 1
     0 1 1 0 0 1 1 1 1 0 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1
     0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     0 0 0 0 0 0 1 1 1 1 0 0 1 0 1 1 1 1 1 1 1 1 1 1 1
     0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1
     0 0 0 0 0 0 0 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     0 0 0 0 0 1 0 1 0 1 1 0 1 0 0 0 1 1 1 1 1 0 1 1 0
     0 0 0 0 1 0 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1
     0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     0 0 0 0 0 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1
     0 0 0 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     0 0 0 0 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     0 0 0 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     0 1 0 1 0 0 0 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1
     0 0 0 0 1 0 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1
     0 1 0 0 0 0 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     0 0 0 0 1 1 0 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1
     0 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     0 0 1 0 1 0 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1
     0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     0 0 0 0 0 0 0 0 1 1 1 0 1 0 0 0 1 1 0 1 1 1 1 1 1
     0 0 0 0 0 0 1 1 0 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1
     0 0 1 0 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     0 0 0 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     0 0 0 1 0 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     0 0 0 0 1 1 0 0 1 1 1 0 1 0 1 0 1 0 1 1 1 1 1 1 1
     0 0 0 0 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1]
)
dogs[:Dogs] = size(dogs[:Y], 1)
dogs[:Trials] = size(dogs[:Y], 2)

dogs[:xa] = mapslices(cumsum, dogs[:Y], 2)
dogs[:xs] = mapslices(x -> collect(1:25) - x, dogs[:xa], 2)
dogs[:y] = 1 - dogs[:Y][:, 2:25]


## Model Specification
model = Model(

  y = Stochastic(2,
    (Dogs, Trials, alpha, xa, beta, xs) ->
      UnivariateDistribution[
        begin
          p = exp(alpha * xa[i, j] + beta * xs[i, j])
          Bernoulli(p)
        end
        for i in 1:Dogs, j in 1:Trials-1
      ],
    false
  ),

  alpha = Stochastic(
    () -> Truncated(Flat(), -Inf, -1e-5)
  ),

  A = Logical(
    alpha -> exp(alpha)
  ),

  beta = Stochastic(
    () -> Truncated(Flat(), -Inf, -1e-5)
  ),

  B = Logical(
    beta -> exp(beta)
  )

)



## Initial Values
inits = [
  Dict(:y => dogs[:y], :alpha => -1, :beta => -1),
  Dict(:y => dogs[:y], :alpha => -2, :beta => -2)
]


## Sampling Scheme
scheme = [Slice([:alpha, :beta], 1.0)]
setsamplers!(model, scheme)

" A factory which makes a function which takes values from a vector and
    puts them into a structure of the same shape as fillable"
function makeFiller(fillable, whichParts)
  partFillers = Function[]
  i = 1
  for part in whichParts
    function enclose(myI, myPart)
      function partFill!(target,v)
        target[myPart] = v[myI]
      end
      return(partFill!)
    end
    push!(partFillers,enclose(i,part))
    i += 1
  end
  function fill!(target, v)
    for partFiller in partFillers
      partFiller(target,v)
    end
  end
  return fill!
end

f! = makeFiller(inits[1],[:alpha, :beta])
inits[1]
f!(inits[1],[1,2])

"Takes a model, an init value, and a list of parameters to optimize over, and optimizes

    notes:
      m should already have inputs"
function optimOver(m::Model, init, params::paramNames::Vector{Symbol})
  current = copy(init)
  fill! = makeFiller(init, params)
  setinits!(m, current)
  function logpdfFor(v)
  ## This currently has side-effects. That's bad style but I'll fix it later
    fill!(current,v)
    return logpdf(m)
  end
end
## MCMC Simulations
mcmc_master!(mm, 1:iters, burnin, thin, 1:chains, verbose)
sim = mcmc(model, dogs, inits, 10000, burnin=2500, thin=2, chains=2)
describe(sim)
