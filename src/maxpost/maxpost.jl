using Distributions
using Optim
using ForwardDiff
using Mamba

import ForwardDiff:
        Dual

import Distributions:
        logpdf

import Mamba:
        setinits!

" A factory which makes a function which takes values from a vector and
    puts them into a structure of the same shape as fillable"
function makeFiller(fillable, whichParts)
  partFillers = Function[]
  i = 1
  for part in whichParts
    function enclose(myI, myPart)
      function partFill!{T<:Number}(target,v::Vector{T})
        target[myPart] = v[myI]
      end
      return(partFill!)
    end
    push!(partFillers,enclose(i,part))
    i += 1
  end
  function fill!{T<:Number}(target, v::Vector{T})
    for partFiller in partFillers
      partFiller(target,v)
    end
  end
  return fill!
end


"Takes a model, an init value, and a list of parameters to optimize over, and optimizes

    notes:
      m should already have inputs"
function optimOver(m::Model, init, params::Vector{Symbol})
  current = copy(init)
  fill! = makeFiller(init, params)
  setinits!(m, current)
  function logpdfFor{T<:Number}(v::Vector{T})
  ## This currently has side-effects. That's bad style but I'll fix it later
    fill!(current,v)
    return -logpdf(m)
  end
  optimize(logpdfFor, Float64[init[param] for param in params], method=BFGS(), autodiff=true)
end

immutable DualNormal <: ContinuousUnivariateDistribution
    μ::Dual
    σ::Dual

    #Normal(μ::Dual, σ::Dual) = (@_args(Normal, σ > zero(σ)); new(μ, σ))
    Normal(μ::Dual, σ::Float64) = Normal(μ, Dual(σ,0.0))
    Normal(μ::Float64, σ::Dual) = Normal(Dual(μ,0.0), σ)
end



function setinits!(s::ScalarStochastic, m::Model, x::Dual)
  s.distr = s.eval(m)
  setmonitor!(s, s.monitor)
end


type GossipyNormal <: ContinuousUnivariateDistribution
  Z::Normal
end

Normalish = Union{Normal, DualNormal, GossipyNormal}

DNum = Union{Dual, Float64}
function logpdf(d::Normalish, v::Dual) #qqqq still needed??
  -log(2)-log(π)-log(d.σ)-(v-d.μ)^2/2/d.σ
end

function logpdf(d::GossipyNormal, v::Float64)
  logpdf(d.Z,v)
end

minimodel = Model(
  y = Stochastic(1,
    (mu, s2) ->  Normal(mu, sqrt(s2)),
    false
  ),

  mu = Stochastic(
    () -> Normal(0, sqrt(1000))
  ),

  s2 = Stochastic(
    () -> Normal(2, 0.001)
  )

)

minidata = Dict{Symbol, Any}(
              :y => [1, 3, 3, 3, 5]
            )
miniinit = Dict{Symbol, Any}(
              :y => minidata[:y],
              :mu => 2.5,
              :s2 => 1
            )
setsamplers!(minimodel, [NUTS([:mu, :s2])])
setinputs!(minimodel, minidata)
setinits!(minimodel, miniinit)
optimOver(minimodel, miniinit, [:mu,:s2])
logpdf(minimodel)
minimodel
z = Normal()
logpdf(z,[1,2,3])





model = Model(

  y = Stochastic(1,
    (mu, s2) ->  MvNormal(mu, sqrt(s2)),
    false
  ),

  mu = Logical(1,
    (xmat, beta) -> xmat * beta,
    false
  ),

  beta = Stochastic(1,
    () -> MvNormal(2, sqrt(1000))
  ),

  s2 = Stochastic(
    () -> InverseGamma(0.001, 0.001)
  )

)
line = Dict{Symbol, Any}(
  :x => [1, 2, 3, 4, 5],
  :y => [1, 3, 3, 3, 5]
)
line[:xmat] = [ones(5) line[:x]]
## Initial Values
## Initial Values
inits = [
  Dict{Symbol, Any}(
    :y => line[:y],
    :beta => rand(Normal(0, 1), 2),
    :s2 => rand(Gamma(1, 1))
  )
  for i in 1:3
]



setinputs!(model, line)
setinits!(model, inits)
logpdf(model)

function f(v::Array{Number,1})
  v
end
dualvec = Array{}(Dual{Real},5)
typeof(dualvec) <: Array{Dual
f(Array(Dual{Real},9))
