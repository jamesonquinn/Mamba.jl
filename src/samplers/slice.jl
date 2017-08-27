#################### Slice Sampler ####################

#################### Types and Constructors ####################

const SliceForm = Union{Univariate, Multivariate}

type SliceTune{F<:SliceForm} <: SamplerTune
  logf::Nullable{Function}
  width::Union{Float64, Vector{Float64}}

  SliceTune{F}() where F<:SliceForm = new()

  SliceTune{F}(x::AbstractVariateVals, width) where F<:SliceForm =
    SliceTune{F}(x, width, Nullable{Function}())

  SliceTune{F}(x::AbstractVariateVals, width, logf::Function) where F<:SliceForm =
    SliceTune{F}(x, width, Nullable{Function}(logf))

  SliceTune{F}(x::AbstractVariateVals, width::Real, logf::Nullable{Function}) where
    F<:SliceForm = new(logf, Float64(width))

  SliceTune{F}(x::AbstractVariateVals, width::Vector, logf::Nullable{Function}) where
    F<:SliceForm = new(logf, convert(Vector{Float64}, width))
end


const SliceUnivariate = SamplerVariate{VS,SliceTune{Univariate}} where VS
const SliceMultivariate = SamplerVariate{VS,SliceTune{Multivariate}} where VS

validate{F<:SliceForm}(v::SamplerVariate{SliceTune{F}}) =
  validate(v, v.tune.width)

validate{F<:SliceForm}(v::SamplerVariate{SliceTune{F}}, width::Float64) = v

function validate{F<:SliceForm}(v::SamplerVariate{SliceTune{F}}, width::Vector)
  n = length(v.value)
  length(width) == n ||
    throw(ArgumentError("length(width) differs from variate length $n"))
  v
end


#################### Sampler Constructor ####################

function Slice{T<:Real, F<:SliceForm}(params::ElementOrVector{Symbol},
                                      width::ElementOrVector{T},
                                      ::Type{F}=Multivariate;
                                      transform::Bool=false)
  tunetype = SliceTune{F}
  samplerfx = function(model::M, block::Integer) where M <: AbstractModel
    block = SamplingBlock{M}(model, block, transform)
    v = SamplerVariate{vstype(model),tunetype}(block, width)
    sample!(v, x -> logpdf!(block, x))
    relist(block, v)
  end
  Sampler(params, samplerfx, tunetype())
end


#################### Sampling Functions ####################

sample!(v::Union{SliceUnivariate, SliceMultivariate}) = sample!(v, v.tune.logf)


function sample!(v::SliceUnivariate, logf::Function)
  logf0 = logf(v.value)

  for k in keys(v.value)
    x = v.value[k...]
    lower = x - v.tune.width * rand()
    upper = lower + v.tune.width

    p0 = logf0 + log(rand())

    v.value[k...] = rand(Uniform(lower, upper))
    while true
      logf0 = logf(v.value)
      logf0 < p0 || break
      value = v.value[k...]
      if value < x
        lower = value
      else
        upper = value
      end
      v.value[k...] = rand(Uniform(lower, upper))
    end
  end

  v
end


function sample!(v::SliceMultivariate, logf::Function)
  p0 = logf(asvec(v.value)) + log(rand())

  n = length(v.value)
  r = rand(n)
  lower = asvec(v.value) .- (v.tune.width .* r)
  upper = lower .+ v.tune.width

  x = (v.tune.width .* rand(n)) .+ lower
  while logf(x) < p0
    println("qqqq slice x $(x) l $(lower) u $(upper) av $(asvec(v.value))")

    for i in 1:n

      value = x[i]
      if value < v.value[i]
        lower[i] = value
      else
        upper[i] = value
      end
      println("qqqq2 slice x $(x[i]) l $(lower[i]) u $(upper[i])")
      if lower[i] < upper[i]
        x[i] = rand(Uniform(lower[i], upper[i]))
      else
        println("Bad Uniform distribution in slice sampler... what happened?")
        println("$(v.tune.width) $(r) $(v.tune.width .* r) $(asvec(v.value) .- v.tune.width .* r)")
        println("$(value) $(i) $(x) $(lower) $(upper)  $(asvec(v.value))")
        throw(ArgumentError("Bad Uniform distribution in slice sampler."))
      end

    end
  end
  v.value[:] = x

  v
end
