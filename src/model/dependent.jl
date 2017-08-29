#################### Dependent ####################

const depfxargs = [(:model, Mamba.AbstractModel)]

function myvaltype(x::Type{SVT}) where SVT<:ScalarVariateType
  myvaltype(SVT,ScalarVariateType)
end
function myvaltype(x::Type{SVT},::Type{ScalarVariateType}) where SVT
  SVT
end
function myvaltype(x::VectorVariateVals{SVT}) where SVT
  SVT
end
function myvaltype(x::DictVariateVals{SVT}) where SVT
  SVT
end
function myvaltype(x::Variate{VS}) where VS
  myvaltype(VS)
end
function myvaltype(x::Type{X}) where X<:Variate
  myvaltype(V,Variate)
end
function myvaltype(x::Type{Variate{SVT}},::Type{Variate}) where SVT
  SVT
end
function myvaltype(x::AbstractModel{V}) where V
  myvaltype(V)
end

function vstype(x::Y) where Y<:AbstractModel{V} where V
  V
end
function vstype(x::Type{Y}) where Y<:AbstractModel{V} where V
  V
end



function myvaltype(x::Type{X}) where X<:ArrayVariateVals{SVT,N} where N where SVT <: ScalarVariateType
  SVT
end
function myvaltype(x::Type{X}) where X<:DictVariateVals{SVT} where SVT <: ScalarVariateType
  SVT
end
#myvaltype{VV}(x::AbstractDependent{VV}) = VV
#NodeType{VV}(x::AbstractDependent{VV}) = VV

#asvec(x::SymDictVariateVals) = asvec(first(values(x.value))) #TODO: remove. Hack, because Sampler somehow leaves a one-key SymDict on the outside.

asvec(x::DictVariateVals) = Float64[x[k] for k in keys(x)]

#################### Base Methods ####################
function start(nd::NestedDictVariateVals)
  start(nd.value)
end

function next(nd::NestedDictVariateVals,state)
  next(nd.value,state)
end

function done(nd::NestedDictVariateVals,state)
  done(nd.value,state)
end

function keyvals(nd::SymDictVariateVals)
  nd.value
end

function keyvals(nd::VecDictVariateVals)
  [Pair(i,nd.value[i]) for i in 1:length(nd.value)]
end

function keys(nd::NestedDictVariateVals)
  mykeys = Vector{Tuple}()
  for kv in keyvals(nd) #TODO: refactor this as iterator using start, next, done
    if typeof(kv[2]) <: ScalarVariateType
      append!(mykeys,[(kv[1],)])
    else
      append!(mykeys,[(kv[1],k...) for k in keys(kv[2])])
    end
  end
  mykeys
end

function length(nd::NestedDictVariateVals)
  length(keys(nd))
end

function keys(ndd::AbstractElasticDependent)
  keys(ndd.value)
end

function length(ndd::AbstractElasticDependent)
  length(collect(keys(ndd)))
end

function ndims(ndd::AbstractElasticDependent)
  k = keys(ndd)
  if length(k) > 0
    return length(k[1])
  else
    return 1 #TODO: fix. this is just a guess.
  end
end

type KeyIter
  lims::Vector{Int}
end
start(k::KeyIter) = fill(1,length(k.lims))
function next(k::KeyIter,s)
  rval = (s...)
  for i in 1:length(s)
    if s[i]<k.lims[i]
      s[i] += 1
      return (rval,s)
    end
    s[i] = 1
  end
  return (rval,())
end
done(k::KeyIter,s) = (s == ())
length(k::KeyIter) = *(k.lims...)
letyep(k::KeyIter) = Tuple

function keys(x::VS) where VS<:ArrayVariateVals{SVT,N} where {SVT,N}
  KeyIter(Int[size(x)...])
end

function Base.show(io::IO, d::AbstractDependent)
  msg = string(ifelse(isempty(d.monitor), "An un", "A "),
               "monitored node of type \"", summary(d), "\"\n")
  print(io, msg)
  show(io, d.value)
end

function Base.showall(io::IO, d::AbstractDependent)
  show(io, d)
  print(io, "\nFunction:\n")
  show(io, "text/plain", first(code_typed(d.eval)))
  print(io, "\n\nSource Nodes:\n")
  show(io, d.sources)
  print(io, "\n\nTarget Nodes:\n")
  show(io, d.targets)
end

dims(d::AbstractDependent) = size(d.value)

function names(d::AbstractDependent)
  names(d, d.symbol)
end

function setmonitor!(d::AbstractFixedDependent, monitor::Bool)
  value = monitor ? Int[0] : Int[]
  setmonitor!(d, value)
end

function setmonitor!(d::AbstractElasticDependent, monitor::Bool) #TODO: If this works, just consolidate with above
  value = monitor ? Int[0] : Int[]
  setmonitor!(d, value)
end

function setmonitor!(d::AbstractFixedDependent, monitor::Vector{Int})
  values = monitor
  n = length(unlist(d))
  if n > 0 && !isempty(monitor)
    if monitor[1] == 0
      values = collect(1:n)
    elseif minimum(monitor) < 1 || maximum(monitor) > n
      throw(BoundsError())
    end
  end
  d.monitor = values
  d
end

function setmonitor!(d::AbstractElasticDependent, monitor::Vector{Int})
    d.monitor = collect(1:ndims(d))
    d
end

#################### Conversions ###############################

function convert{SVT<:ScalarVariateType}(T::Type{Vector{SVT}}, v::VecDictVariateVals)
  convert(T,v.value) #Works only if all values are scalars
end

function convert{SVT<:ScalarVariateType}(T::Type{SVT}, v::VecDictVariateVals)
  length(v.value) == 1 || throw(BoundsError())
  vec = convert(Vector{T},v.value) #Works only if all values are scalars
  vec[1]
end

function flatten{SVT<:ScalarVariateType}(v::VecDictVariateVals{SVT})
  convert(Vector{SVT},v)
end

function promote_rule{SVT<:ScalarVariateType,N}(::Type{Mamba.VecDictVariateVals{SVT}}, ::Type{Array{SVT,N}})
  Array{SVT,N}
end

const NumOrVVal = Union{Number, AbstractVariateVals}

function fixop(opp)
  @eval function ($opp)(x::NumOrVVal,y::NumOrVVal)
    ($opp)(promote(x,y)...)
  end
end

for op in [:asdt]
  fixop(op)
end

#################### Distribution Fallbacks ####################

unlist(d::AbstractDependent, transform::Bool=false) =
  unlist(d, d.value, transform)

unlist(d::AbstractDependent, x::Real, transform::Bool=false) = [x]

unlist(d::AbstractDependent, x::AbstractArray, transform::Bool=false) = vec(x)

unlist(d::AbstractElasticDependent, x::Associative, transform::Bool=false) = x

relist(d::AbstractFixedDependent, x::AbstractArray, transform::Bool=false) =
  relistlength(d, x, transform)[1]

relist(d::AbstractElasticDependent, x::Associative, transform::Bool=false) =
  x

logpdf(d::AbstractDependent, transform::Bool=false) = 0.0

logpdf(d::AbstractDependent, x, transform::Bool=false) = 0.0


#################### Logical ####################

@promote_scalarvariate ScalarLogical


#################### Constructors ####################

function Logical(f::Function, monitor::Union{Bool, Vector{Int}}=true)
  value = Float64(NaN)
  fx, src = modelfxsrc(depfxargs, f)
  l = ScalarLogical(value, :nothing, Int[], fx, src, Symbol[])
  setmonitor!(l, monitor)
end

function Logical(d::Integer, f::Function,
                 monitor::Union{Bool, Vector{Int}}=true)
  value = Array{Float64}(fill(0, d)...)
  fx, src = modelfxsrc(depfxargs, f)
  l = ArrayLogical{1,Float64,Array{Float64}}(value, :nothing, Int[], fx, src, Symbol[])
  setmonitor!(l, monitor)
end

function getindex(X::NestedDictVariateVals,i::Union{Symbol,Int64}...)
  return X.value[i[1]][i[2:end]...]
end

function getindex(X::NestedDictVariateVals,i::Union{Symbol,Int64})
  X.value[i]
end

function getindex(X::NestedDictVariateVals,all_i::Vector)
  [getindex(X,i) for i in all_i]
end

getindex(X::NestedDictVariateVals,i::Tuple) = getindex(X,i...)

function ensureIndexReady!(X::NestedDictVariateVals{SVT},i,subval::Bool=false) where SVT
  #generically, do nothing
  if subval
    if !haskey(X.value,i)
      X.value[i] = VecDictVariateVals{SVT}()
    end
  end
end

function ensureIndexReady!(X::VecDictVariateVals{SVT},i::Integer,subval::Bool=false) where SVT
  l = length(X.value)
  if l<i
    append!(X.value,fill(NaN,i-l))
    if subval
      X.value[i] = VecDictVariateVals{SVT}()
    end
  end
end

function setindex!(X::NestedDictVariateVals{SVT},v,i::Union{Symbol, Int64}...) where SVT
  ensureIndexReady!(X,i[1],true)
  X.value[i[1]][i[2:end]...] = v
end

function setindex_unsafe!(X::NestedDictVariateVals{SVT},v,i::Union{Symbol, Int64}) where SVT
  X.value[i] = v
end

function setindex!(X::NestedDictVariateVals{SVT},v,i::Union{Symbol, Int64}) where SVT
  ensureIndexReady!(X,i)
  X.value[i] = v
end

function setindex!(X::NestedDictVariateVals{SVT},v::Vector,::Colon) where SVT
  l = length(v)
  X[l] = v[l]
  for i in l-1:-1:1
    setindex_unsafe!(X,v[i],i)
  end
end

function copy(X::SymDictVariateVals{SVT}) where SVT
  newX = typeof(X)()
  for (key,value) in X
    newX[key] = copy(value)
  end
  newX
end

function copy(X::VecDictVariateVals{SVT}) where SVT
  newX = typeof(X)()
  l = length(X.value)
  newX[l] = X[l]
  for i in l-1:-1:1
    newX.value[i] = copy(X.value[i])
  end
  newX
end


#################### Updating ####################

function setinits!(l::AbstractLogical, m::AbstractModel, ::Any=nothing)
  l.value = l.eval(m)
  setmonitor!(l, l.monitor)
end

function update!(l::AbstractLogical, m::AbstractModel)
  l.value = l.eval(m)
  l
end


#################### Distribution Methods ####################

relistlength(d::ScalarLogical, x::AbstractArray, transform::Bool=false) =
  (x[1], 1)

function relistlength(d::ArrayLogical, x::AbstractArray, transform::Bool=false)
  n = length(d)
  value = reshape(x[1:n], size(d))
  (value, n)
end


#################### Stochastic ####################

#################### Base Methods ####################

@promote_scalarvariate ScalarStochastic

function Base.showall(io::IO, s::AbstractStochastic)
  show(io, s)
  print(io, "\n\nDistribution:\n")
  show(io, s.distr)
  print(io, "\nFunction:\n")
  show(io, "text/plain", first(code_typed(s.eval)))
  print(io, "\n\nSource Nodes:\n")
  show(io, s.sources)
  print(io, "\n\nTarget Nodes:\n")
  show(io, s.targets)
end


#################### Constructors ####################

function Stochastic(f::Function, monitor::Union{Bool, Vector{Int}}=true)
  value = Float64(NaN)
  fx, src = modelfxsrc(depfxargs, f)
  s = ScalarStochastic(value, :nothing, Int[], fx, src, Symbol[],
                       NullUnivariateDistribution())
  setmonitor!(s, monitor)
end

function Stochastic(d::Integer, f::Function,
                    monitor::Union{Bool, Vector{Int}}=true)
  value = Array{Float64}(fill(0, d)...)
  fx, src = modelfxsrc(depfxargs, f)
  s = ArrayStochastic{1,myvaltype(value),typeof(value)}(value, :nothing, Int[], fx, src, Symbol[],
                      NullUnivariateDistribution())
  setmonitor!(s, monitor)
end

function Stochelastic(d::Integer, f::Function,
                    monitor::Union{Bool, Vector{Int}}=true)

  value = VecDictVariateVals{Float64}()
  k = fill(1,d)
  value[k...] = NaN
  fx, src = modelfxsrc(depfxargs, f)
  s = DictStochastic{myvaltype(value),typeof(value)}(value, :nothing, Int[], fx, src, Symbol[],
                      NullUnivariateDistribution())
  setmonitor!(s, monitor)
end


#################### Updating ####################

function setinits!(s::AbstractStochastic, m::AbstractModel, x)
  throw(ArgumentError("incompatible initial value for node : $(s.symbol). s::$(typeof(s)), m::$(typeof(m))"))
end

function setinits!(s::ScalarStochastic, m::AbstractModel, x::Real)
  s.value = convert(myvaltype(s), x)
  s.distr = s.eval(m)
  setmonitor!(s, s.monitor)
end

function setinits!(s::ArrayStochastic, m::AbstractModel, x::DenseArray)
  s.value = convert(typeof(s.value), copy(x))
  s.distr = s.eval(m)
  if !isa(s.distr, UnivariateDistribution) && dims(s) != dims(s.distr)
    throw(DimensionMismatch("incompatible distribution for stochastic node"))
  end
  setmonitor!(s, s.monitor)
end

function setinits!(s::ScalarStochastic, m::ElasticModel, x::Real)
  s.value = convert(myvaltype(s), x)
  s.distr = s.eval(m)
  setmonitor!(s, s.monitor)
end

function setinits!(s::DictStochastic, m::ElasticModel, x::Real)
  s.value = convert(VecDictVariateVals{myvaltype(s)}, [x])
  s.distr = s.eval(m)
  setmonitor!(s, s.monitor)
end

function setinits!(s::DictStochastic, m::ElasticModel, x)
  vtype = myvaltype(s)
  s.value = VecDictVariateVals{vtype}([vtype(xi) for xi in x])
  s.distr = s.eval(m)
  if !isa(s.distr, UnivariateDistribution) && false # dims(s) != dims(s.distr)
    throw(DimensionMismatch("incompatible distribution for stochastic node"))
  end
  setmonitor!(s, s.monitor)
end


function update!(s::AbstractStochastic, m::AbstractModel)
  s.distr = s.eval(m)
  s
end


#################### Distribution Methods ####################

function unlist(s::AbstractFixedStochastic, transform::Bool=false; monitoronly=false)
  lvalue = unlist(s, s.value, transform)

  monitoronly ? lvalue[s.monitor] : lvalue
end

function unlist(s::AbstractFixedStochastic, x::Real, transform::Bool=false)
  unlist(s, [x], transform)
end

function unlist(s::AbstractFixedStochastic, x::AbstractArray, transform::Bool=false)
  transform ? unlist_sub(s.distr, link_sub(s.distr, x)) :
              unlist_sub(s.distr, x)
end

function relist(s::AbstractFixedStochastic, x::AbstractArray, transform::Bool=false)
  relistlength(s, x, transform)[1]
end

function relistlength(s::AbstractFixedStochastic, x::AbstractArray,
                      transform::Bool=false)
  value, n = relistlength_sub(s.distr, s, x)
  (transform ? invlink_sub(s.distr, value) : value, n)
end

function logpdf(s::AbstractStochastic, transform::Bool=false)
  logpdf(s, s.value, transform)
end

function logpdf(s::AbstractStochastic, x, transform::Bool=false)
  logpdf_sub(s.distr, x, transform)
end

rand(s::AbstractStochastic) = rand_sub(s.distr, s.value)
