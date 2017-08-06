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
function myvaltype(x::Type{X}) where X<:DictVariateVals{SVT,N} where N where SVT <: ScalarVariateType
  SVT
end
#myvaltype{VV}(x::AbstractDependent{VV}) = VV
#NodeType{VV}(x::AbstractDependent{VV}) = VV
keytype{K,SVT}(x::DictVariateVals{SVT,K}) = K
#myvaltype{VV}(x::DictVariateVals{K,VV} where K) = VV #qqqq apparently this is in Base already?

#################### Base Methods ####################
function start(nd::NestedDictVariateVals)
  start(nd.vals)
end

function next(nd::NestedDictVariateVals,state)
  next(nd.vals,state)
end

function done(nd::NestedDictVariateVals,state)
  done(nd.vals,state)
end

function keyvals(nd::SymDictVariateVals)
  nd.vals
end

function keyvals(nd::VecDictVariateVals)
  [Pair(i,nd.vals[i]) for i in 1:length(nd.vals)]
end

function keys(nd::NestedDictVariateVals)
  mykeys = Vector{Tuple}()
  for kv in keyvals(nd) #TODO: refactor this as iterator using start, next, done
    if typeof(kv[2]) <: ScalarVariateType
      append!(mykeys,[(kv[1],)])
    else
      append!(mykeys,[tuple(kv[1],k) for k in keys(kv[2])])
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
    print(k)
    return length(k[1])
  else
    return 1 #qqqq this is just a guess
  end
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
  convert(T,v.vals) #Works only if all values are scalars
end

function convert{SVT<:ScalarVariateType}(T::Type{SVT}, v::VecDictVariateVals)
  length(v.vals) == 1 || throw(BoundsError())
  vec = convert(Vector{T},v.vals) #Works only if all values are scalars
  vec[1]
end

function flatten{SVT<:ScalarVariateType}(v::VecDictVariateVals{SVT})
  convert(Vector{SVT},v)
end

function promote_rule{SVT<:ScalarVariateType,N}(::Type{Mamba.VecDictVariateVals{SVT}}, ::Type{Array{SVT,N}})
  Array{SVT,N}
end

NumOrVVal = Union{Number, AbstractVariateVals}

macro fixop(op)
  return :( $op(x::NumOrVVal,y::NumOrVVal) = $op(promote(x,y)...) )
end
for op in [+,-,*,/,^]
  @fixop(op)
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
  l = ArrayLogical(value, :nothing, Int[], fx, src, Symbol[])
  setmonitor!(l, monitor)
end

function getindex(X::NestedDictVariateVals,i...)
  if length(i) == 1
    return X.vals[i[1]]
  else
    return X.vals[i[1]][i[2:end]]
  end
end

function ensureIndexReady!(X::NestedDictVariateVals,i::Integer)
  #generically, do nothing
end

function ensureIndexReady!(X::VecDictVariateVals,i::Integer)
  l = length(X.vals)
  if l<i
    append!(X.vals,fill(NaN,i-l))
  end
end

function setindex!(X::NestedDictVariateVals,v,i...)
  ensureIndexReady!(X,i[1])
  if length(i) == 1
    X.vals[i[1]] = v
  else
    if !haskey(X.vals,i[1])
      X.vals[i[1]] = VecDictVariateVals{typeof(v)}()
    end
    X.vals[i[1]][i[2:end]] = v
  end
end


#################### Updating ####################

function setinits!(l::AbstractLogical, m::Model, ::Any=nothing)
  l.value = l.eval(m)
  setmonitor!(l, l.monitor)
end

function update!(l::AbstractLogical, m::Model)
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
  s = DictStochastic{keytype(value),myvaltype(value),typeof(value)}(value, :nothing, Int[], fx, src, Symbol[],
                      NullUnivariateDistribution())
  setmonitor!(s, monitor)
end


#################### Updating ####################

function setinits!(s::AbstractStochastic, m::AbstractModel, x)
  throw(ArgumentError("incompatible initial value for node : $(s.symbol)"))
end

function setinits!(s::ScalarStochastic, m::Model, x::Real)
  s.value = convert(myvaltype(s), x)
  s.distr = s.eval(m)
  setmonitor!(s, s.monitor)
end

function setinits!(s::ArrayStochastic, m::Model, x::DenseArray)
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

function unlist(s::AbstractFixedStochastic, transform::Bool=false)
  unlist(s, s.value, transform)
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
