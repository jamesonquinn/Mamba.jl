
################################## Dependent ####################

const depfxargs = [(:model, Mamba.Model)]


#################### Base Methods ####################

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

function setmonitor!(d::AbstractDependent, monitor::Bool)
  value = monitor ? Int[0] : Int[]
  setmonitor!(d, value)
end

function setmonitor!(d::AbstractDependent, monitor::Vector{Int})
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


#################### Distribution Fallbacks ####################

unlist(d::AbstractDependent, transform::Bool=false) =
  unlist(d, d.value, transform)



unlist(d::AbstractDependent, x::Real, transform::Bool=false) = [x]

unlist(d::AbstractDependent, x::AbstractArray, transform::Bool=false) = vec(x)

relist(d::AbstractDependent, x::AbstractArray, transform::Bool=false) =
  relistlength(d, x, transform)[1]

logpdf(d::AbstractDependent, transform::Bool=false) = 0.0

logpdf(d::AbstractDependent, x, transform::Bool=false) = 0.0


#################### Logical ####################

@promote_scalarvariate ScalarLogical


#################### Constructors ####################

function Logical(f::Function, monitor::Union{Bool, Vector{Int}}=true)
  value = Float64(NaN)
  fx, src = modelfxsrc(depfxargs, f)
  l = ScalarLogical(OneObservedValue(value), :nothing, Int[], fx, src, Symbol[])
  setmonitor!(l, monitor)
end

function Logical(d::Integer, f::Function,
                 monitor::Union{Bool, Vector{Int}}=true)
  value = Array{Float64}(fill(0, d)...)
  fx, src = modelfxsrc(depfxargs, f)
  l = ArrayLogical(ObservedValues(value), :nothing, Int[], fx, src, Symbol[])
  setmonitor!(l, monitor)
end


#################### Updating ####################
function deint(x::Real)
    x
end
function deint(x::Integer)
    Float64(x)
end
function deint{T<:Integer}(x::AbstractArray{T})
    value = similar(x,Float64)
    value = convert(typeof(value),x)
    value
end
function deint{T<:Real}(x::AbstractArray{T})
    x
end

ObservedValue(x) = deint(x)

function setinits!(l::AbstractLogical, m::Model, ::Any=nothing)
  l.value = ObservedValue(l.eval(m))
  #size(value) == ()) ? OneObservedValue(value) : ObservedValues(value)
  setmonitor!(l, l.monitor)
end

function update!(l::AbstractLogical, m::Model)
  l.value = ObservedValue(l.eval(m))
  #l.value = (size(value) == ()) ? OneObservedValue(value) : ObservedValues(value)
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
  s = ScalarStochastic(OneObservedValue(value), :nothing, Int[], fx, src, Symbol[],
                       NullUnivariateDistribution())
  setmonitor!(s, monitor)
end

function Stochastic(d::Integer, f::Function,
                    monitor::Union{Bool, Vector{Int}}=true)
  value = Array{Float64}(fill(0, d)...)
  fx, src = modelfxsrc(depfxargs, f)
  s = ArrayStochastic(ObservedValues(value), :nothing, Int[], fx, src, Symbol[],
                      NullUnivariateDistribution())
  setmonitor!(s, monitor)
end


#################### Updating ####################
function setinits!(s::ScalarStochastic, m::Model, x::Real)
  s.value = OneObservedValue(x)
  s.distr = s.eval(m)
  setmonitor!(s, s.monitor)
end

function setinits!(s::ArrayStochastic, m::Model, x::AbstractArray)
  s.value = ObservedValues(x)
  s.distr = s.eval(m)
  if !isa(s.distr, UnivariateDistribution) && dims(s) != dims(s.distr)
    throw(DimensionMismatch("incompatible distribution for stochastic node: "*string(dims(s))*", "*string(dims(s.distr))))
  end
  setmonitor!(s, s.monitor)
end

function setinits!(s::AbstractStochastic, m::Model, x)
  throw(ArgumentError("incompatible initial value for node : $(s.symbol)"))
end

function update!(s::AbstractStochastic, m::Model)
  s.distr = s.eval(m)
  s
end


#################### Distribution Methods ####################

function unlist(s::AbstractStochastic, transform::Bool=false)
  unlist(s, s.value, transform)
end

function unlist(s::AbstractStochastic, x::Real, transform::Bool=false)
  unlist(s, [x], transform)
end

function unlist(s::AbstractStochastic, x::AbstractArray, transform::Bool=false)
  print(transform ? (@which unlist_sub(s.distr, link_sub(s.distr, x))) :
              @which unlist_sub(s.distr, x))
  transform ? unlist_sub(s.distr, link_sub(s.distr, x)) :
              unlist_sub(s.distr, x)
  print("\nfinished unlist\n")
end

function unlist(s::AbstractStochastic, x::ObservedValues, transform::Bool=false)
  print(transform ? (@which unlist_sub(s.distr, link_sub(s.distr, x))) :
              @which unlist_sub(s.distr, x))
  transform ? unlist_sub(s.distr, link_sub(s.distr, x)) :
              unlist_sub(s.distr, x)
end

function relist(s::AbstractStochastic, x::AbstractArray, transform::Bool=false)
  relistlength(s, x, transform)[1]
end

function relistlength(s::AbstractStochastic, x::AbstractArray,
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
