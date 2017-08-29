#################### Model Simulation ####################

function gettune(m::AbstractModel, block::Integer)
  block == 0 && return gettune(m)
  m.samplers[block].tune
end

function gettune(m::AbstractModel)
  Any[gettune(m, i) for i in 1:length(m.samplers)]
end

function settune!(m::AbstractModel, tune, block::Integer)
  block == 0 && return settune!(m, tune)
  m.samplers[block].tune = tune
end

function settune!(m::AbstractModel, tune::Vector{Any})
  nsamplers = length(m.samplers)
  ntune = length(tune)
  nsamplers == ntune ||
    throw(DimensionMismatch(
      "tried to assign $ntune tune elements to $nsamplers samplers"
    ))

  for i in 1:nsamplers
    settune!(m, tune[i], i)
  end
end


function gradlogpdf(m::AbstractModel, block::Integer=0, transform::Bool=false;
                    dtype::Symbol=:forward)
  x0 = unlist(m, block, transform)
  value = gradlogpdf!(m, x0, block, transform, dtype=dtype)
  relist!(m, x0, block, transform)
  value
end

function gradlogpdf{T<:Real}(m::AbstractModel, x::AbstractVector{T}, block::Integer=0,
                             transform::Bool=false; dtype::Symbol=:forward)
  x0 = unlist(m, block)
  value = gradlogpdf!(m, x, block, transform, dtype=dtype)
  relist!(m, x0, block)
  value
end

function gradlogpdf!{T<:Real}(m::AbstractModel, x::AbstractVector{T}, block::Integer=0,
                              transform::Bool=false; dtype::Symbol=:forward)
  f = x -> logpdf!(m, x, block, transform)
  gradient(f, convert(Vector{T}, x), dtype)
end


function logpdf(m::AbstractModel, block::Integer=0, transform::Bool=false)
  params = keys(m, :block, block)
  targets = keys(m, :target, block)
  logpdf(m, params, transform) + logpdf(m, setdiff(targets, params))
end

function logpdf(m::AbstractModel, nodekeys::Vector{Symbol}, transform::Bool=false)
  lp = 0.0
  for key in nodekeys
    lp += logpdf(m[key], transform)
    isfinite(lp) || break
  end
  lp
end
#
# function logpdf(m::AbstractModel, nodekeys::Vector{Symbol}, transform::Bool=false;
#       kargs...)
#   logpdf(m,nodekeys,transform) #ignore index. Very inefficient but works. TODO: implement single-index logpdf
#   # lp = 0.0
#   # for key in nodekeys
#   #   lp += logpdf(m[key], transform; kargs...)
#   #   isfinite(lp) || break
#   # end
#   # lp
# end

function logpdf{T<:Real}(m::AbstractModel, x::AbstractArray{T}, block::Integer=0,
                         transform::Bool=false)
  x0 = unlist(m, block)
  lp = logpdf!(m, x, block, transform)
  relist!(m, x0, block)
  lp
end

function logpdf!{T<:ScalarVariateType}(m::AbstractModel, x::AbstractVariateVals{T}, block::Integer=0,
                          transform::Bool=false)
  params = keys(m, :block, block)
  targets = keys(m, :target, block)
  m[params] = relist(m, x, params, transform)
  lp = logpdf(m, setdiff(params, targets), transform)
  for key in targets
    isfinite(lp) || break
    node = m[key]
    update!(node, m)
    lp += key in params ? logpdf(node, transform) : logpdf(node)
  end
  lp
end


function sample!(m::AbstractModel, block::Integer=0)
  m.iter += 1
  isoneblock = block != 0
  blocks = isoneblock ? block : 1:length(m.samplers)
  for b in blocks
    sampler = m.samplers[b]
    value = sampler.eval(m, b)
    if value != nothing
      m[sampler.params] = value
      update!(m, b)
    end
  end
  m.iter -= isoneblock
  m
end


function unlist(m::AbstractModel, block::Integer=0, transform::Bool=false)
  unlist(m, keys(m, :block, block), transform)
end

function unlist(m::AbstractModel, monitoronly::Bool)
  f = function(key)
    node = m[key]
    println("qqqq ", @which unlist(node))
    r = unlist(node; monitoronly=monitoronly)
  end
  vcat(map(f, keys(m, :dependent))...)
end

function unlist(m::ElasticModel, monitoronly::Bool)
  f = function(key)
    node = m[key]
    r = unlist(node; monitoronly=monitoronly)
    VecDictVariateVals{myvaltype(m)}(r)
  end
  SymDictVariateVals{myvaltype(m)}(k => f(k) for k in keys(m, :dependent))
end

function unlist(m::Model, nodekeys::Vector{Symbol}, transform::Bool=false)
  vcat(map(key -> unlist(m[key], transform), nodekeys)...)
end

function unlist{SVT}(m::ElasticModel{SVT}, nodekeys::Vector{Symbol}, transform::Bool=false)
  function fixtype(v::ScalarVariateType)
    SVT(v)
  end
  function fixtype(v)
    VecDictVariateVals{SVT}(Vector{LeafOrBranch{SVT}}([fixtype(sv) for sv in v]))
  end
  if length(nodekeys) == 1
    return fixtype(unlist(m[nodekeys[1]], transform))
  else
    return SymDictVariateVals{SVT}(key => fixtype(unlist(m[key], transform)) for key in nodekeys)
  end
end


function relist(m::AbstractModel, x, block::Integer=0,
                         transform::Bool=false)
  relist(m, x, keys(m, :block, block), transform)
end

function relist{T<:Real}(m::Model, x::AbstractArray{T},
                         nodekeys::Vector{Symbol}, transform::Bool=false)
  values = Dict{Symbol,Any}()
  N = length(x)
  offset = 0
  for key in nodekeys
    value, n = relistlength(m[key], view(x, (offset + 1):N), transform)
    values[key] = value
    offset += n
  end
  offset == length(x) ||
    throw(ArgumentError("incompatible number of values to put in nodes"))
  values
end

function relist(m::AbstractModel, x,
                         nodekeys::Vector{Symbol}, transform::Bool=false)
  x
end

function relist!(m::AbstractModel, x, block::Integer=0,
                 transform::Bool=false)
  nodekeys = keys(m, :block, block)
  values = relist(m, x, nodekeys, transform)
  for key in nodekeys
    try
      m[key].value = values[key]
    catch
      println("Failed: m[key].value = values[key]") #qqqq
      println(values[key])
      println(typeof(m[key].value))
    end
  end
  update!(m, block)
end

function relist!{T<:Real}(m::AbstractModel, x::AbstractVariateVals{T}, nodekey::Symbol,
                          transform::Bool=false)
  node = m[nodekey]
  m[nodekey] = relist(node, x, transform)
  update!(m, node.targets)
end


function update!(m::AbstractModel, block::Integer=0)
  nodekeys = block == 0 ? keys(m, :dependent) : m.samplers[block].targets
  update!(m, nodekeys)
end

function update!(m::AbstractModel, nodekeys::Vector{Symbol})
  for key in nodekeys
    update!(m[key], m)
  end
  m
end
