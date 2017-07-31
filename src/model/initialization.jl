#################### Model Initialization ####################

function setinits!(m::AbstractModel, inits::Dict{Symbol, Any})
  m.hasinputs || throw(ArgumentError("inputs must be set before inits"))
  m.iter = 0
  for key in keys(m, :dependent)
    node = m[key]
    if isa(node, AbstractStochastic)
      haskey(inits, key) ||
        throw(ArgumentError("missing initial value for node : $key"))
      setinits!(node, m, inits[key])
    else
      setinits!(node, m)
    end
  end
  m.hasinits = true
  m
end

function setinits!(m::AbstractModel, inits::Vector{Dict{Symbol, Any}})
  n = length(inits)
  myStateType = AbstractModelState{nodetype(m)}
  m.states = Array{myStateType}(n)
  for i in n:-1:1
    setinits!(m, inits[i])
    m.states[i] = myStateType(unlist(m), deepcopy(gettune(m)))
  end
  m
end

function setinputs!(m::AbstractModel, inputs::Dict{Symbol, Any})
  for key in keys(m, :input)
    haskey(inputs, key) ||
      throw(ArgumentError("missing inputs for node : $key"))
    isa(inputs[key], AbstractDependent) &&
      throw(ArgumentError("inputs cannot be Dependent types"))
    m.nodes[key] = deepcopy(inputs[key])
  end
  m.hasinputs = true
  m
end

function setsamplers!{T<:Sampler}(m::AbstractModel, samplers::Vector{T})
  m.samplers = deepcopy(samplers)
  for sampler in m.samplers
    sampler.targets = keys(m, :target, sampler.params)
  end
  m
end
