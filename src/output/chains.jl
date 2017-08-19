#################### Chains ####################

#################### DictChainVal ####################

function myvaltype(x::AbstractDictChainVal{SVT}) where SVT
  SVT
end

function getindex(cv::DictChainVal,i1::Int,i2::Int,i3::Tuple)
  cv.vals[i1,i2][i3]
end

function setindex!(cv::DictChainVal,v,i1::Int,i2::Int,i3::Tuple)
  cv.vals[i1,i2][i3] = v
end

function getindex(cv::VecDictChainVal,i1::Int,i3::Tuple)
  cv.vals[i1][i3]
end

function setindex!(cv::VecDictChainVal,v,i1::Int,i3::Tuple)
  cv.vals[i1][i3] = v
end

function getindex(cv::DictChainVal,i1::Int,i2::Int,i3::Colon)
  cv.vals[i1,i2]
end

function setindex!(cv::DictChainVal,v,i1::Int,i2::Int,i3::Union{Colon,UnitRange}) #TODO: figure out why UnitRange is necessary here
  cv.vals[i1,i2] = v
end

# const IntOrColon = Union{Int,Colon}
#
#
# function getindex(cv::DictChainVal,i1::IntOrColon,i2::IntOrColon,i3::Colon)
#   cv.vals[i1,i2]
# end
#
# function setindex!(cv::DictChainVal,v,i1::IntOrColon,i2::IntOrColon,i3::Colon)
#   cv.vals[i1,i2] = v
# end
#
# function getindex(cv::DictChainVal,i1::IntOrColon,i2::IntOrColon,i3::Tuple)
#   cv.vals[i1,i2]
# end
#
# function setindex!(cv::DictChainVal,v,i1::IntOrColon,i2::IntOrColon,i3::Tuple)
#   cv.vals[i1,i2] = v
# end

function cat(d::Int, cvs::AbstractVariateVals{SVT}...) where SVT #qqqq I think this type is too loose, maybe have to go rescue old version of this function?
  allvals = [cv.vals for cv in cvs]
  DictChainVal{SVT}(cat(d,[cv.vals for cv in cvs]...)) #will fail for d==3
  #problem: Cannot `convert` an object of type Array{Mamba.DictVariateVals{Float64},2} to an object of type Mamba.AbstractDictChainVal{SVT,2} where SVT
  #ie, we're passing in something shaped like the one attribute, and it's trying to "convert" not "construct"
  #qqqq above comments ara a misdiagnosis (?), problem was lack of {SVT}
  #DictChainVal(cat(d,[cv.vals for cv in cvs]...)) #will fail for d==3
end
function vcat(cvs::DictChainVal...)
  cat(2,cvs...)
end

function size(x::AbstractDictChainVal)
  (size(x.vals)...,-1)
end

function size(x::AbstractDictChainVal,i::Int)
  size(x)[i]
end

function size(x::AbstractDictChainVal,ii::Int...)
  size(x)[[i for i in ii]]
end

function size(x::AbstractChains,vargs...)
  size(x.value,vargs...)
end

function fill!(x::Mamba.AbstractDictChainVal,v)
  fill!(x.vals,NullDictVariateVals{myvaltype(x)}())
end

#################### Constructors ####################

function MakeChainVal(VS_T::Type{VS},l1,l2,l3) where VS<: ArrayVariateVals
  Array{myvaltype(VS_T)}(l1,l2,l3)
end

function MakeChainVal(VST::Type{VS},l1,l2,l3) where VS<: DictVariateVals
  MakeDictChainVal(myvaltype(VST),l1,l2,l3)
end

function Chains(iters::Integer, params::Integer, vst::VST=VectorVariateVals{Float64};

               start::Integer=1, thin::Integer=1, chains::Integer=1,
               names::Vector{T}=AbstractString[]) where T<:AbstractString where VST<:Type{VS} where VS<: AbstractVariateVals

  value = MakeChainVal(VS, length(start:thin:iters), chains, params)
  fill!(value, NaN)
  Chains(value, start=start, thin=thin, names=names)
end

function Chains{T<:Real, U<:AbstractString, V<:Integer}(value::Array{T, 3};
               start::Integer=1, thin::Integer=1,
               names::Vector{U}=AbstractString[], chains::Vector{V}=Int[])
  n, m, p = size(value)


  if isempty(names)
    names = map(i -> "Param$i", 1:p)
  elseif length(names) != p
    throw(DimensionMismatch("size(value, 3) and names length differ")) #qq
  end
  if isempty(chains)
    chains = collect(1:m)
  elseif length(chains) != m
    throw(DimensionMismatch("size(value, 2) and chains length differ")) #qq
  end

  #v = convert(Array{Float64, 3}, value) #hard-coded Float64
  Chains(value, range(start, thin, n), AbstractString[names...], Int[chains...])
end

function Chains{T<:Real, U<:AbstractString, V<:Integer}(value::AbstractChainVal{T}, namecheck::Bool=false;

               start::Integer=1, thin::Integer=1,
               names::Vector{U}=AbstractString[], chains::Vector{V}=Int[])
  n, m, p = size(value)
  if isempty(chains)
    chains = collect(1:m)
  elseif length(chains) != m
    throw(DimensionMismatch("size(value, 2) and chains length differ")) #qq
  end

  #v = convert(Array{Float64, 3}, value) #hard-coded Float64
  Chains(value, range(start, thin, n), AbstractString[names...], Int[chains...])
end

function Chains{T<:Real, U<:AbstractString}(value::Matrix{T};
               start::Integer=1, thin::Integer=1,
               names::Vector{U}=AbstractString[], chains::Integer=1)
  Chains(reshape(value, size(value, 1), size(value, 2), 1), start=start,
         thin=thin, names=names, chains=Int[chains])
end

function Chains{T<:Real}(value::Vector{T};
               start::Integer=1, thin::Integer=1,
               names::AbstractString="Param1", chains::Integer=1)
  Chains(reshape(value, length(value), 1, 1), start=start, thin=thin,
         names=AbstractString[names], chains=Int[chains])
end


#################### Indexing ####################

function Base.getindex(c::Chains, window, chains, names) #qq
  inds1 = window2inds(c, window)
  inds2 = names2inds(c, names)
  Chains(c.value[inds1, chains, inds2], #qq
         start = first(c) + (first(inds1) - 1) * step(c),
         thin = step(inds1) * step(c), names = c.names[inds2],
         chains = c.chains[chains])
end

function Base.setindex!(c::AbstractChains, value, iters, chains, names) #qq
  setindex!(c.value, value, iters2inds(c, iters), chains, names2inds(c, names)) #qq
end

macro mapiters(iters, c)
  quote
    ($(esc(iters)) - first($(esc(c)))) / step($(esc(c))) + 1.0
  end
end

window2inds(c::AbstractChains, window) =
  throw(ArgumentError("$(typeof(window)) iteration indexing is unsupported"))
window2inds(c::AbstractChains, ::Colon) = window2inds(c, 1:size(c, 1))
window2inds(c::AbstractChains, window::Range) = begin
  range = @mapiters(window, c)
  a = max(ceil(Int, first(range)), 1)
  b = step(window)
  c = min(floor(Int, last(range)), size(c.value, 1))
  a:b:c
end

iters2inds(c::AbstractChains, iters) = iters
iters2inds(c::AbstractChains, ::Colon) = 1:size(c.value, 1)
iters2inds(c::AbstractChains, iters::Range) =
  convert(StepRange{Int, Int}, @mapiters(iters, c))
iters2inds(c::AbstractChains, iter::Real) = Int(@mapiters(iter, c))
iters2inds{T<:Real}(c::AbstractChains, iters::Vector{T}) =
  Int[@mapiters(i, c) for i in iters]

names2inds(c::AbstractChains, names) = names
names2inds(c::AbstractChains, ::Colon) = 1:size(c.value, 2)
names2inds(c::AbstractChains, name::Real) = [name]
names2inds(c::AbstractChains, name::AbstractString) = names2inds(c, [name])
names2inds{T<:AbstractString}(c::AbstractChains, names::Vector{T}) =
  indexin(names, c.names)


#################### Concatenation ####################

function Base.cat(dim::Integer, c1::AbstractChains, args::AbstractChains...)
  dim == 1 ? catTime(c1, args...) :
  dim == 2 ? catChains(c1, args...) :
  dim == 3 ? catParams(c1, args...) :
    throw(ArgumentError("cannot concatenate along dimension $dim"))
end

function catTime(c1::AbstractChains, args::AbstractChains...)
  range = c1.range
  for c in args
    last(range) + step(range) == first(c) ||
      throw(ArgumentError("noncontiguous chain iterations"))
    step(range) == step(c) ||
      throw(ArgumentError("chain thinning differs"))
    range = first(range):step(range):last(c)
  end

  names = c1.names
  all(c -> c.names == names, args) ||
    throw(ArgumentError("chain names differ"))

  chains = c1.chains
  all(c -> c.chains == chains, args) ||
    throw(ArgumentError("sets of chains differ"))

  value = cat(1, c1.value, map(c -> c.value, args)...)
  Chains(value, start=first(range), thin=step(range), names=names,
         chains=chains)
end

function catParams(c1::AbstractChains, args::AbstractChains...)
  range = c1.range
  all(c -> c.range == range, args) ||
    throw(ArgumentError("chain ranges differ"))

  names = c1.names
  n = length(names)
  for c in args
    names = union(names, c.names)
    n += length(c.names)
    n == length(names) ||
      throw(ArgumentError("non-unique chain names"))
  end

  chains = c1.chains
  all(c -> c.chains == chains, args) ||
    throw(ArgumentError("sets of chains differ"))

  value = cat(3, c1.value, map(c -> c.value, args)...) #qq
  Chains(value, start=first(range), thin=step(range), names=names,
         chains=chains)
end

function catChains(c1::AbstractChains, args::AbstractChains...)
  range = c1.range
  all(c -> c.range == range, args) ||
    throw(ArgumentError("chain ranges differ"))

  names = c1.names
  all(c -> c.names == names, args) ||
    begin
      errStr = "chain names differ:"
      for c in args
        if c.names != names
          errStr = string(errStr," ", string(c.names),
                       " ; ",
                       string(names))
        end
      end
      throw(ArgumentError(errStr))
    end

  value = cat(2, c1.value, map(c -> c.value, args)...) #qq
  Chains(value, start=first(range), thin=step(range), names=names)
end

Base.hcat(c1::AbstractChains, args::AbstractChains...) = cat(3, c1, args...)

Base.vcat(c1::AbstractChains, args::AbstractChains...) = cat(1, c1, args...)


#################### Base Methods ####################

function Base.keys(c::AbstractChains)
  c.names
end

function Base.show(io::IO, c::AbstractChains)
  print(io, "Object of type \"$(summary(c))\"\n\n")
  println(io, header(c))
  show(io, c.value)
end

function Base.size(c::AbstractChains)
  dim = size(c.value)
  last(c), dim[2], dim[3]
end

function Base.size(c::AbstractChains, ind)
  size(c)[ind]
end

Base.first(c::AbstractChains) = first(c.range)
Base.step(c::AbstractChains) = step(c.range)
Base.last(c::AbstractChains) = last(c.range)


#################### Auxilliary Functions ####################

function combine(c::AbstractChains)
  n, m, p = size(c.value)
  value = Array{Float64}(n * m, p)
  for j in 1:p
    idx = 1
    for i in 1:n, k in 1:m
      value[idx, j] = c.value[i, k, j]
      idx += 1
    end
  end
  value
end

function header(c::AbstractChains)
  string(
    "Iterations = $(first(c)):$(last(c))\n",
    "Thinning interval = $(step(c))\n",
    "Chains = $(join(map(string, c.chains), ","))\n",
    "Samples per chain = $(length(c.range))\n"
  )
end

function indiscretesupport(c::AbstractChains,
                           bounds::Tuple{Real, Real}=(0, Inf))
  nrows, nchains, nvars = size(c.value)
  result = Array{Bool}(nvars * (nrows > 0))
  for i in 1:nvars
    result[i] = true
    for j in 1:nrows, k in 1:nchains
      x = c.value[j, i, k]
      if !isinteger(x) || x < bounds[1] || x > bounds[2]
        result[i] = false
        break
      end
    end
  end
  result
end

function link(c::AbstractChains)
  cc = copy(c.value)
  for j in 1:length(c.names)
    x = cc[:, :, j]
    if minimum(x) > 0.0
      cc[:, :, j] = maximum(x) < 1.0 ? logit(x) : log(x)
    end
  end
  cc
end
