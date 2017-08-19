#################### Syntactic sugar ####################
|(a::Void,b)=b
const Maybe{T} = Union{T, Void}
isnull(::Void) = true

#################### Model Expression Operators ####################
TypeLike = Union{UnionAll,DataType,Type}
UofVofTs = Vector{Tuple{Symbol, X}} where X<:TypeLike

function modelfx(literalargs::UofVofTs, f::Function)
  modelfxsrc(literalargs, f)[1]
end

function modelfxsrc(literalargs::UofVofTs, f::Function)
  args = Expr(:tuple, map(arg -> Expr(:(::), arg[1], arg[2]), literalargs)...)
  expr, src = modelexprsrc(f, literalargs)
  fx = eval(Expr(:function, args, expr))
  (fx, src)
end

dtof(t::DataType) = t

dtof(t::TypeVar) = t.ub

dtof(t::UnionAll) = t

function modelexprsrc(f::Function, literalargs::UofVofTs)
  m = first(methods(f).ms)
  argnames = Vector{Any}(m.nargs)
  ccall(:jl_fill_argnames, Void, (Any, Any), m.source, argnames)

  fkeys = Symbol[argnames[2:end]...]
  if isa(m.sig, UnionAll)
    msig = m.sig.body
  else
    msig = m.sig
  end
  ftypes = TypeLike[dtof(p) for p in msig.parameters[2:end]]
  n = length(fkeys)

  literalinds = Int[]
  for (key, T) in literalargs
    i = findfirst(fkey -> fkey == key, fkeys)
    if i != 0
      if ftypes[i] <: T
        push!(literalinds, i)
      else
        println(string("not literalinds: ",key," ",i," ",i != 0 && ftypes[i]," ",T)) #TODO: better error reporting, remove
      end
    end
  end
  nodeinds = setdiff(1:n, literalinds)

  all(T -> T == Any, ftypes[nodeinds]) ||
    throw(ArgumentError(string("model node arguments are not all of type Any: ",[(fkeys,ftypes) for i in nodeinds])))

  modelargs = Array{Any}(n)
  for i in nodeinds
    modelargs[i] = Expr(:call, :getval,
                        Expr(:ref, :model, QuoteNode(fkeys[i]))) #getval(model[fkey])
  end
  for i in literalinds
    modelargs[i] = fkeys[i]
  end
  expr = Expr(:block, Expr(:(=), :f, f), Expr(:call, :f, modelargs...))

  (expr, fkeys[nodeinds])
end


#################### Mathematical Operators ####################

isprobvec(p::AbstractVector) = isprobvec(convert(Vector{Float64}, p))

cummean(x::AbstractArray) = mapslices(cummean, x, 1) #qq

function cummean{T<:Real}(x::AbstractVector{T})
  y = similar(x, Float64)
  xs = 0.0
  for i in 1:length(x)
    xs += x[i]
    y[i] = xs / i
  end
  y
end

dot(x) = dot(x, x)

invlogit(x::Real) = 1.0 / (exp(-x) + 1.0)
invlogit(x::AbstractArray) = map(invlogit, x)

logit(x::Real) = log(x / (1.0 - x))
logit(x::AbstractArray) = map(logit, x)

## Csorgo S and Faraway JJ. The exact and asymptotic distributions of the
## Cramer-von Mises statistic. Journal of the Royal Statistical Society,
## Series B, 58: 221-234, 1996.
function pcramer(q::Real)
  p = 0.0
  for k in 0:3
    c1 = 4.0 * k + 1.0
    c2 = c1^2 / (16.0 * q)
    p += gamma(k + 0.5) / factorial(k) * sqrt(c1) * exp(-c2) * besselk(0.25, c2)
  end
  p / (pi^1.5 * sqrt(q))
end


#################### Auxiliary Functions ####################

## pmap2 is a partial work-around for the pmap issue in julia 0.4.0 of worker
## node errors being blocked.  In single-processor mode, pmap2 calls map
## instead to avoid the error handling issue.  In multi-processor mode, pmap is
## called and will apply its error processing.

function pmap2(f::Function, lsts::AbstractArray)
  if (nprocs() > 1) & (VERSION < v"0.5-")
    @everywhere importall Mamba
    pmap(f, lsts)
  else
    map(f, lsts)
  end
end
