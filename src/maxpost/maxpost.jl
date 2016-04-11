
using Optim

" A factory which makes a function which takes values from a vector and
    puts them into a structure of the same shape as fillable"
function makeFiller(fillable, whichParts)
  partFillers = Function[]
  i = 1
  for part in whichParts
    function enclose(myI, myPart)
      function partFill!(target,v::Vector{Float64})
        target[myPart] = v[myI]
      end
      return(partFill!)
    end
    push!(partFillers,enclose(i,part))
    i += 1
  end
  function fill!(target, v::Vector{Float64})
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
  function logpdfFor(v::Vector{Float64})
  ## This currently has side-effects. That's bad style but I'll fix it later
    fill!(current,v)
    return -logpdf(m)
  end
  optimize(logpdfFor, Float64[init[param] for param in params], xtol = 0.05, grtol = 1e-9, ftol=1e-9)
end
