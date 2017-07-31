


abstract type NestedDictVariateVals{SVT<:Real} <: DictVariateVals{Tuple,SVT} end

type VecDictVariateVals{SVT} <: NestedDictVariateVals{SVT} end

const AbstractVariateVal{SVT<:Real} = Union{SVT, (DictVariateVals{XX,SVT} where XX)}

type Variate{K<:AbstractVariateVal} end


function myvaltype(x::Type{X}) where X<:DictVariateVals{N,SVT} where N where SVT <: Real
  SVT
end

function myvaltype(v::Type{Variate{K}}) where K
  myvaltype(K)
end

x = Variate{VecDictVariateVals{X}} where X <: Real

myvaltype(x)
