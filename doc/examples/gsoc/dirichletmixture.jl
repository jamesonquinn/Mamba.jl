using Mamba


## Data
pumps = Dict{Symbol, Any}(
  :y => [5, 1, 5, 14, 3, 19, 1, 1, 4, 22],
  :t => [94.3, 15.7, 62.9, 126, 5.24, 31.4, 1.05, 1.05, 2.1, 10.5]
)
pumps[:N] = length(pumps[:y])


import Base:
    valtype

type QQQQ{B<:Real} end

valtype{VT}(a::QQQQ{VT}) = VT 

a = QQQQ{Int}()
b = QQQQ{Float64}()
valtype(a)
valtype(b)
