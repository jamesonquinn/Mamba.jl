using Distributions
using StatsBase
using IterTools

import Base:
    rand!, length
import Distributions: logpdf

immutable DirichletPInt <: DiscreteMultivariateDistribution
    alpha::Float64
    len::Int64
    basedist::Nullable{Distribution}

    function DirichletPInt(alpha::Float64,n::Int64)
        new(alpha,n,Nullable{Distribution}())
    end
end

function DirichletPIntMarginal(dp::DirichletPInt,v::AbstractVector{Int64},i::Int64,newi::Maybe{Int64}=nothing)
  s = 1:max(v...)
  if isnull(newi)
    lastWeight = Float64[dp.alpha]
  else
    lastWeight = Float64[]
  end
  c = [Float64(cc) for cc in chain(counts(v,s),lastWeight)]
  c[v[i]] -= 1
  tot = length(v) - 1. + dp.alpha
  if !isnull(newi)
    c[newi] = dp.alpha
  end
  c /= tot
  return Categorical(c)
end


Base.show(io::IO, d::DirichletPInt) = show(io, d, (:alpha,))

# Properties

params(d::DirichletPInt) = (d.alpha,)
#@inline partype{T<:Real}(d::Dirichlet{Int64}) = T


#
# function entropy(d::DirichletPInt)
#     α = d.alpha
#     α0 = d.alpha0
#     k = length(α)
#
#     en = d.lmnB + (α0 - k) * digamma(α0)
#     for j in 1:k
#         @inbounds αj = α[j]
#         en -= (αj - 1.0) * digamma(αj)
#     end
#     return en
# end

#
# function dirichlet_mode!{T <: Real}(r::Vector{Int64}, α::Vector{Int64}, α0::T)
#     k = length(α)
#     s = α0 - k
#     for i = 1:k
#         @inbounds αi = α[i]
#         if αi <= one(T)
#             error("Dirichlet has a mode only when alpha[i] > 1 for all i" )
#         end
#         @inbounds r[i] = (αi - one(T)) / s
#     end
#     return r
# end
#
# dirichlet_mode{T <: Real}(α::Vector{Int64}, α0::T) = dirichlet_mode!(Vector{Int64}(length(α)), α, α0)
#
# mode(d::Dirichlet) = dirichlet_mode(d.alpha, d.alpha0)
# mode(d::DirichletCanon) = dirichlet_mode(d.alpha, sum(d.alpha))
#
# modes(d::Dirichlet) = [mode(d)]


# Evaluation

function insupport(d::DirichletPInt, x::AbstractVector{Int64})
    return true
end

function insupport(d::DirichletPInt, x::AbstractVector{Float64})
    return all(v -> v % 1 == 0,x)
end

length(d::DirichletPInt) = d.len

function _logpdfcounts(d::DirichletPInt, c::Vector{Int64},l::Int64)
    a = d.alpha
    n = d.len
    g = sum(map(x->x>0, c))

    p = lgamma(a)+g*log(a)-lgamma(a+n)

    for i in 1:length(c)
        if c[i] != 0
            @inbounds p += lgamma(c[i])
        end
    end
    return p
end

function logpdf(d::DirichletPInt, x::AbstractVector{Int64})
    s = 1:max(x...)
    c = counts(x,s)
    l = length(x)
    return _logpdfcounts(d,c,l)
end

logpdf(d::DirichletPInt, x::AbstractVector{Float64}) = logpdf(d,[Int(v) for v in x])

# sampling

function rand!(d::DirichletPInt, x::AbstractVector{Int64})
    x[1] = curmax = T(1)
    n = length(x)
    α = d.alpha
    for i in 2:n
        r = Int64(floor(rand() * (i - 1 + α))) + 1
        if r > i - 1
            curmax += 1
            @inbounds x[i] = curmax
        else
            @inbounds x[i] = x[r]
        end
    end
    x
end

#######################################
#
#  Estimation
#
#######################################

immutable DirichletPIntStats <: SufficientStats
    counts::Vector{Float64}
    tw::Float64              # total sample weights

    DirichletPIntStats(counts::Vector{Float64}, tw::Real) = new(slogp, Float64(tw))
end

length(ss::DirichletPIntStats) = length(s.counts)
#
# mean_logp(ss::DirichletStats) = ss.slogp * inv(ss.tw)
#
# function suffstats(::Type{Dirichlet}, P::AbstractMatrix{Float64})
#     K = size(P, 1)
#     n = size(P, 2)
#     slogp = zeros(K)
#     for i = 1:n
#         for k = 1:K
#             @inbounds slogp[k] += log(P[k,i])
#         end
#     end
#     DirichletStats(slogp, n)
# end
#
# function suffstats(::Type{Dirichlet}, P::AbstractMatrix{Float64}, w::AbstractArray{Float64})
#     K = size(P, 1)
#     n = size(P, 2)
#     if length(w) != n
#         throw(ArgumentError("Inconsistent argument dimensions."))
#     end
#
#     tw = 0.
#     slogp = zeros(K)
#
#     for i = 1:n
#         @inbounds wi = w[i]
#         tw += wi
#         for k = 1:K
#             @inbounds slogp[k] += log(P[k,i]) * wi
#         end
#     end
#     DirichletStats(slogp, tw)
# end
#
# # fit_mle methods
#
# ## Initialization
#
# function _dirichlet_mle_init2(μ::Vector{Float64}, γ::Vector{Float64})
#     K = length(μ)
#
#     α0 = 0.
#     for k = 1:K
#         @inbounds μk = μ[k]
#         @inbounds γk = γ[k]
#         ak = (μk - γk) / (γk - μk * μk)
#         α0 += ak
#     end
#     α0 /= K
#
#     multiply!(μ, α0)
# end
#
# function dirichlet_mle_init(P::AbstractMatrix{Float64})
#     K = size(P, 1)
#     n = size(P, 2)
#
#     μ = Vector{Float64}(K)  # E[p]
#     γ = Vector{Float64}(K)  # E[p^2]
#
#     for i = 1:n
#         for k = 1:K
#             @inbounds pk = P[k, i]
#             @inbounds μ[k] += pk
#             @inbounds γ[k] += pk * pk
#         end
#     end
#
#     c = 1.0 / n
#     for k = 1:K
#         μ[k] *= c
#         γ[k] *= c
#     end
#
#     _dirichlet_mle_init2(μ, γ)
# end
#
# function dirichlet_mle_init(P::AbstractMatrix{Float64}, w::AbstractArray{Float64})
#     K = size(P, 1)
#     n = size(P, 2)
#
#     μ = Vector{Float64}(K)  # E[p]
#     γ = Vector{Float64}(K)  # E[p^2]
#     tw = 0.
#
#     for i = 1:n
#         @inbounds wi = w[i]
#         tw += wi
#         for k = 1:K
#             pk = P[k, i]
#             @inbounds μ[k] += pk * wi
#             @inbounds γ[k] += pk * pk * wi
#         end
#     end
#
#     c = 1.0 / tw
#     for k = 1:K
#         μ[k] *= c
#         γ[k] *= c
#     end
#
#     _dirichlet_mle_init2(μ, γ)
# end
#
# ## Newton-Ralphson algorithm
#
# function fit_dirichlet!(elogp::Vector{Float64}, α::Vector{Float64};
#     maxiter::Int=25, tol::Float64=1.0e-12, debug::Bool=false)
#     # This function directly overrides α
#
#     K = length(elogp)
#     length(α) == K || throw(ArgumentError("Inconsistent argument dimensions."))
#
#     g = Vector{Float64}(K)
#     iq = Vector{Float64}(K)
#     α0 = sum(α)
#
#     if debug
#         objv = dot(α - 1.0, elogp) + lgamma(α0) - sum(lgamma(α))
#     end
#
#     t = 0
#     converged = false
#     while !converged && t < maxiter
#         t += 1
#
#         # compute gradient & Hessian
#         # (b is computed as well)
#
#         digam_α0 = digamma(α0)
#         iz = 1.0 / trigamma(α0)
#         gnorm = 0.
#         b = 0.
#         iqs = 0.
#
#         for k = 1:K
#             @inbounds ak = α[k]
#             @inbounds g[k] = gk = digam_α0 - digamma(ak) + elogp[k]
#             @inbounds iq[k] = - 1.0 / trigamma(ak)
#
#             @inbounds b += gk * iq[k]
#             @inbounds iqs += iq[k]
#
#             agk = abs(gk)
#             if agk > gnorm
#                 gnorm = agk
#             end
#         end
#         b /= (iz + iqs)
#
#         # update α
#
#         for k = 1:K
#             @inbounds α[k] -= (g[k] - b) * iq[k]
#             @inbounds if α[k] < 1.0e-12
#                 α[k] = 1.0e-12
#             end
#         end
#         α0 = sum(α)
#
#         if debug
#             prev_objv = objv
#             objv = dot(α - 1.0, elogp) + lgamma(α0) - sum(lgamma(α))
#             @printf("Iter %4d: objv = %.4e  ch = %.3e  gnorm = %.3e\n",
#                 t, objv, objv - prev_objv, gnorm)
#         end
#
#         # determine convergence
#
#         converged = gnorm < tol
#     end
#
#     Dirichlet(α)
# end
#
#
# function fit_mle(::Type{Dirichlet}, P::AbstractMatrix{Float64};
#     init::Vector{Float64}=Float64[], maxiter::Int=25, tol::Float64=1.0e-12)
#
#     α = isempty(init) ? dirichlet_mle_init(P) : init
#     elogp = mean_logp(suffstats(Dirichlet, P))
#     fit_dirichlet!(elogp, α; maxiter=maxiter, tol=tol)
# end
#
# function fit_mle(::Type{Dirichlet}, P::AbstractMatrix{Float64}, w::AbstractArray{Float64};
#     init::Vector{Float64}=Float64[], maxiter::Int=25, tol::Float64=1.0e-12)
#
#     n = size(P, 2)
#     length(w) == n || throw(ArgumentError("Inconsistent argument dimensions."))
#
#     α = isempty(init) ? dirichlet_mle_init(P, w) : init
#     elogp = mean_logp(suffstats(Dirichlet, P, w))
#     fit_dirichlet!(elogp, α; maxiter=maxiter, tol=tol)
# end
