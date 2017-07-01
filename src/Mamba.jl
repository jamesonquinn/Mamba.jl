using Distributions

module Mamba

  #################### Imports ####################

  import Base: cor, dot
  import Base.LinAlg: Cholesky
  import Calculus: gradient
  import Compose: Context, context, cm, gridstack, inch, MeasureOrNumber, mm,
         pt, px
  import Distributions:
         ## Generic Types
         Continuous, ContinuousUnivariateDistribution, Distribution,
         MatrixDistribution, Multivariate, MultivariateDistribution, PDiagMat,
         PDMat, ScalMat, Truncated, Univariate, UnivariateDistribution,
         ValueSupport,
         ## ContinuousUnivariateDistribution Types
         Arcsine, Beta, BetaPrime, Biweight, Cauchy, Chi, Chisq, Cosine,
         Epanechnikov, Erlang, Exponential, FDist, Frechet, Gamma, Gumbel,
         InverseGamma, InverseGaussian, Kolmogorov, KSDist, KSOneSided, Laplace,
         Levy, Logistic, LogNormal, NoncentralBeta, NoncentralChisq,
         NoncentralF, NoncentralT, Normal, NormalCanon, Pareto, Rayleigh,
         SymTriangularDist, TDist, TriangularDist, Triweight, Uniform, VonMises,
         Weibull,
         ## DiscreteUnivariateDistribution Types
         Bernoulli, Binomial, Categorical, DiscreteUniform, Geometric,
         Hypergeometric, NegativeBinomial, NoncentralHypergeometric, Pareto,
         PoissonBinomial, Skellam,
         ## MultivariateDistribution Types
         Dirichlet, Multinomial, MvNormal, MvNormalCanon, MvTDist,
         VonMisesFisher,
         ## MatrixDistribution Types
         InverseWishart, Wishart,
         ## Methods
         dim, gradlogpdf, insupport, isprobvec, logpdf, logpdf!, maximum,
         minimum, pdf, quantile, rand, sample!, support
  import Gadfly: draw, Geom, Guide, Layer, layer, PDF, PGF, Plot, plot, PNG, PS,
         render, Scale, SVG, Theme
  import LightGraphs: DiGraph, add_edge!, out_neighbors,
         topological_sort_by_dfs, vertices
  import Showoff: showoff
  import StatsBase: autocor, autocov, countmap, counts, describe, predict,
         quantile, sample, sem, summarystats

  include("distributions/pdmats2.jl")
  importall .PDMats2


  #################### Types ####################

  typealias ElementOrVector{T} Union{T, Vector{T}}


  #################### Variate Types ####################

  abstract ScalarVariate <: Real
  abstract ArrayVariate{N} <: DenseArray{Float64, N}

  typealias AbstractVariate Union{ScalarVariate, ArrayVariate}
  typealias VectorVariate ArrayVariate{1}
  typealias MatrixVariate ArrayVariate{2}


  #################### Distribution Types ####################

  typealias DistributionStruct Union{Distribution,
                                     Array{UnivariateDistribution},
                                     Array{MultivariateDistribution}}


  #################### Dependent Types ####################

  type ScalarLogical <: ScalarVariate
    value::Float64
    symbol::Symbol
    monitor::Vector{Int}
    eval::Function
    sources::Vector{Symbol}
    targets::Vector{Symbol}
  end

  type ArrayLogical{N} <: ArrayVariate{N}
    value::Array{Float64, N}
    symbol::Symbol
    monitor::Vector{Int}
    eval::Function
    sources::Vector{Symbol}
    targets::Vector{Symbol}
  end

  type ScalarStochastic <: ScalarVariate
    value::Float64
    symbol::Symbol
    monitor::Vector{Int}
    eval::Function
    sources::Vector{Symbol}
    targets::Vector{Symbol}
    distr::UnivariateDistribution
  end

  type ArrayStochastic{N} <: ArrayVariate{N}
    value::Array{Float64, N}
    symbol::Symbol
    monitor::Vector{Int}
    eval::Function
    sources::Vector{Symbol}
    targets::Vector{Symbol}
    distr::DistributionStruct
  end

  typealias AbstractLogical Union{ScalarLogical, ArrayLogical}
  typealias AbstractStochastic Union{ScalarStochastic, ArrayStochastic}
  typealias AbstractDependent Union{AbstractLogical, AbstractStochastic}

typealias FlatStateVal{U<:Real} Vector{U}
typealias UnflatStateVal{U<:Real} Dict{Union{Symbol,Tuple{Symbol,Int}},Vector{U}}
typealias AbstractStateVal{U} Union{FlatStateVal{U}, UnflatStateVal{U}}

type StateValVec{S<:AbstractStateVal}
    vals::Vector{S}
end
type StateValMatrix{S<:AbstractStateVal}
    vals::Matrix{S}
end





typealias FlatStateValVec{S<:FlatStateVal} Array{Float64,2}
typealias FlatStateValMatrix{S<:FlatStateVal} Array{Float64,3}







typealias AbstractStateValVec{S<:AbstractStateVal} Union{StateValVec{S}, FlatStateValVec{S}}
typealias AbstractStateValMatrix{S<:AbstractStateVal} Union{StateValMatrix{S}, FlatStateValMatrix{S}}


  #################### Sampler Types ####################

  type Sampler{T}
    params::Vector{Symbol}
    eval::Function
    tune::T
    targets::Vector{Symbol}
  end


  abstract SamplerTune

  type SamplerVariate{T<:SamplerTune,S<:AbstractStateVal} <: VectorVariate
    value::S
    tune::T

    function SamplerVariate{U<:Real}(x::FlatStateVal{U}, tune::T)
      v = new(x, tune)
      validate(v)
    end

    function SamplerVariate{U<:Real}(x::FlatStateVal{U}, pargs...; kargs...)
      value = convert(Vector{Float64}, x)
      SamplerVariate{T}(value, T(value, pargs...; kargs...))
    end

    function SamplerVariate(x::AbstractStateVal, pargs...; kargs...)
      SamplerVariate{T}(x, T(x, pargs...; kargs...))
    end
  end


  #################### Model Types ####################

  type ModelGraph
    graph::DiGraph
    keys::Vector{Symbol}
  end

  type ModelState{T<:AbstractStateVal}
    value::T
    tune::Vector{Any}
  end

  type ModelF{S<:FlatStateVal}
    nodes::Dict{Symbol, Any}
    samplers::Vector{Sampler}
    states::FlatStateValVec{S}
    iter::Int
    burnin::Int
    hasinputs::Bool
    hasinits::Bool
  end

  type ModelNF{S<:FlatStateVal}
    nodes::Dict{Symbol, Any}
    samplers::Vector{Sampler}
    states::StateValVec{S}
    iter::Int
    burnin::Int
    hasinputs::Bool
    hasinits::Bool
  end

  typealias Model{S<:AbstractStateVal} Union{ModelF{S}, ModelNF{S}}

  #################### Chains Type ####################

  abstract AbstractChains{S<:AbstractStateVal}

  immutable Chains{S<:AbstractStateVal} <: AbstractChains{S}
    value::AbstractStateValMatrix{S}
    range::Range{Int}
    names::Vector{AbstractString}
    chains::Vector{Int}
  end

  immutable ModelChains{S<:AbstractStateVal} <: AbstractChains{S}
    value::AbstractStateValMatrix{S}
    range::Range{Int}
    names::Vector{AbstractString}
    chains::Vector{Int}
    model::Model
  end


  #################### Includes ####################

  include("progress.jl")
  include("utils.jl")
  include("variate.jl")

  include("distributions/constructors.jl")
  include("distributions/distributionstruct.jl")
  include("distributions/extensions.jl")
  include("distributions/pdmatdistribution.jl")
  include("distributions/transformdistribution.jl")

  include("model/dependent.jl")
  include("model/graph.jl")
  include("model/initialization.jl")
  include("model/mcmc.jl")
  include("model/model.jl")
  include("model/simulation.jl")

  include("output/chains.jl")
  include("output/chainsummary.jl")
  include("output/fileio.jl")
  include("output/gelmandiag.jl")
  include("output/gewekediag.jl")
  include("output/heideldiag.jl")
  include("output/mcse.jl")
  include("output/modelchains.jl")
  include("output/modelstats.jl")
  include("output/rafterydiag.jl")
  include("output/stats.jl")
  include("output/plot.jl")

  include("samplers/sampler.jl")

  include("samplers/abc.jl")
  include("samplers/amm.jl")
  include("samplers/amwg.jl")
  include("samplers/bhmc.jl")
  include("samplers/bia.jl")
  include("samplers/bmc3.jl")
  include("samplers/bmg.jl")
  include("samplers/dgs.jl")
  include("samplers/hmc.jl")
  include("samplers/mala.jl")
  include("samplers/miss.jl")
  include("samplers/nuts.jl")
  include("samplers/rwm.jl")
  include("samplers/slice.jl")
  include("samplers/slicesimplex.jl")


  #################### Exports ####################

  export
    AbstractChains,
    AbstractDependent,
    AbstractLogical,
    AbstractStochastic,
    AbstractVariate,
    ArrayLogical,
    ArrayStochastic,
    ArrayVariate,
    Chains,
    Logical,
    MatrixVariate,
    Model,
    ModelChains,
    Sampler,
    SamplerTune,
    SamplerVariate,
    ScalarLogical,
    ScalarStochastic,
    ScalarVariate,
    Stochastic,
    VectorVariate

  export
    BDiagNormal,
    Flat,
    SymUniform

  export
    autocor,
    changerate,
    cor,
    describe,
    dic,
    draw,
    gelmandiag,
    gettune,
    gewekediag,
    gradlogpdf,
    gradlogpdf!,
    graph,
    graph2dot,
    heideldiag,
    hpd,
    insupport,
    invlink,
    invlogit,
    link,
    logit,
    logpdf,
    logpdf!,
    mcmc,
    mcse,
    plot,
    predict,
    quantile,
    rafterydiag,
    rand,
    readcoda,
    relist,
    relist!,
    sample!,
    setinits!,
    setinputs!,
    setmonitor!,
    setsamplers!,
    summarystats,
    unlist,
    update!

  export
    ABC,
    AMM, AMMVariate,
    AMWG, AMWGVariate,
    BHMC, BHMCVariate,
    BMC3, BMC3Variate,
    BMG, BMGVariate,
    DiscreteVariate,
    DGS, DGSVariate,
    HMC, HMCVariate,
    BIA, BIAVariate,
    MALA, MALAVariate,
    MISS,
    NUTS, NUTSVariate,
    RWM, RWMVariate,
    Slice, SliceMultivariate, SliceUnivariate,
    SliceSimplex, SliceSimplexVariate

  export
    cm,
    inch,
    mm,
    pt,
    px


  #################### Deprecated ####################

  include("deprecated.jl")

end
