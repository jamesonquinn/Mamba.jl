using Distributions

module Mamba

  #################### Imports ####################

  import Base: cor, dot, valtype, getindex, get, length, keys, setindex!,
         start, next, done, ndims, convert, promote_rule,
         +, -, *, /, ^
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

  ElementOrVector{T} = Union{T, Vector{T}}


  #################### Variate Types ####################

  const ScalarVariateType = Real
  const ArrayVariateVals{SVT,N} = Array{SVT, N} #where SVT <: Real #but this constraint cannot be expressed in Julia
  abstract type DictVariateVals{SVT,K} <: Associative{K, SVT} end #where SVT <: Real #but this constraint cannot be expressed in Julia


  const AbstractVariateVals{SVT} = Union{SVT, (ArrayVariateVals{SVT,XX} where XX), (DictVariateVals{SVT,XX} where XX)} where SVT<:ScalarVariateType
  const VectorVariateVals{SVT} = ArrayVariateVals{SVT,1} where SVT<:ScalarVariateType
  const MatrixVariateVals{SVT} = ArrayVariateVals{SVT,2} where SVT<:ScalarVariateType

  abstract type Variate{VS<:AbstractVariateVals}  end

  const ScalarVariate = Variate{SVT} where SVT<:ScalarVariateType

  abstract type ArrayVariate{N,SVT,V} <: Variate{V} end
  const ValidArrayVariate = ArrayVariate{N,SVT,ArrayVariateVals{SVT,N}} where SVT<:ScalarVariateType where N



  const ValidVectorVariate = ArrayVariate{1,SVT,VectorVariateVals{SVT}} where SVT<:ScalarVariateType


  const ValidMatrixVariate = ArrayVariate{2,SVT,MatrixVariateVals{SVT}} where SVT<:ScalarVariateType

  abstract type DictVariate{K,SVT,V} <: Variate{V} end
  const ValidDictVariate = DictVariate{N,SVT,Variate{VS}} where VS<:DictVariateVals{SVT,N} where SVT<:ScalarVariateType where N



  #################### Distribution Types ####################

  const DistributionStruct = Union{Distribution,
                                   Array{UnivariateDistribution},
                                   Array{MultivariateDistribution},
                                   Associative{Any,MultivariateDistribution}}


  #################### Concrete DictVariateVals Types ####################
  abstract type NestedDictVariateVals{SVT<:ScalarVariateType} <: DictVariateVals{SVT,Tuple} end

  const LeafOrBranch{SVT} = Union{SVT,NestedDictVariateVals{SVT}} where SVT<:ScalarVariateType
  type SymDictVariateVals{SVT} <: NestedDictVariateVals{SVT}
      vals::Dict{Symbol,LeafOrBranch{SVT}}
  end

  type VecDictVariateVals{SVT} <: NestedDictVariateVals{SVT}
      vals::Vector{LeafOrBranch{SVT}}
      qqqq::Bool #TODO: remove

      function VecDictVariateVals{SVT}() where SVT<:ScalarVariateType
        new{SVT}(Vector{LeafOrBranch{SVT}}(),true)
      end

      function VecDictVariateVals{SVT}(x
            ::Union{Vector{SVT},Vector{Union{SVT,NestedDictVariateVals{SVT}}}}) where SVT<:ScalarVariateType
        vdv = VecDictVariateVals{SVT}()
        for i in 1:length(x)
          vdv[i] = x[i]
        end
        vdv
      end
  end

  #################### Dependent Types ####################

  type ScalarLogical{V} <: Variate{V}
    value::V
    symbol::Symbol
    monitor::Vector{Int}
    eval::Function
    sources::Vector{Symbol}
    targets::Vector{Symbol}
  end

  type ArrayLogical{N,SVT,V} <: ArrayVariate{N,SVT,V}
    value::V
    symbol::Symbol
    monitor::Vector{Int}
    eval::Function
    sources::Vector{Symbol}
    targets::Vector{Symbol}
  end

  type ScalarStochastic{V} <: Variate{V}
    value::V
    symbol::Symbol
    monitor::Vector{Int}
    eval::Function
    sources::Vector{Symbol}
    targets::Vector{Symbol}
    distr::UnivariateDistribution
  end

  type ArrayStochastic{N,SVT,V} <: ArrayVariate{N,SVT,V}
    value::V
    symbol::Symbol
    monitor::Vector{Int}
    eval::Function
    sources::Vector{Symbol}
    targets::Vector{Symbol}
    distr::DistributionStruct
    function ArrayStochastic{N,SVT,V}(value::ArrayVariateVals, symbol::Symbol, monitor, eval, sources, targets, distr) where {N,SVT,V}
      new{ndims(value),myvaltype(value),typeof(value)}(value, symbol, monitor, eval, sources, targets, distr)
    end
  end

  type DictStochastic{K,SVT,V} <: DictVariate{K,SVT,V}
    value::V
    symbol::Symbol
    monitor::Vector{Int}
    eval::Function
    sources::Vector{Symbol}
    targets::Vector{Symbol}
    distr::DistributionStruct
  end




const AbstractLogical{SVT} = Union{ScalarLogical{SVT}, (ArrayLogical{N,SVT} where N)}
const AbstractStochastic{SVT} = Union{ScalarStochastic{SVT}, (ArrayStochastic{N,SVT} where N), (DictStochastic{K,SVT} where K)}
const AbstractDependent = Union{AbstractLogical, AbstractStochastic}

const AbstractFixedDependent = Union{ScalarLogical, ArrayLogical, ScalarStochastic, ArrayStochastic}
const AbstractElasticDependent = Union{DictStochastic}
const AbstractElasticDependent = Union{DictStochastic} #TODO: DRY; use set difference.
const AbstractFixedStochastic = Union{ScalarStochastic, ArrayStochastic}


  #################### Sampler Types ####################

  type Sampler{T}
    params::Vector{Symbol}
    eval::Function
    tune::T
    targets::Vector{Symbol}
  end

  abstract type SamplerTune end

  abstract type AbstractSamplingBlock end

  abstract type (AbstractSamplerVariate{VS<:AbstractVariateVals,T<:SamplerTune}
          <: Variate{VS}) end

  type SamplerVariate{VS,T<:SamplerTune} <: AbstractSamplerVariate{VS,T}
    value::VS
    tune::T

    function SamplerVariate{VS,T}(x::VS, tune::T) where VS<:AbstractVariateVals where T<:SamplerTune
      v = new{VS,T}(x, tune)
      validate(v)
    end

    function SamplerVariate{VS,T}(x::VS, pargs...; doconvert::String="true", kargs...) where VS<:ArrayVariateVals where T<:SamplerTune
      value = convert(Vector{Float64}, x)
      SamplerVariate(value,pargs...; doconvert=false, kargs...)
    end


    function SamplerVariate{VS,T}(value::VS, pargs...; doconvert::Bool=false, kargs...) where VS<:AbstractVariateVals where T<:SamplerTune
      t = T(value, pargs...; kargs...)
      SamplerVariate{VS,T}(value, t)
    end
  end

  const FlatSamplerVariate{T} = SamplerVariate{VectorVariateVals{Float64},T}

  const DictSamplerVariate{T} = SamplerVariate{DictVariateVals{Float64,Tuple},T}


  #################### Model Types ####################

  type ModelGraph
    graph::DiGraph
    keys::Vector{Symbol}
  end

  type AbstractModelState{StateType} #where StateType <: Variate
    value::StateType
    tune::Vector{Any}
  end

  type AbstractModel{StateType} #where StateType <: Variate
    nodes::Dict{Symbol, Any}
    samplers::Vector{Sampler}
    states::Vector{AbstractModelState{StateType}}
    iter::Int
    burnin::Int
    hasinputs::Bool
    hasinits::Bool
  end

  const Model{SVT} = AbstractModel{ArrayVariateVals{SVT,1}}

  const ModelState = AbstractModelState{ArrayVariateVals{Float64,1}}

  const ElasticModel{SVT} = AbstractModel{DictVariateVals{SVT,Tuple}}


  #################### Chains Type ####################

  abstract type AbstractChains end

  immutable Chains <: AbstractChains
    value::Array{Float64, 3}
    range::Range{Int}
    names::Vector{AbstractString}
    chains::Vector{Int}
  end

  immutable ModelChains <: AbstractChains
    value::Array{Float64, 3}
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

  #include("maxpost/maxpost.jl")


  #################### Exports ####################

  export
    AbstractChains,
    AbstractDependent,
    AbstractLogical,
    AbstractStochastic,
    Variate,
    ArrayLogical,
    ArrayStochastic,
    ArrayVariate,
    Chains,
    Logical,
    ValidMatrixVariate,
    Model, AbstractModel, ElasticModel,
    ModelChains,
    Sampler,
    SamplerTune,
    SamplerVariate,
    ScalarLogical,
    ScalarStochastic,
    ScalarVariate,
    Stochastic, Stochelastic,
    ValidVectorVariate

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

  export
    get


  #################### Deprecated ####################

  include("deprecated.jl")

end
