using Distributions

module Mamba

  #################### Imports ####################

  import Base: cor, dot,
         valtype, getindex, get, length, keys, setindex!, copy,
         start, next, done, ndims, convert, promote_rule, size, fill!,
         cat, vcat, hcat,
         display,
         +, -, *, /, ^, |
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
         minimum, pdf, quantile, rand, sample!, support, isprobvec
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
  abstract type DictVariateVals{SVT} <: Associative{Tuple, SVT} end #where SVT <: Real #but this constraint cannot be expressed in Julia

  type NullDictVariateVals{SVT} <: DictVariateVals{SVT} end


  const AbstractVariateVals{SVT} = Union{SVT, (ArrayVariateVals{SVT,XX} where XX), DictVariateVals{SVT}} where SVT<:ScalarVariateType
  const VectorVariateVals{SVT} = ArrayVariateVals{SVT,1} where SVT<:ScalarVariateType
  const MatrixVariateVals{SVT} = ArrayVariateVals{SVT,2} where SVT<:ScalarVariateType

  abstract type Variate{VS<:AbstractVariateVals}  end

  const ScalarVariate = Variate{SVT} where SVT<:ScalarVariateType

  abstract type ArrayVariate{N,SVT,V} <: Variate{V} end
  const ValidArrayVariate = ArrayVariate{N,SVT,ArrayVariateVals{SVT,N}} where SVT<:ScalarVariateType where N



  const ValidVectorVariate = ArrayVariate{1,SVT,VectorVariateVals{SVT}} where SVT<:ScalarVariateType


  const ValidMatrixVariate = ArrayVariate{2,SVT,MatrixVariateVals{SVT}} where SVT<:ScalarVariateType

  abstract type DictVariate{SVT,V} <: Variate{V} end
  const ValidDictVariate = DictVariate{SVT,Variate{VS}} where VS<:DictVariateVals{SVT} where SVT<:ScalarVariateType



  #################### Distribution Types ####################

  const DistributionStruct = Union{Distribution,
                                   Array{UnivariateDistribution},
                                   Array{MultivariateDistribution},
                                   Associative{Any,MultivariateDistribution}}


  #################### Concrete DictVariateVals Types ####################
  abstract type NestedDictVariateVals{SVT<:ScalarVariateType} <: DictVariateVals{SVT} end

  const LeafOrBranch{SVT} = Union{SVT,NestedDictVariateVals{SVT}} where SVT<:ScalarVariateType
  type SymDictVariateVals{SVT} <: NestedDictVariateVals{SVT}
      value::Dict{Symbol,LeafOrBranch{SVT}}


      function SymDictVariateVals{SVT}() where SVT<:ScalarVariateType
        new{SVT}(Dict{Symbol,LeafOrBranch{SVT}}())
      end

      function SymDictVariateVals{SVT}(kv) where SVT<:ScalarVariateType
        new{SVT}(Dict{Symbol,LeafOrBranch{SVT}}(kv))
      end
  end

  type VecDictVariateVals{SVT} <: NestedDictVariateVals{SVT}
      value::Vector{LeafOrBranch{SVT}}
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

  type DictStochastic{SVT,V} <: DictVariate{SVT,V}
    value::V
    symbol::Symbol
    monitor::Vector{Int}
    eval::Function
    sources::Vector{Symbol}
    targets::Vector{Symbol}
    distr::DistributionStruct
  end




const AbstractLogical{SVT} = Union{ScalarLogical{SVT}, (ArrayLogical{N,SVT} where N)}
const AbstractStochastic{SVT} = Union{ScalarStochastic{SVT}, (ArrayStochastic{N,SVT} where N), DictStochastic{SVT}}
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

  const DictSamplerVariate{T} = SamplerVariate{DictVariateVals{Float64},T}


  #################### Model Types ####################

  type ModelGraph
    graph::DiGraph
    keys::Vector{Symbol}
  end

  type AbstractModelState{StateType} #where StateType <: Variate
    value::StateType
    tune::Vector{Any}

    #function AbstractModelState{K}(val::K,tune::Vector{Any}) where K<:DictVariateVals{SVT} where SVT

  end

    function MakeAbstractModelState(val::VS,tune::Vector{Any}) where VS<:DictVariateVals{SVT} where SVT #TODO: Why  AbstractVariateVals and not Variate?
      AbstractModelState{DictVariateVals{SVT}}(val,tune)
      #new{DictVariateVals{SVT}}(val,tune)
      #Necessary due to the following ridiculous error:
      #MethodError: Cannot `convert` an object of type Mamba.AbstractModelState{Mamba.SymDictVariateVals{Float64}} to an object of type Mamba.AbstractModelState{Mamba.DictVariateVals{Float64}}
    end
    function MakeAbstractModelState(val::VS,tune::Vector{Any}) where VS<:ArrayVariateVals{SVT} where SVT #TODO: Why  AbstractVariateVals and not Variate?
      AbstractModelState{ArrayVariateVals{SVT,1}}(val,tune)
      #new{DictVariateVals{SVT}}(val,tune)
      #Necessary due to the following ridiculous error:
      #MethodError: Cannot `convert` an object of type Mamba.AbstractModelState{Mamba.SymDictVariateVals{Float64}} to an object of type Mamba.AbstractModelState{Mamba.DictVariateVals{Float64}}
    end

  type AbstractModel{StateType} #where StateType <: Variate
    nodes::Dict{Symbol, Any}
    samplers::Vector{Sampler}
    states::Vector{T} where T<:AbstractModelState{StateType}
    iter::Int
    burnin::Int
    hasinputs::Bool
    hasinits::Bool

    function AbstractModel{StateType}(nodes::Dict{Symbol, Any},
            iter::Integer, burnin::Integer) where StateType
      new{StateType}(nodes, Sampler[], AbstractModelState{StateType}[],
            iter, burnin, false, false)
    end
  end

  const Model{SVT} = AbstractModel{ArrayVariateVals{SVT,1}}

  const ModelState = AbstractModelState{ArrayVariateVals{Float64,1}}

  const ElasticModel{SVT} = AbstractModel{DictVariateVals{SVT}}


  #################### Chains Type ####################

  immutable AbstractDictChainVal{SVT,N}
    value::Array{DictVariateVals{SVT},N}
    #uselessExtraComponent::Bool
  end

  const DictChainVal{SVT} = AbstractDictChainVal{SVT,2}

  function MakeDictChainVal(svt::Type{SVT},ds::Int...) where SVT<:ScalarVariateType
    AbstractDictChainVal{SVT,2}(Array{DictVariateVals{SVT},2}(ds[1:end-1]...))
  end

  const VecDictChainVal{SVT} = AbstractDictChainVal{SVT,1}

  const AbstractChainVal{SVT} = Union{Array{SVT, 3}, DictChainVal{SVT}}

  abstract type AbstractChains{V} end #where V<:ChainVal

  immutable Chains{V} <: AbstractChains{V} #where V<:ChainVal
    value::V
    range::Range{Int}
    names::Vector{AbstractString}
    chains::Vector{Int}
  end

  immutable ModelChains{V} <: AbstractChains{V} #where V<:ChainVal
    value::V
    range::Range{Int}
    names::Vector{AbstractString}
    chains::Vector{Int}
    model::AbstractModel
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
  include("distributions/dirichletprocess.jl")

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

  # include("samplers/abc.jl")
  # include("samplers/amm.jl")
  # include("samplers/amwg.jl")
  # include("samplers/bhmc.jl")
  # include("samplers/bia.jl")
  # include("samplers/bmc3.jl")
  include("samplers/bmg.jl")
  include("samplers/dgs.jl")
  # include("samplers/hmc.jl")
  # include("samplers/mala.jl")
  # include("samplers/miss.jl")
  # include("samplers/nuts.jl")
  # include("samplers/rwm.jl")
  include("samplers/slice.jl")
  include("samplers/slicesimplex.jl")
  include("samplers/GSOC/rjs.jl")

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
    SymUniform,
    DirichletPInt

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
    RJS, RJSVariate,
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
