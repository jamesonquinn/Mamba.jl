# News

## Version Updates

### 0.11.0
* Initial release for julia 0.6.
* Branched off of *Mamba* 0.10.1.

### 0.10.1
* Replace Graphs package dependency with LightGraphs..

### 0.10.0
* Initial release for julia 0.5.
* Branched off of *Mamba* 0.9.2.

### 0.9.2
* Removed 0.9.1 deprecates.
* Added logic to ABC sampler to skip distributional evaluations if candidate draws are not in the prior distribution support.
* Updates for Distributions 0.10.0 package compatibility to fix MvNormal TypeError.
* Added vector indexing to BMC3 and BMG samplers.
* Implemented Binary Individual Adaptation (BIA) sampler.

### 0.9.1
* Removed 0.8.1 and 0.9.0 deprecates.
* Updates for Distributions 0.9.0 package compatibility to fix ambiguous new definition warnings and StackOverflowError.
* Updated documentation links to new readthedocs.io subdomains.

### 0.9.0
* Extended ABC sampler to allow specification of a decay rate for monotonically decreasing tolerances (``epsilon``) and perturbations of tolerance by random exponential variates.
* Added GK distribution example for ABC sampling.
* Revised stand-alone sampler interfaces (see documentation for full details).
    * Deprecated sampling functions ``amm!()``, ``amwg!()``, ``bhmc!()``, ``bmc3!()``, ``bmg!()``, ``dgs!()``, ``hmc!()``, ``mala!()``, ``nuts!()``, ``rwm!()``, ``slice!()``, and ``slicesimplex!()``; and replaced them with ``sample!()``.
    * Moved specifications of tuning parameters and target densities from sampling functions to ``SamplerVariate`` constructors.
    * Added target density fields to tuning parameter types.
    * Changed the following tuning parameter types/fields.
        * ``AMMTune.beta::Real`` -> ``Float64``
        * ``AMMTune.scale::Real`` -> ``Float64`` and redefined to be on the parameter scale instead of the covariance scale.
        * ``AMWGTune.target::Real`` -> ``Float64``
        * ``AMMTune.SigmaF::Cholesky{Float64}`` -> ``SigmaL::LowerTriangular{Float64}``
        * ``DGSTune`` -> ``DSTune``
        * ``HMCTune.SigmaF::Cholesky{Float64}`` -> ``SigmaL::Union{UniformScaling{Int}, LowerTriangular{Float64}}``
        * ``MALATune.scale::Float64`` -> ``epsilon::Float64``
        * ``MALATune.SigmaF::Cholesky{Float64}`` -> ``SigmaL::Union{UniformScaling{Int}, LowerTriangular{Float64}}``
        * ``RWMTune.scale::Union{Real, Vector}`` -> ``Union{Float64, Vector{Float64}}``
        * ``SliceTune.width::Union{Real, Vector}`` -> ``Union{Float64, Vector{Float64}}``
* Renamed ``Model`` method ``simulate!()`` to ``sample!()``.
* Removed 0.7 deprecates.

### 0.8.2
* Modified ``dgs!()`` to require parameter support in Matrix columns instead of rows to improve performance.

### 0.8.1
* Added ``AbstractChains`` read and write methods.
* Fixed restarting of multiple chains in the case of samplers with adaptively tuned parameters.
* Renamed ``Model`` method ``tune()`` to ``gettune()``.
* Simplified user-defined distributions examples.
* Optimized performance of DGS, MISS, and SliceSimplex samplers and simulation engine.

### 0.8.0
* Simplified sampler interfaces.
* Implemented an approximate Bayesian computation (ABC) sampler.
* Implemented a random walk Metropolis (RWM) sampler.

### 0.7.4
* Simplified implementations of sampling function types and constructors.
* Implemented ``SamplerTune`` and ``SamplerVariate`` types.
* Extended contour plots from ``ModelChains`` to ``AbstractChains``.

### 0.7.3
* Parallelize ``logpdf()`` method for ``ModelChains``.
* Update BMG algorithm and interface.
* Simplify BMC3 interface.
* Rename sampler BMMG to BMC3.
* Fix errant reinitialization of samplers when restarting chains with ``mcmc()``.

### 0.7.2
* Deprecated model expression syntax, in favor of model function syntax, for the construction of ``Logical`` and ``Stochastic`` nodes and ``Sampler`` objects.
* Implemented a Hamiltonian Monte Carlo (HMC) sampler.
* Removed ``chain`` field from ``Model`` type.

### 0.7.1
* Implemented Binary Hamiltonian Monte Carlo (BHMC) and Binary Metropolised Gibbs (BMG) samplers for binary model parameters.
* Implemented pairwise posterior density contour plots.
* Implemented the Metropolis-Adjusted Langevin Algorithm (MALA) sampler.
* Added ``first()``, ``step()``, and ``last()`` methods for getting ``AbstractChains`` iteration information.
* Implemented indexing of ``ModelChains`` by model node symbols.
* Added ``logpdf()`` method for ``ModelChains``.
* Relaxed requirement that all sampled nodes be monitored for the calculation of DIC and for the simulation of draws from posterior predictive distributions.
* Removed ``dependents`` field from ``Model`` type.

### 0.7.0
* Implemented function syntax for specification of nodes and user-defined ``Sampler`` constructors.
* Changed DSG sampler support field and arguments from Vector to Matrix.
* Added PGF graphics format to ``Chains`` draw function.
* Removed ``AbstractDependent`` linklength field.

### 0.6.3
* Added support for model specification of stochastic nodes with ``Array{MultivariateDistribution}`` structures containing distributions of different lengths.
* Added a Slice Simplex (SliceSimplex) sampler for parameters, like probability vectors, defined on simplexes.
* Added ``AbstractDependent`` ``logpdf()`` methods for evaluating log-densities at specified values.
* Renamed ``AbstractDependent`` methods ``link()/invlink()`` to ``unlist()/relist()``.
* Implemented an ``AbstractStochastic`` ``rand()`` method for random sampling of node values.
* Implemented ``Chains`` concatenation methods.
* Implemented a ``Chains`` ``readcoda()`` method for importing CODA files.

### 0.6.2
* Added support for ``Array{MultivariateDistribution}`` to the missing values (MISS) sampler.
* Added a Binary Modified Metropolised Gibbs (BMMG) sampler for binary model parameters.
* Added bar plots for the summary of ``AbstractChains`` that contain values simulated for discrete model parameters.
* Compatibility updates for julia 0.4 release candidate 2.

### 0.6.1
* Compatibility updates for julia 0.4 prerelease.

### 0.6.0
* Stable release for julia 0.4.
* Added support for the specification of ``MultivariateDistribution`` arrays in stochastic nodes.
* Arrays in stochastic nodes must now be declared as a ``UnivariateDistribution[]`` or as a ``MultivariateDistribution[]``.  In previous package versions, arrays could be declared as a generic ``Distribution[]`` array.  This is no longer allowable due to the need to distinguish between arrays of univariate and multivariate distributions.
* The following changes were made to internal data structures and generally do not affect the user interface (i.e., model specification, sampling function calls, and sampler output diagnostics and summaries):
    * The ``VariateType``, which aliases ``Float64``, is deprecated and will be removed in a future version.
    * The abstract ``Variate`` type was separated into ``ScalarVariate`` and ``ArrayVariate`` types which are subtypes of ``Real`` and ``DenseArray``, respectively.
    * ``AbstractVariate`` was defined as the ``Union(ScalarVariate,ArrayVariate)``.
    * The ``Logical`` type was separated into ``ScalarLogical`` and ``ArrayLogical`` types which are subtypes of ``ScalarVariate`` and ``ArrayVariate``, respectively.
    * ``AbstractLogical`` was defined as the ``Union(ScalarLogical,ArrayLogical)``.
    * The ``Stochastic`` type was separated into ``ScalarStochastic`` and ``ArrayStochastic`` types which are subtypes of ``ScalarVariate`` and ``ArrayVariate``, respectively.
    * ``AbstractStochastic`` was defined as the ``Union(ScalarStochastic,ArrayStochastic)``.
    * ``AbstractDependent`` was defined as the ``Union(AbstractLogical,AbstractStochastic)``.
    * The ``nlink`` field of logical and stochastic types was renamed to ``linklength``.
    * An abstract ``AbstractChains`` type was implemented, the ``model`` field removed from the ``Chains`` type, and a new ``ModelChains`` type created to provide the ``model`` field.
* Added an example of sampling different parameter blocks with different stand-alone samplers (``amwg!`` and ``slice!``).
* Removed the ``insupport`` method for stochastic types.

### 0.5.2
* Applied *Mamba* changes through 0.4.12.

### 0.5.1
* Applied *Mamba* changes through 0.4.7 and updated compatibility with julia 0.4.0-dev.

### 0.5.0
* Branched off of *Mamba* 0.4.4.
* Initial release for the under-development version of julia 0.4.  *Mamba* 0.5.x releases exist to incorporate changes being made in the nightly builds of julia, and should be considered unstable.  They may contain compatibility issues or serious bugs.  Most users are advised to use *Mamba* 0.4.x releases for julia 0.3 or to wait for stable *Mamba* 0.6.x releases for julia 0.4.

### 0.4.12
* Updated documentation for user-defined distributions.

### 0.4.11
* Implemented `Chains` method function ``changerate`` to calculate parameter state space change rates (per iteration).
* Updated `Truncated` constructor for `Flat` distributions for compatibility with latest *Distributions* package.
* Simplified documentation instructions for user-defined univariate distributions.

### 0.4.10
* Optimized code and improved handling of sampler output in the Gelman convergence diagnostic.
* Added ``ask`` argument to the ``draw`` plot method.

### 0.4.9
* Added Heidelberger and Welch, and Raftery and Lewis convergence diagnostics.
* Added documentation and illustrations for all included diagnostics.
* Added documentation for creating user-defined distributions.

### 0.4.8
* Fixed `BoundsError()` occurring with autocorrelation plots.

### 0.4.7
* Improved formatting and outputting of posterior summary statistics
* Improved efficiency of DOT formatting of `Model` graphs.
* Exported `graph2dot` function to allow in-line processing of `Model` graphs with GraphViz package.

### 0.4.6
* Added `verbose` argument to `mcmc` methods to suppress printing of sampler progress to the console.
* Fixed calculation of effective sample size.

### 0.4.5
* Replaced the `Scale.discrete_color` function deprecated in the *Gadfly* package with `Scale.color_discrete`.

### 0.4.4
* Require julia 0.3.4 to remove version restriction on the *Colors* package.
* Call new *Distributions* package methods to get `InverseGamma` shape and scale parameters in the tutorial example.

### 0.4.3
* Fixed `ERROR: too many parameters for type Truncated`.

### 0.4.2
* Added support for optional arguments in `Chains` plot method.
* Implemented direct grid sampling for discrete univariate stochastic nodes with finite support.

### 0.4.1
* Updated and documented `predict` function as an official part of the package.
* Reorganized `Chains` methods documentation.

### 0.4.0

* Added support for user add-on packages and functions to allow for their inclusion in `Model` specifications.
* Added experimental `predict` (posterior prediction) function.
* Required the *Cairo* package.
* Removed deprecated `MCMC*` types and `slicewg` and `SliceWG` functions.
* Fixed `ERROR: GenericMvNormal not defined`.
* Distributions `DiagNormal` and `IsoNormal` removed and replaced by `MvNormal`.
* Distributions `DiagNormalCanon` and `IsoNormalCanon` removed and replaced by `MvNormalCanon`.
* Distributions `DiagTDist` and `IsoTDist` removed and replaced by `MvTDist`.

### 0.3.8

* Updated to fix warning and work with the latest versions of the *PDMat* and *Distributions* packages.

### 0.3.7

* Extend `Chains` draw method to allow automatic outputting of multiple plot grids to different files.
* Add `Chains` plot method to accommodate vectors of plot types.
* Fix variance calculation in `gewekediag()`.

### 0.3.6

* Fix for convert errors triggered by the *Color* package beginning with its version 0.3.9.

### 0.3.5

* Documentation updates only - primarily the addition of results to examples.
* No changes made to the source code.

### 0.3.4

* Added distributions documentation.
* Added jaws repeated measures analysis of variance example.

### 0.3.3

* Fixed the `rand` method definition error (`type DataType has no field body`) that began occurring with late release candidates and final release of julia 0.3.

### 0.3.2

* Fixed tuning parameter overwrites occurring with `pmap()` in single-processor mode.

### 0.3.1

* Added `chains` field to `Chains` type for tracking purposes.
* Fixed `mcmc` to accommodate restarting of chains subsetted by parameters and/or chains.
* Fixed plot legends to properly reference the chains being displayed.
* Added support for sampling of positive-definite matrices specified with Wishart or InverseWishart distributions.
* Added a block-diagonal normal (`BDiagNormal`) distribution.

### 0.3.0

* Implemented restarting of MCMC chains.
* Deprecated `slicewg` and `SliceWG`.  Replaced with `:univar` option to `slice` and `Slice`.

### 0.2.1

* Updated documentation.
* Simplified parallel code.

### 0.2.0

* Automatically load *Distributions* package.
* Implemented parallel execution of parallel chains on multi-processor systems.

### 0.1.0

* Removed `MCMC` prefix from type names, and deprecated `MCMC*` types.

### 0.0.2

* Renamed package from *MCMCsim* to *Mamba*.

### 0.0.1

* Initial public release.
