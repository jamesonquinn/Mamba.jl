using Mamba
using IterTools

ns2 = [10,5]
ns3 = [20,7,13]
ns4 = [6,4,3,2]

## Data
sizes = ns2
twoClusters = Dict{Symbol, Any}( #Easy data, with clusters [Normal(0,10),Normal(-12,5),Normal(5,2)]
  :y => [2.0, 5.9, 9.3, 2.8, 6.0, 2.7, 4.5, 8.5, 3.6, 8.3, 3.6, 8.9, 7.7, 6.7, 3.2,
  6.8, 3.3, 2.9, 5.6, 6.1, 4.1, 0.7, 4.5, 2.0, 0.0, 4.0, 2.8, 5.2, 0.0, 9.1,
  -3.2, -7.2, -4.0, -5.0, -6.3, -7.3, -7.5, -5.1, -5.5, -9.1, -3.8, -0.1, -2.6,
  -7.2, -4.3],
  :truetags => collect(chain([fill(i,sizes[i]*3) for i in 1:length(sizes)]...)),
  :alpha => 1.
)

sizes = ns3
threeClusters = Dict{Symbol, Any}( #Easy data, with clusters [Normal(0,10),Normal(-12,5),Normal(5,2)]
  :y => [10.4, 8.7, -11.9, -13.0, -2.5, -17.5, -12.9, 24.3, 23.7, 3.6, 3.2, -6.9,
  -0.7, -20.0, 2.5, 2.8, 13.2, 3.0, 3.1, 12.6, -5.3, -0.9, -0.8, -4.2, 0.4,
  -11.4, -11.2, -0.4, -29.4, 18.9, 1.7, 3.3, -4.0, -2.4, -20.0, 6.6, 0.2, -9.1,
  7.4, 6.2, 1.1, 8.8, 3.8, -9.0, 1.6, 3.9, -3.5, -4.4, 10.3, 14.6, 3.3, 13.2,
  -19.0, 18.1, -2.3, 14.0, 1.1, 5.8, -2.9, -11.7, -12.4, -14.4, -11.9, -8.2,
  -13.9, -11.5, -5.2, -5.1, -11.0, -6.3, -3.7, -16.6, -10.3, -9.3, -13.5, -10.3,
  -16.7, -21.7, -11.1, -15.9, -14.1, 2.9, 3.4, 2.4, 7.0, 3.4, 5.6, 6.9, 5.6, 5.5,
  4.2, 1.3, 6.7, 4.4, 4.4, 7.6, 7.2, 5.8, 5.6, 5.6, 4.6, 3.0, 3.6, 8.2, 6.1, 4.6,
  2.3, 2.3, 8.6, 6.4, 3.4, -1.5, 5.0, 9.2, 8.6, 1.7, 5.3, 6.3, 3.7, 5.4],
  :truetags => collect(chain([fill(i,sizes[i]*3) for i in 1:length(sizes)]...)),
  :alpha => 1.
)

sizes = ns4
fourClusters = Dict{Symbol, Any}( #tougher to fit, with clusters [Normal(0,10),Normal(-8,5),Normal(3,2),Normal(-3,1)]
  :y => [0.6, -17.4, -6.8, 3.0, 11.3, -12.3, -17.6, 10.5, 8.7, 9.3, -11.9, 6.4, -2.9,
  1.7, -7.4, 16.6, 1.5, -7.9, -6.0, -7.3, -8.7, -9.3, -7.4, -11.8, -4.1, -11.4,
  -16.4, -8.1, -1.7, -9.0, 0.9, 2.7, 2.2, 5.1, 1.5, 4.4, 4.4, 1.4, 2.4, -5.5,
  -3.3, -2.5, -3.9, -2.6, -4.2],
  :truetags => collect(chain([fill(i,sizes[i]*3) for i in 1:length(sizes)]...)),
  :alpha => .2
)

for dat in (twoClusters,threeClusters,fourClusters)
  dat[:N] = length(dat[:y])
  dat[:mu0] = mean(dat[:y])
  dat[:sig0] = 2 * std(dat[:y])
  dat[:sigshape] = .25
  dat[:hypershape] = .1
  dat[:hyperscale] = .1
end



## Model Specification
model = Model(

  y = Stochelastic(1,
    (mu, sig, T, N) ->
      begin
        UnivariateDistribution[
          begin
            Normal(mu[Int(T[i])], sig[Int(T[i])])
          end
          for i in 1:N
        ]
      end,
    false
  ),

  T = Stochelastic(1,
    (alpha, N) -> DirichletPInt(alpha, N),
    false
  ),

  mu = Stochelastic(1,
    (mu0, sig0, T, G) ->
      begin
        UnivariateDistribution[
          begin
            Normal(mu0, sig0)
          end
          for i in 1:Int(G)
        ]
      end,
    false
  ),

  sig = Stochelastic(1,
    (sigshape, sigscale, T, G) ->
      begin
        UnivariateDistribution[
          begin
            InverseGamma(sigshape, sigscale)
          end
          for i in 1:Int(G)
        ]
      end,
    false
  ),

  sigscale = Stochastic(
    (hypershape, hyperscale) -> InverseGamma(hypershape, hyperscale),
    false
  ),

  G = Stochastic(
    () -> DiscreteUniform(1,1e6), #who cares — never is checked
    false
  )

)


## Initial Values
function makeInits(dat)
  inits = [
    Dict(:y => dat[:y], :T => repeat([1,1,2], outer=Int(dat[:N]//3)),
         :mu => [13.,-13.], :sig => [5.,15.],
         :sigscale => dat[:sigshape] * std(dat[:y]),
         :G => max(dat[:truetags]...)+0.), #evil hack!
    Dict(:y => dat[:y], :T => repeat([1,2,1], outer=Int(dat[:N]//3)),
         :mu => [13.,-13.], :sig => [2.,10.],
         :sigscale => dat[:sigshape] * std(dat[:y]),
         :G => max(dat[:truetags]...)+0.) #evil hack!
  ]
end


## Sampling Scheme
scheme = [DGS(:T),
          Slice(:mu, 8.0),
          Slice(:sig, 8.0),
          Slice(:sigscale, 8.0),
          RJS(:T, model, :G),
          Slice(:mu, 4.0),
          Slice(:sig, 4.0)]
setsamplers!(model, scheme)


Profile.init(n = 10^7, delay = 0.02)
## MCMC Simulations
dat = twoClusters

sim = mcmc(model, dat, makeInits(dat), 150, burnin=100, thin=2,
chains=2)

#@profile sim = mcmc(model, dat, makeInits(dat), 300, burnin=100, thin=2,
#chains=2)
#describe(sim)