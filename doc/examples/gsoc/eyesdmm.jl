using Mamba
using IterTools


ns = [20,7,13]

## Data
eyes = Dict{Symbol, Any}(
  :y => [10.4, 8.7, -11.9, -13.0, -2.5, -17.5, -12.9, 24.3, 23.7, 3.6, 3.2, -6.9,
  -0.7, -20.0, 2.5, 2.8, 13.2, 3.0, 3.1, 12.6, -5.3, -0.9, -0.8, -4.2, 0.4,
  -11.4, -11.2, -0.4, -29.4, 18.9, 1.7, 3.3, -4.0, -2.4, -20.0, 6.6, 0.2, -9.1,
  7.4, 6.2, 1.1, 8.8, 3.8, -9.0, 1.6, 3.9, -3.5, -4.4, 10.3, 14.6, 3.3, 13.2,
  -19.0, 18.1, -2.3, 14.0, 1.1, 5.8, -2.9, -11.7, -12.4, -14.4, -11.9, -8.2,
  -13.9, -11.5, -5.2, -5.1, -11.0, -6.3, -3.7, -16.6, -10.3, -9.3, -13.5, -10.3,
  -16.7, -21.7, -11.1, -15.9, -14.1, 2.9, 3.4, 2.4, 7.0, 3.4, 5.6, 6.9, 5.6, 5.5,
  4.2, 1.3, 6.7, 4.4, 4.4, 7.6, 7.2, 5.8, 5.6, 5.6, 4.6, 3.0, 3.6, 8.2, 6.1, 4.6,
  2.3, 2.3, 8.6, 6.4, 3.4, -1.5, 5.0, 9.2, 8.6, 1.7, 5.3, 6.3, 3.7, 5.4],
  :truetags => collect(chain([fill(i,ns[i]*3) for i in 1:3]...)),
  :N => 120, #should be divisible by 4 for inits below to work
  :alpha => 1.
)


eyes[:mu0] = mean(eyes[:y])
eyes[:sig0] = 2 * std(eyes[:y])
eyes[:sigshape] = .25
eyes[:hypershape] = .1
eyes[:hyperscale] = .1




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
    (mu0, sig0, G) ->
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
    (sigshape, sigscale, G) ->
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
inits = [
  Dict(:y => eyes[:y], :T => repeat([1,2], outer=Int(eyes[:N]//2)),
       :mu => [535.,540.], :sig => [10.,15.],
       :sigscale => eyes[:sigshape] * std(eyes[:y]),
       :G => 7.), #evil hack! Should be 2.
  Dict(:y => eyes[:y], :T => repeat([1,1,2,2], outer=Int(eyes[:N]//4)),
       :mu => [550.,551.], :sig => [15.,25.],
       :sigscale => eyes[:sigshape] * std(eyes[:y]),
       :G => 7.) #evil hack! Should be 2.
]



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
sim = mcmc(model, eyes, inits, 200, burnin=100, thin=2,
chains=2)

@profile sim = mcmc(model, eyes, inits, 200, burnin=100, thin=2,
chains=2)
#describe(sim)
