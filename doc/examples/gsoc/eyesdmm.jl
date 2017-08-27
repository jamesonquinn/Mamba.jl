using Mamba

## Data
eyes = Dict{Symbol, Any}(
  :y =>
    [529.0, 530.0, 532.0, 533.1, 533.4, 533.6, 533.7, 534.1, 534.8, 535.3,
     535.4, 535.9, 536.1, 536.3, 536.4, 536.6, 537.0, 537.4, 537.5, 538.3,
     538.5, 538.6, 539.4, 539.6, 540.4, 540.8, 542.0, 542.8, 543.0, 543.5,
     543.8, 543.9, 545.3, 546.2, 548.8, 548.7, 548.9, 549.0, 549.4, 549.9,
     550.6, 551.2, 551.4, 551.5, 551.6, 552.8, 552.9, 553.2],
  :N => 48, #should be divisible by 4 for inits below to work
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
inits = [
  Dict(:y => eyes[:y], :T => repeat([1,2], outer=Int(eyes[:N]//2)),
       :mu => [535.,540.], :sig => [10.,15.],
       :sigscale => eyes[:sigshape] * std(eyes[:y]),
       :G => 20.), #evil hack! Should be 2.
  Dict(:y => eyes[:y], :T => repeat([1,1,2,2], outer=Int(eyes[:N]//4)),
       :mu => [550.,551.], :sig => [15.,25.],
       :sigscale => eyes[:sigshape] * std(eyes[:y]),
       :G => 20.) #evil hack! Should be 2.
]



## Sampling Scheme
scheme = [DGS(:T),
          Slice(:mu, 8.0),
          Slice(:sig, 8.0),
          Slice(:sigscale, 8.0),
          RJS(:T, model, :G)]
setsamplers!(model, scheme)


## MCMC Simulations
sim = mcmc(model, eyes, inits, 1000, burnin=250, thin=2,
chains=2)
#describe(sim)
