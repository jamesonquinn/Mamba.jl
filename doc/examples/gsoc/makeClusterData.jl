using IterTools
using Distributions

dists = [Normal(0,10),Normal(-12,5),Normal(5,2)]
ns = [20,7,13]

vals = collect(chain([rand(dists[i],ns[i]*3) for i in 1:3]...))
truetags = collect(chain([fill(i,ns[i]*3) for i in 1:3]...))

plot(layer(x=vals,Geom.density(bandwidth=2),Theme(default_color="white")),
    layer(x=vals,color=truetags,Geom.density(bandwidth=2)))

println(repr([round(v,1) for v in vals]))
