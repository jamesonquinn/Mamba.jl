using IterTools
using Distributions


c2dists = [Normal(5,3),Normal(-5,3)]
c2ns = [2,1]

c3dists = [Normal(0,10),Normal(-12,5),Normal(5,2)]
c3ns = [20,7,13]

c4dists = [Normal(0,10),Normal(-8,5),Normal(3,2),Normal(-3,1)]
c4ns = [6,4,3,2]



function makeDat(dists,ns,fac)
  vals = collect(chain([rand(dists[i],ns[i]*fac) for i in 1:length(dists)]...))
  truetags = collect(chain([fill(i,ns[i]*fac) for i in 1:length(dists)]...))
  (vals,truetags)
end

(vals,truetags) = makeDat(c2dists, c2ns, 15)

println(repr([round(v,1) for v in vals]))

using Gadfly

plot(layer(x=vals,Geom.density(bandwidth=2),Theme(default_color="white")),
    layer(x=vals,color=truetags,Geom.density(bandwidth=2)))
