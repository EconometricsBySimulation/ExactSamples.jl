using Random, OnlineStats, StatsBase, Gadfly, DataFrames, Optim, SumOfSquares

n = 100000
w = randn(n)
mean(w), var(w), skewness(w), kurtosis(w)

invlog(x) = -log.(-x .- minimum(-x) .+ 1)

w1 = log.(w .- minimum(w) .+ 1)
w2 = invlog(w)

cor(ordinalrank(w1),ordinalrank(w2))
skewness.([w1, w2])
kurtosis.([w1, w2])

dt = DataFrame(x = [w..., w1..., w2...], type = [fill(1, n)..., fill(2, n)..., fill(3, n)...])

wp  = plot(x=w, Geom.histogram)
w1p = plot(x=w1, Geom.histogram)
w2p = plot(x=w2, Geom.histogram)

hstack(wp, w1p, w2p)

Gadfly.set_default_plot_size(20cm, 14cm)

plot(dt, x=:x, color = :type, Geom.histogram(density=true))

signlog(x) = 

k1 =  sign.(w .- mean(w)) .* log.(log.(abs.(t1) .+ 1) .+ 1)
k2 =  sign.(w .- mean(w)) .* exp.(abs.(t1) .+ 1)

cor(ordinalrank(k1),ordinalrank(k2))

skewness.([k1,k2])
kurtosis.([k1,k2])

k1p = plot(x=k1, Geom.histogram)
k2p = plot(x=k2, Geom.histogram)

cor(ordinalrank(k1), ordinalrank(w))

bounded(x) = min(1, max(0, x))

standardize!(x::Array) = x[:] = (x .- mean(x))./var(x)^.5

# standardize!(w)
# standardize!(w1)
# standardize!(w2)
# standardize!(k1)
# standardize!(k2)

function f(x; skewtarget = .35, kurttarget = .4)
    χ = x .|> bounded |> x -> [bounded(1 - sum(x)), x...]
    z = w.*χ[1] + w1.*χ[2] + w2.*χ[3] + k1.*χ[4] + k2.*χ[5]
    (skewtarget - skewness(z))^2 + (kurttarget - kurtosis(z))^2
end


skewness.([w1, w2, k1, k2])
kurtosis.([w1, w2, k1,k2])

x0 = [0.0, 0.0, 0.0, 0.0]
b = optimize(f, x0)
Optim.minimizer(b) .|> bounded

w3 = Optim.minimizer(b) .|> bounded |> x -> (w.*(1 - sum(x)) + w1*x[1] + w2*x[2] + k1*x[3] + k2*x[4])

skewness(w3), kurtosis(w3)

plot(x=w3, Geom.histogram(bincount=100))

cor(ordinalrank(w),ordinalrank(w3))

################### polyscaler

polyscaler(x, scale) = sign.(x) .* (abs.(x .+ sign.(x))).^scale .- sign.(x) 

cor(polyscaler(w, .03) |> ordinalrank, w |> ordinalrank)

x -> polyscaler(x,.0001) |> x -> (skewness(x), kurtosis(x))

plot(x -> polyscaler(x,.0001), -1, 1)

function F(x; skewtarget = 1.5, kurttarget = -1)
    χ = [1 - sum(x), x...]
    z = w.*χ[1] + w1.*χ[2] + w2.*χ[3] + k1.*χ[4] + k2.*χ[5]
    (skewtarget - skewness(z))^2 + (kurttarget - kurtosis(z))^2
end

x0 = [0.0, 0.0, 0.0, 0.0]
b = optimize(F, x0)
Optim.minimizer(b)

w3 = Optim.minimizer(b) |> x -> (w.*(1 - sum(x)) + w1*x[1] + w2*x[2] + k1*x[3] + k2*x[4])

skewness(w3), kurtosis(w3)
plot(x=w3, Geom.histogram(bincount=100))

cor(ordinalrank(w),ordinalrank(w3))
