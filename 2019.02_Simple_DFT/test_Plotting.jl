include("./System.jl")
include("./KPoints.jl")
include("./Units.jl")
include("./Visualisation.jl")
using LinearAlgebra
using PyPlot


# Silicon
Z = 14
a = 5.431020504 * angströmToBohr
silicon = build_diamond_system(a, Z)

kpath = [(:L, :Γ), (:Γ, :X), (:X, :U), (:K, :Γ)]
# kpath = [(:Γ, :X), (:X, :W), (:W, :K), (:K, :Γ), (:Γ, :L),
#         (:L, :U), (:U, :W), (:W, :L), (:L, :K)]
n_points = 5
kpoints = KPoints(silicon, kpath, n_points)


# Dummy band plot
values = ones(length(kpoints.kpoints))
plot_bands(kpoints, [[v] for v in values])

savefig("plot.pdf", bbox_inches="tight")
