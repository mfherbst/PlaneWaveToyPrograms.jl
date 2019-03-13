include("./System.jl")
include("./BrilloinZonePath.jl")
include("./Units.jl")
using LinearAlgebra
using PyPlot


# Silicon
Z = 14
a = 5.431020504 * angströmToBohr
silicon = build_diamond_system(a, Z)

path = [(:L, :Γ), (:Γ, :X), (:X, :U), (:K, :Γ)]
# path = [(:Γ, :X), (:X, :W), (:W, :K), (:K, :Γ), (:Γ, :L),
#         (:L, :U), (:U, :W), (:W, :L), (:L, :K)]
n_kpoints = 5
kpath = BrilloinZonePath(silicon, path, n_kpoints)

# Dummy plot
plot_quantity(kpath, [[1] for v in 1:length(kpath.kpoints)])
savefig("plot_BZ_path.pdf", bbox_inches="tight")
