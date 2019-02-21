#!/usr/bin/env julia

# Aim: Understand pseudopotentials
#      Idea: Evaluate T + V_{ext}
#            versus
#            T + V_{ext} + V_{PP} = T + V_{ext} + V_{nl} + V_{loc}

using FFTW
using LinearAlgebra
using StaticArrays
using Unitful
using UnitfulAtomic
using PyPlot

# Define Ångström unit
@unit Å "Å" Ångström (1//10)u"nm" false

"""
System to be modelled
"""
struct System
    """Lattice constant"""
    a::Float64

    """3x3 lattice vectors, in columns. |Γ| = det(A)"""
    A::Matrix{Float64}

    """List of atomic positions in the unit cell"""
    atoms::Vector{SVector{3, Float64}}

    """charge of each atom"""
    Zs::Vector{Float64}

    # Derived quantities
    """Reciprocal lattice"""
    B::Matrix{Float64}

    """Volume of the unit cell"""
    unit_cell_volume::Float64

    """High-symmetry points in the reciprocal lattice"""
    high_sym_points::Dict{Symbol, Vector{Float64}}
end


"""
Constructor for the system

A      Lattice vectors as columns
atoms  Atom positions
Z      Atom change (communal to all)
fft_supersampling   Supersampling used during FFT
"""
function System(a, A, atoms, Zs, high_sym_points)
    B = 2π * inv(Array(A'))
    unit_cell_volume = det(A)
    System(a, A, atoms, Zs, B, unit_cell_volume, high_sym_points)
end


"""
Construct a system of the diamond structure
"""
function build_diamond_system(a, Z)
    A = a / 2 .* [[0 1 1.]
                  [1 0 1.]
                  [1 1 0.]]
    τ = a / 8 .* @SVector[1, 1, 1]
    atoms = [τ, -τ]
    Zs = [Z, Z]

    high_sym_points=Dict{Symbol, Vector{Float64}}(
        :Γ => 2π/a .* [  0,   0,   0],
        :X => 2π/a .* [  0,   1,   0],
        :W => 2π/a .* [1/2,   1,   0],
        :K => 2π/a .* [3/4, 3/4,   0],
        :L => 2π/a .* [1/2, 1/2, 1/2],
        :U => 2π/a .* [1/4,   1, 1/4],
    )
    System(a, A, atoms, Zs, high_sym_points)
end

#
# ------------------------------------------------------
#

"""Plane-wave discretisation model for the system"""
struct Model
    """Volume of the unit cell"""
    unit_cell_volume::Float64

    """Maximal kinetic energy |G|^2"""
    Ecut::Float64

    """Supersampling used during FFT"""
    fft_supersampling::Float64

    # Derived quantities
    """List of G vectors"""
    Gs::Vector{SVector{3, Float64}}

    """Reciprocal lattice coordinates of G vectors"""
    G_coords::Vector{SVector{3, Int}}

    """Index of the DC component"""
    idx_DC::Int

    # FFT parameters
    """Size of the FFT grid"""
    fft_size::Int

    """Translation table to transform from the
    plane-wave basis to the indexing convention needed for the FFT
    (DC component in the middle, others periodically, asymmetrically around it"""
    G_to_fft::Vector{SVector{3, Int}}

    # Space to store planned FFT operators from FFTW
    """Plan for forward FFT"""
    fft_plan::typeof(plan_fft!(im * randn(2, 2, 2)))

    """Plan for inverse FFT"""
    ifft_plan::typeof(plan_ifft!(im * randn(2, 2, 2)))
end


function Model(S::System, Ecut; fft_supersampling=2.)
    # Fill G_coords
    G_coords = Vector{SVector{3, Int}}()

    # Figure out upper bound n_max on number of basis functions for any dimension
    # Want |B*(i j k)|^2 <= Ecut
    n_max = ceil(Int, sqrt(Ecut) / opnorm(S.B))

    # Running index of selected G vectors
    ig = 1
    idx_DC = nothing # Index of the DC component inside G_coords
    for i=-n_max-1:n_max+1, j=-n_max-1:n_max+1, k=-n_max-1:n_max+1
        coord = [i; j; k]
        if coord == [0;0;0]
            idx_DC = ig
        end

        G = S.B * coord
        if sum(abs2, G) < Ecut
            # add to basis
            push!(G_coords,@SVector[i,j,k])
            ig += 1

            # Explicitly assert that n_max is large enough
            @assert all(abs.(coord) .<= n_max)
        end
    end
    @assert idx_DC != nothing && G_coords[idx_DC] == [0;0;0]
    Gs = [S.B*G_ind for G_ind in G_coords]
    n_G = length(Gs)

    # Setup FFT
    # Ensure a power of 2 for fast FFTs (because of the tree-like
    # divide and conquer structure of the FFT)2
    fft_size = nextpow(2, fft_supersampling * (2n_max + 1))
    G_to_fft = zeros(SVector{3, Int}, n_G)
    for ig in 1:n_G
        # Put the DC component at (1,1,1) and wrap the others around
        ifft = mod.(G_coords[ig], fft_size).+1
        G_to_fft[ig] = ifft
    end

    tmp  = Array{ComplexF64}(undef, fft_size, fft_size, fft_size)
    fft_plan = plan_fft!(tmp)
    ifft_plan = plan_ifft!(tmp)

    Model(S.unit_cell_volume, Ecut, fft_supersampling, Gs, G_coords,
          idx_DC, fft_size, G_to_fft, fft_plan, ifft_plan)
end

#
# ------------------------------------------------------
#

function g_to_r(M::Model, F_fourier)
    F_real = zeros(ComplexF64, M.fft_size, M.fft_size, M.fft_size)
    for ig=1:length(M.Gs)
        F_real[M.G_to_fft[ig]...] = F_fourier[ig]
    end
    F_real = M.ifft_plan * F_real  # Note: This destroys data in ψ_real
    # IFFT has a normalization factor of 1/length(ψ),
    # but the normalisation convention used in this code is
    # e_G(x) = e^iGx / sqrt(|Γ|), so we need to use the factor
    # below in order to match both conventions.
    F_real .*= (length(F_real) / sqrt(M.unit_cell_volume))
    @assert norm(imag(F_real)) < 1e-10
    real(F_real)
end


function r_to_g(M::Model, F_real)
    # This is needed, because S.fft_plan destroys data in ψ_real
    F_real = copy(F_real)

    # Do FFT on the full FFT plan, but only keep within
    # the n_G from the kinetic energy cutoff -> Lossy Compression of data
    F_fourier_extended = M.fft_plan * F_real
    F_fourier = zeros(ComplexF64, length(M.Gs))
    for ig=1:length(F_fourier)
        F_fourier[ig] = F_fourier_extended[M.G_to_fft[ig]...]
    end
    # Again adjust normalisation as in to_real
    F_fourier .*= (sqrt(M.unit_cell_volume) / length(F_real))
end

#
# ------------------------------------------------------
#

function kinetic_fourier(M::Model, k::Vector)
    n_G = length(M.Gs)
    T = zeros(ComplexF64, n_G, n_G)
    for ig in 1:n_G, jg in 1:n_G
        if ig == jg # Kinetic energy is diagonal
            T[ig, jg] = sum(abs2, k + M.Gs[ig]) / 2
        end
    end
    T
end


"""Generator for nuclear attration potential"""
function pot_generator_nuclear(S::System, ΔG)
    if ΔG == [0,0,0]
        return 0.0
    end
    sum(
        -4π * S.Zs[i] * sqrt(S.unit_cell_volume) * cis(dot(ΔG, R)) / sum(abs2, ΔG)
        for (i, R) in enumerate(S.atoms)
    )
end


"""Build potential matrix from inner function"""
function build_potential(M::Model, pot_generator)
    n_G = length(M.Gs)
    V = zeros(ComplexF64, n_G, n_G)
    for ig = 1:n_G, jg = 1:n_G
        V[ig,jg] = pot_generator(M.Gs[ig] - M.Gs[jg])
    end
    V
end


function pot_pp_loc_real(S::System, M::Model)
    # TODO
end

function pot_pp_nloc_real(S::System, M::Model)
    # TODO
end

function hamiltonian_fourier(S::System, M::Model, k::Vector)
    Tk = kinetic_fourier(M, k)
    Vext = build_potential(M, G -> pot_generator_nuclear(S, G))

    # Vloc_real = pot_pp_loc_real(S, M)
    # Vnloc_real = pot_pp_nloc_real(S, M)
    # Vpp = r_to_g(Vloc_real + Vnloc_real)
    Vpp = 0

    # Build and check Hamiltonian
    Hk = Tk .+ Vext .+ Vpp
    @assert maximum(imag(Hk)) < 1e-12
    Hk = real(Hk)
    @assert maximum(transpose(Hk) - Hk) < 1e-12
    Hk
end

#
# ------------------------------------------------------
#

"""
Take a system and a PW model and compute the eigenvalues
and eigenvectors at a set of kpoints.

Returns two Arrays, the first the eigenvalues at each
k point, the second the eigenvectors at each k point.
The ordering of the axis is such that the first index
indicates the k point and the others are the indices
of eigenvalues or eigenvectors.
"""
function compute_kpoints(S::System, M::Model, kpoints; n_bands=nothing)
    n_G = length(M.Gs)
    n_bands = something(n_bands, n_G)
    @assert n_bands <= n_G

    λs = Matrix{Float64}(undef, length(kpoints), n_bands)
    vs = Array{Float64}(undef, length(kpoints), n_G, n_bands)
    for (ik, k) in enumerate(kpoints)
        Hk = hamiltonian_fourier(S, M, k)
        λs[ik, :], vs[ik, :, :] = eigen(Symmetric(Hk), 1:n_bands)
    end
    λs, vs
end


"""
Given a system, a list of tuples of high-symmetry points
and the number of points in each k-direction in a uniform mesh in the
reciprocal unit cell, build the list of k points and the accumulated
distance travelled in reciprocal space.
"""
function kpoints_from_path(S::System, path::Array{Tuple{Symbol,Symbol}}, n_points)
    # Transform path to actual coordinates
    plot_path = [(S.high_sym_points[p[1]], S.high_sym_points[p[2]])
                 for p in path]

    ks = []
    accu_k_length = Vector([0.])
    for (st, en) in plot_path
        kdiff = (en - st) / n_points
        newks = [st .+ fac .* kdiff for fac in 0:n_points-1]
        append!(ks, newks)
        append!(accu_k_length, accumulate(+, norm.(diff(newks));
                                          init=accu_k_length[end]))
        push!(accu_k_length, accu_k_length[end] + norm(kdiff))
    end
    push!(ks, plot_path[end][end])
    @assert length(ks) == length(accu_k_length)
    ks, accu_k_length
end

#
# ------------------------------------------------------
#

function plot_potential(S::System, M::Model, pot_generator)
    V_fourier = [pot_generator(G) for G in M.Gs]
    V_real = g_to_r(M, V_fourier)
    @assert r_to_g(M, V_real) ≈ V_fourier

    # TODO Build x coordinate (using S.A)
    V_cut = [V_real[i,i,i] for i in 1:M.fft_size]
    figure()
    eV = uconvert(u"eV", 1u"Eh_au").val
    plot(V_cut * eV)
    title("Potential along (x,x,x)")
end


function plot_bands(S::System, M::Model, accu_length, ks, λs)
    n_ks, n_bands = size(λs)
    eV = uconvert(u"eV", 1u"Eh_au").val

    figure()
    for ib in 1:n_bands
        plot(accu_length, eV .* λs[:,ib], "r-")
    end
    # legend()

    high_sym_indices = [
        i for (i,k) in enumerate(ks)
        if any(norm(k .- sp) < 1e-14
               for sp in values(S.high_sym_points))
    ]
    for idx in high_sym_indices
        axvline(x=accu_length[idx], color="grey",
                linewidth=0.5)
    end

    get_label(i) = first(string(lal) for (lal, val) in S.high_sym_points
                         if norm(val .- ks[i]) < 1e-14)
    xticks([accu_length[idx] for idx in high_sym_indices],
           [get_label(idx) for idx in high_sym_indices])
    nothing
end

#
# ------------------------------------------------------
#

function main()
    # Silicon because it's fun
    Z = 14
    a = austrip(5.431020504Å)

    # Ecut= 45 * (2π / a)^2    # Cutoff for PW basis
    Ecut= 28 * (2π / a)^2    # Cutoff for PW basis
    # Ecut= 10 * (2π / a)^2    # Cutoff for PW basis
    n_points = 90            # Number of points in kpoints sampling
    n_bands = 10             # Number of bands to compute

    # Build structure and PW model
    S = build_diamond_system(a, Z)
    M = Model(S, Ecut)

    # Construct kpoints for plotting bands:
    # direction:     Λ         Δ         Σ
    plot_path = [(:L, :Γ), (:Γ, :X), (:X, :U), (:K, :Γ)]
    ks, accu_length = kpoints_from_path(S, plot_path, n_points)

    # Compute along path:
    λs, vs = compute_kpoints(S, M, ks, n_bands=n_bands)

    # Plot
    plot_potential(S, M, G -> pot_generator_nuclear(S, G))
    plot_bands(S, M, accu_length, ks, λs)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
