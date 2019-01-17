#!/usr/bin/env julia

using StaticArrays
using FFTW
using Roots
using PyPlot
using NLsolve
using NLSolversBase
using Printf
using Statistics
using LinearAlgebra

# We solve the TF equation
# ρ = (εF - Vh - Vext)^3/2
# where
# Vext = sum of G1(x-Ra) for Ra running over the atoms in the unit cell
# Vh = G1 * ρ (periodic convolution)
# G1 = 4π ∑_G |G|^-2 e^iGx (solution of -ΔG1 = 4π \sum_R δ_R)
# subject to
# ∫ρ = N (number of electrons per unit cell, = Z * number of atoms in unit cell)

# Let A be the real-space lattice
# We discretize any A-periodic function ψ using plane waves:
# ψ = ∑_G c_G e_G(x), with G the reciprocal lattice vectors,
# and e_G the orthonormal basis e_G(x) = e^iGx / sqrt(|Γ|), with Γ the unit cell
# We evaluate the TF equation in real space through FFTs

# Note, that it is the convention to normalise e_G(x) = e^iGx / sqrt(|Γ|),
# which makes them orthornormal, such that a mass matrix can be avoided.

"""
State of the calculation
"""
struct Structure
    """3x3 lattice vectors, in columns. |Γ| = det(A)"""
    A::Matrix{Float64}

    """List of atomic positions in the unit cell"""
    atoms::Vector{SVector{3, Float64}}

    """charge of each atom"""
    Z::Float64

    """Maximal kinetic energy |G|^2"""
    Ecut::Float64

    """Supersampling used during FFT"""
    fft_supersampling::Float64

    # Derived quantities
    """Reciprocal lattice"""
    B::Matrix{Float64}

    """Volume of the unit cell"""
    unit_cell_volume::Float64

    """List of G vectors"""
    Gs::Vector{SVector{3, Float64}}

    """Number of G vectors"""
    n_G::Int

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

"""
Constructor for the structure function

A      Lattice vectors as columns
atoms  Atom positions
Z      Atom change (communal to all)
fft_supersampling   Supersampling used during FFT
"""
function Structure(A, atoms, Z, Ecut; fft_supersampling=2.)
    B = 2π * inv(A')
    unit_cell_volume = det(A)

    # Fill G_coords
    G_coords = Vector{SVector{3, Int}}()

    # Figure out upper bound n_max on number of basis functions for any dimension
    # Want |B*(i j k)|^2 <= Ecut
    n_max = ceil(Int, sqrt(Ecut) / opnorm(B))

    # Running index of selected G vectors
    ig = 1
    idx_DC = nothing # Index of the DC component inside G_coords
    for i=-n_max-1:n_max+1, j=-n_max-1:n_max+1, k=-n_max-1:n_max+1
        coord = [i; j; k]
        if coord == [0;0;0]
            idx_DC = ig
        end

        G = B * coord
        # println("$i $j $k $(sum(abs2, G))")
        if sum(abs2, G) < Ecut
            # add to basis
            push!(G_coords,@SVector[i,j,k])
            ig += 1

            # Explicitly assert that n_max is large enough
            @assert all(abs.(coord) .<= n_max)
        end
    end
    @assert idx_DC != nothing && G_coords[idx_DC] == [0;0;0]
    Gs = [B*G_ind for G_ind in G_coords]
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

    a = Array{ComplexF64}(undef, fft_size, fft_size, fft_size)
    fft_plan = plan_fft!(a)
    ifft_plan = plan_ifft!(a)


    Structure(A, atoms, Z, Ecut, fft_supersampling,
              B, unit_cell_volume, Gs, n_G, G_coords, idx_DC,
              fft_size, G_to_fft, fft_plan, ifft_plan)
end

"""
Compute the external coulombic potential induced by the atom grid
on the Fourier grid (in Fourier space)
"""
function pot_atom_grid(S::Structure)
    V_cell = sqrt(S.unit_cell_volume)
    pot = zeros(ComplexF64, S.n_G)
    for iatom = 1:length(S.atoms)
        R = S.atoms[iatom]
        for ig = 1:S.n_G
            if ig != S.idx_DC
                G = S.Gs[ig]
                pot[ig] -= 4π * S.Z * V_cell * cis(dot(G, R)) / sum(abs2, G)
            end
        end
    end
    pot
end

"""
Compute the external potential induced by an atom grid,
where the repulsion acts as 1/|G|^4 (in Fourier space)
"""
function pot_atom_grid4(S::Structure)
    abs4(x) = abs2(x)^2
    V_cell = sqrt(S.unit_cell_volume)
    pot = zeros(ComplexF64, S.n_G)
    for iatom = 1:length(S.atoms)
        R = S.atoms[iatom]
        for ig = 1:S.n_G
            if ig != S.idx_DC
                G = S.Gs[ig]
                pot[ig] -= 4π * S.Z * V_cell * cis(dot(G, R)) / sum(abs4, G)
            end
        end
    end
    pot
end

"""
Computes the Hartree potential associated with ρ (in Fourier space)
"""
function pot_hartree(S::Structure, ρ)
    pot = zeros(ComplexF64,S.n_G)
    @assert length(ρ) == S.n_G
    for ig = 1:S.n_G
        G = S.Gs[ig]
        if ig != S.idx_DC
            pot[ig] += 4π * ρ[ig] / sum(abs2, G)
        end
    end
    pot
end

"""
Transform the wave function from Fourier space to real space
"""
function to_real(S::Structure, ψ_fourier)
    ψ_real = zeros(ComplexF64, S.fft_size, S.fft_size, S.fft_size)
    for ig=1:S.n_G
        ψ_real[S.G_to_fft[ig]...] = ψ_fourier[ig]
    end
    ψ_real = S.ifft_plan * ψ_real  # Note: This destroys data in ψ_real
    # IFFT has a normalization factor of 1/length(ψ),
    # but the normalisation convention used in this code is
    # e_G(x) = e^iGx / sqrt(|Γ|), so we need to use the factor
    # below in order to match both conventions.
    ψ_real .*= (length(ψ_real) / sqrt(S.unit_cell_volume))
    @assert norm(imag(ψ_real)) < 1e-10
    real(ψ_real)
end

"""
Transform the wave function from real space to Fourier space
"""
function to_fourier(S::Structure, ψ_real)
    # This is needed, because S.fft_plan destroys data in ψ_real
    ψ_real = copy(ψ_real)

    # Do FFT on the full FFT plan, but only keep within
    # the n_G from the kinetic energy cutoff -> Lossy Compression of data
    ψ_fourier_extended = S.fft_plan * ψ_real
    ψ_fourier = zeros(ComplexF64, S.n_G)
    for ig=1:S.n_G
        ψ_fourier[ig] = ψ_fourier_extended[S.G_to_fft[ig]...]
    end
    # Again adjust normalisation as in to_real
    ψ_fourier .*= (sqrt(S.unit_cell_volume) / length(ψ_real))
end

"""
Computes the density corresponding to a total potential, i.e.
solve the equation ∫(εF - Vext - Vh)^3/2 = N for εF.

Both Vtot (total potential) and the output density are in
real space.
"""
function computeρ(S::Structure, Vtot, N, tol)
    # Buffer for the guess for ρ in the root finding problem
    # sketched in the docstring
    ρguess = copy(Vtot)

    # Fast x^(3/2)
    reg_pow(x) = x < 0 ? zero(x) : sqrt(x*x*x)

    # Form the density in-place and return it
    makeρ!(ρguess, ε, Vtot) = (ρguess .= reg_pow.(ε .- Vtot))

    # Function to update ρguess and return the number of particles
    # it represents
    function compute_n_elec!(ε, ρguess, Vtot, unit_cell_volume)
        makeρ!(ρguess, ε, Vtot)
        mean(ρguess) * unit_cell_volume
    end

    # Use defined functions to find the Fermi level
    # giving the requested number of particles
    εF = find_zero(ε -> compute_n_elec!(ε, ρguess, Vtot, S.unit_cell_volume) - N, 0.0,
                   rtol=0.0, atol=tol, xatol=0.0, xrtol=0.0, verbose=false)
    ρguess
end

"""
Take an external potential, density and particle number
and build the next iterated density

VextF   external potential in Fourier space
ρF      density in Fourier space
"""
function nextρF(S::Structure, VextF, ρF, N, tol)
    VtotF = VextF + pot_hartree(S, ρF)
    # Not sure why this is the case:
    @assert abs(VtotF[S.idx_DC]) < 1e-10

    Vtot = to_real(S, VtotF)
    ρ = computeρ(S, Vtot, N, tol)
    @assert all(ρ .>= 0)
    newρF = to_fourier(S, ρ)

    # Because of charge neutrality:
    #     DC component of newρF = mean average of newρF = N / |Γ|
    # since the normalisation convention we use is e_G = exp(-irG) / sqrt(|Γ|),
    # this implies newρF[S.idx_DC] = N / |Γ| * sqrt(|Γ|) = N / sqrt(|Γ|)
    @assert abs(newρF[S.idx_DC] - N / sqrt(S.unit_cell_volume)) < tol

    # Because of inversion symmetry in the system:
    @assert norm(imag(newρF)) < 1e-10
    real(newρF)
end

"""
Solve the TF equation in a primitive cubic crystal with an atom of charge Z

Return the total potential and final electron density
on the grid in real space.
"""
function solve_TF_pc(L, Z, Ecut; pot_external=pot_atom_grid, fft_supersampling=2.)
    A = L * Matrix(Diagonal(ones(3)))
    atoms = [[0.0,0.0,0.0]]
    N = Z*length(atoms)
    tol = 1e-10

    S = Structure(A, atoms, Z, Ecut; fft_supersampling=fft_supersampling)
    println("Computing with Z=$Z, fft_size=$(S.fft_size), pot_external=$(pot_atom_grid)")

    VextF = pot_external(S)

    # Test we did not mess up Fourier transform (should be in a test suite)
    @assert to_fourier(S, to_real(S, VextF)) ≈ VextF

    # Start from a constant density
    # We want ∫ C_DC * e^{0} \D G = N
    # Thus C_DC (DC Fourier coefficient) = N / ∫ e^{0} \D G = N / sqrt(|Γ|),
    # because of our normalisation convention.
    ρ0F = zeros(S.n_G)
    ρ0F[S.idx_DC] = N / sqrt(S.unit_cell_volume)
    @assert mean(to_real(S, ρ0F)) * S.unit_cell_volume ≈ N

    # The function to be solved for with NLsolve
    residual!(residual, ρF) = (residual .= nextρF(S, VextF, ρF, N, tol) .- ρF)

    # work around https://github.com/JuliaNLSolvers/NLsolve.jl/issues/202
    od = OnceDifferentiable(residual!, identity, ρ0F, ρ0F, [])
    ρF  = nlsolve(od, ρ0F, method=:anderson, m=5, xtol=tol,
                 ftol=0.0, show_trace=true).zero

    VtotF = VextF + pot_hartree(S, ρF)
    Vtot = to_real(S, VtotF)
    ρ = computeρ(S, Vtot, N, tol)
    return S, ρ, Vtot
end

function plot_dependency_on_Z()
    Ecut = 5000
    L = 1.0
    Zs = 1:2:20

    Ekins = empty(Zs, Float64)
    ρs = empty(Zs, Array{Float64})
    Vtots = empty(Zs, Array{Float64})
    for Z in Zs
        S, ρ, Vtot = solve_TF_pc(L, Z, Ecut; pot_external=pot_atom_grid)

        Ekin = mean(ρ.^(5/3)) * S.unit_cell_volume
        push!(Ekins, Ekin)
        push!(ρs, ρ[1, :, 1])
        push!(Vtots, Vtot[1, :, 1])
    end

    # Plot potential energy
    figure()
    title("Potential energy (real space)")
    for (i, Z) in enumerate(Zs)
        plot(Vtots[i], "-x", label="Z=$(@sprintf("%.2f", Z))")
    end
    legend()
    show()

    # Plot density
    figure()
    title("Density (real space)")
    for (i, Z) in enumerate(Zs)
        plot(ρs[i], "-x", label="Z=$(@sprintf("%.2f", Z))")
    end
    legend()
    show()

    # Compare against the known exact assymptotics of the
    # Thomas-Fermi-Atom with respect to Z
    mat = hcat(ones(length(Zs)), Zs .^ (7/3))
    a, b = mat \ Ekins
    relabs = sqrt(sum(abs2, Ekins .- (a .+ b .* Zs .^ (7/3))) / var(Ekins))

    figure()
    plot(Zs .^ (7/3), Ekins, "-x")
    plot(Zs .^ (7/3), a .+ b .* Zs .^ (7/3), "-")
    title("L=$L, Ecut=$Ecut, $(@sprintf("%.2f",a)) + $(@sprintf("%.2f",b)) * Z^{7/3}," *
          "relerr=$(@sprintf("%.4f",relabs))")
    show()
end

function plot_dependency_on_Ecut()
    Ecuts = 50 * collect(1:4:200)
    L = 1.0
    Z = 5

    Ekins = empty(Ecuts, Float64)
    ρs = empty(Ecuts, Array{Float64})
    Vtots = empty(Ecuts, Array{Float64})
    n_Gs = empty(Ecuts, Int)
    for Ecut in Ecuts
        S, ρ, Vtot = solve_TF_pc(L, Z, Ecut; pot_external=pot_atom_grid4)

        Ekin = mean(ρ.^(5/3)) * S.unit_cell_volume
        push!(Ekins, Ekin)
        push!(ρs, ρ[1, :, 1])
        push!(Vtots, Vtot[1, :, 1])
        push!(n_Gs, S.n_G)
    end

    # Compute the grid point coordinates
    grid_points = [collect(1:length(ρ)) / length(ρ) for (i, ρ) in enumerate(ρs)]

    # Plot energies
    figure()
    title("Kinetic energy vs n_G")
    semilogy(n_Gs[1:end-1], abs.(Ekins[1:end-1] .- Ekins[end]), "-x")
    legend()
    show()

    # Plot potential energy
    figure()
    title("Potential energy (real space)")
    for (i, Ecut) in enumerate(Ecuts)
        plot(grid_points[i], Vtots[i], "-x", label="Ecut=$Ecut")
    end

    # Super rough fit to get an idea of the cusp
    before = grid_points[end] .<= 0.5
    after = grid_points[end] .> 0.5
    α = (Vtots[end][4] - Vtots[end][3]) / grid_points[end][1]
    β = Vtots[end][1]
    fit = β .+ α .* abs.(grid_points[end]) .* before .+
               α .* abs.(grid_points[end] .- 1) .* after
    plot(grid_points[end], fit, "-", label="Ref")
    legend()
    show()

    # Plot density
    figure()
    title("Density (real space)")
    for (i, Ecut) in enumerate(Ecuts)
        plot(grid_points[i], ρs[i], "-x", label="Ecut=$Ecut")
    end
    legend()
    show()

    println("Final kinetic energy is $(Ekins[end])")
end

function plot_dependency_on_supersampling()
    Ecut = 5000
    L = 1.0
    Z = 5
    supersamplings = [1., 2., 3., 6.]

    Vtots = empty(supersamplings, Array{Float64})
    for ssample in supersamplings
        S, ρ, Vtot = solve_TF_pc(L, Z, Ecut; pot_external=pot_atom_grid4,
                                 fft_supersampling=ssample)
        push!(Vtots, Vtot[1, :, 1])
    end

    # Compute the grid point coordinates
    grid_points = [collect(1:length(V)) / length(V)
                   for (i, V) in enumerate(Vtots)]

    # Plot potential energy
    figure()
    title("Potential energy (real space)")
    for (i, ssample) in enumerate(supersamplings)
        plot(grid_points[i], Vtots[i], "-x", label="fft_supersampling=$ssample")
    end
    legend()
    show()
end

function run_profiler()
    # using Profile
    # using ProfileView
    # Profile.clear()
    # @profile solve_TF_pc(1,1,10000)
    # ProfileView.view()
end

function main()
    plot_dependency_on_Ecut()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
