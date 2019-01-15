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
    n_max = ceil(Int, sqrt(Ecut) / opnorm(B))  # TODO Why spectral norm?

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
    # TODO Why? Ensure a power of 2 for fast FFTs
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
                # TODO Check I get it right
                pot[ig] -= 4π * S.Z * V_cell * cis(dot(G, R)) / sum(abs2, G)
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
    mul!(ψ_real, S.ifft_plan, ψ_real)  # TODO must not be aliased
    # TODO
    # IFFT has a normalization factor of 1/length(ψ)
    # so adjust to the fact that fft_size != sqrt(unit_cell_volume)
    ψ_real .*= (length(ψ_real) / sqrt(S.unit_cell_volume))
    @assert norm(imag(ψ_real)) < 1e-10
    real(ψ_real)
end

"""
Transform the wave function from real space to Fourier space
"""
function to_fourier(S::Structure, ψ_real)
    ψ_fourier_extended = S.fft_plan*ψ_real
    ψ_fourier = zeros(ComplexF64,S.n_G)
    for ig=1:S.n_G
        ψ_fourier[ig] = ψ_fourier_extended[S.G_to_fft[ig]...]
    end
    ψ_fourier .*= (sqrt(S.unit_cell_volume)/length(ψ_real))
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

    # TODO Different method? Laxer convergence criterion?
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
    @assert abs(VtotF[S.idx_DC]) < 1e-10  # TODO Why?

    Vtot = to_real(S, VtotF)
    ρ = computeρ(S, Vtot, N, tol)
    @assert all(ρ .>= 0)

    newρF = to_fourier(S, ρ)
    @assert abs(newρF[S.idx_DC] - N / sqrt(S.unit_cell_volume)) < tol  # TODO Why?
    @assert norm(imag(newρF)) < 1e-10 # only true because of inversion symmetry
    real(newρF)
end

"""
Solve the TF equation in a primitive cubic crystal with an atom of charge Z

Return the total potential and final electron density
on the grid in real space.
"""
function solve_TF_pc(L, Z, Ecut)
    A = L * Matrix(Diagonal(ones(3)))
    atoms = [[0.0,0.0,0.0]]
    N = Z*length(atoms)
    tol = 1e-10

    S = Structure(A, atoms, Z, Ecut; fft_supersampling=2.)
    println("Computing with Z=$Z, fft_size=$(S.fft_size)")

    Vext = pot_atom_grid(S)

    # Test we did not mess up Fourier transform (should be in a test suite)
    @assert to_fourier(S, to_real(S, Vext)) ≈ Vext

    # Start from a constant density
    # We want ∫ C_DC * e^{0} \D G = N
    # Thus C_DC (DC Fourier coefficient) = N / ∫ e^{0} \D G = N / sqrt(|Γ|)
    # TODO Why square root?
    ρ0F = zeros(S.n_G)
    ρ0F[S.idx_DC] = N / sqrt(S.unit_cell_volume)
    @assert mean(to_real(S, ρ0F)) * S.unit_cell_volume ≈ N

    # The payload function to be solved for with NLsolve
    payload!(residual, ρF) = (residual .= nextρF(S, Vext, ρF, N, tol) .- ρF)

    # TODO Häh?
    # work around https://github.com/JuliaNLSolvers/NLsolve.jl/issues/202
    od = OnceDifferentiable(payload!, identity, ρ0F, ρ0F, [])
    ρF  = nlsolve(od, ρ0F, method=:anderson, m=5, xtol=tol,
                 ftol=0.0, show_trace=true).zero

    # TODO filtered / unfiltered
    # # method 1
    # # energy from the filtered ρ
    # return real(mean(complex(to_real(S,ρ)).^(5/3)) * S.unit_cell_volume)

    # method 2
    # extract energy from the unfiltered ρ
    # computationally suboptimal but not critical
    VtotF = Vext + pot_hartree(S, ρF)
    Vtot = to_real(S, VtotF)
    ρ = computeρ(S, Vtot, N, tol)
    return S, ρ, Vtot
end

function main()
    Ecut = 5000
    L = 1.0
    Zs = 1:10

    Es = empty(Zs, Float64)
    ρs = empty(Zs, Array{Float64})
    Vtots = empty(Zs, Array{Float64})
    for Z in Zs
        S, ρ, Vtot = solve_TF_pc(L, Z, Ecut)
        midpoint = ceil(Int, S.fft_size / 2)

        # TODO only kinetic energy, right?
        E = mean(ρ.^(5/3)) * S.unit_cell_volume
        push!(Es, E)
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

    # TODO Hä?
    mat = hcat(ones(length(Zs)), Zs .^ (7/3))
    a, b = mat \ Es
    relabs = sqrt(sum(abs2, Es .- (a .+ b .* Zs .^ (7/3))) / var(Es))

    figure()
    plot(Zs .^ (7/3), Es, "-x")
    plot(Zs .^ (7/3), a .+ b .* Zs .^ (7/3), "-")
    title("L=$L, Ecut=$Ecut, $(@sprintf("%.2f",a)) + $(@sprintf("%.2f",b)) * Z^{7/3}," *
          "relerr=$(@sprintf("%.4f",relabs))")
    show()

    # using Profile
    # using ProfileView
    # Profile.clear()
    # @profile solve_TF_pc(1,1,10000)
    # ProfileView.view()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
