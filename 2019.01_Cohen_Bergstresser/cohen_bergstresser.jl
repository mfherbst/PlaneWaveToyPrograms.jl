#!/usr/bin/env julia

using StaticArrays
using LinearAlgebra
using PyPlot


"""
We solve the Cohen-Bergstresser model, i.e. we compute the lower part of the
spectrum of
   H = T + Vext
where
    Vext = ∑_G V_G e^(iGx)

If A is the real-space lattice and Γ the unit cell, we discretise
any A-periodic function ψ using plane waves:
   ψ = ∑_G c_G e_G(x)
with G the reciprocal lattice vectors and e_G the orthonormal basis
   e_G(x) = e^iGx / sqrt(|Γ|)

We solve the complete problem in Fourier space.
"""

"""
Static data of the calculation
"""
struct Data
    """3x3 lattice vectors, in columns. |Γ| = det(A)"""
    A::Matrix{Float64}

    """List of atomic positions in the unit cell"""
    atoms::Vector{SVector{3, Float64}}

    """List of atom charges"""
    Z::Float64

    """Maximal kinetic energy |G|^2"""
    Ecut::Float64

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
end


"""
Constructor for the data structure

A      Lattice vectors as columns
atoms  Atom positions
Zs     Atom changes
"""
function Data(A, atoms, Z, Ecut)
    B = 2π * inv(A')
    unit_cell_volume = det(A)

    # Fill G_coords
    G_coords = Vector{SVector{3, Int}}()

    # Figure out upper bound n_max on number of basis functions for any dimension
    # Want |B*(i j k)|^2 <= Ecut
    n_max = ceil(Int, sqrt(Ecut) / opnorm(B))

    # Running index of selected G vectors
    ig = 1
    for i=-n_max-1:n_max+1, j=-n_max-1:n_max+1, k=-n_max-1:n_max+1
        coord = [i; j; k]

        G = B * coord
        # println("$i $j $k $(sum(abs2, G))")
        if sum(abs2, G) < Ecut
            # add to basis
            push!(G_coords, @SVector[i,j,k])
            ig += 1

            # Explicitly assert that n_max is large enough
            @assert all(abs.(coord) .<= n_max)
        end
    end
    Gs = [B*G_ind for G_ind in G_coords]
    n_G = length(Gs)

    Data(A, atoms, Z, Ecut, B,
         unit_cell_volume, Gs, n_G, G_coords)
end


function kinetic(S::Data, k)
    T = zeros(ComplexF64, S.n_G, S.n_G)
    for ig in 1:S.n_G, jg in 1:S.n_G
        # Kinetic energy
        if ig == jg
            T[ig, jg] = sum(abs2, k) / 2 - sum(abs2, S.Gs[ig]) / 2
        end
    end
    T
end


function potential(S::Data, k, τ)
    # Set Fourier terms for potential
    V_sym = Dict{Int64,Float64}()
    V_asym = Dict{Int64,Float64}()
    if S.Z == 14  # Si
        V_sym[3]  = -0.21
        V_sym[8]  = 0.04
        V_sym[11] = 0.08
    else
        throw(ErrorException("Z == $(S.Z) not implemented"))
    end

    V = zeros(ComplexF64, S.n_G, S.n_G)
    for ig in 1:S.n_G, jg in 1:S.n_G
        Gcj = S.G_coords[jg]
        Gci = S.G_coords[ig]
        Gcsq = sum(abs2, Gci .- Gcj)
        ΔG = S.Gs[ig] - S.Gs[jg]

        # Symmetric and antisymmetric structure factor
        S_sym = cos(dot(τ, ΔG))
        S_asym = sin(dot(τ, ΔG))

        # Construct potential
        V[ig, jg] += S_sym * get(V_sym, Gcsq, 0)
        V[ig, jg] -= im * S_asym * get(V_asym, Gcsq, 0)
    end
    V
end

function potential_real(S::Data, τ, xs)
    """
    Evaluate potential on a set of real points
    """

    # Set Fourier terms for potential
    V_sym = Dict{Int64,Float64}()
    V_asym = Dict{Int64,Float64}()
    if S.Z == 14  # Si
        V_sym[3]  = -0.21
        V_sym[8]  = 0.04
        V_sym[11] = 0.08
    else
        throw(ErrorException("Z == $(S.Z) not implemented"))
    end

    V = zeros(ComplexF64, size(xs))
    for ig in 1:S.n_G
        for (ix, x) in enumerate(xs)
            G = S.Gs[ig]
            Gc = S.G_coords[ig]
            Gcsq = sum(abs2, Gc)

            # Symmetric and antisymmetric structure factor
            S_sym = cos(dot(τ, G))
            S_asym = sin(dot(τ, G))

            # FT factor
            V[ix] += (S_sym * get(V_sym, Gcsq, 0)
                      + im * S_asym * get(V_asym, Gcsq, 0)) * exp(-im * dot(G, x))
        end
    end
    @assert maximum(imag(V)) < 1e-12
    real(V)
end


"""
Compute a structure correpsonding to a diamond-type fcc structure.
"""
function compute_diamond_structure(L, Z, Ecut, kpoints)
    a = 1.
    A = L * Matrix(Diagonal(ones(3)))
    τ = L / 8 .* @SVector[a, a, a]
    atoms = [τ, -τ]
    S = Data(A, atoms, Z, Ecut)

    println("max G = $(maximum(maximum, S.G_coords))")

    λs = empty(kpoints, Vector{Float64})
    vs = empty(kpoints, Matrix{Float64})
    for k in kpoints
        Hk = potential(S, k, τ) + kinetic(S, k)

        @assert maximum(imag(Hk)) < 1e-12
        Hk = real(Hk)
        @assert maximum(transpose(Hk) - Hk) < 1e-12
        Hk = Symmetric(Hk)

        # Diagonalise
        λ, v = eigen(Hk)
        push!(λs, λ)
        push!(vs, v)
    end

    λs, vs
end

function main()
    L = 1.0
    Z = 14
    Ecut = 1000
    ks = map(x -> [0,0,x], 0:0.5:2π / L)
    λs, vs = compute_diamond_structure(L, Z, Ecut, ks)

    # --
    a = 1.
    A = L * Matrix(Diagonal(ones(3)))
    τ = L / 8 .* @SVector[a, a, a]
    atoms = [τ, -τ]
    S = Data(A, atoms, Z, Ecut)
    xs = map(x -> [x,x,x], 0:0.025:L)
    xabs = map(x -> x[3], xs)
    V = potential_real(S, τ, xs)
    figure()
    title("Potential along (x,x,x)")
    plot(xabs, V)
    show()
    # --

    return
    kabs = map(x -> sqrt(sum(abs2, x)), ks)
    figure()
    for i in [1,10]
        plot(kabs, map(x -> x[i], λs), label="Band $i")
    end
    legend()
    show()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
