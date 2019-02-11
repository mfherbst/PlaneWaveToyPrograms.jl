#!/usr/bin/env julia

using StaticArrays
using LinearAlgebra
using PyPlot
include("visualise.jl")


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
    B = 2π * inv(Array(A'))
    unit_cell_volume = det(A)

    # Fill G_coords
    G_coords = Vector{SVector{3, Int}}()

    # Figure out upper bound n_max on number of basis functions for any dimension
    # Want |B*(i j k)|^2 <= Ecut
    n_max = 5 * ceil(Int, sqrt(Ecut) / opnorm(B))
    print(n_max)

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
            T[ig, jg] = sum(abs2, k + S.Gs[ig]) / 2
        end
    end
    T
end


function potential(S::Data, k, τ)
    RyToHartree = 1/2

    # Set Fourier terms for potential
    V_sym = Dict{Int64,Float64}()
    V_asym = Dict{Int64,Float64}()
    if S.Z == 14  # Si
        V_sym[3]  = -0.21 * RyToHartree
        V_sym[8]  = 0.04 * RyToHartree
        V_sym[11] = 0.08 * RyToHartree
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

    RyToHartree = 1/2

    # Set Fourier terms for potential
    V_sym = Dict{Int64,Float64}()
    V_asym = Dict{Int64,Float64}()
    if S.Z == 14  # Si
        V_sym[3]  = -0.21 * RyToHartree
        V_sym[8]  = 0.04 * RyToHartree
        V_sym[11] = 0.08 * RyToHartree
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
function compute(S::Data, kpoints)
    λs = empty(kpoints, Vector{Float64})
    vs = empty(kpoints, Matrix{Float64})
    τ = S.atoms[1]
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

function construct_diamond_structure(a, Z, Ecut)
    A = a / 2 .* [[0 1 1.]
                 [1 0 1.]
                 [1 1 0.]]
    τ = a / 8 .* @SVector[1, 1, 1]
    atoms = [τ, -τ]
    S = Data(A, atoms, Z, Ecut)
    S
end

function main()
    angströmToBohr = 1/0.52917721067
    a = 5.43 * angströmToBohr
    Z = 14
    Ecut = 17
    S = construct_diamond_structure(a, Z, Ecut)

    println("max G = $(maximum(maximum, S.G_coords))")

    println("A\n", S.A)
    println("B\n", S.B)
    println("A^T * B\n",transpose(S.A) * S.B)
    println("")

    kdelta = 0.05
    high_sym_points = [
        [0.5, 0.5, 0.5],    # L point
        [0.0, 0.0, 0.0],    # Γ point
        [1.0, 0.0, 0.0],    # X point
        #[1.0, 0.5, 0.0],    # W point
        [0.75, 0.75, 0.0],  # K point
        [0.0, 0.0, 0.0],    # Γ point
        #[0.5, 0.5, 0.5]     # L point
        #[1.0,0.25,0.25]     # U point
    ]
    ks = []
    for i in 2:size(high_sym_points, 1)
        linear(x) = S.B * high_sym_points[i - 1] +
            x * S.B * (high_sym_points[i] - high_sym_points[i - 1])
        newks = map(x -> linear(x), 0:kdelta:1)
        if length(ks) > 0 && newks[1] == ks[end]
            ks = vcat(ks, newks[2:end])
        else
            ks = vcat(ks, newks)
        end
    end
    println("Considered k points:")
    println(ks)
    println("-- -- -- -- -- -- --")
    # Other sampling:
    #   - Monkhorst-Pack mesh (H. J. Monkhorst and J. D. Pack, Phys. Rev. B 13, 5188 (1976). )
    #   - Chadi and Cohen
    λs, vs = compute(S, ks)

    # The band structure should be periodic with k
    # (at least in infinite - basis size)
    λtest1, _ = compute(S, [k .+ S.B[:, 1] for k in ks])
    λtest2, _ = compute(S, [k .+ S.B[:, 2] for k in ks])
    λtest3, _ = compute(S, [k .+ S.B[:, 3] for k in ks])

    λdiff1 = (λs .- λtest1)
    λdiff2 = (λs .- λtest2)
    λdiff3 = (λs .- λtest3)
    # figure()
    # for i in 1:5
    #     plot(HartreeToEv .* map(x -> x[i], λdiff1), label="λdiff1 $i")
    #     plot(HartreeToEv .* map(x -> x[i], λdiff2), label="λdiff2 $i")
    #     plot(HartreeToEv .* map(x -> x[i], λdiff3), label="λdiff3 $i")
    # end
    # legend()
    # show()

    HartreeToEv = 1 / 27.21138602
    max1 = HartreeToEv .* [maximum(map(x -> abs(x[i]), λdiff1)) for i in 1:3]
    max2 = HartreeToEv .* [maximum(map(x -> abs(x[i]), λdiff2)) for i in 1:3]
    max3 = HartreeToEv .* [maximum(map(x -> abs(x[i]), λdiff3)) for i in 1:3]
    print("max1 = $max1\n")
    print("max2 = $max2\n")
    print("max3 = $max3\n")
    @assert maximum(max1) < 1e-4
    @assert maximum(max2) < 1e-4
    @assert maximum(max3) < 1e-4

    #
    # Plotting
    #
    close()

    # Plot lattice
    figure()
    origin = a/8 .* ones(3)
    plot_lattice(S.A, S.atoms, radius=0.5*a, origin=origin .+ a.*[0.5,0.5,0.5])
    plot_cube(origin + a.*[0.5,0.5,0.5], a, "r-")
    PyPlot.plot3D(a.*[1/8, 9/8], a.*[1/8, 9/8], a.*[1/8,9/8], "b-")
    xlabel("x")
    ylabel("y")
    zlabel("z")

    # Plot Potential
    xs = map(x -> [x,x,x] .+ origin, 0:0.0125:a)
    xabs = map(x -> x[3], xs)
    τ = S.atoms[1]
    V = potential_real(S, τ, xs)
    figure()
    title("Potential along (x,x,x)")
    plot(xabs, HartreeToEv .* V)
    show()

    # Plot bands
    figure()
    for i in 1:10
        plot(HartreeToEv .* map(x -> x[i], λs), "x-", label="Band $i")
    end
    # legend()

    high_sym_indices = [
        i for (i,k) in enumerate(ks)
        if any(norm(k .- S.B * sp) < 1e-14
               for sp in high_sym_points)
    ]
    for idx in high_sym_indices
        axvline(x=idx-1, color="grey",
                linewidth=0.5)
    end
    show()

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
