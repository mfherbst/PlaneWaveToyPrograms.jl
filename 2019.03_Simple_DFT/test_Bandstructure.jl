using FFTW
include("SphericalHarmonics.jl")
include("System.jl")
include("BrilloinZonePath.jl")
include("BrilloinZoneMesh.jl")
include("PlaneWaveBasis.jl")
include("Units.jl")
include("PspHgh.jl")
using LinearAlgebra
using PyPlot
using ProgressMeter
using IterativeSolvers

# TODO qsq should only be computed once for each k

#
# Terms
#
"""A k-Block of the kinetic matrix"""
struct KineticBlock
    pw::PlaneWaveBasis
    idx_kpoint::Int

    """Cache for the terms |G + k|^2; size: n_G"""
    qsq::Vector{Float64}
end
function KineticBlock(pw::PlaneWaveBasis, idx_kpoint::Int)
    n_G = length(pw.Gmask[idx_kpoint])
    qsq = Vector{Float64}(undef, n_G)
    for (icont, ig) in enumerate(pw.Gmask[idx_kpoint])
        qsq[icont] = sum(abs2, pw.Gs[ig] + pw.kpoints[idx_kpoint])
    end
    KineticBlock(pw, idx_kpoint, qsq)
end
function LinearAlgebra.mul!(Y::SubArray, Tk::KineticBlock, B::SubArray)
    Y .= Diagonal(Tk.qsq / 2) * B
end
Base.size(Tk::KineticBlock) = (length(Tk.qsq), length(Tk.qsq))


function nuclear_attraction_generator_1e(system::System, G::Vector{Float64})
    if sum(abs.(G)) == 0 return 0.0 end  # Ignore DC component (net zero charge)
    sum(
        -4π / system.unit_cell_volume   # spherical Hankel transform prefactor
        * system.Zs[i] / sum(abs2, G)   # potential
        * cis(-dot(G, R))               # structure factor
        for (i, R) in enumerate(system.atoms)
    )
end


function psp_local_generator_1e(system::System, psp::PspHgh, G::Vector{Float64})
    if sum(abs.(G)) == 0 return 0.0 end  # Ignore DC component (net zero charge)
    pot = eval_psp_local_fourier(psp, G)
    sum(
        4π / system.unit_cell_volume  # Prefactor spherical Hankel transform
        * pot * cis(dot(G, R))        # potential and structure factor
        for R in system.atoms
    )
end

struct LocalPotentialBlock
    pw::PlaneWaveBasis
    idx_kpoint::Int
    idx_to_fft::Vector{Vector{Int}}

    potential_real::Array{Float64, 3}
    potential_1e_real::Array{Float64, 3}
end
function LocalPotentialBlock(pw::PlaneWaveBasis, idx_kpoint::Int, generator_1e)
    # Fill 1e potential in Fourier space and transform to real space
    V1e = generator_1e.(pw.Gs)
    potential_1e_real = G_to_R(pw, V1e)

    # Test FFT is an identity to the truncated potential in real space
    @assert maximum(abs.(G_to_R(pw, R_to_G!(pw, copy(potential_1e_real)))
                         - potential_1e_real)) < 1e-12

    # Note: This fails for small grids with a low Ecut
    @assert maximum(abs.(R_to_G!(pw, copy(potential_1e_real)) - V1e)) < 1e-12

    # Check the potential has no imaginary part
    @assert norm(imag(potential_1e_real)) < 1e-12
    potential_1e_real = real(potential_1e_real)

    # Prepare idx_to_fft for FFTs of the Psi
    idx_to_fft = pw.idx_to_fft[pw.Gmask[idx_kpoint]]
    @assert length(idx_to_fft) == length(pw.Gmask[idx_kpoint])

    potential_real = potential_1e_real
    LocalPotentialBlock(pw, idx_kpoint, idx_to_fft, potential_real, potential_1e_real)
end
function LinearAlgebra.mul!(Y::SubArray, Vk::LocalPotentialBlock, B::SubArray)
    n_G, n_vec = size(B)
    for ivec in 1:n_vec
        B_real = G_to_R(Vk.pw, B[:, ivec], idx_to_fft=Vk.idx_to_fft)
        Y[:, ivec] = R_to_G!(Vk.pw, Vk.potential_real .* B_real,
                             idx_to_fft=Vk.idx_to_fft)
    end
    Y
end


"""A k-Block of the non-local part of the pseudopotential"""
struct PspNonLocalBlock
    pw::PlaneWaveBasis
    idx_kpoint::Int

    """
    Cache for the projection vectors. For each l the Vector
    contains an array of size (n_G, n_proj, 2*lmax+1),
    where n_proj are the number of projectors for this l.

    These quantities are called ̂p_i^{l,m} in doc.tex. Note,
    that compared to doc.tex these quantities miss the factor
    i^l to stay in real arithmetic for them.
    """
    projection_vectors::Vector{Array{Float64, 3}}

    """The coefficients to employ between projection vectors"""
    projection_coefficients::Vector{Matrix{Float64}}

    """
    Cache for the structure factor
    """
    structure_factor::Matrix{ComplexF64}
end
function PspNonLocalBlock(pw::PlaneWaveBasis, idx_kpoint::Int, system::System, psp::PspHgh)
    k = pw.kpoints[idx_kpoint]
    n_G = length(pw.Gmask[idx_kpoint])
    n_atoms = length(system.atoms)

    # Evaluate projection vectors
    projection_vectors = Vector{Array{Float64, 3}}(undef, psp.lmax + 1)
    for l in 0:psp.lmax
        n_proj = size(psp.h[l + 1], 1)
        proj_l = zeros(Float64, n_G, n_proj, 2l + 1)
        for m in -l:l
            for iproj in 1:n_proj
                for (icont, ig) in enumerate(pw.Gmask[idx_kpoint])
                    # Compute projector for q and add it to proj_l
                    # including structure factor
                    q = pw.Gs[ig] + k
                    radial_il = eval_psp_projection_radial(psp, iproj, l, sum(abs2, q))
                    proj_l[icont, iproj, l + m + 1] = radial_il * ylm_real(l, m, q)
                    # im^l *
                end # ig
            end  # iproj
        end  # m
        projection_vectors[l + 1] = proj_l
    end  # l

    structure_factor = Matrix{ComplexF64}(undef, n_G, length(system.atoms))
    for (iatom, R) in enumerate(system.atoms)
        for (icont, ig) in enumerate(pw.Gmask[idx_kpoint])
            structure_factor[icont, iatom] = cis(dot(R, pw.Gs[ig]))
        end
    end

    projection_coefficients = psp.h
    PspNonLocalBlock(pw, idx_kpoint, projection_vectors, projection_coefficients,
                     structure_factor)
end
function LinearAlgebra.mul!(Y::SubArray, Vk::PspNonLocalBlock, B::SubArray)
    n_G, n_vec = size(B)
    n_atoms = size(Vk.structure_factor, 2)
    lmax = length(Vk.projection_vectors) - 1

    # TODO Maybe precompute this?
    # Amend projection vector by structure factor
    projsf = [
        broadcast(*, reshape(Vk.projection_vectors[l + 1], n_G, :, 2l+1, 1),
                  reshape(Vk.structure_factor, n_G, 1, 1, n_atoms))
        for l in 0:lmax
    ]

    # Compute product of transposed projection operator
    # times B for each angular momentum l
    projtB = Vector{Array{ComplexF64, 4}}(undef, lmax + 1)
    for l in 0:lmax
        n_proj = size(Vk.projection_vectors[l + 1], 2)
        projsf_l = projsf[l + 1]
        @assert size(projsf_l) == (n_G, n_proj, 2l + 1, n_atoms)

        # TODO use dot
        # Perform application of projector times B as matrix-matrix product
        projtB_l = conj(transpose(reshape(projsf_l, n_G, :))) *  B
        @assert size(projtB_l) ==  (n_proj * (2l + 1) * n_atoms, n_vec)

        projtB[l + 1] = reshape(projtB_l, n_proj, 2l + 1, n_atoms, n_vec)
    end

    # Compute contraction of above result with coefficients h
    # and another projector
    Ω = Vk.pw.unit_cell_volume
    Y[:] = zeros(ComplexF64, n_G, n_vec)
    for l in 0:lmax, midx in 1:2l + 1, iatom in 1:n_atoms
        h_l = Vk.projection_coefficients[l + 1]
        projsf_l = projsf[l + 1]
        projtB_l = projtB[l + 1]
        Y .+= projsf_l[:, :, midx, iatom] * (h_l * projtB_l[:, midx, iatom, :] / Ω)
    end

    Y
end


struct HamiltonianBlock{NonLocalPotential}
    pw::PlaneWaveBasis
    idx_kpoint::Int
    size::Tuple{Int,Int}

    T_k::KineticBlock
    Vloc_k::LocalPotentialBlock
    Vnloc_k::NonLocalPotential
end
function HamiltonianBlock(pw::PlaneWaveBasis, idx_kpoint::Int, system::System;
                          psp::PspHgh)
    T_k = KineticBlock(pw::PlaneWaveBasis, idx_kpoint::Int)
    Vloc_k = LocalPotentialBlock(pw, idx_kpoint,
                                 G -> psp_local_generator_1e(system, psp, G))
    Vnloc_k = PspNonLocalBlock(pw, idx_kpoint, system, psp)
    HamiltonianBlock(pw, idx_kpoint, size(T_k), T_k, Vloc_k, Vnloc_k)
end
function HamiltonianBlock(pw::PlaneWaveBasis, idx_kpoint::Int, system::System; psp::Nothing)
    T_k = KineticBlock(pw::PlaneWaveBasis, idx_kpoint::Int)
    Vloc_k = LocalPotentialBlock(pw, idx_kpoint,
                                 G -> nuclear_attraction_generator_1e(system, G))
    HamiltonianBlock{Nothing}(pw, idx_kpoint, size(T_k), T_k, Vloc_k, nothing)
end
function HamiltonianBlock(pw::PlaneWaveBasis, idx_kpoint::Int, system::System)
    HamiltonianBlock(pw, idx_kpoint, system)
end
function LinearAlgebra.mul!(Y::Matrix, H::HamiltonianBlock, B::Matrix)
    mul!(view(Y,:,:), H, view(B, :, :))
end
function LinearAlgebra.mul!(Y::Vector, H::HamiltonianBlock, B::Vector)
    mul!(view(Y,:,1), H, view(B, :, 1))
end
function LinearAlgebra.mul!(Y::SubArray, H::HamiltonianBlock{Nothing},
                            B::SubArray)
    mul!(Y, H.T_k, B)
    Y2 = Array{ComplexF64}(undef, size(Y))
    mul!(view(Y2,:,:), H.Vloc_k, B)
    Y .+= Y2
end
function LinearAlgebra.mul!(Y::SubArray, H::HamiltonianBlock, B::SubArray)
    mul!(Y, H.T_k, B)
    Y2 = Array{ComplexF64}(undef, size(Y))
    mul!(view(Y2,:,:), H.Vloc_k, B)
    Y .+= Y2
    mul!(view(Y2,:,:), H.Vnloc_k, B)
    Y .+= Y2
end
Base.size(H::HamiltonianBlock, idx::Int) = H.size[idx]
Base.size(H::HamiltonianBlock) = H.size
Base.eltype(H::HamiltonianBlock) = ComplexF64


"""
Kinetic-enery based preconditioner.
Applies 1 / (|k + G|^2 + α) to the vectors, when called with ldiv!

The rationale is to dampen the high-kinetic energy parts of the
Hamiltonian and decreases their size, thus make the Hamiltonian
more well-conditioned
"""
struct KineticPreconditionerBlock
    # TODO Check what this guy is called in the literature normally
    #      e.g. Kresse-Furtmüller paper
    pw::PlaneWaveBasis
    idx_kpoint::Int
    qsq::Vector{Float64}
    α::Float64
    diagonal::Vector{Float64}
end
function KineticPreconditionerBlock(H::HamiltonianBlock; α=0)
    qsq = H.T_k.qsq
    diagonal = 1 ./ (qsq ./ 2 .+ 1e-6 .+ α)
    KineticPreconditionerBlock(H.pw, H.idx_kpoint, qsq, α, diagonal)
end
function LinearAlgebra.ldiv!(Y, KinP::KineticPreconditionerBlock, B)
    Y .= Diagonal(KinP.diagonal) * B
end

#
# ---------------------------------------------------------
#

"""
Take a system and a PlaneWaveBasis and compute
the eigenvalues and eigenvectors at each kpoint
of the PlaneWaveBasis.

Returns two Arrays, the first the eigenvalues at each
k point, the second the eigenvectors at each k point.
"""
function compute_bands(pw::PlaneWaveBasis, system::System;
                       psp=nothing, n_bands=nothing)
    n_k = length(pw.kpoints)
    n_G_min = minimum(length.(pw.Gmask))
    n_bands = something(n_bands, n_G_min)
    @assert n_bands <= n_G_min

    λs = Vector{Vector{Float64}}(undef, n_k)
    vs = Vector{Matrix{ComplexF64}}(undef, n_k)

    pbar = Progress(n_k, desc="Computing k points: ",
                    dt=0.5, barglyphs=BarGlyphs("[=> ]"))
    for idx_kpoint in 1:n_k
        H = HamiltonianBlock(pw, idx_kpoint, system, psp=psp)
        largest = false  # Want smallest eigenpairs
        res = lobpcg(H, largest, n_bands, P=KineticPreconditionerBlock(H, α=0.1))

        @assert maximum(imag(res.λ)) < 1e-12
        λs[idx_kpoint] = real(res.λ)
        vs[idx_kpoint] = res.X
        next!(pbar)
    end
    λs, vs
end

#
# ---------------------------------------------------------
#

"""
Quickly test a few silicon k points
"""
function quicktest_silicon()
    kpoints = [
        [0,0,0],
        [0.229578295126352, 0.229578295126352, 0.000000000000000],
        [0.401762016471116, 0.401762016471116, 0.114789147563176],
        [0.280595694022912, 0.357121792398363, 0.280595694022912],
    ]
    ref = [
        [0.067966083141126, 0.470570565964348, 0.470570565966131,
         0.470570565980086, 0.578593208202602],
        [0.105959302042882, 0.329211057388161, 0.410969129077501,
         0.451613404615669, 0.626861886537186],
        [0.158220020418481, 0.246761395395185, 0.383362969225928,
         0.422345289771740, 0.620994908900183],
        [0.138706889457309, 0.256605657080138, 0.431494061152506,
         0.437698454692923, 0.587160336593700]
    ]

    Z = 14
    a = 5.431020504 * ÅtoBohr
    silicon = build_diamond_system(a, Z)
    Ecut = 15  # Hartree
    pw = PlaneWaveBasis(silicon, kpoints, Ecut)
    psp = PspHgh("./psp/CP2K-pade-Si-q4.hgh")

    # Check on the Hamiltonian
    opH = HamiltonianBlock(pw, 1, silicon; psp=psp)
    H = Matrix{ComplexF64}(undef, size(opH))
    mul!(H, opH, Matrix{ComplexF64}(I, size(opH)))
    @assert maximum(abs.(conj(transpose(H)) - H)) < 1e-14
    @assert maximum(abs.(imag(H))) < 1e-12

    λs, vs = compute_bands(pw, silicon, psp=psp, n_bands=5)
    for i in 1:length(ref)
        println(λs[i] - ref[i])
        @assert maximum(abs.(ref[i] - λs[i])[1:4]) < 1e-8
        @assert maximum(abs.(ref[i] - λs[i])[1:4]) < 1e-4
    end
end


function main()
    #
    # Setup system (Silicon) and model (Plane Wave basis)
    #
    Z = 14
    a = 5.431020504 * ÅtoBohr
    silicon = build_diamond_system(a, Z)
    bkmesh = build_diamond_bzmesh(silicon)

    Ecut = 15  # Hartree
    pw = PlaneWaveBasis(silicon, bkmesh.kpoints, Ecut)
    psp = PspHgh("./psp/CP2K-pade-Si-q4.hgh")

    # TODO Minimise Energy wrt. density

    #
    # Compute and plot bands
    #
    path = [(:L, :Γ), (:Γ, :X), (:X, :U), (:K, :Γ)]
    # path = [(:Γ, :X), (:X, :W), (:W, :K), (:K, :Γ), (:Γ, :L),
    #         (:L, :U), (:U, :W), (:W, :L), (:L, :K)]
    n_kpoints = 6
    kpath = BrilloinZonePath(silicon, path, n_kpoints)

    # Form new pw basis with the kpoints for above path
    pw = substitute_kpoints(pw, kpath.kpoints)
    println("FFT grid size: $(prod(pw.fft_size))")

    # Compute bands and plot
    λs, vs = compute_bands(pw, silicon, psp=psp, n_bands=10)
    plot_quantity(kpath, λs)
    savefig("bands_Si.pdf", bbox_inches="tight")
end

# TODO Put structure-factors into PlaneWaveBasis (or maybe some precomuted data object)

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
