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


"""Nuclear attration potential matrix element <e_G|V|e_{G+ΔG}>"""
function elem_nuclear_attration(ΔG, system::System)
    if norm(ΔG) <= 1e-14  # Should take care of DC component
        return 0.0
    end
    sum(
        -4π / system.unit_cell_volume   # spherical Hankel transform prefactor
        * S.Zs[i] / sum(abs2, ΔG)       # potential
        * cis(-dot(ΔG, R))              # structure factor
        for (i, R) in enumerate(S.atoms)
    )
end


"""A k-Block of the nuclear attraction matrix"""
struct NuclearAttractionBlock
    pw::PlaneWaveBasis
    idx_kpoint::Int
    system::System
    data::Array{ComplexF64,2}  # TODO temporary
end
function NuclearAttractionBlock(pw::PlaneWaveBasis, idx_kpoint::Int, system::System)
    n_G = length(pw.Gmask[idx_kpoint])
    V = zeros(ComplexF64, n_G, n_G)
    for (icont, ig) in enumerate(pw.Gmask[idx_kpoint]),
            (jcont, jg) in enumerate(pw.Gmask[idx_kpoint])
        ΔG = system.Gs[ig] - system.Gs[jg]
        V[icont, jcont] = elem_nuclear_attration(ΔG, system)
    end
    NuclearAttractionBlock(pw, idx_kpoint, system, V)
end
LinearAlgebra.mul!(Y::SubArray, Vk::NuclearAttractionBlock, B::SubArray) = mul!(Y, Vk.data, B)


"""
Local pseudopotential part matrix element <e_G|V|e_{G+ΔG}>
"""
function elem_psp_loc(ΔG, system::System, psp::PspHgh)
    if norm(ΔG) <= 1e-14  # Should take care of DC component
        return 0.0        # (net zero charge)
    end

    sum(
        4π / system.unit_cell_volume    # spherical Hankel transform prefactor
        * psp_loc(psp, ΔG)              # potential
        * cis(-dot(ΔG, R))              # structure factor
        for R in system.atoms
    )
end


"""A k-Block of the local part of the pseudopotential"""
struct PspLocalBlock
    pw::PlaneWaveBasis
    idx_kpoint::Int
    system::System
    psp::PspHgh
    data::Array{ComplexF64,2}  # TODO temporary
end
function PspLocalBlock(pw::PlaneWaveBasis, idx_kpoint::Int, system::System, psp::PspHgh)
    k = pw.kpoints[idx_kpoint]
    n_G = length(pw.Gmask[idx_kpoint])
    V = zeros(ComplexF64, n_G, n_G)

    for (icont, ig) in enumerate(pw.Gmask[idx_kpoint]),
            (jcont, jg) in enumerate(pw.Gmask[idx_kpoint])
        ΔG = pw.Gs[ig] - pw.Gs[jg]
        V[icont, jcont] = elem_psp_loc(ΔG, system, psp)
    end
    PspLocalBlock(pw, idx_kpoint, system, psp, V)
end
LinearAlgebra.mul!(Y::SubArray, Vk::PspLocalBlock, B::SubArray) = mul!(Y, Vk.data, B)


"""
Matrix element of the nonlocal part of the pseudopotential.
Effectively computes <e_G1|V_k|e_G2> where e_G1 and e_G2
are plane waves and V_k is the fiber of the nonlocal part
for the k-point k.

Misses the structure factor and a factor of 1 / Ω.
"""
function elem_psp_nloc(k, G1, G2, psp::PspHgh)
    function calc_b(psp, i, l, m, Gk)
        qsq = sum(abs2, Gk)
        proj_il = eval_projection_radial(psp, i, l, qsq)
        ylm_real(l, m, Gk) * proj_il
    end

    accu = 0
    for l in 0:psp.lmax, m in -l:l
        hp = psp.h[l + 1]
        @assert ndims(hp) == 2
        for ij in CartesianIndices(hp)
            i, j = ij.I
            accu += (
                  calc_b(psp, i, l, m, G1 + k)
                * hp[ij] * calc_b(psp, j, l, m, G2 + k)
            )
        end
    end
    accu
end


"""A k-Block of the non-local part of the pseudopotential"""
struct PspNonLocalBlock
    pw::PlaneWaveBasis
    idx_kpoint::Int
    system::System
    psp::PspHgh
    size::Tuple{Int,Int}
    data::Array{ComplexF64,2}  # TODO temporary
end
function PspNonLocalBlock(pw::PlaneWaveBasis, idx_kpoint::Int, system::System, psp::PspHgh)
    k = pw.kpoints[idx_kpoint]
    n_G = length(pw.Gmask[idx_kpoint])
    V = zeros(ComplexF64, n_G, n_G)

    for (icont, ig) in enumerate(pw.Gmask[idx_kpoint]),
            (jcont, jg) in enumerate(pw.Gmask[idx_kpoint])
        for R in system.atoms
            pot = elem_psp_nloc(k, pw.Gs[ig], pw.Gs[jg], psp)

            # Add to potential after incorporating structure and volume factors
            ΔG = pw.Gs[ig] - pw.Gs[jg]
            V[icont, jcont] += pot * cis(dot(ΔG, R)) / system.unit_cell_volume
        end
    end
    PspNonLocalBlock(pw, idx_kpoint, system, psp, size(V), V)
end
LinearAlgebra.mul!(Y::SubArray, Vk::PspNonLocalBlock, B::SubArray) = mul!(Y, Vk.data, B)


"""
Local pseudopotential part matrix element <e_G|V|e_{G+ΔG}>
"""
function elem_psp_loc(ΔG, system::System, psp::PspHgh)
    if norm(ΔG) <= 1e-14  # Should take care of DC component
        return 0.0        # (net zero charge)
    end

    sum(
        4π / system.unit_cell_volume    # spherical Hankel transform prefactor
        * psp_loc(psp, ΔG)              # potential
        * cis(-dot(ΔG, R))              # structure factor
        for R in system.atoms
    )
end


struct HamiltonianBlock{LocalPotential, NonLocalPotential}
    pw::PlaneWaveBasis
    idx_kpoint::Int
    size::Tuple{Int,Int}

    T_k::KineticBlock
    Vloc_k::LocalPotential
    Vnloc_k::NonLocalPotential
end
function HamiltonianBlock(pw::PlaneWaveBasis, idx_kpoint::Int, system::System;
                          psp::PspHgh)
    T_k = KineticBlock(pw::PlaneWaveBasis, idx_kpoint::Int)
    Vloc_k = PspLocalBlock(pw, idx_kpoint, system, psp)
    Vnloc_k = PspNonLocalBlock(pw, idx_kpoint, system, psp)
    PspNonLocalBlock(pw, idx_kpoint, system, psp)
    HamiltonianBlock(pw, idx_kpoint, size(T_k), T_k, Vloc_k, Vnloc_k)
end
function HamiltonianBlock(pw::PlaneWaveBasis, idx_kpoint::Int, system::System)
    T_k = KineticBlock(pw::PlaneWaveBasis, idx_kpoint::Int)
    Vloc_k = NuclearAttractionBlock(pw, idx_kpoint, system)
    HamiltonianBlock(pw, idx_kpoint, size(T_k), T_k, Vloc_k, nothing)
end
function LinearAlgebra.mul!(Y::Matrix, H::HamiltonianBlock, B::Matrix)
    mul!(view(Y,:,:), H, view(B, :, :))
end
function LinearAlgebra.mul!(Y::Vector, H::HamiltonianBlock, B::Vector)
    mul!(view(Y,:,1), H, view(B, :, 1))
end
function LinearAlgebra.mul!(Y::SubArray, H::HamiltonianBlock{T, Nothing} where T,
                            B::SubArray)
    mul!(Y, H.T_k, B)
    Y2 = similar(Y)
    mul!(view(Y2,:,:), H.Vloc_k, B)
    Y .+= Y2
end
function LinearAlgebra.mul!(Y::SubArray, H::HamiltonianBlock, B::SubArray)
    mul!(Y, H.T_k, B)
    Y2 = similar(Y)
    mul!(view(Y2,:,:), H.Vloc_k, B)
    Y .+= Y2
    mul!(view(Y2,:,:), H.Vnloc_k, B)
    Y .+= Y2
end
Base.size(H::HamiltonianBlock, idx::Int) = H.size[idx]
Base.size(H::HamiltonianBlock) = H.size
Base.eltype(H::HamiltonianBlock) = ComplexF64

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
        res = lobpcg(H, largest, n_bands)
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

    opH = HamiltonianBlock(pw, 1, silicon; psp=psp)
    H = Matrix{Float64}(undef, size(opH))
    mul!(H, opH, Matrix{Float64}(I, size(opH)))
    @assert maximum(abs.(conj(transpose(H)) - H)) < 1e-12

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

    # Compute bands and plot
    λs, vs = compute_bands(pw, silicon, psp=psp, n_bands=10)
    plot_quantity(kpath, λs)
    savefig("bands_Si.pdf", bbox_inches="tight")
end

# TODO Put structure-factors into PlaneWaveBasis (or maybe some precomuted data object)

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
