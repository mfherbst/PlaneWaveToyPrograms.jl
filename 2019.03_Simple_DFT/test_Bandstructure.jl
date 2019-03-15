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
"""A k-Block of the kinetic operator"""
struct KineticBlock
    pw::PlaneWaveBasis
    idx_kpoint::Int
end
function LinearAlgebra.mul!(Y::AbstractMatrix, A::KineticBlock, B::AbstractMatrix)
    # TODO
end


"""Build the kinetic energy matrix"""
function kinetic(pw::PlaneWaveBasis, idx_kpoint::Int)
    n_G = length(pw.Gmask[idx_kpoint])
    k = pw.kpoints[idx_kpoint]
    T = zeros(ComplexF64, n_G, n_G)
    for (icont, ig) in enumerate(pw.Gmask[idx_kpoint])
        T[icont, icont] = sum(abs2, k + pw.Gs[ig]) / 2
    end
    T
end


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


"""Build full matrix of nuclear attraction potential"""
function pot_nuclear_attration(pw::PlaneWaveBasis, idx_kpoint::Int, system::System)
    n_G = length(pw.Gmask[idx_kpoint])
    V = zeros(ComplexF64, n_G, n_G)
    for (icont, ig) in enumerate(pw.Gmask[idx_kpoint]),
            (jcont, jg) in enumerate(pw.Gmask[idx_kpoint])
        ΔG = system.Gs[ig] - system.Gs[jg]
        V[icont, jcont] = elem_nuclear_attration(ΔG, system)
    end
    V
end


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
        proj_il = eval_projection_vector(psp, i, l, qsq)
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


"""
Build the full matrix for the nonlocal part of the pseudopotential
"""
function pot_psp_nloc(pw::PlaneWaveBasis, idx_kpoint::Int, system::System, psp::PspHgh)
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
    V
end


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


"""Build full matrix of local pseudopotential part"""
function pot_psp_loc(pw::PlaneWaveBasis, idx_kpoint::Int, system::System, psp::PspHgh)
    k = pw.kpoints[idx_kpoint]
    n_G = length(pw.Gmask[idx_kpoint])
    V = zeros(ComplexF64, n_G, n_G)

    for (icont, ig) in enumerate(pw.Gmask[idx_kpoint]),
            (jcont, jg) in enumerate(pw.Gmask[idx_kpoint])
        ΔG = pw.Gs[ig] - pw.Gs[jg]
        V[icont, jcont] = elem_psp_loc(ΔG, system, psp)
    end
    V
end


struct HamiltonianBlock
    data::Matrix
end
function HamiltonianBlock(pw::PlaneWaveBasis, idx_kpoint::Int, system::System;
                          psp::Union{PspHgh,Nothing})
    T = kinetic(pw, idx_kpoint)

    Vext = 0
    Vpsp = 0
    if psp == nothing
        Vext = pot_nuclear_attration(pw, idx_kpoint, system)
    else
        Vloc = pot_psp_loc(pw, idx_kpoint, system, psp)
        Vnloc = pot_psp_nloc(pw, idx_kpoint, system, psp)
        Vpsp  = Vloc .+ Vnloc
    end

    # Build and check Hamiltonian
    H = T .+ Vext .+ Vpsp
    @assert maximum(abs.(conj(transpose(H)) - H)) < 1e-12
    HamiltonianBlock(H)
end
LinearAlgebra.mul!(Y::Matrix, A::HamiltonianBlock, B::Matrix)     = mul!(Y, A.data, B)
LinearAlgebra.mul!(Y::Vector, A::HamiltonianBlock, B::Vector)     = mul!(Y, A.data, B)
LinearAlgebra.mul!(Y::SubArray, A::HamiltonianBlock, B::SubArray) = mul!(Y, A.data, B)
Base.size(H::HamiltonianBlock, idx::Int) = size(H.data)[idx]
Base.eltype(H::HamiltonianBlock) = eltype(H.data)

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
        H = HamiltonianBlock(pw, idx_kpoint, system, psp=psp).data
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
