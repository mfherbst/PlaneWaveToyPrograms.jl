using FFTW
include("SphericalHarmonics.jl")
include("System.jl")
include("BrilloinZonePath.jl")
include("BrilloinZoneMesh.jl")
include("PlaneWaveBasis.jl")
include("Units.jl")
include("PspHgh.jl")
using Printf
using LinearAlgebra
using PyPlot
using ProgressMeter
using IterativeSolvers
import IterativeSolvers: LOBPCGResults

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
import Base: *
function *(Tk::KineticBlock, B::Array)
    Y = similar(B)
    mul!(view(Y,:,:), Tk, view(B,:,:))
end


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

    potential_1e_real::Array{Float64, 3}
    potential_2e_real::Array{Float64, 3}
end
function LocalPotentialBlock(pw::PlaneWaveBasis, idx_kpoint::Int, generator_1e,
                             potential_2e_real::Array{ComplexF64, 3})
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

    @assert norm(imag(potential_2e_real)) < 1e-12
    potential_2e_real = real(potential_2e_real)
    LocalPotentialBlock(pw, idx_kpoint, idx_to_fft, potential_1e_real,
                        potential_2e_real)
end
function LinearAlgebra.mul!(Y::SubArray, Vk::LocalPotentialBlock, B::SubArray)
    n_G, n_vec = size(B)
    for ivec in 1:n_vec
        B_real = G_to_R(Vk.pw, B[:, ivec], idx_to_fft=Vk.idx_to_fft)
        VkB_real = (Vk.potential_1e_real .+ Vk.potential_2e_real) .* B_real
        Y[:, ivec] = R_to_G!(Vk.pw, VkB_real, idx_to_fft=Vk.idx_to_fft)
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
        projtB_l = adjoint(reshape(projsf_l, n_G, :)) *  B
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
import Base: *
function *(Vk::PspNonLocalBlock, B::Array)
    Y = similar(B)
    mul!(view(Y,:,:), Vk, view(B,:,:))
end


struct HamiltonianBlock{NonLocalPotential}
    pw::PlaneWaveBasis
    idx_kpoint::Int
    size::Tuple{Int,Int}

    T_k::KineticBlock
    Vloc_k::LocalPotentialBlock
    Vnloc_k::NonLocalPotential
end
function HamiltonianBlock(pw::PlaneWaveBasis, idx_kpoint::Int, system::System,
                          psp::PspHgh)
    HamiltonianBlock(pw, idx_kpoint, system, psp, zeros(ComplexF64, pw.fft_size...))
end
function HamiltonianBlock(pw::PlaneWaveBasis, idx_kpoint::Int, system::System,
                          psp::PspHgh, potential_2e_real::Array{ComplexF64, 3})
    T_k = KineticBlock(pw::PlaneWaveBasis, idx_kpoint::Int)
    Vloc_k = LocalPotentialBlock(pw, idx_kpoint,
                                 G -> psp_local_generator_1e(system, psp, G),
                                 potential_2e_real)
    Vnloc_k = PspNonLocalBlock(pw, idx_kpoint, system, psp)
    HamiltonianBlock{PspNonLocalBlock}(pw, idx_kpoint, size(T_k), T_k, Vloc_k, Vnloc_k)
end
function HamiltonianBlock(pw::PlaneWaveBasis, idx_kpoint::Int, system::System, psp::Nothing)
    T_k = KineticBlock(pw::PlaneWaveBasis, idx_kpoint::Int)
    Vloc_k = LocalPotentialBlock(pw, idx_kpoint,
                                 G -> nuclear_attraction_generator_1e(system, G))
    HamiltonianBlock{Nothing}(pw, idx_kpoint, size(T_k), T_k, Vloc_k, nothing)
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
import Base: *
function *(H::HamiltonianBlock, B::Array)
    Y = similar(B)
    mul!(Y, H, B)
end
Base.size(H::HamiltonianBlock, idx::Int) = H.size[idx]
Base.size(H::HamiltonianBlock) = H.size
Base.eltype(H::HamiltonianBlock) = ComplexF64

struct Hamiltonian{NonLocalPotential}
    pw::PlaneWaveBasis
    blocks::Vector{HamiltonianBlock{NonLocalPotential}}
    ρ::Array{Float64, 3}
    potential_2e_real::Array{ComplexF64, 3}
end
function Hamiltonian(pw::PlaneWaveBasis, system::System,
                     psp::Union{PspHgh, Nothing};
                     ρguess::Union{Array{Float64, 3}, Nothing}=nothing)
    potential_2e_real = zeros(ComplexF64, pw.fft_size...)
    if ρguess == nothing
        ρguess = zeros(Float64, pw.fft_size...)
    else
        potential_2e_real = solve_poisson(pw::PlaneWaveBasis, ρguess)
        println("norm(imag(potential_2e_real)) == $(norm(imag(potential_2e_real)))")
        @assert norm(imag(potential_2e_real)) < 1e-12
    end
    # TODO density to block
    blocks = [HamiltonianBlock(pw, idx_kpoint, system, psp, potential_2e_real)
              for idx_kpoint in 1:length(pw.kpoints)]
    if psp == nothing
        Hamiltonian{Nothing}(pw, blocks, ρguess, potential_2e_real)
    else
        Hamiltonian{PspNonLocalBlock}(pw, blocks, ρguess, potential_2e_real)
    end
end
function solve_poisson(pw::PlaneWaveBasis, ρ::Array{Float64, 3})
    ρ_Fourier = R_to_G!(pw, copy(ρ))
    VHartree_Fourier = 4π*[ρ_Fourier[ig] / sum(abs2, G) for (ig, G) in enumerate(pw.Gs)]
    VHartree_Fourier[pw.idx_DC] = 0.0
    G_to_R(pw, VHartree_Fourier)
end
function substitute_density!(H::Hamiltonian, ρ::Array{Float64, 3})
    H.ρ[:] = ρ
    H.potential_2e_real[:] = solve_poisson(H.pw, ρ)
    @assert norm(imag(H.potential_2e_real)) < 1e-12
    for b in H.blocks
        b.Vloc_k.potential_2e_real[:] = real(H.potential_2e_real[:])
    end
    H
end

function lobpcg_full(H::Hamiltonian, largest::Bool, n_bands::Int;
                preconditioner=nothing, guess=nothing, kwargs...)
    n_k = length(H.blocks)
    res = Array{LOBPCGResults}(undef, n_k)
    for (ik, Hk) in enumerate(H.blocks)
        if preconditioner != nothing
            P = preconditioner(Hk)
        else
            P = nothing
        end
        if guess != nothing
            @assert length(guess) ≥ n_k
            @assert size(guess[ik], 2) == n_bands
            @assert size(guess[ik], 1) == size(Hk, 2)
            res[ik] = lobpcg(Hk, largest, guess[ik], P=P; kwargs...)
        else
            res[ik] = lobpcg(Hk, largest, n_bands, P=P; kwargs...)
        end
        println("lobpcg $ik niter: $(res[ik].iterations) converged: $(res[ik].converged)")
        @assert all(res[ik].converged)
    end
    res
end


function self_consistent_field!(H::Hamiltonian, bzmesh::BrilloinZoneMesh,
                                occupation::Vector{Float64};
                                n_bands::Int=sum(occupation .> 0),
                                PsiGuess=nothing, tol=1e-6)
    @assert n_bands ≥ length(occupation)

    if n_bands > length(occupation)
        occs = zeros(Float64, n_bands)
        occs[1:length(occupation)] = occupation
    else
        occs = occupation
    end
    println("#################\n#-- SCF start --#\n#################")
    println("occs: $occs")

    ene_old = NaN
    Psi = nothing
    if PsiGuess != nothing
        Psi = PsiGuess
        @assert size(PsiGuess[1], 2) == n_bands
        ene_old = compute_energy(H, bzmesh, Psi, occs)["total"]
        println("Starting ene:       $(ene_old)")
    end

    β_mix = 0.2
    ρ_old = H.ρ
    for i in 1:100
        println("\n#\n# ITER $i\n#")
        make_precond(Hk) = KineticPreconditionerBlock(Hk, α=0.1)
        largest = false
        res = lobpcg_full(H, largest, n_bands, tol=tol / 100,
                          guess=Psi, preconditioner=make_precond)
        Psi = [st.X for st in res]
        println("\nEvals:")
        for (i, st) in enumerate(res)
            println("$i  $(real(st.λ))")
        end

        ρ = compute_density_stupid(H.pw, bzmesh, Psi, occs)

        H = substitute_density!(H, ρ)
        ene = compute_energy(H, bzmesh, Psi, occupation)
        println("energy: kin=$(ene["kinetic"])  e2e=$(ene["e2e"])  " *
                "e1e_loc=$(ene["e1e_loc"])  e_nloc=$(ene["e_nloc"])")
        diff = ene["total"] - ene_old
        println("iter: $i   E=$(ene["total"])   ΔE=$diff")
        if abs(diff) < tol
            println("converged")
            return H, ene
        end

        if norm(ρ_old) > 0
            ρ = β_mix * H.ρ + (1 - β_mix) * ρ_old
            H = substitute_density!(H, ρ)
        end

        ene_old = ene["total"]
        ρ_old = ρ
    end
    return H, Dict(String, Float64)("total"=>ene_old)
end


Wavefunction = Vector{Matrix{ComplexF64}}
function compute_energy(H::Hamiltonian, bzmesh::BrilloinZoneMesh,
                        Psi::Wavefunction, occupation::Vector{Float64})
    @assert real(H.potential_2e_real) == H.blocks[1].Vloc_k.potential_2e_real
    @assert H.pw.kpoints == bzmesh.kpoints

    dVol = H.pw.unit_cell_volume / prod(H.pw.fft_size)
    e2e = 0.5 * sum(H.blocks[1].Vloc_k.potential_2e_real .* H.ρ) * dVol
    e1e_loc = sum(H.blocks[1].Vloc_k.potential_1e_real .* H.ρ) * dVol

    e_kin = 0.0
    for ik in 1:length(bzmesh.kpoints)
        Psi_k = Psi[ik]
        w_k = bzmesh.weights[ik]
        e_kin += w_k * occupation[ik] * tr(adjoint(Psi_k) * (H.blocks[ik].T_k * Psi_k))
    end

    e_nloc = 0.0
    if H.blocks[1].Vnloc_k != nothing
        # TODO One could be more clever about this and directly
        #      use the projections
        for ik in 1:length(bzmesh.kpoints)
            Psi_k = Psi[ik]
            w_k = bzmesh.weights[ik]
            e_nloc += (
                w_k * occupation[ik] * tr(adjoint(Psi_k) * (H.blocks[ik].Vnloc_k * Psi_k))
            )
        end
    end

    # TODO Nuclear repulsion and psp core energy

    total = e_kin + e2e + e1e_loc + e_nloc
    @assert imag(total) < 1e-12
    Dict{String, Float64}(
        "kinetic" => real(e_kin),
        "e2e"     => real(e2e),
        "e1e_loc" => real(e1e_loc),
        "e_nloc"  => real(e_nloc),
        "total"   => real(total)
    )
end

function purify_print(Mat; tol=1e-14)
    for ij in CartesianIndices(Mat)
        if abs(Mat[ij]) < 1e-14
            Mat[ij] = 0
        end
    end

    str = ""
    for i in 1:size(Mat, 1)
        for j in 1:size(Mat, 2)
            if Mat[i,j] < tol
                str *= @sprintf "%8.4g " 0
            else
                str *= @sprintf "%8.4g " Mat[i,j]
            end
        end
        str *= "\n"
    end
    str
end

function compute_density_stupid(pw::PlaneWaveBasis, bzmesh::BrilloinZoneMesh,
                                Psi::Wavefunction, occupation::Vector{Float64})
    n_fft = prod(pw.fft_size)
    n_k = length(bzmesh.kpoints)
    @assert n_k == length(Psi)
    for idx_kpoint in 1:n_k
        @assert length(pw.Gmask[idx_kpoint]) == size(Psi[idx_kpoint], 1)
        @assert length(occupation) == size(Psi[idx_kpoint], 2)
    end

    println("\n# density computation")
    ρ = zeros(Float64, pw.fft_size...)
    for idx_kpoint in 1:n_k
        println("-- idx_kpoint=$idx_kpoint")
        Ψ_k = Psi[idx_kpoint]
        idx_fft = pw.idx_to_fft[pw.Gmask[idx_kpoint]]
        weight = bzmesh.weights[idx_kpoint]
        n_states = size(Ψ_k, 2)

        # Fourier-transform the wave functions to real space
        Ψ_k_real = Array{ComplexF64}( undef, pw.fft_size..., n_states)
        for istate in 1:n_states
            Ψ_k_real[:, :, :, istate] = G_to_R(pw, Ψ_k[:, istate]; idx_to_fft=idx_fft)
        end

        # Orthonormalise in real space
        # TODO Why is this needed ?
        Ψ_k_real_mat = reshape(Ψ_k_real, n_fft, n_states)
        Udagger = (
              sqrt(n_fft / pw.unit_cell_volume)
            * inv(sqrt(adjoint(Ψ_k_real_mat) * Ψ_k_real_mat))
        )
        println("Udagger: \n$(purify_print(real(Udagger)))\n----")
        Ψ_k_real_mat = Ψ_k_real_mat * Udagger

        # Assert orthonormality of orbitals at k-point
        println("non-orthonormality: ",
                maximum(abs.(adjoint(Ψ_k_real_mat) * Ψ_k_real_mat  - (n_fft / pw.unit_cell_volume) * I)))
        @assert maximum(abs.(adjoint(Ψ_k_real_mat) * Ψ_k_real_mat
                             - (n_fft / pw.unit_cell_volume) * I)) < 1e-10
        Ψ_k_real = reshape(Ψ_k_real_mat, pw.fft_size..., n_states)

        # Add the density from this kpoint
        for istate in 1:n_states
            # TODO Assert imaginary part is not too large!
            ρ .+= (weight * occupation[istate]
                   * real(Ψ_k_real[:, :, :, istate] .* conj(Ψ_k_real[:, :, :, istate]))
            )
        end
    end

    @assert maximum(imag(ρ)) < 1e-12
    ρ = real(ρ)

    # Ensure that there is no negative ρ element
    # TODO Why is this needed?
    for ijk in CartesianIndices(ρ)
        ρ[ijk] = max(ρ[ijk], 0)
    end


    # Renormalize rho
    # TODO Why is this needed?
    n_electrons_integrated = sum(ρ) * pw.unit_cell_volume / prod(pw.fft_size)
    println("n_electrons_integrated:  $(n_electrons_integrated)")
    #@assert abs(n_electrons_integrated - sum(occupation)) < 1e-12
    ρ = sum(occupation) / n_electrons_integrated * ρ

    println("# end density computation\n")

    ρ
end


"""
Kinetic-energy based preconditioner.
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
function compute_bands(pw::PlaneWaveBasis, system::System,
                       potential_2e_real::Array{ComplexF64, 3};
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
        H = HamiltonianBlock(pw, idx_kpoint, system, psp, potential_2e_real)
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
    ref_noHartree_noXC = [
        [0.067966083141126, 0.470570565964348, 0.470570565966131,
         0.470570565980086, 0.578593208202602],
        [0.105959302042882, 0.329211057388161, 0.410969129077501,
         0.451613404615669, 0.626861886537186],
        [0.158220020418481, 0.246761395395185, 0.383362969225928,
         0.422345289771740, 0.620994908900183],
        [0.138706889457309, 0.256605657080138, 0.431494061152506,
         0.437698454692923, 0.587160336593700]
    ]
    ref_noXC = [
        [0.185538688128930, 0.639782864332682, 0.646228816341722,
         0.652579750538309, 0.702926833017312],
        [0.232356507255950, 0.479585478143240, 0.515892938933947,
         0.614879052936140, 0.726232570409365],
        [0.306640270578223, 0.373925930428464, 0.479563402554229,
         0.560825484067980, 0.664124433312495],
        [0.284385454518986, 0.359761906442770, 0.577352003566477,
         0.588385362008877, 0.688717726672163]
    ]

    Z = 14
    a = 5.431020504 * ÅtoBohr
    silicon = build_diamond_system(a, Z)
    occupation = [2.0, 2.0, 2.0, 2.0]
    bzmesh = build_diamond_bzmesh(silicon)

    Ecut = 15  # Hartree
    pw = PlaneWaveBasis(silicon, bzmesh.kpoints, Ecut)
    psp = PspHgh("./psp/CP2K-pade-Si-q4.hgh")

    # Check on the Hamiltonian
    opH = HamiltonianBlock(pw, 1, silicon, psp)
    H = Matrix{ComplexF64}(undef, size(opH))
    mul!(H, opH, Matrix{ComplexF64}(I, size(opH)))
    @assert maximum(abs.(H' - H)) < 1e-14
    @assert maximum(abs.(imag(H))) < 1e-12

    Ham = Hamiltonian(pw, silicon, psp)
    Ham, ene = self_consistent_field!(Ham, bzmesh, occupation, n_bands=4, tol=1e-8)
    potential_2e_real = solve_poisson(pw, Ham.ρ)

    pw = substitute_kpoints(pw, kpoints)
    λs, vs = compute_bands(pw, silicon, potential_2e_real, psp=psp, n_bands=5)
    ref = ref_noXC
    for i in 1:length(ref)
        println(λs[i] - ref[i])
        @assert maximum(abs.(ref[i] - λs[i])[1:4]) < 1e-4
        @assert maximum(abs.(ref[i] - λs[i])) < 1e-3
    end

    @assert abs(ene["total"]   -  3.1661644264) < 1e-5
    @assert abs(ene["e1e_loc"] - -1.7783908803) < 1e-4
    @assert abs(ene["kinetic"] -  3.0074897969) < 1e-4
    @assert abs(ene["e_nloc"]  -  1.5085540922) < 1e-4
    @assert abs(ene["e2e"]     -  0.4285114176) < 1e-4
end


function bands_silicon()
    #
    # Setup system (Silicon) and model (Plane Wave basis)
    #
    Z = 14
    a = 5.431020504 * ÅtoBohr
    silicon = build_diamond_system(a, Z)
    occupation = [2.0, 2.0, 2.0, 2.0]
    bzmesh = build_diamond_bzmesh(silicon)

    Ecut = 25  # Hartree
    pw = PlaneWaveBasis(silicon, bzmesh.kpoints, Ecut)
    psp = PspHgh("./psp/CP2K-pade-Si-q4.hgh")

    # Run SCF to minimise wrt. density and get final 2e potential
    H = Hamiltonian(pw, silicon, psp)
    H, ene = self_consistent_field!(H, bzmesh, occupation, n_bands=8, tol=1e-6)
    potential_2e_real = solve_poisson(pw, H.ρ)

    #
    # Compute and plot bands
    #
    path = [(:L, :Γ), (:Γ, :X), (:X, :U), (:K, :Γ)]
    # path = [(:Γ, :X), (:X, :W), (:W, :K), (:K, :Γ), (:Γ, :L),
    #         (:L, :U), (:U, :W), (:W, :L), (:L, :K)]
    n_kpoints = 7
    kpath = BrilloinZonePath(silicon, path, n_kpoints)

    # Form new pw basis with the kpoints for above path
    pw = substitute_kpoints(pw, kpath.kpoints)
    println("FFT grid size: $(prod(pw.fft_size))")

    # Compute bands and plot
    λs, vs = compute_bands(pw, silicon, potential_2e_real, psp=psp, n_bands=15)
    plot_quantity(kpath, λs)
    savefig("bands_Si.pdf", bbox_inches="tight")
end

# TODO Put structure-factors into PlaneWaveBasis (or maybe some precomuted data object)

main() = bands_silicon()

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
