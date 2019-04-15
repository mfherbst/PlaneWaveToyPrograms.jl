using FFTW
using LinearAlgebra
using Printf
include("SphericalHarmonics.jl")
include("System.jl")
include("BrilloinZonePath.jl")
include("BrilloinZoneMesh.jl")
include("PlaneWaveBasis.jl")
include("Units.jl")
include("PspHgh.jl")
include("libxc.jl")
include("blocks.jl")
include("compute_ewald.jl")
using PyPlot
using ProgressMeter
using IterativeSolvers
import IterativeSolvers: LOBPCGResults

# TODO qsq should only be computed once for each k

#
# Hamiltonian
#
Wavefunction = Vector{Matrix{ComplexF64}}
FunctionalXC = Vector{Functional}


struct Hamiltonian{NonLocalPotential}
    system::System
    pw::PlaneWaveBasis
    blocks::Vector{HamiltonianBlock{NonLocalPotential}}
    ρ::Array{Float64, 3}
    xc::FunctionalXC
    potential_2e_real::Array{ComplexF64, 3}
    Zs::Vector{Float64}  # Charges for each atom
end


function Hamiltonian(pw::PlaneWaveBasis, system::System,
                     psp::Union{PspHgh, Nothing},
                     xc::FunctionalXC;
                     ρguess::Union{Array{Float64, 3}, Nothing}=nothing)
    potential_2e_real = zeros(ComplexF64, pw.fft_size...)
    if ρguess == nothing
        ρguess = zeros(Float64, pw.fft_size...)
    else
        potential_2e_real = solve_poisson(pw, ρguess) + compute_xc(xc, ρguess)
        println("norm(imag(potential_2e_real)) == $(norm(imag(potential_2e_real)))")
        @assert norm(imag(potential_2e_real)) < 1e-12
    end
    # TODO density to block
    blocks = [HamiltonianBlock(pw, idx_kpoint, system, psp, potential_2e_real)
              for idx_kpoint in 1:length(pw.kpoints)]
    if psp == nothing
        Hamiltonian{Nothing}(system, pw, blocks, ρguess, xc,
                             potential_2e_real, system.Zs)
    else
        Zs = psp.Zion .* ones(size(system.Zs))
        Hamiltonian{PspNonLocalBlock}(system, pw, blocks, ρguess, xc,
                                      potential_2e_real, Zs)
    end
end


function solve_poisson(pw::PlaneWaveBasis, ρ::Array{Float64, 3})
    ρ_Fourier = R_to_G!(pw, copy(ρ))
    VHartree_Fourier = 4π*[ρ_Fourier[ig] / sum(abs2, G) for (ig, G) in enumerate(pw.Gs)]
    VHartree_Fourier[pw.idx_DC] = 0.0
    G_to_R(pw, VHartree_Fourier)
end


function compute_xc(xc::FunctionalXC, ρ::Array{Float64, 3})
    accu = zeros(Float64, size(ρ)...)
    for i in 1:length(xc)
        if xc[i].family == FunctionalFamily(1)
            accu += evaluate_lda_potential(xc[i], ρ)
        else
            error("Not implemented, functional family $(xc[i].family)")
        end
    end
    accu
end


function substitute_density!(H::Hamiltonian, ρ::Array{Float64, 3})
    H.ρ[:] = ρ
    H.potential_2e_real[:] = solve_poisson(H.pw, ρ) + compute_xc(H.xc, ρ)
    @assert norm(imag(H.potential_2e_real)) < 1e-12
    for b in H.blocks
        b.Vloc_k.potential_2e_real[:] = real(H.potential_2e_real[:])
    end
    H
end


#
# LOBPCG preconditioner
#
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
# SCF
#
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


function compute_density_stupid(pw::PlaneWaveBasis, bzmesh::BrilloinZoneMesh,
                                Psi::Wavefunction, occupation::Vector{Float64};
                                lobpcg_tol=1e-6)
    n_fft = prod(pw.fft_size)
    n_k = length(bzmesh.kpoints)
    @assert n_k == length(Psi)
    for idx_kpoint in 1:n_k
        @assert length(pw.Gmask[idx_kpoint]) == size(Psi[idx_kpoint], 1)
        @assert length(occupation) == size(Psi[idx_kpoint], 2)
    end

    ρ = zeros(ComplexF64, pw.fft_size...)
    for idx_kpoint in 1:n_k
        Ψ_k = Psi[idx_kpoint]
        idx_fft = pw.idx_to_fft[pw.Gmask[idx_kpoint]]
        weight = bzmesh.weights[idx_kpoint]
        n_states = size(Ψ_k, 2)

        # Fourier-transform the wave functions to real space
        Ψ_k_real = zeros(ComplexF64, pw.fft_size..., n_states)
        for istate in 1:n_states
            Ψ_k_real[:, :, :, istate] = G_to_R(pw, Ψ_k[:, istate]; idx_to_fft=idx_fft)
        end

        # TODO I am not quite sure why this is needed here
        #      maybe this points at an error in the normalisation of the
        #      Fourier transform
        Ψ_k_real /= sqrt(pw.unit_cell_volume)

        # Check for orthonormality of the Ψ_k_reals
        Ψ_k_real_mat = reshape(Ψ_k_real, n_fft, n_states)
        Ψ_k_real_overlap = adjoint(Ψ_k_real_mat) * Ψ_k_real_mat
        max_nondiag = maximum(abs.(Ψ_k_real_overlap - I * (n_fft / pw.unit_cell_volume)))
        @assert max_nondiag < lobpcg_tol

        # Add the density from this kpoint
        for istate in 1:n_states
            ρ .+= (weight * occupation[istate]
                   * Ψ_k_real[:, :, :, istate] .* conj(Ψ_k_real[:, :, :, istate])
            )
        end
    end

    # Check ρ is real and positive and properly normalised
    @assert maximum(imag(ρ)) < 1e-12
    ρ = real(ρ)
    @assert minimum(ρ) ≥ 0

    n_electrons = sum(ρ) * pw.unit_cell_volume / n_fft
    @assert abs(n_electrons - sum(occupation)) < 1e-9

    ρ
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

    ene_old = NaN
    Psi = nothing
    if PsiGuess != nothing
        Psi = PsiGuess
        @assert size(PsiGuess[1], 2) == n_bands
        ene_old = compute_energy(H, bzmesh, Psi, occs)["total"]
        println("Starting ene:       $(ene_old)")
    end

    β_mix = 0.3
    ρ_old = H.ρ
    for i in 1:100
        println("#\n# SCF iter $i\n#")
        make_precond(Hk) = KineticPreconditionerBlock(Hk, α=0.1)
        largest = false
        res = lobpcg_full(H, largest, n_bands, tol=tol / 100,
                          guess=Psi, preconditioner=make_precond)
        Psi = [st.X for st in res]
        println("\nlobpcg evals:")
        for (i, st) in enumerate(res)
            println("$i  $(real(st.λ))")
        end

        ρ = compute_density_stupid(H.pw, bzmesh, Psi, occs, lobpcg_tol=tol / 100)
        H = substitute_density!(H, ρ)
        ene = compute_energy(H, bzmesh, Psi, occupation)

        # Display convergence
        diff = ene["total"] - ene_old
        println()
        for key in keys(ene)
            if key != "total"
                @printf("%10s =  %16.10g\n", key, ene[key])
            end
        end
        @printf("%10s =  %16.10g\n", "total", ene["total"])
        @printf("%10s =  %16.10g\n", "Δtotal", diff)
        if abs(diff) < tol
            println("\n#\n#-- SCF converged\n#")
            return H, ene
        end

        if norm(ρ_old) > 0
            ρ = β_mix * H.ρ + (1 - β_mix) * ρ_old
            H = substitute_density!(H, ρ)
        end

        ene_old = ene["total"]
        ρ_old = ρ

        println()
    end
    return H, Dict(String, Float64)("total" => ene_old)
end


#
# Energy
#
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
        e_kin += w_k * tr(occupation[ik] * adjoint(Psi_k) * (H.blocks[ik].T_k * Psi_k))
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

    e_nuclear = compute_ewald(H.system, Zs=H.Zs)

    # TODO psp core energy

    total = e_kin + e2e + e1e_loc + e_nloc + e_nuclear
    @assert imag(total) < 1e-12
    Dict{String, Float64}(
        "e_nuclear" => e_nuclear,
        "kinetic" => real(e_kin),
        "e2e"     => real(e2e),
        "e1e_loc" => real(e1e_loc),
        "e_nloc"  => real(e_nloc),
        "total"   => real(total),
    )
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
    ref = [
        [-0.178072428810715, 0.261616125031914, 0.262784140913747,
          0.263316413356142, 0.353800852071277],
        [-0.133973089764325, 0.106057092768291, 0.158523970429309,
          0.236043055254138, 0.369766606388843],
        [-0.068165395186365, 0.010628148024691, 0.122579725282070,
          0.190369766912598, 0.325184211947709],
        [-0.089624321030248, 0.004334893864463, 0.203685563865404,
          0.214828905505821, 0.327528699634624]
    ]

    ene_ref_noXC = Dict{String, Float64}(
        "e1e_loc" => -1.7783908803,
        "kinetic" =>  3.0074897969,
        "e_nloc"  =>  1.5085540922,
        "e2e"     =>  0.4285114176,
        "total"   =>  3.1661644264,
    )
    ene_ref = Dict{String, Float64}(
        "e1e_loc" => -2.1757127578,
        "kinetic" =>  3.2107081817,
        "e_nloc"  =>  1.5804553245,
        "e2e"     => -1.8327072303,
        "total"   =>  0.7827435181,
    )

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

    xc = Functional.(["lda_x", "lda_c_vwn"])
    Ham = Hamiltonian(pw, silicon, psp, xc)
    Ham, ene = self_consistent_field!(Ham, bzmesh, occupation, n_bands=4, tol=1e-8)
    potential_2e_real = solve_poisson(pw, Ham.ρ) + compute_xc(xc, Ham.ρ)

    pw = substitute_kpoints(pw, kpoints)
    λs, vs = compute_bands(pw, silicon, potential_2e_real, psp=psp, n_bands=5)

    # ref = ref_noXC
    for i in 1:length(ref)
        println("eval $i ", λs[i] - ref[i])
        @assert maximum(abs.(ref[i] - λs[i])[1:4]) < 5e-5
        @assert maximum(abs.(ref[i] - λs[i])) < 1e-3
    end

    # TODO I think there is still some issue with the energy computation
    #      because XC and Hartree energy should use different density prefactors
    #      (e.g. Hartree should use α + β density and XC only α density

    # ene_ref = ene_ref_noXC
    for k in keys(ene_ref)
        @printf("%8s  %16.10g\n", k, abs(ene[k] - ene_ref[k]))
        if k == "e1e_loc"
            @assert abs(ene[k] - ene_ref[k]) < 5e-4
        else
            @assert abs(ene[k] - ene_ref[k]) < 1e-4
        end
    end
    @assert abs(ene["total"] - ene_ref["total"]) < 1e-6
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
    println("FFT grid size: $(prod(pw.fft_size))")
    psp = PspHgh("./psp/CP2K-pade-Si-q4.hgh")

    # Run SCF to minimise wrt. density and get final 2e potential
    xc = Functional.(["lda_x", "lda_c_vwn"])
    H = Hamiltonian(pw, silicon, psp, xc)
    H, ene = self_consistent_field!(H, bzmesh, occupation, n_bands=8, tol=1e-6)
    potential_2e_real = solve_poisson(pw, H.ρ) + compute_xc(H.xc, H.ρ)

    #
    # Compute and plot bands
    #
    # path = [(:L, :Γ), (:Γ, :X), (:X, :U), (:K, :Γ)]
    path = [(:Γ, :X), (:X, :W), (:W, :K), (:K, :Γ), (:Γ, :L),
            (:L, :U), (:U, :W), (:W, :L), (:L, :K)]
    n_kpoints = 7
    kpath = BrilloinZonePath(silicon, path, n_kpoints)

    # Form new pw basis with the kpoints for above path
    pw = substitute_kpoints(pw, kpath.kpoints)

    # Compute bands and plot
    λs, vs = compute_bands(pw, silicon, potential_2e_real, psp=psp, n_bands=15)
    plot_quantity(kpath, λs)
    savefig("bands_Si.pdf", bbox_inches="tight")

    return kpath, λs
end

function dump_bandstructure(kpath, λs, file)
    open(file, "w") do fp
        for (i, acculen) in enumerate(kpath.accumulated_arclength)
            @assert length(λs[i]) ≥ 5
            λ = λs[i][1:5]
            @printf(fp, "%20.12f%20.12f%20.12f%20.12f%20.12f%20.12f\n", acculen, λ...)
        end
    end
end

main() = bands_silicon()

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
