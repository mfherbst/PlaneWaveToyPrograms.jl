#!/usr/bin/env julia

using PWDFT
include("CorePseudo.jl")

# This file tests main.jl against PWDFT

function main()
    #
    # Setup silicon structure and model for CorePseudo
    #
    Z = 14
    a = 5.431020504 * angströmToBohr

    Ecut= 83 * (2π / a)^2    # Cutoff for PW basis in CorePseudo

    # Build structure and PW model
    S = build_diamond_system(a, Z)
    psp = PspHgh(Z)
    M = Model(S, psp, Ecut)

    #
    # Setup PWDFT, using our coords
    #
    atoms = init_atoms_xyz_string(
        """
        2

        Si  0.0  0.0  0.0
        Si  0.25  0.25  0.25
        """, in_bohr=true)
    atoms.LatVecs = gen_lattice_fcc(10.2631)
    atoms.LatVecs = Array(S.A)
    atoms.positions = zeros(3, length(S.atoms))
    for i in 1:length(S.atoms)
        atoms.positions[:,i] = Vector(S.atoms[i])
    end

    PWDFT_cut = 15.0         # This is equal to the above Ecut for CorePseudo
    pspPWfile = "../../Codes/PWDFT.jl/pseudopotentials/pade_gth/Si-q4.gth"
    pspPW = PsPot_GTH(pspPWfile)

    kpoints = KPoints(atoms)
    ik = 1
    kpoints.k[:, ik] = [0,0,0]
    @assert length(kpoints.k[1, :]) == 1

    #
    # Setup PWDFT plane-wave grid and copy kpoints and g vectors to CorePseudo
    #
    pw = PWGrid(PWDFT_cut, atoms.LatVecs, kpoints=kpoints)
    @assert length(pw.gvecw.idx_gw2g) == 1
    M.Gs = [SVector{3}(pw.gvec.G[:,i]) for i in pw.gvecw.idx_gw2g[ik]]

    #
    # local pseudo-potential test
    #
    function PWDFT_psp_local(k)
        pw.gvecw.kpoints.k[:, ik] = k
        Ngw = pw.gvecw.Ngw[ik]

        Ham = Hamiltonian(atoms, [pspPWfile], 0/0, kpoints=pw.gvecw.kpoints,
                          pw=pw)
        Ham.ik = ik

        Psi_unit = Matrix{ComplexF64}(I, Ngw, Ngw)
        return op_V_Ps_loc(Ham, Psi_unit)
    end
    k = [0.2,0.4,0.1]
    Vloc_ref = PWDFT_psp_local(k)
    Vloc_this = build_potential(M, G -> pot_generator_psp_loc(S, M.psp, G))
    @assert Vloc_ref ≈ Vloc_this

    #
    # Test non-local pseudopotential projection vector expressions
    #
    for l in 0:M.psp.lmax
        hp = M.psp.h[l + 1]
        for i in 1:size(hp)[1]
            for G in M.Gs
                qsq = sum(abs2, G)
                this = eval_projection_vector(M.psp, i, l, qsq) / sqrt(S.unit_cell_volume)
                ref = eval_proj_G(pspPW, l, i, sqrt(qsq), S.unit_cell_volume)
                @assert this ≈ ref
            end
        end
    end
    # end test projection vector
    #
    # Test nonlocal pseudopotentials
    #
    function PWDFT_psp_nonlocal(k)
        pw.gvecw.kpoints.k[:, ik] = k
        Ngw = pw.gvecw.Ngw[ik]

        Ham = Hamiltonian(atoms, [pspPWfile], 0/0, kpoints=pw.gvecw.kpoints,
                          pw=pw)
        Ham.ik = ik

        Psi_unit = Matrix{ComplexF64}(I, Ngw, Ngw)
        op_V_Ps_nloc(Ham, Psi_unit)
    end
    k = [0.2,0.4,0.1]
    Vthis = pot_psp_nloc_fourier(S, M, M.psp, k)
    Vref = PWDFT_psp_nonlocal(k)
    @assert Vref ≈ Vthis

    #
    # Test values at three points
    #
    n_bands = 5

    kpts = [0.000000,0.000000,0.000000]
    ref = [0.0679660831, 0.4705705660, 0.4705705660, 0.4705705660, 0.5785932082]
    Hk = hamiltonian_fourier(S, M, kpts)
    ev, _ = eigen(Symmetric(Hk), 1:n_bands)
    @assert ev[1:4] ≈ ref[1:4]
    @assert abs(ev[5] - ref[5]) < 1e-4

    kpts = [0.229578,0.229578,0.000000]
    ref = [0.1059593020, 0.3292110574, 0.4109691291, 0.4516134046, 0.6268618865]
    Hk = hamiltonian_fourier(S, M, kpts)
    ev, _ = eigen(Symmetric(Hk), 1:n_bands)
    @assert maximum(abs.(ev - ref)) < 1e-4

    kpts = [0.401762,0.401762,0.114789]
    ref = [0.1582200204, 0.2467613954, 0.3833629692, 0.4223452898, 0.6209949089]
    Hk = hamiltonian_fourier(S, M, kpts)
    ev, _ = eigen(Symmetric(Hk), 1:n_bands)
    @assert maximum(abs.(ev - ref)[1:4]) < 1e-4
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
