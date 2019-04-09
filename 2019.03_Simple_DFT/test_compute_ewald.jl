using LinearAlgebra
using FFTW
include("System.jl")
include("compute_ewald.jl")
include("Units.jl")





# The Madelung constant is defined as
#    γ_E = -α (Z_i Z_j)^2 / (2R)
# where (2R) is the nearest-neighbour distance for ionic crystals
# and R = R_\text{ws} for elemental crystals



# Madelung α for selected cases
α_NaCl = 1.74757
α_wurtzite = 1.63870
α_zincblende = 1.63806
α_bcc = 1.79186
α_fcc = 1.79175
α_hcp = 1.79168
α_sc = 1.76012
α_diamond = 1.67085


function test_diamond_silicon(η=nothing)
    Z = 14
    a = 5.431020504 * ÅtoBohr
    silicon = build_diamond_system(a, Z)

    γ_E = compute_ewald(silicon, η=η)
    α = - γ_E / Z^2 * 2 * a

    println("Madelung from γ_E: $α")
    println("Madelung lit:      $α_diamond  (reference)")

    ref = -102.8741963352893
    println()
    println("Ewald sum γ_E:     $γ_E")
    println("Ewald sum ref:     $ref")
    println("Ewald sum   Δ:     $(γ_E - ref)")
    @assert abs(γ_E - ref) < 1e-8
end


function test_N2(η=nothing)
    atoms = ÅtoBohr .* [[0., 0, 0], [1.25, 0, 0]]
    Zs = [5.0, 5.0]
    nitrogen = build_system_simple_cubic(16.0, atoms, Zs)
    γ_E = compute_ewald(nitrogen, η=η)

    ref = 1.790634595  # TODO source?
    println("Ewald sum γ_E:     $γ_E")
    println("Ewald sum ref:     $ref")
    println("Ewald sum   Δ:     $(γ_E - ref)")
    @assert abs(γ_E - ref) < 1e-7
end


function test_H2(η=nothing)
    atoms = ÅtoBohr .* [
        [3.83653478, 4.23341768, 4.23341768],
        [4.63030059, 4.23341768, 4.23341768],
    ]
    Zs = [1.0, 1.0]
    hydrogen = build_system_simple_cubic(16.0, atoms, Zs)
    γ_E = compute_ewald(hydrogen, η=η)

    ref = 0.31316999  # TODO source?
    println("Ewald sum γ_E:     $γ_E")
    println("Ewald sum ref:     $ref")
    println("Ewald sum   Δ:     $(γ_E - ref)")
    @assert abs(γ_E - ref) < 1e-7
end


function run_pwdft_H2(η=nothing)
    atoms = init_atoms_xyz_string(
        """
        2

        H      3.83653478       4.23341768       4.23341768
        H      4.63030059       4.23341768       4.23341768
        """
    )
    atoms.LatVecs = gen_lattice_sc(16.0)
    atoms.Zvals = [1.0, 1.0]

    eta = nothing
    if η != nothing
        eta = 4η^2
    end
    ewald_pwdft = calc_E_NN(atoms, eta=eta)
    ref = 0.31316999  # TODO source?
    println("Ewald sum γ_E:     $ewald_pwdft")
    println("Ewald sum ref:     $ref")
    println("Ewald sum   Δ:     $(ewald_pwdft - ref)")
    @assert abs(γ_E - ref) < 1e-8
end


function test_LiH(η=nothing)
    atoms = ÅtoBohr .* [
        [4.23341768, 4.23341768, 5.04089768],
        [4.23341768, 4.23341768, 3.42593768],
    ]
    Zs = [1.0, 1.0]
    lih = build_system_simple_cubic(16.0, atoms, Zs)
    γ_E = compute_ewald(lih, η=η)

    ref = -0.02196861  # TODO source?
    println("Ewald sum γ_E:     $γ_E")
    println("Ewald sum ref:     $ref")
    println("Ewald sum   Δ:     $(γ_E - ref)")
    @assert abs(γ_E - ref) < 1e-8
end


function test_H(η=nothing)
    atoms = [[0.0, 0, 0]]
    Zs = [1.0]
    hydrogen = build_system_simple_cubic(16.0, atoms, Zs)
    γ_E = compute_ewald(hydrogen, η=η)

    ref = -0.088665545  # TODO source?
    println("Ewald sum γ_E:     $γ_E")
    println("Ewald sum ref:     $ref")
    println("Ewald sum   Δ:     $(γ_E - ref)")
    @assert abs(γ_E - ref) < 1e-8
end
