# TODO Determine crystal symmetry and other things using spglib


"""
System to be modelled
"""
struct System
    """3x3 lattice vectors, in columns. |Γ| = det(A)"""
    A::Matrix{Float64}   # TODO rename e.g. metric_direct

    """List of atomic positions in the unit cell"""
    atoms::Vector{Vector{Float64}}

    """charge of each atom"""
    Zs::Vector{Float64}

    # Derived quantities
    """Reciprocal lattice"""
    B::Matrix{Float64}   # TODO rename e.g. metric_reciprocal

    """Volume of the unit cell"""
    unit_cell_volume::Float64

    """High-symmetry points in the reciprocal lattice"""
    high_sym_points::Dict{Symbol, Vector{Float64}}

    """A nice human-readable descriptor for the system"""
    description::String
end

"""
Constructor for the system

A      Lattice vectors as columns
atoms  Atom positions
Z      Atom change (communal to all)
fft_supersampling   Supersampling used during FFT
"""
function System(A, atoms, Zs, high_sym_points, description)
    B = 2π * inv(Array(A'))
    unit_cell_volume = det(A)
    System(A, atoms, Zs, B, unit_cell_volume, high_sym_points, description)
end

"""
Construct a system of the diamond structure
"""
function build_system_diamond(a, Z)
    A = a / 2 .* [[0 1 1.]
                  [1 0 1.]
                  [1 1 0.]]
    τ = a / 8 .* [1, 1, 1]
    atoms = [τ, -τ]
    Zs = [Z, Z]

    high_sym_points=Dict{Symbol, Vector{Float64}}(
        :Γ => 2π/a .* [  0,   0,   0],
        :X => 2π/a .* [  0,   1,   0],
        :W => 2π/a .* [1/2,   1,   0],
        :K => 2π/a .* [3/4, 3/4,   0],
        :L => 2π/a .* [1/2, 1/2, 1/2],
        :U => 2π/a .* [1/4,   1, 1/4],
    )
    System(A, atoms, Zs, high_sym_points, "diamond")
end


function build_system_sc(a, atoms::Vector{Vector{Float64}},
                         Zs::Vector{Float64})
    A = Matrix(a * I, 3, 3)
    high_sym_points=Dict{Symbol, Vector{Float64}}(
        :Γ => 2π/a .* [  0,   0,   0],
        :R => 2π/a .* [1/2, 1/2, 1/2],
        :X => 2π/a .* [  0, 1/2,   0],
        :M => 2π/a .* [1/2, 1/2,   0],
    )
    System(A, atoms, Zs, high_sym_points, "simple_cubic")
end

build_diamond_system = build_system_diamond
build_system_simple_cubic = build_system_sc
