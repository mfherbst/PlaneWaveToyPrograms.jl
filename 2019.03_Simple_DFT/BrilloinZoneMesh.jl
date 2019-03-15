# TODO More spglib in this

"""
Brilloin zone mesh to employ for kpoint sampling
"""
struct BrilloinZoneMesh
    """
    The system to which this mesh refers to
    """
    system::System

    """
    uniform mesh before reduction by symmetry
    """
    mesh::Tuple{Int64,Int64,Int64}

    """
    coordinates of the k-Points to use for BZ sampling
    """
    kpoints::Vector{Vector{Float64}}

    """
    Integration weight for each kpoint above
    """
    weights::Vector{Float64}

    """
    Arc distance between neighbouring kpoints of the
    kpoints path in reciprocal space implied by above vector.

    Notice that is usually (but not always) the l2-norm of
    the difference between adjacent kpoints.
    """
    arclength::Vector{Float64}
end

function build_diamond_bzmesh(system::System)
    @assert system.description == "diamond"

    mesh = (3,3,3)
    kpoints = [
        [  0.0000000000,  0.0000000000,  0.0000000000],
        [ -0.3333333333,  0.3333333333,  0.3333333333],
        [ -0.0000000000,  0.0000000000,  0.6666666667],
        [  0.6666666667, -0.6666666667, -0.0000000000],
    ]
    weights = [0.0370370370, 0.2962962963, 0.2222222222, 0.4444444444]
    arclength = norm.(diff(kpoints))
    BrilloinZoneMesh(system, mesh, kpoints, weights, arclength)
end


