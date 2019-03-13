# TODO More spglib in this

"""
KPoint mesh to employ
"""
struct KPoints
    """
    k-Points to use for the BZ sampling
    """
    kpoints::Vector{Vector{Float64}}

    """
    Accumulated distance of the kpoints
    for plotting of bands
    """
    accumulated_kdistance::Vector{Float64}

    # TODO Will probably be needed later
    # """
    # Integration weight for each k-Point
    # """
    # weight::Vector{Float64}

    # TODO This is not really followd right now
    # """
    # k-Point mesh employed
    # """
    # mesh::Tuple{Int64,Int64,Int64}

    """
    Map of indices which are high-symmetry points
    to their high-symmetry point labels
    """
    labels::Dict{Int64, Symbol}

    """
    The system, which was used to generate these k-points
    """
    system::System
end

"""
Given a system, a list of tuples of high-symmetry points as the plotting
path and the number of mesh points in each direction of a uniform mesh
in reciprocal space, build a KPoints object.
"""
function KPoints(system::System, path::Array{Tuple{Symbol,Symbol}}, nkpoints::Int64)
    # Transform path to actual coordinates
    plot_path = [(system.high_sym_points[p[1]], system.high_sym_points[p[2]])
                 for p in path]

    # Factorise B, since we'll need it a lot in here
    Bfac = factorize(system.B)

    
    labels = Dict{Int64, Symbol}()
    kpoints = []
    accumulated_kdistance = [0.]
    for (st, en) in path
        stkp = system.high_sym_points[st]
        enkp = system.high_sym_points[en]
        labels[length(kpoints) + 1] = st

        kdiff = (enkp - stkp) / nkpoints
        newkpoints = [stkp .+ fac .* kdiff for fac in 0:nkpoints-1]
        append!(kpoints, newkpoints)
        append!(accumulated_kdistance,
                accumulated_kdistance[end] .+ norm(kdiff) * collect(1:nkpoints))
    end
    labels[length(kpoints) + 1] = path[end][end]
    push!(kpoints, system.high_sym_points[path[end][end]])
    @assert length(kpoints) == length(accumulated_kdistance)

    KPoints(kpoints, accumulated_kdistance, labels, system)
end

"""
Given a list of vectors of the same length, return the accumlated distance
between them.
"""
function accumulated_distance(vecs::Vector{Vector{T}} where T <: Number)
    @assert length(vecs) > 1
    distances = [norm(vecs[i] - vecs[i + 1]) for i in 1:length(vecs) - 1]
    vcat([0], accumulate(+, distances))
end
