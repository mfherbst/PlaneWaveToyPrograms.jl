struct BrilloinZonePath
    """
    The system to which this mesh refers to
    """
    system::System

    """
    coordinates of the k-Points to use for BZ sampling
    """
    kpoints::Vector{Vector{Float64}}

    """
    Accumulated arclength in reciproacal space following
    above kpoints path.
    """
    accumulated_arclength::Vector{Float64}

    """
    Map from indices to the symbols identifying them
    """
    labels::Dict{Int64, Symbol}
end

"""
Given a system, a list of tuples of high-symmetry points as the plotting
path and the number of mesh points to use on each segment, compute the list
of kpoints where computation needs to accurr along with the accumulated
arc length in reciprocal space following this paths.
"""
function BrilloinZonePath(system::System, path::Array{Tuple{Symbol,Symbol}},
                          n_kpoints::Int64)
    # Transform path to actual coordinates
    plot_path = [(system.high_sym_points[p[1]], system.high_sym_points[p[2]])
                 for p in path]

    labels = Dict{Int64, Symbol}()
    kpoints = []
    accumulated_arclength = [0.]
    for (st, en) in path
        stkp = system.high_sym_points[st]
        enkp = system.high_sym_points[en]
        labels[length(kpoints) + 1] = st

        kdiff = (enkp - stkp) / n_kpoints
        newkpoints = [stkp .+ fac .* kdiff for fac in 0:n_kpoints-1]
        append!(kpoints, newkpoints)
        append!(accumulated_arclength,
                accumulated_arclength[end] .+ norm(kdiff) * collect(1:n_kpoints))
    end
    labels[length(kpoints) + 1] = path[end][end]
    push!(kpoints, system.high_sym_points[path[end][end]])
    @assert length(kpoints) == length(accumulated_arclength)

    BrilloinZonePath(system, kpoints, accumulated_arclength, labels)
end

"""
Plot a quantity or a list of quantities (e.g. energies)
each kpoint of a BrilloinZonePath
"""
function plot_quantity(kpath::BrilloinZonePath,
                       values::Vector{Vector{T}} where T <: Number)
    n_ks = length(kpath.kpoints)
    n_bands = minimum(length.(values))
    kdistance = kpath.accumulated_arclength

    figure()
    for ib in 1:n_bands
        plot(kdistance, [values[ik][ib] for ik in 1:n_ks], "rx-")
    end

    high_sym_indices = collect(keys(kpath.labels))
    for idx in high_sym_indices
        axvline(x=kdistance[idx], color="grey", linewidth=0.5)
    end
    xticks([kdistance[idx] for idx in high_sym_indices],
           [kpath.labels[idx] for idx in high_sym_indices])

    nothing
end
