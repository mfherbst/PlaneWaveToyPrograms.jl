"""
Plot a band of data for each kpoint
"""
function plot_bands(kpoints::KPoints, values::Vector{Vector{T}} where T <: Number)
    n_ks = length(kpoints.kpoints)
    n_bands = minimum(length.(values))
    kdistance = kpoints.accumulated_kdistance

    figure()
    for ib in 1:n_bands
        plot(kdistance, [values[ik][ib] for ik in 1:n_ks], "rx-")
    end

    high_sym_indices = collect(keys(kpoints.labels))
    for idx in high_sym_indices
        axvline(x=kdistance[idx], color="grey", linewidth=0.5)
    end
    xticks([kdistance[idx] for idx in high_sym_indices],
           [kpoints.labels[idx] for idx in high_sym_indices])

    nothing
end
