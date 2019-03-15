struct PlaneWaveBasis
    #
    # Plane-Wave mesh
    #
    """The kpoints of the basis (== BrilloinZoneMesh)"""
    kpoints::Vector{Vector{Float64}}

    """Volume of the unit cell"""
    unit_cell_volume::Float64

    """Maximal kinetic energy |G + k|^2 in Hartree"""
    Ecut::Float64

    """Wave vectors of the plane wave basis functions in reciprocal space"""
    Gs::Vector{Vector{Float64}}

    """Index of the DC fourier component in the plane-wave mesh"""
    idx_DC::Int

    """
    Masks (list of indices), which select the plane waves required for each
    kpoint, such that the resulting plane-wave basis reaches the
    selected Ecut threshold.
    """
    Gmask::Vector{Vector{Int}}

    #
    # Fast-fourier transform
    #
    """Supersampling used for Fast-Fourier transformations"""
    fft_supersampling::Float64

    # FFT parameters
    """Size of the FFT grid in each dimension"""
    fft_size::Vector{Int}

    """Translation table to transform from the
    plane-wave basis to the indexing convention needed for the FFT
    (DC component in the middle, others periodically, asymmetrically around it"""
    idx_to_fft::Vector{Vector{Int}}

    # Space to store planned FFT operators from FFTW
    """Plan for forward FFT"""
    FFT::typeof(plan_fft!(im * randn(2, 2, 2)))

    """Plan for inverse FFT"""
    iFFT::typeof(plan_ifft(im * randn(2, 2, 2)))
end


function PlaneWaveBasis(system::System, kpoints::Vector{Vector{Float64}}, Ecut;
                        fft_supersampling=2)
    # We want the set of wavevectors {G} to be chosen such that |G|^2/2 ≤ Ecut,
    # i.e. |G| ≤ 2 * sqrt(Ecut). Additionally the representation of the electron
    # density and the exchange-correlation term needs more wavevectors
    # (see doc.tex for details), namely for the density an fft_supersampling
    # of 2. This requires |G| ≤ fft_supersampling * sqrt(2 * Ecut),
    # such that the cutoff for |G|^2 can be computed to
    cutoff_Gsq = 2 * fft_supersampling^2 * Ecut

    # Construct the plane-wave grid
    coords, Gs = construct_pw_grid(system, cutoff_Gsq, kpoints=kpoints)

    # Index of the DC component inside Gs
    idx_DC = findfirst(isequal([0, 0, 0]), coords)
    @assert idx_DC != nothing

    # Select the Gs falling in the energy range determined by Ecut
    # for each kpoint
    Gmash = Vector{Vector{Int}}(undef, length(kpoints))
    for (ik, k) in enumerate(kpoints)
        Gmash[ik] = findall(G -> sum(abs2, k + G) ≤ 2 * Ecut, Gs)
    end

    # Maximal and minimal coordinates along each direction
    max_coords = maximum(abs.(hcat(coords...)), dims=2)
    min_coords = minimum(abs.(hcat(coords...)), dims=2)

    # Form and optimise FFT grid dimensions
    fft_size = reshape(max_coords .- min_coords .+ 1, :)
    fft_size = optimise_fft_grid(fft_size)

    # Translation table from plane-wave to FFT grid
    n_G = length(Gs)
    idx_to_fft = [1 .+ mod.(coords[ig], fft_size) for ig in 1:n_G]

    tmp = Array{ComplexF64}(undef, fft_size...)
    fft_plan = plan_fft!(tmp)  # can play with FFTW flags here
    ifft_plan = plan_ifft(tmp) # can play with FFTW flags here

    PlaneWaveBasis(kpoints, system.unit_cell_volume, Ecut, Gs,
                   idx_DC, Gmash, fft_supersampling, fft_size,
                   idx_to_fft, fft_plan, ifft_plan)
end


"""
Optimise an FFT grid such that the number of grid points in each dimenion
agrees well with a fast FFT algorithm.
"""
function optimise_fft_grid(n_grid_points::Vector{Int})
    # Ensure a power of 2 in the number of grid points
    # in each dimension for a fast FFT
    # (because of the tree-like divide and conquer structure of the FFT)
    return map(x -> nextpow(2, x), n_grid_points)
end


"""
Construct a plane-wave grid, which is able to provide a discretisation
of the passed system at the provided kpoints such that at least the
resulting cutoff is reached at all points, that is that
for all kpoints k and plane-wave vectors G, we have
|k + G|^2 ≤ cutoff_Gsq
"""
function construct_pw_grid(system::System, cutoff_Gsq::Number;
                           kpoints::Vector{Vector{Float64}} = [[0, 0, 0]])
    B = system.B  # Reciprocal lattice vectors

    # For a particular k-Point, the coordinates [m n o] of the
    # complementory reciprocal lattice vectors satisfy
    #     |B * [m n o] + k|^2 ≤ cutoff_Gsq
    # Now
    #     |B * [m n o] + k| ≥ abs(|B * [m n o]| - |k|) = |B * [m n o]| - |k|
    # provided that |k| ≤ |B|, which is typically the case. Therefore
    #     |λ_{min}(B)| * |[m n o]| ≤ |B * [m n o]| ≤ sqrt(cutoff_Gsq) + |k|
    # (where λ_{min}(B) is the smallest eigenvalue of B), such that
    #     |[m n o]| ≤ (sqrt(cutoff_Gsq) + |k|) / |λ_{min}(B)|
    # In the extremal case, m = o = 0, such that
    #    n_max_trial = (sqrt(cutoff_Gsq) + |k|) / |λ_{min}(B)|

    eig_B = eigvals(B)
    max_k = maximum(norm.(kpoints))

    # Check the assumption in above argument is true
    @assert max_k ≤ maximum(abs.(eig_B))

    # Use the above argument to figure out a trial upper bound n_max
    trial_n_max = ceil(Int, (max_k + sqrt(cutoff_Gsq)) / minimum(abs.(eig_B)))

    # Go over the trial range (extending trial_n_max by one for safety)
    trial_n_range = -trial_n_max-1:trial_n_max+1

    # Determine actual n_max
    n_max = 0
    for coord in CartesianIndices((trial_n_range, trial_n_range, trial_n_range))
        G = B * [coord.I...]

        if any(sum(abs2, G + k) ≤ cutoff_Gsq for k in kpoints)
            @assert all(abs.([coord.I...]) .<= trial_n_max)
            n_max = max(n_max, maximum(abs.([coord.I...])))
        end
    end

    # Now fill returned quantities
    n_range = -n_max:n_max
    coords = [[coord.I...]
                 for coord in CartesianIndices((n_range, n_range, n_range)) ]
    coords = reshape(coords, :)
    Gs = [B * coord for coord in coords]
    return coords, Gs
end


"""
Take an existing Plane-wave basis and replace its kpoints without altering
the plane-wave vectors, i.e. without altering the Gs
"""
function substitute_kpoints(pw::PlaneWaveBasis, kpoints::Vector{Vector{Float64}})
    # Compute new Gmask and substitute the old one
    Gmask = Vector{Vector{Int}}(undef, length(kpoints))
    for (ik, k) in enumerate(kpoints)
        Gmask[ik] = findall(G -> sum(abs2, k + G) ≤ 2 * pw.Ecut, pw.Gs)
    end

    PlaneWaveBasis(kpoints, pw.unit_cell_volume, pw.Ecut, pw.Gs,
                   pw.idx_DC, Gmask, pw.fft_supersampling, pw.fft_size,
                   pw.idx_to_fft, pw.FFT, pw.iFFT)
end


#
# Perform FFT
#
function G_to_R(pw::PlaneWaveBasis, F_fourier)
    @assert length(F_fourier) == length(pw.Gs)
    F_real = zeros(ComplexF64, pw.fft_size...)
    for ig=1:length(pw.Gs)
        F_real[pw.idx_to_fft[ig]...] = F_fourier[ig]
    end
    F_real = pw.iFFT * F_real  # Note: This destroys data in F_real

    # IFFT has a normalization factor of 1/length(ψ),
    # but the normalisation convention used in this code is
    # e_G(x) = e^iGx / sqrt(|Γ|), so we need to use the factor
    # below in order to match both conventions.
    F_real .*= (length(F_real) / sqrt(pw.unit_cell_volume))
    @assert norm(imag(F_real)) < 1e-10
    real(F_real)
end


function R_to_G(pw::PlaneWaveBasis, F_real)
    @assert [size(F_real)...] == pw.fft_size

    # Do FFT on the full FFT plan, but only keep within
    # the n_G from the kinetic energy cutoff -> Lossy Compression of data
    F_fourier_extended = pw.FFT * F_real
    F_fourier = zeros(ComplexF64, length(pw.Gs))
    for ig=1:length(F_fourier)
        F_fourier[ig] = F_fourier_extended[pw.idx_to_fft[ig]...]
    end
    # Again adjust normalisation as in G_to_R
    F_fourier .*= (sqrt(pw.unit_cell_volume) / length(F_real))
end
