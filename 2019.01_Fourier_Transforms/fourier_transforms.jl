#!/usr/bin/env julia

using FFTW
using PyPlot

# Experiments with the inverse Fourier transform of a
# potential, which can be expressed as a multiplicative
# 1/G^n factor in Fourier space

function compute(potential, Gcut, supersampling)
    # Setup G vectors
    Gs = collect(-Gcut:Gcut)
    n_G = length(Gs)

    # Setup Fourier transform
    fft_size = nextpow(2, supersampling * (2Gcut + 1))
    a = Array{ComplexF64}(undef, fft_size)
    ifft_plan = plan_ifft(a)

    G_to_fft = mod.(Gs, fft_size) .+ 1

    # Build potential data
    potF = zeros(ComplexF64, fft_size)
    for (ig, G) in enumerate(Gs)
        potF[G_to_fft[ig]] = potential(G)
    end

    # Fourier transform it
    pot = ifft_plan * potF
    # IFFT has a normalization factor of 1/length(ψ),
    # so we undo this here
    pot *= length(pot)
    @assert maximum(imag(pot)) < 1e-10
    pot = real(pot)

    # Coordinates of the Gvectors corresponding to
    # the indices in the FFT array potF
    Gcoords = fftshift(-Int(fft_size//2):Int(fft_size//2)-1)
    Rcoords = collect(0:length(pot) - 1) ./ (length(pot) - 1)
    return Rcoords, pot, Gcoords, real(potF)
end

function plot_potential(potential; Gcuts=[30, 60, 100, 200, 400, 1000], supersampling=[2, 4])
    plot_Gcuts = Vector{Float64}()
    plot_supersampling = Vector{Float64}()
    plot_Rcoords = Vector{Vector{Float64}}()
    plot_pots = Vector{Vector{Float64}}()
    plot_Gcoords = Vector{Vector{Float64}}()
    plot_potFs = Vector{Vector{Float64}}()

    for Gcut in Gcuts
        for ssample in supersampling
            Rcoords, pot, Gcoords, potF = compute(potential, Gcut, ssample)

            push!(plot_Gcuts, Gcut)
            push!(plot_supersampling, ssample)
            push!(plot_Rcoords, Rcoords)
            push!(plot_pots, pot)
            push!(plot_Gcoords, Gcoords)
            push!(plot_potFs, potF)
        end
    end

    figure()
    title("Potential (in fourier space)")
    for (i, Gcut) in enumerate(plot_Gcuts)
        plot(plot_Gcoords[i], plot_potFs[i], "x",
             label="Gcut=$Gcut, supersampling=$(plot_supersampling[i])")
    end
    legend()
    show()

    figure()
    title("Potential (in real space)")
    for (i, Gcut) in enumerate(plot_Gcuts)
        plot(plot_Rcoords[i], plot_pots[i], "-",
             label="Gcut=$Gcut, supersampling=$(plot_supersampling[i])")
    end
    legend()
    show()
end

function main()
    Z = 5              # Amplitude of the potential
    function potential_inv_G(G, power)
        if G == 0
            return 0
        else
            return -4π * Z / abs(G)^power
        end
    end

    plot_potential(G -> potential_inv_G(G, 0.5),
                   Gcuts=[100, 5000], supersampling=[2, 10])
    plot_potential(G -> potential_inv_G(G, 1),
                   Gcuts=[100, 5000], supersampling=[2, 10])
    plot_potential(G -> potential_inv_G(G, 2),
                   Gcuts=[100, 5000], supersampling=[2, 10])
    plot_potential(G -> potential_inv_G(G, 4),
                   Gcuts=[100, 5000], supersampling=[2, 10])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
