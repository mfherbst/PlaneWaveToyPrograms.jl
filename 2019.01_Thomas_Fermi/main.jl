using StaticArrays
using FFTW
using Roots
using PyPlot
using NLsolve
using NLSolversBase
using Printf
using Statistics

# We solve the TF equation
# ρ = (εF - Vh - Vext)^3/2
# Vext = sum of G1(x-Ra) for Ra running over the atoms in the unit cell
# Vh = G1 * ρ (periodic convolution)
# G1 = 4π ∑_G |G|^-2 e^iGx (solution of -ΔG1 = 4π \sum_R δ_R)
# ∫ρ = N (number of electrons per unit cell, = Z * number of atoms in unit cell)

# We discretize any A-periodic function ψ on plane waves:
# ψ = ∑_G c_G e_G(x), with G the reciprocal lattice vectors,
# and e_G the orthonormal basis e_G(x) = e^iGx / sqrt(|Γ|), with Γ the unit cell
# We evaluate the TF equation in real space through FFTs
struct Structure
    A::Matrix{Float64} # 3x3 lattice vectors, in columns. |Γ| = det(A)
    atoms::Vector{SVector{3,Float64}} # list of atomic positions in the unit cell
    Z::Float64 # charge
    Ecut::Float64 # max kinetic energy |G|^2

    B::Matrix{Float64} # reciprocal lattice
    vol_unit_cell::Float64

    Ng::Int # number of G vectors
    Gs_ind::Vector{SVector{3,Int}} # indices of G vectors
    ind_DC::Int # index of the DC component
    Gs::Vector{SVector{3,Float64}} # G vectors
    fft_size::Int # size of the FFT grid
    G_to_fft::Vector{SVector{3,Int}}
    fft_to_G::Array{Int,3}
    fft_plan::typeof(plan_fft!(im*randn(2,2,2)))
    ifft_plan::typeof(plan_ifft!(im*randn(2,2,2)))
end

function Structure(A,atoms,Z,Ecut)
    vol_unit_cell = det(A)
    B = 2π*inv(A')

    # Figure out upper bound Nmax on number of basis functions for any dimension
    # Want |B*(i j k)|^2 <= Ecut
    Nmax = ceil(Int,sqrt(Ecut)/opnorm(B))
    Gs_ind = Vector{SVector{3,Int}}()
    ind = 1
    ind_DC = -1
    for i=-Nmax-1:Nmax+1, j=-Nmax-1:Nmax+1, k=-Nmax-1:Nmax+1
        G = B*[i;j;k]
        # println("$i $j $k $(sum(abs2,G))")
        if i == 0 && j == 0 && k == 0
            ind_DC = ind
        end
        if sum(abs2,G) < Ecut
            # add to basis
            push!(Gs_ind,@SVector[i,j,k])
            ind += 1
            @assert abs(i) <= Nmax && abs(j) <= Nmax && abs(k) <= Nmax # to be on the safe side that Nmax is large enough
        end
    end
    @assert ind_DC != -1 && Gs_ind[ind_DC] == [0;0;0]
    Gs = [B*G_ind for G_ind in Gs_ind]
    Ng = length(Gs)

    supersampling = 2
    fft_size = nextpow(2,supersampling*(2Nmax+1)) # ensure a power of 2 for fast FFTs
    G_to_fft = zeros(SVector{3,Int},Ng)
    fft_to_G = zeros(Int,fft_size,fft_size,fft_size)
    fft_to_G .= -1 # unaffected
    for ig in 1:Ng
        # Put the DC component at (1,1,1) and wrap the others around
        ifft = mod.(Gs_ind[ig],fft_size).+1
        G_to_fft[ig] = ifft
        fft_to_G[ifft...] = ig
    end

    a = Array{ComplexF64}(undef, fft_size, fft_size, fft_size)
    fft_plan = plan_fft!(a)
    ifft_plan = plan_ifft!(a)

    Structure(A,atoms,Z,Ecut,B,vol_unit_cell,Ng,Gs_ind,ind_DC,Gs,fft_size,G_to_fft,fft_to_G, fft_plan, ifft_plan)
end

# G function in Fourier space
function coulomb(S::Structure)
    pot = zeros(ComplexF64,S.Ng)
    for iat = 1:length(S.atoms)
        R = S.atoms[iat]
        for ig = 1:S.Ng
            if ig != S.ind_DC
                G = S.Gs[ig]
                pot[ig] += 4π*S.Z*cis(dot(G,R))/sum(abs2,G)*sqrt(S.vol_unit_cell)
            end
        end
    end
    pot
end

# Computes the Hartree potential associated with ρ
function hartree(S::Structure, ρ)
    pot = zeros(ComplexF64,S.Ng)
    @assert length(ρ) == S.Ng
    for ig = 1:S.Ng
        G = S.Gs[ig]
        if ig != S.ind_DC
            pot[ig] += 4π*ρ[ig]/sum(abs2,G)
        end
    end
    pot
end

function to_real(S::Structure, ψ_four)
    ψ_real = zeros(ComplexF64, S.fft_size, S.fft_size, S.fft_size)
    for ig=1:S.Ng
        ψ_real[S.G_to_fft[ig]...] = ψ_four[ig]
    end
    mul!(ψ_real, S.ifft_plan, ψ_real)
    ψ_real .*= (length(ψ_real) / sqrt(S.vol_unit_cell)) # IFFT have a normalization factor of 1/length(ψ)
    @assert norm(imag(ψ_real)) < 1e-10
    real(ψ_real)
end
function to_fourier(S::Structure, ψ_real)
    ψ_four_extended = S.fft_plan*ψ_real
    ψ_four = zeros(ComplexF64,S.Ng)
    for ig=1:S.Ng
        ψ_four[ig] = ψ_four_extended[S.G_to_fft[ig]...]
    end
    ψ_four .*= (sqrt(S.vol_unit_cell)/length(ψ_real))
end

# Computes the density corresponding to a total potential, both in real space
# solve the equation ∫(εF - Vext - Vh)^3/2 = N for εF
function Vtot_to_ρ(S::Structure, Vtot, N, tol)
    reg_pow(x) = x < 0 ? zero(x) : sqrt(x*x*x) #faster than x^3/2
    ρreal = copy(Vtot) # acts as a buffer
    makeρ!(ρreal,ε,Vtot) = (ρreal .= reg_pow.(ε .- Vtot))
    function Nρ(ε,ρreal,Vtot,vol_unit_cell) # compute number of electrons in ρ, modifies the buffer
        makeρ!(ρreal,ε,Vtot)
        mean(ρreal)*vol_unit_cell
    end

    εF = find_zero(ε -> Nρ(ε,ρreal,Vtot,S.vol_unit_cell) - N, 0.0, rtol=0.0, atol=tol, xatol=0.0, xrtol=0.0, verbose=false)
    ρreal
end

function new_ρ(S::Structure, Vext, ρ, N, tol)
    # build total potential
    Vh = hartree(S,ρ)
    Vtot = Vh + Vext
    @assert abs(Vtot[S.ind_DC]) < 1e-10
    Vtot = to_real(S, Vtot)

    ρreal = Vtot_to_ρ(S,Vtot,N,tol)
    @assert all(ρreal .>= 0)

    newρ = to_fourier(S,ρreal)
    @assert abs(newρ[S.ind_DC] - N/sqrt(S.vol_unit_cell)) < tol
    # newρ[S.ind_DC] = N/sqrt(S.vol_unit_cell) #enforce exactly
    @assert norm(imag(newρ)) < 1e-10 #only true because of inversion symmetry
    real(newρ)
end

function compute(L,Z,Ecut)
    A = L*Matrix(Diagonal(ones(3)))
    atoms = [[0.0,0.0,0.0]]
    # Z = 1
    N = Z*length(atoms)
    tol = 1e-10

    S = Structure(A,atoms,Z,Ecut)
    println("Computing with Z=$Z, fft_size=$(S.fft_size)")

    Vext = -coulomb(S)
    @assert to_fourier(S,to_real(S,Vext)) ≈ Vext # should really be in a test suite

    ρ0 = zeros(S.Ng)
    # Start from a constant density. Want int of C e0 = N => C = N/int(e0) = N/sqrt(|Γ|)
    ρ0[S.ind_DC] = N/sqrt(S.vol_unit_cell)
    @assert mean(to_real(S,ρ0))*S.vol_unit_cell ≈ N

    F!(resid,ρ) = (resid .= new_ρ(S, Vext, ρ, N, tol) .- ρ)

    od   = OnceDifferentiable(F!,identity,ρ0,ρ0,[]) # work around https://github.com/JuliaNLSolvers/NLsolve.jl/issues/202
    ρ = nlsolve(od,ρ0,method=:anderson, m=5,xtol=tol,ftol=0.0, show_trace=true).zero

    # # method 1
    # # energy from the filtered ρ
    # return real(mean(complex(to_real(S,ρ)).^(5/3))*S.vol_unit_cell)

    # method 2
    # extract energy from the unfiltered ρ
    # computationally suboptimal but not critical
    Vh = hartree(S,ρ)
    Vtot = Vh + Vext
    Vtot = to_real(S, Vtot)
    ρreal = Vtot_to_ρ(S,Vtot,ρ,N,tol)
    mean(ρreal.^(5/3))*S.vol_unit_cell


    figure()
    x = range(0,stop=L,length=S.fft_size)
    plot(x,ρreal[:,1,1])
    STOP

    # mean(ρreal.^(5/3))*S.vol_unit_cell
end

Ecut = 5000
Zs = 1:10
L = 1.0
Es = compute.(L,Zs,Ecut)
# figure()
# plot(Zs.^(7/3),Es,"-x")
# plot(Zs,Es./(Zs.^(7/3)),"-x")

mat = hcat(ones(length(Zs)), Zs.^(7/3))
a,b = mat \ Es
relabs = sqrt(sum(abs2,Es .- (a.+b.*Zs.^(7/3)))/var(Es))

figure()
plot(Zs.^(7/3),Es,"-x")
plot(Zs.^(7/3),a.+b.*Zs.^(7/3),"-")
title("L=$L,Ecut=$Ecut,$(@sprintf("%.2f",a))+$(@sprintf("%.2f",b))*Z^7/3,relerr=$(@sprintf("%.4f",relabs))")


# using Profile
# using ProfileView
# Profile.clear()
# @profile compute(1,1,10000)
# ProfileView.view()
