using PyPlot

# We solve the Kronig-Penney model -Δ + V with
# V = Z \sum δ_{an}
# in a supercell of size aL

# This does not use k points to be able to treat defects
# We discretize in a basis of plane waves |q> = e^iqx / sqrt(aL), q = 2π/a * (N, ..., -1/L,0,1/L, ..., N)

const Z = -5.0 #potential strength
const Zdef = +0.0 # defect strength
const a = 1 #lattice constant
const N = 20 #number of plane waves per unit cell
const L = 20 #supercell size

# all plane waves that can fit in a box of size L
const qrange = 2π/a*(-N:(1/L):N)
# to get a FFT-like formula we need that x * qj is 2pi*k/(2NL+1) for some k => xk = k*L/(2NL+1)
const xrange = a*(0:L/(2*N*L+1):L)[1:end-1]
# sanity checks
const Nb = length(qrange)
@assert Nb == (2N)*L+1
@assert length(xrange) == Nb

function build_ham()
    A = zeros(Complex128,Nb,Nb)
    for i=1:Nb,j=1:Nb
        # A[i,j] = <qi|H|qj>
        qi,qj = qrange[i],qrange[j]

        # Kinetic
        if i == j
            A[i,i] = abs2(qi)
        end

        # Periodic Kronig-Penney potential
        for l=0:L-1
            A[i,j] += Z*exp(im*(qj-qi)*a*l)/(a*L)
        end

        # Defect
        ldef = div(L,2)
        A[i,j] += Zdef*exp(im*(qj-qi)*a*ldef)/(a*L)
    end
    return Hermitian(A)
end

# explicit fourier->real map (slow Fourier transform). Can be implemented more efficiently with an FFT
function build_four_to_real()
    Q = zeros(Complex128,Nb,Nb)
    for ix = 1:Nb
        for iq = 1:Nb
            Q[ix,iq] = exp(im*qrange[iq]*xrange[ix])/sqrt(a*L)
        end
    end
    Q
end

A = build_ham()
Q = build_four_to_real()
@assert diagm(diag(Q'Q)) ≈ (Q'Q)[1,1]*eye(Nb) #ensure unitarity up to constant factor
F2R = Q
R2F = inv(Q)

E,V = eig(A)
# @assert (E[L+1] - E[L]) >  (E[L] - E[1]) #ensure good separation of bands
# density matrix
Pq = V[:,1:L]*V[:,1:L]' # in q space
Px = F2R*Pq*R2F # in x space
ρ = diag(Px)

close(:all)

# Plot the spectrum = collapsed band structure
figure()
plot(E,"x") # 

# Plot the Bloch waves
figure()
for iplot=1:5
    plot(real(F2R*V[:,iplot]))
end

# Plot the density
figure()
plot(xrange,ρ)
# plot(xrange,(Px[div(end,2),:])) #locality of the density matrix
