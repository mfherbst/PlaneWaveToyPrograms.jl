
"""
Structure to store the parameters for a
Hartwigsen, Goedecker, Teter, Hutter separable dual-space
Gaussian pseudopotential (1998).
"""
struct PspHgh
    """Pseudopotential identifying string"""
    identifier::String

    """Ionic charge (total charge - valence electrons)"""
    Zion::Float64

    """Range of local Gaussian ionic charge distribution"""
    rloc::Float64

    """Coefficients for the local part"""
    c::Vector{Float64}

    """Maximal angular momentum in the non-local part"""
    lmax::Int

    """Projector radius parameter for each angular momentum"""
    rp::Vector{Float64}

    """
    Non-local potential coefficients, one matrix for each AM channel
    """
    h::Vector{Matrix{Float64}}
end

function PspHgh(path::AbstractString)
    open(path, "r") do io
        PspHgh(io)
    end
end

function PspHgh(io::IOStream)
    # This reading function is very preliminary.
    # It does not really do proper checking

    lines = readlines(io)
    identifier = lines[1]

    # lines[2] contains the number of projectors for each AM channel
    m = match(r"^ *(([0-9]+ *)+)", lines[2])
    n_elec = [parse(Int, part) for part in split(m[1])]
    Zion = Float64(sum(n_elec))
    lmax = length(n_elec) - 1

    # lines[3] contains rloc nloc and coefficients for it
    m = match(r"^ *([-.0-9]+) *([0-9]+) *(([-.0-9]+ *)+)", lines[3])
    rloc = parse(Float64, m[1])
    nloc = parse(Int, m[2])
    c = [parse(Float64, part) for part in split(m[3])]
    @assert length(c) == nloc

    # lines[4] contanis (lmax + 1) again
    m = match(r"^ *([0-9]+)", lines[4])
    @assert lmax == parse(Int, m[1]) - 1

    rp = Vector{Float64}(undef, lmax + 1)
    h = Vector{Matrix{Float64}}(undef, lmax + 1)
    cur = 5  # Current line to parse
    for l in 0:lmax
        m = match(r"^ *([-.0-9]+) *([0-9]+) *(([-.0-9]+ *)+)", lines[cur])
        rp[l + 1] = parse(Float64, m[1])
        nproj = parse(Int, m[2])
        h[l + 1] = Matrix{Float64}(undef, nproj, nproj)

        hcoeff = [parse(Float64, part) for part in split(m[3])]
        for i in 1:nproj
            for j in i:nproj
                h[l + 1][j, i] = h[l + 1][i, j] = hcoeff[j - i + 1]
            end

            cur += 1
            if cur > length(lines)
                break
            end
            m = match(r"^ *(([-.0-9]+ *)+)", lines[cur])
            hcoeff = [parse(Float64, part) for part in split(m[1])]
        end
    end
    PspHgh(identifier, Zion, rloc, c, lmax, rp, h)
end

#
# -------
#

"""
Evaluate the local part of the pseudopotential in reciprocal space
taking a ΔG, a difference between plane waves as input. Effectively
computes <e_G|Vloc|e_{G+ΔG}> without taking into account the
structure factor and the (4π / Ω) spherical Hankel transform prefactor.
"""
function eval_psp_local_fourier(psp::PspHgh, ΔG)
    rloc = psp.rloc
    C(idx) = idx <= length(psp.c) ? psp.c[idx] : 0.0
    Grsq = sum(abs2, ΔG) * rloc^2

    (
        - psp.Zion / sum(abs2, ΔG) * exp(-Grsq / 2)
        + sqrt(π/2) * rloc^3 * exp(-Grsq / 2) * (
            + C(1)
            + C(2) * (  3 -       Grsq                       )
            + C(3) * ( 15 -  10 * Grsq +      Grsq^2         )
            + C(4) * (105 - 105 * Grsq + 21 * Grsq^2 - Grsq^3)
        )
   )
end


"""
Evaluate the local part of the pseudopotential in real space taking
a position vector r as input.
"""
function eval_psp_local_real(psp::PspHgh, r::Vector)
    rloc = psp.rloc
    C(idx) = idx <= length(psp.c) ? psp.c[idx] : 0.0
    rrsq = sum(abs2, r) / rloc

    (
        - psp.Zion / norm(r) * erf(norm(r) / sqrt(2) / rloc)
        + exp(-rrsq / 2) * ( C(1) + C(2) * rrsq + C(3) * rrsq^2 + C(4) * rrsq^3 )
    )
end


"""
Evaluate the radial part of a projector at a reciprocal point q.
Compared to the rigorous derivation in doc.tex this expresison
misses a factor i^l to avoid complex arithmetic. Compared to the ones presented
in the GTH and HGH papers it misses a factor of 1/sqrt(Ω), which is added
by the caller.
"""
function eval_psp_projection_radial(psp::PspHgh, i, l, qsq::Number)
    rp = psp.rp[l + 1]
    q = sqrt(qsq)
    qrsq = qsq * rp^2
    common = 4 * pi^(5 / 4) * sqrt(2^(l + 1) * rp^(2 * l + 3)) * exp(-qrsq / 2)

    if l == 0
        if i == 1 return common end
        # Note: In the next case the HGH paper has an error.
        #       The first 8 in equation (8) should not be under the sqrt-sign
        #       This is the right version (as shown in the GTH paper)
        if i == 2 return common *    2  / sqrt(15)  * (3  -   qrsq         ) end
        if i == 3 return common * (4/3) / sqrt(105) * (15 - 10qrsq + qrsq^2) end
    end

    if l == 1  # verify expressions
        if i == 1 return common * 1     /    sqrt(3) * q end
        if i == 2 return common * 2     /  sqrt(105) * q * ( 5 -   qrsq         ) end
        if i == 3 return common * 4 / 3 / sqrt(1155) * q * (35 - 14qrsq + qrsq^2) end
    end

    throw(ErrorException("Did not implement case of i == $i and l == $l"))
end
