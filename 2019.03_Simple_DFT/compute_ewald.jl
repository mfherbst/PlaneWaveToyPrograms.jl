using SpecialFunctions: erfc


"""
Compute nuclear-nuclear repulsion energy per unit cell
in a uniform background of negative charge following the
Ewald summation procedure. The convergence parameter η
is by default chosen automatically depending on the
plane-wave mesh, but an explicit value may be provided
as well.

system    System to compute the energy for
Zs        Charges to use for the atoms in System (by default
          chosen as system.Zs, but may be modified, e.g. for
          pseudopotentials.
"""
function compute_ewald(system::System; Zs=nothing, η=nothing)
    # All numerical precision constants in this code have been
    # optimised for Float64
    etype = Float64

    if Zs == nothing
        Zs = system.Zs
    end
    if η == nothing
        # Balance between reciprocal summation and real-space summation
        # with a slight bias towards recirpocal summation
        η = sqrt(sqrt(1.69 * norm(system.B ./ 2π) / norm(system.A))) / 2
    end

    #
    # Numerical cutoffs
    #
    # The largest argument to the exp(-x) function
    # to obtain a numerically meaningful contribution.
    # The +5 is for safety
    max_exponent = -(log(eps(etype)) + 5)

    # The largest argument to the erfc function for various precisions.
    # To get an idea:
    #     erfc(8) ≈ 1e-29
    #     erfc(10) ≈ 2e-45
    max_erfc_arg = 8
    if etype != Float64
        error("Not implemented.")
    end

    #
    # Reciprocal space sum
    #
    # Initialise reciprocal sum with correction term for charge neutrality
    sum_recip = - (sum(Zs)^2 / 4η^2)

    # Function to return the indices corresponding
    # to a particular shell
    function shell_indices(ish)
        [[i,j,k] for i in -ish:ish for j in -ish:ish for k in -ish:ish
         if maximum(abs.([i,j,k])) == ish]
    end

    # Loop over reciprocal-space shells
    gsh = 0
    any_term_contributes = true
    while any_term_contributes
        # Notice that the first gsh this loop processes is 1
        # in other words G == 0 is implicitly excluded.
        gsh += 1
        any_term_contributes = false

        # Compute G vectors and moduli squared for this
        # shell patch
        idcs = shell_indices(gsh)
        Gs = [system.B * coord for coord in shell_indices(gsh)]
        Gsqs = [sum(abs2, G) for G in Gs]

        accu = 0.0
        for (ig, G) in enumerate(Gs)
            # Check if the Gaussian exponent is small enough
            # for this term to contribute to the reciprocal sum
            exponent = Gsqs[ig] / 4η^2
            if exponent > max_exponent * 10
                continue
            end

            cos_strucfac = sum(Zi * cos(dot(system.atoms[iat], G))
                               for (iat, Zi) in enumerate(Zs))
            sin_strucfac = sum(Zi * sin(dot(system.atoms[iat], G))
                               for (iat, Zi) in enumerate(Zs))
            sum_strucfac = cos_strucfac * cos_strucfac + sin_strucfac * sin_strucfac

            any_term_contributes = true
            sum_recip += sum_strucfac * exp(-exponent) / Gsqs[ig]
        end
    end
    # Amend sum_recip by proper scaling factors:
    sum_recip = sum_recip * 4π / system.unit_cell_volume

    #
    # Real-space sum
    #
    # Initialise real-space sum with correction term for uniform background
    sum_real = -2η / sqrt(π) * sum(Zs .^ 2)

    # Loop over real-space shells
    rsh = -1
    any_term_contributes = true
    while any_term_contributes || rsh <= 0
        # In this loop the first rsh, which is processed is rsh == 0
        rsh += 1
        any_term_contributes = false

        # Loop over R vectors for this shell patch
        for R in [system.A * coord for coord in shell_indices(rsh)]
            for (iat, ti) in enumerate(system.atoms), (jat, tj) in enumerate(system.atoms)
                # Compute norm of the distance
                dist = norm(ti - tj - R)

                # Avoid zero denominators
                if dist <= 1e-24
                    continue
                end

                # erfc decays very quickly, so cut off at some point
                if η * dist > max_erfc_arg
                    continue
                end

                any_term_contributes = true
                sum_real += Zs[iat] * Zs[jat] * erfc(η * dist) / dist
            end # iat, jat
        end # R
    end

    # println("gsh = $(gsh)    rsh = $(rsh)")

    # Return total sum, amended by 1/2 (because of double counting)
    (sum_recip + sum_real) / 2
end


function compute_pickard(system::System; Zs=nothing, Rc=nothing, Rd=nothing)


# TODO



end



# TODO Take a look at the approximate Ewald summation using semi-analytic
#      terms by Chris Pickard, which scales very well and replaces Fourier-space part.


