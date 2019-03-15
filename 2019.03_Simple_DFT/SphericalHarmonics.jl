using LinearAlgebra

# These guys are probably implemented in some package properly

function ylm_real(l::Int64, m::Int64, rvec::AbstractVector)
    @assert 0 ≤ l
    @assert -l ≤ m ≤ l
    @assert length(rvec) == 3
    x, y, z = rvec
    r = norm(rvec)

    if l == 0 # s
        if m ==  0 return sqrt(1 / 4π) end
    end

    # Catch cases of numerically very small r
    if r <= 1e-14
        return 0.0
    end

    if l == 1 # p
        if m == -1 return sqrt(3 / 4π) * y / r end
        if m ==  0 return sqrt(3 / 4π) * z / r end
        if m ==  1 return sqrt(3 / 4π) * x / r end
    end

    if l == 2 # d
        if m == -2 return sqrt(15 / 4π) * (x / r) * (y / r) end
        if m == -1 return sqrt(15 / 4π) * (y / r) * (z / r) end
        if m ==  0 return sqrt(5 / 16π) * (2z^2 - x^2 - y^2) / r^2 end
        if m ==  1 return sqrt(15 / 4π) * (x / r) * (z / r) end
        if m ==  2 return sqrt(15 / 4π) * (x^2 - y^2) / r^2 end
    end

    throw(ErrorException("The case l = $l and m = $m is not implemented"))
end
