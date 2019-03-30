using Libdl

push!(DL_LOAD_PATH, @__DIR__)

LIBXC_HANDLE = Libdl.dlopen_e("libxc")
if LIBXC_HANDLE  == C_NULL
    libxc_available_functionals() = []
    return
end


"""Macro to insert symbol resolution code for LIBXC symbols"""
macro xcsym(sym)
    return :( Libdl.dlsym(LIBXC_HANDLE, $sym) )
end


"""LibXC xc_func_type holder for functional info"""
mutable struct XcFuncType
end


"""Return the version of the libxc library."""
function libxc_version()
    varray = zeros(Cint, 3)
    ccall(@xcsym(:xc_version), Cvoid, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
          pointer(varray, 1), pointer(varray, 2), pointer(varray, 3))
    VersionNumber(varray[1], varray[2], varray[3])
end


"""Get the list of available libxc functionals"""
function libxc_available_functionals()
    n_xc = ccall(@xcsym(:xc_number_of_functionals), Cint, ())
    max_string_length = ccall(@xcsym(:xc_maximum_name_length), Cint, ())

    funcnames = Vector{String}(undef, n_xc)
    for i in 1:n_xc
        funcnames[i] = "\0"^(max_string_length + 2)
    end
    ccall(@xcsym(:xc_available_functional_names), Cvoid, (Ptr{Ptr{UInt8}}, ),
          funcnames)

    [string(split(funcnames[i], "\0")[1]) for i in 1:n_xc]
end


struct Functional
    number::Int
    name::String
end
function Functional(name::String)
    number = ccall(@xcsym(:xc_functional_get_number), Cint, (Cstring, ), name)
    if number == -1
        error("Functional $name is not known.")
    end
    Functional(number, name)
end


"""
Compute LDA-type functional energy and derivatives
(0 => energy, 1 => potential, 2 => 2nd derivative, 3 => 3rd derivative)
"""
function evaluate_lda(func::Functional, ρ::Array{Float64, 3};
                      derivatives=[0, 1])
    sort!(derivatives)
    n_spin = 1  # Hard-coded right now

    # Init and alloc
    ptr = ccall(@xcsym(:xc_func_alloc), Ptr{XcFuncType}, ())
    ret = ccall(@xcsym(:xc_func_init), Cint, (Ptr{XcFuncType}, Cint, Cint),
                ptr, func.number, n_spin)

    if derivatives == [1]
        V_XC = zeros(Float64, size(ρ)...)
        ccall(@xcsym(:xc_lda_vxc), Cvoid,
              (Ptr{XcFuncType}, Cint, Ptr{Float64}, Ptr{Float64}),
              ptr, length(ρ), ρ, V_XC)
        return Dict(1 => V_XC)
    elseif derivatives == [0]
        E_XC = zeros(Float64, size(ρ)...)
        ccall(@xcsym(:xc_lda_exc), Cvoid,
              (Ptr{XcFuncType}, Cint, Ptr{Float64}, Ptr{Float64}),
              ptr, length(ρ), ρ, E_XC)
        return Dict(0 => E_XC)
    elseif derivatives == [0, 1]
        E_XC = zeros(Float64, size(ρ)...)
        V_XC = zeros(Float64, size(ρ)...)
        ccall(@xcsym(:xc_lda_exc_vxc), Cvoid,
              (Ptr{XcFuncType}, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
              ptr, length(ρ), ρ, E_XC, V_XC)
        return Dict(0 => E_XC, 1 => V_XC)
    else
        error("Not implemented: $derivatives")
    end

    # end and free
    ccall(@xcsym(:xc_func_end), Cvoid, (Ptr{XcFuncType}, ), ptr )
    ccall(@xcsym(:xc_func_free), Cvoid, (Ref{XcFuncType}, ), ptr )
end


# LDA_X = Functional("lda_x")
# LDA_C = Functional("lda_c_vwn")
# PBE_X = Functional("gga_x_pbe")
# PBE_C = Functional("gga_c_pbe")
