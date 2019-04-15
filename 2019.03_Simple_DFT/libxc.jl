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

"""LibXC xc_func_info holder"""
mutable struct XcFuncInfoType
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

@enum FunctionalKind begin
    functional_exchange             = 0
    functional_correlation          = 1
    functional_exchange_correlation = 2
    functional_kinetic              = 3
end

@enum FunctionalFamily begin
    family_unknown  = -1
    family_lda      = 1
    family_gga      = 2
    family_mggai    = 4
    family_lca      = 8
    family_oep      = 16
    family_hyb_gga  = 32
    family_hyb_mgga = 64
end


mutable struct Functional
    number::Int
    name::String
    kind::FunctionalKind
    family::FunctionalFamily
    n_spin::Int

    # Pointer holding the LibXC representation of this functional
    pointer::Ptr{XcFuncType}
end


function Functional(name::String; n_spin::Int = 1)
    @assert n_spin == 1

    number = ccall(@xcsym(:xc_functional_get_number), Cint, (Cstring, ), name)
    if number == -1
        error("Functional $name is not known.")
    end

    function pointer_cleanup(ptr::Ptr{XcFuncType})
        if ptr != C_NULL
            ccall(@xcsym(:xc_func_end), Cvoid, (Ptr{XcFuncType}, ), ptr)
            ccall(@xcsym(:xc_func_free), Cvoid, (Ref{XcFuncType}, ), ptr)
        end
    end

    pointer = ccall(@xcsym(:xc_func_alloc), Ptr{XcFuncType}, ())
    try
        # Initialise to the desired functional
        ret = ccall(@xcsym(:xc_func_init), Cint, (Ptr{XcFuncType}, Cint, Cint),
                    pointer, number, n_spin)
        if ret != 0
            error("Something went wrong initialising the functional")
        end

        ptr_info = ccall(@xcsym(:xc_func_get_info), Ptr{XcFuncInfoType},
                         (Ptr{XcFuncType}, ), pointer)
        kind = ccall(@xcsym(:xc_func_info_get_kind), Cint, (Ptr{XcFuncInfoType}, ),
                     ptr_info)
        family = ccall(@xcsym(:xc_func_info_get_family), Cint, (Ptr{XcFuncInfoType}, ),
                       ptr_info)
        # flags = ccall(@xcsym(:xc_func_info_get_flags), Cint, (Ptr{XcFuncInfoType}, ),
        #               ptr_info)

        # Make functional and attach finaliser for cleaning up the pointer
        func = Functional(number, name, FunctionalKind(kind),
                          FunctionalFamily(family), n_spin, pointer)
        finalizer(cls -> pointer_cleanup(cls.pointer), func)
        return func
    catch
        pointer_cleanup(pointer)
        rethrow()
    end
end


function evaluate_lda_energy(func::Functional, ρ::Array{Float64, 3})
    @assert func.family == FunctionalFamily(1)
    @assert func.n_spin == 1

    E_XC = zeros(Float64, size(ρ)...)
    ccall(@xcsym(:xc_lda_exc), Cvoid, (Ptr{XcFuncType}, Cint, Ptr{Float64}, Ptr{Float64}),
          func.pointer, length(ρ), ρ, E_XC)
    return E_XC
end


function evaluate_lda_potential(func::Functional, ρ::Array{Float64, 3})
    @assert func.family == FunctionalFamily(1)
    @assert func.n_spin == 1

    Vρ_XC = zeros(Float64, size(ρ)...)
    ccall(@xcsym(:xc_lda_vxc), Cvoid, (Ptr{XcFuncType}, Cint, Ptr{Float64}, Ptr{Float64}),
          func.pointer, length(ρ), ρ, Vρ_XC)
    return Vρ_XC
end


function evaluate_lda!(func::Functional, ρ::Array{Float64, 3},
                       E_XC::Array{Float64, 3}, Vρ_XC::Array{Float64, 3})
    @assert func.family == FunctionalFamily(1)
    @assert func.n_spin == 1
    @assert size(E_XC) == size(ρ)
    @assert size(Vρ_XC) == size(ρ)

    ccall(@xcsym(:xc_lda_exc_vxc), Cvoid,
                 (Ptr{XcFuncType}, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
                 func.pointer, length(ρ), ρ, E_XC, Vρ_XC)
    E_XC, Vρ_XC
end

"""
Evaluate the energy of a GGA functional

ρ      Density on the grid
σ      Contracted density gradient (∇ρ \cdot ∇ρ) on a grid
"""
function evaluate_gga_energy(func::Functional, ρ::Array{Float64, 3}, σ::Array{Float64, 3})
    @assert func.family == FunctionalFamily(2)
    @assert func.n_spin == 1

    E_XC = zeros(Float64, size(ρ)...)
    ccall(@xcsym(:xc_gga_exc), Cvoid,
          (Ptr{XcFuncType}, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
          func.pointer, length(ρ), ρ, σ, E_XC)
    return E_XC
end


function evaluate_gga_potential(func::Functional, ρ::Array{Float64, 3},
                                σ::Array{Float64, 3})
    @assert func.family == FunctionalFamily(2)
    @assert func.n_spin == 1

    Vρ_XC = zeros(Float64, size(ρ)...)
    Vσ_XC = zeros(Float64, size(ρ)...)
    ccall(@xcsym(:xc_gga_vxc), Cvoid,
          (Ptr{XcFuncType}, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
          func.pointer, length(ρ), ρ, ρ, Vρ_XC, Vσ_XC)
    return Vρ_XC, Vσ_XC
end


function evaluate_gga!(func::Functional, ρ::Array{Float64, 3},
                       σ::Array{Float64, 3}, E_XC::Array{Float64, 3},
                       Vρ_XC::Array{Float64, 3}, Vσ_XC::Array{Float64, 3})
    @assert func.family == FunctionalFamily(2)
    @assert func.n_spin == 1
    @assert size(E_XC) == size(ρ)
    @assert size(Vρ_XC) == size(ρ)
    @assert size(Vσ_XC) == size(ρ)

    ccall(@xcsym(:xc_gga_exc_vxc), Cvoid,
                 (Ptr{XcFuncType}, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                  Ptr{Float64}, Ptr{Float64}),
                 func.pointer, length(ρ), ρ, σ, E_XC, Vρ_XC, Vσ_XC)
    E_XC, Vρ_XC, Vσ_XC
end
