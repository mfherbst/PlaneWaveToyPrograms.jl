import PyPlot

function plot_cube(center, length, options)
    lower = .5 .*[[length,length,-length],
                  [length,-length,-length],
                  [-length,-length,-length],
                  [-length,length,-length],
                  [length,length,-length]]
    xs = [p[1] + center[1] for p in lower]
    ys = [p[2] + center[2] for p in lower]
    zs = [p[3] + center[3] for p in lower]
    PyPlot.plot3D(xs, ys, zs, options)

    upper = [p .+ [0,0,length] for p in lower]
    xs = [p[1] + center[1] for p in upper]
    ys = [p[2] + center[2] for p in upper]
    zs = [p[3] + center[3] for p in upper]
    PyPlot.plot3D(xs, ys, zs, options)

    for (i, l) in enumerate(lower)
        u = upper[i]
        xs = [l[1], u[1]] .+ center[1]
        ys = [l[2], u[2]] .+ center[2]
        zs = [l[3], u[3]] .+ center[3]
        PyPlot.plot3D(xs, ys, zs, options)
    end
end

function plot_lattice(A, atoms; radius=2, origin=[0.0,0.0,0.0])
    xs = []
    ys = []
    zs = []
    rnge = -ceil(10*radius):ceil(10*radius)
    for kn in rnge, ln in rnge, mn in rnge
        new = [
               a + kn * A[:,1] + ln * A[:,2] + mn * A[:,3]
               for a in atoms
              ]
        new = [n for n in new if maximum(abs.(n .- origin)) <= radius]
        append!(xs, [v[1] for v in new])
        append!(ys, [v[2] for v in new])
        append!(zs, [v[3] for v in new])
    end
    PyPlot.scatter3D(xs, ys, zs)
end

function test()
    a = 1
    A = a / 2 * [[1,1,0]  [0,1,1]  [1,0,1]]
    τ = a / 8 .* [1, 1, 1]
    atoms = [τ, -τ]
    plot_lattice(A, atoms)
    plot_cube(A, "y-")
    plot_cube(Matrix(Diagonal(a .* ones(3))), "r-")
end
