#!/usr/bin/env python3
from sympy import S, exp, expand_func, gamma, integrate, oo, pi, sqrt, symbols
from sympy.functions.special.bessel import jn


def proj(r, a, l, i):
    return sqrt(2)*r**(l+2*(S(i)-1)) * exp(-r**2/a**2/2) / (
        a**(S(l)+(4*S(i)-1)/S(2)) * sqrt(gamma(S(l) + (4*S(i)-1)/S(2)))
    )


def integrand_extra(r, k, l):
    return 4 * pi * S(1j)**l * r**2 * expand_func(jn(l, k * r))


def integrand(r, k, a, l, i):
    return (proj(r, a, l, i) * integrand_extra(r, k, l)).simplify()


def derive_stuff(l, i):
    a, k, r = symbols('a k r', positive=True)

    print()
    print("l={}    i={}".format(l,i))
    print()

    print("Projector     ", proj(r, a, l, i))
    print("Normalisation ", integrate((r * proj(r, a, l, i))**2, (r, 0, oo)))
    print("Integrand     ", integrand(r, k, a, l, i))

    res = integrate(integrand(r, k, a, l, i), (r, 0, oo))
    print("FT            ", res)


derive_stuff(l=0, i=1)
derive_stuff(l=0, i=2)
derive_stuff(l=1, i=1)
