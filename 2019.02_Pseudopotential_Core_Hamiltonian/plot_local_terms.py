#!/usr/bin/env python3
import numpy as np

from matplotlib import pyplot as plt

from sympy import erf, exp, sin, sqrt, symbols

a, k, r = symbols('a k r', positive=True)


def integr_erf(r, k, a):
    return -(r*sin(r*k)/k) * erf(r/sqrt(2)/a)/r


def integr_exp(r, k, a):
    return exp(-(r/a)**2/2)


rs = np.linspace(0, 100, 1001)
i1 = integr_erf(r, 1, 1)
i2 = integr_exp(r, 1, 1)

plt.plot(rs, np.array([i1.evalf(subs={r: rval}) for rval in rs]), label="erf")
plt.plot(rs, np.array([i2.evalf(subs={r: rval}) for rval in rs]), label="exp")
plt.plot(rs, rs <= 1, label="1_{<=1}")
plt.legend()
plt.show()
