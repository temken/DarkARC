import os
import sys

import numpy as np
from sympy.physics.wigner import wigner_3j
from scipy.interpolate import RectBivariateSpline

from tabulation import qMin, qMax, kMin, kMax, lPrime_max

# The standard ionization form factor
def tabulate_standard_form_factor(element, n, l, gridsize):
    filepath = "../data/standard_formfactor/" + element.Shell_Name(n, l) + ".txt"
    if os.path.exists(filepath) == False:
        FF2 = [[0 for x in range(gridsize)] for y in range(gridsize)]
        kGrid = np.logspace(np.log10(kMin), np.log10(kMax), gridsize)
        for lPrime in range(lPrime_max + 1):
            for L in range(abs(l - lPrime), l + lPrime + 1):
                print(lPrime, L)

                # Import integral table
                radial_integral = np.loadtxt("../data/integral_1/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
                for ki in range(gridsize):
                    k = kGrid[ki]
                    for qi in range(gridsize):
                        FF2[ki][qi] += 4 * pow(k, 3) / pow(2 * np.pi, 3) * (2 * l + 1) * (2 * lPrime + 1) * (2 * L + 1) * pow(radial_integral[ki][qi], 2) * pow(wigner_3j(l, lPrime, L, 0, 0, 0), 2)
        np.savetxt(filepath, FF2)


def interpolate_standard_form_factor(element, n, l):
    filepath = "../data/standard_formfactor/" + element.Shell_Name(n, l) + ".txt"
    if os.path.exists(filepath):
        zGrid = np.loadtxt(filepath)
        steps = len(zGrid)
        qGrid = np.logspace(np.log10(qMin), np.log10(qMax), steps)
        kGrid = np.logspace(np.log10(kMin), np.log10(kMax), steps)
        return RectBivariateSpline(kGrid, qGrid, zGrid)
