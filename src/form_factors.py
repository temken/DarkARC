import os
import sys

from tabulation import *

import numpy as np
import mpmath as mp
from sympy import N
from sympy.physics.wigner import wigner_3j, gaunt
from scipy.interpolate import RectBivariateSpline
from scipy.special import sph_harm

from tabulation import qMin, qMax, kMin, kMax, lPrime_max

# The standard atomic and ionization form factor
# def tabulate_atomic_form_factor_1(element, n, l, m, lPrime,mPrime, gridsize):
#     filepath = "../data/atomic_formfactor_1/" + element.Shell_Name(n, l) + str(m)+"_"+str(lPrime)+str(mPrime) + ".txt"
#     if os.path.exists(filepath) == False:
#         FF = [[0 for x in range(gridsize)] for y in range(gridsize)]
#         for L in range(abs(l-lPrime),l+lPrime+1):
#             radial_integral = np.loadtxt("../data/integral_1/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
#             for ki in range(gridsize):
#                 for qi in range(gridsize):
#                     FF[ki][qi] += pow(1j,L) * radial_integral[ki][qi] * pow(-1,mPrime) * (2*L+1) * math.sqrt((2*l+1)*(2*lPrime+1)) * wigner_3j(l, lPrime, L, 0, 0, 0) * wigner_3j(l, lPrime, L, m, -mPrime, 0)
#         np.savetxt(filepath, FF)

def tabulate_ionization_form_factor_1(element, n, l, gridsize):
    filepath = "../data/form_factor_1/" + element.Shell_Name(n, l) + ".txt"
    if os.path.exists(filepath) == False:
        FF2 = [[0 for x in range(gridsize)] for y in range(gridsize)]
        kGrid = np.logspace(np.log10(kMin), np.log10(kMax), gridsize)
        for lPrime in range(lPrime_max + 1):
            for L in range(abs(l - lPrime), l + lPrime + 1):
                radial_integral = np.loadtxt("../data/integral_1/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
                for ki in range(gridsize):
                    k = kGrid[ki]
                    for qi in range(gridsize):
                        FF2[ki][qi] += 4 * pow(k, 3) / pow(2 * np.pi, 3) * (2 * l + 1) * (2 * lPrime + 1) * (2 * L + 1) * pow(radial_integral[ki][qi], 2) * pow(wigner_3j(l, lPrime, L, 0, 0, 0), 2)
        np.savetxt(filepath, FF2)

def atomic_formfactor_scalar(element,n,l,m,kPrime,lPrime,mPrime,q):
    f12 = 0
    dlog10k = np.log10(kMax/kMin) / (gridsize - 1)
    dlog10q = np.log10(qMax/qMin) / (gridsize - 1)
    ki =int( round(np.log10(kPrime/kMin) / dlog10k) )
    qi =int( round(np.log10(q/qMin) / dlog10q) )
    for L in range(abs(l-lPrime),l+lPrime+1):
        radial_integral_1 = np.loadtxt("../data/integral_1/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
        f12 += np.sqrt(4*np.pi) * pow(1j,L) * radial_integral_1[ki][qi] * (-1)**mPrime * np.sqrt(2*L+1) * N(gaunt(l,lPrime,L,m,-mPrime,0))
    return f12

def atomic_formfactor_vector(component,element,n,l,m,kPrime,lPrime,mPrime,q):
    f12 = 0
    dlog10k = np.log10(kMax/kMin) / (gridsize - 1)
    dlog10q = np.log10(qMax/qMin) / (gridsize - 1)
    ki =int( round(np.log10(kPrime/kMin) / dlog10k) )
    qi =int( round(np.log10(q/qMin) / dlog10q) )
    for lHat in [l-1,l+1]:
        for L in range(abs(lHat-lPrime),lHat+lPrime+1):
            radial_integral_2 = np.loadtxt("../data/integral_2/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
            radial_integral_3 = np.loadtxt("../data/integral_3/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
            # for M in range(-L,L+1):
            for mHat in range(m-1,m+2):
                f12 += pow(1j,L) * (grad_Ylm_coefficient(component,l,m,lHat,mHat) * radial_integral_2[ki][qi] + er_Ylm_coefficient(component,l,m,lHat,mHat) * radial_integral_3[ki][qi]) * (-1)**mPrime * np.sqrt(4*np.pi) * np.sqrt(2*L+1) * N(gaunt(lHat,lPrime,L,mHat,-mPrime,0))
                # print("\t",[component,l,m,lHat,mHat],grad_Ylm_coefficient(component,l,m,lHat,mHat),er_Ylm_coefficient(component,l,m,lHat,mHat),N(gaunt(lHat,lPrime,L,mHat,-mPrime,0)))
    return 1j / mElectron *f12

def grad_Ylm(l,m,theta,phi):
    unit_vectors =[ np.array([1,0,0]) , np.array([0,1,0]) ,  np.array([0,0,1]) ]
    gradYlm = np.array([0+0j,0+0j,0+0j])
    for i in range(3):
        for lHat in range(l-1,l+2,2):
            for mHat in range(m-1,m+2):
                if mHat in range(-lHat,lHat+1): 
                    gradYlm[i] += grad_Ylm_coefficient(i+1,l,m,lHat,mHat) * sph_harm(mHat,lHat,phi,theta)

    return gradYlm

def grad_Ylm_coefficient(component,l,m,lHat,mHat):
    if component not in [1,2,3]:
        sys.exit("Error in spherical_harmonic_gradient_coefficient(): Component out of bound.")
    if component == 1:
        if (lHat not in [l-1,l+1]) or (mHat not in [m-1,m+1]):
            return 0.0
        elif lHat == l+1 and mHat == m+1:
            return l/2 * math.sqrt((l+m+1) * (l+m+2) / (2*l+3) / (2*l+1))
        elif lHat == l+1 and mHat == m-1:
            return -l/2 * math.sqrt((l-m+1) * (l-m+2) / (2*l+3) / (2*l+1))
        elif lHat == l-1 and mHat == m+1:
            return (l+1)/2 * math.sqrt((l-m-1) * (l-m) / (2*l-1) / (2*l+1))
        elif lHat == l-1 and mHat == m-1:
            return -(l+1)/2 * math.sqrt((l+m-1) * (l+m) / (2*l-1) / (2*l+1))
    elif component == 2:
        if (lHat not in [l-1,l+1]) or (mHat not in [m-1,m+1]):
            return 0.0
        elif lHat == l+1 and mHat == m+1:
            return -1j*l/2 * math.sqrt((l+m+1) * (l+m+2) / (2*l+3) / (2*l+1))
        elif lHat == l+1 and mHat == m-1:
            return -1j*l/2 * math.sqrt((l-m+1) * (l-m+2) / (2*l+3) / (2*l+1))
        elif lHat == l-1 and mHat == m+1:
            return -1j*(l+1)/2 * math.sqrt((l-m-1) * (l-m) / (2*l-1) / (2*l+1))
        elif lHat == l-1 and mHat == m-1:
            return -1j*(l+1)/2 * math.sqrt((l+m-1) * (l+m) / (2*l-1) / (2*l+1))
    elif component == 3:
        if (lHat not in [l-1,l+1]) or (mHat != m):
            return 0.0
        elif lHat == l+1:
            return  -l * math.sqrt((l-m+1) * (l+m+1) / (2*l+3) / (2*l+1))
        elif lHat == l-1:
            return (1+l) * math.sqrt((l-m) * (l+m) / (2*l-1) / (2*l+1))


def er_Ylm(l,m,theta,phi):
    unit_vectors =[ np.array([1,0,0]) , np.array([0,1,0]) ,  np.array([0,0,1]) ]
    erYlm = np.array([0+0j,0+0j,0+0j])
    for i in range(3):
        for lHat in range(l-1,l+2,2):
            for mHat in range(m-1,m+2):
                if mHat in range(-lHat,lHat+1): 
                    erYlm[i] += er_Ylm_coefficient(i+1,l,m,lHat,mHat) * sph_harm(mHat,lHat,phi,theta)

    return erYlm


def er_Ylm_coefficient(component,l,m,lHat,mHat):
    if component not in [1,2,3]:
        sys.exit("Error in spherical_harmonic_gradient_coefficient(): Component out of bound.")
    if component == 1:
        if (lHat not in [l-1,l+1]) or (mHat not in [m-1,m+1]):
            return 0.0
        elif lHat == l+1 and mHat == m+1:
            return -1./2 * math.sqrt((l+m+1) * (l+m+2) / (2*l+3) / (2*l+1))
        elif lHat == l+1 and mHat == m-1:
            return 1./2 * math.sqrt((l-m+1) * (l-m+2) / (2*l+3) / (2*l+1))
        elif lHat == l-1 and mHat == m+1:
            return 1./2 * math.sqrt((l-m-1) * (l-m) / (2*l-1) / (2*l+1))
        elif lHat == l-1 and mHat == m-1:
            return -1./2 * math.sqrt((l+m-1) * (l+m) / (2*l-1) / (2*l+1))
    elif component == 2:
        if (lHat not in [l-1,l+1]) or (mHat not in [m-1,m+1]):
            return 0.0
        elif lHat == l+1 and mHat == m+1:
            return 1j/2 * math.sqrt((l+m+1) * (l+m+2) / (2*l+3) / (2*l+1))
        elif lHat == l+1 and mHat == m-1:
            return 1j/2 * math.sqrt((l-m+1) * (l-m+2) / (2*l+3) / (2*l+1))
        elif lHat == l-1 and mHat == m+1:
            return -1j/2 * math.sqrt((l-m-1) * (l-m) / (2*l-1) / (2*l+1))
        elif lHat == l-1 and mHat == m-1:
            return -1j/2 * math.sqrt((l+m-1) * (l+m) / (2*l-1) / (2*l+1))
    elif component == 3:
        if (lHat not in [l-1,l+1]) or (mHat != m):
            return 0.0
        elif lHat == l+1:
            return  math.sqrt((l-m+1) * (l+m+1) / (2*l+3) / (2*l+1))
        elif lHat == l-1:
            return math.sqrt((l-m) * (l+m) / (2*l-1) / (2*l+1))
