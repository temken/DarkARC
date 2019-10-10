import os
import sys

import numpy as np
import mpmath as mp
from sympy.physics.wigner import wigner_3j
from scipy.interpolate import RectBivariateSpline
from scipy.special import sph_harm
import math

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

def tabulate_ionization_form_factor_2(element, n, l, gridsize):
    for component in range(1,4):
        filepath = "../data/form_factor_2/" + element.Shell_Name(n, l)+"_"+ str(component) + ".txt"
        if os.path.exists(filepath) == False:
            kGrid = np.logspace(np.log10(kMin), np.log10(kMax), gridsize)
            FF = np.array([[0.0+0.0j for x in range(gridsize)] for y in range(gridsize)])
            for lPrime in range(0,lPrime_max+1):
                for lHat in [l-1,l+1]:
                    for L in range(abs(lHat-lPrime),lHat+lPrime+1):
                        radial_integral_2 = np.loadtxt("../data/integral_2/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
                        radial_integral_3 = np.loadtxt("../data/integral_3/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
                        for m in range(-l,l+1):
                            for mPrime in range(-lPrime,lPrime+1):
                                for mHat in range(m-1,m+2):
                                    if mHat ==  mPrime:
                                        print("i=",component,"\tn=",n,"\tl=",l,"\tm=",m,"\tl'=",lPrime,"\tm'=",mPrime,"\tlHat=",lHat,"\tmHat=",mHat,"\tL=",L)
                                        for ki in range(gridsize):
                                            k = kGrid[ki]
                                            for qi in range(gridsize):
                                                FF[ki][qi] += 4j * pow(k, 3) / pow(2 * np.pi, 3) * pow(1j,L) * (grad_Ylm_coefficient(component,l,m,lHat,mHat) * radial_integral_2[ki][qi] + er_Ylm_coefficient(component,l,m,lHat,mHat) * radial_integral_3[ki][qi]) * pow(-1,mPrime) * (2*L+1) * math.sqrt((2*lPrime+1) * (2*lHat+1)) * wigner_3j(lHat, lPrime, L, 0, 0, 0) * wigner_3j(lHat, lPrime, L, mHat, -mPrime, 0)
            np.savetxt(filepath, FF)

def tabulate_ionization_form_factor_3(element, n, l, gridsize):
    for component in range(1,4):
        filepath = "../data/form_factor_3/" + element.Shell_Name(n, l)+"_"+ str(component) + ".txt"
        if os.path.exists(filepath) == False:
            kGrid = np.logspace(np.log10(kMin), np.log10(kMax), gridsize)
            FF = np.array([[0.0+0.0j for x in range(gridsize)] for y in range(gridsize)])
            for lPrime in range(0,lPrime_max+1):
                for Lprime in range(abs(l-lPrime),l+lPrime+1):
                    radial_integral_1 = np.loadtxt("../data/integral_1/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(Lprime) + ".txt")
                    for lHat in [l-1,l+1]:
                        for L in range(abs(lHat-lPrime),lHat+lPrime+1):
                            radial_integral_2 = np.loadtxt("../data/integral_2/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
                            radial_integral_3 = np.loadtxt("../data/integral_3/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
                            for m in range(-l,l+1):
                                for mPrime in range(-lPrime,lPrime+1): 
                                    if m==mPrime:
                                        if component == 1 or component == 2:
                                            mHatRange = [m-1,m+1]
                                        else:
                                            mHatRange = [m]
                                        for mHat in mHatRange:
                                            if mHat ==  mPrime:
                                                print("i=",component,"\tn=",n,"\tl=",l,"\tm=",m,"\tl'=",lPrime,"\tm'=",mPrime,"\tlHat=",lHat,"\tmHat=",mHat,"\tL=",L,"\tL'=",Lprime)
                                                for ki in range(gridsize):
                                                    k = kGrid[ki]
                                                    for qi in range(gridsize):
                                                        f12 = pow(1j,Lprime) * radial_integral_1[ki][qi] * (2*Lprime+1) * math.sqrt((2*l+1) * (2*lPrime+1)) * wigner_3j(l, lPrime, Lprime, 0, 0, 0) * wigner_3j(l, lPrime, Lprime, m, -mPrime, 0)
                                                        F12 = 1j * pow(1j,L) * (grad_Ylm_coefficient(component,l,m,lHat,mHat) * radial_integral_2[ki][qi] + er_Ylm_coefficient(component,l,m,lHat,mHat) * radial_integral_3[ki][qi]) * pow(-1,mPrime) * (2*L+1) * math.sqrt((2*lPrime+1) * (2*lHat+1)) * wigner_3j(lHat, lPrime, L, 0, 0, 0) * wigner_3j(lHat, lPrime, L, mHat, -mPrime, 0)
                                                        FF[ki][qi] += 4 * pow(k, 3) / pow(2 * np.pi, 3) * F12 * np.conj(f12)

            np.savetxt(filepath, FF)
                    



# def interpolate_standard_form_factor(element, n, l):
#     filepath = "../data/standard_formfactor/" + element.Shell_Name(n, l) + ".txt"
#     if os.path.exists(filepath):
#         zGrid = np.loadtxt(filepath)
#         steps = len(zGrid)
#         qGrid = np.logspace(np.log10(qMin), np.log10(qMax), steps)
#         kGrid = np.logspace(np.log10(kMin), np.log10(kMax), steps)
#         return RectBivariateSpline(kGrid, qGrid, zGrid)

# The new vectorial atomic form factor
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
