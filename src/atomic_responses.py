import os
import sys
import time

import numpy as np
from sympy.physics.wigner import gaunt
from scipy.special import sph_harm

from units import *
from wave_functions import *
from vector_spherical_harmonics import *

from radial_integrals_tabulation import qMin, qMax, kMin, kMax, lPrime_max, gridsize

# THIS FUNCTION REQUIRES THE FINISHED TABULATION OF THE RADIAL INTEGRALS
def main():
	start_tot = time.time()

	####################################################################################
	print("\nCompare the two alternative implementations of the four atomic response functions.")
	element = Ar
	n = 3
	l = 0
	kPrime = 0.1 * keV
	q1 = 0 * keV
	q2 = 0 * keV
	q3 = 1 * keV
	q = np.sqrt(q1*q1+q2*q2+q3*q3)

	for response in range(3,5):
		print(response,atomic_response_function(response,element,n,l,kPrime,q),atomic_response_alternative(response,element,n,l,kPrime,q1,q2,q3))

	####################################################################################
	# print("\nTest if sum_{m mPrime}f_{1->2}F_{1->2} is parallel to q, which is relevant for atomic response function W_2.")
	# element = Xe
	# n = 4
	# l = 0
	# kPrime = 1 * keV
	# q1 = 1 * keV
	# q2 = 10 * keV
	# q3 = 100.0 * keV
	# q = np.sqrt(q1*q1+q2*q2+q3*q3)
	# print("q=",q/keV,"keV")
	# result = [0,0,0]
	# for m in range(-l,l+1):
	# 	for lPrime in range(7):
	# 		for mPrime in range(-lPrime,lPrime+1):
	# 			f12s = atomic_formfactor_scalar_alternative(element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
	# 			result[0] += np.real(f12s * np.conj(atomic_formfactor_vector_alternative(1,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)))
	# 			result[1] += np.real(f12s * np.conj(atomic_formfactor_vector_alternative(2,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)))
	# 			result[2] += np.real(f12s * np.conj(atomic_formfactor_vector_alternative(3,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)))

	# response_2 = 4 * pow(kPrime, 3) / pow(2 * np.pi, 3) * q/mElectron * np.sqrt(result[0]*result[0]+result[1]*result[1]+result[2]*result[2])
	# print("A = sum_{m mPrime}f_{1->2}F_{1->2} = ",result)
	# cross_product = [result[1]*q3-result[2]*q2 , result[2]*q1-result[0]*q3, result[0]*q2-result[1]*q1]
	# print("Cross product (A x q) =", cross_product,"\n")

	# print("W_2 = ",response_2)

	# print("\nTest if the norm of this vector is accurately reproduced using the frame in which the z axis is pointing along q.")
	# result = [0,0,0]
	# for m in range(-l,l+1):
	# 	for lPrime in range(7):
	# 		for mPrime in range(-lPrime,lPrime+1):
	# 			f12s = atomic_formfactor_scalar(element,n,l,m,kPrime,lPrime,mPrime,q)
	# 			result[2] += np.real(f12s * np.conj(atomic_formfactor_vector(3,element,n,l,m,kPrime,lPrime,mPrime,q)))

	# response_2_alternative = 4 * pow(kPrime, 3) / pow(2 * np.pi, 3) * q/mElectron * np.sqrt(result[2]*result[2])
	# print("W_2 = ",response_2_alternative,"\n")

	####################################################################################
	end_tot = time.time()
	print("\nProcessing time:\t", end_tot - start_tot, "s\n")
	####################################################################################

def atomic_response_function(response,element,n,l,kPrime,q):
	Wion = 0
	for lPrime in range(lPrime_max+1):
		for m  in range(-l,l+1):
			for mPrime in range(-lPrime,lPrime+1):
				Wion += electronic_transition_response(response,element,n,l,m,kPrime,lPrime,mPrime,q)
	if response == 3:
		return 4 * pow(kPrime, 3) / pow(2 * np.pi, 3) * np.linalg.norm(Wion)
	else:
		return 4 * pow(kPrime, 3) / pow(2 * np.pi, 3) * Wion

def electronic_transition_response(response,element,n,l,m,kPrime,lPrime,mPrime,q):
	W12 = 0
	if response == 1:
		f12scalar = atomic_formfactor_scalar(element,n,l,m,kPrime,lPrime,mPrime,q)
		W12 = f12scalar * np.conj(f12scalar)
	elif response == 2:
		f12scalar = atomic_formfactor_scalar(element,n,l,m,kPrime,lPrime,mPrime,q)
		f12vector = atomic_formfactor_vector(3,element,n,l,m,kPrime,lPrime,mPrime,q)
		W12 = q/mElectron * f12scalar * np.conj(f12vector)
	elif response == 3:
		f12vector1 = atomic_formfactor_vector(1,element,n,l,m,kPrime,lPrime,mPrime,q)
		f12vector2 = atomic_formfactor_vector(2,element,n,l,m,kPrime,lPrime,mPrime,q)
		f12vector3 = atomic_formfactor_vector(3,element,n,l,m,kPrime,lPrime,mPrime,q)
		W12 = f12vector1 * np.conj(f12vector1) + f12vector2 * np.conj(f12vector2) + f12vector3 * np.conj(f12vector3)
	elif response == 4:
		f12vector3 = atomic_formfactor_vector(3,element,n,l,m,kPrime,lPrime,mPrime,q)
		qf = q/mElectron * f12vector3
		W12 = qf * np.conj(qf)
	else:
		sys.exit("Error in atomic_formfactor(): Response out of bound.")
	return np.real(W12)


# Scalar atomic form factor
def atomic_formfactor_scalar(element,n,l,m,kPrime,lPrime,mPrime,q):
	f12 = 0
	dlog10k = np.log10(kMax/kMin) / (gridsize - 1)
	dlog10q = np.log10(qMax/qMin) / (gridsize - 1)
	ki =int( round(np.log10(kPrime/kMin) / dlog10k) )
	qi =int( round(np.log10(q/qMin) / dlog10q) )
	for L in range(abs(l-lPrime),l+lPrime+1):
		radial_integral_1 = np.loadtxt("../data/radial_integral_1/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
		f12 += np.sqrt(4*np.pi) * pow(1j,L) * radial_integral_1[ki][qi] * (-1)**mPrime * np.sqrt(2*L+1) * float(gaunt(l,lPrime,L,m,-mPrime,0))
	return f12

# Vectorial atomic form factor
def atomic_formfactor_vector(component,element,n,l,m,kPrime,lPrime,mPrime,q):
	f12 = 0
	dlog10k = np.log10(kMax/kMin) / (gridsize - 1)
	dlog10q = np.log10(qMax/qMin) / (gridsize - 1)
	ki =int( round(np.log10(kPrime/kMin) / dlog10k) )
	qi =int( round(np.log10(q/qMin) / dlog10q) )
	for lHat in [l-1,l+1]:
		for L in range(abs(lHat-lPrime),lHat+lPrime+1):
			radial_integral_2 = np.loadtxt("../data/radial_integral_2/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
			radial_integral_3 = np.loadtxt("../data/radial_integral_3/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
			for mHat in range(m-1,m+2):
				f12 += pow(1j,L) * (VSH_coefficients_Y(component,l,m,lHat,mHat) * radial_integral_2[ki][qi] + VSH_coefficients_Psi(component,l,m,lHat,mHat) * radial_integral_3[ki][qi]) * (-1)**mPrime * np.sqrt(4*np.pi) * np.sqrt(2*L+1) * float(gaunt(lHat,lPrime,L,mHat,-mPrime,0))
	return 1j / mElectron *f12


# Alternative definition of functions with full 3 vector for q.

def atomic_response_alternative(response,element,n,l,kPrime,q1,q2,q3):
	Wion = 0
	for m  in range(-l,l+1):
		for lPrime in range(lPrime_max+1):
			for mPrime in range(-lPrime,lPrime+1):
				Wion += electronic_transition_response_alternative(response,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
	if response == 3:
		return 4 * pow(kPrime, 3) / pow(2 * np.pi, 3) * np.linalg.norm(Wion)
	else:
		return 4 * pow(kPrime, 3) / pow(2 * np.pi, 3) * Wion

def electronic_transition_response_alternative(response,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3):
	W12 = 0
	if response == 1:
		f12scalar = atomic_formfactor_scalar_alternative(element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		W12 = f12scalar * np.conj(f12scalar)
	elif response == 2:
		f12scalar = atomic_formfactor_scalar_alternative(element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		f12vector1 = atomic_formfactor_vector_alternative(1,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		f12vector2 = atomic_formfactor_vector_alternative(2,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		f12vector3 = atomic_formfactor_vector_alternative(3,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		W12 = 1/mElectron *( q1 * f12scalar * np.conj(f12vector1) +q2 * f12scalar * np.conj(f12vector2)+q3 * f12scalar * np.conj(f12vector3) )
	elif response == 3:
		f12vector1 = atomic_formfactor_vector_alternative(1,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		f12vector2 = atomic_formfactor_vector_alternative(2,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		f12vector3 = atomic_formfactor_vector_alternative(3,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		W12 = f12vector1 * np.conj(f12vector1) + f12vector2 * np.conj(f12vector2) + f12vector3 * np.conj(f12vector3)
	elif response == 4:
		f12vector1 = atomic_formfactor_vector_alternative(1,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		f12vector2 = atomic_formfactor_vector_alternative(2,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		f12vector3 = atomic_formfactor_vector_alternative(3,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		qf = q1/mElectron * f12vector1 + q2/mElectron * f12vector2 + q3/mElectron * f12vector3
		W12 = qf * np.conj(qf)
	else:
		sys.exit("Error in atomic_formfactor(): Response out of bound.")
	return np.real(W12)

def atomic_formfactor_scalar_alternative(element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3):
	f12 = 0
	q = np.sqrt(q1*q1+q2*q2+q3*q3)
	theta_q = np.arccos(q3/q)
	phi_q =  np.arctan2(q2,q1)
	dlog10k = np.log10(kMax/kMin) / (gridsize - 1)
	dlog10q = np.log10(qMax/qMin) / (gridsize - 1)
	ki =int( round(np.log10(kPrime/kMin) / dlog10k) )
	qi =int( round(np.log10(q/qMin) / dlog10q) )
	for L in range(abs(l-lPrime),l+lPrime+1):
		radial_integral_1 = np.loadtxt("../data/radial_integral_1/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
		for M in range(-L,L+1):
			if L >= 0 and abs(M) <=L:
				new = 4*np.pi * pow(1j,L) * np.conj(sph_harm(M,L,phi_q,theta_q)) * radial_integral_1[ki][qi] * (-1)**mPrime * float(gaunt(l,lPrime,L,m,-mPrime,M))
				f12 += new
	return f12

def atomic_formfactor_vector_alternative(component,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3):
	f12 = 0
	q = np.sqrt(q1*q1+q2*q2+q3*q3)
	theta_q = np.arccos(q3/q)
	phi_q =  np.arctan2(q2,q1)
	# print("angles:",theta_q/np.pi,phi_q/np.pi)
	dlog10k = np.log10(kMax/kMin) / (gridsize - 1)
	dlog10q = np.log10(qMax/qMin) / (gridsize - 1)
	ki =int( round(np.log10(kPrime/kMin) / dlog10k) )
	qi =int( round(np.log10(q/qMin) / dlog10q) )
	for lHat in [l-1,l+1]:
		for L in range(abs(lHat-lPrime),lHat+lPrime+1):
			radial_integral_2 = np.loadtxt("../data/radial_integral_2/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
			radial_integral_3 = np.loadtxt("../data/radial_integral_3/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
			for M in range(-L,L+1):
				for mHat in range(m-1,m+2):
					new =  1.0/mElectron * 4*np.pi * pow(1j,L+1) * np.conj(sph_harm(M,L,phi_q,theta_q)) * (VSH_coefficients_Y(component,l,m,lHat,mHat) * radial_integral_2[ki][qi] + VSH_coefficients_Psi(component,l,m,lHat,mHat) * radial_integral_3[ki][qi]) * (-1)**mPrime * float(gaunt(lHat,lPrime,L,mHat,-mPrime,M))
					f12 += new
	return f12

if __name__ == "__main__":
	main()
