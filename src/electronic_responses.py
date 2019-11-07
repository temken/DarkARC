import os
import sys

import numpy as np
from sympy.physics.wigner import gaunt
from scipy.special import sph_harm

from units import *
from radial_integrals_tabulation import qMin, qMax, kMin, kMax, lPrime_max, gridsize


def electronic_ionization_response(response,element,n,l,kPrime,q):
	Wion = 0
	for lPrime in range(lPrime_max+1):
		for m  in range(-l,l+1):
			for mPrime in range(-lPrime,lPrime+1):
				Wion += electronic_transition_response(response,element,n,l,m,kPrime,lPrime,mPrime,q)
	if response == 3:
		return 4 * pow(kPrime, 3) / pow(2 * np.pi, 3) * np.linalg.norm(Wion)
	else:
		return 4 * pow(kPrime, 3) / pow(2 * np.pi, 3) * Wion

def electronic_ionization_response_2(response,element,n,l,kPrime,q1,q2,q3):
	Wion = 0
	for m  in range(-l,l+1):
		for lPrime in range(lPrime_max+1):
			for mPrime in range(-lPrime,lPrime+1):
				Wion += electronic_transition_response_2(response,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
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
		f12scalar = atomic_formfactor_scalar(element,n,l,m,kPrime,lPrime,mPrime,q)
		f12vector1 = atomic_formfactor_vector(1,element,n,l,m,kPrime,lPrime,mPrime,q)
		f12vector2 = atomic_formfactor_vector(2,element,n,l,m,kPrime,lPrime,mPrime,q)
		f12vector3 = atomic_formfactor_vector(3,element,n,l,m,kPrime,lPrime,mPrime,q)
		W12 = [ f12scalar * np.conj(f12vector1) , f12scalar * np.conj(f12vector2) , f12scalar * np.conj(f12vector3) ]
	elif response == 4:
		f12vector1 = atomic_formfactor_vector(1,element,n,l,m,kPrime,lPrime,mPrime,q)
		f12vector2 = atomic_formfactor_vector(2,element,n,l,m,kPrime,lPrime,mPrime,q)
		f12vector3 = atomic_formfactor_vector(3,element,n,l,m,kPrime,lPrime,mPrime,q)
		W12 = f12vector1 * np.conj(f12vector1) + f12vector2 * np.conj(f12vector2) + f12vector3 * np.conj(f12vector3)
	elif response == 5:
		f12vector3 = atomic_formfactor_vector(3,element,n,l,m,kPrime,lPrime,mPrime,q)
		qf = q/mElectron * f12vector3
		W12 = qf * np.conj(qf)
	elif response == 6:
		f12vector1 = atomic_formfactor_vector(1,element,n,l,m,kPrime,lPrime,mPrime,q)
		f12vector2 = atomic_formfactor_vector(2,element,n,l,m,kPrime,lPrime,mPrime,q)
		W12 = 2.0 * q/mElectron * np.imag(f12vector2 * np.conj(f12vector1))
	else:
		sys.exit("Error in atomic_formfactor(): Response out of bound.")
	return np.real(W12)

def electronic_transition_response_2(response,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3):
	W12 = 0
	if response == 1:
		f12scalar = atomic_formfactor_scalar_2(element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		W12 = f12scalar * np.conj(f12scalar)
	elif response == 2:
		f12scalar = atomic_formfactor_scalar_2(element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		f12vector1 = atomic_formfactor_vector_2(1,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		f12vector2 = atomic_formfactor_vector_2(2,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		f12vector3 = atomic_formfactor_vector_2(3,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		W12 = 1/mElectron *( q1 * f12scalar * np.conj(f12vector1) +q2 * f12scalar * np.conj(f12vector2)+q3 * f12scalar * np.conj(f12vector3) )
	elif response == 3:
		f12scalar = atomic_formfactor_scalar_2(element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		f12vector1 = atomic_formfactor_vector_2(1,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		f12vector2 = atomic_formfactor_vector_2(2,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		f12vector3 = atomic_formfactor_vector_2(3,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		W12 = [ f12scalar * np.conj(f12vector1) , f12scalar * np.conj(f12vector2) , f12scalar * np.conj(f12vector3) ]
	elif response == 4:
		f12vector1 = atomic_formfactor_vector_2(1,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		f12vector2 = atomic_formfactor_vector_2(2,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		f12vector3 = atomic_formfactor_vector_2(3,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		W12 = f12vector1 * np.conj(f12vector1) + f12vector2 * np.conj(f12vector2) + f12vector3 * np.conj(f12vector3)
	elif response == 5:
		f12vector1 = atomic_formfactor_vector_2(1,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		f12vector2 = atomic_formfactor_vector_2(2,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		f12vector3 = atomic_formfactor_vector_2(3,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		qf = q1/mElectron * f12vector1 + q2/mElectron * f12vector2 + q3/mElectron * f12vector3
		W12 = qf * np.conj(qf)
	elif response == 6:
		f12vector1 = atomic_formfactor_vector_2(1,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		f12vector2 = atomic_formfactor_vector_2(2,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		f12vector3 = atomic_formfactor_vector_2(3,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
		W12 = q1/mElectron * 1j * (f12vector2 * np.conj(f12vector3)-f12vector3 * np.conj(f12vector2)) + q2/mElectron * 1j * (f12vector3 * np.conj(f12vector1)-f12vector1 * np.conj(f12vector3))+ q3/mElectron * 1j * (f12vector1 * np.conj(f12vector2)-f12vector2 * np.conj(f12vector1))
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
		radial_integral_1 = np.loadtxt("../data/integral_1/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
		f12 += np.sqrt(4*np.pi) * pow(1j,L) * radial_integral_1[ki][qi] * (-1)**mPrime * np.sqrt(2*L+1) * float(gaunt(l,lPrime,L,m,-mPrime,0))
	return f12
def atomic_formfactor_scalar_2(element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3):
	f12 = 0
	q = np.sqrt(q1*q1+q2*q2+q3*q3)
	theta_q = np.arccos(q3/q)
	phi_q =  np.arctan2(q2,q1)
	dlog10k = np.log10(kMax/kMin) / (gridsize - 1)
	dlog10q = np.log10(qMax/qMin) / (gridsize - 1)
	ki =int( round(np.log10(kPrime/kMin) / dlog10k) )
	qi =int( round(np.log10(q/qMin) / dlog10q) )
	for L in range(abs(l-lPrime),l+lPrime+1):
		radial_integral_1 = np.loadtxt("../data/integral_1/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
		for M in range(-L,L+1):
			if L >= 0 and abs(M) <=L:
				new = 4*np.pi * pow(1j,L) * np.conj(sph_harm(M,L,phi_q,theta_q)) * radial_integral_1[ki][qi] * (-1)**mPrime * float(gaunt(l,lPrime,L,m,-mPrime,M))
				f12 += new
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
			radial_integral_2 = np.loadtxt("../data/integral_2/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
			radial_integral_3 = np.loadtxt("../data/integral_3/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
			for mHat in range(m-1,m+2):
				f12 += pow(1j,L) * (grad_Ylm_coefficient(component,l,m,lHat,mHat) * radial_integral_2[ki][qi] + er_Ylm_coefficient(component,l,m,lHat,mHat) * radial_integral_3[ki][qi]) * (-1)**mPrime * np.sqrt(4*np.pi) * np.sqrt(2*L+1) * float(gaunt(lHat,lPrime,L,mHat,-mPrime,0))
	return 1j / mElectron *f12
def atomic_formfactor_vector_2(component,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3):
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
			radial_integral_2 = np.loadtxt("../data/integral_2/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
			radial_integral_3 = np.loadtxt("../data/integral_3/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
			for M in range(-L,L+1):
				for mHat in range(m-1,m+2):
					new =  1.0/mElectron * 4*np.pi * pow(1j,L+1) * np.conj(sph_harm(M,L,phi_q,theta_q)) * (grad_Ylm_coefficient(component,l,m,lHat,mHat) * radial_integral_2[ki][qi] + er_Ylm_coefficient(component,l,m,lHat,mHat) * radial_integral_3[ki][qi]) * (-1)**mPrime * float(gaunt(lHat,lPrime,L,mHat,-mPrime,M))
					f12 += new
	return f12


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
			return l/2 * np.sqrt((l+m+1) * (l+m+2) / (2*l+3) / (2*l+1))
		elif lHat == l+1 and mHat == m-1:
			return -l/2 * np.sqrt((l-m+1) * (l-m+2) / (2*l+3) / (2*l+1))
		elif lHat == l-1 and mHat == m+1:
			return (l+1)/2 * np.sqrt((l-m-1) * (l-m) / (2*l-1) / (2*l+1))
		elif lHat == l-1 and mHat == m-1:
			return -(l+1)/2 * np.sqrt((l+m-1) * (l+m) / (2*l-1) / (2*l+1))
	elif component == 2:
		if (lHat not in [l-1,l+1]) or (mHat not in [m-1,m+1]):
			return 0.0
		elif lHat == l+1 and mHat == m+1:
			return -1j*l/2 * np.sqrt((l+m+1) * (l+m+2) / (2*l+3) / (2*l+1))
		elif lHat == l+1 and mHat == m-1:
			return -1j*l/2 * np.sqrt((l-m+1) * (l-m+2) / (2*l+3) / (2*l+1))
		elif lHat == l-1 and mHat == m+1:
			return -1j*(l+1)/2 * np.sqrt((l-m-1) * (l-m) / (2*l-1) / (2*l+1))
		elif lHat == l-1 and mHat == m-1:
			return -1j*(l+1)/2 * np.sqrt((l+m-1) * (l+m) / (2*l-1) / (2*l+1))
	elif component == 3:
		if (lHat not in [l-1,l+1]) or (mHat != m):
			return 0.0
		elif lHat == l+1:
			return  -l * np.sqrt((l-m+1) * (l+m+1) / (2*l+3) / (2*l+1))
		elif lHat == l-1:
			return (1+l) * np.sqrt((l-m) * (l+m) / (2*l-1) / (2*l+1))


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
			return -1./2 * np.sqrt((l+m+1) * (l+m+2) / (2*l+3) / (2*l+1))
		elif lHat == l+1 and mHat == m-1:
			return 1./2 * np.sqrt((l-m+1) * (l-m+2) / (2*l+3) / (2*l+1))
		elif lHat == l-1 and mHat == m+1:
			return 1./2 * np.sqrt((l-m-1) * (l-m) / (2*l-1) / (2*l+1))
		elif lHat == l-1 and mHat == m-1:
			return -1./2 * np.sqrt((l+m-1) * (l+m) / (2*l-1) / (2*l+1))
	elif component == 2:
		if (lHat not in [l-1,l+1]) or (mHat not in [m-1,m+1]):
			return 0.0
		elif lHat == l+1 and mHat == m+1:
			return 1j/2 * np.sqrt((l+m+1) * (l+m+2) / (2*l+3) / (2*l+1))
		elif lHat == l+1 and mHat == m-1:
			return 1j/2 * np.sqrt((l-m+1) * (l-m+2) / (2*l+3) / (2*l+1))
		elif lHat == l-1 and mHat == m+1:
			return -1j/2 * np.sqrt((l-m-1) * (l-m) / (2*l-1) / (2*l+1))
		elif lHat == l-1 and mHat == m-1:
			return -1j/2 * np.sqrt((l+m-1) * (l+m) / (2*l-1) / (2*l+1))
	elif component == 3:
		if (lHat not in [l-1,l+1]) or (mHat != m):
			return 0.0
		elif lHat == l+1:
			return  np.sqrt((l-m+1) * (l+m+1) / (2*l+3) / (2*l+1))
		elif lHat == l-1:
			return np.sqrt((l-m) * (l+m) / (2*l-1) / (2*l+1))
