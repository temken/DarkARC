import sys
import numpy as np
from sympy.physics.wigner import gaunt

from units import *
from radial_integrals import *

lPrime_max = 10

def main():
	response = 1
	element = Xe
	n = 5
	l = 1
	# Ee = 3000 * eV
	# kPrime = np.sqrt(2 * mElectron * Ee)
	kPrime = 0.1 * keV
	q = 1.0 * keV
	print(lPrime_max, atomic_response_function(response, element, n, l, kPrime, q))


def atomic_response_function(response,element,n,l,kPrime,q):
	Wion = 0
	for lPrime in range(lPrime_max+1):
		for m  in range(-l,l+1):
			for mPrime in range(-lPrime,lPrime+1):
				Wion +=  electronic_transition_response(response,element,n,l,m,kPrime,lPrime,mPrime,q)
		print(lPrime, 4 * pow(kPrime, 3) / pow(2 * np.pi, 3) * Wion)
	if response == 3:
		Wion = 4 * pow(kPrime, 3) / pow(2 * np.pi, 3) * np.linalg.norm(Wion)
	else:
		Wion =  4 * pow(kPrime, 3) / pow(2 * np.pi, 3) * Wion

	if response == 1:
		# radial_integral(integral, element, n, l, kPrime, lPrime, L, q,method)
		radial_integral_1 = radial_integral(1,element,n,l, kPrime,l,0,q,"numpy-stepwise") 
		radial_integral_one = radial_integral(1,element,n,l, kPrime,l,0,0,"numpy-stepwise")
		correction = 4 * pow(kPrime, 3) / pow(2 * np.pi, 3) * (2 * l + 1) * (radial_integral_one * radial_integral_one - 2 * radial_integral_one * radial_integral_1)
		if(correction < 0):
			Wion += correction
	return Wion

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

def atomic_formfactor_scalar(element,n,l,m,kPrime,lPrime,mPrime,q):
	f12 = 0
	for L in range(abs(l-lPrime),l+lPrime+1):
		radial_integral_1 = radial_integral(1, element, n, l, kPrime, lPrime, L, q, "numpy-stepwise")
		f12 += np.sqrt(4*np.pi) * pow(1j,L) * radial_integral_1 * (-1)**mPrime * np.sqrt(2*L+1) * float(gaunt(l,lPrime,L,m,-mPrime,0))
	return f12

if __name__ == "__main__":
	main()