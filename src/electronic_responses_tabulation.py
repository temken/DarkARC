

from wave_functions import *
from electronic_responses import *
from radial_integrals_tabulation import qMin, qMax, kMin, kMax, lPrime_max, gridsize

def main():
	# Test W_3
	element = Xe
	
	n=5
	l=0
	# m=1
	kPrime = 1 * keV
	# lPrime = 4	
	# mPrime = 1
	q1 = -1.0 * keV
	q2 = 1.0 * keV
	q3 = -1.0 * keV
	q = np.sqrt(q1*q1+q2*q2+q3*q3)

	# f12s = atomic_formfactor_scalar_2(element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
	# print("Scalar f = ",f12s)
	# f12v = [0,0,0]
	# f12v[0] = atomic_formfactor_vector_2(1,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
	# f12v[1] = atomic_formfactor_vector_2(2,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
	# f12v[2] = atomic_formfactor_vector_2(3,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)

	result = [0,0,0]
	for m in range(-l,l+1):
		for lPrime in range(7):
			for mPrime in range(-lPrime,lPrime+1):
				f12s = atomic_formfactor_scalar_2(element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
				result[0] += np.real(f12s * np.conj(atomic_formfactor_vector_2(1,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)))
				result[1] += np.real(f12s * np.conj(atomic_formfactor_vector_2(2,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)))
				result[2] += np.real(f12s * np.conj(atomic_formfactor_vector_2(3,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)))

	norm = 4 * pow(kPrime, 3) / pow(2 * np.pi, 3) * np.sqrt(result[0]*result[0]+result[1]*result[1]+result[2]*result[2])
	print("result = ",result)
	print("norm = ",norm,"\n")
	result = [0,0,0]
	for m in range(-l,l+1):
		for lPrime in range(7):
			for mPrime in range(-lPrime,lPrime+1):
				f12s = atomic_formfactor_scalar(element,n,l,m,kPrime,lPrime,mPrime,q)
				# result[0] += np.real(f12s * np.conj(atomic_formfactor_vector(1,element,n,l,m,kPrime,lPrime,mPrime,q)))
				# result[1] += np.real(f12s * np.conj(atomic_formfactor_vector(2,element,n,l,m,kPrime,lPrime,mPrime,q)))
				result[2] += np.real(f12s * np.conj(atomic_formfactor_vector(3,element,n,l,m,kPrime,lPrime,mPrime,q)))

	norm = 4 * pow(kPrime, 3) / pow(2 * np.pi, 3) * np.sqrt(result[2]*result[2])
	print("result = ",result)
	print("norm = ",norm,"\n")

	####################################################################################
	# Tabulate the standard ionization form factor after the radial integrals have been computed
	# element = Ar
	# n = 3
	# l = 1
	# # tabulate_standard_form_factor(element, n, l, gridsize)
	# tabulate_ionization_form_factor_3(element, n, l, gridsize)
	
	# args=[]
	# for n in range(element.nMax, 0, -1):
	# 	for l in range(element.lMax[n - 1], -1, -1):
	# 		args.append([element,n,l,gridsize])
	# 		# tabulate_standard_form_factor(element, n, l, gridsize)

	# with multiprocessing.Pool() as pool:
	# 	pool.starmap(tabulate_standard_form_factor,args)	
    ####################################################################################


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

# def tabulate_ionization_form_factor_1(element, n, l, gridsize):
#     filepath = "../data/form_factor_1/" + element.Shell_Name(n, l) + ".txt"
#     if os.path.exists(filepath) == False:
#         FF2 = [[0 for x in range(gridsize)] for y in range(gridsize)]
#         kGrid = np.logspace(np.log10(kMin), np.log10(kMax), gridsize)
#         for lPrime in range(lPrime_max + 1):
#             for L in range(abs(l - lPrime), l + lPrime + 1):
#                 radial_integral = np.loadtxt("../data/integral_1/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
#                 for ki in range(gridsize):
#                     k = kGrid[ki]
#                     for qi in range(gridsize):
#                         FF2[ki][qi] += 4 * pow(k, 3) / pow(2 * np.pi, 3) * (2 * l + 1) * (2 * lPrime + 1) * (2 * L + 1) * pow(radial_integral[ki][qi], 2) * pow(wigner_3j(l, lPrime, L, 0, 0, 0), 2)
#         np.savetxt(filepath, FF2)

if __name__ == "__main__":
	main()