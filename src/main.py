import time

from units import *
from radial_integrals import *
from tabulation import *
from form_factors import *

import numpy as np

def main():
	processes = 0
	start_tot = time.time()

	####################################################################################
	# Test W_6
	element = Ar
	
	n=2
	l=0
	# m=1
	kPrime = 1 * keV
	# lPrime = 4	
	# mPrime = 1
	q1 = -1.0 * keV
	q2 = 1.0 * keV
	q3 = -1.0 * keV
	q = np.sqrt(q1*q1+q2*q2+q3*q3)

	result = 0
	for m in range(-l,l+1):
		for lPrime in range(7):
			for mPrime in range(-lPrime,lPrime+1):
				f12 = [0,0,0]
				f12[0] = atomic_formfactor_vector_2(1,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
				f12[1] = atomic_formfactor_vector_2(2,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
				f12[2] = atomic_formfactor_vector_2(3,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
				result += 2 * q1/mElectron * np.imag( np.conj(f12[1]) * f12[2])
				result += 2 * q2/mElectron * np.imag( np.conj(f12[2]) * f12[0])
				result += 2 * q3/mElectron * np.imag( np.conj(f12[0]) * f12[1])
				# print(2 * q1/mElectron * np.imag( np.conj(f12[1]) * f12[2]) )#,2 * q2/mElectron * np.imag( np.conj(f12[2]) * f12[0]),2 * q3/mElectron * np.imag( np.conj(f12[0]) * f12[1]))
				print(result)
	print(4 * pow(kPrime, 3) / pow(2 * np.pi, 3) * result)

	result = 0
	for m in range(-l,l+1):
		for lPrime in range(7):
			for mPrime in range(-lPrime,lPrime+1):
				f12 = [0,0,0]
				f12[0] = atomic_formfactor_vector(1,element,n,l,m,kPrime,lPrime,mPrime,q)
				f12[1] = atomic_formfactor_vector(2,element,n,l,m,kPrime,lPrime,mPrime,q)
				# f12[2] = atomic_formfactor_vector_2(3,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
				# result += 2 * q1/mElectron * np.imag( np.conj(f12[1]) * f12[2])
				# result += 2 * q2/mElectron * np.imag( np.conj(f12[2]) * f12[0])
				result += 2 * q/mElectron * np.imag( np.conj(f12[0]) * f12[1])
				# print(2 * q/mElectron * np.imag( np.conj(f12[0]) * f12[1]))
				print(result)
	print(4 * pow(kPrime, 3) / pow(2 * np.pi, 3) * result)

	####################################################################################


	####################################################################################
	# Test W_3
	# element = Xe
	
	# n=5
	# l=0
	# # m=1
	# kPrime = 1 * keV
	# # lPrime = 4	
	# # mPrime = 1
	# q1 = -1.0 * keV
	# q2 = 1.0 * keV
	# q3 = -1.0 * keV
	# q = np.sqrt(q1*q1+q2*q2+q3*q3)

	# # f12s = atomic_formfactor_scalar_2(element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
	# # print("Scalar f = ",f12s)
	# # f12v = [0,0,0]
	# # f12v[0] = atomic_formfactor_vector_2(1,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
	# # f12v[1] = atomic_formfactor_vector_2(2,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
	# # f12v[2] = atomic_formfactor_vector_2(3,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)

	# result = [0,0,0]
	# for m in range(-l,l+1):
	# 	for lPrime in range(7):
	# 		for mPrime in range(-lPrime,lPrime+1):
	# 			f12s = atomic_formfactor_scalar_2(element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)
	# 			result[0] += np.real(f12s * np.conj(atomic_formfactor_vector_2(1,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)))
	# 			result[1] += np.real(f12s * np.conj(atomic_formfactor_vector_2(2,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)))
	# 			result[2] += np.real(f12s * np.conj(atomic_formfactor_vector_2(3,element,n,l,m,kPrime,lPrime,mPrime,q1,q2,q3)))

	# norm = 4 * pow(kPrime, 3) / pow(2 * np.pi, 3) * np.sqrt(result[0]*result[0]+result[1]*result[1]+result[2]*result[2])
	# print("result = ",result)
	# print("norm = ",norm,"\n")

	# result = [0,0,0]
	# for m in range(-l,l+1):
	# 	for lPrime in range(7):
	# 		for mPrime in range(-lPrime,lPrime+1):
	# 			f12s = atomic_formfactor_scalar(element,n,l,m,kPrime,lPrime,mPrime,q)
	# 			# result[0] += np.real(f12s * np.conj(atomic_formfactor_vector(1,element,n,l,m,kPrime,lPrime,mPrime,q)))
	# 			# result[1] += np.real(f12s * np.conj(atomic_formfactor_vector(2,element,n,l,m,kPrime,lPrime,mPrime,q)))
	# 			result[2] += np.real(f12s * np.conj(atomic_formfactor_vector(3,element,n,l,m,kPrime,lPrime,mPrime,q)))

	# norm = 4 * pow(kPrime, 3) / pow(2 * np.pi, 3) * np.sqrt(result[2]*result[2])
	# print("result = ",result)
	# print("norm = ",norm,"\n")

	# print(electronic_response(3,element,n,l,kPrime,q))
	# print(electronic_response_2(3,element,n,l,kPrime,q1,q2,q3))

	# print("\nVector F = ",f12v)
	# print("\t|F| = ",np.sqrt(f12v[0]*np.conj(f12v[0])+f12v[1]*np.conj(f12v[1])+f12v[2]*np.conj(f12v[2])))
	# print("\t<q,F> = ",q1*f12v[0],"+",q2*f12v[1],"+",q3*f12v[2]," = ",q1*f12v[0]+q2*f12v[1]+q3*f12v[2])
	# print("\nProduct f*F = ", [f12s*np.conj(f12v[0]),f12s*np.conj(f12v[1]),f12s*np.conj(f12v[2])])

	# print("\n\nq along z axis")
	# f12s = atomic_formfactor_scalar(element,n,l,m,kPrime,lPrime,mPrime,q)
	# print("Scalar f = ",f12s)
	# f12v = [0,0,0]
	# f12v[0] = atomic_formfactor_vector(1,element,n,l,m,kPrime,lPrime,mPrime,q)
	# f12v[1] = atomic_formfactor_vector(2,element,n,l,m,kPrime,lPrime,mPrime,q)
	# f12v[2] = atomic_formfactor_vector(3,element,n,l,m,kPrime,lPrime,mPrime,q)
	# print("\nVector F = ",f12v)
	# print("\t|F| = ",np.sqrt(f12v[0]*np.conj(f12v[0])+f12v[1]*np.conj(f12v[1])+f12v[2]*np.conj(f12v[2])))
	# print("\t<q,F> = ",q*f12v[2])
	# print("\nProduct f*F = ", [f12s*np.conj(f12v[0]),f12s*np.conj(f12v[1]),f12s*np.conj(f12v[2])])
	####################################################################################





	####################################################################################
	# Test the six electronic responses
	# element = Ar
	
	# n=2
	# l=0
	# kPrime = 0.1 * keV
	# q1 = 0.0 * keV
	# q2 = 0.1 * keV
	# q3 = 0.1 * keV
	# q = np.sqrt(q1*q1+q2*q2+q3*q3)

	# print("q=",q/keV," keV")
	# for response in [3,6]:
	# 	print(response,electronic_response(response,element,n,l,kPrime,q))

	# for response in [3,6]:
	# 	print(response,electronic_response_2(response,element,n,l,kPrime,q1,q2,q3))
	####################################################################################

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


	####################################################################################
	# Count number of tables to be created for one shell
	# integral = 1
	# element = Xe
	# n = 5
	# l = 1

	# counter = 0
	# 	        # print(n,l)
	# for lPrime in range(lPrime_max + 1):
	#     if integral == 1:
	#         Lmin = abs(l - lPrime)
	#         Lmax = l + lPrime
	#     else:
	#         Lmin = min(abs(l - 1 - lPrime), abs(l + 1 - lPrime))
	#         Lmax = l + lPrime + 1
	#     for L in range(Lmin, Lmax + 1):  # new form factor
	#         # for L in range( abs(l-lPrime) , l+lPrime+1): #standard form factor
	#         print(n, l, lPrime, L)
	#         counter += 1
	# print(counter)
	####################################################################################

	####################################################################################
	# Count number of tables to be created in total
	# integral = 1

	# counter = 0
	# for integral in [1,2,3]:
	# 	for element in [Xe,Ar]:
	# 		nMin = 1
	# 		# if element == Xe:
	# 		# 	nMin = 4
	# 		# elif element == Ar:
	# 		# 	nMin = 3
	# 		for n in range(element.nMax, nMin-1, -1):
	# 		    for l in range(element.lMax[n - 1], -1, -1):
	# 		        # print(n,l)
	# 		        for lPrime in range(lPrime_max + 1):
	# 		            if integral == 1:
	# 		                Lmin = abs(l - lPrime)
	# 		                Lmax = l + lPrime
	# 		            else:
	# 		                Lmin = min(abs(l - 1 - lPrime), abs(l + 1 - lPrime))
	# 		                Lmax = l + lPrime + 1
	# 		            for L in range(Lmin, Lmax + 1):  # new form factor
	# 		                # for L in range( abs(l-lPrime) , l+lPrime+1): #standard form factor
	# 			            print(n, l, lPrime, L)
	# 			            counter += 1
	# print(counter)
	# print("Time in total:\t", counter * 3,"hr")
	# print("Time for one node:\t", counter * 3 / 32 ,"hr")
	# print("Time for five nodes:\t", counter * 3 / 32 / 5 ,"hr")
	####################################################################################
	
	####################################################################################
	# Test individual integrals with different methods
	# integral = 3
	# element = Xe
	# n = 4
	# l = 1
	# lPrime = 0
	# L = 0
	# qGrid = np.logspace(np.log10(qMin),np.log10(qMax),gridsize)
	# kGrid = np.logspace(np.log10(kMin),np.log10(kMax),gridsize)
	# k = kGrid[10]
	# q = qGrid[99]

	
	# create_integration_method_table(integral,element,n,l,lPrime,L,gridsize)
	# k = 100*keV
	# q = 1*keV
	# evaluate_integration_methods(integral, element, n, l, k, lPrime, L, q,True)

	# # for method in ["quadosc","Hankel","analytic","tanh-sinh-stepwise","numpy-stepwise"]:
	# #     print(method)
	# #     start = time.time()
	# #     int1 = radial_integral(integral,element,n,l,k,lPrime,L,q,method)
	# #     end = time.time()
	# #     print(int1,"\t(", end-start,"s)\n")
	####################################################################################

		####################################################################################
	# # Tabulate radial integral for an atomic shell in parallel
	# integral = 3
	# element = Ar
	# # n = 3
	# # l = 0

	# job_list = list()
	# done_jobs= 0
	# for n in range(element.nMax, 0, -1):
	# 	for l in range(element.lMax[n - 1], -1, -1):
	# 		for lPrime in range(lPrime_max+1):
	# 			if integral == 1:
	# 				Lmin = abs(l - lPrime)
	# 				Lmax = l + lPrime
	# 			else:
	# 				Lmin = min(abs(l - 1 - lPrime), abs(l + 1 - lPrime))
	# 				Lmax = l + lPrime + 1
	# 			for L in range(Lmin, Lmax + 1):
	# 				filepath = "../data/integral_" + str(integral) + "/" + element.Shell_Name(n,l) + "_" + str(lPrime)+ "_" + str(L) + ".txt"
	# 				if os.path.exists(filepath) == False :
	# 					job_list.append([integral,element,n,l,lPrime,L])
	# 				else:
	# 					done_jobs += 1
	# print("Previous progress: ",done_jobs,"/",len(job_list)+done_jobs)

	# counter = 1
	# number_of_jobs = len(job_list)
	# for job in job_list:
	# 	print("Job ", counter ," / ", number_of_jobs)
	# 	tabulate_integral(job[0],job[1],job[2],job[3],job[4],job[5],gridsize,processes)
	# 	counter += 1

	####################################################################################


	end_tot = time.time()
	print("\nProcessing time:\t", end_tot - start_tot, "s\n")

if __name__ == "__main__":
    main()
