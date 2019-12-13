import time
import numpy as np
import multiprocessing
from tqdm import tqdm
from sympy.physics.wigner import gaunt, wigner_3j


from wave_functions import *
from electronic_responses import *
from radial_integrals_tabulation import qMin, qMax, kMin, kMax, lPrime_max, gridsize

def main():
	processes = multiprocessing.cpu_count()
	start_tot = time.time()

	####################################################################################
	#Test all electronic responses
	# element = Xe
	# n = 4
	# l = 2
	# kPrime = 0.1 * keV
	# q1 = 0 * keV
	# q2 = 0 * keV
	# q3 = 0.10 * keV
	# q = np.sqrt(q1*q1+q2*q2+q3*q3)

	# for response in range(1,7):
	# 	print(response,electronic_ionization_response(response,element,n,l,kPrime,q),electronic_ionization_response_2(response,element,n,l,kPrime,q1,q2,q3))

	####################################################################################
	#Test W_3

	element = Xe
	n = 4
	l = 0
	kPrime = 1 * keV
	q1 = 2.5 * keV
	q2 = 1.25 * keV
	q3 = 5.0 * keV
	q = np.sqrt(q1*q1+q2*q2+q3*q3)
	print("q=",q/keV,"keV")
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
	# result = [0,0,0]
	# for m in range(-l,l+1):
	# 	for lPrime in range(7):
	# 		for mPrime in range(-lPrime,lPrime+1):
	# 			f12s = atomic_formfactor_scalar(element,n,l,m,kPrime,lPrime,mPrime,q)
	# 			result[2] += np.real(f12s * np.conj(atomic_formfactor_vector(3,element,n,l,m,kPrime,lPrime,mPrime,q)))

	# norm = 4 * pow(kPrime, 3) / pow(2 * np.pi, 3) * np.sqrt(result[2]*result[2])
	# print("result = ",result)
	# print("norm = ",norm,"\n")

	####################################################################################
	# Tabulate the electronic ionization responses after the radial integrals have been computed

	# element = Ar
	# n = 2
	# l = 0
	# tabulate_electronic_ionization_response(5,element,n,l,gridsize)

	# args=[]
	# counter_total = 0
	# for element in [Ar,Xe]:
	# 	for response in [1,2,3,4,5]:
	# 		for n in range(element.nMax, 0, -1):
	# 			for l in range(element.lMax[n - 1], -1, -1):
	# 				filepath = "../data/response_"+str(response)+"/" + element.Shell_Name(n, l) + ".txt"
	# 				if os.path.exists(filepath) == False:
	# 					args.append([response,element,n,l,gridsize])
	# 					counter_total += 1

	# print("Start tabulation of electronic ionization responses using",processes,"cores. Total number of tables:",counter_total)   
	# with multiprocessing.Pool(processes) as pool:
	# 	pool.starmap(tabulate_electronic_ionization_response,args)	
	
	####################################################################################
	end_tot = time.time()
	print("\nProcessing time:\t", end_tot - start_tot, "s\n")
	####################################################################################

def tabulate_electronic_ionization_response(response,element,n,l,gridsize):
	filepath = "../data/response_"+str(response)+"/" + element.Shell_Name(n, l) + ".txt"
	if os.path.exists(filepath) == False:
		print("Tabulation of response",response,"of",element.Shell_Name(n,l),"started.")
		result = [[0 for x in range(gridsize)] for y in range(gridsize)]
		kGrid = np.logspace(np.log10(kMin), np.log10(kMax), gridsize)
		qGrid = np.logspace(np.log10(qMin), np.log10(qMax), gridsize)
		
		if response == 1:
			for lPrime in range(lPrime_max + 1):
				for L in range(abs(l - lPrime), l + lPrime + 1):
					radial_integral = np.loadtxt("../data/integral_1/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
					for ki in range(gridsize):
						k = kGrid[ki]
						for qi in range(gridsize):
							result[ki][qi] += 4 * pow(k, 3) / pow(2 * np.pi, 3) * (2 * l + 1) * (2 * lPrime + 1) * (2 * L + 1) * pow(radial_integral[ki][qi], 2) * pow(wigner_3j(l, lPrime, L, 0, 0, 0), 2)
		
		elif response == 2:
			for lPrime in range(lPrime_max + 1):
				for L in range(abs(l - lPrime), l + lPrime + 1):
					radial_integral_1 = np.loadtxt("../data/integral_1/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
					for lHat in [l-1,l+1]:
						for LHat in range(abs(lHat-lPrime),lHat+lPrime+1):
							radial_integral_2 = np.loadtxt("../data/integral_2/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(LHat) + ".txt")
							radial_integral_3 = np.loadtxt("../data/integral_3/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(LHat) + ".txt")
							for m in range(-l,l+1):
								mHat = m
								Gaunt_Sum = 0.0
								for mPrime in range(-lPrime,lPrime+1):
									Gaunt_Sum += float(gaunt(l,lPrime,L,m,-mPrime,0) * gaunt(lHat,lPrime,LHat,mHat,-mPrime,0))
								if(Gaunt_Sum != 0.0): 
									for ki in range(gridsize):
										k = kGrid[ki]
										for qi in range(gridsize):
											q = qGrid[qi]
											result[ki][qi] += 1/mElectron * 4 * pow(k, 3) / pow(2 * np.pi, 3) * 4*np.pi * q/mElectron * pow(-1j,LHat+1) * pow(1j,L) * radial_integral_1[ki][qi] * (grad_Ylm_coefficient(3,l,m,lHat,mHat) * radial_integral_2[ki][qi] + er_Ylm_coefficient(3,l,m,lHat,mHat) * radial_integral_3[ki][qi]) * np.sqrt(2*L+1) * np.sqrt(2*LHat+1) * Gaunt_Sum
		elif response == 3:
			for lPrime in range(lPrime_max + 1):
				for L in range(abs(l - lPrime), l + lPrime + 1):
					radial_integral_1 = np.loadtxt("../data/integral_1/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
					for lHat in [l-1,l+1]:
						for LHat in range(abs(lHat-lPrime),lHat+lPrime+1):
							radial_integral_2 = np.loadtxt("../data/integral_2/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(LHat) + ".txt")
							radial_integral_3 = np.loadtxt("../data/integral_3/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(LHat) + ".txt")
							for m in range(-l,l+1):
								mHat = m
								Gaunt_Sum = 0.0
								for mPrime in range(-lPrime,lPrime+1):
									Gaunt_Sum += float(gaunt(l,lPrime,L,m,-mPrime,0) * gaunt(lHat,lPrime,LHat,mHat,-mPrime,0))
								if(Gaunt_Sum != 0.0): 
									for ki in range(gridsize):
										k = kGrid[ki]
										for qi in range(gridsize):
											result[ki][qi] += 1/mElectron * 4 * pow(k, 3) / pow(2 * np.pi, 3) * 4*np.pi * pow(-1j,LHat+1) * pow(1j,L) * radial_integral_1[ki][qi] * (grad_Ylm_coefficient(3,l,m,lHat,mHat) * radial_integral_2[ki][qi] + er_Ylm_coefficient(3,l,m,lHat,mHat) * radial_integral_3[ki][qi]) * np.sqrt(2*L+1) * np.sqrt(2*LHat+1) * Gaunt_Sum
			for ki in range(gridsize):
				for qi in range(gridsize):
					result[ki][qi] = np.linalg.norm(result[ki][qi])
		
		elif response == 4:
			for lPrime in range(lPrime_max + 1):
				for lHat in [l-1,l+1]:
					for L in range(abs(lHat-lPrime),lHat+lPrime+1):
						radial_integral_2 = np.loadtxt("../data/integral_2/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
						radial_integral_3 = np.loadtxt("../data/integral_3/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
						for lHat2 in [l-1,l+1]:
							for L2 in range(abs(lHat2-lPrime),lHat2+lPrime+1):
								radial_integral_2_conj = np.loadtxt("../data/integral_2/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L2) + ".txt")
								radial_integral_3_conj = np.loadtxt("../data/integral_3/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L2) + ".txt")
								for m in range(-l,l+1):
									for mHat in range(m-1,m+2):
										for mHat2 in range(m-1,m+2):
											Gaunt_Sum = 0.0
											for mPrime in range(-lPrime,lPrime+1):
												Gaunt_Sum += float(gaunt(lHat,lPrime,L,mHat,-mPrime,0) * gaunt(lHat2,lPrime,L2,mHat2,-mPrime,0))
											if(Gaunt_Sum != 0.0):
												for i in range(1,4):
													for ki in range(gridsize):
														k = kGrid[ki]
														for qi in range(gridsize):
															result[ki][qi] += 1/mElectron/mElectron * 4 * pow(k, 3) / pow(2 * np.pi, 3) * 4*np.pi * pow(1j,L) * pow(-1j,L2) * (grad_Ylm_coefficient(i,l,m,lHat,mHat) * radial_integral_2[ki][qi] + er_Ylm_coefficient(i,l,m,lHat,mHat) * radial_integral_3[ki][qi]) * np.conj(grad_Ylm_coefficient(i,l,m,lHat2,mHat2) * radial_integral_2_conj[ki][qi] + er_Ylm_coefficient(i,l,m,lHat2,mHat2) * radial_integral_3_conj[ki][qi]) * np.sqrt(2*L+1) * np.sqrt(2*L2+1) * Gaunt_Sum
		elif response == 5:
			for lPrime in range(lPrime_max + 1):
				for lHat in [l-1,l+1]:
					for L in range(abs(lHat-lPrime),lHat+lPrime+1):
						radial_integral_2 = np.loadtxt("../data/integral_2/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
						radial_integral_3 = np.loadtxt("../data/integral_3/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L) + ".txt")
						for lHat2 in [l-1,l+1]:
							for L2 in range(abs(lHat2-lPrime),lHat2+lPrime+1):
								radial_integral_2_conj = np.loadtxt("../data/integral_2/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L2) + ".txt")
								radial_integral_3_conj = np.loadtxt("../data/integral_3/" + element.Shell_Name(n, l) + "_" + str(lPrime) + "_" + str(L2) + ".txt")
								for m in range(-l,l+1):
									for mHat in range(m-1,m+2):
										for mHat2 in range(m-1,m+2):
											Gaunt_Sum = 0.0
											for mPrime in range(-lPrime,lPrime+1):
												Gaunt_Sum += float(gaunt(lHat,lPrime,L,mHat,-mPrime,0) * gaunt(lHat2,lPrime,L2,mHat2,-mPrime,0))
											if(Gaunt_Sum != 0.0):
												for ki in range(gridsize):
													k = kGrid[ki]
													for qi in range(gridsize):
														q = qGrid[qi]
														result[ki][qi] += pow(q/mElectron,2) * 1/mElectron/mElectron * 4 * pow(k, 3) / pow(2 * np.pi, 3) * 4*np.pi * pow(1j,L) * pow(-1j,L2) * (grad_Ylm_coefficient(3,l,m,lHat,mHat) * radial_integral_2[ki][qi] + er_Ylm_coefficient(3,l,m,lHat,mHat) * radial_integral_3[ki][qi]) * np.conj(grad_Ylm_coefficient(3,l,m,lHat2,mHat2) * radial_integral_2_conj[ki][qi] + er_Ylm_coefficient(3,l,m,lHat2,mHat2) * radial_integral_3_conj[ki][qi]) * np.sqrt(2*L+1) * np.sqrt(2*L2+1) * Gaunt_Sum


			
		else:
			sys.exit("Error in tabulate_electronic_ionization_responses(): Response out of bound.")
		np.savetxt(filepath, np.real(result))
		print("Tabulation of response",response,"of",element.Shell_Name(n,l),"finished.")


if __name__ == "__main__":
	main()