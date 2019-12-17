import time
import numpy as np
import multiprocessing
from tqdm import tqdm
from sympy.physics.wigner import gaunt, wigner_3j

from vector_spherical_harmonics import *
from wave_functions import *
from electronic_responses import *
from radial_integrals_tabulation import qMin, qMax, kMin, kMax, lPrime_max, gridsize

def main():
	processes = multiprocessing.cpu_count()
	start_tot = time.time()

	####################################################################################
	# Tabulate the electronic ionization responses after the radial integrals have been computed

	# element = Ar
	# n = 2
	# l = 0
	# tabulate_atomic_response_function(5,element,n,l,gridsize)

	args=[]
	counter_total = 0
	for element in [Ar,Xe]:
		for response in [1,2,3,4,5]:
			for n in range(element.nMax, 0, -1):
				for l in range(element.lMax[n - 1], -1, -1):
					filepath = "../data/response_"+str(response)+"/" + element.Shell_Name(n, l) + ".txt"
					if os.path.exists(filepath) == False:
						args.append([response,element,n,l,gridsize])
						counter_total += 1

	print("Start tabulation of electronic ionization responses using",processes,"cores. Total number of tables:",counter_total)   
	with multiprocessing.Pool(processes) as pool:
		pool.starmap(tabulate_electronic_ionization_response,args)	
	
	####################################################################################
	end_tot = time.time()
	print("\nProcessing time:\t", end_tot - start_tot, "s\n")
	####################################################################################

def tabulate_atomic_response_function(response,element,n,l,gridsize):
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
											result[ki][qi] += 1/mElectron * 4 * pow(k, 3) / pow(2 * np.pi, 3) * 4*np.pi * q/mElectron * pow(-1j,LHat+1) * pow(1j,L) * radial_integral_1[ki][qi] * (VSH_coefficients_Psi(3,l,m,lHat,mHat) * radial_integral_3[ki][qi] + VSH_coefficients_Y(3,l,m,lHat,mHat) * radial_integral_2[ki][qi]) * np.sqrt(2*L+1) * np.sqrt(2*LHat+1) * Gaunt_Sum
		
		elif response == 3:
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
															result[ki][qi] += 1/mElectron/mElectron * 4 * pow(k, 3) / pow(2 * np.pi, 3) * 4*np.pi * pow(1j,L) * pow(-1j,L2) * (VSH_coefficients_Psi(i,l,m,lHat,mHat) * radial_integral_3[ki][qi] + VSH_coefficients_Y(i,l,m,lHat,mHat) * radial_integral_2[ki][qi]) * np.conj(VSH_coefficients_Psi(i,l,m,lHat2,mHat2) * radial_integral_3_conj[ki][qi] + VSH_coefficients_Y(i,l,m,lHat2,mHat2) * radial_integral_2_conj[ki][qi]) * np.sqrt(2*L+1) * np.sqrt(2*L2+1) * Gaunt_Sum
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
												for ki in range(gridsize):
													k = kGrid[ki]
													for qi in range(gridsize):
														q = qGrid[qi]
														result[ki][qi] += pow(q/mElectron,2) * 1/mElectron/mElectron * 4 * pow(k, 3) / pow(2 * np.pi, 3) * 4*np.pi * pow(1j,L) * pow(-1j,L2) * (VSH_coefficients_Psi(3,l,m,lHat,mHat) * radial_integral_3[ki][qi] + VSH_coefficients_Y(3,l,m,lHat,mHat) * radial_integral_2[ki][qi]) * np.conj(VSH_coefficients_Psi(3,l,m,lHat2,mHat2) * radial_integral_3_conj[ki][qi] + VSH_coefficients_Y(3,l,m,lHat2,mHat2) * radial_integral_2_conj[ki][qi]) * np.sqrt(2*L+1) * np.sqrt(2*L2+1) * Gaunt_Sum


			
		else:
			sys.exit("Error in tabulate_electronic_ionization_responses(): Response out of bound.")
		np.savetxt(filepath, np.real(result))
		print("Tabulation of response",response,"of",element.Shell_Name(n,l),"finished.")


if __name__ == "__main__":
	main()