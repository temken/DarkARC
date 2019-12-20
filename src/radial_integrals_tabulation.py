import numpy as np
import os.path
import multiprocessing
from tqdm import tqdm
import time

from units import *
from radial_integrals import *
from wave_functions import l_names

# Parameters on the table's size and argument intervals
lPrime_max = 7
qMin = 0.1*keV
qMax = 1000*keV
kMin = 0.1*keV
kMax = 100*keV
gridsize = 100
# Integration method hierarchy
methods_hierarchy = ["analytic","numpy-stepwise", "quadosc"]

# Parallel tabularization of the radial integrals
def main():
	processes = 0
	start_tot = time.time()
	
	####################################################################################
	# # Find the best integration method for a specific integral
	# element = Ar
	# n = 3
	# l = 1
	# integral = 2
	# lPrime = 2
	# L = 1
	# k = 5*keV
	# q = 1000*keV

	# print("\nFind the best integration method for a given set of parameters.\n")
	# best_method = evaluate_integration_methods(integral, element, n, l, k, lPrime, L, q,True);

	####################################################################################
	# Tabulate the radial integrals for one orbital (nl)
	element = Ar
	integral = 1
	n = 3
	l = 1
	print("Tabulate the radial integral",integral,"for",element.name,n,l_names[l],".\n")

	job_list = list()
	done_jobs= 0
	counter_total = 0
	for lPrime in range(lPrime_max + 1):
		if integral == 1:
			Lmin = abs(l - lPrime)
			Lmax = l + lPrime
		else:
			Lmin = min(abs(l - 1 - lPrime), abs(l + 1 - lPrime))
			Lmax = l + lPrime + 1
		for L in range(Lmin, Lmax + 1):
			counter_total += 1
			filepath = "../data/radial_integral_" + str(integral) + "/" + element.Shell_Name(n,l) + "_" + str(lPrime)+ "_" + str(L) + ".txt"
			if os.path.exists(filepath) == False:
				job_list.append([integral,element,n,l,lPrime,L])
			else:
				done_jobs += 1
	print("Required tables in total:\t",counter_total)
	print("Previous progress:\t\t\t",done_jobs,"/",counter_total)
	print("Estimated time:\t\t\t\t", len(job_list) * 3," core hrs\n")

	counter = 1
	number_of_jobs = len(job_list)
	for job in job_list:
		print("Job ", counter ," / ", number_of_jobs)
		tabulate_integral(job[0],job[1],job[2],job[3],job[4],job[5],gridsize,processes)
		counter += 1

	####################################################################################
	# # Radial integrals for all responses and atomic orbitals
	# element = Ar
	# print("Tabulate the radial integrals for",element.name,".\n")

	# # Count number of tables to be created in total, and check for previously completed integral tables
	# job_list = list()
	# done_jobs= 0

	# counter_total = 0
	# for integral in range(1,4):
	# 	for n in range(element.nMax, 0, -1):
	# 		for l in range(element.lMax[n - 1], -1, -1):
	# 			for lPrime in range(lPrime_max + 1):
	# 				if integral == 1:
	# 					Lmin = abs(l - lPrime)
	# 					Lmax = l + lPrime
	# 				else:
	# 					Lmin = min(abs(l - 1 - lPrime), abs(l + 1 - lPrime))
	# 					Lmax = l + lPrime + 1
	# 				for L in range(Lmin, Lmax + 1):
	# 					counter_total += 1
	# 					filepath = "../data/radial_integral_" + str(integral) + "/" + element.Shell_Name(n,l) + "_" + str(lPrime)+ "_" + str(L) + ".txt"
	# 					if os.path.exists(filepath) == False:
	# 						job_list.append([integral,element,n,l,lPrime,L])
	# 					else:
	# 						done_jobs += 1
	# print("Required tables in total:\t",counter_total)
	# print("Previous progress:\t\t\t",done_jobs,"/",counter_total)
	# print("Estimated time:\t\t\t\t", len(job_list) * 3," core hrs\n")

	# counter = 1
	# number_of_jobs = len(job_list)
	# for job in job_list:
	# 	print("Job ", counter ," / ", number_of_jobs)
	# 	tabulate_integral(job[0],job[1],job[2],job[3],job[4],job[5],gridsize,processes)
	# 	counter += 1
	
	####################################################################################

	end_tot = time.time()
	print("\nProcessing time:\t", end_tot - start_tot, "s\n")


# Parallel tabulation functions
def radial_integral_wrapper(args):
	integral, element, n, l, k, lPrime, L, q,integration_methods = args
	integration_method = get_integration_method(integration_methods, k,q)
	result = radial_integral(integral, element, n, l, k, lPrime, L, q,integration_method)
	if result == False:
		print("Warning: Analytic method did not converge, use numpy-stepwise instead.")
		result = radial_integral(integral, element, n, l, k, lPrime, L, q,"numpy-stepwise")
	return result

def tabulate_integral(integral,element,n,l,lPrime,L,steps,processes = 0):
	filepath = "../data/radial_integral_" + str(integral) + "/" + element.Shell_Name(n,l) + "_" + str(lPrime)+ "_" + str(L) + ".txt"
	if os.path.exists(filepath) == False:
		if processes == 0: processes = multiprocessing.cpu_count()
		
		integration_methods = create_integration_method_table(integral,element,n,l,lPrime,L,steps,processes)
	 
		qGrid = np.logspace(np.log10(qMin),np.log10(qMax),steps)
		kGrid = np.logspace(np.log10(kMin),np.log10(kMax),steps)

		args = []
		for k in kGrid:
			for q in qGrid:
				args.append([integral,element,n,l,k,lPrime,L,q,integration_methods])
		print("Start tabulation: Radial integral",integral,"for",element.Shell_Name(n,l),"with lPrime =",lPrime,",\tL =",L,", using ",processes," cores.")   
		with multiprocessing.Pool(processes) as pool:
			table = list(tqdm(pool.imap(radial_integral_wrapper,args), total=steps*steps,desc=element.Shell_Name(n,l)+"_"+str(lPrime)+str(L)))

		nested_table = [[table[ki*steps+qi] for qi in range(steps)] for ki in range(steps)]
		np.savetxt(filepath,nested_table,delimiter = '\t')
		print("Finish tabulation:\tn=",n,"\tl=",l,"\tlPrime=",lPrime,"\tL=",L)

# 3.) Identify the fastest accurate integration method across the parameter grid of the table.
def create_integration_method_table(integral,element,n,l,lPrime,L,steps,processes = 0):
	filepath = '../data/integration_methods_'+str(integral)+'/' + element.Shell_Name(n,l)+'_'+str(lPrime)+"_"+str(L)+'.txt'
	if os.path.exists(filepath):
		print("Method table exists already and will be imported.")
		methods = np.loadtxt(filepath,dtype = 'str')
		coarse_gridsize = len(methods)
	else:
		if processes == 0: processes = multiprocessing.cpu_count()
		print("Method table must be created for ",element.Shell_Name(n,l)," with lPrime = ",lPrime," and L = ",L,"\tCores:\t",processes)
		coarse_gridsize = steps // 10 +1
		
		qGridCoarse = np.logspace(np.log10(qMin),np.log10(qMax),coarse_gridsize)
		kGridCoarse = np.logspace(np.log10(kMin),np.log10(kMax),coarse_gridsize)

		args = []
		for k in kGridCoarse:
			for q in qGridCoarse:
				args.append([integral,element,n,l,k,lPrime,L,q])
		
		with multiprocessing.Pool(processes) as pool:
			table = list(tqdm(pool.imap(evaluate_integration_methods_wrapper,args), total=coarse_gridsize*coarse_gridsize,desc="Integration methods"))
 
		methods = [[table[ki*coarse_gridsize+qi] for qi in range(coarse_gridsize)] for ki in range(coarse_gridsize)]
		np.savetxt(filepath,methods,fmt="%s",delimiter = '\t')
	return methods

def evaluate_integration_methods_wrapper(args):
	integral, element, n, l, kPrime, lPrime, L, q = args
	return evaluate_integration_methods(integral, element, n, l, kPrime, lPrime, L, q)

# Function to evaluate the different methods, and return the fastest working method.
def evaluate_integration_methods(integral, element, n, l, kPrime, lPrime, L, q,output = False):
	# Compute the integral with different methods
	results = []
	if output: print("Initial integration results:")
	for method in methods_hierarchy:
		t_1 = time.time()
		result = radial_integral(integral, element, n,l, kPrime, lPrime, L, q, method)
		t_2 = time.time()
		duration = t_2 - t_1
		if result == result:
			results.append([method, float(result), duration])
			if output: print(method,"\t",float(result),"\t",duration)
			if method == "numpy-stepwise":
				break

	# Identify methods giving correct results
	working_methods = []
	# 1. Check for dublicates
	for result in results:
		for other_result in (x for x in results if x != result):
			if math.isclose(result[1], other_result[1], rel_tol=1e-2):
				working_methods.append(result)
				break
	if output:
		print("Dublicates:")
		if len(working_methods) == 0:
			print("None.")
		else:
			for result in working_methods:
				print(result[0],"\t",result[1],"\t",result[2])
	# 2. Without dublicates, we need to compare to quadosc
	if len(working_methods) == 0:
		if any("quadosc" in result for result in results):
			for result in results:
				if result[0] == "quadosc":
					working_methods.append(result)
					if output: print(result[0],"\t",result[1],"\t",result[2])
					break
		else:
			t_1 = time.time()
			result_quadosc = radial_integral(integral, element, n, l, kPrime, lPrime, L, q, "quadosc")
			t_2 = time.time()
			duration = t_2 - t_1
			results.append(["quadosc", float(result_quadosc), duration])
			if output: print(results[-1][0],"\t",results[-1][1],"\t",results[-1][2])
			for result in results:
				for other_result in (x for x in results if x != result):
					if math.isclose(result[1], other_result[1],rel_tol = 1e-2):
						working_methods.append(result)
			if len(working_methods) == 0:
				print("Warning in evaluate_integration_methods(): No dublicates found for ",[integral, element, n, l, kPrime, lPrime, L, q],".")
				for result in results:
					if result[0] == "numpy-stepwise":
						working_methods.append(result)
						break

	# Return the fastest of the good methods
	working_methods.sort(key = lambda x: x[2])
	if output:
		print("Working methods:")
		for result in working_methods:
			print(result[0],"\t",result[1],"\t",result[2])
		print("\nBest method:\t",working_methods[0][0])
	return working_methods[0][0]

def get_integration_method(methods, k,q):
	gridsize = len(methods)
	qGridCoarse = np.logspace(np.log10(qMin),np.log10(qMax),gridsize)
	kGridCoarse = np.logspace(np.log10(kMin),np.log10(kMax),gridsize)
	for qi in range(1,gridsize):
		if qGridCoarse[qi] > q :
			break;
	for ki in range(1,gridsize):
		if kGridCoarse[ki] > k :
			break;
	potential_methods = [methods[ki][qi],methods[ki-1][qi],methods[ki][qi-1],methods[ki-1][qi-1]]
	for method in reversed(methods_hierarchy):
		if method in potential_methods:
			return method

if __name__ == "__main__":
	main()