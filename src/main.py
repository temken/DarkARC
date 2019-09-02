import time
from mpi4py import MPI

from units import *
from radial_integrals import *
from tabulation import *
from form_factors import *

start_tot = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
	print("MPI processes: ", size)


####################################################################################
# Tabulate radial integral for an atomic shell in parallel
integral = 1
element = Xe
gridsize = 100
n = 5
l = 1

job_list = list()
done_jobs= 0
# for n in range(element.nMax, 0, -1):
#     for l in range(element.lMax[n - 1], -1, -1):
for lPrime in range(lPrime_max+1):
	if integral == 1:
		Lmin = abs(l - lPrime)
		Lmax = l + lPrime
	else:
		Lmin = min(abs(l - 1 - lPrime), abs(l + 1 - lPrime))
		Lmax = l + lPrime + 1
	for L in range(Lmin, Lmax + 1):
		filepath = "../data/integral_" + str(integral) + "/" + element.Shell_Name(n,l) + "_" + str(lPrime)+ "_" + str(L) + ".txt"
		if os.path.exists(filepath) == False :
			job_list.append([integral,element,n,l,lPrime,L])
		else:
			done_jobs += 1
my_jobs = job_list[divide_work(len(job_list),size,rank)[0]:divide_work(len(job_list),size,rank)[1]+1]

if rank == 0: print("Progress: ",done_jobs,"/",len(job_list)+done_jobs)

for job in my_jobs:
	tabulate_integral(job[0],job[1],job[2],job[3],job[4],job[5],gridsize,rank)

####################################################################################




####################################################################################
# Test individual integrals with different methods
# integral = 1
# element = Ar
# n = 3
# l = 1
# lPrime = 7
# L = 8
# qGrid = np.logspace(np.log10(qMin),np.log10(qMax),gridsize)
# kGrid = np.logspace(np.log10(kMin),np.log10(kMax),gridsize)
# k = kGrid[10]
# q = qGrid[99]

# k = 20*keV
# q = 0.01*keV

# evaluate_integration_methods(integral, element, n, l, k, lPrime, L, q,True)
# for method in ["quadosc","Hankel","analytic","tanh-sinh-stepwise","numpy-stepwise"]:
#     print(method)
#     start = time.time()
#     int1 = radial_integral(integral,element,n,l,k,lPrime,L,q,method)
#     end = time.time()
#     print(int1,"\t(", end-start,"s)\n")
####################################################################################



####################################################################################
# Tabulate the standard ionization form factor after the radial integrals have been computed
# element = Ar
# n = 3
# l = 1
# gridsize = 100

# tabulate_standard_form_factor(element, n, l, gridsize)
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
# for element in [Xe]:
# 	for n in range(element.nMax, 3, -1):
# 	    for l in range(element.lMax[n - 1], -1, -1):
# 	        # print(n,l)
# 	        for lPrime in range(lPrime_max + 1):
# 	            if integral == 1:
# 	                Lmin = abs(l - lPrime)
# 	                Lmax = l + lPrime
# 	            else:
# 	                Lmin = min(abs(l - 1 - lPrime), abs(l + 1 - lPrime))
# 	                Lmax = l + lPrime + 1
# 	            for L in range(Lmin, Lmax + 1):  # new form factor
# 	                # for L in range( abs(l-lPrime) , l+lPrime+1): #standard form factor
# 		            print(n, l, lPrime, L)
# 		            counter += 1
# print(counter)
####################################################################################

end_tot = time.time()
if rank == 0:
	print("\nProcessing time:\t", end_tot - start_tot, "s\n")
