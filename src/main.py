import time

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0: print("MPI processes: ",size)

from units import *
from radial_integrals import *
from tabulation import *

start_tot = time.time()

integral = 1
element = Ar
gridsize = 100

n = 3
l = 1

job_list = list()
done_jobs= 0
for lPrime in range(lPrime_max+1):
	for L in range( abs(l-lPrime) , l+lPrime+1):
		filepath = "../data/integral_" + str(integral) + "/" + element.Shell_Name(n,l) + "_" + str(lPrime)+ "_" + str(L) + ".txt"
		if os.path.exists(filepath) == False :
			job_list.append([integral,element,n,l,lPrime,L])
		else:
			done_jobs += 1
		# tabulate_integral(integral,element,n,l,lPrime,L,100,rank)
my_jobs = job_list[divide_work(len(job_list),size,rank)[0]:divide_work(len(job_list),size,rank)[1]+1]

if rank == 0: print("Progress: ",done_jobs,"/",len(job_list)+done_jobs)

for job in my_jobs:
	tabulate_integral(job[0],job[1],job[2],job[3],job[4],job[5],gridsize,rank)

# lPrime = 1 
# L = 0

# k = 1.0 * keV
# q = 0.5 * keV
# # print("\nChosen method: ",get_integration_method(methods,k,q))
# for method in ["quadosc","analytic","Hankel"]:
#     print(method)
#     start = time.time()
#     int1 = radial_integral(integral,element,n,l,k,lPrime,L,q,method)
#     end = time.time()
#     print(int1,"\t(", end-start,"s)\n")


# counter = 0
# for n in range(element.nMax,0,-1):
#     for l in range(element.lMax[n-1],-1,-1):
#         # print(n,l)
#         for lPrime in range(lPrime_max+1):
#             for L in range( min(abs(l-1-lPrime),abs(l+1-lPrime)) , l+1+lPrime+1):
#             	print(n,l,lPrime,L)
#             	counter += 1

# print(counter)

end_tot = time.time()
if rank == 0: print("\nProcessing time:\t", end_tot-start_tot,"s\n")