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
lPrime_max = 7
gridsize = 100

n = 3
l = 1

job_list = list()
for lPrime in range(lPrime_max+1):
	for L in range( min(abs(l-1-lPrime),abs(l+1-lPrime)) , l+1+lPrime+1):
		filepath = "../data/integral_" + str(integral) + "/" + element.Shell_Name(n,l) + "_" + str(lPrime)+ "_" + str(L) + ".txt"
		if os.path.exists(filepath) == False :
			job_list.append([integral,element,n,l,lPrime,L])
		# tabulate_integral(integral,element,n,l,lPrime,L,100,rank)
my_jobs = job_list[divide_work(len(job_list),size,rank)[0]:divide_work(len(job_list),size,rank)[1]+1]

for job in my_jobs:
	tabulate_integral(job[0],job[1],job[2],job[3],job[4],job[5],gridsize,rank)

print(len(job_list))

lPrime = 7 
L = 7

# start1 = time.time()
# create_integration_method_table(integral,element,n,l,lPrime,L,10)
# end1 = time.time()
# start2 = time.time()
# tabulate_integral(integral,element,n,l,lPrime,L,100)
# end2 = time.time()

# print(end1-start1,end2-start2)

# k = 0.2 * keV
# q = 600 * keV
# print("\nChosen method: ",get_integration_method(methods,k,q))
# for method in ["quadosc","analytic","Hankel"]:
#     print(method)
#     start = time.time()
#     int1 = radial_integral(integral,element,n,l,k,lPrime,L,q,method)
#     end = time.time()
#     print(int1,"\t(", end-start,"s)\n")


# counter = 0
# for n in range(element.nMax,0,-1):
#     for l in range(element.lMax[n-1],-1,-1):
#         print(n,l)
        # for lPrime in range(lPrime_max+1):
        #     for L in range( min(abs(l-1-lPrime),abs(l+1-lPrime)) , l+1+lPrime+1):
        #        counter += 1

# print(counter)

end_tot = time.time()
print("\nProcessing time:\t", end_tot-start_tot,"s\n")