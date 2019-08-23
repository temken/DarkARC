import time

from units import *
from radial_integrals import *
from tabulation import *

start_tot = time.time()


lPrime_max = 7

integral = 1
element = Ar
n = 3
l = 1
lPrime = 7 
L = 8

create_integration_method_table(integral,element,n,l,lPrime,L,100)
tabulate_integral(integral,element,n,l,lPrime,L,100)

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