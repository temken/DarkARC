import numpy as np
import os.path

from units import *
from radial_integrals import *

qMin = 0.1*keV
qMax = 1000*keV
kMin = 0.2*keV
kMax = 100*keV

def tabulate_integral(integral,element,n,l,lPrime,L,steps=100):
    filepath = "../data/I_" + str(integral) + "_" + element.Shell_Name(n,l) + "_" + str(lPrime)+ "_" + str(L) + ".txt"
    qGrid = np.logspace(np.log10(qMin),np.log10(qMax),steps)
    kGrid = np.logspace(np.log10(kMin),np.log10(kMax),steps)

    integration_methods = create_integration_method_table(integral,element,n,l,lPrime,L,steps)
    table = list()
    counter = 0;

    for k in kGrid:
        q_list=list()
        for q in qGrid:
            integration_method = get_integration_method(integration_methods, k,q)
            result = radial_integral(integral, element, n, l, k, lPrime, L, q,integration_method)
            counter+=1
            print("q = ", q/keV," keV\tk = ",k/keV," keV\tI = ",result,"\tMethod: ",integration_method,"\tProgress: ",100*counter/steps/steps)
            q_list.append(result)
        table.append(q_list)
    np.savetxt(filepath,table,delimiter = '\t')

def create_integration_method_table(integral,element,n,l,lPrime,L,steps):
    filepath = '../data/integration_methods_'+str(integral)+'_' + element.Shell_Name(n,l)+'_'+str(lPrime)+str(L)+'.txt'
    if os.path.exists(filepath):
        print("Method table exists already and will be imported.")
        methods = np.loadtxt(filepath,dtype = 'str')
        coarse_gridsize = len(methods)

    else:
        print("Method table must be created.")
        coarse_gridsize = steps //10 +1
        print("Start evaluating the coarse grid of size ",coarse_gridsize,"x",coarse_gridsize," for ",element.Shell_Name(n,l)," with lPrime = ",lPrime," and L = ",L)
        qGridCoarse = np.logspace(np.log10(qMin),np.log10(qMax),coarse_gridsize)
        kGridCoarse = np.logspace(np.log10(kMin),np.log10(kMax),coarse_gridsize)
        methods = [["quadosc" for x in range(coarse_gridsize)] for y in range(coarse_gridsize)]
        ki= 0
        for k in kGridCoarse:
            qi= 0
            for q in qGridCoarse:
                methods[ki][qi] = identify_integration_method(integral,element,n,l,k,lPrime,L,q)
                print(ki,"\t",qi,"\t",methods[ki][qi])
                qi += 1
            ki += 1
        np.savetxt(filepath,methods,fmt="%s" ,delimiter = '\t')
    
    print("quadosc / Hankel / analytic [%] : ",sum( (m == 'quadosc').sum() for m in methods)/coarse_gridsize/coarse_gridsize*100, " / ",sum( (m == 'Hankel').sum() for m in methods)/coarse_gridsize/coarse_gridsize*100, " / ",sum( (m == 'analytic').sum() for m in methods)/coarse_gridsize/coarse_gridsize*100)
    return methods

def identify_integration_method(integral,element,n,l,k,lPrime,L,q):
    integral_exact = radial_integral_quadosc(integral,element,n,l,k,lPrime,L,q)
    # Three methods
    integral_Hankel = radial_integral_hankel(integral,element,n,l,k,lPrime,L,q)
    deviation = abs(integral_exact-integral_Hankel)/abs(integral_exact)
    if deviation < 0.01:
        return "Hankel"
    else:
        integral_analytic = radial_integral_analytic(integral,element,n,l,k,lPrime,L,q)
        deviation = abs(integral_exact-integral_analytic)/abs(integral_exact)
        if deviation < 0.01:
            return "analytic"
        else:
            deviation = abs(integral_analytic-integral_Hankel)/abs(integral_analytic)
            if deviation < 0.01:
                print("Warning: quadosc might be wrong for ",[element,n,l,k,lPrime,L,q])
            return "quadosc"

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
    # print(potential_methods)
    if "quadosc" in potential_methods:
        return "quadosc"
    elif "analytic" in potential_methods:
        return "analytic"
    else:
        return "Hankel"