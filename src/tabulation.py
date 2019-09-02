import numpy as np
import os.path

from units import *
from radial_integrals import *

lPrime_max = 7
qMin = 0.1*keV
qMax = 1000*keV
kMin = 0.1*keV
kMax = 100*keV

methods_hierarchy = ["analytic", "Hankel", "numpy-stepwise", "quadosc"]

def tabulate_integral(integral,element,n,l,lPrime,L,steps,rank):
    filepath = "../data/integral_" + str(integral) + "/" + element.Shell_Name(n,l) + "_" + str(lPrime)+ "_" + str(L) + ".txt"
    if os.path.exists(filepath) == False:

        print("Start tabulation - Rank: ",rank,"\tn =",n,"\tl =",l,"\tlPrime =",lPrime,"\tL =",L)

        qGrid = np.logspace(np.log10(qMin),np.log10(qMax),steps)
        kGrid = np.logspace(np.log10(kMin),np.log10(kMax),steps)

        integration_methods = create_integration_method_table(integral,element,n,l,lPrime,L,steps,rank)
        table = list()
        counter = 0;

        for k in kGrid:
            q_list=list()
            for q in qGrid:
                integration_method = get_integration_method(integration_methods, k,q)
                result = radial_integral(integral, element, n, l, k, lPrime, L, q,integration_method)
                if result == False:
                    print("Warning: Analytic method did not converge, use numpy-stepwise instead.")
                    result = radial_integral(integral, element, n, l, k, lPrime, L, q,"numpy-stepwise")
                counter+=1
                if counter % 100 == 0:
                    print("Rank: ",rank,"\tn=",n,"\tl=",l,"\tlPrime=",lPrime,"\tL=",L,"Progress: ",100*counter/steps/steps," %")
                # print("q = ", q/keV," keV\tk = ",k/keV," keV\tI = ",result,"\tMethod: ",integration_method,"\tProgress: ",100*counter/steps/steps)
                q_list.append(result)
            table.append(q_list)
        np.savetxt(filepath,table,delimiter = '\t')
        print("Finish tabulation - Rank: ",rank,"\tn=",n,"\tl=",l,"\tlPrime=",lPrime,"\tL=",L)


def create_integration_method_table(integral,element,n,l,lPrime,L,steps,rank):
    filepath = '../data/methods_'+str(integral)+'/' + element.Shell_Name(n,l)+'_'+str(lPrime)+str(L)+'.txt'
    if os.path.exists(filepath):
        print("Rank: ",rank,"\tMethod table exists already and will be imported.")
        methods = np.loadtxt(filepath,dtype = 'str')
        coarse_gridsize = len(methods)

    else:
        print("Rank: ",rank,"\tMethod table must be created for ",element.Shell_Name(n,l)," with lPrime = ",lPrime," and L = ",L)
        coarse_gridsize = steps // 10 +1
        # print("Start evaluating the coarse grid of size ",coarse_gridsize,"x",coarse_gridsize," for ",element.Shell_Name(n,l)," with lPrime = ",lPrime," and L = ",L)
        qGridCoarse = np.logspace(np.log10(qMin),np.log10(qMax),coarse_gridsize)
        kGridCoarse = np.logspace(np.log10(kMin),np.log10(kMax),coarse_gridsize)
        methods = [["quadosc" for x in range(coarse_gridsize)] for y in range(coarse_gridsize)]
        ki= 0
        for k in kGridCoarse:
            qi= 0
            for q in qGridCoarse:
                methods[ki][qi] = evaluate_integration_methods(integral,element,n,l,k,lPrime,L,q,False)
                print("Rank = ",rank,"\tn = ",n,"\tl = ",l,"\tl' = ",lPrime,"\tki = ",ki,"\tqi = ",qi,"\t",methods[ki][qi])
                qi += 1
            ki += 1
        np.savetxt(filepath,methods,fmt="%s" ,delimiter = '\t')
    # print("quadosc / Hankel / analytic [%] : ",sum( (m == 'quadosc').sum() for m in methods)/coarse_gridsize/coarse_gridsize*100, " / ",sum( (m == 'Hankel').sum() for m in methods)/coarse_gridsize/coarse_gridsize*100, " / ",sum( (m == 'analytic').sum() for m in methods)/coarse_gridsize/coarse_gridsize*100)
    return methods

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
            if math.isclose(result[1], other_result[1], rel_tol=1e-3):
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
        print("Fastest method:\t",working_methods[0][0])
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


def divide_work(no_of_jobs,no_of_processes,rank):
    count = no_of_jobs // no_of_processes;
    remainder = no_of_jobs % no_of_processes;
    start=0
    stop=0
    if rank < remainder:
        start = rank*(count+1)
        stop = start + count
    else:
        start = rank * count + remainder;
        stop = start + (count - 1);
    return [start,stop]