import numpy as np
import os.path
import multiprocessing
from tqdm import tqdm

from units import *
from radial_integrals import *

lPrime_max = 7
qMin = 0.1*keV
qMax = 1000*keV
kMin = 0.1*keV
kMax = 100*keV
gridsize = 100

methods_hierarchy = ["analytic", "Hankel", "numpy-stepwise", "quadosc"]

# def radial_integral_wrapper(integral, element, n, l, k, lPrime, L, q,integration_methods):
def radial_integral_wrapper(args):
    integral, element, n, l, k, lPrime, L, q,integration_methods = args
    integration_method = get_integration_method(integration_methods, k,q)
    result = radial_integral(integral, element, n, l, k, lPrime, L, q,integration_method)
    if result == False:
        print("Warning: Analytic method did not converge, use numpy-stepwise instead.")
        result = radial_integral(integral, element, n, l, k, lPrime, L, q,"numpy-stepwise")
    # print("n=",n,"\tl=",l,"\tlPrime=",lPrime,"\tL=",L)
    return result

def tabulate_integral(integral,element,n,l,lPrime,L,steps,processes = 0):
    filepath = "../data/integral_" + str(integral) + "/" + element.Shell_Name(n,l) + "_" + str(lPrime)+ "_" + str(L) + ".txt"
    if os.path.exists(filepath) == False:
        if processes == 0: processes = multiprocessing.cpu_count()
        
        integration_methods = create_integration_method_table(integral,element,n,l,lPrime,L,steps,processes)
     
        qGrid = np.logspace(np.log10(qMin),np.log10(qMax),steps)
        kGrid = np.logspace(np.log10(kMin),np.log10(kMax),steps)

        args = []
        for k in kGrid:
            for q in qGrid:
                # args.append((integral,element,n,l,k,lPrime,L,q,integration_methods))
                args.append([integral,element,n,l,k,lPrime,L,q,integration_methods])
        print("Start tabulation for ",element.Shell_Name(n,l),"with lPrime =",lPrime,",\tL =",L,", using ",processes," cores.")   
        with multiprocessing.Pool(processes) as pool:
            # table = pool.starmap(radial_integral_wrapper,args)
            table = list(tqdm(pool.imap(radial_integral_wrapper,args), total=steps*steps,desc=element.Shell_Name(n,l)+"_"+str(lPrime)+str(L)))

        nested_table = [[table[ki*steps+qi] for qi in range(steps)] for ki in range(steps)]
        np.savetxt(filepath,nested_table,delimiter = '\t')
        print("Finish tabulation:\tn=",n,"\tl=",l,"\tlPrime=",lPrime,"\tL=",L)

def create_integration_method_table(integral,element,n,l,lPrime,L,steps,processes = 0):
    filepath = '../data/methods_'+str(integral)+'/' + element.Shell_Name(n,l)+'_'+str(lPrime)+str(L)+'.txt'
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
                # args.append((integral,element,n,l,k,lPrime,L,q))
                args.append([integral,element,n,l,k,lPrime,L,q])
        
        with multiprocessing.Pool(processes) as pool:
            # table = pool.starmap(evaluate_integration_methods,args)
            table = list(tqdm(pool.imap(evaluate_integration_methods_wrapper,args), total=coarse_gridsize*coarse_gridsize,desc="Integration methods"))
 
        methods = [[table[ki*coarse_gridsize+qi] for qi in range(coarse_gridsize)] for ki in range(coarse_gridsize)]
        np.savetxt(filepath,methods,fmt="%s",delimiter = '\t')
    return methods

def evaluate_integration_methods_wrapper(args):
    integral, element, n, l, kPrime, lPrime, L, q = args
    return evaluate_integration_methods(integral, element, n, l, kPrime, lPrime, L, q)

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
    # print("n = ",n,"\tl = ",l,"\tl' = ",lPrime,"\tL = ",L,"\tk = ",round(kPrime/keV,1),"keV\tq = ",round(q/keV,1),"keV\t",working_methods[0][0])
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