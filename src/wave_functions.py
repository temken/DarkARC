import numpy as np
import mpmath as mp
import scipy.special as special

from units import *


# Initial state wave function class
class Initial_Wavefunctions:
    def __init__(self,name,C,Z,nlj,E_B): 
        self.name = name
        self.C_nlj = C
        self.Z_lj = Z
        self.n_lj = nlj
        self.binding_energies = E_B
        self.Z_eff = []
        for n in range(len(E_B)):
            self.Z_eff.append([])
            for l in range(len(E_B[n])):
                self.Z_eff[n].append((n+1) * np.sqrt(-2*E_B[n][l] / au))
                # self.Z_eff[n][l] = (n+1) * mp.sqrt(-2*E_B[n][l] / au)
                # self.Z_eff[n][l] = (n+1) * np.sqrt(-2*E_B[n][l] / au)
        self.nMax = len(E_B)
        self.lMax = [len(x)-1 for x in E_B]

    def Shell_Name(self,n,l):
        l_names =["s","p","d","f","g"]
        return self.name+"_"+str(n)+l_names[l]

    def R(self,n,l,r):
        radial_wavefunction = 0
        for j in range(len(self.C_nlj[n-1][l])):
            radial_wavefunction += mp.power(a0,-3/2) * self.C_nlj[n-1][l][j] * mp.power(2 * self.Z_lj[l][j], self.n_lj[l][j]+1/2) / mp.sqrt(mp.factorial(2 * self.n_lj[l][j])) * mp.power(r/a0,self.n_lj[l][j]-1) * mp.exp(-self.Z_lj[l][j] * r/a0)
        return radial_wavefunction

    def R_2(self,n,l,r):
        radial_wavefunction = 0
        for j in range(len(self.C_nlj[n-1][l])):
            radial_wavefunction += np.power(a0,-3/2) * self.C_nlj[n-1][l][j] * np.power(2 * self.Z_lj[l][j], self.n_lj[l][j]+1/2) / np.sqrt(np.math.factorial(2 * self.n_lj[l][j])) * np.power(r/a0,self.n_lj[l][j]-1) * np.exp(-self.Z_lj[l][j] * r/a0)
        return radial_wavefunction
    

    def dRdr(self,n,l,r):
        derivative = 0
        for j in range(len(self.C_nlj[n-1][l])):
            derivative += self.C_nlj[n-1][l][j] * mp.power(2 * self.Z_lj[l][j], self.n_lj[l][j]+1/2) / mp.sqrt(mp.factorial(2 * self.n_lj[l][j])) * (self.n_lj[l][j]-1) / a0 * mp.power(r/a0,self.n_lj[l][j]-2) * mp.exp(-self.Z_lj[l][j] * r/a0)
            derivative += self.C_nlj[n-1][l][j] * mp.power(2 * self.Z_lj[l][j], self.n_lj[l][j]+1/2) / mp.sqrt(mp.factorial(2 * self.n_lj[l][j])) * mp.power(r/a0,self.n_lj[l][j]-1) * (-self.Z_lj[l][j] /a0) * mp.exp(-self.Z_lj[l][j] * r/a0)
        return mp.power(a0,-3/2) * derivative

    def dRdr_2(self,n,l,r):
        derivative = 0
        for j in range(len(self.C_nlj[n-1][l])):
            derivative += self.C_nlj[n-1][l][j] * np.power(2 * self.Z_lj[l][j], self.n_lj[l][j]+1/2) / np.sqrt(np.math.factorial(2 * self.n_lj[l][j])) * (self.n_lj[l][j]-1) / a0 * np.power(r/a0,self.n_lj[l][j]-2) * np.exp(-self.Z_lj[l][j] * r/a0)
            derivative += self.C_nlj[n-1][l][j] * np.power(2 * self.Z_lj[l][j], self.n_lj[l][j]+1/2) / np.sqrt(np.math.factorial(2 * self.n_lj[l][j])) * np.power(r/a0,self.n_lj[l][j]-1) * (-self.Z_lj[l][j] /a0) * np.exp(-self.Z_lj[l][j] * r/a0)
        return np.power(a0,-3/2) * derivative

    def Z_effective(self,n,l):
        return self.Z_eff[n-1][l]

# Final state wave function
def R_final_kl_2(r,k,l,Z_eff):  
    np1f1 = np.vectorize(mp.hyp1f1)
    result = 4 * np.pi * (2*k*r)**l * np.exp(pi * Z_eff /2/k/a0 + (special.loggamma(l+1-1j * Z_eff/ k / a0)).real) / np.math.factorial(2*l+1) * (np.exp(-1j*k*r) * np1f1(l+1+1j*Z_eff/k/a0,(2*l+2),2j*k*r,maxterms=1000000)).real
    return result
    # return 4 * np.pi * (2*k*r)**l * abs(special.gamma(l+1-1j * Z_eff/ k / a0)) * np.exp(pi * Z_eff /2/k/a0) / np.math.factorial(2*l+1) * (np.exp(-1j*k*r) * mp.hyp1f1(l+1+1j*Z_eff/k/a0,(2*l+2),2j*k*r,maxterms=1000000)).real

def R_final_kl(r,k,l,Z_eff):
    return 4 * pi * (2*k*r)**l * abs( mp.gamma(l+1 - 1j*Z_eff / k / a0) ) * mp.exp(pi * Z_eff /2/k/a0) / mp.factorial(2*l+1) * (mp.expj(-k*r) * mp.hyp1f1(l+1+1j*Z_eff/k/a0,(2*l+2),2j*k*r,maxterms=1000000)).real


# Argon
C = [[ [0.316405, 0.542760, 0.167691, 0.000408, 0.002431, -0.000861, -0.000422, 0.000066, -0.000061, 0.000009] ] , [[0.079148, -0.507823, 0.059900, -0.026389, 0.832638, 0.295522, 0.000217, 0.002203, 0.001423, 0.000186] , [0.002436, -0.114774, -0.503175, -0.427033, 0.009669, -0.004825, 0.000231, -0.000098]] , [[0.035512, -0.181267, 0.026500, 0.006280, 0.111836, 0.385604, 0.000070, -0.376901, -0.593561, -0.229971],[0.001854, -0.042064, -0.095603, -0.194233, 0.005891, 0.366141, 0.526490, 0.249866]]]
Z = [[25.5708, 15.6262, 22.3994, 10.53, 7.0534, 5.412, 46.7052, 3.7982, 2.5495, 1.7965],[26.6358, 12.7337, 7.3041, 5.3353, 20.7765, 3.3171, 2.0947, 1.378]]
n = [[1, 1, 2, 2, 2, 2, 3, 3, 3, 3],[2, 2, 2, 2, 3, 3, 3, 3]]
E_B = [[-118.610349*au],[-12.322152*au,-9.571464*au],[-1.277352*au,-0.591016*au]]
Ar = Initial_Wavefunctions("Ar",C,Z,n,E_B)

# Xenon
C = [ [ [-0.965401, -0.040350, 0.001890, -0.003868, -0.000263, 0.000547, -0.000791, 0.000014, -0.000013, -0.000286, 0.000005, -0.000003, 0.000001] ] , [ [0.313912, 0.236118, -0.985333, 0.000229, -0.346825, 0.345786, -0.120941, -0.005057, 0.001528, -0.151508, -0.000281, 0.000134, -0.000040] , [0.051242, 0.781070, 0.114910, -0.000731, 0.000458, 0.083993, -0.000265, 0.000034, 0.009061, -0.000014, 0.000006, -0.000002] ] , [[-0.140382, -0.125401, 0.528161, -0.000435, 0.494492, -1.855445, 0.128637, -0.017980, 0.000792, 0.333907, -0.000228, 0.000191, -0.000037],[0.000264, 0.622357, -0.009861, -0.952677, -0.337900, -0.026340, -0.000384, -0.001665, 0.087491, 0.000240, -0.000083, 0.000026],[0.220185, 0.603140, 0.194682, -0.014369, 0.049865, -0.000300, 0.000418, -0.000133]] , [[0.064020, 0.059550, -0.251138, 0.000152, -0.252274, 1.063559, -0.071737, -0.563072, -0.697466, -0.058009, -0.018353, 0.00292, -0.000834],[0.013769, -0.426955, 0.045088, 0.748434, 0.132850, 0.059406, -0.679569, -0.503653, -0.149635, -0.014193, 0.000528, -0.000221],[-0.013758, -0.804573, 0.260624, 0.007490, 0.244109, 0.597018, 0.395554, 0.039786]] , [[-0.022510, -0.021077, 0.088978, -0.000081, 0.095199, -0.398492, 0.025623, 0.274471, 0.291110, 0.011171, -0.463123, -0.545266, -0.167779],[-0.005879, 0.149040, -0.018716, -0.266839, -0.031096, -0.024100, 0.267374, 0.161460, 0.059721, -0.428353, -0.542284, -0.201667]] ]
Z = [ [54.9179, 47.25, 26.0942, 68.1771, 16.8296, 12.0759, 31.903, 8.0145, 5.8396, 14.7123, 3.8555, 2.6343, 1.8124] , [58.7712, 22.6065, 48.9702, 13.4997, 9.8328, 40.2591, 7.1841, 5.1284, 21.533, 3.4469, 2.2384, 1.4588] , [19.9787, 12.2129, 8.6994, 27.7398, 15.941, 6.058, 4.099, 2.5857] ]
n = [ [1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5] , [2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5] , [3, 3, 3, 4, 4, 4, 4, 4] ]
E_B = [ [-1224.397767 * au] , [-189.230111 * au,-177.782438 * au] , [-40.175652 * au,-35.221651 * au,-26.118859 * au] , [-7.856291 * au,-6.008328 * au,-2.777871 * au] , [-0.944407 * au,-0.457283 * au] ]
Xe = Initial_Wavefunctions("Xe",C,Z,n,E_B)