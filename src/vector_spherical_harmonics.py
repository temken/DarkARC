import numpy as np
from scipy.special import sph_harm

# Vector spherical harmonics (VSH) following the conventions of of Barrera et al.

# First VSH \vec{Y}_{lm} = Y_\lm \vec{r}

def vector_spherical_harmonics_Y(l,m,theta,phi):
	unit_vectors =[ np.array([1,0,0]) , np.array([0,1,0]) ,  np.array([0,0,1]) ]
	erYlm = np.array([0+0j,0+0j,0+0j])
	for i in range(3):
		for lHat in range(l-1,l+2,2):
			for mHat in range(m-1,m+2):
				if mHat in range(-lHat,lHat+1): 
					erYlm[i] += VSH_coefficients_Y(i+1,l,m,lHat,mHat) * sph_harm(mHat,lHat,phi,theta)

	return erYlm


def VSH_coefficients_Y(component,l,m,lHat,mHat):
	if component not in [1,2,3]:
		sys.exit("Error in VSH_coefficients_Y(): Component out of bound.")
	if component == 1:
		if (lHat not in [l-1,l+1]) or (mHat not in [m-1,m+1]):
			return 0.0
		elif lHat == l+1 and mHat == m+1:
			return -1./2 * np.sqrt((l+m+1) * (l+m+2) / (2*l+3) / (2*l+1))
		elif lHat == l+1 and mHat == m-1:
			return 1./2 * np.sqrt((l-m+1) * (l-m+2) / (2*l+3) / (2*l+1))
		elif lHat == l-1 and mHat == m+1:
			return 1./2 * np.sqrt((l-m-1) * (l-m) / (2*l-1) / (2*l+1))
		elif lHat == l-1 and mHat == m-1:
			return -1./2 * np.sqrt((l+m-1) * (l+m) / (2*l-1) / (2*l+1))
	elif component == 2:
		if (lHat not in [l-1,l+1]) or (mHat not in [m-1,m+1]):
			return 0.0
		elif lHat == l+1 and mHat == m+1:
			return 1j/2 * np.sqrt((l+m+1) * (l+m+2) / (2*l+3) / (2*l+1))
		elif lHat == l+1 and mHat == m-1:
			return 1j/2 * np.sqrt((l-m+1) * (l-m+2) / (2*l+3) / (2*l+1))
		elif lHat == l-1 and mHat == m+1:
			return -1j/2 * np.sqrt((l-m-1) * (l-m) / (2*l-1) / (2*l+1))
		elif lHat == l-1 and mHat == m-1:
			return -1j/2 * np.sqrt((l+m-1) * (l+m) / (2*l-1) / (2*l+1))
	elif component == 3:
		if (lHat not in [l-1,l+1]) or (mHat != m):
			return 0.0
		elif lHat == l+1:
			return  np.sqrt((l-m+1) * (l+m+1) / (2*l+3) / (2*l+1))
		elif lHat == l-1:
			return np.sqrt((l-m) * (l+m) / (2*l-1) / (2*l+1))

# Second VSH \vec{\Psi}_{lm} = r \grad{Y_\lm} 

def vector_spherical_harmonics_Psi(l,m,theta,phi):
	unit_vectors =[ np.array([1,0,0]) , np.array([0,1,0]) ,  np.array([0,0,1]) ]
	gradYlm = np.array([0+0j,0+0j,0+0j])
	for i in range(3):
		for lHat in range(l-1,l+2,2):
			for mHat in range(m-1,m+2):
				if mHat in range(-lHat,lHat+1): 
					gradYlm[i] += VSH_coefficients_Psi(i+1,l,m,lHat,mHat) * sph_harm(mHat,lHat,phi,theta)

	return gradYlm

def VSH_coefficients_Psi(component,l,m,lHat,mHat):
	if component not in [1,2,3]:
		sys.exit("Error in VSH_coefficients_Psi(): Component out of bound.")
	if component == 1:
		if (lHat not in [l-1,l+1]) or (mHat not in [m-1,m+1]):
			return 0.0
		elif lHat == l+1 and mHat == m+1:
			return l/2 * np.sqrt((l+m+1) * (l+m+2) / (2*l+3) / (2*l+1))
		elif lHat == l+1 and mHat == m-1:
			return -l/2 * np.sqrt((l-m+1) * (l-m+2) / (2*l+3) / (2*l+1))
		elif lHat == l-1 and mHat == m+1:
			return (l+1)/2 * np.sqrt((l-m-1) * (l-m) / (2*l-1) / (2*l+1))
		elif lHat == l-1 and mHat == m-1:
			return -(l+1)/2 * np.sqrt((l+m-1) * (l+m) / (2*l-1) / (2*l+1))
	elif component == 2:
		if (lHat not in [l-1,l+1]) or (mHat not in [m-1,m+1]):
			return 0.0
		elif lHat == l+1 and mHat == m+1:
			return -1j*l/2 * np.sqrt((l+m+1) * (l+m+2) / (2*l+3) / (2*l+1))
		elif lHat == l+1 and mHat == m-1:
			return -1j*l/2 * np.sqrt((l-m+1) * (l-m+2) / (2*l+3) / (2*l+1))
		elif lHat == l-1 and mHat == m+1:
			return -1j*(l+1)/2 * np.sqrt((l-m-1) * (l-m) / (2*l-1) / (2*l+1))
		elif lHat == l-1 and mHat == m-1:
			return -1j*(l+1)/2 * np.sqrt((l+m-1) * (l+m) / (2*l-1) / (2*l+1))
	elif component == 3:
		if (lHat not in [l-1,l+1]) or (mHat != m):
			return 0.0
		elif lHat == l+1:
			return  -l * np.sqrt((l-m+1) * (l+m+1) / (2*l+3) / (2*l+1))
		elif lHat == l-1:
			return (1+l) * np.sqrt((l-m) * (l+m) / (2*l-1) / (2*l+1))