# Units and constants

from math import pi

#Energy
GeV = 1.0;
eV = 1.0E-9*GeV;
keV = 1.0E-6*GeV;
MeV = 1.0E-3*GeV;
TeV = 1.0E3*GeV;
au = 27.211386245988*eV

#Mass
gram = 5.60958884493318e23*GeV;
kg = 1e3*gram;

#Length
cm = 5.067730214314311e13/GeV;
meter = 100*cm;
km = 1000*meter;
fm = 1e-15*meter;
pb = 1e-36*(cm)**2;
parsec = 3.0857e16*meter;
kpc = 1e3*parsec;
Mpc = 1e6*parsec;
a0  =  5.29177e-11*meter; #Bohr radius

#Time
sec = 299792458*meter;
minute = 60*sec;
hour = 60*minute;
day = 24*hour;
year = 365.24*day;

#More energy
erg = gram*(cm/sec)**2;
Joule = kg*(meter/sec)**2;

#Temperature
Kelvin = 8.6173303e-14*GeV;

#Angle
deg = pi/180.0;
    
#Masses
mPlanck =  1.2209E19*GeV;
mProton = 0.938*GeV ;
mElectron =  0.511*MeV;
mNucleon  =  0.932*GeV;

#Coupling constants
aEM  =  1.0/137.035999139;
GNewton = mPlanck**(-2);
GFermi  =  1.16637e-5*GeV**(-2);

#Reference momentum transfer
qref  =  aEM*mElectron

#Unit conversion function
def in_units(quantity,dimension):
    return quantity/dimension