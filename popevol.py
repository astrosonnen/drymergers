import numpy as np
from drymergers.cgsconstants import *
from scipy.interpolate import splrep,splev
from drymergers import shmrs
from drymergers import galpop_recipes as recipes
from scipy.integrate import quad
from drymergers import fitting_tools


A = 0.0104
xitilde = 9.72e-3
alpha = 0.133
beta = -1.995
gamma = 0.263
etap = 0.0993

betaR = 0.6

apar = 1.11
bpar = 1.10
Mdot = 46.1/yr

Ngrid = 1001
lmstar_grid = np.linspace(9., 13., Ngrid)

def izfunc(z, z_0):
    return apar*(z - z_0) - (apar-1.)*np.log((1.+z)/(1.+z_0))


class population:

    def __init__(self, z_0=2., nobj=100, mstar_chab_0=None, mhalo_0=None, veldisp_0=None):

        self.z_0 = z_0
        self.nobj = nobj
        self.mstar_chab_0 = mstar_chab_0
        self.mhalo_0 = mhalo_0
        self.veldisp_0 = veldisp_0
        self.z = None
        self.mhalo = None
        self.mstar_chab = None
        self.veldisp = None
        self.xieff = None
        self.rfuncatxieff = None
        self.dlnsigma2_dz = None
        self.vdisp_coeff = None


    def evolve(self, ximin=0.03, dz=0.01, z_low=0., \
               vdisp_coeff=(2.40, 0.18), v200sigma_rat2 = 0.65, vorbv200_rat = 1.1):

        self.z = np.arange(z_low, self.z_0, dz)
        nz = len(self.z)

        self.mhalo = np.zeros((self.nobj, nz), dtype='float')
        self.mstar_chab = 0.*self.mhalo
        self.veldisp = 0.*self.mhalo
        self.dlnsigma2_dz = 0.*self.mhalo

        self.mstar_chab[:, -1] = self.mstar_chab_0
        self.veldisp[:, -1] = self.veldisp_0

        self.vdisp_coeff = np.zeros((nz, 2), dtype='float')
        self.vdisp_coeff[-1] = vdisp_coeff

        integral = xitilde*quad(lambda x: x**(beta+1./3.)*np.exp(x**gamma), ximin/xitilde, 1./xitilde)[0]
        
        for i in range(self.nobj):
            self.mhalo[i, :] = 1e12*((self.mhalo_0[i]/1e12)**(1.-bpar) - (1-bpar)/H0*Mdot/1e12*izfunc(self.z, self.z_0))**(1./(1.-bpar)) ### C: halo mass evolution in z 

        for i in range(nz-2, -1, -1): # loop in z, from high to low

            lmhalo_grid = shmrs.mhfunc(lmstar_grid, self.z[i]) # A: evaluates Mh(M*, z) on a grid
            lmstar_spline = splrep(lmhalo_grid, lmstar_grid) # A: creates spline interpolation of M*(Mh), for faster evaluation

            def rfunc(mhalo): # stellar to halo mass ratio, calculated by spline interpolation
                return 10.**(splev(np.log10(mhalo), lmstar_spline) - np.log10(mhalo))

            lmchab_here = splev(np.log10(self.mhalo[:, i]), lmstar_spline)
            self.mstar_chab = 10.**lmchab_here
            
            
            
            for j in range(self.nobj): # loop over individual galaxies

                self.dlnsigma_dz[j, i] = -A*integral*(self.mhalo[j, i]/1e12)**alpha*(1.+self.z[i])**etap

                self.veldisp[j, i] = self.veldisp[j, i+1]*(1. - 0.5*self.dlnsigma2_dz[j, i]*dz)
                
            # now fits for the vdisp - mstar relation and imf - mstar relation
            fit_vdisp_coeff = fitting_tools.fit_mchab_dep(np.log10(self.mstar_chab[:, i]), np.log10(self.veldisp[:, i]), \
                                                 guess=vdisp_coeff)[0]

            vdisp_coeff = fit_vdisp_coeff
            self.vdisp_coeff[i, :] = vdisp_coeff # updates M*-sigma relation


