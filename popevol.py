import numpy as np
from drymergers.cgsconstants import *
from scipy.interpolate import splrep,splev
from drymergers import shmrs
from drymergers import galpop_recipes as recipes
from scipy.integrate import quad
from drymergers import fitting_tools
import sys

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

# gammap parameters
a_gammap = 0.6
b_gammap = -0.73

def fgfunc(xi):
    return a_gammap * xi + b_gammap

def izfunc(z, z_0):
    return apar*(z - z_0) - (apar-1.)*np.log((1.+z)/(1.+z_0))

class population:

    def __init__(self, z_0=2., nobj=100, mstar_chab_0=None, mhalo_0=None, veldisp_0=None, gammap_0=None):

        self.z_0 = z_0
        self.nobj = nobj
        self.mstar_chab_0 = mstar_chab_0
        self.mstar_salp_0 = mstar_chab_0*10.**0.25
        self.mhalo_0 = mhalo_0
        self.veldisp_0 = veldisp_0
        self.z = None
        self.mhalo = None
        self.mstar_chab = None
        self.mstar_true = None
        self.veldisp = None
        self.aimf = None
        self.xieff = None
        self.rfuncatxieff = None
        self.dlnsigma2_dz = None
        self.vdisp_coeff = None
        self.imf_coeff = None
        self.gammap_0 = gammap_0
        self.gammap = None

    def evolve(self, ximin=0.03, dz=0.01, z_low=0., imf_recipe='mstar-vdisp', imf_coeff=(0.2, 0.2, 0.5), \
               vdisp_coeff=(2.40, 0.18), do_gammap=False, v200sigma_rat2 = 0.65, vorbv200_rat = 1.1, model="B10"):

        self.z = np.arange(z_low, self.z_0, dz)
        nz = len(self.z)

        self.mhalo = np.zeros((self.nobj, nz), dtype='float')
        self.mstar_chab = 0.*self.mhalo
        self.mstar_salp = 0.*self.mhalo
        self.mstar_true = 0.*self.mhalo
        self.veldisp = 0.*self.mhalo
        self.gammap = 0.*self.mhalo
        self.aimf = 0.*self.mhalo
        self.dlnsigma2_dz = 0.*self.mhalo

        self.mstar_chab[:, -1] = self.mstar_chab_0
        self.mstar_salp[:, -1] = self.mstar_salp_0
        self.veldisp[:, -1] = self.veldisp_0
        if do_gammap:
            self.gammap[:, -1] = self.gammap_0

        self.vdisp_coeff = np.zeros((nz, 2), dtype='float')
        self.vdisp_coeff[-1] = vdisp_coeff

        self.imf_coeff = np.zeros((nz, len(imf_coeff)), dtype='float')
        self.imf_coeff[-1] = imf_coeff

        for i in range(self.nobj):
            self.mhalo[i, :] = 1e12*((self.mhalo_0[i]/1e12)**(1.-bpar) - (1-bpar)/H0*Mdot/1e12*izfunc(self.z, self.z_0))**(1./(1.-bpar)) ### C: halo mass evolution in z 
            if imf_recipe == 'mstar':
                self.aimf[i, -1] = 10.**recipes.limf_func_mstar(np.log10(self.mstar_salp_0[i]), imf_coeff)
            elif imf_recipe == 'vdisp':
                self.aimf[i, -1] = 10.**recipes.limf_func_vdisp(np.log10(self.veldisp_0[i]), imf_coeff)
            elif imf_recipe == 'mstar-vdisp':
                self.aimf[i, -1] = 10.**recipes.limf_func_mstarvdisp(np.log10(self.mstar_salp_0[i]), np.log10(self.veldisp_0[i]), imf_coeff)

        self.mstar_true[:, -1] += self.mstar_salp_0 * self.aimf[:, -1]

        self.xieff = 0.*self.mhalo
        self.rfuncatxieff = 0.*self.mhalo

        for i in range(nz-2, -1, -1): # loop in z, from high to low

            if (model=="B10" or model=="L12_sig_mod1" or model=="L12_sig_mod2" or model=="SNT17_original" or model=="SNT17_corrected"):

                lmhalo_grid = shmrs.mhfunc(lmstar_grid, self.z[i], model) # A: evaluates Mh(M*, z) on a grid
                lmhalo_spline = splrep(lmhalo_grid, lmstar_grid) # A: creates spline interpolation of M*(Mh), for faster evaluation
    
                def rfunc(mhalo, model): # stellar to halo mass ratio, calculated by spline interpolation
                    return 10.**(splev(np.log10(mhalo), lmhalo_spline) - np.log10(mhalo))
                
            elif(model=="B13"):
                def rfunc(mhalo, model): # stellar to halo mass ratio, calculated by spline interpolation
                    return 10.**(shmrs.mstarfunc(np.log10(mhalo), self.z[i], model) - np.log10(mhalo))

            else:
                print("Wrong way!")
                sys.exit(1000)
                                
            for j in range(self.nobj): # loop over individual galaxies
                imz = quad(
                    lambda xi: rfunc(xi*self.mhalo[j, i], model)*xi**(beta+1.)*np.exp((xi/xitilde)**gamma), ximin, 1.)[0]
                imxiz = quad(
                    lambda xi: rfunc(xi*self.mhalo[j, i], model)*xi**(beta+2.)*np.exp((xi/xitilde)**gamma), ximin, 1.)[0]

                self.xieff[j, i] = imxiz/imz # effective merger halo mass ratio
                self.rfuncatxieff[j, i] = rfunc(self.xieff[j, i]*self.mhalo[j, i], model)

                dmstar_chab_dz = -A*imz*self.mhalo[j, i]*(self.mhalo[j, i]/1e12)**alpha*(1.+self.z[i])**etap # change in mchab due to mergers in this timestep (Nipoti 2012)

                def integrand(xi):
                    lmstar_here = np.log10(xi*self.mhalo[j, i]*rfunc(xi*self.mhalo[j, i], model))
                    thing = recipes.satellite_imf(lmstar_here, z=self.z[i], recipe=imf_recipe, coeff=imf_coeff, \
                                                  lvdisp=self.vdisp_coeff[i+1, 0] + self.vdisp_coeff[i+1, 1]*(lmstar_here - 11.))* \
                            rfunc(xi*self.mhalo[j, i], model)*xi**(beta+1.)*np.exp((xi/xitilde)**gamma)
                    return thing

                imtz = quad(integrand, ximin, 1.)[0]

                dmstar_true_dz = -A*imtz*self.mhalo[j,i]*(self.mhalo[j, i]/1e12)**alpha*(1.+self.z[i])**etap  

                def epsilon(xi):
                    mstar_sat = rfunc(self.mhalo[j, i]*xi, model)*xi*self.mhalo[j, i]
                    vdisp_sat = 10.**(self.vdisp_coeff[i+1, 0] + self.vdisp_coeff[i+1, 1]*(np.log10(mstar_sat) - 11.)) # satellite velocity dispersion is given by M*-sigma relation at previous timestep
                    return (vdisp_sat/self.veldisp[j, i+1])**2

                def epsilon_orb(xi):
                    return xi * v200sigma_rat2 * (vorbv200_rat**2/(1. + xi) - 1.) ### C: epsilon_orb,corr

                isigma2 = quad(lambda xi: ((1.+xi*epsilon(xi) - epsilon_orb(xi))/(1.+xi) - 1)*xi**beta*np.exp((xi/xitilde)**gamma), \
                               ximin, 1.)[0]

                self.dlnsigma2_dz[j, i] = -A*isigma2*(self.mhalo[j, i]/1e12)**alpha*(1.+self.z[i])**etap

                self.mstar_chab[j, i] = self.mstar_chab[j, i+1] - dmstar_chab_dz*dz
                self.mstar_salp[j, i] = self.mstar_chab[j, i]*10.**0.25
                self.mstar_true[j, i] = self.mstar_true[j, i+1] - dmstar_true_dz*dz

                self.veldisp[j, i] = self.veldisp[j, i+1]*(1. - 0.5*self.dlnsigma2_dz[j, i]*dz)

                if do_gammap:
                    igz = quad(
                    lambda xi: fgfunc(xi)*rfunc(xi*self.mhalo[j, i], model)*xi**(beta+1.)*np.exp((xi/xitilde)**gamma), ximin, 1.)[0]
                    dgammap = -A*igz*self.mhalo[j, i]*(self.mhalo[j, i]/1e12)**alpha*(1.+self.z[i])**etap

                    self.gammap[j, i] = self.gammap[j, i+1] - dgammap*dz

            # now fits for the vdisp - mstar relation and imf - mstar relation
            fit_vdisp_coeff = fitting_tools.fit_mchab_dep(np.log10(self.mstar_chab[:, i]), np.log10(self.veldisp[:, i]), \
                                                 guess=vdisp_coeff)[0]

            vdisp_coeff = fit_vdisp_coeff
            self.vdisp_coeff[i, :] = vdisp_coeff # updates M*-sigma relation
            self.aimf[:, i] = self.mstar_true[:, i] / self.mstar_salp[:, i]

            if imf_recipe == 'mstar':
                imf_coeff = fitting_tools.fit_mstar_only(np.log10(self.mstar_salp[:, i]), np.log10(self.aimf[:, i]), \
                                                   guess=imf_coeff)[0]
            elif imf_recipe == 'vdisp':
                 imf_coeff = fitting_tools.fit_sigma_only(np.log10(self.veldisp[:, i]), np.log10(self.aimf[:, i]), \
                                                   guess=imf_coeff)[0]
            elif imf_recipe == 'mstar-vdisp':
                  imf_coeff = fitting_tools.fit_mstar_sigma_fixed_z(np.log10(self.mstar_salp[:, i]), np.log10(self.veldisp[:, i]), np.log10(self.aimf[:, i]), \
                                                   guess=imf_coeff)[0]

            self.imf_coeff[i, :] = imf_coeff