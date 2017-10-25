import numpy as np
from lensingtools import sersic,generalized_her,gNFW,NFW,truncNFW
from vdmodel_2013 import sigma_model
from vdmodel_2013.profiles import deVaucouleurs,gen_her
import pickle
from scipy.interpolate import splev,splrep

f = open('/data3/sonnen/Python/lensingtools/gen_her_reff_spline.dat','r')
reff_spline = pickle.load(f)
f.close()

drydir = '/data3/sonnen/Python/drymergers/'

f = open(drydir+'gNFW_grid_s2s_deV_0.5reff.txt','r')
grid = np.loadtxt(f)
f.close()

gamma_grid = grid[:,0]
s2s_grid = grid[:,1]

s2s_gnfw_spline = splrep(gamma_grid,s2s_grid)

s2s_deV = 0.1904

class NTB09_model:

    def __init__(self,mstar=1.,reff=1.,gammastar=1.5,mdm=1.,rs=10.,rvir=1.):
        self.mstar = mstar
        self.reff = reff
        self.gammastar = gammastar
        self.mdm = mdm
        self.rs = rs
        self.rvir = rvir
        self.apar = None
        self.sigma2 = None
        self.mein = None

    def get_apar(self):
        self.apar = self.reff/splev(self.gammastar,reff_spline)
        #self.apar = self.reff/1.243

    def get_sigma(self):

        self.get_apar()

        norm = self.mdm/truncNFW.M3d(np.inf,self.rs,self.rvir)

        s2_halo = sigma_model.sigma2general(lambda r: norm*truncNFW.M3d(r,self.rs,self.rvir),aperture=0.5*self.reff,seeing=None,lp_pars=(self.apar,self.gammastar),light_profile=gen_her)

        s2_stars = sigma_model.sigma2general(lambda r: self.mstar*generalized_her.M3d(r,self.apar,self.gammastar),aperture=0.5*self.reff,seeing=None,lp_pars=(self.apar,self.gammastar),light_profile=gen_her)

        self.sigma2 = s2_stars + s2_halo
        
    def get_mein(self):

        norm = self.mdm/truncNFW.M3d(np.inf,self.rs,self.rvir)
        
        self.mein = 0.5*self.mstar + norm*truncNFW.M2d(self.reff,self.rs,self.rvir)



class deV_gNFW:

    def __init__(self,mstar=1.,reff=1.,mdm=1.,rs=6.,gammadm=1.):
        self.mstar = mstar
        self.reff = reff
        self.mdm = mdm
        self.rs = rs
        self.gammadm = gammadm
        self.sigma2 = None
        self.mein_dm = None

    def quick_sigma(self):

        s2_halo = self.mein_dm*splev(self.gammadm,s2s_gnfw_spline)

        s2_stars = self.mstar*s2s_deV

        self.sigma2 = (s2_halo + s2_stars)/self.reff
        

    def mein_dm_from_mein(self,mein):

        self.mein_dm = mein - 0.5*self.mstar



