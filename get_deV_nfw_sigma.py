#This code calculates model brightness-weighted velocity dispersions within a circular aperture of half an effective radius for a de Vaucouleurs profile and a gNFW.

from numpy import *
from vdmodel_2013 import sigma_model
from vdmodel_2013.profiles import deVaucouleurs
from lensingtools import sersic, NFW
import pickle


reff = 1.
aperture = 0.5*reff
rs = 10.*reff


f = open('/data3/sonnen/Python/lensingtools/deV_dyn.dat','r')
rdyn_deV,M3d_deV = pickle.load(f)
f.close()

s2stars = sigma_model.sigma2general((rdyn_deV,M3d_deV),aperture,reff,seeing=None,light_profile=deVaucouleurs)
print s2stars


s2halo = sigma_model.sigma2general(lambda r: NFW.M3d(r,rs)/NFW.M2d(1.,rs),aperture,lp_pars=reff,seeing=None,light_profile=deVaucouleurs)
print s2halo

