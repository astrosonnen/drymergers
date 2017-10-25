#This code calculates model brightness-weighted velocity dispersions within a circular aperture of half an effective radius for a de Vaucouleurs profile and a gNFW.

from numpy import *
from vdmodel_2013 import sigma_model
from vdmodel_2013.profiles import deVaucouleurs
from lensingtools import sersic, gNFW
import pickle

Ngam = 271
gammas = linspace(0.1,2.8,Ngam)

s2s = 0.*gammas

reff = 1.
aperture = 0.5*reff
rs = 10.*reff


f = open('/data3/sonnen/Python/lensingtools/deV_dyn.dat','r')
rdyn_deV,M3d_deV = pickle.load(f)
f.close()

f = open('/data3/sonnen/Python/SL2S/gNFW_rs30kpc_M3d_dyn.dat','r')
rdyn,M3d = pickle.load(f)
f.close()

s2stars = sigma_model.sigma2general((rdyn_deV,M3d_deV),aperture,reff,seeing=None,light_profile=deVaucouleurs)
print s2stars

lines = []
lines.append('#gamma\tsigma2\tM3d/M2d\n')

ratios = 0.*gammas

for i in arange(Ngam):
    norm = 1./gNFW.M2d(reff,rs,gammas[i])
    #s2s[i] = sigma_model.sigma2general((rdyn,M3d[i,:]/gNFW.M2d(reff*(30./6.),30,gammas[i])),aperture*30/6.,lp_pars=reff*(30./6.),seeing=None,light_profile=deVaucouleurs)
    s2s[i] = sigma_model.sigma2general(lambda r: norm*gNFW.M3d(r,rs,gammas[i]),aperture,lp_pars=reff,seeing=None,light_profile=deVaucouleurs)
    ratios[i] = gNFW.M3d(reff,rs,gammas[i])*norm

#s2s /= 6./30.

for i in arange(Ngam):
    lines.append('%4.2f %7.5f %7.5f\n'%(gammas[i],s2s[i],ratios[i]))

f = open('gNFW_rs10reff_grid_s2s_deV_0.5reff.txt','w')
f.writelines(lines)
f.close()

