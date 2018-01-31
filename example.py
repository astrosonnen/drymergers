import numpy as np
from drymergers import shmrs, popevol, galpop_recipes as recipes
import pickle
from ConfigParser import SafeConfigParser

config = SafeConfigParser()
config.read("parameters.ini")

n_gal   = eval(config.get("my_parameters_example", "n_gal"  )) # sample size
model   = eval(config.get("my_parameters_example", "model"  )) # adopted model for the SHMR
z_start = eval(config.get("my_parameters_example", "z_start")) # initial redshift
z_end   = eval(config.get("my_parameters_example", "z_end"  )) # final redshift
ximin   = eval(config.get("my_parameters_example", "xi_min" )) # minimum merger mass ratio
lmhmin  = eval(config.get("my_parameters_example", "lmhmin" )) # minimum halo mass (in log10)
lmhmax  = eval(config.get("my_parameters_example", "lmhmax" )) # maximum halo mass (in log10)

print(n_gal, model, z_start, ximin, z_end)

imf_coeff = (0.21, 0.26, 0.) # coefficients describing the IMF normalization of galaxies at z_start

vdisp_coeff = (2.34 + 0.20*(np.log10(1. + z_start) - np.log10(1.2)), 0.18) # coefficients of mchab-vdisp relation (from Auger et al. 2010, z-dependence from Mason et al. 2015)

outname = "pop_model/pop_model_"+str(n_gal)+"_"+str(model)+"_"+str(z_start)+"_"+str(z_end)+"_"+str(lmhmin)+"_"+str(lmhmax)+".dat"

lmhalos = shmrs.generate_halos(100000, z=z_start, lmhmin=lmhmin, lmhmax=lmhmax) # generate halos. Exclude clusters.
lmstars = shmrs.generate_mstar(lmhalos, z=z_start, scat=0, model=model) # assigns stellar masses, using B10 model.
selection = lmstars > 10.3 # makes a cut in stellar mass
indices = np.arange(100000)
ind_sample = np.random.choice(indices[selection], n_gal) # selects a random sample of n_gal objects ### C: np.random.choice(...) - Generates a random sample from a given 1-D array

lmhalo_sample = lmhalos[ind_sample]
lmstar_sample = lmstars[ind_sample]

vdisp_sample = 10.**(vdisp_coeff[0] + vdisp_coeff[1]*(lmstar_sample - 11.) + np.random.normal(0., 0.04, n_gal))

aimf_z2_sample = 0.*lmstar_sample ### C:  Unused!!!

pop = popevol.population(z_0=z_start, nobj=n_gal, mstar_chab_0=10.**lmstar_sample, mhalo_0=10.**lmhalo_sample, veldisp_0=vdisp_sample)

# =============================================================================
# logmhalo=[]
# logmstarlogmhalo=[]
# 
# for i in range(len(shmrs.logmhalo[0])):
# #    print(shmrs.logmhalo[0][i], shmrs.logmstarlogmhalo[0][0][i])
#     logmhalo.append(shmrs.logmhalo[0][i])
#     logmstarlogmhalo.append(shmrs.logmstarlogmhalo[0][0][i])
# 
# data=np.array([logmhalo,logmstarlogmhalo])
# data=data.T
# np.savetxt("B10_L12_SNT17/"+str(model)+"_"+str(z_start)+"_"+str(z_end)+".dat", data)
# =============================================================================

pop.evolve(z_low=z_end, imf_recipe='mstar-vdisp', imf_coeff=imf_coeff, vdisp_coeff=vdisp_coeff, ximin=ximin, v200sigma_rat2=0.6, vorbv200_rat=0.75, model=model)

f = open(outname, 'w')

pickle.dump(pop, f)
f.close()