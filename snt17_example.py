import numpy as np
from drymergers import shmrs, popevol, galpop_recipes as recipes
import pickle


Ngal = 100 # sample size
#%%
z_0 = 2. # starting redshift
ximin = 0.03 # minimum merger mass ratio

imf_coeff = (0.21, 0.26, 0.) # coefficients describing the IMF normalization of galaxies at z_0

vdisp_coeff = (2.34 + 0.20*(np.log10(1. + z_0) - np.log10(1.2)), 0.18) # coefficients of mchab-vdisp relation (from Auger et al. 2010, z-dependence from Mason et al. 2015)

outname = 'pop_model_'+str(Ngal)+'.dat' ### C: introduction of 'str(Ngal)'
#%%
lmhalos = shmrs.generate_halos(100000, z=2., lmhmax=14.) # generate halos. Exclude clusters.
lmstars = shmrs.generate_mstar(lmhalos, z=2., scat=0.18) # assigns stellar masses, using B10 model.
selection = lmstars > 10.3 # makes a cut in stellar mass
indices = np.arange(100000)
ind_sample = np.random.choice(indices[selection], Ngal) # selects a random sample of Ngal objects ### C: np.random.choice(...) - Generates a random sample from a given 1-D array
#%%
lmhalo_sample = lmhalos[ind_sample]
lmstar_sample = lmstars[ind_sample]
#%%
vdisp_sample = 10.**(vdisp_coeff[0] + vdisp_coeff[1]*(lmstar_sample - 11.) + np.random.normal(0., 0.04, Ngal))
#%%
aimf_z2_sample = 0.*lmstar_sample ### C:  Unused!!!
#%%
pop = popevol.population(z_0=z_0, nobj=Ngal, mstar_chab_0=10.**lmstar_sample, mhalo_0=10.**lmhalo_sample, \
                         veldisp_0=vdisp_sample)
#%%
pop.evolve(z_low=0., imf_recipe='mstar-vdisp', imf_coeff=imf_coeff, vdisp_coeff=vdisp_coeff, ximin=ximin, v200sigma_rat2=0.6, vorbv200_rat=0.75)
#%%
f = open(outname, 'w')

#%%
pickle.dump(pop, f)
f.close()
#%%