import numpy as np
import shmrs
from drymergers import popevol, galpop_recipes as recipes
import pickle
import ndinterp
from scipy.interpolate import splrep


Ngal = 1000 # sample size

nv200 = 5
nvorb = 5

v200sigma_rat_grid = np.linspace(0.2, 1., nv200)
vorbv200_rat_grid = np.linspace(0.5, 1.5, nvorb)

vdisp_b_grid = np.zeros((nv200, nvorb))
vdisp_a_grid = np.zeros((nv200, nvorb))

v200sigma_rat_spline = splrep(v200sigma_rat_grid, np.arange(nv200))
vorbv200_rat_spline = splrep(vorbv200_rat_grid, np.arange(nvorb))

axes = {0: v200sigma_rat_spline, 1: vorbv200_rat_spline}

z_0 = 2.
ximin = 0.03

modelname = 'mstar'
imf_coeff = (0.21, 0.26, 0.)

vdisp_coeff = (2.34 + 0.20*(np.log10(3.) - np.log10(1.2)), 0.18)

lmhalos = shmrs.generate_halos(100000, z=2., lmhmax=13.5)
lmstars = shmrs.generate_mstar(lmhalos, z=2., scat=0.18)
selection = lmstars > 10.3
indices = np.arange(100000)
ind_sample = np.random.choice(indices[selection], Ngal)

lmhalo_sample = lmhalos[ind_sample]
lmstar_sample = lmstars[ind_sample]
print len(lmstar_sample[lmstar_sample>12])

vdisp_sample = 10.**(vdisp_coeff[0] + vdisp_coeff[1]*(lmstar_sample - 11.) + np.random.normal(0., 0.04, Ngal))

aimf_z2_sample = 0.*lmstar_sample

pop = popevol.population(z_0=z_0, nobj=Ngal, mstar_chab_0=10.**lmstar_sample, mhalo_0=10.**lmhalo_sample, \
                         veldisp_0=vdisp_sample)

f = open('pop_for_grid_1000.dat', 'w')
pickle.dump(pop, f)
f.close()

saved = False

for i in range(nv200):
    for j in range(nvorb):
        print i, j

        pop.evolve(z_low=0., imf_recipe='mstar-vdisp', imf_coeff=imf_coeff, vdisp_coeff=vdisp_coeff, ximin=ximin, v200sigma_rat2=v200sigma_rat_grid[i], vorbv200_rat=vorbv200_rat_grid[j])

        vdisp_b_grid[i, j] = pop.vdisp_coeff[20, 0]
        vdisp_a_grid[i, j] = pop.vdisp_coeff[20, 1]

        if not saved:
            f = open('popevol_for_grid_1000.dat', 'w')
            pickle.dump(pop, f)
            f.close()

            saved = True

b_ndinterp = ndinterp.ndInterp(axes, vdisp_b_grid, order=1)
a_ndinterp = ndinterp.ndInterp(axes, vdisp_a_grid, order=1)

f = open('vdisp_grid_b2.dat', 'w')
pickle.dump(b_ndinterp, f)
f.close()

f = open('vdisp_grid_a2.dat', 'w')
pickle.dump(a_ndinterp, f)
f.close()

