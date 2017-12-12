import pymc
import numpy as np
import pickle
import pylab
from plotters import probcontour


f = open('vdisp_grid_b2.dat', 'r')
b_interp = pickle.load(f)
f.close()

f = open('vdisp_grid_a2.dat', 'r')
a_interp = pickle.load(f)
f.close()

a_obs = 0.18
a_err = 0.03

b_obs = 2.34
b_err = 0.01

v200_min = 0.2
v200_max = 1.
vorb_min = 0.5
vorb_max = 1.5

v200_par = pymc.Uniform('v200sigma_rat', lower=v200_min, upper=v200_max, value=0.6)
vorb_par = pymc.Uniform('vorbv200_rat', lower=vorb_min, upper=vorb_max, value=1.1)

pars = [v200_par, vorb_par]

@pymc.deterministic
def like(v200=v200_par, vorb=vorb_par):

    point = np.array((v200, vorb)).reshape((1, 2))

    a = a_interp.eval(point)
    b = b_interp.eval(point)

    return -0.5*(a - a_obs)**2/a_err**2 -0.5*(b - b_obs)**2/b_err**2

@pymc.stochastic
def logp(value=0., observed=True, p=pars):
    return like

M = pymc.MCMC(pars)
M.sample(60000, 10000)

pylab.plot(M.trace('v200sigma_rat')[:])
pylab.plot(M.trace('vorbv200_rat')[:])
#pylab.show()
pylab.close()

probcontour(M.trace('v200sigma_rat')[:], M.trace('vorbv200_rat')[:], style='r')
pylab.xlabel('$(v_{200}/\sigma)^2$', fontsize=16)
pylab.ylabel('$v/v_{200}$', fontsize=16)
pylab.savefig('orbital_pars_inference_v2.png')
#pylab.show()

