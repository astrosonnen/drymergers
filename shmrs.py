import numpy as np
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq
from scipy.integrate import quad


h = 0.7
N = 1001

lmstar_grid = np.linspace(9., 12.5, N)


def mhalo_dist(lmhalo, z):    #halo mass distribution from Tinker et al. 2008. Not normalized.
    # I believe this function returns dN/dlogNh. MUST DOUBLE-CHECK!!!
    lmref = 10. - np.log10(0.7)
    logsigminus = -0.64 + 0.11*(lmhalo - lmref) + 0.012*(lmhalo - lmref)**2 + 0.0006*(lmhalo - lmref)**3
    dlogsigminusdlogm = 0.11 + 2.*0.012*(lmhalo - lmref) + 3.*0.0006*(lmhalo - lmref)**2
    sigma = 10.**-logsigminus

    A = 0.186*(1.+z)**(-0.14)
    a = 1.47*(1.+z)**(-0.06)
    Delta = 200.
    alpha = 10.**(-(0.75/(np.log10(Delta/75.)))**1.2)
    b = 2.57*(1.+z)**alpha
    c = 1.19

    return ((sigma/b)**-a + 1)*np.exp(-c/sigma**2)/10.**lmhalo*dlogsigminusdlogm


def generate_halos(N, lmhmin=12., lmhmax=14., z=0.):
    """
    generates a random sample of halo masses from Tinker et al. 2008 distribution
    :param N: int. Sample size.
    :param lmhmin: minimum halo mass (log10)
    :param lmhmax: maximum halo mass (log10)
    :param z: redshift
    :return: numpy.ndarray of halo masses
    """
    lmhs = np.linspace(lmhmin, lmhmax, 101)
    dist_spline = splrep(lmhs, mhalo_dist(lmhs, z)) ### C: splrep(...) - Find the B-spline representation of 1-D curve.

    def intfunc(lmh):
        return splint(lmhmin, lmh, dist_spline) ### C: splint(...) - Evaluate the definite integral of a B-spline between two given points.

    norm = intfunc(lmhmax)

    x = np.random.rand(N)*norm
    lmhalos = 0.*x

    for i in range(0, N):
        lmhalos[i] = brentq(lambda lmh: intfunc(lmh) - x[i], lmhmin, lmhmax) ### C: brentq(...) - Find a root of a function in a bracketing interval using Brent's method.


    return lmhalos


def generate_mstar(lmhalos, z=0., scat=0.1):
    return mstarfunc(lmhalos, z) + np.random.normal(0., scat, len(np.atleast_1d(lmhalos))) ### C: np.atleast_1d(***) - Convert inputs to arrays with at least one dimension.


def mhfunc(lmstar, z):
    """
    stellar-to-halo mass relation from Leauthaud et al. 2012.
    :param lmstar:
    :param z:
    :return:
    """

    h_l12 = 0.72

    mstar00 = 10.7871 + np.log10(h_l12/h)
    mstar0z = 0.3623
    m10 = 12.4131 + np.log10(h_l12/h)
    m1z = 0.3785
    beta0 = 0.4477
    betaz = 0.02564
    delta0 = 0.56
    deltaz = 0.
    gamma0 = 0.8202
    gammaz = 1.8617

    logm1 = m10 + m1z*z
    logms0 = mstar00 + mstar0z*z
    beta = beta0 + betaz*z
    delta = delta0 + deltaz*z
    gamma = gamma0 + gammaz*z

    return logm1 + beta*(lmstar-logms0) + (10.**(lmstar-logms0))**delta/(1.+(10.**(lmstar-logms0))**(-gamma)) - 0.5

def b10_mhfunc(lmstar, z):
    """
    stellar-to-halo mass relation from Behroozi et al. 2010.
    :param lmstar:
    :param z:
    :return:
    """

    h_b10 = 0.72

    mstar00 = 10.7871 + np.log10(h_l12/h)
    mstar0z = 0.3623
    m10 = 12.4131 + np.log10(h_l12/h)
    m1z = 0.3785
    beta0 = 0.4477
    betaz = 0.02564
    delta0 = 0.56
    deltaz = 0.
    gamma0 = 0.8202
    gammaz = 1.8617

    logm1 = m10 + m1z*z
    logms0 = mstar00 + mstar0z*z
    beta = beta0 + betaz*z
    delta = delta0 + deltaz*z
    gamma = gamma0 + gammaz*z

    return logm1 + beta*(lmstar-logms0) + (10.**(lmstar-logms0))**delta/(1.+(10.**(lmstar-logms0))**(-gamma)) - 0.5


def lmhalo_given_lmstar(lmstar, z):

    scat_mstar = 0.20

    lmhalo_grid = mhfunc(lmstar_grid, z)
    lmstar_spline = splrep(lmhalo_grid, lmstar_grid)

    def intfunc(lmhalo):
        lmstar_mu = splev(lmhalo, lmstar_spline)
        p_mstar = 1./(2.*np.pi)/scat_mstar*np.exp(-0.5*(lmstar - lmstar_mu)**2/scat_mstar**2)
        p_mhalo = mhalo_dist(lmhalo, z)
        return p_mstar*p_mhalo

    num = quad(lambda lmhalo: intfunc(lmhalo)*lmhalo, lmhalo_grid[0], lmhalo_grid[-1])[0]
    den = quad(lambda lmhalo: intfunc(lmhalo), lmhalo_grid[0], lmhalo_grid[-1])[0]

    return num/den


def mstarfunc(lmhalo, z=0.):
    """
    inverse function of mhfunc
    :param lmhalo:
    :param z:
    :return:
    """
    lmhalos = mhfunc(lmstar_grid, z)
    lmstar_spline = splrep(lmhalos, lmstar_grid) ### C: splerp(...) - Find the B-spline representation of 1-D curve.
    return splev(lmhalo, lmstar_spline) ### C: splerv(...) - Evaluate a B-spline or its derivatives.
    

def rstarh(lmhalo, z):

    lmhalos = mhfunc(lmstar_grid, z)

    lmh_spline = splrep(lmhalos, lmstar_grid)

    return 10.**(splev(lmhalo, lmh_spline) - lmhalo)

