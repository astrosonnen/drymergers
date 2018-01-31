import numpy as np
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq
from scipy.integrate import quad

h = 0.7
N = 1001

lmstar_grid = np.linspace(9., 12.5, N)

logmhalo=[]
logmstarlogmhalo=[]

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

def generate_mstar(lmhalos, z=0., scat=0.1, model="B10"):
    return mstarfunc(lmhalos, z, model) + np.random.normal(0., scat, len(np.atleast_1d(lmhalos))) ### C: np.atleast_1d(***) - Convert inputs to arrays with at least one dimension.

def mhfunc(lmstar, z, model):
    """
    stellar-to-halo mass relation
    :param lmstar:
    :param z:
    :model: stellar-to-halo mass relation from different authors - B10 (Behroozi et al. 2010), L12 (Leauthaud et al. 2012), B13 (Behroozi et al. 2013)
    :return:
    """
    if (model=="B10"):

        scale_factor = 1/(1+z) 

# # # # # # # # Weighted Average of values between 0.8 and 1.0 # # # # # # # #
# =============================================================================
#         if (z>=0.8 and z<=1.0):
#             mstar00_comm = (np.abs(z-0.8)* 11.09 +np.abs(z-1)*10.72)/(np.abs(z-0.8)+np.abs(z-1))
#             mstar0a_comm = (np.abs(z-0.8)*  0.56 +np.abs(z-1)* 0.55)/(np.abs(z-0.8)+np.abs(z-1))
#             m10_comm     = (np.abs(z-0.8)* 12.27 +np.abs(z-1)*12.35)/(np.abs(z-0.8)+np.abs(z-1))
#             m1a_comm     = (np.abs(z-0.8)*(-0.84)+np.abs(z-1)* 0.28)/(np.abs(z-0.8)+np.abs(z-1))
#             beta0_comm   = (np.abs(z-0.8)*  0.65 +np.abs(z-1)* 0.44)/(np.abs(z-0.8)+np.abs(z-1))
#             betaa_comm   = (np.abs(z-0.8)*  0.31 +np.abs(z-1)* 0.18)/(np.abs(z-0.8)+np.abs(z-1))
#             delta0_comm  = (np.abs(z-0.8)*  0.56 +np.abs(z-1)* 0.57)/(np.abs(z-0.8)+np.abs(z-1))
#             deltaa_comm  = (np.abs(z-0.8)*(-0.12)+np.abs(z-1)* 0.17)/(np.abs(z-0.8)+np.abs(z-1))
#             gamma0_comm  = (np.abs(z-0.8)*  1.12 +np.abs(z-1)* 1.56)/(np.abs(z-0.8)+np.abs(z-1))
#             gammaa_comm  = (np.abs(z-0.8)*(-0.53)+np.abs(z-1)* 2.51)/(np.abs(z-0.8)+np.abs(z-1))
#     
#         mstar00 = 10.72 if z>=0. and z<0.8 else 11.09 if z>1. and z<4. else mstar00_comm if z>=0.8 and z<=1 else np.nan
#         mstar0a =  0.55 if z>=0. and z<0.8 else  0.56 if z>1. and z<4. else mstar0a_comm if z>=0.8 and z<=1 else np.nan
#         m10     = 12.35 if z>=0. and z<0.8 else 12.27 if z>1. and z<4. else     m10_comm if z>=0.8 and z<=1 else np.nan
#         m1a     =  0.28 if z>=0. and z<0.8 else -0.84 if z>1. and z<4. else     m1a_comm if z>=0.8 and z<=1 else np.nan
#         beta0   =  0.44 if z>=0. and z<0.8 else  0.65 if z>1. and z<4. else   beta0_comm if z>=0.8 and z<=1 else np.nan
#         betaa   =  0.18 if z>=0. and z<0.8 else  0.31 if z>1. and z<4. else   betaa_comm if z>=0.8 and z<=1 else np.nan
#         delta0  =  0.57 if z>=0. and z<0.8 else  0.56 if z>1. and z<4. else  delta0_comm if z>=0.8 and z<=1 else np.nan
#         deltaa  =  0.17 if z>=0. and z<0.8 else -0.12 if z>1. and z<4. else  delta0_comm if z>=0.8 and z<=1 else np.nan
#         gamma0  =  1.56 if z>=0. and z<0.8 else  1.12 if z>1. and z<4. else  delta0_comm if z>=0.8 and z<=1 else np.nan
#         gammaa  =  2.51 if z>=0. and z<0.8 else -0.53 if z>1. and z<4. else  delta0_comm if z>=0.8 and z<=1 else np.nan
#                 
#         logm1 = m10 + m1a*(scale_factor-1)
#         logms0 = mstar00 + mstar0a*(scale_factor-1)
#         beta = beta0 + betaa*(scale_factor-1)
#         delta = delta0 + deltaa*(scale_factor-1)
#         gamma = gamma0 + gammaa*(scale_factor-1)
#     
#         logmhalo.append(logm1 + beta*(lmstar-logms0) + (10.**(lmstar-logms0))**delta/(1.+(10.**(lmstar-logms0))**(-gamma)) - 0.5)
# 
#         logmstarlogmhalo.append(lmstar-logmhalo)
#         
#         return logm1 + beta*(lmstar-logms0) + (10.**(lmstar-logms0))**delta/(1.+(10.**(lmstar-logms0))**(-gamma)) - 0.5
# =============================================================================

        mstar00 = 10.72 if z>=0. and z<=1.0 else np.nan
        mstar0a =  0.55 if z>=0. and z<=1.0 else np.nan
        m10     = 12.35 if z>=0. and z<=1.0 else np.nan
        m1a     =  0.28 if z>=0. and z<=1.0 else np.nan
        beta0   =  0.44 if z>=0. and z<=1.0 else np.nan
        betaa   =  0.18 if z>=0. and z<=1.0 else np.nan
        delta0  =  0.57 if z>=0. and z<=1.0 else np.nan
        deltaa  =  0.17 if z>=0. and z<=1.0 else np.nan
        gamma0  =  1.56 if z>=0. and z<=1.0 else np.nan
        gammaa  =  2.51 if z>=0. and z<=1.0 else np.nan

        logm1 = m10 + m1a*(scale_factor-1)
        logms0 = mstar00 + mstar0a*(scale_factor-1)
        beta = beta0 + betaa*(scale_factor-1)
        delta = delta0 + deltaa*(scale_factor-1)
        gamma = gamma0 + gammaa*(scale_factor-1)

        logmhalo.append(logm1 + beta*(lmstar-logms0) + (10.**(lmstar-logms0))**delta/(1.+(10.**(lmstar-logms0))**(-gamma)) - 0.5)

        logmstarlogmhalo.append(lmstar-logmhalo)
 
        return logm1 + beta*(lmstar-logms0) + (10.**(lmstar-logms0))**delta/(1.+(10.**(lmstar-logms0))**(-gamma)) - 0.5

    elif (model=="L12_sig_mod1" or model=="L12_sig_mod2"):

        scale_factor = 1/(1+z) 

        if (model=="L12_sig_mod1"):
            logm1  = 12.520 if z>=0.22 and z<0.48 else 12.725 if z>=0.48 and z<0.74 else 12.722 if z>=0.74 and z<=1 else np.nan
            logms0 = 10.916 if z>=0.22 and z<0.48 else 11.038 if z>=0.48 and z<0.74 else 11.100 if z>=0.74 and z<=1 else np.nan
            beta   =  0.457 if z>=0.22 and z<0.48 else  0.466 if z>=0.48 and z<0.74 else  0.470 if z>=0.74 and z<=1 else np.nan
            delta  =  0.566 if z>=0.22 and z<0.48 else  0.610 if z>=0.48 and z<0.74 else  0.393 if z>=0.74 and z<=1 else np.nan
            gamma  =  1.530 if z>=0.22 and z<0.48 else  1.950 if z>=0.48 and z<0.74 else  2.510 if z>=0.74 and z<=1 else np.nan

        if (model=="L12_sig_mod2"):
            logm1  = 12.518 if z>=0.22 and z<0.48 else 12.724 if z>=0.48 and z<0.74 else 12.726 if z>=0.74 and z<=1 else np.nan
            logms0 = 10.917 if z>=0.22 and z<0.48 else 11.038 if z>=0.48 and z<0.74 else 11.100 if z>=0.74 and z<=1 else np.nan
            beta   =  0.456 if z>=0.22 and z<0.48 else  0.466 if z>=0.48 and z<0.74 else  0.470 if z>=0.74 and z<=1 else np.nan
            delta  =  0.582 if z>=0.22 and z<0.48 else  0.620 if z>=0.48 and z<0.74 else  0.470 if z>=0.74 and z<=1 else np.nan
            gamma  =  1.480 if z>=0.22 and z<0.48 else  1.930 if z>=0.48 and z<0.74 else  2.380 if z>=0.74 and z<=1 else np.nan

        logmhalo.append(logm1 + beta*(lmstar-logms0) + (10.**(lmstar-logms0))**delta/(1.+(10.**(lmstar-logms0))**(-gamma)) - 0.5)

        logmstarlogmhalo.append(lmstar-logmhalo)

        return logm1 + beta*(lmstar-logms0) + (10.**(lmstar-logms0))**delta/(1.+(10.**(lmstar-logms0))**(-gamma)) - 0.5

    elif (model=="SNT17_original" or model=="SNT17_corrected"):
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

        if (model=="SNT17_corrected"):
            z = -z/(1.+z)

        logm1 = m10 + m1z*z
        logms0 = mstar00 + mstar0z*z
        beta = beta0 + betaz*z
        delta = delta0 + deltaz*z
        gamma = gamma0 + gammaz*z

        logmhalo.append(logm1 + beta*(lmstar-logms0) + (10.**(lmstar-logms0))**delta/(1.+(10.**(lmstar-logms0))**(-gamma)) - 0.5)

        logmstarlogmhalo.append(lmstar-logmhalo)

        return logm1 + beta*(lmstar-logms0) + (10.**(lmstar-logms0))**delta/(1.+(10.**(lmstar-logms0))**(-gamma)) - 0.5

    else:
        print("Bye bye!")

# =============================================================================
# =============================================================================
# # #def mhfunc(lmstar, z):
# # #    """
# # #    stellar-to-halo mass relation from Leauthaud et al. 2012.
# # #    :param lmstar:
# # #    :param z:
# # #    :return:
# # #    """
# # #
# # #    h_l12 = 0.72
# # #
# # #    mstar00 = 10.7871 + np.log10(h_l12/h)
# # #    mstar0z = 0.3623
# # #    m10 = 12.4131 + np.log10(h_l12/h)
# # #    m1z = 0.3785
# # #    beta0 = 0.4477
# # #    betaz = 0.02564
# # #    delta0 = 0.56
# # #    deltaz = 0.
# # #    gamma0 = 0.8202
# # #    gammaz = 1.8617
# # #
# # #    logm1 = m10 + m1z*z
# # #    logms0 = mstar00 + mstar0z*z
# # #    beta = beta0 + betaz*z
# # #    delta = delta0 + deltaz*z
# # #    gamma = gamma0 + gammaz*z
# # #
# # #    return logm1 + beta*(lmstar-logms0) + (10.**(lmstar-logms0))**delta/(1.+(10.**(lmstar-logms0))**(-gamma)) - 0.5
# # #
# # #def b10_mhfunc(lmstar, z):
# # #    """
# # #    stellar-to-halo mass relation from Behroozi et al. 2010.
# # #    :param lmstar:
# # #    :param z:
# # #    :return:
# # #    """
# # #
# # #    h_b10 = 0.72
# # #
# # #    mstar00 = 10.7871 + np.log10(h_l12/h)
# # #    mstar0z = 0.3623
# # #    m10 = 12.4131 + np.log10(h_l12/h)
# # #    m1z = 0.3785
# # #    beta0 = 0.4477
# # #    betaz = 0.02564
# # #    delta0 = 0.56
# # #    deltaz = 0.
# # #    gamma0 = 0.8202
# # #    gammaz = 1.8617
# # #
# # #    logm1 = m10 + m1z*z
# # #    logms0 = mstar00 + mstar0z*z
# # #    beta = beta0 + betaz*z
# # #    delta = delta0 + deltaz*z
# # #    gamma = gamma0 + gammaz*z
# # #
# # #    return logm1 + beta*(lmstar-logms0) + (10.**(lmstar-logms0))**delta/(1.+(10.**(lmstar-logms0))**(-gamma)) - 0.5
# =============================================================================
# =============================================================================

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

def mstarfunc(lmhalo, z=0., model="B10"):
    """
    inverse function of mhfunc
    :param lmhalo:
    :param z:
    :model: stellar-to-halo mass relation from different authors - B10 (Behroozi et al. 2010), L12 (Leauthaud et al. 2012), B13 (Behroozi et al. 2013)
    :return:
    """
    if (model=="B10" or model=="L12_sig_mod1" or model=="L12_sig_mod2" or model=="SNT17_original" or model=="SNT17_corrected"):

        lmhalos = mhfunc(lmstar_grid, z, model)
        lmstar_spline = splrep(lmhalos, lmstar_grid) ### C: splerp(...) - Find the B-spline representation of 1-D curve.
        return splev(lmhalo, lmstar_spline) ### C: splev(...) - Evaluate a B-spline or its derivatives.

    elif(model=="B13"):

        a_scale_factor = 1./(1.+z)

        # *** Intrinsic parameters from Behroozi et al. 2013 *** #

        M_10        = 11.514
        M_1a        = -1.793
        M_1z        = -0.251
        epsilon_0   = -1.777
        epsilon_a   = -0.006
        epsilon_z   = -0.000
        epsilon_a2  = -0.119
        alpha_0     = -1.412
        alpha_a     =  0.731
        delta_0     =  3.508
        delta_a     =  2.608
        delta_z     = -0.043
        gamma_0     =  0.316
        gamma_a     =  1.319
        gamma_z     =  0.279

        ni          = np.exp(-4*a_scale_factor**2)
        log_M_1     = M_10 + (M_1a*(a_scale_factor-1) + M_1z*z)*ni
        log_epsilon = epsilon_0 + (epsilon_a*(a_scale_factor-1) + epsilon_z*z)*ni + epsilon_a2*(a_scale_factor-1)
        alpha       = alpha_0 + (alpha_a*(a_scale_factor-1))*ni
        delta       = delta_0 + (delta_a*(a_scale_factor-1) + delta_z*z)*ni
        gamma       = gamma_0 + (gamma_a*(a_scale_factor-1) + gamma_z*z)*ni

        func_l_Mh_M1 = lambda x: -np.log10(10**(alpha*x) + 1) + delta*(((np.log10(1 + np.exp(x)))**gamma)/(1 + np.exp(10**(-x))))

        return (log_epsilon + log_M_1) + func_l_Mh_M1(lmhalo-log_M_1) - func_l_Mh_M1(0)

def rstarh(lmhalo, z):

    lmhalos = mhfunc(lmstar_grid, z)

    lmh_spline = splrep(lmhalos, lmstar_grid)

    return 10.**(splev(lmhalo, lmh_spline) - lmhalo)