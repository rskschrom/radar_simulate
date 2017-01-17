import numpy as np
from scipy.special import gamma

# module of common utility functions

# angular moments (2d gaussian)
def angularGaussian(width_deg):
    width_rad = width_deg*np.pi/180.
    r = np.exp(-2.*width_rad**2.)
    a1 = (1./4.)*(1.+r)**2.
    a2 = (1./4.)*(1.-r**2.)
    a3 = (3./8.+1./2.*r+1./8*r**4.)**2.
    a4 = (3./8.-1./2.*r+1./8*r**4.)*(3./8.+1./2.*r+1./8*r**4.)
    a5 = 1./8.*(3./8.+1./2.*r+1./8*r**4.)*(1.-r**4.)
    a6 = 0.
    a7 = 1./2*r*(1.+r)
    moment_arr = np.array([a1, a2, a3, a4, a5, a6, a7])
    return moment_arr

# present in radar met units
def logify(parameter):
    lparameter = 10.*np.log10(parameter)
    return lparameter

# create exponential distribution
def exponentialDist(n0, lam, max_dim):
    numbins = len(max_dim)-1
    dmax_dim = max_dim[1:numbins+1]-max_dim[0:numbins]
    n = n0*np.exp(-lam*max_dim)
    return n, dmax_dim

# create gamma distribution
def gammaDist(nu, ni, an, max_dim):
    numbins = len(max_dim)-1
    dmax_dim = max_dim[1:numbins+1]-max_dim[0:numbins]
    lam = 1./(2.*an)
    n = 2.*ni/gamma(nu)*lam**nu*max_dim**(nu-1.)*np.exp(-lam*max_dim)
    return n, dmax_dim

def maxwellMixing(rho_snow, eps_ice):
# estimate refractive index (Maxwell-Garnett)
    rho_ice = 920. 
    eps_fac = (eps_ice-1.)/(eps_ice+2.)
    eps_snow = (1.+2.*rho_snow/rho_ice*eps_fac)/(1.-rho_snow/rho_ice*eps_fac)
    return eps_snow
