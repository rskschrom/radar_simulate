import numpy as np
from util import logify

# function to integrate parameter over distribution
def integrate(param, n, dmax_dim):
    numbins = len(n)
    avg_n = (n[1:numbins]+n[0:numbins-1])/2.
    avg_param = (param[1:numbins]+param[0:numbins-1])/2.
    int_param = np.sum(avg_n*dmax_dim*avg_param)
    return int_param

# zh function
def calculateZh(shh, svv, wavl, n, dmax_dim, ang_moments):
    # set parameters
    a2 = ang_moments[1]
    a4 = ang_moments[3]
    zh_param = np.abs(shh)**2.+\
               -2.*np.real(np.conj(shh)*(shh-svv))*a2+\
               np.abs(shh-svv)**2.*a4
    zh = integrate(zh_param, n, dmax_dim)*4.*wavl**4./(np.pi**4.*kw2)
    return zh

# zv function
def calculateZv(shh, svv, wavl, n, dmax_dim, ang_moments):
    # set parameters
    a1 = ang_moments[0]
    a3 = ang_moments[2]
    zv_param = np.abs(shh)**2.+\
               -2.*np.real(np.conj(shh)*(shh-svv))*a1+\
               np.abs(shh-svv)**2.*a3
    zv = integrate(zv_param, n, dmax_dim)*4.*wavl**4./(np.pi**4.*kw2)
    return zv

# zdr function
def calculateZdr(shh, svv, wavl, n, dmax_dim, ang_moments):
    zh = calculateZh(shh, svv, wavl, n, dmax_dim, ang_moments)
    zv = calculateZv(shh, svv, wavl, n, dmax_dim, ang_moments)
    zdr = zh/zv
    return zdr

# kdp function
def calculateKdp(shh, svv, wavl, n, dmax_dim, ang_moments):
    # note: shh and svv should be forward amplitudes
    a7 = ang_moments[6]
    kdp_param = np.real(shh-svv)*a7
    kdp = integrate(kdp_param, n, dmax_dim)*0.18*wavl/np.pi
    return kdp

# rhohv function
def calculateRhoHV(shh, svv, wavl, n, dmax_dim, ang_moments):
    a1 = ang_moments[0]
    a2 = ang_moments[1]
    a5 = ang_moments[4]
    zh = calculateZh(shh, svv, wavl, n, dmax_dim, ang_moments)
    zv = calculateZv(shh, svv, wavl, n, dmax_dim, ang_moments)
    rhohv_param = np.abs(shh)**2.+np.abs(shh-svv)**2.*a5-np.conj(shh)*(shh-svv)*a1+\
                  -shh*(np.conj(shh)-np.conj(svv))*a2
    rhohv = np.abs(integrate(rhohv_param, n, dmax_dim))*4.*wavl**4./(np.pi**4.*kw2*np.sqrt(zh*zv))
    return rhohv

# ldr function
def calculateLdr(shh, svv, wavl, n, dmax_dim, ang_moments):
    a5 = ang_moments[4]
    zh = calculateZh(shh, svv, wavl, n, dmax_dim, ang_moments)
    ldr_param = np.abs(shh-svv)**2.*a5
    ldr = integrate(ldr_param, n, dmax_dim)*4.*wavl**4./(np.pi**4.*kw2*zh)
    return ldr

# function to return everthing for ease of use
def calculateEverything(shh, svv, wavl, n, dmax_dim, ang_moments):
    zh = calculateZh(shh, svv, wavl, n, dmax_dim, ang_moments)
    zv = calculateZv(shh, svv, wavl, n, dmax_dim, ang_moments)
    zdr = calculateZdr(shh, svv, wavl, n, dmax_dim, ang_moments)
    kdp = calculateKdp(shh, svv, wavl, n, dmax_dim, ang_moments)
    rhohv = calculateRhoHV(shh, svv, wavl, n, dmax_dim, ang_moments)
    ldr = calculateLdr(shh, svv, wavl, n, dmax_dim, ang_moments)
    return zh, zv, zdr, kdp, rhohv, ldr

# function to return everthing for ease of use
def calculateEverythingLog(shh, svv, wavl, n, dmax_dim, ang_moments):
    zh, zv, zdr, kdp, rhohv, ldr = calculateEverything(shh, svv, wavl, n, dmax_dim, ang_moments)
    zH = logify(zh)
    zV = logify(zv)
    zDR = logify(zdr)
    lDR = logify(ldr)
    return zH, zV, zDR, kdp, rhohv, lDR

# function to return everthing for ease of use (use for bin output)
def calculateEverythingBin(shh, svv, wavl, n, ang_moments):
    zh, zv, zdr, kdp, rhohv, ldr = calculateEverything(shh, svv, wavl, n, dmax_dim, ang_moments)
    zH = logify(zh)
    zV = logify(zv)
    zDR = logify(zdr)
    lDR = logify(ldr)
    return zH, zV, zDR, kdp, rhohv, lDR
