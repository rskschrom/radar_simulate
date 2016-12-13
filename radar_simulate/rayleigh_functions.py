'''
Calculate the scattering amplitudes for particles that satisfy the Rayleigh approximation
using formulas from Bohren and Huffman (1983)
'''
import numpy as np

# homogeneous spheroids
def scatSpheroidArr(diel, thickness,
                    maxDim, wavelength):
    numcalc = len(thickness)
    c = thickness/2.
    a = maxDim/2.    
    rad = (c*a**2.)**(1./3.)

    #Rayleigh formulas
    alp = c/a

    f = np.empty([numcalc])
    lv = np.empty([numcalc])
    lh = np.empty([numcalc])

    #oblate spheroids
    f[alp<1.] = np.sqrt(1.0/alp[alp<1.]**2-1.0)
    lv[alp<1.] = (1.0+f[alp<1.]**2)/(f[alp<1.]**2)*(1.0-np.arctan(f[alp<1.])/f[alp<1.])

    #prolate spheroids
    f[alp>1.] = np.sqrt(1.0-1.0/alp[alp>1.]**2)
    lv[alp>1.] = (1.0-f[alp>1.]**2)/(f[alp>1.]**2)*(1.0/(2.0*f[alp>1.])*np.log((1.0+f[alp>1.])/(1.0-f[alp>1.]))-1.0)

    # spheres
    lv[alp==1.] = 1./3.

    lh = (1.0-lv)/2.0

    # scattering amplitudes
    com_fact = np.pi**2.*(2.*rad)**3./(6.*wavelength**2.)
    shh = com_fact*1./(lh+1./(diel-1.))
    svv = com_fact*1./(lv+1./(diel-1.))

    return shh, svv

# core-shell spheroids
def scatTwoLayerSpheroidArr(dielCore, dielShell,
                            thickness, maxDim,
                            wavelength, coreDepthFrac):
    numcalc = len(thickness)
    c = thickness/2.
    a = maxDim/2.    
    rad = (c*a**2.)**(1./3.)

    #Rayleigh formulas
    alp = c/a

    f = np.empty([numcalc])
    lv = np.empty([numcalc])
    lh = np.empty([numcalc])

    #oblate spheroids
    f[alp<1.] = np.sqrt(1.0/alp[alp<1.]**2-1.0)
    lv[alp<1.] = (1.0+f[alp<1.]**2)/(f[alp<1.]**2)*(1.0-np.arctan(f[alp<1.])/f[alp<1.])

    #prolate spheroids
    f[alp>1.] = np.sqrt(1.0-1.0/alp[alp>1.]**2)
    lv[alp>1.] = (1.0-f[alp>1.]**2)/(f[alp>1.]**2)*(1.0/(2.0*f[alp>1.])*np.log((1.0+f[alp>1.])/(1.0-f[alp>1.]))-1.0)

    # spheres
    lv[alp==1.] = 1./3.

    lh = (1.0-lv)/2.0

    # two layer part
    volf = (1.-coreDepthFrac)*(1.-coreDepthFrac*c/a)**2.
    vol_part = np.pi**2*(2.0*rad)**3/(6*wavelength**2)

    e1 = dielCore
    e2 = dielShell

    numer_h = (e2-1.)*(e2+(e1-e2)*lh*(1.-volf))+volf*e2*(e1-e2)
    denom_h = (e2+(e1-e2)*lh*(1.-volf))*(1.+(e2-1.)*lh)+volf*lh*e2*(e1-e2)

    numer_v = (e2-1.)*(e2+(e1-e2)*lv*(1.-volf))+volf*e2*(e1-e2)
    denom_v = (e2+(e1-e2)*lv*(1.-volf))*(1.+(e2-1.)*lv)+volf*lv*e2*(e1-e2)

    shh = vol_part*numer_h/denom_h
    svv = vol_part*numer_v/denom_v

    return shh, svv
