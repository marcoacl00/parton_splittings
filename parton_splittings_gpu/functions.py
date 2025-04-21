import numpy as np
from scipy.integrate import quad


def fasit2Ncdiag(t,L,p1,p2,z, omega, Omega, qF):
    pre= -2*1j*omega
    
    num = 2*omega*Omega/np.tan(Omega*t)
    den = 2*omega*Omega/np.tan(Omega*t)+1j*qF*(z**2+(1-z)**2)*(L-t)
    
    return pre*(1-num/den*np.exp(-1j*(p1**2+p2**2)/den))


def fasit2Ncdiagint(L,p1,p2,z, omega, Omega, qF):
    def real_fas(t,L,p1,p2,z):
        return np.real(fasit2Ncdiag(t,L,p1,p2,z, omega, Omega, qF))
    def imag_fas(t,L,p1,p2,z):
        return np.imag(fasit2Ncdiag(t,L,p1,p2,z, omega, Omega, qF))
    re = quad(real_fas,0,L,args=(L,p1,p2,z))[0]
    im = quad(imag_fas,0,L,args=(L,p1,p2,z))[0]
    
    return re + 1j*im

