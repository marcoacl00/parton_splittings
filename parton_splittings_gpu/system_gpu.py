import numpy as np #for cpu
import cupy as cp #for gpu

prec = np.float32
prec_c = np.complex64

class phsys:
    """This class contains all parameters and methods 
    exclusive to the chosen configuration 
    (E[GeV], z, qF[GeV^2/fm], L[grid_size fm])"""
    
    def __init__(self, E, z, qF, L, Ncmode = "LNcFac", vertex = "gamma_qq", parallel = "gpu"):
        fm = prec(5.067)
        self.fm = fm
        self.E = prec(E * fm)
        self.qhat = prec(qF * fm**2)
        self.z = z
        self.omega = prec(self.E * z * (1 - z))
        self.Omega = (1 - 1j)/2 * prec(np.sqrt(self.qhat / self.omega))
        self.Ncmode = Ncmode
        self.vertex = vertex
        self.L = prec(L)
        self.t = 0.005 #initial time
        self.Nu1 = None
        self.Nu2 = None
        self.Nv1 = None
        self.Nv2 = None
        self.U1 = None
        self.U2 = None
        self.V1 = None
        self.V2 = None
        self.du1 = None
        self.du2 = None
        self.dv1 = None
        self.dv2 = None
        self.Fsol = None
        self.parallel = parallel

        if vertex == "gamma_qq":
            self.Nsig = 2
        else:
            raise ValueError("Other vertices not yet available")
        

    def set_dim(self, Nu1: int, Nu2: int, Nv1: int, Nv2: int):
        """Set the dimensions of the system for each axis u1, u2, v1 and v2"""
        self.Nu1, self.Nu2, self.Nv1, self.Nv2 = Nu1, Nu2, Nv1, Nv2

        if self.parallel == "gpu":
            #keep grid on GPU already
            self.U1 = cp.linspace(-self.L/2, self.L/2, Nu1, dtype=prec)
            self.U2 = cp.linspace(-self.L/2, self.L/2, Nu2, dtype=prec)
            self.V1 = cp.linspace(-self.L/2, self.L/2, Nv1, dtype=prec)
            self.V2 = cp.linspace(-self.L/2, self.L/2, Nv2, dtype=prec)

        else:

            self.U1 = np.linspace(-self.L/2, self.L/2, Nu1, dtype=prec)
            self.U2 = np.linspace(-self.L/2, self.L/2, Nu2, dtype=prec)
            self.V1 = np.linspace(-self.L/2, self.L/2, Nv1, dtype=prec)
            self.V2 = np.linspace(-self.L/2, self.L/2, Nv2, dtype=prec)

        self.du1 = self.U1[1] - self.U1[0]
        self.du2 = self.U2[1] - self.U2[0]
        self.dv1 = self.V1[1] - self.V1[0]
        self.dv2 = self.V2[1] - self.V2[0]

    def print_dim(self):
        """Print the dimensions of the system for all axis u1, u2, v1 and v2"""
        print(self.Nu1, self.Nu2, self.Nv1, self.Nv2)

    def beta(self, t):
        """The frequency of spatial oscilations"""
        co = self.omega * self.Omega / 2 * 1 / np.tan(self.Omega * t) #already precision complex64
        return np.real(co)
    
    
    def eps(self, t):
        """Damp factor on the source term"""
        co = self.omega * self.Omega / 2 * 1 / np.tan(self.Omega * t)
        return np.imag(co) 

    def dbeta(self, t):
        """Derivative of the frequency of spatial oscilations"""
        det = 1e-12
        return (self.beta(t+det) - self.beta(t))/det 
    

    def init_fsol(self):
        """Initialize the grid for the solution. The dimensions of the system
        must be provided beforehand"""
        if self.Nu1 == None:
            raise ValueError("No value for the dimensions provided")
        else:
            if self.vertex == "gamma_qq":
                if self.parallel == "gpu":
                    self.Fsol = cp.zeros(shape=(2, self.Nu1, self.Nu2, self.Nv1, self.Nv2), dtype=prec_c)
                else:
                    self.Fsol = np.zeros(shape=(2, self.Nu1, self.Nu2, self.Nv1, self.Nv2), dtype=prec_c)
            else: 
                raise TypeError("Other vertices not yet available")
            
    def F_in_out(self, theta):
        Finout = np.real(-2 * (1 - np.exp(-1j * np.tan(self.Omega * self.t) / 
                                          (2* self.omega * self.Omega)* 
                                          self.omega **2 * theta**2)))
        
        return Finout
    
    def set_fsol(self, arr):
        self.Fsol = arr

    def increase_t(self, t):
        self.t += t
    
    def set_t(self, t):
        self.t = t
            
    def return_fsol(self):
        """Returs the solution grid"""
        return self.Fsol        
    
    def dirac_v1(self, j1):
        retval = 0
        if j1 == self.Nv1//2 or j1 == self.Nv1//2 - 1:
            retval = 1/(2*self.dv1)

        return retval
        
    def dirac_v2(self, j2):
        retval = 0
        if j2 == self.Nv2//2 or j2 == self.Nv2//2 - 1:
            retval = 1/(2*self.dv2)
        
        return retval 
    
    def ddirac_v1(self, j1):
        retval = 0
        if j1 == self.Nv1//2-1:
            retval =  1/(self.dv1 ** 2)
        
        if j1 == self.Nv1//2:
            retval =  -1/(self.dv1 ** 2)
        
        return retval 
        
    def ddirac_v2(self, j2):
        retval = 0
        if j2 == self.Nv2//2-1:
            retval =  1/(self.dv2 ** 2)
        
        if j2 == self.Nv2//2:
            retval =  -1/(self.dv2 ** 2)
        
        return retval 
    
    def source_term(self, t, i1, i2, j1, j2):

        constant = 1j * self.omega / np.pi
        damp = np.exp(-self.eps(t) * (self.U1[i1]*self.U1[i1] + self.U2[i2]*self.U2[i2]))
        ux = self.U1[i1]
        uy = self.U2[i2]
        deltav1 = self.dirac_v1(j1)
        ddeltav1 = self.ddirac_v1(j1)
        deltav2 = self.dirac_v2(j1)
        ddeltav2 = self.ddirac_v2(j1)
 

        non_hom_term = constant * damp * (ux * ddeltav1 *  deltav2 + 
                                          uy * deltav1 * ddeltav2 ) / (ux**2 + uy**2)

        return non_hom_term
    
    def Kin(self, sig, i1, i2, j1, j2, dt):
        
        f = self.Fsol
        du1 = self.du1
        du2 = self.du2
        dv1 = self.dv1
        dv2 = self.dv2
        u1 = self.U1[i1]
        u2 = self.U2[i2]

        beta_t = self.beta(self.t)
        beta_t12 = self.beta(self.t + dt/2)
        beta_t1 = self.beta(self.t + dt)

        omega = self.omega

        #a bunch of array extractions to compute the derivative
        f_ = f[sig, i1, i2, j1, j2]
        f_uxplus1 = f[sig, i1+1, i2, j1, j2]
        f_uxminus1 = f[sig, i1-1, i2, j1, j2]
        f_uyplus1 = f[sig, i1, i2+1, j1, j2]
        f_uyminus1 = f[sig, i1, i2-1, j1, j2]
        f_vxplus1 = f[sig, i1, i2, j1+1, j2]
        f_vxminus1 = f[sig, i1, i2, j1-1, j2]
        f_vyplus1 = f[sig, i1, i2, j1, j2+1]
        f_vyminus1 = f[sig, i1, i2, j1, j2-1]

        #finite difference approx for derivatives
        deriv_ux = (f_uxplus1 - f_uxminus1)/(2*du1)
        deriv_uy = (f_uyplus1 - f_uyminus1)/(2*du2)

        deriv2_ux = (f_uxplus1 - 2 * f_ + f_uxminus1)/(du1**2)
        deriv2_uy = (f_uyplus1 - 2 * f_ + f_uyplus1)/(du2**2)

        deriv2_vx = (f_vxplus1 - 2 * f_ + f_vxminus1)/(dv1**2)
        deriv2_vy = (f_vyplus1 - 2 * f_ + f_vyminus1)/(dv2**2)

        deriv2_u = deriv2_ux + deriv2_uy
        deriv2_v = deriv2_vx + deriv2_vy

        dir_deriv = u1 * deriv_ux + u2 * deriv_uy

        beta_eff = (beta_t + 4*beta_t12 + beta_t1)/6

        return -1/(2 * omega) * (deriv2_u + 4j * beta_eff * dir_deriv - deriv2_v) 
    
    
    def V_LargeNc_gamma_qq(self, sig1, sig2, i1, i2, j1, j2):
        """The potential matrix for gamma -> qqbar in the large Nc factorized (diag) limit"""
        u1 = self.U1[i1]
        u2 = self.U2[i2]
        v1 = self.V1[j1]
        v2 = self.V2[j2]

        if sig1 == 0 and sig2 == 0:
        
            u_sqrd = u1**2 + u2**2
            v_sqrd =  v1**2 + v2**2

            element = u_sqrd + v_sqrd

        elif (sig1 == 0 and sig2 == 1) or (sig1 == 1 and sig2 == 0):

            element = 0
        
        elif sig1 == 1 and sig2 == 1:
            u1_minus_v1_sqrd = (u1 - v1)**2
            u2_minus_v2_sqrd = (u2 - v2)**2
            z = self.z
            g_z = z**2 + (1-z)**2

            element = g_z * (u1_minus_v1_sqrd + u2_minus_v2_sqrd)
            
        else:
            element = 0
            print("ERROR ON POTENTIAL")

    
        return -self.qhat / (4.0) * element 
    
    def V_FiniteNc_gamma_qq(self, sig1, sig2, i1, i2, j1, j2):
        """The Finite-Nc potential matrix for gamma -> qqbar"""
        Nc = 3
        CF = (Nc**2 - 1)/Nc
        u1 = self.U1[i1]
        u2 = self.U2[i2]
        v1 = self.V1[j1]
        v2 = self.V2[j2]

        if sig1 == 0 and sig2 == 0:
        
            u_sqrd = u1**2 + u2**2
            v_sqrd =  v1**2 + v2**2
            u_dot_v = u1*v1 + u2*v2
            prefac = 1/(CF * Nc)

            element = u_sqrd + v_sqrd + prefac * u_dot_v

        elif sig1 == 0 and sig2 == 1:

            u_dot_v = u1*v1 + u2*v2
            prefac = -1/(CF * Nc)

            element = prefac * u_dot_v
        
        elif sig1 == 1 and sig2 == 0:

            u1_minus_v1_sqrd = (u1 - v1)**2
            u2_minus_v2_sqrd = (u2 - v2)**2
            z = self.z
            g_z = z*(1-z)

            prefac = Nc/(CF)
            element = prefac * (u1_minus_v1_sqrd + u2_minus_v2_sqrd)
        
        elif sig1 == 1 and sig2 == 1:
            u1_minus_v1_sqrd = (u1 - v1)**2
            u2_minus_v2_sqrd = (u2 - v2)**2
            z = self.z
            prefac = CF - Nc*z*(1-z)
            element = g_z * (u1_minus_v1_sqrd + u2_minus_v2_sqrd)
            
        else:
            raise ValueError("Error on potential")

        return -self.qhat / (4.0) * element 
    

    




