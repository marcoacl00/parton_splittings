import numpy as np 
import cupy as cp #comment if not going to use GPU
from scipy import special


class phsys:
    """Simulation in position space.
    This class contains all parameters and methods 
    exclusive to the chosen configuration.
    Recommended only for very small momentum modes.
    (E[GeV], z, qF[GeV^2/fm], Lu[grid_size_u fm]),  Lv[grid_size_v fm]). \n
    Extra params: NcMode (LNcFac, LNc, FNc) 
    vertex (currently only "gamma_qq"),
    optimization : "default" (with np/simd), and "gpu" 
    prec: precision type (default float32, better for gpu)
    """
    
    def __init__(self, E, z, qF, Lu, Lv, Ncmode = "LNcFac", vertex = "gamma_qq", optimization = "gpu", prec = np.float32):

        #float precision and optimization type
        self.prec = prec
        self.prec_c = np.result_type(1j * prec(1))
        self.optimization = optimization

        fm = prec(5.067)
        self.fm = fm
        self.E = prec(E * fm)
        self.qhat = prec(qF * fm**2)
        self.z = z
        self.omega = prec(self.E * z * (1 - z))
        self.Omega = (1 - 1j)/2 * prec(np.sqrt(self.qhat / self.omega))
        self.Ncmode = Ncmode
        self.vertex = vertex
        self.Lu = prec(Lu)
        self.Lv = prec(Lv)

        self.t = 1e-4 #initial time
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
        

        if vertex == "gamma_qq":
            self.Nsig = 2
        else:
            raise ValueError("Other vertices not yet available")
        

    def set_dim(self, Nu1: int, Nu2: int, Nv1: int, Nv2: int):
        """Set the dimensions of the system for each axis u1, u2, v1 and v2"""
        self.Nu1, self.Nu2, self.Nv1, self.Nv2 = Nu1, Nu2, Nv1, Nv2

        if self.optimization == "gpu":
            #keep grid on GPU already
            self.U1 = cp.linspace(-self.Lu/2, self.Lu/2, Nu1, dtype=self.prec)
            self.U2 = cp.linspace(-self.Lu/2, self.Lu/2, Nu2, dtype=self.prec)
            self.V1 = cp.linspace(-self.Lv/2, self.Lv/2, Nv1, dtype=self.prec)
            self.V2 = cp.linspace(-self.Lv/2, self.Lv/2, Nv2, dtype=self.prec)

        else:

            self.U1 = np.linspace(-self.Lu/2, self.Lu/2, Nu1, dtype=self.prec)
            self.U2 = np.linspace(-self.Lu/2, self.Lu/2, Nu2, dtype=self.prec)
            self.V1 = np.linspace(-self.Lv/2, self.Lv/2, Nv1, dtype=self.prec)
            self.V2 = np.linspace(-self.Lv/2, self.Lv/2, Nv2, dtype=self.prec)

        self.du1 = self.U1[1] - self.U1[0]
        self.du2 = self.U2[1] - self.U2[0]
        self.dv1 = self.V1[1] - self.V1[0]
        self.dv2 = self.V2[1] - self.V2[0]

    def print_dim(self):
        """Print the dimensions of the system for all axis u1, u2, v1 and v2"""
        print(self.Nu1, self.Nu2, self.Nv1, self.Nv2)

    def beta(self, t):
        """The frequency of spatial oscilations"""
        co = self.omega * self.Omega * 0.5 * 1 / np.tan(self.Omega * t) #already precision complex64
        return np.real(co)
    
    
    def eps(self, t):
        """Damp factor on the source term"""
        co = self.omega * self.Omega * 0.5 * 1 / np.tan(self.Omega * t)
        return np.imag(co) 

    def dbeta(self, t):
        """Derivative of the frequency of spatial oscilations"""
        det = 1e-3
        return (self.beta(t+det) - self.beta(t-det))/(2*det) 
    

    def init_fsol(self):
        """Initialize the grid for the solution. The dimensions of the system
        must be provided beforehand"""
        if self.Nu1 == None:
            raise ValueError("No value for the dimensions provided")
        else:
            if self.vertex == "gamma_qq":
                if self.optimization == "gpu":
                    self.Fsol = cp.zeros(shape=(2, self.Nu1, self.Nu2, self.Nv1, self.Nv2), dtype=self.prec_c)
                else:
                    self.Fsol = np.zeros(shape=(2, self.Nu1, self.Nu2, self.Nv1, self.Nv2), dtype=self.prec_c)
            else: 
                raise TypeError("Other vertices not yet available")
            
    def F_in_out(self, theta):

        Finout = np.real(-2 * (1 - np.exp(-1j * np.tan(self.Omega * self.t) / 
                                          (2* self.omega * self.Omega)* 
                                          self.omega**2 * theta**2)))
        
        return Finout
    
    def set_fsol(self, arr):
        self.Fsol = arr

    def increase_t(self, dt):
        self.t += dt
    
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

    def source_term_array(self, t):

        constant = 1j * self.omega / np.pi
        ux = self.U1[:, None, None, None]
        uy = self.U2[None, :, None, None]

        damp = np.exp(-self.eps(t) * (ux**2 + uy**2))

        one_u2 = 1 / (ux**2 + uy**2)
        
        origin_v1 = self.Nv1//2
        origin_v2 = self.Nv2//2

        deltav1 = .0 * self.V1
        deltav2 = .0 * self.V2
        ddeltav1 = .0 * self.V1
        ddeltav2 = .0 * self.V2

        deltav1[origin_v1] =  self.dirac_v1(origin_v1)
        deltav1[origin_v1-1] =  self.dirac_v1(origin_v1-1)
        ddeltav1[origin_v1] =  self.ddirac_v1(origin_v1)
        ddeltav1[origin_v1-1] =  self.ddirac_v1(origin_v1-1)

        deltav2[origin_v2] =  self.dirac_v2(origin_v2)
        deltav2[origin_v2-1] =  self.dirac_v2(origin_v2-1)
        ddeltav2[origin_v2] =  self.ddirac_v2(origin_v2)
        ddeltav2[origin_v2-1] =  self.ddirac_v2(origin_v2-1)

        delta_term = (ux * ddeltav1[None, None, :, None] *  deltav2[None, None, None, :] + 
                      uy * ddeltav2[None, None, None, :] * deltav1[None, None, :, None]) 
 

        non_hom_term = constant * damp * delta_term * one_u2

        return non_hom_term



    def V_LargeNc_gamma_qq_par(self, sig1, sig2):
        """The potential matrix for gamma -> qqbar in the large Nc factorized (diag) limit"""
        u1 = self.U1[:, None, None, None]
        u2 = self.U2[None, :, None, None]
        v1 = self.V1[None, None, :, None]
        v2 = self.V2[None, None, None, :]

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
    
    def V_FiniteNc_gamma_qq_par(self, sig1, sig2):
        """The Finite-Nc potential matrix for gamma -> qqbar"""
        Nc = 3
        CF = (Nc**2 - 1)/Nc
        u1 = self.U1[:, None, None, None]
        u2 = self.U2[None, :, None, None]
        v1 = self.V1[None, None, :, None]
        v2 = self.V2[None, None, None, :]

        if sig1 == 0 and sig2 == 0:
            u_sqrd = u1**2 + u2**2
            v_sqrd = v1**2 + v2**2
            u_dot_v = u1 * v1 + u2 * v2
            prefac = 1 / (CF * Nc)
            element = u_sqrd + v_sqrd + prefac * u_dot_v

        elif sig1 == 0 and sig2 == 1:
            u_dot_v = u1 * v1 + u2 * v2
            prefac = -1 / (CF * Nc)
            element = prefac * u_dot_v

        elif sig1 == 1 and sig2 == 0:
            u1_minus_v1_sqrd = (u1 - v1) ** 2
            u2_minus_v2_sqrd = (u2 - v2) ** 2
            z = self.z
            g_z = z * (1 - z)
            element = g_z * (u1_minus_v1_sqrd + u2_minus_v2_sqrd)

        elif sig1 == 1 and sig2 == 1:
            u1_minus_v1_sqrd = (u1 - v1) ** 2
            u2_minus_v2_sqrd = (u2 - v2) ** 2
            z = self.z
            prefac = CF - Nc * z * (1 - z)
            element = prefac * (u1_minus_v1_sqrd + u2_minus_v2_sqrd)

        else:
            raise ValueError("Error on potential")

        return -0.25 * self.qhat * element
    

    
class physys_new:
    """Simulation in momentum space (k, l).
    This class contains all parameters and methods 
    exclusive to the chosen configuration.
    This is the recommended method.
    (E[GeV], z, qF[GeV^2/fm], Lp[grid_size_p fm]),  Lq[grid_size_q fm]). \n
    Extra params: NcMode (LNcFac, LNc, FNc), 
    vertex (currently only "gamma_qq"),
    optimization : "default" (with np/simd), and "gpu" 
    prec: precision type (default float32, better for gpu)
    """
    
    def __init__(self, E, z, qF, Lk, Ll, Ncmode = "LNcFac", vertex = "gamma_qq", optimization = "gpu", damp = 0, prec = np.float32):

        #float precision and optimization type
        self.prec = prec
        self.prec_c = np.result_type(1j * prec(1))
        self.optimization = optimization

        fm = prec(5.067)
        self.fm = fm
        self.E = prec(E * fm)
        self.qhat = prec(qF * fm**2)
        self.z = z
        self.omega = prec(self.E * z * (1 - z))
        self.Omega = (1 - 1j)/2 * prec(np.sqrt(self.qhat / self.omega))
        self.Ncmode = Ncmode
        self.vertex = vertex
        self.Lk = prec(Lk * fm)
        self.Ll = prec(Ll * fm)
        self.damp = prec(damp)
        self.xp = cp if optimization == "gpu" else np
        self.t = 1e-3 #initial time
        self.optimization = optimization
        self.Nk1 = None
        self.Nk2 = None
        self.Nl1 = None
        self.Nl2 = None
        self.K1 = None
        self.K2 = None
        self.L1 = None
        self.L2 = None
        self.dk1 = None
        self.dk2 = None
        self.dl1 = None
        self.dl2 = None
        self.Fsol = None

        if vertex == "gamma_qq":
            self.Nsig = 2
        else:
            raise ValueError("Other vertices not yet available")
        
    def set_dim(self, Nk1: int, Nk2: int, Nl1: int, Nl2: int):
        """Set the dimensions of the system for each axis p1, p2, q1 and q2"""
        self.Nk1, self.Nk2, self.Nl1, self.Nl2 = Nk1, Nk2, Nl1, Nl2

        if self.optimization == "gpu":
            #keep grid on GPU already
            deltae = 1e-4  # to avoid numerical issues at the edges
            print("Using GPU optimization")
            self.K1 = cp.linspace(-deltae, self.Lk - deltae, Nk1, dtype=self.prec)
            self.K2 = cp.linspace(0, self.Lk, Nk2, dtype=self.prec)
            self.L1 = cp.linspace(-self.Ll/2, self.Ll/2, Nl1, dtype=self.prec)
            self.L2 = cp.linspace(-self.Ll/2, self.Ll/2, Nl2, dtype=self.prec)

        else:
            deltae = 1e-3  # to avoid numerical issues at the edges
            print("Warning: Using CPU. Simulation will take significantly longer.")
            self.K1 = np.linspace(-deltae, self.Lk - deltae, Nk1, dtype=self.prec)
            self.K2 = np.linspace(0, self.Lk, Nk2, dtype=self.prec)
            self.L1 = np.linspace(-self.Ll/2, self.Ll/2, Nl1, dtype=self.prec)
            self.L2 = np.linspace(-self.Ll/2, self.Ll/2, Nl2, dtype=self.prec)

        self.dk1 = self.K1[1] - self.K1[0]
        self.dk2 = self.K2[1] - self.K2[0]
        self.dl1 = self.L1[1] - self.L1[0]
        self.dl2 = self.L2[1] - self.L2[0]

    def print_dim(self):
        """Print the dimensions of the system for all axis p1, p2, q1 and q2"""
        print(self.Nk1, self.Nk2, self.Nl1, self.Nl2)

    def beta(self, t):
        """The frequency of spatial oscilations"""
        co = self.omega * self.Omega * 0.5 * 1 / np.tan(self.Omega * t)
        return np.real(co)
    
    def eps(self, t):
        """Damp factor on the source term"""
        co = self.omega * self.Omega * 0.5 * 1 / np.tan(self.Omega * t)
        return np.imag(co)

    
    def source_term_array(self, t):
        """Source term in momentum space"""
        beta = self.beta(t)
        epsi = self.eps(t)
        Gamma = epsi - 1j * beta

        constant = 2 * 1j * self.omega
        K1 = self.K1[:, None, None, None]
        K2 = self.K2[None, :, None, None]
        L1 = self.L1[None, None, :, None]
        L2 = self.L2[None, None, None, :]

        ksqrd = K1**2 + K2**2
        lsqrd = L1**2 + L2**2

        kdotl = K1 * L1 + K2 * L2

        fac1 = 1 / (1 + 4 * kdotl / (4 * ksqrd + lsqrd))
        fac2 = 1 - np.exp(-(ksqrd + lsqrd/4 + kdotl) / (4 * Gamma))

        

        non_hom_term = - constant * fac1 * fac2 

        return non_hom_term 
    
    def init_fsol(self):
        """Initialize the grid for the solution. The dimensions of the system
        must be provided beforehand"""
        if self.Nk1 == None:
            raise ValueError("No value for the dimensions provided")
        else:
            if self.vertex == "gamma_qq":
                if self.optimization == "gpu":
                    self.Fsol = cp.zeros(shape=(2, self.Nk1, self.Nk2, self.Nl1, self.Nl2), dtype=self.prec_c)
                else:
                    self.Fsol = np.zeros(shape=(2, self.Nk1, self.Nk2, self.Nl1, self.Nl2), dtype=self.prec_c)
            else: 
                raise TypeError("Other vertices not yet available")
            
    def set_fsol(self, arr):
        """Set the solution grid"""
        self.Fsol = arr

    def increase_t(self, dt):
        """Increase the time by dt"""
        self.t += dt

    def set_t(self, t):
        """Set the time to t"""
        self.t = t
    
    def F_in_out(self, theta):
        """Calculate the in-out function for the source term"""
        Finout = np.real(-2 * (1 - np.exp(-1j * np.tan(self.Omega * self.t) / (2* self.omega * self.Omega)* self.omega**2 * theta**2)))
        
        return Finout


