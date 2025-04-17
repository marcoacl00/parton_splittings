import numpy as np

class phsys:
    def __init__(self, E, z, qF, L, Ncmode = "LNcFac", vertex = "gamma_qq"):
        fm = 5.067
        self.fm = fm
        self.E = E * fm
        self.qhat = qF * fm**2
        self.z = z
        self.omega = self.E * z * (1 - z)
        self.Omega = (1 - 1j)/2 * np.sqrt(self.qhat / self.omega)
        self.Ncmode = Ncmode
        self.vertex = vertex
        self.L = L
        self.t = 0.005
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
        self.U1 = np.linspace(-self.L/2, self.L/2, Nu1)
        self.U2 = np.linspace(-self.L/2, self.L/2, Nu2)
        self.V1 = np.linspace(-self.L/2, self.L/2, Nv1)
        self.V2 = np.linspace(-self.L/2, self.L/2, Nv2)
        self.du1 = self.U1[1] - self.U1[0]
        self.du2 = self.U2[1] - self.U2[0]
        self.dv1 = self.V1[1] - self.V1[0]
        self.dv2 = self.V2[1] - self.V2[0]

    def print_dim(self):
        """Print the dimensions of the system for all axis u1, u2, v1 and v2"""
        print(self.Nu1, self.Nu2, self.Nv1, self.Nv2)

    def beta(self, t):
        """The frequency of spatial oscilations"""
        co = self.omega * self.Omega / 2 * 1 / np.tan(self.Omega * t)
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
                self.Fsol = np.zeros(shape=(2, self.Nu1, self.Nu2, self.Nv1, self.Nv2))  + 0j  
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
    
    def V_LargeNc_gamma_qq(self, sig1, sig2, i1, i2, j1, j2):
        """The potential matrix for gamma -> qqbar in the large Nc factorized limit"""
        if sig1 == 0 and sig2 == 0:
            element = (self.U1[i1]*self.U1[i1] + self.U2[i2]*self.U2[i2]  + 
                       self.V1[j1]*self.V1[j1] + self.V2[j2]*self.V2[j2])
        
        elif sig1 == 0 and sig2 == 1:
            element = .0
        
        elif sig1 == 1 and sig2 == 0:
            #element =  2*z* (1-z)*((U[i1] - U[j1])*(U[i1] - U[j1]) + (U[i2] - U[j2])*(U[i2] - U[j2]))
            element = 0
        
        elif sig1 == 1 and sig2 == 1:
            element = (self.z **2 + (1-self.z)**2) * ((self.U1[i1] - self.V1[j1])*(self.U1[i1] - self.V1[j1]) + (self.U2[i2] - self.V2[j2])*(self.U2[i2] - self.V2[j2]))
        else:
            element = 0
            print("ERROR ON POTENTIAL")
    
        return -self.qhat / (4.0) * element 
    
    def dirac_v1(self, j1):
        retval = 0
        if j1 == self.Nv1//2 or j1 == self.Nv1//2 - 1:
            retval = 1/(2*self.dv1)

        return retval
    
    def dirac_gaussian(self, x):
        eps = 0.002
        return 1/np.sqrt(np.pi * eps) * np.exp(-x**2 / eps)
    
    def ddirac_gaussian(self, x):
        eps = 0.002
        return 1/np.sqrt(np.pi * eps) * (-2 * x / eps) * np.exp(-x**2 / eps)
        
    def dirac_v2(self, j2):
        retval = 0
        if j2 == self.Nv2//2 or j2 == self.Nv2//2 - 1:
            retval = 1/(2*self.dv2)
        
        return retval 
    
    def ddirac_v1(self, j2):
        retval = 0
        if j2 == self.Nv1//2-1:
            retval =  1/(self.dv1 ** 2)
        
        if j2 == self.Nv1//2:
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


    

