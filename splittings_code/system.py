import numpy as np 
import cupy as cp #comment if not going to use GPU
from scipy import special

class physys3D:
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
    
    def __init__(self, E, z, qtilde, Lk, Ll, Ncmode = "LNcFac", vertex = "gamma_qq", optimization = "gpu", prec = np.float64):

        #float precision and optimization type
        self.prec = prec
        self.prec_c = np.result_type(1j * prec(1))
        self.optimization = optimization

        fm = prec(5.067)
        self.fm = fm
        self.E = prec(E * fm)
        self.qtilde = prec(qtilde * fm**2)
        self.z = z
        self.omega = prec(self.E * z * (1 - z))


        self.Ncmode = Ncmode
        self.vertex = vertex
        self.Lk = prec(Lk * fm)
        self.Ll = prec(Ll * fm)
        self.xp = cp if optimization == "gpu" else np
        self.t = 1e-8 #initial time
        
        if self.vertex == "gamma_qq":
            if self.Ncmode == "LNc" or "LNcFac":
                CF = 3/2 #now we use this for the computation of qab
            else:
                CF = (3**2 - 1) / (2 * 3)
            qab = self.qtilde * CF
            self.Omega = (1 - 1j)/2 * prec(np.sqrt(qab / self.omega))


        elif self.vertex == "q_qg":
            Nc = 3
            CA = Nc
            CF = (Nc**2 - 1) / (2 * Nc)
            if self.Ncmode == "LNc" or self.Ncmode == "LNcFac":
                CF = 3/2
                
            qab = 0.5 * ((CA) * (z) + CF * (1-z)**2) * self.qtilde
            
            self.Omega = (1 - 1j)/2 * prec(np.sqrt(qab / self.omega))


        elif self.vertex == "g_gg":
            Nc = 3
            CA = Nc
            CF = (Nc**2 - 1) / (2 * Nc)

            qab = (1 - z + z**2) * self.qtilde * CA 
            
            self.Omega = (1 - 1j)/2 * prec(np.sqrt(qab / self.omega))



        self.optimization = optimization
        self.Nk = None
        self.Nl = None
        self.Npsi = None
        self.K = None
        self.L = None
        self.psi = None
        self.dk = None
        self.dl = None
        self.dpsi = None
        self.Fsol = None   

        if vertex == "gamma_qq":
            self.Nsig = 2
        elif vertex == "q_qg":
            self.Nsig = 3
        elif vertex == "g_gg":
            self.Nsig = 8
        else:
            raise ValueError("Invalid vertex")
            

    def set_dim(self, Nk: int, Nl: int, Npsi: int):
        """Set the dimensions of the system for each axis p1, p2, q1 and q2"""
        self.Nk, self.Nl, self.Npsi = Nk, Nl, Npsi

        if self.optimization == "gpu":
            #keep grid on GPU already
            deltae = self.Lk / (Nk - 1)
            print("Using GPU optimization")
            self.K = cp.linspace(deltae, self.Lk + deltae, Nk, dtype=self.prec)
            self.L = cp.linspace(0, self.Ll, Nl, dtype=self.prec)
            self.psi = cp.linspace(0, 2*np.pi, Npsi, dtype=self.prec)

        else:
            deltae = self.Lk / (Nk - 1)
            print("Warning: Using CPU. Simulation can take longer.")
            self.K = np.linspace(deltae, self.Ll + deltae, Nk, dtype=self.prec)
            self.L = np.linspace(0, self.Ll, Nl, dtype=self.prec)
            self.psi = np.linspace(0, 2*np.pi, Npsi, dtype=self.prec)

        self.dk = self.K[1] - self.K[0]
        self.dl = self.L[1] - self.L[0]
        #self.psi = self.psi[:-1]  #to avoid double counting 0 and 2pi
        self.dpsi = self.psi[1] - self.psi[0]
        print(self.psi)

    def print_dim(self):
        """Print the dimensions of the system for all axis p1, p2, q1 and q2"""
        print(self.Nk, self.Nl, self.Npsi)

    def beta(self, t):
        """The frequency of spatial oscilations"""
        co = self.omega * self.Omega * 0.5 * 1 / np.tan(self.Omega * t)
        return co


    def source_term_array(self, t):
        """Source term in momentum space"""

        Gamma = -1j * self.beta(t) 

        constant = 2 * 1j * self.omega
        K = self.K[:, None, None]
        L = self.L[None, :, None]
        psi = self.psi[None, None, :]

        ksqrd = K**2
        lsqrd = L**2
        kdotl = K * L * np.cos(psi)

        fac1 = (ksqrd - lsqrd)/(ksqrd+ lsqrd + 2 * kdotl)

        fac2 = 1 - np.exp(-(ksqrd + lsqrd + 2*kdotl) / (4 * Gamma))

        non_hom_term = - constant * fac1 * fac2 

        return non_hom_term
    
    
    def init_fsol(self):
        """Initialize the grid for the solution. The dimensions of the system
        must be provided beforehand"""
        if self.Nk == None or self.Nl == None or self.Npsi == None: 
            raise ValueError("No value for some of the dimensions provided")
        else:
            if self.optimization == "gpu":
                self.Fsol = cp.zeros(shape=(self.Nsig, self.Nk, self.Nl, self.Npsi), dtype=self.prec_c)
            else:
                self.Fsol = np.zeros(shape=(self.Nsig, self.Nk, self.Nl, self.Npsi), dtype=self.prec_c)


    def set_fsol(self, arr):
        """Set the solution grid"""
        self.Fsol = arr

    def increase_t(self, dt):
        """Increase the time by dt"""
        self.t += dt

    def set_t(self, t):
        """Set the time to t"""
        self.t = t
