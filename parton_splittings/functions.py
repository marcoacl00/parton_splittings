from .system import *
#sfrom test_folders.Lanckzos_4D import *
from numba import jit
from scipy.integrate import quad

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@jit
def Kin(f, sig, i1, i2, j1, j2, U1, U2, V1, V2, omega, beta_t, beta_t1, beta_t12):

    du1 = U1[1] - U1[0]
    du2 = U2[1] - U2[0]

    dv1 = V1[1] - V1[0]
    dv2 = V2[1] - V2[0]

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

    dir_deriv = U1[i1] * deriv_ux + U2[i2] * deriv_uy

    beta_eff = (beta_t + 4*beta_t12 + beta_t1)/6

    return -1/(2 * omega) * (deriv2_u + 4j * beta_eff * dir_deriv - deriv2_v) 



@jit
def V_LargeNc(sig1, sig2, z, qF, U1, U2, V1, V2, i1, i2, j1, j2):
    if sig1 == 0 and sig2 == 0:

        u_sqrd = U1[i1]**2 + U2[i2]**2 
        v_sqrd = V1[j1]**2 + V2[j2]**2

        element = u_sqrd + v_sqrd 
    
    elif sig1 == 0 and sig2 == 1:
        element = .0
    
    elif sig1 == 1 and sig2 == 0:
        #element =  2*z* (1-z)*((U[i1] - U[j1])*
        #(U[i1] - U[j1]) + (U[i2] - U[j2])*(U[i2] - U[j2]))
        element = .0
    
    elif sig1 == 1 and sig2 == 1:

        uminusv_sqrd = (U1[i1] - V1[j1])**2 + (U2[i2] - V2[j2])**2

        element = (z **2 + (1-z)**2) * uminusv_sqrd
    else:
        element = 0
        print("ERROR ON POTENTIAL")
    
    return -qF / (4.0) * element  

@jit
def eff_pot_gamqq(VLNc, i1, i2, U1, U2, 
                    omega, beta_t, debeta_t,
                    beta_t1, debta_t1, 
                    beta_t12, debta_t12):

    #simpson rule for integration in betas and dbetas
    beta_eff = (beta_t + 4 * beta_t12 + beta_t1)/6
    dbeta_eff = (debeta_t + 4 * debta_t12 + debta_t1)/6

    u_sqrd = U1[i1]**2 + U2[i2]**2
    ret_val = 1j * VLNc +  beta_eff * u_sqrd
    
    ret_val -= 1/(2 * omega) * (4j * beta_eff - 4 * dbeta_eff * u_sqrd)

    return ret_val



@jit(nopython = True, parallel = True)
def apply_hamil(f, 
                z, qF, U1, U2, V1, V2, 
                beta_t, dbeta_t, 
                beta_t1, debta_t1, 
                beta_t12, debta_t12, 
                omega):
    
    g = .0 * f + 0j
    Nu1 = len(U1)
    Nu2 = len(U2)
    Nv1 = len(V1)
    Nv2 = len(V2)

    for sigp in range(2):
        for i1 in range(1, Nu1-1):
            for i2 in range(1, Nu2-1):
                for j1 in range(1, Nv1-1):
                    for j2 in range(1, Nv2-1):

                        VLNc = V_LargeNc(sigp, sigp,
                                          z, qF, U1, U2, V1, V2, 
                                          i1, i2, j1, j2)
                        
                        K = Kin(f, sigp, 
                                i1, i2, j1, j2, 
                                U1, U2, V1, V2, 
                                omega, 
                                beta_t, beta_t1, beta_t12)
                        
                        V = eff_pot_gamqq(VLNc, 
                                          i1, i2, U1, U2, 
                                          omega, 
                                          beta_t, dbeta_t, 
                                          beta_t1, debta_t1, 
                                          beta_t12, debta_t12)
                        

                        g[sigp, i1, i2, j1, j2] = K + V * f[sigp, i1, i2, j1, j2]

    return g



def compute_fourier_torch_chunks(fit, U1, U2, V1, V2, 
                                 px, py, beta, 
                                 chunksize_U1 = 8, 
                                 chunksize_U2 = 8):
    

    du1 = U1[1] - U1[0]
    du2 = U2[1] - U2[0]
    dv1 = V1[1] - V1[0]
    dv2 = V2[1] - V2[0]
    dVol = du1 * du2 * dv1 * dv2

    total_sum = 0
    
    # Process U1 in chunks
    V1 = torch.tensor(V1, dtype=torch.float32, device=device)
    V2 = torch.tensor(V2, dtype=torch.float32, device=device)
    
    chunk_cont = 1
    for start1 in range(0, len(U1), chunksize_U1):
        for start2 in range(0, len(U2), chunksize_U2):
        

            end1 = min(start1 + chunksize_U1, len(U1))
            end2 = min(start2 + chunksize_U2, len(U2))
            U1_chunk = torch.tensor(U1[start1:end1], 
                                    dtype=torch.float32, device=device)
            
            U2_chunk = torch.tensor(U2[start2:end2], 
                                    dtype=torch.float32, device=device)

            x_dom_torcha = torch.cartesian_prod(U1_chunk, U2_chunk, V1, V2)
            
            # Evaluate the function `f` on the grid
            ux_torcha = x_dom_torcha[:, 0]
            uy_torcha = x_dom_torcha[:, 1]
            vx_torcha = x_dom_torcha[:, 2]

            arg = beta * (ux_torcha**2 + uy_torcha**2) - px * (ux_torcha - vx_torcha)


            with torch.no_grad():

                fit_val = fit(x_dom_torcha).T
                chunk_sum = torch.sum(fit_val[0]*torch.cos(arg) - fit_val[1] * torch.sin(arg))

            # Accumulate the result
            
            total_sum += chunk_sum

            del x_dom_torcha, arg, chunk_sum

            torch.cuda.empty_cache()  # Frees unused memory
            torch.cuda.synchronize() 

            print("Chunk ", chunk_cont, " computed")
            chunk_cont +=1

    return total_sum * dVol



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

