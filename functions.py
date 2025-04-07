from system import *
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

    deriv_ux = (f[sig, i1+1, i2, j1, j2] - f[sig, i1-1, i2, j1, j2])/(2*du1)
    deriv_uy = (f[sig, i1, i2+1, j1, j2] - f[sig, i1, i2-1, j1, j2])/(2*du2)

    deriv2_ux = (f[sig, i1+1, i2, j1, j2] - 2 * f[sig, i1, i2, j1, j2] + f[sig, i1-1, i2, j1, j2])/(du1**2)
    deriv2_uy = (f[sig, i1, i2+1, j1, j2] - 2 * f[sig, i1, i2, j1, j2] + f[sig, i1, i2-1, j1, j2])/(du2**2)

    deriv2_vx = (f[sig, i1, i2, j1+1, j2] - 2 * f[sig, i1, i2, j1, j2] + f[sig, i1, i2, j1-1, j2])/(dv1**2)
    deriv2_vy = (f[sig, i1, i2, j1, j2 + 1] - 2 * f[sig, i1, i2, j1, j2] + f[sig, i1, i2, j1, j2-1])/(dv2**2)

    return -1/(2 * omega) * (deriv2_ux + deriv2_uy + 4j * (beta_t + 4*beta_t12 + beta_t1)/6 * (U1[i1] * deriv_ux + U2[i2] * deriv_uy) + \
                             - deriv2_vx - deriv2_vy ) 

def Kin_chunks(f, sig, U1, U2, V1, V2, omega, beta_t, beta_t1, cs_u1 = 16, cs_u2 = 16, cs_v1 = 16, cs_v2 = 16):

    du1 = U1[1] - U1[0]
    du2 = U2[1] - U2[0]

    dv1 = V1[1] - V1[0]
    dv2 = V2[1] - V2[0]

    kin_f = np.zeros_like(f[sig])

    for su1 in range(1, len(U1)-1, cs_u1):
        for su2 in range(1, len(U2)-1, cs_u2):
            for sv1 in range(1, len(V1)-1, cs_v1):
                for sv2 in range(1, len(V2)-1, cs_v2):

                    eu1 = min(su1 + cs_u1, len(U1)-1)
                    eu2 = min(su2 + cs_u2, len(U2)-1)

                    ev1 = min(sv1 + cs_v1, len(V1)-1)
                    ev2 = min(sv1 + cs_v2, len(V2)-1)
            
                    kin_f[su1:eu1, su2:eu2, sv1:ev1, sv2:ev2] = (f[sig, su1 + 1:eu1+1, su2:eu2, sv1:ev1, sv2:ev2]  -2 * f[sig, su1:eu1, su2:eu2, sv1:ev1, sv2:ev2] + f[sig, su1 - 1:eu1-1, su2:eu2, sv1:ev1, sv2:ev2]) / du1**2

                    kin_f[su1:eu1, su2:eu2, sv1:ev1, sv2:ev2] += (f[sig, su1:eu1, su2+1:eu2+1, sv1:ev1, sv2:ev2]  -2 * f[sig, su1:eu1,su2:eu2, sv1:ev1, sv2:ev2] + f[sig, su1:eu1, su2-1:eu2-1, sv1:ev1, sv2:ev2]) / du2**2

                    kin_f[su1:eu1, su2:eu2, sv1:ev1, sv2:ev2] += -(f[sig, su1:eu1, su2:eu2, sv1+1:ev1+1, sv2:ev2]  -2 * f[sig, su1:eu1,su2:eu2, sv1:ev1, sv2:ev2] + f[sig, su1:eu1, su2:eu2, sv1-1:ev1-1, sv2:ev2]) / dv1**2

                    kin_f[su1:eu1, su2:eu2, sv1:ev1, sv2:ev2] += -(f[sig, su1:eu1, su2:eu2, sv1:ev1, sv2+1:ev2+1]  -2 * f[sig, su1:eu1,su2:eu2, sv1:ev1, sv2:ev2] + f[sig, su1:eu1, su2:eu2, sv1:ev1, sv2-1:ev2-1]) / dv2**2

                    kin_f[su1:eu1, su2:eu2, sv1:ev1, sv2:ev2] += 4j * (beta_t + beta_t1)/2 * (U1[su1:eu1] * (f[sig, su1+1:eu1+1, su2:eu2, sv1:ev1, sv2:ev2] - f[sig, su1-1:eu1-1, su2:eu2, sv1-1:ev1-1, sv2:ev2]) / (2*du1) + U2[su2:eu2] * (f[sig, su1:eu1, su2+1:eu2+1, sv1:ev1, sv2:ev2] - f[sig, su1:eu1, su2+1:eu2+1, sv1:ev1, sv2:ev2]) / (2*du2))

                    kin_f *= -1/(2 *omega)


    return kin_f


@jit
def V_LargeNc(sig1, sig2, z, qF, U1, U2, V1, V2, i1, i2, j1, j2):
    if sig1 == 0 and sig2 == 0:
        element = (U1[i1]**2 + U2[i2]**2 + V1[j1]**2 + V2[j2]**2)
    
    elif sig1 == 0 and sig2 == 1:
        element = .0
    
    elif sig1 == 1 and sig2 == 0:
        #element =  2*z* (1-z)*((U[i1] - U[j1])*(U[i1] - U[j1]) + (U[i2] - U[j2])*(U[i2] - U[j2]))
        element = 0
    
    elif sig1 == 1 and sig2 == 1:
        element = (z **2 + (1-z)**2) * ((U1[i1] - V1[j1])**2 + (U2[i2] - V2[j2])**2)
    else:
        element = 0
        print("ERROR ON POTENTIAL")
    
    return -qF / (4.0) * element  

@jit
def eff_pot_gamqq(VLNc, i1, i2, U1, U2, omega, beta_t, debeta_t, beta_t1, debta_t1, beta_t12, debta_t12):

    ret_val = 1j * VLNc + (debeta_t + 4 * debta_t12 + debta_t1)/6 * (U1[i1]**2 + U2[i2]**2)
    
    ret_val -= 1/(2 * omega) * (4j * (beta_t + beta_t1)/2 - 4 * (beta_t**2 + + 4 * beta_t12**2 + beta_t1**2)/6 * (U1[i1]**2 + U2[i2]**2))

    return ret_val

@jit(nopython = True, parallel = True)
def apply_hamil(f, z, qF, U1, U2, V1, V2, beta_t, dbeta_t, beta_t1, debta_t1, beta_t12, debta_t12, omega):
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

                        VLNc = V_LargeNc(sigp, sigp, z, qF, U1, U2, V1, V2, i1, i2, j1, j2)
                        K = Kin(f, sigp, i1, i2, j1, j2, U1, U2, V1, V2, omega, beta_t, beta_t1, beta_t12)
                        V = eff_pot_gamqq(VLNc, i1, i2, U1, U2, omega, beta_t, dbeta_t, beta_t1, debta_t1, beta_t12, debta_t12)

                        g[sigp, i1, i2, j1, j2] = K + V * f[sigp, i1, i2, j1, j2]

    return g


def compute_fourier(f, beta, U1, U2, V1, V2, px, py):
        sum = 0j
        du1 = U1[1] - U1[0]
        du2 = U2[1] - U2[0]
        dv1 = V1[1] - V1[0]
        dv2 = V2[1] - V2[0]
        dVol = du1 * du2 * dv1 * dv2
        for i1 in range(1, len(U1)-1):
            for i2 in range(1, len(U2)-1):
                for j1 in range(1, len(V1)-1):
                    for j2 in range(1, len(V2)-1):
                        #sum[sigma] +=  compute_value_cell(X1, i1, i2, j1, j2, px, py)
                        osc = np.exp(1j * beta * (U1[i1]**2 + U2[i2]**2))
                        exponent_factor = np.exp(-1j * (px * (U1[i1] - V1[j1]) + py * (U2[i2] - V2[j2])))
                        sum += exponent_factor * osc * f[1, i1, i2, j1, j2]
            

        return sum * dVol


def compute_fourier_torch_chunks(fit, U1, U2, V1, V2, px, py, beta, chunksize_U1 = 8, chunksize_U2 = 8):
    

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
            U1_chunk = torch.tensor(U1[start1:end1], dtype=torch.float32, device=device)
            U2_chunk = torch.tensor(U2[start2:end2], dtype=torch.float32, device=device)

            x_dom_torcha = torch.cartesian_prod(U1_chunk, U2_chunk, V1, V2)
            
            # Evaluate the function `f` on the grid

            arg = beta * (x_dom_torcha[:, 0]**2 + x_dom_torcha[:, 1]**2) - px * (x_dom_torcha[:, 0] - x_dom_torcha[:, 2])

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

