import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col

from matplotlib.colors import LinearSegmentedColormap

from matplotlib.animation import FuncAnimation

from matplotlib import rcParams

rcParams['font.size'] = 8
rcParams['mathtext.fontset'] = 'custom'

from scipy import special
from scipy.integrate import quad

from numba import jit

import os



def main(N, Evar, z, qFvar, L, theta, NcMode):
    fm = 5.067

    E = Evar * fm
    omega = E*z*(1-z)  
    print(qFvar)
    qF = qFvar * fm * fm
    Omega = (1.0 - 1.0j)/2.0 * np.sqrt(qF / omega)
    Nc = 3
    CF = (Nc * Nc - 1)/(2*Nc)

    print("Omega = ", omega)

    Nu1 = 2*N
    Nu2 = N
    Nv1 = 2*N
    Nv2 = N

    #grid parameters
    U1 = np.linspace(-L, L, Nu1)
    U2 = np.linspace(-L/2, L/2, Nu2)
    V1 = np.linspace(-L, L, Nv1)
    V2 = np.linspace(-L/2, L/2, Nv2)


    origin_v1 = Nv1//2
    origin_v2 = Nv2//2


    du1 = U1[1] - U1[0]
    du2 = U2[1] - U2[0]
    dv1 = V1[1] - V1[0]
    dv2 = V2[1] - V2[0]
    ht = 0.005

    #DIRAC DELTA AND DERIVATIVE
    delta_v1 = np.zeros(Nv1)
    d_delta_v1 = np.zeros(Nv1)

    delta_v1[origin_v1-1] = 1/(2 * dv1)
    delta_v1[origin_v1] = 1/(2 * dv1)
    
    d_delta_v1[origin_v1-1] = 1/(dv1*dv1)
    d_delta_v1[origin_v1] = -1/(dv1*dv1) 

    delta_v2 = np.zeros(Nv2)
    d_delta_v2 = np.zeros(Nv2)
    
    delta_v2[origin_v2-1] = 1/(2 * dv2)
    delta_v2[origin_v2] = 1/(2 * dv2)
    
    d_delta_v2[origin_v2-1] = 1/(dv2*dv2)
    d_delta_v2[origin_v2] = -1/(dv2*dv2) 


    #potential matrix V_{sig1, sig2}
    def V(sig1, sig2, i1, i2, j1, j2):

        if sig1 == 0 and sig2 == 0:
            element = (CF * (U1[i1]*U1[i1] + U2[i2]*U2[i2] + V1[j1]*V1[j1] + V2[j2]*V2[j2]) + 1.0/Nc * (U1[i1]*V1[j1] + U2[i2]*V2[j2]))
        
        elif sig1 == 0 and sig2 == 1:
            element = - 1.0/Nc * (U1[i1]*V1[j1] + U2[i2]*V2[j2])
        
        elif sig1 == 1 and sig2 == 0:
            element =  Nc*z* (1-z)*((U1[i1] - V1[j1])*(U1[i1] - V1[j1]) + (U2[i2] - V2[j2])*(U2[i2] - V2[j2]))
        
        elif sig1 == 1 and sig2 == 1:
            element = (CF - Nc*z*(1-z)) * ((U1[i1] - V1[j1])*(U1[i1] - V1[j1]) + (U2[i2] - V2[j2])*(U2[i2] - V2[j2]))
        else:
            element = 0
            print("ERROR ON POTENTIAL")
        
        return -qF / (4.0 * CF) * element 
    
    @jit(nopython=True)
    def V_LargeNc(sig1, sig2, i1, i2, j1, j2):
        if sig1 == 0 and sig2 == 0:
            element = (U1[i1]*U1[i1] + U2[i2]*U2[i2] + V1[j1]*V1[j1] + V2[j2]*V2[j2])
        
        elif sig1 == 0 and sig2 == 1:
            element = .0
        
        elif sig1 == 1 and sig2 == 0:
            #element =  2*z* (1-z)*((U[i1] - U[j1])*(U[i1] - U[j1]) + (U[i2] - U[j2])*(U[i2] - U[j2]))
            element = 0
        
        elif sig1 == 1 and sig2 == 1:
            element = (z **2 + (1-z)**2) * ((U1[i1] - V1[j1])*(U1[i1] - V1[j1]) + (U2[i2] - V2[j2])*(U2[i2] - V2[j2]))
        else:
            element = 0
            print("ERROR ON POTENTIAL")
        
        return -qF / (4.0) * element  

    @jit(nopython = True)
    def source_term(t, i1, i2, j1, j2):
        constant = 1j * omega / np.pi
        phase_fac = np.exp(0.5 * 1j * omega * Omega  / np.tan(Omega * (t)) * (U1[i1]*U1[i1] + U2[i2]*U2[i2]))

        non_hom_term = constant * phase_fac * (U1[i1] * d_delta_v1[j1] * delta_v2[j2] + U2[i2] * d_delta_v2[j2] * delta_v1[j1] )/ \
        (U1[i1]*U1[i1] + U2[i2]*U2[i2] + 1e-2 * du1**2)

        return non_hom_term 


    def fasit2Ncdiag(t,L,p1,p2,z):
        pre= -2*1j*omega
        
        num = 2*omega*Omega/np.tan(Omega*t)
        den = 2*omega*Omega/np.tan(Omega*t)+1j*qF*(z**2+(1-z)**2)*(L-t)
        
        return pre*(1-num/den*np.exp(-1j*(p1**2+p2**2)/den))

    def fasit2Ncdiagint(L,p1,p2,z):
        def real_fas(t,L,p1,p2,z):
            return np.real(fasit2Ncdiag(t,L,p1,p2,z))
        def imag_fas(t,L,p1,p2,z):
            return np.imag(fasit2Ncdiag(t,L,p1,p2,z))
        re = quad(real_fas,0,L,args=(L,p1,p2,z))[0]
        im = quad(imag_fas,0,L,args=(L,p1,p2,z))[0]
        
        return re + 1j*im



    def compute_V():
        V_field = np.zeros(shape=(2, 2, Nu1, Nu2, Nv1, Nv2)) + 0j
        for sig in range(2):
            for sigp in range(2):
                for i1 in range(1,Nu1-1):
                    for i2 in range(1,Nu2-1):
                        for j1 in range(1,Nv1-1):
                            for j2 in range(1,Nv2-1):
                                V_field[sig, sigp, i1,i2,j1,j2] = V(sig, sigp, i1, i2, j1, j2) + 0j
                                
        return V_field
    
    @jit(nopython=True, parallel=True)
    def compute_V_Nc():
        V_field2 = np.zeros(shape=(2, 2, Nu1, Nu2, Nv1, Nv2)) + 0j
        for sig in range(2):
            for sigp in range(2):
                for i1 in range(1,Nu1-1):
                    for i2 in range(1,Nu2-1):
                        for j1 in range(1,Nv1-1):
                            for j2 in range(1,Nv2-1):
                                V_field2[sig, sigp, i1,i2,j1,j2] = V_LargeNc(sig, sigp, i1, i2, j1, j2) + 0j
                                
        return V_field2

    @jit(parallel = True, nopython = True)
    def compute_non_hom(t):
        NHT = np.zeros(shape=(2, Nu1, Nu2, Nv1, Nv2)) + 0j
        for sig in range(2):
            for i1 in range(1, Nu1-1):
                for i2 in range(1, Nu2-1):
                    for j1 in range(1, Nv1-1):
                        for j2 in range(1, Nv2-1):
                            NHT[sig, i1,i2,j1,j2] = source_term(t, i1, i2, j1, j2)
                                
        return NHT
    
    
    def compute_derivative_4D(F, hu1, hu2, hv1, hv2):

        part_u1 = np.zeros(shape=(2, Nu1, Nu2, Nv1, Nv2)) +0j
       
        
        part_u1[:, 1:-1, 1:-1, 1:-1,1:-1] = F[:, 2:, 1:-1, 1:-1,1:-1] - 2 * F[:, 1:-1, 1:-1, 1:-1,1:-1] + F[:, :-2, 1:-1, 1:-1,1:-1]
        part_u1*= 1/(hu1**2)


        part_u2 = np.zeros(shape=(2, Nu1, Nu2, Nv1, Nv2)) +0j


        part_u2[:, 1:-1, 1:-1, 1:-1,1:-1] = F[:, 1:-1, 2:, 1:-1,1:-1] - 2 * F[:, 1:-1, 1:-1, 1:-1,1:-1] + F[:, 1:-1, :-2, 1:-1,1:-1]
        part_u2 *= 1/(hu2**2)
        
        part_v1 = .0 * part_u1
        part_v1[:, 1:-1, 1:-1, 1:-1,1:-1] = F[:, 1:-1, 1:-1, 2:,1:-1] - 2 * F[:, 1:-1, 1:-1, 1:-1,1:-1] + F[:, 1:-1, 1:-1, :-2,1:-1]
        part_v1 *= 1/(hv1**2)

        part_v2 = .0 * part_u1
        part_v2[:, 1:-1, 1:-1, 1:-1,1:-1] = F[:, 1:-1, 1:-1, 1:-1,2:] - 2 * F[:, 1:-1, 1:-1, 1:-1,1:-1] + F[:, 1:-1, 1:-1, 1:-1,:-2]
        part_v2 *= 1/(hv2**2)

        return np.array([part_u1, part_u2, part_v1, part_v2])

    @jit(nopython=True, parallel = True)
    def compute_derivative2_u1(F, sig, i1,i2,j1,j2, hu1):

        deriv = F[sig, i1 + 1, i2, j1, j2] - 2 * F[sig, i1 , i2, j1, j2] + F[sig, i1-1 , i2, j1, j2]
        deriv *= (1/hu1**2)

        return deriv

    @jit(nopython=True, parallel = True)
    def compute_derivative2_u2(F, sig, i1,i2,j1,j2, hu1):

        deriv = F[sig, i1 , i2 + 1, j1, j2] - 2 * F[sig, i1 , i2, j1, j2] +  F[sig, i1 , i2-1, j1, j2]
        deriv *= 1/(hu1**2)

        return deriv
    
    @jit(nopython=True, parallel = True)
    def compute_derivative2_v1(F, sig, i1,i2,j1,j2, hu1):

        deriv = F[sig, i1, i2, j1 + 1, j2] - 2 * F[sig, i1 , i2, j1, j2] +  F[sig, i1 , i2, j1 - 1, j2]
        deriv *= 1/(hu1**2)

        return deriv
    
    @jit(nopython=True, parallel = True)
    def compute_derivative2_v2(F, sig, i1,i2,j1,j2, hu1):

        deriv = F[sig, i1, i2, j1, j2 + 1] - 2 * F[sig, i1 , i2, j1, j2] +  F[sig, i1, i2, j1, j2 - 1]
        deriv *= 1/(hu1**2)

        return deriv
    
    @jit(nopython=True, parallel = True)
    def compute_kin_term_deriv(F, c_L, lamb):
            f_1 = .0 * F
            for sig in range(2):
                for i1 in range(1,Nu1-1):
                    for i2 in range(1,Nu2-1):
                        for j1 in range(1,Nv1-1):
                            for j2 in range(1,Nv2-1):
                                ddu1 = compute_derivative2_u1(F, sig, i1, i2, j1, j2, du1)
                                ddu2 = compute_derivative2_u2(F, sig, i1, i2, j1, j2, du2)
                                ddv1 = compute_derivative2_v1(F, sig, i1, i2, j1, j2, dv1)
                                ddv2 = compute_derivative2_v2(F, sig, i1, i2, j1, j2, dv2)
                                f_1[sig, i1, i2, j1, j2] = ddu1 + ddu2 - ddv1 - ddv2
                            
            f_1 *= (c_L / lamb)
            return f_1

    @jit(nopython=True, parallel = True)
    def compute_pot_term(F, Vfield, lamb):
        f_1 = .0 * F
        for sig in range(2):
            for sigp in range(2):
                for i1 in range(1,Nu1-1):
                    for i2 in range(1,Nu2-1):
                        for j1 in range(1,Nv1-1):
                            for j2 in range(1,Nv2-1):
                                V = V_LargeNc(sig, sigp, i1, i2, j1, j2) + 0j
                                f_1[sig, i1, i2, j1, j2] =  1j * V *F[sigp, i1, i2, j1, j2] / lamb
                            
        return f_1
    


    if(NcMode == "LargeNc"):
        VfieldNc = compute_V_Nc()
        print("V LargeNc computed \n")


    elif(NcMode == "FiniteNc"):
        VfieldNc = compute_V()
        print("V FiniteNc computed \n")

    else:
        print("ERROR")


    #Spectral properties of the hamiltonian for Faber expansion
    lam_re_max = 4 * (1/du1**2 + 1/du2**2 )/ (2*omega)  
    lam_re_min = -4 * (1/dv1**2 + 1/dv2**2 )/ (2*omega)    
    lam_im_min = -60 
    lam_im_max = 0  

    #print(8/du**2 / (2*omega)  )
    
    c = (lam_re_max - lam_re_min)/2
    l = (lam_im_max - lam_im_min)/2

    lambda_F = (l**(2/3) + c**(2/3))**(3/2) / 2


    csc = c/lambda_F
    lsc = l/lambda_F

    print(csc)
    print(lsc)

    gamma_0 = ((lam_re_max + lam_re_min) + 1j * (lam_im_min + lam_im_max))/(2*lambda_F) 
    gamma_1 = (((csc**(2/3) + lsc**(2/3)) * (csc**(4/3) - lsc**(4/3))))/(4)
    def coeff(m, dt1):
        exp_arg = 2*dt1*np.sqrt(gamma_1+0j)
        return (-1j/np.sqrt(gamma_1+0j))**m * np.exp(-1j * dt1 * gamma_0) * special.jv(m, exp_arg) #* np.exp(np.imag(exp_arg))

    print("Scaling factor = ", lambda_F, "\n")

    htau = ht * lambda_F

    print("Gamma_0 =", gamma_0)
    print("Gamma_1 =", gamma_1)

    coeff_list = []
    m = 0
    while True:
        coeff_list.append(coeff(m, htau))
        if abs(coeff_list[-1]) < 1e-6:
            break
        else:
            m += 1
    print(coeff_list)
   

    coeff_arr = np.array(coeff_list)

    Np = np.max(np.size(coeff_arr))
    print("Polynomial number = ", Np, "\n")

    c_L = -1/(2 * omega) 

    def faber_exp(f, Np, Vfield): #compute U(ht)*f
        """Computes e^(-iHdt)f using the Faber expansion. 
        Takes a 4D vector F, the number of polynomials Np and the potential field V"""
        fH_0 = f
        
        #fH_1 calculation

        fH_1 = compute_kin_term_deriv(fH_0, c_L, lambda_F) 
        fH_1 += compute_pot_term(fH_0, Vfield, lambda_F)

        fH_1 -= gamma_0 * fH_0

        #print("Before-derivative ", np.max(fH_1))

        fH_2 = compute_kin_term_deriv(fH_1, c_L, lambda_F) 
        fH_2 += compute_pot_term(fH_1, Vfield, lambda_F)


        fH_2 += -gamma_0 * fH_1 - 2 * gamma_1*fH_0

        #print("Post-gammas ", np.max(fH_2))

        

        Uf_est = coeff_arr[0] * fH_0 + coeff_arr[1] * fH_1 + coeff_arr[2] * fH_2
        #print(coeff(0, dt))
        #print(coeff(1, dt))
        #print(coeff(2, dt))

        for k in range(3, Np):
            fH_0 = fH_1
            fH_1 = fH_2


            fH_2 = compute_kin_term_deriv(fH_1, c_L, lambda_F) / lambda_F
            fH_2 += compute_pot_term(fH_1, c_L, lambda_F)

            fH_2 += -gamma_0 * fH_1 - gamma_1*fH_0

            Uf_est += coeff_arr[k] * fH_2

            #print("Polynomial ", k, " processed")
        print("Polynomials processed")



        return Uf_est


    T_prime = np.arange(0, L, ht)

    #Fp_prime = np.zeros(len(T_prime))
    Fmed_Nc = .0 * T_prime
    t0 = 1e-6 * ht

    

    @jit(parallel = True, nopython = True)
    def compute_value_cell(F, i1, i2, j1, j2, px, py):
        val = 0 + 0j
        for ii1 in [-1, 0,1]:
            for ii2 in [-1, 0,1]:
                for jj1 in [-1, 0,1]:
                    for jj2 in [-1, 0,1]:
                        exponent_factor = np.exp(-1j * (px * (U1[i1 + ii1] - V1[j1 + jj1]) + py * (U2[i2 + ii2] - V2[j2 + jj2])))
                        val += F[i1 + ii1, i2 + ii2, j1 + jj1, j2 + jj2] * exponent_factor
        
        return val / 3**4


    @jit(parallel = True, nopython = True)                    
    def compute_fourier(X, px, py):
        sum = np.array([.0j, .0j])
        dVol = du1 * du2 * dv1 * dv2
        for sigma in range(2):
            #X1 = X[sigma, :]
            for i1 in range(1, Nu1-1):
                for i2 in range(1, Nu2-1):
                    for j1 in range(1, Nv1-1):
                        for j2 in range(1, Nv2-1):
                            #sum[sigma] +=  compute_value_cell(X1, i1, i2, j1, j2, px, py)
                            exponent_factor = np.exp(-1j * (px * (U1[i1] - V1[j1]) + py * (U2[i2] - V2[j2])))
                            sum[sigma] += exponent_factor * X[sigma, i1, i2, j1, j2]

        return sum * dVol
    
    output_name = []

    for th in range(len(theta)):

        output_name.append(f"./results_files/Fmed_inin_N={N}_E={Evar}GeV_z={z}_theta={np.round(theta[th], 3)}_dt={ht}_L={L}_qF={np.round(qFvar,2)}_{NcMode}.txt")
        print("Open:", output_name[th], "\n")
        with open(output_name[th], 'w') as file:
            for t in range (len(T_prime)):
                Fmed_Nc[t] = np.real((fasit2Ncdiagint(T_prime[t] + t0, omega*theta[th], 0, z) ))
            file.write(f"{T_prime[0]} \t {0.000}\t {0.000}\n")
    

    F = np.zeros(shape=(2, Nu1, Nu2, Nv1, Nv2)) + 0j
    for i in range (1, len(T_prime)):
        yi  = F + 1/2 * compute_non_hom(T_prime[i-1] + t0) * ht 
        
        F = faber_exp(yi,  Np, VfieldNc) 

        F += 1/2 * compute_non_hom(T_prime[i]) * ht 

        for th in range(len(theta)):
            Fp_prime = np.real(compute_fourier(F, omega*theta[th], 0)[1])    
            Fmed_Nc= np.real((fasit2Ncdiagint(T_prime[i] + t0, omega*theta[th], 0, z) ))

            with open(output_name[th], 'a') as file:
                file.write(f"{T_prime[i]} \t {theta[th]**2 / 2 * Fp_prime}\t {theta[th]**2 / 2 *Fmed_Nc}\n")

            print(f"F_med_Theta{theta[th]} =", theta[th]**2 / 2 * Fp_prime)
        print("Processed time step t = ", T_prime[i])






