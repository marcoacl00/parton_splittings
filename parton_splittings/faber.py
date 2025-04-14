from .functions import *
from scipy import special


def faber_params(qF, du1, du2, dv1, dv2, beta_t, dbeta_t, omega, L):

    lam_re_max = 2 * (1/du1**2 + 1/du2**2 + (2 / (2*du1) + 2/ (2*du2)) * L * beta_t)/ omega + (2*beta_t**2 / omega  + dbeta_t) * L**2

    lam_re_min = -2 * (1/dv1**2 + 1/dv2**2 + (2/ (2*du1) + 2 / (2*du1)) * L * beta_t)/ omega  

    lam_im_max = 0
    lam_im_min = -2*qF - L**2*beta_t/omega

    c = (lam_re_max - lam_re_min)/2
    l = (lam_im_max - lam_im_min)/2

    lambda_F = (l**(2/3) + c**(2/3))**(3/2) / 2

    csc = c/lambda_F
    lsc = l/lambda_F

    gamma_0 = ((lam_re_max + lam_re_min) + 1j * (lam_im_min + lam_im_max))/(2*lambda_F) 
    gamma_1 = (((csc**(2/3) + lsc**(2/3)) * (csc**(4/3) - lsc**(4/3))))/(4)
    
    return lambda_F, gamma_0, gamma_1


def coeff(m, dt1, gamma_0, gamma_1, lambF):

    exp_arg = 2 * lambF * dt1*np.sqrt(gamma_1+0j)
    return (-1j/np.sqrt(gamma_1+0j))**m * np.exp(-1j * lambF * dt1 * gamma_0) * special.jv(m, exp_arg) #* np.exp(np.imag(exp_arg))
   
    
@jit
def faber_exp(f, gamma_0, gamma_1, coeff_arr, lambda_F, z, qF, U1, U2, V1, V2, beta_t, dbeta_t, beta_t1, debta_t1, beta_t12, debta_t12, omega):
        """Computes e^(-iHdt)f using the Faber expansion. 
        Takes a 4D vector F, the number of polynomials Np and the potential field V"""
        fH_0 = f

        fH_1 = apply_hamil(fH_0, z, qF, U1, U2, V1, V2, beta_t, dbeta_t, beta_t1, debta_t1, beta_t12, debta_t12, omega) / lambda_F
        fH_1 -= gamma_0 * fH_0

        fH_2 = apply_hamil(fH_1, z, qF, U1, U2, V1, V2, beta_t, dbeta_t, beta_t1, debta_t1, beta_t12, debta_t12, omega)  / lambda_F
        fH_2 += -gamma_0 * fH_1 - 2 * gamma_1*fH_0

        Uf_est = coeff_arr[0] * fH_0 + coeff_arr[1] * fH_1 + coeff_arr[2] * fH_2

        for k in range(3, len(coeff_arr)):
            fH_0 = fH_1
            fH_1 = fH_2

            fH_2 = apply_hamil(fH_1, z, qF, U1, U2, V1, V2, beta_t, dbeta_t, beta_t1, debta_t1, beta_t12, debta_t12, omega)  / lambda_F

            fH_2 += -gamma_0 * fH_1 - gamma_1*fH_0

            Uf_est += coeff_arr[k] * fH_2

        print("Polynomials processed")



        return Uf_est



def faber_par(sis):
     
    return faber_params(sis.qhat,
             sis.du1,
             sis.du2,
             sis.dv1,
             sis.dv2,
             sis.beta(sis.t),
             sis.dbeta(sis.t),
             sis.omega,
             sis.L)

def faber_expand(sis, ht):
    lambF, gamma0, gamma1 = faber_par(sis)

    print(lambF, gamma0, gamma1)

    
    coeff_array = [coeff(0, ht, gamma0, gamma1, lambF)]

    m = 1
    while (np.abs(coeff_array[-1]) > 1e-5 or m < 6):
        coeff_array.append(coeff(m, ht, gamma0, gamma1, lambF))
        m+=1

    print("Number of faber coefficients = ", len(coeff_array))
    
    return faber_exp(sis.Fsol, gamma0, gamma1, coeff_array, lambF, 
              sis.z, sis.qhat, sis.U1, sis.U2, sis.V1, sis.V2, 
              sis.beta(sis.t), sis.dbeta(sis.t), sis.beta(sis.t + ht), sis.dbeta(sis.t + ht), sis.beta(sis.t + ht/2), sis.dbeta(sis.t + ht/2), sis.omega)
