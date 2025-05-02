from .system_gpu import *
from .hamiltonian import *


from scipy import special
from tqdm import tqdm


def faber_params(sys):

    qF = sys.qhat
    du1 = sys.du1
    du2 = sys.du2
    dv1 = sys.dv1
    dv2 = sys.dv2
    beta_t = sys.beta(sys.t)
    dbeta_t = sys.dbeta(sys.t)

    Lu = sys.Lu
    Lv = sys.Lv
    omega = sys.omega

    lam_re_max_deriv = 2 * (1/du1**2 + 1/du2**2 + 
                            (2 / (2*du1) + 2/ (2*du2)) * Lu * beta_t)/ omega #deriv contribution
    
    lam_re_max_extr = (2*beta_t**2 / omega  + dbeta_t) * Lu**2 #extra contribution from beta * u^2 terms
    lam_re_max =  lam_re_max_deriv + lam_re_max_extr

    #Minimum real eigenvalue
    lam_re_min = -2 * (1/dv1**2 + 1/dv2**2 + 
                       (2/ (2*du1) + 2 / (2*du1)) * Lv * beta_t)/ omega  
    
    #Maximum imag eigenvalue
    lam_im_max = 0

    #Minimum imag eigenvalue
    lam_im_min = -qF - Lu**2*beta_t/omega

    #Elipse params and scaling factor 
    c = (lam_re_max - lam_re_min)/2
    l = (lam_im_max - lam_im_min)/2

    lambda_F = (l**(2/3) + c**(2/3))**(3/2) / 2

    one_lamb = 1/lambda_F

    csc = c/lambda_F
    lsc = l/lambda_F

    gamma_0 = ((lam_re_max + lam_re_min) + 1j * (lam_im_min + lam_im_max)) * 0.5 * one_lamb 
    gamma_1 = (((csc**(2/3) + lsc**(2/3)) * (csc**(4/3) - lsc**(4/3)))) * 0.25
    
    return lambda_F, gamma_0, gamma_1



def coeff(m, dt1, gamma_0, gamma_1, lambF):

    sqrt_term = (-1j/np.sqrt(gamma_1+0j))**m
    jv_arg = 2 * lambF * dt1*np.sqrt(gamma_1+0j)
    jv_arg = jv_arg.get()
    exp_arg = -1j * lambF * dt1 * gamma_0


    bessel = special.jv(int(m),jv_arg)

    return  sqrt_term * np.exp(exp_arg) *  bessel


def faber_expand(sys, ht):
    """Computes e^(-iHdt)f using the Faber expansion. 
    Takes a 4D vector F, the number of polynomials 
    Np and the potential field V"""
    
    lambF, gamma0, gamma1 = faber_params(sys)
    one_lamb = 1/lambF

    print("Faber params: lamb = ", lambF, "gamma0 = ", gamma0, "gamma1 = ", gamma1)

    
    m = 1
    coeff_array = [coeff(0, ht, gamma0, gamma1, lambF)] #compute first coefficient
    while (np.abs(coeff_array[-1]) > 1e-7 or m < 6):
        coeff_array.append(coeff(m, ht, gamma0, gamma1, lambF))
        m += 1
    print("Number of polynomials = ", m)
    fH_0 = 1.0 * sys.Fsol

    fH_1 = apply_hamil(sys, fH_0, ht) * one_lamb
    fH_1 -= gamma0 * fH_0

    fH_2 = apply_hamil(sys, fH_1, ht) * one_lamb
    fH_2 += -gamma0 * fH_1 - 2 * gamma1*fH_0

    Uf_est = coeff_array[0] * fH_0 + coeff_array[1] * fH_1 + coeff_array[2] * fH_2

    for k in tqdm(range(3, len(coeff_array))):

        fH_0 = fH_1
        fH_1 = fH_2
        fH_2 = apply_hamil(sys, fH_1, ht) * one_lamb

        fH_2 += -gamma0 * fH_1 - gamma1*fH_0

        Uf_est += coeff_array[k] * fH_2

    print("Polynomials processed")



    return Uf_est
   
    

