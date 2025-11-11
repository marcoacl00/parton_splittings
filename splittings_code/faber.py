from .system import *
from .hamiltonian import *


from scipy import special
from tqdm import tqdm


def faber_params3D(sys):

    #estimate spectrum of hamiltonian (effective)
    qtilde = sys.qtilde
    dk = sys.dk
    dl = sys.dl
    dpsi = sys.dpsi

    Lk = sys.Lk
    Ll = sys.Ll
    omega = sys.omega

    #Maximum real eigenvalue estimate
    lam_re_max = 2 * Lk * Ll / (omega) 

    #Minimum real eigenvalue

    lam_re_min = -2 * Lk * Ll / (omega) -  qtilde / 4 * 1/dl**2 - 3 * qtilde / 4 * 1/dk**2

    
    #Maximum imag eigenvalue
    d_d2_eig = - 8 * (1/dk**2) - 8 * (1/dl**2) - 4/dl**2 * 1/dpsi**2

    lam_im_min= qtilde / 4 * (d_d2_eig)


    #Minimum imag eigenvalue

    lam_im_max = 0

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


def coeff(m, dt1, gamma_0, gamma_1, lambF, optim):

    sqrt_term = (-1j/np.sqrt(gamma_1+0j))**m
    jv_arg = 2 * lambF * dt1*np.sqrt(gamma_1+0j)
    if optim == "gpu":
        jv_arg = jv_arg.get()
    exp_arg = -1j * lambF * dt1 * gamma_0


    bessel = special.jv(int(m),jv_arg)

    return  sqrt_term * np.exp(exp_arg) *  bessel



def faber_expand3D(sys, ht, gamma0, gamma1, one_lamb, coeff_array=None):
    """Computes e^(-iHdt)f using the Faber expansion. 
    Takes a 4D vector F, the number of polynomials 
    Np and the potential field V"""

    #print("Faber params: lamb = ", lambF, "gamma0 = ", gamma0, "gamma1 = ", gamma1)

    
    fH_0 = 1.0 * sys.Fsol
    fH_1 = apply_hamil_3D(sys, fH_0) * one_lamb
   
    fH_1 -= gamma0 * fH_0
    fH_2 = apply_hamil_3D(sys, fH_1) * one_lamb
    fH_2 += -gamma0 * fH_1 - 2 * gamma1*fH_0
    Uf_est = coeff_array[0] * fH_0 + coeff_array[1] * fH_1 + coeff_array[2] * fH_2
    #print("norm of zero order term: ", np.linalg.norm(fH_0))
    #print("norm of first order term: ", np.linalg.norm(fH_1))
    #print("norm of second order term: ", np.linalg.norm(fH_2))
    for k in range(3, len(coeff_array)):
        fH_0 = fH_1
        fH_1 = fH_2
        fH_2 = apply_hamil_3D(sys, fH_1) * one_lamb
        fH_2 += -gamma0 * fH_1 - gamma1*fH_0
        Uf_est += coeff_array[k] * fH_2
        #print(f"norm of order {k} term: ", np.linalg.norm(coeff_array[k] * fH_2))


    return Uf_est

