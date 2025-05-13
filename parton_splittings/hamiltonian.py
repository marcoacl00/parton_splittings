from .system_gpu import *

def Kin(sys, sig, i1, i2, j1, j2, dt):
    
    f = sys.Fsol
    du1 = sys.du1
    du2 = sys.du2
    dv1 = sys.dv1
    dv2 = sys.dv2
    u1 = sys.U1[i1]
    u2 = sys.U2[i2]

    beta_t = sys.beta(sys.t)
    beta_t12 = sys.beta(sys.t + dt/2)
    beta_t1 = sys.beta(sys.t + dt)

    omega = sys.omega

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
    deriv2_uy = (f_uyplus1 - 2 * f_ + f_uyminus1)/(du2**2)

    deriv2_vx = (f_vxplus1 - 2 * f_ + f_vxminus1)/(dv1**2)
    deriv2_vy = (f_vyplus1 - 2 * f_ + f_vyminus1)/(dv2**2)

    deriv2_u = deriv2_ux + deriv2_uy
    deriv2_v = deriv2_vx + deriv2_vy

    dir_deriv = u1 * deriv_ux + u2 * deriv_uy

    beta_eff = (beta_t + 4*beta_t12 + beta_t1)/6

    return -1/(2 * omega) * (deriv2_u + 4j * beta_eff * dir_deriv - deriv2_v) 

    
def Kin_par(sys, f, sig, dt):
    """!! This will return something with a reduced dimension
        Must be taken into account when computing H.f"""
    du1 = sys.du1
    du2 = sys.du2
    dv1 = sys.dv1
    dv2 = sys.dv2
    u1 = sys.U1[1:-1, None, None, None]
    u2 = sys.U2[None, 1:-1, None, None]

    beta_t = sys.beta(sys.t)
    beta_t12 = sys.beta(sys.t + dt/2)
    beta_t1 = sys.beta(sys.t + dt)

    omega = sys.omega

    #a bunch of array extractions to compute the derivative
    f_ = f[sig, 1:-1,  1:-1,  1:-1,  1:-1]
    f_uxplus1 = f[sig, 2:, 1:-1, 1:-1, 1:-1]
    f_uxminus1 = f[sig, :-2, 1:-1, 1:-1, 1:-1]
    f_uyplus1 = f[sig, 1:-1, 2:, 1:-1, 1:-1]
    f_uyminus1 = f[sig, 1:-1, :-2, 1:-1, 1:-1]
    f_vxplus1 = f[sig, 1:-1, 1:-1, 2:, 1:-1]
    f_vxminus1 = f[sig, 1:-1, 1:-1, :-2, 1:-1]
    f_vyplus1 = f[sig, 1:-1, 1:-1, 1:-1, 2:]
    f_vyminus1 = f[sig, 1:-1, 1:-1, 1:-1, :-2]

    #finite difference approx for derivatives
    deriv_ux = (f_uxplus1 - f_uxminus1)/(2*du1)
    deriv_uy = (f_uyplus1 - f_uyminus1)/(2*du2)

    deriv2_ux = (f_uxplus1 - 2 * f_ + f_uxminus1)/(du1**2)
    deriv2_uy = (f_uyplus1 - 2 * f_ + f_uyminus1)/(du2**2)

    deriv2_vx = (f_vxplus1 - 2 * f_ + f_vxminus1)/(dv1**2)
    deriv2_vy = (f_vyplus1 - 2 * f_ + f_vyminus1)/(dv2**2)

    deriv2_u = deriv2_ux + deriv2_uy
    deriv2_v = deriv2_vx + deriv2_vy


    dir_deriv = u1 * deriv_ux + u2 * deriv_uy


    beta_eff = (beta_t + 4*beta_t12 + beta_t1)/6

    return -1/(2 * omega) * (deriv2_u + 4j * beta_eff * dir_deriv - deriv2_v) 


def V_eff_gamma_qq_LNc(sys, sig, sigp, i1, i2, j1, j2, dt):

    if sig == sigp:
        dt_2 = dt/2
        
        beta_t = sys.beta(sys.t)
        beta_t12 = sys.beta(sys.t + dt_2)
        beta_t1 = sys.beta(sys.t + dt)
        debeta_t = sys.dbeta(sys.t)
        debeta_t12 = sys.dbeta(sys.t + dt_2)
        debeta_t1 = sys.dbeta(sys.t + dt)

        beta_eff = (beta_t + 4 * beta_t12 + beta_t1)/6
        beta2_eff = (beta_t**2 + 4 * beta_t12**2 + beta_t1**2)/6
        dbeta_eff = (debeta_t + 4 * debeta_t12 + debeta_t1)/6

        u_sqrd = sys.U1[i1]**2 + sys.U2[i2]**2

        V_ = sys.V_LargeNc_gamma_qq(sig, sigp, i1, i2, j1, j2)

        ret_val = 1j * V_ +  dbeta_eff * u_sqrd
        ret_val -= 1/(2 * sys.omega) * (4j * beta_eff - 4 * beta2_eff * u_sqrd)

    else: 
        V_ = sys.V_LargeNc_gamma_qq(sig, sigp, i1, i2, j1, j2)

        ret_val = 1j*V_
    
    return ret_val



def V_eff_gamma_qq_LNc_par(sys, sig, sigp, dt):

    if sig == sigp:
        dt_2 = dt/2
        
        beta_t = sys.beta(sys.t)
        beta_t12 = sys.beta(sys.t + dt_2)
        beta_t1 = sys.beta(sys.t + dt)
        debeta_t = sys.dbeta(sys.t)
        debeta_t12 = sys.dbeta(sys.t + dt_2)
        debeta_t1 = sys.dbeta(sys.t + dt)

        one_six = 1/6

        beta_eff = (beta_t + 4 * beta_t12 + beta_t1) * one_six
        beta2_eff = (beta_t**2 + 4 * beta_t12**2 + beta_t1**2) * one_six
        dbeta_eff = (debeta_t + 4 * debeta_t12 + debeta_t1) * one_six

        u_sqrd = sys.U1[1:-1, None, None, None]**2 + sys.U2[None, 1:-1, None, None]**2

        V_ = sys.V_LargeNc_gamma_qq_par(sig, sigp)

        ret_arr = 1j * V_ +  dbeta_eff * u_sqrd
        ret_arr -= 2 / sys.omega * (1j * beta_eff - beta2_eff * u_sqrd)

    else: 

        V_ = sys.V_LargeNc_gamma_qq_par(sig, sigp)

        ret_arr = 1j*V_
    
    return ret_arr


def apply_hamil(sis, f, ht):

    if sis.vertex == "gamma_qq":
        if sis.Ncmode == "LNcFac":

            Hf = np.zeros_like(f)

            Hf[0, 1:-1, 1:-1, 1:-1, 1:-1] = Kin_par(sis, f, 0, ht)

            V_ss = V_eff_gamma_qq_LNc_par(sis, 0, 0, ht)
            Hf[0, 1:-1, 1:-1, 1:-1, 1:-1] += V_ss * f[0, 1:-1, 1:-1, 1:-1, 1:-1]
            V_ss = V_eff_gamma_qq_LNc_par(sis, 0, 1, ht)
            Hf[0, 1:-1, 1:-1, 1:-1, 1:-1] += V_ss * f[1, 1:-1, 1:-1, 1:-1, 1:-1]

            Hf[1, 1:-1, 1:-1, 1:-1, 1:-1] = Kin_par(sis, f, 1, ht)

            V_ss = V_eff_gamma_qq_LNc_par(sis, 1, 0, ht)
            Hf[1, 1:-1, 1:-1, 1:-1, 1:-1] += V_ss * f[0, 1:-1, 1:-1, 1:-1, 1:-1]
            V_ss = V_eff_gamma_qq_LNc_par(sis, 1, 1, ht)
            Hf[1, 1:-1, 1:-1, 1:-1, 1:-1] += V_ss * f[1, 1:-1, 1:-1, 1:-1, 1:-1]

            del V_ss

        else:
            raise TypeError


    else:
        raise TypeError
    
    return Hf









