from .system_gpu import *



def V_eff_gamma_qq_LNc(sis, sig, sigp, i1, i2, j1, j2, dt):

    if sig == sigp:
        dt_2 = dt/2
        
        beta_t = sis.beta(sis.t)
        beta_t12 = sis.beta(sis.t + dt_2)
        beta_t1 = sis.beta(sis.t + dt)
        debeta_t = sis.dbeta(sis.t)
        debeta_t12 = sis.dbeta(sis.t + dt_2)
        debeta_t1 = sis.dbeta(sis.t + dt)

        beta_eff = (beta_t + 4 * beta_t12 + beta_t1)/6
        dbeta_eff = (debeta_t + 4 * debeta_t12 + debeta_t1)/6

        u_sqrd = sis.U1[i1]**2 + sis.U2[i2]**2

        V_ = sis.V_LargeNc_gamma_qq(sig, sigp, i1, i2, j1, j2)

        ret_val = 1j * V_ +  beta_eff * u_sqrd
        ret_val -= 1/(2 * sis.omega) * (4j * beta_eff - 4 * dbeta_eff * u_sqrd)

    else: 
        V_ = sis.V_LargeNc_gamma_qq(sig, sigp, i1, i2, j1, j2)

        ret_val = 1j*V_
    
    return ret_val








