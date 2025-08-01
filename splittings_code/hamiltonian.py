from .system import *

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
    """Computes kinetic operator without applying boundary conditions.
    Ghost points are added, but not physically constrained."""
    
    du1, du2 = sys.du1, sys.du2
    dv1, dv2 = sys.dv1, sys.dv2
    u1 = sys.U1[:, None, None, None]
    u2 = sys.U2[None, :, None, None]

    beta_t = sys.beta(sys.t)
    beta_t12 = sys.beta(sys.t + dt / 2)
    beta_t1 = sys.beta(sys.t + dt)
    beta_eff = (beta_t + 4 * beta_t12 + beta_t1) / 6

    omega = sys.omega

    f_ = f[sig]
    
    # Pad f_ in all dimensions with 1 ghost cell on each side using anti-periodic boundaries

    f_padded = cp.pad(f_, ((1, 1), (1, 1), (1, 1), (1, 1)), mode='edge')  # Pad with ghost cells

    del f_    
    # Compute derivatives on the original domain using ghost values
    deriv_ux = (f_padded[2:, 1:-1, 1:-1, 1:-1] - f_padded[:-2, 1:-1, 1:-1, 1:-1]) / (2 * du1)
    deriv_uy = (f_padded[1:-1, 2:, 1:-1, 1:-1] - f_padded[1:-1, :-2, 1:-1, 1:-1]) / (2 * du2)

    dir_deriv = u1 * deriv_ux + u2 * deriv_uy

    del deriv_ux, deriv_uy

    deriv2_ux = (f_padded[2:, 1:-1, 1:-1, 1:-1] - 2 * f_padded[1:-1, 1:-1, 1:-1, 1:-1] + f_padded[:-2, 1:-1, 1:-1, 1:-1]) / du1**2
    deriv2_uy = (f_padded[1:-1, 2:, 1:-1, 1:-1] - 2 * f_padded[1:-1, 1:-1, 1:-1, 1:-1] + f_padded[1:-1, :-2, 1:-1, 1:-1]) / du2**2

    deriv2_u = deriv2_ux + deriv2_uy

    del deriv2_ux, deriv2_uy

    deriv2_vx = (f_padded[1:-1, 1:-1, 2:, 1:-1] - 2 * f_padded[1:-1, 1:-1, 1:-1, 1:-1] + f_padded[1:-1, 1:-1, :-2, 1:-1]) / dv1**2
    deriv2_vy = (f_padded[1:-1, 1:-1, 1:-1, 2:] - 2 * f_padded[1:-1, 1:-1, 1:-1, 1:-1] + f_padded[1:-1, 1:-1, 1:-1, :-2]) / dv2**2

    deriv2_v = deriv2_vx + deriv2_vy

    del deriv2_vx, deriv2_vy


    return -1 / (2 * omega) * (deriv2_u + 4j * beta_eff * dir_deriv - deriv2_v)


def linear_extrapolation_pad(f_, pad_width):
    f_padded = f_
    ndim = f_.ndim
    
    for axis, (pad_before, pad_after) in enumerate(pad_width):
        shape = list(f_padded.shape)
        
        # Extrapolate at the start
        if pad_before > 0:
            slicer0 = [slice(None)] * ndim
            slicer1 = [slice(None)] * ndim
            slicer0[axis] = 0
            slicer1[axis] = 1
            first = f_padded[tuple(slicer0)]
            second = f_padded[tuple(slicer1)]
            slope = second - first

            # reshape to broadcast: insert pad dimension at correct axis
            first_b = cp.expand_dims(first, axis=axis)
            slope_b = cp.expand_dims(slope, axis=axis)
            offsets = cp.arange(pad_before, 0, -1, dtype=f_.dtype)
            shape_offsets = [1]*ndim
            shape_offsets[axis] = pad_before
            offsets = offsets.reshape(shape_offsets)
            
            start_pad = first_b - slope_b * offsets
            f_padded = cp.concatenate((start_pad, f_padded), axis=axis)

        # Extrapolate at the end
        if pad_after > 0:
            slicerN = [slice(None)] * ndim
            slicerNm1 = [slice(None)] * ndim
            slicerN[axis] = -1
            slicerNm1[axis] = -2
            last = f_padded[tuple(slicerN)]
            penultimate = f_padded[tuple(slicerNm1)]
            slope = last - penultimate

            last_b = cp.expand_dims(last, axis=axis)
            slope_b = cp.expand_dims(slope, axis=axis)
            offsets = cp.arange(1, pad_after + 1, dtype=f_.dtype)
            shape_offsets = [1]*ndim
            shape_offsets[axis] = pad_after
            offsets = offsets.reshape(shape_offsets)
            
            end_pad = last_b + slope_b * offsets
            f_padded = cp.concatenate((f_padded, end_pad), axis=axis)
            
    return f_padded




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


def Hamiltonian_momentum(sys, f, sig, dt):
    """
    Computes Hf.
    This acts as M(partial_p, partial_q, ...) on f.
    Assumes f has shape (2, Np1, Np2, Nq1, Nq2).
    """
    dp1, dp2 = sys.dp1, sys.dp2
    dq1, dq2 = sys.dq1, sys.dq2
    p1 = sys.P1[:, None, None, None]
    p2 = sys.P2[None, :, None, None]
    q1 = sys.Q1[None, None, :, None]
    q2 = sys.Q2[None, None, None, :]

    omega = sys.omega

    f_ = f[sig]

    f_padded = cp.pad(f_, ((1, 1), (1, 1), (1, 1), (1, 1)), mode='edge')  # Pad with ghost cells

    # Second derivatives in momentum space (bulk only)
    deriv2_p1 = (f_padded[2:, 1:-1, 1:-1, 1:-1] 
                 - 2 * f_padded[1:-1, 1:-1, 1:-1, 1:-1] 
                 + f_padded[:-2, 1:-1, 1:-1, 1:-1]) / dp1**2
    
    deriv2_p2 = (f_padded[1:-1, 2:, 1:-1, 1:-1] -
                  2 * f_padded[1:-1, 1:-1, 1:-1, 1:-1] 
                  + f_padded[1:-1, :-2, 1:-1, 1:-1]) / dp2**2
    
    deriv2_q1 = (f_padded[1:-1, 1:-1, 2:, 1:-1] - 
                 2 * f_padded[1:-1, 1:-1, 1:-1, 1:-1] 
                 + f_padded[1:-1, 1:-1, :-2, 1:-1]) / dq1**2
    
    deriv2_q2 = (f_padded[1:-1, 1:-1, 1:-1, 2:] - 
                 2 * f_padded[1:-1, 1:-1, 1:-1, 1:-1] 
                 + f_padded[1:-1, 1:-1, 1:-1, :-2]) / dq2**2

    # Mixed partial derivatives (bulk only)
    deriv2_p1_q1 = (f_padded[2:, 1:-1, 2:, 1:-1] - 
                    f_padded[2:, 1:-1, :-2, 1:-1] - 
                    f_padded[:-2, 1:-1, 2:, 1:-1] + 
                    f_padded[:-2, 1:-1, :-2, 1:-1]) / (4 * dp1 * dq1)
    
    deriv2_p2_q2 = (f_padded[1:-1, 2:, 1:-1, 2:] - 
                    f_padded[1:-1, 2:, 1:-1, :-2] - 
                    f_padded[1:-1, :-2, 1:-1, 2:] + 
                    f_padded[1:-1, :-2, 1:-1, :-2]) / (4 * dp2 * dq2)

    # kinetic term
    kin_term = 1 / (2 * omega) * (p1**2 + p2**2 - q1**2 - q2**2) * f_ 

    if sig == 0:
        V_term = sys.qhat/4 * (deriv2_p1 + deriv2_p2 + deriv2_q1 + deriv2_q2)
    else: 
        g_z = sys.z**2 + (1 - sys.z)**2
        V_term = sys.qhat * g_z / 4 * (deriv2_p1 + deriv2_p2 + deriv2_q1 + deriv2_q2 - 2 * (deriv2_p1_q1 + deriv2_p2_q2))

    result = 1j * V_term + kin_term 


    return result



def Hamiltonian_new_momentum(sys, f):
    """
    Computes Hf.
    This acts as M(partial_p, partial_q, ...) on f.
    Assumes f has shape (2, Np1, Np2, Nq1, Nq2).
    """
    dk1, dk2 = sys.dk1, sys.dk2
    dl1, dl2 = sys.dl1, sys.dl2
    k1 = sys.K1[:, None, None, None]
    k2 = sys.K2[None, :, None, None]
    l1 = sys.L1[None, None, :, None]
    l2 = sys.L2[None, None, None, :]

    omega = sys.omega
    f_0 = f[0]
    f_1 = f[1]


    # Second derivatives in momentum space (bulk only)

    # Pad both f_0 and f_1 with linear extrapolation
    #if sys.optimization == "gpu":
    #    f_0_padded = cp.pad(f_0, ((1, 1), (1, 1), (1, 1), (1, 1)), mode='edge')  # Pad with ghost cells
    #    f_1_padded = cp.pad(f_1, ((1, 1), (1, 1), (1, 1), (1, 1)), mode='edge')  # Pad with ghost cells
    #else: 
    #    f_0_padded = np.pad(f_0, ((1, 1), (1, 1), (1, 1), (1, 1)), mode='edge')  # Pad with ghost cells
    #    f_1_padded = np.pad(f_1, ((1, 1), (1, 1), (1, 1), (1, 1)), mode='edge')  # Pad with ghost cells

    xp = cp if sys.optimization == "gpu" else np
    f_0_padded = xp.pad(f_0, ((1, 1), (1, 1), (1, 1), (1, 1)), mode='edge')  # Pad with ghost cells
    f_1_padded = xp.pad(f_0, ((1, 1), (1, 1), (1, 1), (1, 1)), mode='edge')

    # simple derivatives
    #deriv_k1 = (f_padded[2:, 1:-1, 1:-1, 1:-1] - f_padded[:-2, 1:-1, 1:-1, 1:-1]) / (2 * dk1)
    #deriv_k2 = (f_padded[1:-1, 2:, 1:-1, 1:-1] - f_padded[1:-1, :-2, 1:-1, 1:-1]) / (2 * dk2)


    # Mixed partial derivatives (bulk only)
    # deriv2_k1_l1 = (f_padded[2:, 1:-1, 2:, 1:-1] - 
    #                 f_padded[2:, 1:-1, :-2, 1:-1] - 
    #                 f_padded[:-2, 1:-1, 2:, 1:-1] + 
    #                 f_padded[:-2, 1:-1, :-2, 1:-1]) / (4 * dk1 * dl1)
    
    # deriv2_k2_l2 = (f_padded[1:-1, 2:, 1:-1, 2:] - 
    #                 f_padded[1:-1, 2:, 1:-1, :-2] - 
    #                 f_padded[1:-1, :-2, 1:-1, 2:] + 
    #                 f_padded[1:-1, :-2, 1:-1, :-2]) / (4 * dk2 * dl2)
    

    # kinetic term
    kin_term_1 = 1 / (omega) * (k1 * l1 + k2 * l2) * f_1
    kin_term_0 = 1 / (omega) * (k1 * l1 + k2 * l2) * f_0

    # Derivatives to include in the potential term


    # ∂²f0/∂k1²
    deriv2_k1_0 = (f_0_padded[2:, 1:-1, 1:-1, 1:-1] 
            - 2 * f_0_padded[1:-1, 1:-1, 1:-1, 1:-1] 
            + f_0_padded[:-2, 1:-1, 1:-1, 1:-1]) / dk1**2
    
    # ∂²f0/∂k2²
    deriv2_k2_0 = (f_0_padded[1:-1, 2:, 1:-1, 1:-1] -
            2 * f_0_padded[1:-1, 1:-1, 1:-1, 1:-1] 
            + f_0_padded[1:-1, :-2, 1:-1, 1:-1]) / dk2**2
    
    # ∂²f1/∂k1²
    deriv2_k1_1 = (f_1_padded[2:, 1:-1, 1:-1, 1:-1]
            - 2 * f_1_padded[1:-1, 1:-1, 1:-1, 1:-1] 
            + f_1_padded[:-2, 1:-1, 1:-1, 1:-1]) / dk1**2
    
    # ∂²f1/∂k2²
    deriv2_k2_1 = (f_1_padded[1:-1, 2:, 1:-1, 1:-1] -
            2 * f_1_padded[1:-1, 1:-1, 1:-1, 1:-1] 
            + f_1_padded[1:-1, :-2, 1:-1, 1:-1]) / dk2**2

    # ∂²f0/∂l1²  
    deriv2_l1_0 = (f_0_padded[1:-1, 1:-1, 2:, 1:-1] - 
            2 * f_0_padded[1:-1, 1:-1, 1:-1, 1:-1] 
            + f_0_padded[1:-1, 1:-1, :-2, 1:-1]) / dl1**2

    # ∂²f0/∂l2² 
    deriv2_l2_0 = (f_0_padded[1:-1, 1:-1, 1:-1, 2:] - 
            2 * f_0_padded[1:-1, 1:-1, 1:-1, 1:-1] 
            + f_0_padded[1:-1, 1:-1, 1:-1, :-2]) / dl2**2
    
    
    
    #deriv_l1_0 = (f_0_padded[1:-1, 1:-1, 2:, 1:-1] - f_0_padded[1:-1, 1:-1, :-2, 1:-1]) / (2 * dl1)
    #deriv_l2_0 = (f_0_padded[1:-1, 1:-1, 1:-1, 2:] - f_0_padded[1:-1, 1:-1, 1:-1, :-2]) / (2 * dl2)

    
    if sys.Ncmode == "LNcFac" or sys.Ncmode == "LNc":

        q_4 = sys.qhat * 0.25
        
        V_term_00 = (0.5 * (deriv2_l1_0 + deriv2_l2_0 ) + deriv2_k1_0 + deriv2_k2_0)
            
        g_z = sys.z**2 + (1 - sys.z)**2
        V_term_11 = g_z  * (deriv2_k1_1 + deriv2_k2_1)

        V_1 = V_term_11

        if sys.Ncmode == "LNc":
            V_term_10 =  2 * sys.z * (1 - sys.z) * (deriv2_k1_0 + deriv2_k2_0)
            V_1 += V_term_10

        
        HF_0 = 1j * q_4 * V_term_00 + kin_term_0
        HF_1 = + kin_term_1 + 1j * q_4 * (V_1) 

    else:
        Nc = 3
        CF = (Nc**2 - 1) / (2 * Nc)
        fz =  sys.z * (1 - sys.z)

        # ∂²f1/∂l1² 
        deriv2_l1_1 = (f_1_padded[1:-1, 1:-1, 2:, 1:-1] - 
                2 * f_1_padded[1:-1, 1:-1, 1:-1, 1:-1] 
                + f_1_padded[1:-1, 1:-1, :-2, 1:-1]) / dl1**2
        
        # ∂²f1/∂l2² 
        deriv2_l2_1 = (f_1_padded[1:-1, 1:-1, 1:-1, 2:] - 
                2 * f_1_padded[1:-1, 1:-1, 1:-1, 1:-1] 
                + f_1_padded[1:-1, 1:-1, 1:-1, :-2]) / dl2**2

        q_4_CF = 0.25 * sys.qhat/CF

        V_term_00 = (CF * (0.5 * (deriv2_l1_0 + deriv2_l2_0 + 
                                           deriv2_k1_0 + deriv2_k2_0)) 

                              + 1/Nc * (deriv2_k1_0 + deriv2_k2_0 -
                                         0.25 * (deriv2_l1_0 + deriv2_l1_0)))

        V_term01 = -1/Nc * (deriv2_k1_1 + deriv2_k2_1 
                                      - 0.25 * (deriv2_l1_1 + deriv2_l2_1))

        V_term10 =  Nc * fz  * (deriv2_k1_0 + deriv2_k2_0 )

        V_term11 =  (CF  - Nc * fz)  * (deriv2_k1_1 + deriv2_k2_1)

        HF_0 = kin_term_0 + 1j * q_4_CF *(V_term_00 + V_term01)
        HF_1 = kin_term_1 + 1j * q_4_CF * (V_term10 + V_term11)

    if sys.optimization == "gpu":
        HF = cp.array([HF_0, HF_1])
    else:
        HF = np.array([HF_0, HF_1])

    return HF


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

        u_sqrd = sys.U1[:, None, None, None]**2 + sys.U2[None, :, None, None]**2

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

            #Hf[0, 1:-1, 1:-1, 1:-1, 1:-1] = Kin_par(sis, f, 0, ht)
            Hf[0] = Kin_par(sis, f, 0, ht)

            V_ss = V_eff_gamma_qq_LNc_par(sis, 0, 0, ht)
            Hf[0] += V_ss * f[0]
            V_ss = V_eff_gamma_qq_LNc_par(sis, 0, 1, ht)
            Hf[0] += V_ss * f[1]

            Hf[1] = Kin_par(sis, f, 1, ht)

            V_ss = V_eff_gamma_qq_LNc_par(sis, 1, 0, ht)
            Hf[1] += V_ss * f[0]
            V_ss = V_eff_gamma_qq_LNc_par(sis, 1, 1, ht)
            Hf[1] += V_ss * f[1]

            del V_ss

        else:
            raise TypeError


    else:
        raise TypeError
    
    return Hf




def apply_hamil_new_mom_par(sis, f):
    if sis.vertex == "gamma_qq":
      
        Hf = Hamiltonian_new_momentum(sis, f)

    else:
        raise TypeError
    
    return Hf










