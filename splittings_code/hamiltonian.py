from .system import *

def laplacian_at_origin(f, dl, order=4):
    """
    f: shape (Nk, Nl, Npsi)
    Returns Δf at l=0, shape (Nk,).
    f[:, 0, :] is psi-independent 
    """
    f0 = f[:, 0, :].mean(axis=-1)
    
    if order == 2:
        g1 = f[:, 1, :].mean(axis=-1) - f0
        return 4.0 * g1 / dl**2
    
    if order == 4:
        ring1 = f[:, 1, :].mean(axis=-1)
        ring2 = f[:, 2, :].mean(axis=-1)
        return (16.0 * ring1 - ring2 - 15.0 * f0) / (3.0 * dl**2)
    
    if order == 6:
        ring1 = f[:, 1, :].mean(axis=-1)
        ring2 = f[:, 2, :].mean(axis=-1)
        ring3 = f[:, 3, :].mean(axis=-1)
        return (270.0*ring1 - 27.0*ring2 + 2.0*ring3 - 245.0*f0) / (45.0 * dl**2)
    
    raise ValueError("order must be 2, 4, or 6")

def Hamitlonian3D_gammaqq(sys, f):
    """
    Computes Hf.
    This acts as M(partial_k, partial_l, partial_psi) on f.
    Assumes f has shape (2, Nk, Nl, Npsi).
    """

    dk = sys.dk
    dl = sys.dl
    dpsi = sys.dpsi
    k = sys.K[:, None, None]
    l = sys.L[None, :, None]
    psi = sys.psi[None, None, :]


    omega = sys.omega
    f_0 = f[0]
    f_1 = f[1]


    # Second derivatives in momentum space (bulk only)

    xp = cp if sys.optimization == "gpu" else np


    f_0[:, 0, :] = xp.mean(f_0[:, 0, :], axis=-1, keepdims=True)
    f_1[:, 0, :] = xp.mean(f_1[:, 0, :], axis=-1, keepdims=True)

    # f_0, f_1 have shape (Ns, Nl, Nk)
    # f_0, f_1 have shape (Ns, Nl, Nk)
    # xp = numpy or cupy
    # dk, dl are scalars

    # f_0, f_1 have shape (Nk, Nl, Npsi)
    Nk, Nl, Npsi = f_0.shape

    deriv_k_0  = xp.zeros_like(f_0)
    deriv_l_0  = xp.zeros_like(f_0)
    deriv2_k_0 = xp.zeros_like(f_0)
    deriv2_l_0 = xp.zeros_like(f_0)

    deriv_k_1  = xp.zeros_like(f_1)
    deriv_l_1  = xp.zeros_like(f_1)
    deriv2_k_1 = xp.zeros_like(f_1)
    deriv2_l_1 = xp.zeros_like(f_1)

    # ============================================================
    # ∂/∂k  (axis 0)
    # ============================================================

    # bulk
    deriv_k_0[1:-1, :, :] = (f_0[2:, :, :] - f_0[:-2, :, :]) * (0.5 / dk)
    deriv_k_1[1:-1, :, :] = (f_1[2:, :, :] - f_1[:-2, :, :]) * (0.5 / dk)

    # k = 0
    deriv_k_0[0, :, :] = (-3*f_0[0, :, :] + 4*f_0[1, :, :] - f_0[2, :, :]) * (0.5 / dk)
    deriv_k_1[0, :, :] = (-3*f_1[0, :, :] + 4*f_1[1, :, :] - f_1[2, :, :]) * (0.5 / dk)

    # k = Nk-1
    deriv_k_0[-1, :, :] = (3*f_0[-1, :, :] - 4*f_0[-2, :, :] + f_0[-3, :, :]) * (0.5 / dk)
    deriv_k_1[-1, :, :] = (3*f_1[-1, :, :] - 4*f_1[-2, :, :] + f_1[-3, :, :]) * (0.5 / dk)

    # ============================================================
    # ∂²/∂k²  (axis 0)
    # ============================================================

    # bulk
    deriv2_k_0[1:-1, :, :] = (
        f_0[2:, :, :] - 2*f_0[1:-1, :, :] + f_0[:-2, :, :]
    ) / dk**2

    deriv2_k_1[1:-1, :, :] = (
        f_1[2:, :, :] - 2*f_1[1:-1, :, :] + f_1[:-2, :, :]
    ) / dk**2

    # k = 0
    deriv2_k_0[0, :, :] = (
        2*f_0[0, :, :] - 5*f_0[1, :, :] + 4*f_0[2, :, :] - f_0[3, :, :]
    ) / dk**2

    deriv2_k_1[0, :, :] = (
        2*f_1[0, :, :] - 5*f_1[1, :, :] + 4*f_1[2, :, :] - f_1[3, :, :]
    ) / dk**2

    # k = Nk-1
    deriv2_k_0[-1, :, :] = (
        2*f_0[-1, :, :] - 5*f_0[-2, :, :] + 4*f_0[-3, :, :] - f_0[-4, :, :]
    ) / dk**2

    deriv2_k_1[-1, :, :] = (
        2*f_1[-1, :, :] - 5*f_1[-2, :, :] + 4*f_1[-3, :, :] - f_1[-4, :, :]
    ) / dk**2

    # ============================================================
    # ∂/∂l  (axis 1)
    # ============================================================

    # bulk
    deriv_l_0[:, 1:-1, :] = (f_0[:, 2:, :] - f_0[:, :-2, :]) * (0.5 / dl)
    deriv_l_1[:, 1:-1, :] = (f_1[:, 2:, :] - f_1[:, :-2, :]) * (0.5 / dl)

    # l = 0
    deriv_l_0[:, 0, :] = (
        -11*f_0[:, 0, :] + 18*f_0[:, 1, :] - 9*f_0[:, 2, :] + 2*f_0[:, 3, :]
    ) / (6*dl)

    deriv_l_1[:, 0, :] = (
        -11*f_1[:, 0, :] + 18*f_1[:, 1, :] - 9*f_1[:, 2, :] + 2*f_1[:, 3, :]
    ) / (6*dl)

    # l = Nl-1
    deriv_l_0[:, -1, :] = (
        3*f_0[:, -1, :] - 4*f_0[:, -2, :] + f_0[:, -3, :]
    ) * (0.5 / dl)

    deriv_l_1[:, -1, :] = (
        3*f_1[:, -1, :] - 4*f_1[:, -2, :] + f_1[:, -3, :]
    ) * (0.5 / dl)

    # ============================================================
    # ∂²/∂l²  (axis 1)
    # ============================================================

    # bulk
    deriv2_l_0[:, 1:-1, :] = (
        f_0[:, 2:, :] - 2*f_0[:, 1:-1, :] + f_0[:, :-2, :]
    ) / dl**2

    deriv2_l_1[:, 1:-1, :] = (
        f_1[:, 2:, :] - 2*f_1[:, 1:-1, :] + f_1[:, :-2, :]
    ) / dl**2

    # l = 0
    deriv2_l_0[:, 0, :] = (
        -35*f_0[:, 0, :] + 104*f_0[:, 1, :] - 114*f_0[:, 2, :]
        + 56*f_0[:, 3, :] - 11*f_0[:, 4, :]
    ) / (12*dl**2)

    deriv2_l_1[:, 0, :] = (
        -35*f_1[:, 0, :] + 104*f_1[:, 1, :] - 114*f_1[:, 2, :]
        + 56*f_1[:, 3, :] - 11*f_1[:, 4, :]
    ) / (12*dl**2)

    # l = Nl-1
    deriv2_l_0[:, -1, :] = (
        35*f_0[:, -1, :] - 104*f_0[:, -2, :] + 114*f_0[:, -3, :]
        - 56*f_0[:, -4, :] + 11*f_0[:, -5, :]
    ) / (12*dl**2)

    deriv2_l_1[:, -1, :] = (
        35*f_1[:, -1, :] - 104*f_1[:, -2, :] + 114*f_1[:, -3, :]
        - 56*f_1[:, -4, :] + 11*f_1[:, -5, :]
    ) / (12*dl**2)



    deriv2_psi_0 = xp.zeros_like(f_0)
    deriv2_psi_1 = xp.zeros_like(f_1)


    deriv2_psi_0[:,:,1:-1] = (f_0[:,:,2:] - 2*f_0[:,:,1:-1] + f_0[:,:,:-2]) / dpsi**2
    deriv2_psi_1[:,:,1:-1] = (f_1[:,:,2:] - 2*f_1[:,:,1:-1] + f_1[:,:,:-2]) / dpsi**2

    #psi is between 0 and pi
    deriv2_psi_0[:,:,0]  = (2*f_0[:,:,1] - 2*f_0[:,:,0]) / dpsi**2
    deriv2_psi_1[:,:,0]  = (2*f_1[:,:,1] - 2*f_1[:,:,0]) / dpsi**2

    deriv2_psi_0[:,:,-1]  = (2*f_0[:,:,-2] - 2*f_0[:,:,-1]) / dpsi**2
    deriv2_psi_1[:,:,-1]  = (2*f_1[:,:,-2] - 2*f_1[:,:,-1]) / dpsi**2
    
    # Polar Laplacian (bulk)
    lap_lpsi_0 = deriv2_l_0 + (1/l) * deriv_l_0 + (1/l**2) * deriv2_psi_0
    lap_lpsi_1 = deriv2_l_1 + (1/l) * deriv_l_1  + (1/l**2) * deriv2_psi_1

    # Correct l = 0 limit
    lap_lpsi_0[:,0,:] = laplacian_at_origin(f_0, dl, order=4)[:, None]
    lap_lpsi_1[:,0,:] = laplacian_at_origin(f_1, dl, order=4)[:, None]


    
    # kinetic term
    kin_term_1 = 1 / (omega) * (k * l * xp.cos(psi)) * f_1
    kin_term_0 = 1 / (omega) * (k * l * xp.cos(psi)) * f_0


    if sys.Ncmode == "LNcFac" or sys.Ncmode == "LNc" or sys.Ncmode == "LeadNc":
        
        CF = 3/2
        if sys.Ncmode == "LeadNc":
            CF = 4/3
            
        q_4 = sys.qtilde * 0.25 
        
        V_term_00 = 0.5 * (deriv2_k_0 + 1/k * deriv_k_0 + lap_lpsi_0) 
                            
        g_z = sys.z**2 + (1 - sys.z)**2
        V_term_11 = g_z  * (deriv2_k_1 + deriv_k_1 * 1/(k))

        V_1 = V_term_11

        if sys.Ncmode == "LNc":
            V_term_10 =  2 * sys.z * (1 - sys.z) * (deriv2_k_0 + 1/k * deriv_k_0)
            V_1 += V_term_10

        
        HF_0 = kin_term_0 + 1j * q_4 * CF * V_term_00 
        HF_1 = kin_term_1 + 1j * q_4 * CF * V_1


    elif sys.Ncmode == "LNc_qeff":
        CF = (3**2 - 1) / (2 * 3)
        qeff_4 = CF * sys.qtilde * 0.25

        V_term_00 = 0.5 * (deriv2_k_0 + 1/k * deriv_k_0 + lap_lpsi_0)
        
        g_z = sys.z**2 + (1 - sys.z)**2

        V_term_10 =  2 * sys.z * (1 - sys.z) * (deriv2_k_0 + 1/k * deriv_k_0)


        V_term_11 = g_z  * (deriv2_k_1 + deriv_k_1 * 1/(k))


        HF_0 = kin_term_0 + 1j * qeff_4 * V_term_00 
        HF_1 = kin_term_1 + 1j * qeff_4 * (V_term_11 + V_term_10)



    elif sys.Ncmode == "FNc":
        Nc = 3
        CF = 4/3

        q_4 = 0.25 * sys.qtilde

        fz =  sys.z * (1 - sys.z)

        V_term_00 = 0.5 * CF * (deriv2_k_0 + 1/k * deriv_k_0 + lap_lpsi_0) \
                              + 1/(2*Nc) * (deriv2_k_0 + 1/k * deriv_k_0 - lap_lpsi_0)

        V_term01 = - 1/(2*Nc) * (deriv2_k_1 + 1/k * deriv_k_1 - lap_lpsi_1)

        V_term10 =  Nc * fz  * (deriv2_k_0 + 1/k * deriv_k_0)

        V_term11 =  (CF  - Nc * fz)  * (deriv2_k_1 + 1/k * deriv_k_1)

        HF_0 = kin_term_0 + 1j * q_4 *(V_term_00 + V_term01)
        HF_1 = kin_term_1 + 1j * q_4 * (V_term10 + V_term11)

    #HF_0[:, 0, :] = xp.mean(HF_0[:, 0, :], axis=-1, keepdims=True)
    #HF_1[:, 0, :] = xp.mean(HF_1[:, 0, :], axis=-1, keepdims=True)
    
    if sys.optimization == "gpu":
        return cp.array([HF_0, HF_1])
    else:
        return np.array([HF_0, HF_1])


def Hamiltonian3D_qqg(sys, f):
    """
    Computes Hf.
    This acts as M(partial_k, partial_l, partial_psi) on f.
    Assumes f has shape (3, Nk, Nl, Npsi).
    """

    dk = sys.dk
    dl = sys.dl
    dpsi = sys.dpsi
    k = sys.K[:, None, None]
    l = sys.L[None, :, None]
    psi = sys.psi[None, None, :]
    epsl = dl  # to avoid division by zero

    omega = sys.omega
    f_0 = f[0]
    f_1 = f[1]
    f_2 = f[2]

    # Second derivatives in momentum space (bulk only)

    xp = cp if sys.optimization == "gpu" else np

    # Pad both f_0 and f_1 with linear extrapolation

    f_0[:, 0, :] = xp.mean(f_0[:, 0, :], axis=-1, keepdims=True)
    f_1[:, 0, :] = xp.mean(f_1[:, 0, :], axis=-1, keepdims=True)
    f_2[:, 0, :] = xp.mean(f_2[:, 0, :], axis=-1, keepdims=True)

    f0p = xp.pad(f_0, ((1,1),(1,1),(1,1)), mode="edge")
    f1p = xp.pad(f_1, ((1,1),(1,1),(1,1)), mode="edge")
    f2p = xp.pad(f_2, ((1,1),(1,1),(1,1)), mode="edge")

    # f_0, f_1 have shape (Ns, Nl, Nk)
    # f_0, f_1 have shape (Ns, Nl, Nk)
    # xp = numpy or cupy
    # dk, dl are scalars

    # f_0, f_1 have shape (Nk, Nl, Npsi)
    Nk, Nl, Npsi = f_0.shape

    deriv_k_0  = xp.zeros_like(f_0)
    deriv_l_0  = xp.zeros_like(f_0)
    deriv2_k_0 = xp.zeros_like(f_0)
    deriv2_l_0 = xp.zeros_like(f_0)

    deriv_k_1  = xp.zeros_like(f_1)
    deriv_l_1  = xp.zeros_like(f_1)
    deriv2_k_1 = xp.zeros_like(f_1)
    deriv2_l_1 = xp.zeros_like(f_1)

    deriv_k_2  = xp.zeros_like(f_2)
    deriv_l_2  = xp.zeros_like(f_2)
    deriv2_k_2 = xp.zeros_like(f_2)
    deriv2_l_2 = xp.zeros_like(f_2)
    

    # ============================================================
    # ∂/∂k  (axis 0)
    # ============================================================

    # bulk
    deriv_k_0[1:-1, :, :] = (f_0[2:, :, :] - f_0[:-2, :, :]) * (0.5 / dk)
    deriv_k_1[1:-1, :, :] = (f_1[2:, :, :] - f_1[:-2, :, :]) * (0.5 / dk)
    

    # k = 0
    deriv_k_0[0, :, :] = (-3*f_0[0, :, :] + 4*f_0[1, :, :] - f_0[2, :, :]) * (0.5 / dk)
    deriv_k_1[0, :, :] = (-3*f_1[0, :, :] + 4*f_1[1, :, :] - f_1[2, :, :]) * (0.5 / dk)

    # k = Nk-1
    deriv_k_0[-1, :, :] = (3*f_0[-1, :, :] - 4*f_0[-2, :, :] + f_0[-3, :, :]) * (0.5 / dk)
    deriv_k_1[-1, :, :] = (3*f_1[-1, :, :] - 4*f_1[-2, :, :] + f_1[-3, :, :]) * (0.5 / dk)

    # ============================================================
    # ∂²/∂k²  (axis 0)
    # ============================================================

    # bulk
    deriv2_k_0[1:-1, :, :] = (
        f_0[2:, :, :] - 2*f_0[1:-1, :, :] + f_0[:-2, :, :]
    ) / dk**2

    deriv2_k_1[1:-1, :, :] = (
        f_1[2:, :, :] - 2*f_1[1:-1, :, :] + f_1[:-2, :, :]
    ) / dk**2

    # k = 0
    deriv2_k_0[0, :, :] = (
        2*f_0[0, :, :] - 5*f_0[1, :, :] + 4*f_0[2, :, :] - f_0[3, :, :]
    ) / dk**2

    deriv2_k_1[0, :, :] = (
        2*f_1[0, :, :] - 5*f_1[1, :, :] + 4*f_1[2, :, :] - f_1[3, :, :]
    ) / dk**2

    # k = Nk-1
    deriv2_k_0[-1, :, :] = (
        2*f_0[-1, :, :] - 5*f_0[-2, :, :] + 4*f_0[-3, :, :] - f_0[-4, :, :]
    ) / dk**2

    deriv2_k_1[-1, :, :] = (
        2*f_1[-1, :, :] - 5*f_1[-2, :, :] + 4*f_1[-3, :, :] - f_1[-4, :, :]
    ) / dk**2

    # ============================================================
    # ∂/∂l  (axis 1)
    # ============================================================

    # bulk
    deriv_l_0[:, 1:-1, :] = (f_0[:, 2:, :] - f_0[:, :-2, :]) * (0.5 / dl)
    deriv_l_1[:, 1:-1, :] = (f_1[:, 2:, :] - f_1[:, :-2, :]) * (0.5 / dl)

    # l = 0
    # deriv_l_0[:, 0, :] = (
    #     -11*f_0[:, 0, :] + 18*f_0[:, 1, :] - 9*f_0[:, 2, :] + 2*f_0[:, 3, :]
    # ) / (6*dl)

    # deriv_l_1[:, 0, :] = (
    #     -11*f_1[:, 0, :] + 18*f_1[:, 1, :] - 9*f_1[:, 2, :] + 2*f_1[:, 3, :]
    # ) / (6*dl)

    # l = Nl-1
    deriv_l_0[:, -1, :] = (
        3*f_0[:, -1, :] - 4*f_0[:, -2, :] + f_0[:, -3, :]
    ) * (0.5 / dl)

    deriv_l_1[:, -1, :] = (
        3*f_1[:, -1, :] - 4*f_1[:, -2, :] + f_1[:, -3, :]
    ) * (0.5 / dl)

    # deriv_l_0[:, 0, :] = (
    # -3*f_0[:, 0, :] + 4*f_0[:, 1, :] - f_0[:, 2, :]
    # ) * (0.5 / dl)
    # deriv_l_1[:, 0, :] = (
    #     -3*f_1[:, 0, :] + 4*f_1[:, 1, :] - f_1[:, 2, :]
    # ) * (0.5 / dl)

    # ============================================================
    # ∂²/∂l²  (axis 1)
    # ============================================================

    # bulk
    deriv2_l_0[:, 1:-1, :] = (
        f_0[:, 2:, :] - 2*f_0[:, 1:-1, :] + f_0[:, :-2, :]
    ) / dl**2

    deriv2_l_1[:, 1:-1, :] = (
        f_1[:, 2:, :] - 2*f_1[:, 1:-1, :] + f_1[:, :-2, :]
    ) / dl**2

    # deriv2_l_0[:, 0, :] = (
    # 2*f_0[:, 0, :] - 5*f_0[:, 1, :] + 4*f_0[:, 2, :] - f_0[:, 3, :]
    # ) / dl**2
    # deriv2_l_1[:, 0, :] = (
    
    # 2*f_1[:, 0, :] - 5*f_1[:, 1, :] + 4*f_1[:, 2, :] - f_1[:, 3, :]
    # ) / dl**2

    # l = 0
    # deriv2_l_0[:, 0, :] = (
    #     -35*f_0[:, 0, :] + 104*f_0[:, 1, :] - 114*f_0[:, 2, :]
    #     + 56*f_0[:, 3, :] - 11*f_0[:, 4, :]
    # ) / (12*dl**2)

    # deriv2_l_1[:, 0, :] = (
    #     -35*f_1[:, 0, :] + 104*f_1[:, 1, :] - 114*f_1[:, 2, :]
    #     + 56*f_1[:, 3, :] - 11*f_1[:, 4, :]
    # ) / (12*dl**2)
    



    # l = Nl-1
    deriv2_l_0[:, -1, :] = (
        35*f_0[:, -1, :] - 104*f_0[:, -2, :] + 114*f_0[:, -3, :]
        - 56*f_0[:, -4, :] + 11*f_0[:, -5, :]
    ) / (12*dl**2)

    deriv2_l_1[:, -1, :] = (
        35*f_1[:, -1, :] - 104*f_1[:, -2, :] + 114*f_1[:, -3, :]
        - 56*f_1[:, -4, :] + 11*f_1[:, -5, :]
    ) / (12*dl**2)



    deriv2_psi_0 = xp.zeros_like(f_0)
    deriv2_psi_1 = xp.zeros_like(f_1)
    deriv2_psi_2 = xp.zeros_like(f_2)

    deriv2_psi_0[:,:,1:-1] = (f_0[:,:,2:] - 2*f_0[:,:,1:-1] + f_0[:,:,:-2]) / dpsi**2
    deriv2_psi_1[:,:,1:-1] = (f_1[:,:,2:] - 2*f_1[:,:,1:-1] + f_1[:,:,:-2]) / dpsi**2

    deriv2_psi_0[:,:,0]  = (2*f_0[:,:,1] - 2*f_0[:,:,0]) / dpsi**2
    deriv2_psi_1[:,:,0]  = (2*f_1[:,:,1] - 2*f_1[:,:,0]) / dpsi**2

    deriv2_psi_0[:,:,-1]  = (2*f_0[:,:,-2] - 2*f_0[:,:,-1]) / dpsi**2
    deriv2_psi_1[:,:,-1]  = (2*f_1[:,:,-2] - 2*f_1[:,:,-1]) / dpsi**2



    
    # Polar Laplacian (bulk)
    lap_lpsi_0 = deriv2_l_0 + (1/l) * deriv_l_0 + (1/l**2) * deriv2_psi_0
    lap_lpsi_1 = deriv2_l_1 + (1/l) * deriv_l_1  + (1/l**2) * deriv2_psi_1

    # Correct l = 0 limit
    #lap_lpsi_0[:,0,:] = 2 * deriv2_l_0[:,0,:]
    #lap_lpsi_1[:,0,:] = 2 * deriv2_l_1[:,0,:]
    lap_lpsi_0[:, 0, :] = laplacian_at_origin(f_0, dl, order=2)[:, None]
    lap_lpsi_1[:, 0, :] = laplacian_at_origin(f_1, dl, order=2)[:, None]


    if(sys.Ncmode == "FNc"):
        # repeat the calculation of all derivatives for f_2
        deriv_k_2[1:-1, :, :] = (f_2[2:, :, :] - f_2[:-2, :, :]) * (0.5 / dk)
        deriv_k_2[0, :, :] = (-3*f_2[0, :, :] + 4*f_2[1, :, :] - f_2[2, :, :]) * (0.5 / dk) 
        deriv_k_2[-1, :, :] = (3*f_2[-1, :, :] - 4*f_2[-2, :, :] + f_2[-3, :, :]) * (0.5 / dk)  
        deriv2_k_2[1:-1, :, :] = (f_2[2:, :, :] - 2*f_2[1:-1, :, :] + f_2[:-2, :, :]) / dk**2
        deriv2_k_2[0, :, :] = (2*f_2[0, :, :] - 5*f_2[1, :, :] + 4*f_2[2, :, :] - f_2[3, :, :]) / dk**2
        deriv2_k_2[-1, :, :] = (2*f_2[-1, :, :] - 5*f_2[-2, :, :] + 4*f_2[-3, :, :] - f_2[-4, :, :]) / dk**2
        deriv_l_2[:, 1:-1, :] = (f_2[:, 2:, :] - f_2[:, :-2, :]) * (0.5 / dl)
        deriv_l_2[:, 0, :] = (-11*f_2[:, 0, :] + 18*f_2[:, 1, :] - 9*f_2[:, 2, :] + 2*f_2[:, 3, :]) / (6*dl)
        deriv_l_2[:, -1, :] = (3*f_2[:, -1, :] - 4*f_2[:, -2, :] + f_2[:, -3, :]) * (0.5 / dl)
        deriv2_l_2[:, 1:-1, :] = (f_2[:, 2:, :] - 2*f_2[:, 1:-1, :] + f_2[:, :-2, :]) / dl**2
        deriv2_l_2[:, 0, :] = (-35*f_2[:, 0, :] + 104*f_2[:, 1, :] - 114*f_2[:, 2, :] + 56*f_2[:, 3, :] - 11*f_2[:, 4, :]) / (12*dl**2)
        deriv2_l_2[:, -1, :] = (35*f_2[:, -1, :] - 104*f_2[:, -2, :] + 114*f_2[:, -3, :] - 56*f_2[:, -4, :] + 11*f_2[:, -5, :]) / (12*dl**2)

        #psi derivs and laplacians
        deriv2_psi_2[:,:,1:-1] = (f_2[:,:,2:] - 2*f_2[:,:,1:-1] + f_2[:,:,:-2]) / dpsi**2
        deriv2_psi_2[:,:,0]  = (2*f_2[:,:,1] - 2*f_2[:,:,0]) / dpsi**2
        deriv2_psi_2[:,:,-1]  = (2*f_2[:,:,-2] - 2*f_2[:,:,-1]) / dpsi**2
        lap_lpsi_2 = deriv2_l_2 + (1/l) * deriv_l_2 + (1/l**2) * deriv2_psi_2
        lap_lpsi_2[:,0,:] = laplacian_at_origin(f_2, dl, order=2)[:, None]


    kin_term_1 = 1 / (omega) * (k * l * xp.cos(psi)) * f_1
    kin_term_0 = 1 / (omega) * (k * l * xp.cos(psi)) * f_0
    kin_term_2 = 1 / (omega) * (k * l * xp.cos(psi)) * f_2

    z_corr = 1 - sys.z
    if sys.Ncmode == "FNc":
        Nc = 3
        
        q_4 = 0.25 * sys.qtilde


        V_term_00 = (

            (0.25 * Nc * (1 + 2 * z_corr**2)) * (deriv2_k_0 + 1/k * deriv_k_0 ) 
            + Nc * (lap_lpsi_0) * 0.25
        ) #O(1)

        #V_term01 = 0.0

        V_term02 = (z_corr - 1) * z_corr * (deriv2_k_2 + 1/k * deriv_k_2)  #O(1/Nc²)


        V_term10 =  (-z_corr + 1) * z_corr * (deriv2_k_0 + 1/k * deriv_k_0) #O(1)

        V_term11 = (-(((-1 + z_corr)**2) / (2 * Nc)) + 0.5 * Nc * (1 + z_corr * (-2 + 3 * z_corr)))  \
        * (deriv2_k_1 + 1/k * deriv_k_1)  #O(1)

        #V_term12 = 0.0

        V_term20 =  (z_corr - 1) * z_corr * (deriv2_k_0 + 1/k * deriv_k_0) # O(1/Nc)

        V_term21 = (1/2 * Nc) * (
            (-1 - 2 * (-1 + z_corr) * z_corr) * (deriv2_k_1 + 1/k * deriv_k_1)
            + (lap_lpsi_1)
        ) # O(1)
        
        V_term22 = (
            (-(((-1 + z_corr)**2) / (2 * Nc)) + 0.25 * Nc * (1 - 4 * z_corr + 6 * z_corr**2)) * (deriv2_k_2 + 1/k * deriv_k_2)
            + (Nc * (lap_lpsi_2)) * 0.25
        ) # O(1/Nc)
        

        HF_0 = kin_term_0 +  1j * q_4 *(V_term_00  + V_term02 * Nc**2 / (Nc**3 - Nc))
        HF_1 = kin_term_1 +  1j * q_4 * (V_term10 * (Nc**3 - Nc)/(Nc**2 - 1) + V_term11)
        HF_2 = kin_term_2 +  1j * q_4 * (V_term20 * (Nc**3 - Nc)/ Nc**2 + V_term21 * (Nc**2 - 1)/Nc**2 + V_term22)


    elif sys.Ncmode == "LeadNc":
        Nc = 3
        q_4 = 0.25 * sys.qtilde

        V_term_00 = (
            (
             + 0.25 * Nc * (1 + 2 * sys.z**2)
            ) * (deriv2_k_0 + 1/k * deriv_k_0)

            + Nc * (deriv2_l_0 + 1/(l + epsl) * deriv_l_0 + 1/(l + epsl)**2 * deriv2_psi_0) * 0.25
        ) #O(1)

        

        V_term10 =  (-sys.z + 1) * sys.z * (deriv2_k_0 + 1/k * deriv_k_0) #O(1)

        V_term11 = (-(((-1 + sys.z)**2) / (2 * Nc)) + 0.5 * Nc * (1 + sys.z * (-2 + 3 * sys.z))) \
                    * (deriv2_k_1 + 1/k * deriv_k_1)  #O(1)



        HF_0 = kin_term_0 + 1j * q_4 *(V_term_00)
        HF_1 = kin_term_1 + 1j * q_4 * (V_term10 * (Nc**3 - Nc)/(Nc**2 - 1) + V_term11)
        HF_2 = kin_term_2 

    elif sys.Ncmode == "LNc_qeff":
        Nc = 3
        CA = Nc
        CF = (Nc**2 -1)/(2*Nc)
        qeff = ((1-sys.z) * CA + sys.z**2 * CF) * sys.qtilde

        V_term_00 = ((1 + 2*sys.z**2) / (4 - 4*sys.z + 2*sys.z**2)) * (deriv2_k_0 + 1/k * deriv_k_0 \
                                                    + deriv2_l_0 + 1/(l + epsl) * deriv_l_0 + 1/(l + epsl)**2 * deriv2_psi_0)
        
        V_term_10 = 2 * sys.z * (1 - sys.z) / (2 - 2*sys.z + sys.z**2) * (deriv2_k_0 + 1/k * deriv_k_0)

        V_term_11 = (1 + sys.z * (-2 + 3 * sys.z)) / (2 - 2*sys.z + sys.z**2) * (deriv2_k_1 + 1/k * deriv_k_1)
    

        HF_0 = kin_term_0 + 1j * 0.25 * qeff * V_term_00
        HF_1 = kin_term_1 + 1j * 0.25 * qeff * (V_term_10 + V_term_11)
        HF_2 = kin_term_2

    elif sys.Ncmode == "LNc":
        Nc = 3
        q_4 = 0.25 * sys.qtilde


        V_term_00 = (
            (0.25 * Nc * (3 - 4*sys.z + 2*sys.z**2))
            ) * (deriv2_k_0 + 1/k * deriv_k_0) + Nc * (lap_lpsi_0) * 0.25

        #V_term01 = 0.0

        V_term02 = 0.0


        V_term10 =  sys.z * (1-sys.z) * (deriv2_k_0 + 1/k * deriv_k_0)

        V_term11 = Nc * ( 1 - 2*sys.z + 1.5*sys.z**2
        ) * (deriv2_k_1 + 1/k * deriv_k_1)


        

        HF_0 = kin_term_0 +  1j * q_4 *(V_term_00)
        HF_1 = kin_term_1 +  1j * q_4 * (V_term10 * Nc + V_term11)
        HF_2 = kin_term_2 

    
    elif sys.Ncmode == "LNcFac":

        Nc = 3
        q_4 = sys.qtilde * 0.25
        g_z = 1 + sys.z * (-2 + 3 * sys.z)
            
        V_term_11 = Nc/2 * (g_z  * (deriv2_k_1 + 1/k * deriv_k_1))


        
        HF_1 = kin_term_1 + 1j * q_4 * (V_term_11)
        HF_0 = kin_term_0 + 0.0 * HF_1 
        HF_2 = kin_term_2 + 0.0 * HF_1


    
    return xp.array([HF_0, HF_1, HF_2])


def Hamiltonian3D_ggg(sys, f):
    # this one has 8 components
    dk = sys.dk
    dl = sys.dl
    dpsi = sys.dpsi
    k = sys.K[:, None, None]
    l = sys.L[None, :, None]
    psi = sys.psi[None, None, :]

    omega = sys.omega
    f_list = [f[i] for i in range(6)]

    if sys.Ncmode == "FNc":
        nsig = 6
    elif sys.Ncmode in ("LNc", "LNcFac", "LeadNc"):
        nsig = 2
    else:
        nsig = 6

    xp = cp if sys.optimization == "gpu" else np

    # ensure l=0 slice is averaged over psi like other functions
    for i in range(nsig):
        f_list[i][:, 0, :] = xp.mean(f_list[i][:, 0, :], axis=-1, keepdims=True)

    # allocate derivative arrays
    deriv_k = [xp.zeros_like(f_list[0]) for _ in range(nsig)]
    deriv_l = [xp.zeros_like(f_list[0]) for _ in range(nsig)]
    deriv2_k = [xp.zeros_like(f_list[0]) for _ in range(nsig)]
    deriv2_l = [xp.zeros_like(f_list[0]) for _ in range(nsig)]
    deriv2_psi = [xp.zeros_like(f_list[0]) for _ in range(nsig)]
    lap_lpsi = [xp.zeros_like(f_list[0]) for _ in range(nsig)]

    # compute k-derivatives (axis 0) and k-second derivatives
    for i in range(nsig):
        fi = f_list[i]
        # bulk central
        deriv_k[i][1:-1, :, :] = (fi[2:, :, :] - fi[:-2, :, :]) * (0.5 / dk)
        # k = 0 (one-sided)
        deriv_k[i][0, :, :] = (-3*fi[0, :, :] + 4*fi[1, :, :] - fi[2, :, :]) * (0.5 / dk)
        # k = Nk-1 (one-sided)
        deriv_k[i][-1, :, :] = (3*fi[-1, :, :] - 4*fi[-2, :, :] + fi[-3, :, :]) * (0.5 / dk)

        # second deriv bulk
        deriv2_k[i][1:-1, :, :] = (fi[2:, :, :] - 2*fi[1:-1, :, :] + fi[:-2, :, :]) / dk**2
        # k = 0
        deriv2_k[i][0, :, :] = (2*fi[0, :, :] - 5*fi[1, :, :] + 4*fi[2, :, :] - fi[3, :, :]) / dk**2
        # k = Nk-1
        deriv2_k[i][-1, :, :] = (2*fi[-1, :, :] - 5*fi[-2, :, :] + 4*fi[-3, :, :] - fi[-4, :, :]) / dk**2

    # compute l-derivatives (axis 1) and l-second derivatives
    for i in range(nsig):
        fi = f_list[i]
        # bulk central
        deriv_l[i][:, 1:-1, :] = (fi[:, 2:, :] - fi[:, :-2, :]) * (0.5 / dl)
        # l = 0 (one-sided high-order)
        deriv_l[i][:, 0, :] = (-11*fi[:, 0, :] + 18*fi[:, 1, :] - 9*fi[:, 2, :] + 2*fi[:, 3, :]) / (6*dl)
        # l = Nl-1 (one-sided)
        deriv_l[i][:, -1, :] = (3*fi[:, -1, :] - 4*fi[:, -2, :] + fi[:, -3, :]) * (0.5 / dl)

        # second deriv bulk
        deriv2_l[i][:, 1:-1, :] = (fi[:, 2:, :] - 2*fi[:, 1:-1, :] + fi[:, :-2, :]) / dl**2
        # l = 0 (high-order)
        deriv2_l[i][:, 0, :] = (-35*fi[:, 0, :] + 104*fi[:, 1, :] - 114*fi[:, 2, :]
                                 + 56*fi[:, 3, :] - 11*fi[:, 4, :]) / (12*dl**2)
        # l = Nl-1
        deriv2_l[i][:, -1, :] = (35*fi[:, -1, :] - 104*fi[:, -2, :] + 114*fi[:, -3, :]
                                  - 56*fi[:, -4, :] + 11*fi[:, -5, :]) / (12*dl**2)

    # psi second derivatives (axis 2), use same treatment as other functions (non-periodic endpoints)
    for i in range(nsig):
        fi = f_list[i]
        deriv2_psi[i][:, :, 1:-1] = (fi[:, :, 2:] - 2*fi[:, :, 1:-1] + fi[:, :, :-2]) / dpsi**2
        deriv2_psi[i][:, :, 0] = (2*fi[:, :, 1] - 2*fi[:, :, 0]) / dpsi**2
        deriv2_psi[i][:, :, -1] = (2*fi[:, :, -2] - 2*fi[:, :, -1]) / dpsi**2

        # polar laplacian
        lap_lpsi[i] = deriv2_l[i] + (1.0 / l) * deriv_l[i] + (1.0 / l**2) * deriv2_psi[i]
        # correct l=0 limit
        lap_lpsi[i][:, 0, :] = 2 * laplacian_at_origin(fi, dl, order=2)[:, None]

    # kinetic terms
    kin_term = [1.0 / (omega) * (k * l * xp.cos(psi)) * f_list[i] for i in range(6)]

    # Now assemble HF depending on color mode, using the same V expressions as before
    HF = [kin_term[i] for i in range(6)]

    if sys.Ncmode == "FNc":
        Nc = 3
        q_4 = 0.25 * sys.qtilde
        z = sys.z

        C0Ncfac = Nc**2 * (Nc**2 - 1)
        C1Ncfac = Nc * (Nc**2 - 1)
        C2Ncfac = -2*Nc*(Nc**2 - 1)
        C3Ncfac = 1 # no rescaling here; source term set to 0
        C4Ncfac = Nc * (Nc**2 - 1)
        C5Ncfac = -C4Ncfac

        V0 = (
            -2.0 * (-1 + z) * z * (deriv2_k[1] + 1/k * deriv_k[1]) * C1Ncfac/C0Ncfac
            - 1.0 * (-1 + z) * z * (deriv2_k[2] + 1/k * deriv_k[2]) * C2Ncfac/C0Ncfac
            + (Nc * (3.0/4.0 - z + z**2) * deriv2_k[0] + (Nc * lap_lpsi[0]) * 0.25) 
        )

        V1 = (
            -(deriv2_k[0] + 1/k * deriv_k[0]) * (-1 + z) * z * C0Ncfac/C1Ncfac
            + Nc * (1 - 2*z + 2*z**2) * (deriv2_k[1] + 1/k * deriv_k[1])
        )

  
        V2 = (
             C0Ncfac/C2Ncfac * (0.5 * (deriv2_k[0] + 1/k * deriv_k[0]) - 0.5 * (lap_lpsi[0]))
            + C3Ncfac/C2Ncfac * ((-1 + 2*z - 2*z**2) * (deriv2_k[3] + 1/k * deriv_k[3]) + lap_lpsi[3])
            + C4Ncfac/C2Ncfac * (1.5 * Nc * (deriv2_k[4] + 1/k * deriv_k[4]) - 1.5 * Nc * lap_lpsi[4])
            + C1Ncfac/C2Ncfac * (0.5 * Nc * (1 - 2*z)**2 * (deriv2_k[1] + 1/k * deriv_k[1]) - 0.5 * Nc * lap_lpsi[1])
            + (Nc * (3.0/4.0 - 2*z + 2*z**2) * (deriv2_k[2] + 1/k * deriv_k[2]) + 0.25 * Nc * lap_lpsi[2])
        )
        

        # New V3 according to user's formula (C2->comp0, C1->comp1, C3->comp2,
        # C4->comp3, C5->comp4, C6->comp5). Use laplacians in k and l (lap_lpsi).

        V3 = (
            ((-(1.0/4.0) + z - z**2) * (deriv2_k[2] + 1.0/k * deriv_k[2])
            + 0.25 * lap_lpsi[2]) * C2Ncfac / C3Ncfac

            + ((-(1.0/2.0) * (1 - 2*z)**2) * (deriv2_k[1] + 1.0/k * deriv_k[1])
            + 0.5 * lap_lpsi[1]) * C1Ncfac / C3Ncfac

            + (0.5 * Nc * (deriv2_k[0] + 1.0/k * deriv_k[0])
            - 0.5 * Nc * lap_lpsi[0]) * C0Ncfac / C3Ncfac

            + Nc * (0.25 - 2*z + 2*z**2) * (deriv2_k[3] + 1.0/k * deriv_k[3])
            + 0.75 * Nc * lap_lpsi[3]

            + (Nc**2 * (0.5 + 4*z - 4*z**2) * (deriv2_k[4] + 1.0/k * deriv_k[4])
            - 0.5 * Nc**2 * lap_lpsi[4]) * C4Ncfac / C3Ncfac

            + (Nc**2 * (0.25 + 4*z - 4*z**2) * (deriv2_k[5] + 1.0/k * deriv_k[5])
            - 0.25 * Nc**2 * lap_lpsi[5]) * C5Ncfac / C3Ncfac
        )

    
        V4 = (
            ((-1 + z) * z * (deriv2_k[3] + 1.0 / k * deriv_k[3])) * C3Ncfac / C4Ncfac

            - (3.0 * Nc * (-1 + z) * z * (deriv2_k[5] + 1.0 / k * deriv_k[5])) * C5Ncfac / C4Ncfac

            + (0.25 * (deriv2_k[0] + 1.0 / k * deriv_k[0]) - 0.25 * lap_lpsi[0]) * C0Ncfac / C4Ncfac

            + ((Nc * (0.5 + 2.0 * z - 2.0 * z**2) * (deriv2_k[4] + 1.0 / k * deriv_k[4])
               + 0.5 * Nc * lap_lpsi[4])) 
        )

        V5 = (
            (
            (-(1.0/4.0) + z - z**2) * (deriv2_k[3] + 1.0 / k * deriv_k[3])
            + 0.25 * (lap_lpsi[3])
            ) * C3Ncfac / C5Ncfac

            + (
            0.5 * Nc * (1 - 2*z)**2 * (deriv2_k[4] + 1.0 / k * deriv_k[4])
            - 0.5 * Nc * (lap_lpsi[4])
            ) * (C4Ncfac / C5Ncfac)

            + (
            0.75 * Nc * (1 - 2*z)**2 * (deriv2_k[5] + 1.0 / k * deriv_k[5])
            + 0.25 * Nc * (lap_lpsi[5])
            )
            
        )



        HF[0] += 1j * q_4 * V0
        HF[1] += 1j * q_4 * V1
        HF[2] += 1j * q_4 * V2
        HF[3] += 1j * q_4 * V3
        HF[4] += 1j * q_4 * V4
        HF[5] += 1j * q_4 * V5


        return xp.array(HF)

    elif sys.Ncmode in ("LNc", "LeadNc"):
        Nc = 3
        q_4 = 0.25 * sys.qtilde
        z = sys.z

        V_term_00 = (
            (Nc * (3/4 - sys.z + sys.z**2))
            ) * (deriv2_k[0] + 1/k * deriv_k[0]) + Nc * (lap_lpsi[0]) * 0.25

        #V_term01 = 0.0


        V_term10 =  sys.z * (1-sys.z) * (deriv2_k[0] + 1/k * deriv_k[0])

        V_term11 = Nc * (1 - 2*sys.z + 2*sys.z**2
        ) * (deriv2_k[1] + 1/k * deriv_k[1])


        HF[0] = HF[0] +  1j * q_4 *(V_term_00)
        HF[1] = HF[1] +  1j * q_4 * (V_term10 * Nc + V_term11)

        # append irrelevant components as zero arrays consistent with shape
        return xp.array(HF)

    elif sys.Ncmode == "LNcFac":
        Nc = 3
        q_4 = sys.qtilde * 0.25

        V_term_11 = Nc * (1 - 2*sys.z + 2*sys.z**2
        ) * (deriv2_k[1] + 1/k * deriv_k[1])

        HF[1] += 1j * q_4 * (V_term_11)

        return xp.array(HF)

    else:
        # default: return kinetic only for all components
        return xp.array(HF)


def apply_hamil_3D(sis, f):

    if sis.vertex == "gamma_qq":
        Hf = Hamitlonian3D_gammaqq(sis, f)

    elif sis.vertex == "q_qg":
        Hf = Hamiltonian3D_qqg(sis, f)

    elif sis.vertex == "g_gg":
        Hf = Hamiltonian3D_ggg(sis, f)

    else:
        raise TypeError ("Unknown vertex type specified.")
    
    return Hf










