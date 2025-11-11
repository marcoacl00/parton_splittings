from .system import *


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
    epsl = dl  # to avoid division by zero

    omega = sys.omega
    f_0 = f[0]
    f_1 = f[1]


    # Second derivatives in momentum space (bulk only)

    xp = cp if sys.optimization == "gpu" else np

    f_0_padded = xp.pad(f_0, ((1, 1), (1, 1), (1, 1)), mode='edge')  # Pad with ghost cells
    f_1_padded = xp.pad(f_1, ((1, 1), (1, 1), (1, 1)), mode='edge')  # Pad with ghost cells
    

    

    # required first derivatives

    deriv_k_0 = (f_0_padded[2:, 1:-1, 1:-1] - f_0_padded[:-2, 1:-1, 1:-1]) / (2 * dk)
    deriv_k_1 = (f_1_padded[2:, 1:-1, 1:-1] - f_1_padded[:-2, 1:-1, 1:-1]) / (2 * dk)
    deriv_l_0 = (f_0_padded[1:-1, 2:, 1:-1] - f_0_padded[1:-1, :-2, 1:-1]) / (2 * dl)
    deriv_l_1 = (f_1_padded[1:-1, 2:, 1:-1] - f_1_padded[1:-1, :-2, 1:-1]) / (2 * dl)

    boundary_correction = False
    if boundary_correction:
        deriv_k_0[0, :, :]  = (-11*f_0[0, :, :] + 18*f_0[1, :, :] - 9*f_0[2, :, :] + 2*f_0[3, :, :]) / (6 * dk)
        deriv_k_0[-1, :, :] = (11*f_0[-1, :, :] - 18*f_0[-2, :, :] + 9*f_0[-3, :, :] - 2*f_0[-4, :, :]) / (6 * dk)

        deriv_k_1[0, :, :]  = (-11*f_1[0, :, :] + 18*f_1[1, :, :] - 9*f_1[2, :, :] + 2*f_1[3, :, :]) / (6 * dk)
        deriv_k_1[-1, :, :] = (11*f_1[-1, :, :] - 18*f_1[-2, :, :] + 9*f_1[-3, :, :] - 2*f_1[-4, :, :]) / (6 * dk)

        deriv_l_0[:, 0, :]  = (-11*f_0[:, 0, :] + 18*f_0[:, 1, :] - 9*f_0[:, 2, :] + 2*f_0[:, 3, :]) / (6 * dl)
        deriv_l_0[:, -1, :] = ( 11*f_0[:, -1, :] - 18*f_0[:, -2, :] + 9*f_0[:, -3, :] - 2*f_0[:, -4, :]) / (6 * dl)

        deriv_l_1[:, 0, :]  = (-11*f_1[:, 0, :] + 18*f_1[:, 1, :] - 9*f_1[:, 2, :] + 2*f_1[:, 3, :]) / (6 * dl)
        deriv_l_1[:, -1, :] = ( 11*f_1[:, -1, :] - 18*f_1[:, -2, :] + 9*f_1[:, -3, :] - 2*f_1[:, -4, :]) / (6 * dl)


    #required second derivatives
    deriv2_k_0 = (f_0_padded[2:, 1:-1, 1:-1] - 2 * f_0_padded[1:-1, 1:-1, 1:-1] + f_0_padded[:-2, 1:-1, 1:-1]) / dk**2
    deriv2_k_1 = (f_1_padded[2:, 1:-1, 1:-1] - 2 * f_1_padded[1:-1, 1:-1, 1:-1] + f_1_padded[:-2, 1:-1, 1:-1]) / dk**2
    deriv2_l_0 = (f_0_padded[1:-1, 2:, 1:-1] - 2 * f_0_padded[1:-1, 1:-1, 1:-1] + f_0_padded[1:-1, :-2, 1:-1]) / dl**2
    deriv2_l_1 = (f_1_padded[1:-1, 2:, 1:-1] - 2 * f_1_padded[1:-1, 1:-1, 1:-1] + f_1_padded[1:-1, :-2, 1:-1]) / dl**2

    if boundary_correction:
        deriv2_k_0[0, :, :]  = (35*f_0[0, :, :] - 104*f_0[1, :, :] + 114*f_0[2, :, :] - 56*f_0[3, :, :] + 11*f_0[4, :, :]) / (12 * dk**2)
        deriv2_k_0[-1, :, :] = (35*f_0[-1, :, :] - 104*f_0[-2, :, :] + 114*f_0[-3, :, :] - 56*f_0[-4, :, :] + 11*f_0[-5, :, :]) / (12 * dk**2)
        deriv2_k_1[0, :, :]  = (35*f_1[0, :, :] - 104*f_1[1, :, :] + 114*f_1[2, :, :] - 56*f_1[3, :, :] + 11*f_1[4, :, :]) / (12 * dk**2)
        deriv2_k_1[-1, :, :] = (35*f_1[-1, :, :] - 104*f_1[-2, :, :] + 114*f_1[-3, :, :] - 56*f_1[-4, :, :] + 11*f_1[-5, :, :]) / (12 * dk**2)
        deriv2_l_0[:, 0, :]  = (35*f_0[:, 0, :] - 104*f_0[:, 1, :] + 114*f_0[:, 2, :] - 56*f_0[:, 3, :] + 11*f_0[:, 4, :]) / (12 * dl**2)
        deriv2_l_0[:, -1, :] = (35*f_0[:, -1, :] - 104*f_0[:, -2, :] + 114*f_0[:, -3, :] - 56*f_0[:, -4, :] + 11*f_0[:, -5, :]) / (12 * dl**2)
        deriv2_l_1[:, 0, :]  = (35*f_1[:, 0, :] - 104*f_1[:, 1, :] + 114*f_1[:, 2, :] - 56*f_1[:, 3, :] + 11*f_1[:, 4, :]) / (12 * dl**2)
        deriv2_l_1[:, -1, :] = (35*f_1[:, -1, :] - 104*f_1[:, -2, :] + 114*f_1[:, -3, :] - 56*f_1[:, -4, :] + 11*f_1[:, -5, :]) / (12 * dl**2)


    # Compute second derivative in psi direction (axis=2) with periodic boundary
    deriv2_psi_0 = xp.zeros_like(f_0)
    deriv2_psi_1 = xp.zeros_like(f_1)

    # Bulk (central difference)
    deriv2_psi_0[:, :, 1:-1] = (f_0[:, :, 2:] - 2*f_0[:, :, 1:-1] + f_0[:, :, :-2]) / dpsi**2
    deriv2_psi_1[:, :, 1:-1] = (f_1[:, :, 2:] - 2*f_1[:, :, 1:-1] + f_1[:, :, :-2]) / dpsi**2

    # Periodic boundaries
    deriv2_psi_0[:, :, 0]  = (f_0[:, :, 1] - 2*f_0[:, :, 0] + f_0[:, :, -2]) / dpsi**2
    deriv2_psi_0[:, :, -1] = deriv2_psi_0[:, :, 0]

    deriv2_psi_1[:, :, 0]  = (f_1[:, :, 1] - 2*f_1[:, :, 0] + f_1[:, :, -2]) / dpsi**2
    deriv2_psi_1[:, :, -1] = deriv2_psi_1[:, :, 0]

    
    # kinetic term
    kin_term_1 = 2 / (omega) * (k * l * xp.cos(psi)) * f_1
    kin_term_0 = 2 / (omega) * (k * l * xp.cos(psi)) * f_0


    if sys.Ncmode == "LNcFac" or sys.Ncmode == "LNc" or sys.NcMode == "LeadNc":
        

        CF = 3/2
        if sys.Ncmode == "LeadNc":
            CF = (3**2 - 1) / (2 * 3)


        q_4 = sys.qtilde * 0.25 
        
        
        V_term_00 = 0.5 * (deriv2_k_0 + 1/k * deriv_k_0 + deriv2_l_0 
                           + 1/(l + epsl) * deriv_l_0 + 1 / (l+epsl)**2 * deriv2_psi_0)
            
        g_z = sys.z**2 + (1 - sys.z)**2
        V_term_11 = g_z  * (deriv2_k_1 + deriv_k_1 * 1/(k))

        V_1 = V_term_11

        if sys.Ncmode == "LNc":
            V_term_10 =  2 * sys.z * (1 - sys.z) * (deriv2_k_0 + 1/k * deriv_k_0)
            V_1 += V_term_10

        
        HF_0 = kin_term_0 + 1j * q_4 * CF * V_term_00 
        HF_1 = kin_term_1 + 1j * q_4 * CF * V_1

    elif sys.Ncmode == "FNc":
        Nc = 3
        CF = (Nc**2 - 1) / (2 * Nc)

        q_4 = 0.25 * sys.qtilde

        fz =  sys.z * (1 - sys.z)

        V_term_00 = (0.5 * CF * ((deriv2_k_0 + 1/k * deriv_k_0)
                                  + (deriv2_l_0 + 1/(l + epsl) * deriv_l_0
                                  + 1 / (l+epsl)**2 * deriv2_psi_0))
                                   
                              + 1/(2*Nc) * (deriv2_k_0 + 1/k * deriv_k_0
                                            - (deriv2_l_0 + 1/(l + epsl) * deriv_l_0)
                                         - 1 / (l+epsl)**2 * deriv2_psi_0))
        
        V_term01 = -1/(2*Nc) * (deriv2_k_1 + 1/k * deriv_k_1
                                      - (deriv2_l_1 + 1/(l + epsl) * deriv_l_1)
                                      - 1 / (l+epsl)**2 * deriv2_psi_1)
        
        V_term10 =  Nc * fz  * (deriv2_k_0 + 1/k * deriv_k_0)

        V_term11 =  (CF  - Nc * fz)  * (deriv2_k_1 + 1/k * deriv_k_1)

        HF_0 = kin_term_0 + 1j * q_4 *(V_term_00 + V_term01)
        HF_1 = kin_term_1 + 1j * q_4 * (V_term10 + V_term11)
    
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

    f_0_padded = xp.pad(f_0, ((1, 1), (1, 1), (1, 1)), mode='edge')  # Pad with ghost cells
    f_1_padded = xp.pad(f_1, ((1, 1), (1, 1), (1, 1)), mode='edge')  # Pad with ghost cells
    f_2_padded = xp.pad(f_2, ((1, 1), (1, 1), (1, 1)), mode='edge')  # Pad with ghost cells

    # required first derivatives

    deriv_k_0 = (f_0_padded[2:, 1:-1, 1:-1] - f_0_padded[:-2, 1:-1, 1:-1]) / (2 * dk)
    deriv_k_1 = (f_1_padded[2:, 1:-1, 1:-1] - f_1_padded[:-2, 1:-1, 1:-1]) / (2 * dk)
    deriv_k_2 = (f_2_padded [2:, 1:-1, 1:-1] - f_2_padded[:-2, 1:-1, 1:-1]) / (2 * dk)
    deriv_l_0 = (f_0_padded[1:-1, 2:, 1:-1] - f_0_padded[1:-1, :-2, 1:-1]) / (2 * dl)
    deriv_l_1 = (f_1_padded[1:-1, 2:, 1:-1] - f_1_padded[1:-1, :-2, 1:-1]) / (2 * dl)   
    deriv_l_2 = (f_2_padded[1:-1, 2:, 1:-1] - f_2_padded[1:-1, :-2, 1:-1]) / (2 * dl)
    
    boundary_correction = False
    # apply one-sideded boundary conditions 
    if boundary_correction:
        deriv_k_0[0, :, :]  = (-11*f_0[0, :, :] + 18*f_0[1, :, :] - 9*f_0[2, :, :] + 2*f_0[3, :, :]) / (6 * dk)
        deriv_k_0[-1, :, :] = (11*f_0[-1, :, :] - 18*f_0[-2, :, :] + 9*f_0[-3, :, :] - 2*f_0[-4, :, :]) / (6 * dk)
        deriv_k_1[0, :, :]  = (-11*f_1[0, :, :] + 18*f_1[1, :, :] - 9*f_1[2, :, :] + 2*f_1[3, :, :]) / (6 * dk)
        deriv_k_1[-1, :, :] = (11*f_1[-1, :, :] - 18*f_1[-2, :, :] + 9*f_1[-3, :, :] - 2*f_1[-4, :, :]) / (6 * dk)
        deriv_k_2[0, :, :]  = (-11*f_2[0, :, :] + 18*f_2[1, :, :] - 9*f_2[2, :, :] + 2*f_2[3, :, :]) / (6 * dk)
        deriv_k_2[-1, :, :] = (11*f_2[-1, :, :] - 18*f_2[-2, :, :] + 9*f_2[-3, :, :] - 2*f_2[-4, :, :]) / (6 * dk)
        deriv_l_0[:, 0, :]  = (-11*f_0[:, 0, :] + 18*f_0[:, 1, :] - 9*f_0[:, 2, :] + 2*f_0[:, 3, :]) / (6 * dl)
        deriv_l_0[:, -1, :] = ( 11*f_0[:, -1, :] - 18*f_0[:, -2, :] + 9*f_0[:, -3, :] - 2*f_0[:, -4, :]) / (6 * dl)
        deriv_l_1[:, 0, :]  = (-11*f_1[:, 0, :] + 18*f_1[:, 1, :] - 9*f_1[:, 2, :] + 2*f_1[:, 3, :]) / (6 * dl)
        deriv_l_1[:, -1, :] = ( 11*f_1[:, -1, :] - 18*f_1[:, -2, :] + 9*f_1[:, -3, :] - 2*f_1[:, -4, :]) / (6 * dl)
        deriv_l_2[:, 0, :]  = (-11*f_2[:, 0, :] + 18*f_2[:, 1, :] - 9*f_2[:, 2, :] + 2*f_2[:, 3, :]) / (6 * dl)
        deriv_l_2[:, -1, :] = ( 11*f_2[:, -1, :] - 18*f_2[:, -2, :] + 9*f_2[:, -3, :] - 2*f_2[:, -4, :]) / (6 * dl)

    #required second derivatives
    deriv2_k_0 = (f_0_padded[2:, 1:-1, 1:-1] - 2 * f_0_padded[1:-1, 1:-1, 1:-1] + f_0_padded[:-2, 1:-1, 1:-1]) / dk**2
    deriv2_k_1 = (f_1_padded[2:, 1:-1, 1:-1] - 2 * f_1_padded[1:-1, 1:-1, 1:-1] + f_1_padded[:-2, 1:-1, 1:-1]) / dk**2
    deriv2_k_2 = (f_2_padded[2:, 1:-1, 1:-1] - 2 * f_2_padded[1:-1, 1:-1, 1:-1] + f_2_padded[:-2, 1:-1, 1:-1]) / dk**2
    deriv2_l_0 = (f_0_padded[1:-1, 2:, 1:-1] - 2 * f_0_padded[1:-1, 1:-1, 1:-1] + f_0_padded[1:-1, :-2, 1:-1]) / dl**2
    deriv2_l_1 = (f_1_padded[1:-1, 2:, 1:-1] - 2 * f_1_padded[1:-1, 1:-1, 1:-1] + f_1_padded[1:-1, :-2, 1:-1]) / dl**2
    deriv2_l_2 = (f_2_padded[1:-1, 2:, 1:-1] - 2 * f_2_padded[1:-1, 1:-1, 1:-1] + f_2_padded[1:-1, :-2, 1:-1]) / dl**2

    # apply one-sideded boundary conditions
    if boundary_correction:
        deriv2_k_0[0, :, :]  = (35*f_0[0, :, :] - 104*f_0[1, :, :] + 114*f_0[2, :, :] - 56*f_0[3, :, :] + 11*f_0[4, :, :]) / (12 * dk**2)
        deriv2_k_0[-1, :, :] = (35*f_0[-1, :, :] - 104*f_0[-2, :, :] + 114*f_0[-3, :, :] - 56*f_0[-4, :, :] + 11*f_0[-5, :, :]) / (12 * dk**2)
        deriv2_k_1[0, :, :]  = (35*f_1[0, :, :] - 104*f_1[1, :, :] + 114*f_1[2, :, :] - 56*f_1[3, :, :] + 11*f_1[4, :, :]) / (12 * dk**2)
        deriv2_k_1[-1, :, :] = (35*f_1[-1, :, :] - 104*f_1[-2, :, :] + 114*f_1[-3, :, :] - 56*f_1[-4, :, :] + 11*f_1[-5, :, :]) / (12 * dk**2)
        deriv2_k_2[0, :, :]  = (35*f_2[0, :, :] - 104*f_2[1, :, :] + 114*f_2[2, :, :] - 56*f_2[3, :, :] + 11*f_2[4, :, :]) / (12 * dk**2)
        deriv2_k_2[-1, :, :] = (35*f_2[-1, :, :] - 104*f_2[-2, :, :] + 114*f_2[-3, :, :] - 56*f_2[-4, :, :] + 11*f_2[-5, :, :]) / (12 * dk**2)
        deriv2_l_0[:, 0, :]  = (35*f_0[:, 0, :] - 104*f_0[:, 1, :] + 114*f_0[:, 2, :] - 56*f_0[:, 3, :] + 11*f_0[:, 4, :]) / (12 * dl**2)
        deriv2_l_0[:, -1, :] = (35*f_0[:, -1, :] - 104*f_0[:, -2, :] + 114*f_0[:, -3, :] - 56*f_0[:, -4, :] + 11*f_0[:, -5, :]) / (12 * dl**2)
        deriv2_l_1[:, 0, :]  = (35*f_1[:, 0, :] - 104*f_1[:, 1, :] + 114*f_1[:, 2, :] - 56*f_1[:, 3, :] + 11*f_1[:, 4, :]) / (12 * dl**2)
        deriv2_l_1[:, -1, :] = (35*f_1[:, -1, :] - 104*f_1[:, -2, :] + 114*f_1[:, -3, :] - 56*f_1[:, -4, :] + 11*f_1[:, -5, :]) / (12 * dl**2)
        deriv2_l_2[:, 0, :]  = (35*f_2[:, 0, :] - 104*f_2[:, 1, :] + 114*f_2[:, 2, :] - 56*f_2[:, 3, :] + 11*f_2[:, 4, :]) / (12 * dl**2)
        deriv2_l_2[:, -1, :] = (35*f_2[:, -1, :] - 104*f_2[:, -2, :] + 114*f_2[:, -3, :] - 56*f_2[:, -4, :] + 11*f_2[:, -5, :]) / (12 * dl**2)
    # ----------------------------

    # Apply periodic boundary conditions along axis 2 (psi axis)
    # Pad with wrap mode for periodicity
    f_0_padded_psi = xp.pad(f_0, ((1, 1), (1, 1), (1, 1)), mode='wrap')
    f_1_padded_psi = xp.pad(f_1, ((1, 1), (1, 1), (1, 1)), mode='wrap')
    f_2_padded_psi = xp.pad(f_2, ((1, 1), (1, 1), (1, 1)), mode='wrap')

    deriv2_psi_0 = (f_0_padded_psi[1:-1, 1:-1, 2:] - 2 * f_0_padded_psi[1:-1, 1:-1, 1:-1] + f_0_padded_psi[1:-1, 1:-1, :-2]) / dpsi**2
    deriv2_psi_1 = (f_1_padded_psi[1:-1, 1:-1, 2:] - 2 * f_1_padded_psi[1:-1, 1:-1, 1:-1] + f_1_padded_psi[1:-1, 1:-1, :-2]) / dpsi**2
    deriv2_psi_2 = (f_2_padded_psi[1:-1, 1:-1, 2:] - 2 * f_2_padded_psi[1:-1, 1:-1, 1:-1] + f_2_padded_psi[1:-1, 1:-1, :-2]) / dpsi**2

    kin_term_1 = 2 / (omega) * (k * l * xp.cos(psi)) * f_1
    kin_term_0 = 2 / (omega) * (k * l * xp.cos(psi)) * f_0
    kin_term_2 = 2 / (omega) * (k * l * xp.cos(psi)) * f_2

    if sys.Ncmode == "FNc":
        Nc = 3
        
        q_4 = 0.25 * sys.qtilde


        V_term_00 = (
            (
             + 0.25 * Nc * (1 + 2 * sys.z**2)
            ) * (deriv2_k_0 + 1/k * deriv_k_0)

            + Nc * (deriv2_l_0 + 1/(l + epsl) * deriv_l_0 + 1/(l + epsl)**2 * deriv2_psi_0) * 0.25
        ) #O(1)

        #V_term01 = 0.0

        V_term02 = (sys.z - 1) * sys.z * (deriv2_k_2 + 1/k * deriv_k_2)  #O(1/Nc²)


        V_term10 =  (-sys.z + 1) * sys.z * (deriv2_k_0 + 1/k * deriv_k_0) #O(1)

        V_term11 = (
            -(((-1 + sys.z)**2) / (2 * Nc))
            + 0.5 * Nc * (1 + sys.z * (-2 + 3 * sys.z))
        ) * (deriv2_k_1 + 1/k * deriv_k_1)  #O(1)

        #V_term12 = 0.0

        V_term20 =  (sys.z - 1) * sys.z * (deriv2_k_0 + 1/k * deriv_k_0) # O(1/Nc)

        V_term21 = (1/2 * Nc) * (
            (-1 - 2 * (-1 + sys.z) * sys.z) * (deriv2_k_1 + 1/k * deriv_k_1)
            + (deriv2_l_1 + 1/(l + epsl) * deriv_l_1 + 1/(l + epsl)**2 * deriv2_psi_1)
        ) # O(1)
        
        V_term22 = (
            (-(((-1 + sys.z)**2) / (2 * Nc)) + 0.25 * Nc * (1 - 4 * sys.z + 6 * sys.z**2)) * (deriv2_k_2 + 1/k * deriv_k_2)
            + (Nc * (deriv2_l_2 + 1/(l + epsl) * deriv_l_2 + 1/(l + epsl)**2 * deriv2_psi_2)) * 0.25
        ) # O(1/Nc)
        

        HF_0 = kin_term_0 +  1j * q_4 *(V_term_00  + V_term02 * Nc**2 / (Nc**3 - 1))
        HF_1 = kin_term_1 +  1j * q_4 * (V_term10 * (Nc**3 - 1)/(Nc**2 - 1) + V_term11)
        HF_2 = kin_term_2 +  1j * q_4 * (V_term20 * (Nc**3 - 1)/ Nc**2 + V_term21 * (Nc**2 - 1)/Nc**2 + V_term22)


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
        HF_1 = kin_term_1 + 1j * q_4 * (V_term10 * (Nc**3 - 1)/(Nc**2 - 1) + V_term11)
        HF_2 = kin_term_2 + 0.0 * HF_1


    elif sys.Ncmode == "LNc":
        Nc = 3
        q_4 = 0.25 * sys.qtilde



        V_term_00 = (
            (
            -((-1 + sys.z)**2) / (2 * Nc) + 0.25 * Nc * (1 + 2 * sys.z**2)
            ) * (deriv2_k_0 + 1/k * deriv_k_0)

            + Nc * (deriv2_l_0 + 1/(l + epsl) * deriv_l_0 + 1/(l + epsl)**2 * deriv2_psi_0) * 0.25
        )

        #V_term01 = 0.0

        V_term02 = 0.0


        V_term10 =  (-sys.z + 1) * sys.z * (deriv2_k_0 + 1/k * deriv_k_0)

        V_term11 = (
            + 0.5 * Nc * (1 + sys.z * (-2 + 3 * sys.z))
        ) * (deriv2_k_1 + 1/k * deriv_k_1)

        #V_term12 = 0.0

        V_term20 =  (sys.z - 1) * sys.z * (deriv2_k_0 + 1/k * deriv_k_0)

        V_term21 = (1/2 * Nc) * (
            (-1 - 2 * (-1 + sys.z) * sys.z) * (deriv2_k_1 + 1/k * deriv_k_1)
            + (deriv2_l_1 + 1/(l + epsl) * deriv_l_1 + 1/(l + epsl)**2 * deriv2_psi_1)
        )
        
        V_term22 = (
            (-(((-1 + sys.z)**2) / (2 * Nc)) + 0.25 * Nc * (1 - 4 * sys.z + 6 * sys.z**2)) * (deriv2_k_2 + 1/k * deriv_k_2)
            + (Nc * (deriv2_l_2 + 1/(l + epsl) * deriv_l_2 + 1/(l + epsl)**2 * deriv2_psi_2)) * 0.25
        )
        

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
    #this one has 8 components
    dk = sys.dk
    dl = sys.dl
    dpsi = sys.dpsi
    k = sys.K[:, None, None]
    l = sys.L[None, :, None]
    psi = sys.psi[None, None, :]
    epsl = dl  

    omega = sys.omega
    f_0 = f[0]
    f_1 = f[1]
    f_2 = f[2]
    f_3 = f[3]
    f_4 = f[4]
    f_5 = f[5]
    f_6 = f[6]
    f_7 = f[7]

    if sys.Ncmode == "FNc":
        nsig = 8
    elif sys.Ncmode == "LNc":
        nsig = 2

    # Second derivatives in momentum space (bulk only)
    xp = cp if sys.optimization == "gpu" else np

    # Pad all f_i with linear extrapolation
    f_padded = [xp.pad(f_i, ((1, 1), (1, 1), (1, 1)), mode='edge') for f_i in [f_0, f_1, f_2, f_3, f_4, f_5, f_6, f_7]]

    # required first derivatives
    deriv_k = [(f_padded[i][2:, 1:-1, 1:-1] - f_padded[i][:-2, 1:-1, 1:-1]) / (2 * dk) for i in range(nsig)]
    deriv_l = [(f_padded[i][1:-1, 2:, 1:-1] - f_padded[i][1:-1, :-2, 1:-1]) / (2 * dl) for i in range(nsig)]  

    boundary_correction = False
    # apply one-sideded boundary conditions 
    if boundary_correction:
        for i in range(nsig):
            deriv_k[i][0, :, :]  = (-11*f_padded[i][1, :, :] + 18*f_padded[i][2, :, :] - 9*f_padded[i][3, :, :] + 2*f_padded[i][4, :, :]) / (6 * dk)
            deriv_k[i][-1, :, :] = (11*f_padded[i][-2, :, :] - 18*f_padded[i][-3, :, :] + 9*f_padded[i][-4, :, :] - 2*f_padded[i][-5, :, :]) / (6 * dk)
            deriv_l[i][:, 0, :]  = (-11*f_padded[i][:, 1, :] + 18*f_padded[i][:, 2, :] - 9*f_padded[i][:, 3, :] + 2*f_padded[i][:, 4, :]) / (6 * dl)
            deriv_l[i][:, -1, :] = ( 11*f_padded[i][:, -2, :] - 18*f_padded[i][:, -3, :] + 9*f_padded[i][:, -4, :] - 2*f_padded[i][:, -5, :]) / (6 * dl)

        #required second derivatives
    deriv2_k = [(f_padded[i][2:, 1:-1, 1:-1] - 2 * f_padded[i][1:-1, 1:-1, 1:-1] + f_padded[i][:-2, 1:-1, 1:-1]) / dk**2 for i in range(nsig)]

    deriv2_l = [(f_padded[i][1:-1, 2:, 1:-1] - 2 * f_padded[i][1:-1, 1:-1, 1:-1] + f_padded[i][1:-1, :-2, 1:-1]) / dl**2 for i in range(nsig)]

    # apply one-sideded boundary conditions
    if boundary_correction:
        for i in range(nsig):
            deriv2_k[i][0, :, :]  = (35*f_padded[i][1, :, :] - 104*f_padded[i][2, :, :] + 114*f_padded[i][3, :, :] - 56*f_padded[i][4, :, :] + 11*f_padded[i][5, :, :]) / (12 * dk**2)
            deriv2_k[i][-1, :, :] = (35*f_padded[i][-2, :, :] - 104*f_padded[i][-3, :, :] + 114*f_padded[i][-4, :, :] - 56*f_padded[i][-5, :, :] + 11*f_padded[i][-6, :, :]) / (12 * dk**2)
            deriv2_l[i][:, 0, :]  = (35*f_padded[i][:, 1, :] - 104*f_padded[i][:, 2, :] + 114*f_padded[i][:, 3, :] - 56*f_padded[i][:, 4, :] + 11*f_padded[i][:, 5, :]) / (12 * dl**2)
            deriv2_l[i][:, -1, :] = (35*f_padded[i][:, -2, :] - 104*f_padded[i][:, -3, :] + 114*f_padded[i][:, -4, :] - 56*f_padded[i][:, -5, :] + 11*f_padded[i][:, -6, :]) / (12 * dl**2)

    # ----------------------------

    # Apply periodic boundary conditions along axis 2 (psi axis)
    # Pad with wrap mode for periodicity
    f_padded_psi = [xp.pad(f_i, ((1, 1), (1, 1), (1, 1)), mode='wrap') for f_i in [f_0, f_1, f_2, f_3, f_4, f_5, f_6, f_7]]

    deriv2_psi = [(f_padded_psi[i][1:-1, 1:-1, 2:] - 2 * f_padded_psi[i][1:-1, 1:-1, 1:-1] + f_padded_psi[i][1:-1, 1:-1, :-2]) / dpsi**2 for i in range(nsig)]

    kin_term = [2 / (omega) * (k * l * xp.cos(psi)) * f_padded[i][1:-1, 1:-1, 1:-1] for i in range(8)]
    for i in range(nsig):
        deriv2_psi[i][:, 0, :] = 0
        deriv2_psi[i][:, -1, :] = 0

    HF = []

    if sys.Ncmode == "FNc":
        Nc = 3
        q_4 = 0.25 * sys.qtilde 
        z = sys.z

        C0Ncfac = Nc**2 * (Nc**2 - 1)
        C1Ncfac = Nc * (Nc**2 - 1)
        C2Ncfac = -Nc**2 * (Nc - 1)
        C3Ncfac = Nc * (Nc**2 - 1)
        C4Ncfac = Nc**2 * (4 * Nc - 3)
        C5Ncfac = 2 * (Nc**2 - 1)
        C6Ncfac = Nc**2
        C7Ncfac =  Nc**2 * (Nc**2 - 1)

        # #C0Ncfac = Nc**4 #Nc**4
        # C2Ncfac = 1 #Nc**5
        # C3Ncfac = 1 #Nc**5
        # C4Ncfac = 1 #Nc**3
        # C5Ncfac = 1 #Nc**5
        # C6Ncfac = 1 #Nc**4
        # C7Ncfac = 1 #Nc**2

        
        
        # start from kinetic terms, then add FNc potential for component 0 as specified
        HF = [kin_term[i] for i in range(8)]

        V0 = (
            # M00 term
            Nc * (1 - 2 * z + 2 * z**2) * (deriv2_k[0] + 1/k * deriv_k[0]) #O(Nc)

            # # M01 term
            + ((1.0/4.0 + z - z**2) * (deriv2_k[1] + 1/k * deriv_k[1])
                - 0.25 * (deriv2_l[1] + 1/(l + epsl) * deriv_l[1] + 1/(l + epsl)**2 * deriv2_psi[1])) * C1Ncfac/C0Ncfac # O(1/Nc)

            # # M02 term
             + ((-0.5 + z - z**2) * (deriv2_k[2] + 1/k * deriv_k[2])
                + 0.5 * (deriv2_l[2] + 1/(l + epsl) * deriv_l[2] + 1/(l + epsl)**2 * deriv2_psi[2])) * C2Ncfac/C0Ncfac #O(1/Nc)
            
            # # M03 term
             + (-0.75 * (deriv2_k[3] + 1/k * deriv_k[3])
                + 0.75 * (deriv2_l[3] + 1/(l + epsl) * deriv_l[3] + 1/(l + epsl)**2 * deriv2_psi[3])) * C3Ncfac/C0Ncfac # O(1/Nc)
        )


        V1 = (

            # M10 term
            0.25 * (deriv2_k[0] - deriv2_l[0] - 1/(l + epsl) * deriv_l[0] - 1/(l + epsl)**2 * deriv2_psi[0]) * C0Ncfac/C1Ncfac #O(Nc)

            # M11 term
            + Nc * (0.75 - z + z**2) * deriv2_k[1]  
            + (Nc / 4.0) * (deriv2_l[1] + 1/(l + epsl) * deriv_l[1] + 1/(l + epsl)**2 * deriv2_psi[1]) #O(Nc)
        )
        

        V2 = (
            # M20 term
            ((-0.75 + 4.0 * z - 4.0 * z**2) * (deriv2_k[0] + 1/k * deriv_k[0])
             + 0.75 * (deriv2_l[0] + 1/(l + epsl) * deriv_l[0] + 1/(l + epsl)**2 * deriv2_psi[0])) * C0Ncfac/C2Ncfac #O(1)

            # CM21 term
            + ((0.75 * Nc) * (deriv2_k[1] + 1/k * deriv_k[1])
               - 0.75 * Nc * (deriv2_l[1] + 1/(l + epsl) * deriv_l[1] + 1/(l + epsl)**2 * deriv2_psi[1])) * C1Ncfac/C2Ncfac #O(1)

            # M24 term
            + (Nc * (-0.25 + 2.0 * z - 2.0 * z**2) * (deriv2_k[4] + 1/k * deriv_k[4])
               + 0.25 * Nc * (deriv2_l[4] + 1/(l + epsl) * deriv_l[4] + 1/(l + epsl)**2 * deriv2_psi[4])) * C4Ncfac/C2Ncfac

            # M22 term
            + (Nc * (-0.25 - z + z**2) * (deriv2_k[2] + 1/k * deriv_k[2])
               + 1.25 * Nc * (deriv2_l[2] + 1/(l + epsl) * deriv_l[2] + 1/(l + epsl)**2 * deriv2_psi[2]))

            # M23 term
            + (-1.75 * Nc * (deriv2_k[3] + 1/k * deriv_k[3])
               + 1.75 * Nc * (deriv2_l[3] + 1/(l + epsl) * deriv_l[3] + 1/(l + epsl)**2 * deriv2_psi[3])) * C3Ncfac/C2Ncfac

            # M25 term
            + (0.75 * Nc**2 * (deriv2_k[5] + 1/k * deriv_k[5])
               - 0.75 * Nc**2 * (deriv2_l[5] + 1/(l + epsl) * deriv_l[5] + 1/(l + epsl)**2 * deriv2_psi[5])) * C5Ncfac/C2Ncfac
        )


        V3 = (
            # C3 Nc d^2_k
            C3Ncfac * Nc * (deriv2_k[3] + 1/k * deriv_k[3])

            # - C2 Nc (-1 + z) z d^2_k
            - C2Ncfac * Nc * (-1 + z) * z * (deriv2_k[2] + 1/k * deriv_k[2])

            # + C4 Nc (-1 + z) z d^2_k
            + C4Ncfac * Nc * (-1 + z) * z * (deriv2_k[4] + 1/k * deriv_k[4])

            # + C5 Nc^2 (-1 + z) z d^2_k
            + C5Ncfac * Nc**2 * (-1 + z) * z * (deriv2_k[5] + 1/k * deriv_k[5])

            # + C0 ((1/4 - 2z + 2z^2) d^2_k - (1/4) d^2_l)
            + C0Ncfac * ((0.25 - 2*z + 2*z**2) * (deriv2_k[0] + 1/k * deriv_k[0]) - 0.25 * (deriv2_l[0] + 1/(l + epsl) * deriv_l[0] + 1/(l + epsl)**2 * deriv2_psi[0]))

            # + C1 (-(1/4) Nc d^2_k + (Nc/4) d^2_l)
            + C1Ncfac * (-(0.25) * Nc * (deriv2_k[1] + 1/k * deriv_k[1]) + (Nc / 4.0) * (deriv2_l[1] + 1/(l + epsl) * deriv_l[1] + 1/(l + epsl)**2 * deriv2_psi[1]))

            # + C6 (-(1/2) Nc^2 d^2_k + (1/2) Nc^2 d^2_l)
            + C6Ncfac * (-(0.5) * Nc**2 * (deriv2_k[6] + 1/k * deriv_k[6]) + 0.5 * Nc**2 * (deriv2_l[6] + 1/(l + epsl) * deriv_l[6] + 1/(l + epsl)**2 * deriv2_psi[6]))
        ) / C3Ncfac


        V4 = (
            # C0 term
            C0Ncfac * (0.5 * (1 - 2*z)**2 * (deriv2_k[0] + 1/k * deriv_k[0]) - 0.5 * (deriv2_l[0] + 1/(l + epsl) * deriv_l[0] + 1/(l + epsl)**2 * deriv2_psi[0]))

            # C7 term
            + C7Ncfac * ((-0.5 + 4*z - 4*z**2) * (deriv2_k[7] + 1/k * deriv_k[7]) + 0.5 * (deriv2_l[7] + 1/(l + epsl) * deriv_l[7] + 1/(l + epsl)**2 * deriv2_psi[7]))

            # C1 term
            + C1Ncfac * (0.5 * Nc * (deriv2_k[1] + 1/k * deriv_k[1]) - 0.5 * Nc * (deriv2_l[1] + 1/(l + epsl) * deriv_l[1] + 1/(l + epsl)**2 * deriv2_psi[1]))

            # C4 term
            + C4Ncfac * (Nc * (0.75 - 2*z + 2*z**2) * (deriv2_k[4] + 1/k * deriv_k[4]) + 0.25 * Nc * (deriv2_l[4] + 1/(l + epsl) * deriv_l[4] + 1/(l + epsl)**2 * deriv2_psi[4]))

            # C2 term
            + C2Ncfac * (-0.5 * Nc * (deriv2_k[2] + 1/k * deriv_k[2]) + 0.5 * Nc * (deriv2_l[2] + 1/(l + epsl) * deriv_l[2] + 1/(l + epsl)**2 * deriv2_psi[2]))

            # C3 term
            + C3Ncfac * (-Nc * (deriv2_k[3] + 1/k * deriv_k[3]) + Nc * (deriv2_l[3] + 1/(l + epsl) * deriv_l[3] + 1/(l + epsl)**2 * deriv2_psi[3]))

            # C5 term
            + C5Ncfac * (0.5 * Nc**2 * (deriv2_k[5] + 1/k * deriv_k[5]) - 0.5 * Nc**2 * (deriv2_l[5] + 1/(l + epsl) * deriv_l[5] + 1/(l + epsl)**2 * deriv2_psi[5]))

            # C6 term
            + C6Ncfac * (Nc**2 * (0.5 - 4*z + 4*z**2) * (deriv2_k[6] + 1/k * deriv_k[6]) - 0.5 * Nc**2 * (deriv2_l[6] + 1/(l + epsl) * deriv_l[6] + 1/(l + epsl)**2 * deriv2_psi[6]))
        ) / C4Ncfac


        V5 = (
            # C1 (d^2_k/4 - d^2_l/4)
            (C1Ncfac / C5Ncfac) * (
            0.25 * (deriv2_k[1] + 1/k * deriv_k[1]) - 0.25 * (deriv2_l[1] + 1/(l + epsl) * deriv_l[1] + 1/(l + epsl)**2 * deriv2_psi[1])
            )
            # C2 ((-(3/4) + 2z - 2z^2) d^2_k + (3/4) d^2_l)
            + (C2Ncfac / C5Ncfac) * (
            (-0.75 + 2*z - 2*z**2) * (deriv2_k[2] + 1/k * deriv_k[2]) + 0.75 * (deriv2_l[2] + 1/(l + epsl) * deriv_l[2] + 1/(l + epsl)**2 * deriv2_psi[2])
            )
            # C3 ((-1 + 2z - 2z^2) d^2_k + d^2_l)
            + (C3Ncfac / C5Ncfac) * (
            (-1 + 2*z - 2*z**2) * (deriv2_k[3] + 1/k * deriv_k[3]) + (deriv2_l[3] + 1/(l + epsl) * deriv_l[3] + 1/(l + epsl)**2 * deriv2_psi[3])
            )
            # C7 (-(d^2_k/(2Nc)) + d^2_l/(2Nc))
            + (C7Ncfac / C5Ncfac) * (
            (-1/(2*Nc)) * (deriv2_k[7] + 1/k * deriv_k[7]) + (1/(2*Nc)) * (deriv2_l[7] + 1/(l + epsl) * deriv_l[7] + 1/(l + epsl)**2 * deriv2_psi[7])
            )
            # C5 (Nc (5/4 - 3z + 3z^2) d^2_k - Nc/4 d^2_l)
            + (
            Nc * (5/4 - 3*z + 3*z**2) * (deriv2_k[5] + 1/k * deriv_k[5]) - (Nc/4) * (deriv2_l[5] + 1/(l + epsl) * deriv_l[5] + 1/(l + epsl)**2 * deriv2_psi[5])
            )
            # C6 (-(1/4) Nc d^2_k + (Nc/4) d^2_l)
            + (
            -0.25 * Nc * (deriv2_k[6] + 1/k * deriv_k[6]) + 0.25 * Nc * (deriv2_l[6] + 1/(l + epsl) * deriv_l[6] + 1/(l + epsl)**2 * deriv2_psi[6])
            )
        )


        V6 = (
            # C2 term: -C2Ncfac * (-1 + z) * z * d^2_k[2]
            (-C2Ncfac/C6Ncfac * (-1 + z) * z) * (deriv2_k[2] + 1/k * deriv_k[2])

            # C7 term: (2 * C7Ncfac * (-1 + z) * z / Nc) * d^2_k[7]
            + (2 * C7Ncfac/C6Ncfac * (-1 + z) * z / Nc) * (deriv2_k[7] + 1/k * deriv_k[7])

            # C5 term: C5Ncfac * Nc * (-1 + z) * z * d^2_k[5]
            + C5Ncfac/C6Ncfac * Nc * (-1 + z) * z * (deriv2_k[5] + 1/k * deriv_k[5])

            # C3 term: C3Ncfac * ((1/4 + z - z^2) d^2_k[3] - (1/4) d^2_l[3])
            + C3Ncfac/C6Ncfac * (
            (0.25 + z - z**2) * (deriv2_k[3] + 1/k * deriv_k[3])
            - 0.25 * (deriv2_l[3] + 1/(l + epsl) * deriv_l[3] + 1/(l + epsl)**2 * deriv2_psi[3])
            )

            # C1 term: C1Ncfac * (-(1/4) d^2_k[1] + (1/4) d^2_l[1])
            + C1Ncfac/C6Ncfac * (
            -0.25 * (deriv2_k[1] + 1/k * deriv_k[1])
            + 0.25 * (deriv2_l[1] + 1/(l + epsl) * deriv_l[1] + 1/(l + epsl)**2 * deriv2_psi[1])
            )

            # C6 term: C6Ncfac * (Nc (1/4 + z - z^2) d^2_k[6] + (3 Nc d^2_l[6])/4)
            + C6Ncfac/C6Ncfac * (
            Nc * (0.25 + z - z**2) * (deriv2_k[6] + 1/k * deriv_k[6])
            + 0.75 * Nc * (deriv2_l[6] + 1/(l + epsl) * deriv_l[6] + 1/(l + epsl)**2 * deriv2_psi[6])
            ) 
        )

        V7 = (
            # C7 (1/2 Nc (1 - 2 z)^2 d^2_k[7] + (Nc d^2_l[7])/2)
            0.5 * Nc * (1 - 2*z)**2 * (deriv2_k[7] + 1/k * deriv_k[7])
            + 0.5 * Nc * (deriv2_l[7] + 1/(l + epsl) * deriv_l[7] + 1/(l + epsl)**2 * deriv2_psi[7])

            # C2 (Nc^2 (-(1/4) + z - z^2) d^2_k[2] + 1/4 Nc^2 d^2_l[2])
            + (Nc**2 * (-(1/4) + z - z**2) * (deriv2_k[2] + 1/k * deriv_k[2])
            + 0.25 * Nc**2 * (deriv2_l[2] + 1/(l + epsl) * deriv_l[2] + 1/(l + epsl)**2 * deriv2_psi[2])) * C2Ncfac/C7Ncfac

            # C3 (Nc^2 (-(1/4) + z - z^2) d^2_k[3] + 1/4 Nc^2 d^2_l[3])
            + (Nc**2 * (-(1/4) + z - z**2) * (deriv2_k[3] + 1/k * deriv_k[3])
            + 0.25 * Nc**2 * (deriv2_l[3] + 1/(l + epsl) * deriv_l[3] + 1/(l + epsl)**2 * deriv2_psi[3])) * C3Ncfac/C7Ncfac

            # C5 (Nc^3 (1/4 - z + z^2) d^2_k[5] - 1/4 Nc^3 d^2_l[5])
            + (Nc**3 * (1/4 - z + z**2) * (deriv2_k[5] + 1/k * deriv_k[5])
            - 0.25 * Nc**3 * (deriv2_l[5] + 1/(l + epsl) * deriv_l[5] + 1/(l + epsl)**2 * deriv2_psi[5])) * C5Ncfac/C7Ncfac

            # C6 (Nc^3 (-(1/4) + z - z^2) d^2_k[6] + 1/4 Nc^3 d^2_l[6])
            + (Nc**3 * (-(1/4) + z - z**2) * (deriv2_k[6] + 1/k * deriv_k[6])
            + 0.25 * Nc**3 * (deriv2_l[6] + 1/(l + epsl) * deriv_l[6] + 1/(l + epsl)**2 * deriv2_psi[6])) * C6Ncfac/C7Ncfac
        )


        HF[0] += 1j * q_4 * V0
        HF[1] += 1j * q_4 * V1
        HF[2] += 1j * q_4 * V2
        HF[3] += 1j * q_4 * V3
        HF[4] += 1j * q_4 * V4
        HF[5] += 1j * q_4 * V5
        HF[6] += 1j * q_4 * V6
        HF[7] += 1j * q_4 * V7
        
        # return full 8-component HF
        return xp.array(HF)

    elif sys.Ncmode == "LNc" or sys.Ncmode == "LeadNc":
        Nc = 3
        q_4 = 0.25 * sys.qtilde
        z = sys.z

        C0Ncfac = Nc**4
        C1Ncfac = Nc**3
        if sys.Ncmode == "LeadNc":
            C0Ncfac = Nc**2 * (Nc**2 - 1)
            C1Ncfac = Nc * (Nc**2 - 1)
        
        # C2Ncfac = -Nc**2 * (Nc - 1)
        # C3Ncfac = Nc * (Nc**2 - 1)
        # C4Ncfac = Nc**2 * (4 * Nc - 3)
        # C5Ncfac = 2 * (Nc**2 - 1)
        # C6Ncfac = 1
        # C7Ncfac =  Nc**2 * (Nc**2 - 1)

        # #C0Ncfac = Nc**4 #Nc**4
        # C2Ncfac = 1 #Nc**5
        # C3Ncfac = 1 #Nc**5
        # C4Ncfac = 1 #Nc**3
        # C5Ncfac = 1 #Nc**5
        # C6Ncfac = 1 #Nc**4
        # C7Ncfac = 1 #Nc**2

        # start from kinetic terms, then add FNc potential for component 0 as specified
        HF = [kin_term[i] for i in range(2)]

        V0 = (
            # M00 term
            Nc * (1 - 2 * z + 2 * z**2) * (deriv2_k[0] + 1/k * deriv_k[0]))

        V1 = (

            # M10 term
            0.25 * (deriv2_k[0] - deriv2_l[0] - 1/(l + epsl) * deriv_l[0] - 1/(l + epsl)**2 * deriv2_psi[0]) * C0Ncfac/C1Ncfac

            # M11 term
            + Nc * (0.75 - z + z**2) * deriv2_k[1]  
            + (Nc / 4.0) * (deriv2_l[1] + 1/(l + epsl) * deriv_l[1] + 1/(l + epsl)**2 * deriv2_psi[1])
        )

        HF[0] += 1j * q_4 * V0
        HF[1] += 1j * q_4 * V1
        for i in range(2,8):
            HF.append(0.0 * HF[1])  #irrelevant for large nc
        
        
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










