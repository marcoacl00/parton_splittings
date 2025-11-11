from .faber import *
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.integrate import simpson


def integrate_solution3D(sis):
    array = sis.Fsol[1].get()
    grid_k = sis.K.get()
    grid_psi = sis.psi.get()
    Theta = grid_k / sis.omega
    
    dpsi = grid_psi[1] - grid_psi[0]
    F_med_sim = np.zeros(len(Theta))

    for i in range(len(Theta)):
        integrand = np.real(array[i, 0, :]) * Theta[i]**2 / 2
        # apply Simpson's 1/3 rule
        #
        F_med_sim[i] = (1 / (2 * np.pi)) * simpson(integrand, dx=dpsi)

    return F_med_sim

def simulate3D(sist, ht, t_L, step_save=10):
    """
    Simulates the system in 3D using the Simpson rule for time integration,
    similar to simulate_new_momentum, but keeping the 3D structure.
    """

    sis = sist


    time_list = np.arange(1e-3, t_L, ht)

    lambF, gamma0, gamma1 = faber_params3D(sist)
    one_lamb = 1 / lambF

    print("Faber params: lamb = ", lambF, "gamma0 = ", gamma0, "gamma1 = ", gamma1)

    m = 1
    coeff_array = [coeff(0, ht, gamma0, gamma1, lambF, sist.optimization)]
    while (np.abs(coeff_array[-1]) > 1e-7 or m < 3):
        coeff_array.append(coeff(m, ht, gamma0, gamma1, lambF, sist.optimization))
        m += 1

    o = 1
    coeff_array_2 = [coeff(0, ht * 0.5, gamma0, gamma1, lambF, sist.optimization)]
    while (np.abs(coeff_array_2[-1]) > 1e-7 or o < 3):
        coeff_array_2.append(coeff(o, ht * 0.5, gamma0, gamma1, lambF, sist.optimization))
        o += 1

    print("Number of polynomials = ", m, " and ", o)
    cont = 1
    if sis.vertex == "gamma_qq":
        for _ in tqdm(time_list):
            # Simpson rule: initial non-homogeneous term
            nHom = sis.source_term_array(sis.t)
            nFsol = sis.Fsol
            nFsol[0] += nHom * ht / 6
            nFsol[1] += nHom * ht / 6
            sis.set_fsol(nFsol)

            # Simpson rule: mid-point non-homogeneous term
            # Faber expansion for full step
            f_sol_n = faber_expand3D(sis, ht, gamma0, gamma1, one_lamb, coeff_array)

            sis_aux = sis

            xp = sis.xp
            nHom_dt_2 = xp.array([sis.source_term_array(sis.t + ht / 2), sis.source_term_array(sis.t + ht / 2)])

            sis_aux.set_fsol(nHom_dt_2)

            f_term_2 = faber_expand3D(sis_aux, ht / 2, gamma0, gamma1, one_lamb, coeff_array_2)

            # Simpson rule: final non-homogeneous term
            nHom_end = sis.source_term_array(sis.t + ht)

            # Complete Simpson rule update
            nFsol = f_sol_n + 4 / 6 * f_term_2 * ht
            nFsol[0] += nHom_end * ht / 6
            nFsol[1] += nHom_end * ht / 6

            sis.set_fsol(nFsol)

            if cont % step_save == 0:
                F_med_ii = integrate_solution3D(sis)
                save_dir = "saved_fmeds_ii/gamma_qq/"

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # Save F_med_ii with E, qtilde, and z in the filename
                E_val = sis.E/sis.fm
                qtilde_val = sis.qtilde/sis.fm**2
                z_val = sis.z
                Lk = sis.Lk

                fname = f"F_med_ii_E={round(E_val, 4)}GeV_qtilde={round(qtilde_val, 4)}GeV²_fm_z={z_val}_t{round(sis.t, 4)}_Lk={Lk}GeV⁻¹.npy"
                
                np.save(save_dir + fname, F_med_ii)

            # NaN-safety check
            if np.isnan(nHom).any():
                print("Warning: NaN detected in nHom at t = {}".format(sis.t))
            if np.isnan(nFsol).any():
                raise ValueError("NaN detected in nFsol at t = {}".format(sis.t))

            
            cont += 1
            sis.increase_t(ht)
    
    elif sis.vertex == "q_qg":
        for _ in tqdm(time_list):
            # Simpson rule: initial non-homogeneous term
            nHom = sis.source_term_array(sis.t)
            nFsol = sis.Fsol
            nFsol[0] += nHom * ht / 6
            nFsol[1] += nHom * ht / 6
            nFsol[2] += nHom * ht / 6
            sis.set_fsol(nFsol)

            # Simpson rule: mid-point non-homogeneous term
            # Faber expansion for full step
            f_sol_n = faber_expand3D(sis, ht, gamma0, gamma1, one_lamb, coeff_array)

            sis_aux = sis

            xp = sis.xp
            nHom_dt_2 = xp.array([sis.source_term_array(sis.t + ht / 2), 
                                  sis.source_term_array(sis.t + ht / 2),
                                  sis.source_term_array(sis.t + ht / 2)])

            sis_aux.set_fsol(nHom_dt_2)

            f_term_2 = faber_expand3D(sis_aux, ht / 2, gamma0, gamma1, one_lamb, coeff_array_2)

            # Simpson rule: final non-homogeneous term
            nHom_end = sis.source_term_array(sis.t + ht)

            # Complete Simpson rule update
            nFsol = f_sol_n + 4 / 6 * f_term_2 * ht
            nFsol[0] += nHom_end * ht / 6
            nFsol[1] += nHom_end * ht / 6
            nFsol[2] += nHom_end * ht / 6

            sis.set_fsol(nFsol)

            if cont % step_save == 0:
                F_med_qq = integrate_solution3D(sis)
                save_dir = "saved_fmeds_ii/q_qg/"

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # Save F_med_qq with E, qtilde, and z in the filename
                E_val = sis.E/sis.fm
                qtilde_val = sis.qtilde/sis.fm**2
                z_val = sis.z
                Lk = sis.Lk

                fname = f"F_med_qq_E={round(E_val, 4)}GeV_qtilde={round(qtilde_val, 4)}GeV²_fm_z={z_val}_t{round(sis.t, 4)}_Lk={Lk}GeV⁻¹.npy"
                
                np.save(save_dir + fname, F_med_qq)

            # NaN-safety check
            if np.isnan(nHom).any():
                print("Warning: NaN detected in nHom at t = {}".format(sis.t))
            if np.isnan(nFsol).any():
                raise ValueError("NaN detected in nFsol at t = {}".format(sis.t))

            if _ % step_save == 0:
                print("Processing t = ", round(sis.t, 3))
                # You can add 3D plotting here if needed
            cont += 1
            sis.increase_t(ht)

    elif sis.vertex == "g_gg":
        for _ in tqdm(time_list):
            # Simpson rule: initial non-homogeneous term
            c = np.ones(sis.Nsig)
            c[6] = 0.0

            nHom = sis.source_term_array(sis.t)
            nFsol = sis.Fsol
            for i in range(sis.Nsig):
                nFsol[i] += c[i] * nHom * ht / 6
            sis.set_fsol(nFsol)

            # Simpson rule: mid-point non-homogeneous term
            # Faber expansion for full step
            f_sol_n = faber_expand3D(sis, ht, gamma0, gamma1, one_lamb, coeff_array)

            sis_aux = sis

            xp = sis.xp
            nHom_dt_2 = xp.array([c[_] * sis.source_term_array(sis.t + ht / 2) for _ in range(sis.Nsig)])

            sis_aux.set_fsol(nHom_dt_2)

            f_term_2 = faber_expand3D(sis_aux, ht / 2, gamma0, gamma1, one_lamb, coeff_array_2)

            # Simpson rule: final non-homogeneous term
            nHom_end = sis.source_term_array(sis.t + ht)

            # Complete Simpson rule update
            nFsol = f_sol_n + 4 / 6 * f_term_2 * ht
            for i in range(sis.Nsig):
                nFsol[i] += c[i] * nHom_end * ht / 6

            sis.set_fsol(nFsol)

            if cont % step_save == 0:
                F_med_gg = integrate_solution3D(sis)
                save_dir = "saved_fmeds_ii/g_gg/"

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # Save F_med_gg with E, qtilde, and z in the filename
                E_val = sis.E/sis.fm
                qtilde_val = sis.qtilde/sis.fm**2
                z_val = sis.z
                Lk = sis.Lk

                fname = f"F_med_gg_E={round(E_val, 4)}GeV_qtilde={round(qtilde_val, 4)}GeV²_fm_z={z_val}_t{round(sis.t, 4)}_Lk={Lk}GeV⁻¹.npy"
                
                np.save(save_dir + fname, F_med_gg)

            # NaN-safety check
            if np.isnan(nHom).any():
                print("Warning: NaN detected in nHom at t = {}".format(sis.t))
            if np.isnan(nFsol).any():
                raise ValueError("NaN detected in nFsol at t = {}".format(sis.t))

            sis.increase_t(ht)
            cont += 1

    print("Final time = ", round(sis.t, 3))
    return sis
