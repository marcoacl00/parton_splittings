from .hamiltonian import *
from tqdm import tqdm
from scipy.integrate import simpson
import os

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
        F_med_sim[i] = (1 / (np.pi)) * simpson(integrand, dx=dpsi)

    return F_med_sim

def euler_step3D(sis, ht):
    Fsol = sis.Fsol
    Hf = apply_hamil_3D(sis, Fsol)
    Fsol_new = Fsol + (-1j * ht) * Hf
    return Fsol_new


def simulate_euler(sist, ht, t_L, step_save=500):
    

    """
    Simulates the system in 3D using the Euler method for time integration.
    """

    sis = sist

    time_list = np.arange(0, t_L, ht)
    cont = 1
    for step, t in enumerate(tqdm(time_list)):

        # Euler method: non-homogeneous term
        nHom = sis.source_term_array(sis.t)
        nFsol = sis.Fsol
        
        nFsol[0] += nHom * ht
        nFsol[1] += nHom * ht
        

        sis.set_fsol(nFsol)

        # Euler method: homogeneous term
        Fsol_new = euler_step3D(sis, ht)
        sis.set_fsol(Fsol_new)

        cont += 1
        sis.increase_t(ht)


        if cont % step_save == 0:
            F_med_ii = integrate_solution3D(sis)
            save_dir = "saved_fmeds_ii/gamma_qq/"
            energy_dir = "_{}/".format(int(sis.E/sis.fm))

            spec_dir = sis.Ncmode + energy_dir
            save_dir += spec_dir

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Save F_med_ii with E, qtilde, and z in the filename
            E_val = sis.E/sis.fm
            qtilde_val = sis.qtilde/sis.fm**2
            z_val = sis.z
            Lk = sis.Lk

            print("Maximum Fmed = ", F_med_ii.max())

            fname = f"F_med_ii_E={round(E_val, 4)}GeV_qtilde={round(qtilde_val, 4)}GeV²_fm_z={z_val}_t{round(sis.t, 4)}_Lk={Lk}GeV⁻¹.npy"
            
            np.save(save_dir + fname, F_med_ii)

    

    return sis