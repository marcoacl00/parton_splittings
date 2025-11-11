import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
import re
import sys
import os
from scipy.integrate import simpson

def main(file_name):
    array = np.load("simulations/" + file_name)


    def fasit2Ncdiag(t,L,p1,p2,z, omega, Omega, qF, gz):
        pre= -2*1j*omega
        
        num = 2*omega*Omega/np.tan(Omega*t)
        den = 2*omega*Omega/np.tan(Omega*t)+1j*qF*gz*(L-t)
        
        return pre*(1-num/den*np.exp(-1j*(p1**2+p2**2)/den))
    


    def fasit2Ncdiagint(L,p1,p2,z, omega, Omega, qF, gz):
        def real_fas(t,L,p1,p2,z):
            return np.real(fasit2Ncdiag(t,L,p1,p2,z, omega, Omega, qF, gz))
        def imag_fas(t,L,p1,p2,z):
            return np.imag(fasit2Ncdiag(t,L,p1,p2,z, omega, Omega, qF, gz))
        re = quad(real_fas,0,L,args=(L,p1,p2,z))[0]
        im = quad(imag_fas,0,L,args=(L,p1,p2,z))[0]
        
        return re + 1j*im
    
    def F_in_out(t, omega, Omega, theta):
        """Calculate the in-out function for the source term"""
        Finout = np.real(-2 * (1 - np.exp(-1j * np.tan(Omega *t) / (2* omega * Omega)* omega**2 * theta**2)))
        
        return Finout


    # Extract variables from the file name using regex
    pattern = r"E=([\d\.]+)_z=([\d\.]+)_qtilde=([\d\.]+)_Lk=([\d\.]+)_Ll=([\d\.]+)_Nk=(\d+)_Nl=(\d+)_Npsi=(\d+)_L=([\d\.]+)_NcMode=([^.]+)_vertex=([^.]+)\.npy"
    match = re.search(pattern, file_name)

    if match:
        E = float(match.group(1))
        z = float(match.group(2))
        qtilde_ = float(match.group(3))
        Lk = float(match.group(4))
        Ll = float(match.group(5))
        Nk = int(match.group(6))
        Nl = int(match.group(7))
        Npsi = int(match.group(8))
        L = float(match.group(9))
        NcMode = match.group(10)
        vertex = match.group(11)
    else:
        raise ValueError("Could not extract parameters from file name.")

    print(f"Parameters extracted: E={E}, z={z}, qtilde={qtilde_}, Lk={Lk}, Ll={Ll}, Nk={Nk}, Nl={Nl}, Npsi={Npsi}, L={L}, NcMode={NcMode}")

    fm = 5.067
    # Convert units from GeV to fm
    E = E * fm
    qtilde = qtilde_ * fm**2
    Lk = Lk * fm
    Ll = Ll * fm
    Nc = 3
    CF = (Nc**2 - 1) / (2 * Nc)
    
    CA = Nc

    omega = z * (1-z) * E

    if NcMode == "LNc" or NcMode == "LNcFac":
        CF = Nc/2


    if vertex == "gamma_qq":
        
        qab = qtilde * CF        
        Omega = np.sqrt(qab / omega) * (1-1j)/2

    elif vertex == "q_qg":
            
        qab = 0.5 * ((CA) * (z) + CF * (1-z)**2) * qtilde
        Omega = (1 - 1j)/2 *np.sqrt(qab / omega)

    
    elif vertex == "g_gg":
        qab = (1 - z + z**2) * qtilde * CA
        Omega = np.sqrt(qab / omega) * (1-1j)/2

    

    # Prepare momentum grids
    eps = Lk / (Nk - 1)
    grid_k = np.linspace(0, Lk, Nk) + eps
    grid_l = np.linspace(0, Ll, Nl) 
    grid_psi = np.linspace(0, 2*np.pi, Npsi)



    Theta = grid_k / omega
    # Interpolator for array[1]
    #interp = RegularGridInterpolator((grid_k1, grid_k2, grid_l1, grid_l2), array[1], bounds_error=False)


    f_vals_theo = .0 * Theta * 1j
    f_iout_vals = .0 * Theta * 1j
    f_iout_vals_Nc = .0 * Theta * 1j

    if vertex == 'gamma_qq':
        gz = z**2 + (1 - z)**2

    elif vertex == 'q_qg':
        gz = (1 + z * (-2 + 3 * z))

    else:
        gz = z**2 + (1 - z)**2

    
    

    for i in range(len(grid_k)):
        f_vals_theo[i] = fasit2Ncdiagint(L, grid_k[i], 0, z, omega, Omega, qtilde * CF, gz)
        f_iout_vals[i] = np.real(F_in_out(L, omega, Omega, Theta[i]))
        f_iout_vals_Nc[i] = np.real(F_in_out(L, omega, Omega, Theta[i]))


    F_med_sim = .0 * Theta 

    dpsi = grid_psi[1] - grid_psi[0]
    for i in range(len(Theta)):
        # collect the integrand values for all psi
        integrand = np.real(array[1, i, 0, :]) * Theta[i]**2 / 2


        # apply Simpson's 1/3 rule
        F_med_sim[i] = (1 / (2 * np.pi)) * simpson(integrand, dx=dpsi)
       # F_med_sim[i] = np.real(array[1, i, 0, 0]) * Theta[i]**2 / 2


    Fmed_ii_theo = Theta**2 /2 * np.real(f_vals_theo)

    theta_indexes = np.where(Theta <= 1)[0]
    Theta = Theta[theta_indexes]
    Fmed_ii_theo = Fmed_ii_theo[theta_indexes]
    F_med_sim = F_med_sim[theta_indexes]
    f_iout_vals = f_iout_vals[theta_indexes]
    f_iout_vals_Nc = f_iout_vals_Nc[theta_indexes]

    
    F_in_out_ = F_med_sim + f_iout_vals

    # Plotting F_in-in
    #plt.plot(Theta, Fmed_ii_theo, label=r"Large $N_c$ fac.", linestyle='--', color = 'blue')
    plt.plot(Theta, F_med_sim, '-o', markersize = 2, label="Sim.", color = 'red')
    theta_lim = Theta[-1] if Theta[-1] < 1 else 1
    plt.xlabel(r"$\theta$")
    plt.ylabel("$F^{in-in}$")
    plt.legend()
    plt.title(r"$p^+ = {:.1f}$ GeV, $z = {:.2f}$, $\tilde{{q}} = {:.1f}$ GeV$^2$/fm, $L = {:.1f}$ fm, NcMode = {}, vertex = {}"
              .format(E / fm, z, qtilde / fm**2, L, NcMode, vertex), size=10)
    plt.xlim(0, theta_lim)
    plt.grid(alpha = 0.2)

    # Ensure the saved_results_plots directory exists
    os.makedirs("saved_results_plots", exist_ok=True)
    plt.savefig(f"saved_results_plots/F_in-in_E={E/fm:.1f}_z={z:.2f}_qtilde={qtilde/fm**2:.1f}_L={L:.1f}_NcMode={NcMode}_vertex={vertex}.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    #Plotting F = F_in-in + F_in-out

    plt.plot(Theta, Fmed_ii_theo + f_iout_vals_Nc, label=r"Large $N_c$ fac.", linestyle='--', color = 'blue')
    plt.plot(Theta, F_in_out_, '-o', markersize = 1.8, linewidth = 0.8, label="Numerical", color = 'red')
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$F_{med}$")
    plt.legend()
    plt.title(r"$p^+ = {:.1f}$ GeV, $z = {:.2f}$, $\tilde{{q}} = {:.1f}$ GeV$^2$/fm, $L = {:.1f}$ fm, NcMode = {}, vertex = {}"
              .format(E / fm, z, qtilde / fm**2, L, NcMode, vertex), size=10)
    plt.xlim(0, theta_lim)
    plt.grid(alpha = 0.2)

    # Save the plot
    plt.savefig(f"saved_results_plots/F_med_E={E/fm:.1f}_z={z:.2f}_qtilde={qtilde/fm**2:.1f}_L={L:.1f}_NcMode={NcMode}_vertex={vertex}.pdf", format='pdf', bbox_inches='tight')
    plt.close()


    # Ensure the saved_results directory exists
    os.makedirs("saved_results", exist_ok=True)
    # Save arrays to npy file in saved_results/
    np.save(f"saved_results/F_in-in_arrays_E={E/fm:.1f}_z={z:.2f}_qtilde={qtilde/fm**2:.1f}_L={L:.1f}_NcMode={NcMode}.npy",
            np.array([Theta, Fmed_ii_theo, F_med_sim, F_in_out_], dtype=np.complex128))


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python plots.py <file_name>")
        sys.exit(1)
    
    file_name = sys.argv[1]
    main(file_name)


