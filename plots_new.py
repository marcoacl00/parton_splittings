import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
import re
import sys
import os

def main(file_name):
    array = np.load("simulations/" + file_name)


    def fasit2Ncdiag(t,L,p1,p2,z, omega, Omega, qF):
        pre= -2*1j*omega
        
        num = 2*omega*Omega/np.tan(Omega*t)
        den = 2*omega*Omega/np.tan(Omega*t)+1j*qF*(z**2+(1-z)**2)*(L-t)
        
        return pre*(1-num/den*np.exp(-1j*(p1**2+p2**2)/den))


    def fasit2Ncdiagint(L,p1,p2,z, omega, Omega, qF):
        def real_fas(t,L,p1,p2,z):
            return np.real(fasit2Ncdiag(t,L,p1,p2,z, omega, Omega, qF))
        def imag_fas(t,L,p1,p2,z):
            return np.imag(fasit2Ncdiag(t,L,p1,p2,z, omega, Omega, qF))
        re = quad(real_fas,0,L,args=(L,p1,p2,z))[0]
        im = quad(imag_fas,0,L,args=(L,p1,p2,z))[0]
        
        return re + 1j*im
    
    def F_in_out(t, omega, Omega, theta):
        """Calculate the in-out function for the source term"""
        Finout = np.real(-2 * (1 - np.exp(-1j * np.tan(Omega *t) / (2* omega * Omega)* omega**2 * theta**2)))
        
        return Finout


    # Extract variables from the file name using regex
    pattern = r"E=(?P<E>[\d\.]+)_z=(?P<z>[\d\.]+)_qhat=(?P<qhat>[\d\.]+)_Lk=(?P<Lk>[\d\.]+)_Ll=(?P<Ll>[\d\.]+)_Nk1=(?P<Nk1>\d+)_Nk2=(?P<Nk2>\d+)_Nl1=(?P<Nl1>\d+)_Nl2=(?P<Nl2>\d+)_L=(?P<L>[\d\.]+)_NcMode=(?P<NcMode>[^\.]+).npy"
    match = re.search(pattern, file_name)

    if match:
        E = float(match.group("E")) 
        z = float(match.group("z"))
        qhat = float(match.group("qhat")) 
        Lk = float(match.group("Lk")) 
        Ll = float(match.group("Ll"))
        L = float(match.group("L"))
        Nk1 = int(match.group("Nk1"))
        Nk2 = int(match.group("Nk2"))
        Nl1 = int(match.group("Nl1"))
        Nl2 = int(match.group("Nl2"))
        NcMode = match.group("NcMode")
    else:
        raise ValueError("Could not extract parameters from file name.")

    print(f"Parameters extracted: E={E}, z={z}, qhat={qhat}, Lk={Lk}, Ll={Ll}, L={L}")

    fm = 5.067
    # Convert units from GeV to fm
    E = E * fm
    qhat = qhat * fm**2
    Lk = Lk * fm
    Ll = Ll * fm

    omega = z * (1-z) * E
    Omega = np.sqrt(qhat / omega) * (1-1j)/2

    # Prepare momentum grids
    eps = 1e-4
    grid_k1 = np.linspace(-eps, Lk - eps, Nk1)
    grid_k2 = np.linspace(0, Lk, Nk2)
    grid_l1 = np.linspace(-Ll/2, Ll/2, Nl1)
    grid_l2 = np.linspace(-Ll/2, Ll/2, Nl2)



    Theta = grid_k1 / omega
    # Interpolator for array[1]
    #interp = RegularGridInterpolator((grid_k1, grid_k2, grid_l1, grid_l2), array[1], bounds_error=False)

    # Evaluate at (k, l=0)
    #f_vals_interp = np.zeros_like(mom_theo, dtype=np.complex128)
    #for i in range(len(mom_theo)):
        #f_vals_interp[i] = interp((mom_theo[i], 0, 0, 0))

    # Find the index in grid_k2 closest to 0
    idx_k2_0 = np.argmin(np.abs(grid_k2))
    print(f"Index of grid_k2 closest to 0: {idx_k2_0}")
    print(f"Value of grid_k2 at this index: {grid_k2[idx_k2_0]}")

    print("Considering 0 for l1:", grid_l1[Nl1//2])

    f_vals_theo = .0 * Theta * 1j
    f_iout_vals = .0 * Theta * 1j

    for i in range(len(grid_k1)):
        f_vals_theo[i] = fasit2Ncdiagint(L, grid_k1[i], 0, z, omega, Omega, qhat)
        f_iout_vals[i] = np.real(F_in_out(L, omega, Omega, Theta[i]))
    print(array.shape)


    F_med_sim = .0 * Theta 
    for i in range(len(Theta)):
        F_med_sim[i] = Theta[i]**2 / 2 * np.real(array[1, i, idx_k2_0, Nl1//2, Nl2//2])

    Fmed_ii_theo = Theta**2 /2 * np.real(f_vals_theo)

    # Plotting F_in-in
    plt.plot(Theta, Fmed_ii_theo, label=r"Large $N_c$ fac.", linestyle='--', color = 'blue')
    plt.plot(Theta, F_med_sim, 'o', markersize = 2, label="Sim.", color = 'red')
    theta_lim = Theta[-1] if Theta[-1] < 1 else 1
    plt.xlabel(r"$\theta$")
    plt.ylabel("$F^{in-in}$")
    plt.legend()
    plt.title(r"$E = {:.1f}$ GeV, $z = {:.2f}$, $\hat{{q}} = {:.1f}$ GeV$^2$/fm, $L = {:.1f}$ fm, NcMode = {}"
              .format(E / fm, z, qhat / fm**2, L, NcMode), size=10)
    plt.xlim(0, theta_lim)
    plt.grid(alpha = 0.2)

    # Ensure the saved_results_plots directory exists
    os.makedirs("saved_results_plots", exist_ok=True)
    plt.savefig(f"saved_results_plots/F_in-in_E={E/fm:.1f}_z={z:.2f}_qhat={qhat/fm**2:.1f}_L={L:.1f}_NcMode={NcMode}.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    #Plotting F = F_in-in + F_in-out
    F_in_out_ = F_med_sim + f_iout_vals
    plt.plot(Theta, Fmed_ii_theo + f_iout_vals, label=r"Large $N_c$ fac.", linestyle='--', color = 'blue')
    plt.plot(Theta, F_in_out_, 'o', markersize = 2, label="Sim.", color = 'red')
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$F_{med}$")
    plt.legend()
    plt.title(r"$E = {:.1f}$ GeV, $z = {:.2f}$, $\hat{{q}} = {:.1f}$ GeV$^2$/fm, $L = {:.1f}$ fm, NcMode = {}"
              .format(E / fm, z, qhat / fm**2, L, NcMode), size=10)
    plt.xlim(0, theta_lim)
    plt.grid(alpha = 0.2)

    # Save the plot
    plt.savefig(f"saved_results_plots/F_med_E={E/fm:.1f}_z={z:.2f}_qhat={qhat/fm**2:.1f}_L={L:.1f}_NcMode={NcMode}.pdf", format='pdf', bbox_inches='tight')
    plt.close()


    # Ensure the saved_results directory exists
    os.makedirs("saved_results", exist_ok=True)
    # Save arrays to npy file in saved_results/
    np.save(f"saved_results/F_in-in_arrays_E={E/fm:.1f}_z={z:.2f}_qhat={qhat/fm**2:.1f}_L={L:.1f}_NcMode={NcMode}.npy",
            np.array([Theta, Fmed_ii_theo, F_med_sim, F_in_out_], dtype=np.complex128))


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python plots.py <file_name>")
        sys.exit(1)
    
    file_name = sys.argv[1]
    main(file_name)


