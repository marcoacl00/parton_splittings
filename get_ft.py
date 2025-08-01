from splittings_code import *
import matplotlib.pyplot as plt
import re
import os 
import sys


def main(dir, filename):

    match_E = re.search(r"E=(-?\d+(\.\d+)?)", filename)
    match_z = re.search(r"z=(-?\d+(\.\d+)?)", filename)
    match_qhat =  re.search(r"qhat=(-?\d+(\.\d+)?)", filename)
    match_Lu =  re.search(r"Lu=(-?\d+(\.\d+)?)", filename)
    match_Lv =  re.search(r"Lv=(-?\d+(\.\d+)?)", filename)
    match_Nu1 =  re.search(r"Nu1=(-?\d+(\.\d+)?)", filename)
    match_Nu2 =  re.search(r"Nu2=(-?\d+(\.\d+)?)", filename)
    match_Nv1 =  re.search(r"Nv1=(-?\d+(\.\d+)?)", filename)
    match_Nv2 =  re.search(r"Nv2=(-?\d+(\.\d+)?)", filename)
    match_L =  re.search(r"L=(-?\d+(\.\d+)?)", filename)

    E = float(match_E.group(1))
    z = float(match_z.group(1))
    qhat = float(match_qhat.group(1))
    Lu = float(match_Lu.group(1))
    Lv = float(match_Lv.group(1))
    Nu1 = int(match_Nu1.group(1))
    Nu2 = int(match_Nu2.group(1))
    Nv1 = int(match_Nv1.group(1))
    Nv2 = int(match_Nv2.group(1))
    L = float(match_L.group(1))

    #import system parameters
    sis = phsys(E, z, qhat, Lu, Lv) 
    sis.set_dim(Nu1,Nu2,Nv1,Nv2)  
    sis.set_t(L)

    fsol = np.load(dir + filename)
    fsol = cp.asarray(fsol)

    Theta = np.arange(0.1, 0.9, 0.1)
    Fp_an  = .0 * Theta
    Fp_sim = .0 * Theta

    for th in range(len(Theta)):

        fp = np.real(fasit2Ncdiagint(sis.t, sis.omega*Theta[th], 0, sis.z, sis.omega, sis.Omega, sis.qhat))

        Fp_prime = compute_fourier_torch_chunks(sis, fsol, sis.omega*Theta[th],
                                    chunksize_U1 = 8, 
                                    chunksize_U2 = 8).get()

        Fp_an[th] = Theta[th]*fp
        Fp_sim[th] = Theta[th]*Fp_prime

        print("Theta = c",Theta[th])
        print("Analytical = ", Theta[th]**2 / 2 * fp)
        print("Simulation = ", Theta[th]**2 / 2 *Fp_prime)

    dir = "fourier_results/"
    np.save(dir + "ft_" + filename, np.array([Theta, Fp_an, Fp_sim]))

    #do plotting
    plt.figure(figsize=(10, 6))
    plt.plot(Theta, Fp_an, label='Analytical', marker='o')
    plt.plot(Theta, Fp_sim, label='Simulation', marker='x')
    plt.xlabel('Theta')
    plt.ylabel('Fp')
    plt.title('Comparison of Analytical and Simulation Fp')
    plt.legend()
    plt.grid()
    plt.savefig(dir + "ft_" + filename.replace('.npy', '.pdf'), format='pdf')
    plt.close()
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_ft.py <directory>")
        sys.exit(1)
    main("simulations/", sys.argv[1])