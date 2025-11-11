import subprocess
from itertools import product
import numpy as np

def run_simulation(
    E, z, qtilde, Nk, Nl, Npsi, Lk=None, Ll=None, L_medium=2, ht=0.01, NcMode="FNc", vertex="q_qg"):

    cmd = [
        "python3", "simulate_3D.py",
        "--E", str(E),
        "--z", str(z),
        "--qtilde", str(qtilde),
        "--Nk", str(Nk),
        "--Nl", str(Nl),
        "--Npsi", str(Npsi),
        "--L_medium", str(L_medium),
        "--ht", str(ht),
        "--NcMode", str(NcMode)
        ,"--vertex", str(vertex)
    ]
    if Lk is not None:
        cmd += ["--Lk", str(Lk)]
    if Ll is not None:
        cmd += ["--Ll", str(Ll)]
 

    subprocess.run(cmd, check=True)


def alpha_(omega):
    if omega < 1:
        return 10
    if omega < 1.5:
        return 7
    if omega < 2:
        return 3.5
    if omega < 3:
        return 2.5
    if omega < 5:
        return 2
    if omega < 10:
        return 1.6
    if omega < 15:
        return 1.4
    if omega < 20:
        return 1.3
    if omega < 30:
        return 1.2
    if omega < 50:
        return 1
    if omega < 70:
        return 0.8
    if omega < 100:
        return 0.7
    return 0.6


if __name__ == "__main__":

    #This is where the simulation parameters (both physical and numerical) are set.
    #You can adjust these parameters to test different scenarios or configurations.
    #The parameters are passed to the simulate_new_momentum.py script.

    # Example usage: put here the parameters you want to test
    Zsym = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    Zasym = [1 - Zsym[i] for i in range(len(Zsym)) if Zsym[i] != 0.5]

    Z = sorted(Zsym + Zasym)
    #Z = [0.5]
    E = [500]
    L = [4]

    Qtilde = [1.5] #this is qF/CF, which is color independent

    print("Computing simulations for the following z values:")
    print(Z)
    for z, En, Len, qtilde in product(Z, E, L, Qtilde):
        
        print(f"\n .: Running simulation for:  E = {En} GeV | z = {z} | L = {Len} fm | qtilde = {qtilde} GeV²/fm :. \n")
        
        alpha = alpha_(En*z*(1-z))
        beta = 1 * alpha
        print("alpha =", alpha)

        run_simulation(
            E=En,
            z=z,
            qtilde=qtilde,

            #ajust Lk and Ll based on precision and performance considerationsz
            Nk = 120 if z <= 0.1 or z >= 0.9 else 200,
            Nl = 60,
            Npsi = 32,

            Lk =  alpha * (1-z) * z * En,
            Ll =  beta * (1-z) * z * En,

            L_medium=Len,

            ht= 0.01, #time step
            NcMode="LeadNc", #LNcFac, LNc, FNc
            vertex = "q_qg" # "gamma_qq", "q_qg" or "g_gg"
        )
