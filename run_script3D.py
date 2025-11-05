import subprocess
from itertools import product
import numpy as np

def run_simulation(
    E, z, qhat, Nk, Nl, Npsi, Lk=None, Ll=None, L_medium=2, ht=0.0025, NcMode="FNc", vertex="q_qg"):

    cmd = [
        "python3", "simulate_3D.py",
        "--E", str(E),
        "--z", str(z),
        "--qhat", str(qhat),
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

if __name__ == "__main__":

    #This is where the simulation parameters (both physical and numerical) are set.
    #You can adjust these parameters to test different scenarios or configurations.
    #The parameters are passed to the simulate_new_momentum.py script.

    # Example usage: put here the parameters you want to test
    Z = [0.03]
    E = [200]
    L = [2]
    Qhat = [2]

    for z, En, Len, qhat in product(Z, E, L, Qhat):
        
        print(f"\n .: Running simulation for:  E = {En} GeV | z = {z} | L = {Len} fm | qhat = {qhat} GeV²/fm :. \n")
        
        alpha = 1.8
        beta = 1 * alpha
        run_simulation(
            E=En,
            z=z,
            qhat=qhat,

            #ajust Lk and Ll based on precision and performance considerationsz
            Nk = 120,
            Nl = 60,
            Npsi = 32,

            Lk =  alpha * (1-z) * z * En,
            Ll =  beta * (1-z) * z * En,

            L_medium=Len,

            ht= 0.01, #time step
            NcMode="FNc", #LNcFac, LNc, FNc
            vertex = "g_gg" # "gamma_qq", "q_qg" or "g_gg"
        )
