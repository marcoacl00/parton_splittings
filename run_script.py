import subprocess
import numpy as np


def run_simulation(
    E, z, qhat, Nk1, Nk2, Nl1, Nl2, Lk=None, Ll=None, L_medium=2, ht=0.0025, NcMode="FNc"
):
    cmd = [
        "python3", "simulate_new_momentum.py",
        "--E", str(E),
        "--z", str(z),
        "--qhat", str(qhat),
        "--Nk1", str(Nk1),
        "--Nk2", str(Nk2),
        "--Nl1", str(Nl1),
        "--Nl2", str(Nl2),
        "--L_medium", str(L_medium),
        "--ht", str(ht),
        "--NcMode", str(NcMode)
    ]
    if Lk is not None:
        cmd += ["--Lk", str(Lk)]
    if Ll is not None:
        cmd += ["--Ll", str(Ll)]

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # Example usage: run with default parameters
    Z = [0.1]

    print(Z)

    for z in Z:

        print(f"Running simulation for z = {z}"
              )
        
        En = 500
        alpha = 0.6
        beta = 0.1
        run_simulation(
            E=En,
            z=z,
            qhat=3,
            Nk1=100,
            Nk2=100,
            Nl1= 27,
            Nl2= 27,
            Lk =  alpha * (1-z) * z * En,
            Ll =  beta * (1-z) * z * En,
            L_medium=3,
            ht=0.01,
            NcMode="FNc" #LNcFac, LNc, FNc
        )
