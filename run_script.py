import subprocess


def run_simulation(
    E, z, qhat, Nk1, Nk2, Nl1, Nl2, Lk=None, Ll=None, L_medium=2, ht=0.0025, NcMode="FNc"):

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

    #This is where the simulation parameters (both physical and numerical) are set.
    #You can adjust these parameters to test different scenarios or configurations.
    #The parameters are passed to the simulate_new_momentum.py script.

    # Example usage: put here the parameters you want to test
    Z = [0.2]
    E = [100]
    L = [6]
    Qhat = [2.0]

    for z, En, Len, qhat in zip(Z, E, L, Qhat):
        
        print(f"\n .: Running simulation for:  E = {En} GeV | z = {z} | L = {Len} fm | qhat = {qhat} GeV²/fm :. \n")
        
        alpha = 1.2
        beta = 0.4 * alpha
        run_simulation(

            E=En,
            z=z,
            qhat=qhat,

            #ajust Lk and Ll based on precision and performance considerations
            Nk1=100, #what will actually be plotted
            Nk2=100, #this axis will be set to 0... maybe this is too much?

            Nl1= 31, #these have effect on the non-factorized versions. we only want the value at the origin
            Nl2= 31, 

            #alpha and beta regulates the limits of the momentum grid. since omega = z * (1 - z) * E, limit up to theta = 1 requires alpha = 1. Adjust taking into account behaviour of the med modification factor and the boundary errors.
            #Warning: values near the boundaries of the grid are not reliable, so be careful with the values of alpha and beta.

            Lk =  alpha * (1-z) * z * En,
            Ll =  beta * (1-z) * z * En,

            L_medium=Len,

            ht=0.01,
            NcMode="FNc" #LNcFac, LNc, FNc
        )
