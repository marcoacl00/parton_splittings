from splittings_code import *
import time
import os
import subprocess
import argparse

def main():
    start_time = time.time()  

    
    sis = physys_new(E, z, qhat, Lk, Ll, Ncmode=NcMode, optimization="gpu") #simulation with GPU
    #sis = phsys(E, z, qhat, Lu, Lv, optimization="default", prec=np.float64) #simulation with CPU (much slower)

    sis.set_dim(Nk1,Nk2,Nl1,Nl2)   #Grid dimensions
    sis.init_fsol()

    dir = "simulations/"

    simul = simulate_new_momentum(sis, ht, L_medium)

    if not os.path.exists(dir):
        os.makedirs(dir)

    file_name = "E={}_z={}_qhat={}_Lk={}_Ll={}_Nk1={}_Nk2={}_Nl1={}_Nl2={}_L={}_NcMode={}.npy".format(
        E, z, qhat, Lk, Ll, Nk1, Nk2, Nl1, Nl2, L_medium, NcMode
    )

    end_time = time.time() 

    np.save(dir + file_name, sis.Fsol)

    #print time in hh:mm:ss
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Simulation completed in {int(hours):02}:{int(minutes):02}:{seconds:.2f}")
    print(f"Saved simulation to {file_name}")

    # Run plotting script with the generated file name

    plot_script = "plots_new.py"
    subprocess.run(["python3", plot_script, file_name])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run parton splitting simulation")
    parser.add_argument("--E", type=float, default=100, help="Energy in GeV")
    parser.add_argument("--z", type=float, default=0.1, help="Momentum fraction")
    parser.add_argument("--qhat", type=float, default=1.5, help="qhat in GeV^2/fm")
    parser.add_argument("--Lk", type=float, help="Lk value (default: 0.6*E)")
    parser.add_argument("--Ll", type=float, help="Ll value (default: 0.1*E)")
    parser.add_argument("--Nk1", type=int, default=70, help="Nk1 grid size")
    parser.add_argument("--Nk2", type=int, default=70, help="Nk2 grid size")
    parser.add_argument("--Nl1", type=int, default=31, help="Nl1 grid size")
    parser.add_argument("--Nl2", type=int, default=31, help="Nl2 grid size")
    parser.add_argument("--L_medium", type=float, default=2, help="Medium length in fm")
    parser.add_argument("--ht", type=float, default=0.0025, help="Time step")
    parser.add_argument("--NcMode", type=str, default=None, help="Nc mode (default: LNcFac)")
    args = parser.parse_args()

    # Set parameters from command line or defaults
    E = args.E
    z = args.z
    qhat = args.qhat
    Lk = args.Lk if args.Lk is not None else 0.6 * E
    Ll = args.Ll if args.Ll is not None else 0.1 * E
    Nk1 = args.Nk1
    Nk2 = args.Nk2
    Nl1 = args.Nl1
    Nl2 = args.Nl2
    L_medium = args.L_medium
    ht = args.ht
    NcMode = args.NcMode if args.NcMode is not None else "LNcFac"

    # Print parameters for debugging
    print(f"Parameters: E={E}, z={z}, qhat={qhat}, Lk={Lk}, Ll={Ll}, Nk1={Nk1}, Nk2={Nk2}, Nl1={Nl1}, Nl2={Nl2}, L_medium={L_medium}, ht={ht}, NcMode={NcMode}")

    main()