from main import *

#E[GeV], z, qF, L[fm], theta, NcMode
#NcMode: LargeNc or FiniteNc


parameters = np.loadtxt("run_card.txt")
print(parameters)

for n_par in range(len(parameters)):
    print("Parameters: ", parameters[n_par, :])

    N = int(parameters[n_par, 0])
    En = parameters[n_par, 1]
    z = parameters[n_par, 2]
    qF = parameters[n_par, 3]
    L = parameters[n_par, 4]
    theta = np.linspace(0.1, 0.9, int(parameters[n_par, 5]))
    NcMode = "LargeNc"

    main(N, En, z, qF, L, theta, NcMode)