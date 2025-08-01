This code is executed using the run.script.py, for the momentum simulation with good coordinates. There it's possible to change all simulation parameters.

It is extremely recommended running this code using an NVIDIA GPU (CUDA: CuPy), as it takes substantially less time. CPUs (through numpy) can be used for for very small grids, as its version is not fully optimized yet. If using CPU, you should comment the line "import cupy" if you don't have it installed, in whatever files it shows up.

The position space simulation is not automatized like the momentum space one. But all ingredients are there (simulate_pos.py -> get_ft.py).

**Requirements**

- Numpy, Scipy, Cupy (CUDA), Matplotlib

**KNOWN ISSUES**

- The code is still lacking a proper boundary treatment, so most results will have a poor behavior near it (and in some cases, most of the domain gets affected)

- The Fmed^in-in -> 2 for large theta (and large energies) is still not completely observed, as there are systematic issues that affect this limit (possibly due to boundaries)

- Very high energies are still hard to resolve, specially for low medium length (L <~ 1fm). The solution is extremely oscillatory in this regime; however, it is also true that the modification factor is very small in these regimes, and the finite Nc corrections don't have much impact here


**TO BE DONE**

- We are using 4D grid for (kx, ky, lx, ly). I believe that moving to a polar system of coordinates would allow to reduce the numerica complexity on the (kx, ky) subspace, as we only care about the norm of (kx, ky) 

- Fine tune the boundary condition (maybe think of something around Fmed_in_out)

- A full numerical analysis: varying parameters, do some analytical estimations of numerical errors and corrections, etc.

- Get the q -> qg vertex 
