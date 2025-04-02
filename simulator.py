from faber import *
from fitterNN import *
import matplotlib.pyplot as plt

from joblib import Parallel, delayed


def simulate(sist, ht, t_L):
    cont = 0

    sis = sist

    def compute_nHom_chunk(i1, i2, j1, j2, sis, t):
        """
        Compute a single element of nHom for the given indices.
        """
        return sis.source_term(t, i1, i2, j1, j2)

    
    def compute_nHom_parallel(sis, t):
        """
        Parallel computation of nHom using joblib.
        """
        nHom = np.zeros_like(sis.Fsol, dtype=np.complex128)  # Initialize nHom with the same shape as Fsol

        # Create a list of all index combinations
        index_combinations = [
            (i1, i2, j1, j2)
            for i1 in range(1, sis.Nu1 - 1)
            for i2 in range(1, sis.Nu2 - 1)
            for j1 in range(1, sis.Nv1 - 1)
            for j2 in range(1, sis.Nv2 - 1)
        ]

        # Parallel computation of source_term for all index combinations
        results = Parallel(n_jobs=-1)(  # Use all available CPU cores
            delayed(compute_nHom_chunk)(i1, i2, j1, j2, sis, t)
            for i1, i2, j1, j2 in index_combinations
        )

        # Assign the results back to nHom
        for idx, (i1, i2, j1, j2) in enumerate(index_combinations):
            nHom[:, i1, i2, j1, j2] = results[idx]

        return nHom
    
    

    while sis.t < t_L:

        print("Processing t = ", sis.t)

        nHom = compute_nHom_parallel(sis, sis.t-ht)
        
        #construct Faber evolved solution
        sis.set_fsol(sis.Fsol + 1/2 * nHom * ht)
        f_sol_n = faber_expand(sis, ht)

        #Build non homogneous term at t = sis.t + ht
        nHom = compute_nHom_parallel(sis, sis.t)

        #complete the trapezoidal rule
        f_sol_n += 1/2 * nHom * ht

        #update fsol
        sis.set_fsol(f_sol_n)

        if cont%5 == 0:
            plt.plot(sis.U1, np.real(sis.Fsol[1, :, sis.Nu2//2, sis.Nv1//2, sis.Nv2//2]), label = "Re.")
            plt.plot(sis.U1, np.imag(sis.Fsol[1, :, sis.Nu2//2, sis.Nv1//2, sis.Nv2//2]), label = "Im.")
            plt.xlabel("$u_x$")
            plt.ylabel("$F(u_x, 0, 0, 0)$")
            plt.legend()
            plt.show()

            plt.plot(sis.V1, np.real(sis.Fsol[1,  sis.Nu1//2,  sis.Nu2//2, :, sis.Nv2//2 ]), label = "Real.")
            plt.plot(sis.V1, np.imag(sis.Fsol[1, sis.Nu1//2,  sis.Nu2//2, :,  sis.Nv2//2 ]), label = "Imag.")
            plt.xlabel("$u_x$")
            plt.ylabel("$F(0, 0, v_x, 0)$")
            plt.legend()
            plt.show()
        
        cont +=1
        sis.increase_t(ht)

        print(sist.t)
    
    return sist


