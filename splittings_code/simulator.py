from .faber import *
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


#------Simulation function------#

def simulate(sist, ht, t_L, step_save = 10):

    sis = sist

    ht_cp = 1.0 * ht
    cont = 0
    while sis.t < t_L:

        if sis.t < 0.1:
            ht = 5e-1 * ht_cp
        else: 
            ht = ht_cp

        #construct Faber evolved solution
        nHom = sis.source_term_array(sis.t)
        print("First nHom computed")
        nFsol = sis.Fsol
        nFsol[0] +=  nHom * ht / 2
        nFsol[1] +=  nHom * ht / 2

        sis.set_fsol(nFsol)


        if cont%step_save == 0:
            plt.plot(sis.U1.get(), np.real(sis.Fsol[1, :, sis.Nu2//2, sis.Nv1//2, sis.Nv2//2].get()), 
                     label = "Re.")
            plt.plot(sis.U1.get(), np.imag(sis.Fsol[1, :, sis.Nu2//2, sis.Nv1//2, sis.Nv2//2].get()), 
                     label = "Im.")
            plt.xlabel("$u_x$")
            plt.ylabel("$F(u_x, 0, 0, 0)$")
            plt.legend()
            name = "tu_{}".format(round(sis.t, 3))
            plt.savefig("plot_images/" + name + ".png")
            plt.close()

            plt.plot(sis.V1.get(), np.real(sis.Fsol[1,  sis.Nu1//2,  sis.Nu2//2, :, sis.Nv2//2 ].get()), 
                     label = "Real.")
            plt.plot(sis.V1.get(), np.imag(sis.Fsol[1, sis.Nu1//2,  sis.Nu2//2, :,  sis.Nv2//2 ].get()), 
                     label = "Imag.")
            plt.xlabel("$u_x$")
            plt.ylabel("$F(0, 0, v_x, 0)$")
            plt.legend()
            name = "tv_{}".format(round(sis.t, 3))
            plt.savefig("plot_images/" + name+ ".png")
            plt.close()

        print("Processing t = ", round(sis.t, 3))

        
        

        f_sol_n = faber_expand(sis, ht)

        #Build non homogneous term at t = sis.t + ht
        nHom = sis.source_term_array(sis.t)
        print("Second nHom computed")

        #complete the trapezoidal rule
        nFsol = f_sol_n
        nFsol[0] +=  nHom * ht / 2
        nFsol[1] +=  nHom * ht / 2

      
        sis.set_fsol(nFsol)
            
        
        cont +=1
        
        sis.increase_t(ht)
    
    return sis



def simulate_new_momentum(sist, ht, t_L, step_save = 10):
    """Simulates the system using the momentum method in the (k ,l) space."""
    
    sis = sist

    direc = "plot_images/"
    if not os.path.exists(direc):
        os.makedirs(direc)

    time_list = np.arange(1e-3, t_L, ht)

    lambF, gamma0, gamma1 = faber_params_new_momentum(sist)
    one_lamb = 1/lambF

    print("Faber params: lamb = ", lambF, "gamma0 = ", gamma0, "gamma1 = ", gamma1)
    
    m = 1
    coeff_array = [coeff(0, ht, gamma0, gamma1, lambF, sist.optimization)] #compute first coefficient

    while (np.abs(coeff_array[-1]) > 1e-7 or m < 10):
        coeff_array.append(coeff(m, ht, gamma0, gamma1, lambF, sist.optimization))
        m += 1

    print("Number of polynomials = ", m)

    for _ in tqdm(time_list):

        #construct Faber evolved solution
        nHom = sis.source_term_array(sis.t)
        nFsol = sis.Fsol
        nFsol[0] +=  nHom * ht / 2
        nFsol[1] +=  nHom * ht / 2

        sis.set_fsol(nFsol)

        if _%step_save == 0:
            # Do some plotting to monitor the evolution
            plt.plot(sis.K1.get(), np.real(sis.Fsol[1, :, 0, sis.Nl1//2, sis.Nl2//2].get()), 
                        label = "Re.")
            plt.plot(sis.K1.get(), np.imag(sis.Fsol[1, :, 0, sis.Nl1//2, sis.Nl2//2].get()), 
                        label = "Im.")
            plt.xlabel("$k_x$")
            plt.ylabel("$F(k_x, 0, 0, 0)$")
            plt.legend()
            name = "tk_{}".format(round(sis.t, 3))
            plt.savefig(direc + name + ".png")
            plt.close()

            plt.plot(sis.L1.get(), np.real(sis.Fsol[1,  sis.Nk1//2,  sis.Nk2//2,
                        :, sis.Nl2//2 ].get()), 
                        label = "Real.")
            
            plt.plot(sis.L1.get(), np.imag(sis.Fsol[1, sis.Nk1//2,  sis.Nk2//2,
                        :,  sis.Nl2//2 ].get()), 
                        label = "Imag.")
            plt.xlabel("$l_x$")
            plt.ylabel("$F(Lk/2, Lk/2, l_x, 0)$")
            plt.legend()
            name = "tl_{}".format(round(sis.t, 3))
            plt.savefig(direc + name + ".png")
            plt.close()

        
        f_sol_n = faber_expand_new_momentum(sis, ht, gamma0, gamma1, one_lamb, coeff_array)

        #Build non homogneous term at t = sis.t + ht
        nHom = sis.source_term_array(sis.t + ht)

        #complete the trapezoidal rule
        nFsol = f_sol_n
        nFsol[0] +=  nHom * ht / 2
        nFsol[1] +=  nHom * ht / 2

        # Set the new solution
        sis.set_fsol(nFsol)

        # NaN-safety check
        if np.isnan(nHom).any():
            print("Warning: NaN detected in nHom at t = {}".format(sis.t))
        if np.isnan(nFsol).any():
            raise ValueError("NaN detected in nFsol at t = {}".format(sis.t))   
        
        sis.increase_t(ht)

    print("Final time = ", round(sis.t, 3))
    return sis