from .faber import *
import matplotlib.pyplot as plt

#------Simulation function------#

def simulate(sist, ht, t_L, step_save = 10):

    sis = sist

    ht_cp = 1.0 * ht

    while sis.t < t_L:

        if sis.t < 0.1:
            ht = 5e-1 * ht_cp
        else: 
            ht = ht_cp

        print("Processing t = ", round(sis.t, 3))

        
        #construct Faber evolved solution
        nHom = sis.source_term_array(sis.t)
        print("First nHom computed")
        nFsol = sis.Fsol
        nFsol[0, 1:-1, 1:-1, 1:-1, 1:-1] +=  nHom * ht / 2
        nFsol[1, 1:-1, 1:-1, 1:-1, 1:-1] +=  nHom * ht / 2

        sis.set_fsol(nFsol)

        f_sol_n = faber_expand(sis, ht)

        #Build non homogneous term at t = sis.t + ht
        nHom = sis.source_term_array(sis.t)
        print("Second nHom computed")

        #complete the trapezoidal rule
        nFsol = f_sol_n
        nFsol[0, 1:-1, 1:-1, 1:-1, 1:-1] +=  nHom * ht / 2
        nFsol[1, 1:-1, 1:-1, 1:-1, 1:-1] +=  nHom * ht / 2

        sis.set_fsol(nFsol)

        

        # if cont%step_save == 0:
            # plt.plot(sis.U1.get(), np.real(sis.Fsol[1, :, sis.Nu2//2, sis.Nv1//2, sis.Nv2//2].get()), 
            #          label = "Re.")
            # plt.plot(sis.U1.get(), np.imag(sis.Fsol[1, :, sis.Nu2//2, sis.Nv1//2, sis.Nv2//2].get()), 
            #          label = "Im.")
            # plt.xlabel("$u_x$")
            # plt.ylabel("$F(u_x, 0, 0, 0)$")
            # plt.legend()
            # name = "tu_{}".format(round(sis.t, 3))
            # plt.savefig("plot_images/" + name + ".png")
            # plt.close()

            # plt.plot(sis.V1.get(), np.real(sis.Fsol[1,  sis.Nu1//2,  sis.Nu2//2, :, sis.Nv2//2 ].get()), 
            #          label = "Real.")
            # plt.plot(sis.V1.get(), np.imag(sis.Fsol[1, sis.Nu1//2,  sis.Nu2//2, :,  sis.Nv2//2 ].get()), 
            #          label = "Imag.")
            # plt.xlabel("$u_x$")
            # plt.ylabel("$F(0, 0, v_x, 0)$")
            # plt.legend()
            # name = "tv_{}".format(round(sis.t, 3))
            # plt.savefig("plot_images/" + name+ ".png")
            # plt.close()
            
        
        # cont +=1
        
        sis.increase_t(ht)
    
    return sis


