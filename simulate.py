from parton_splittings_gpu import *
import time

start_time = time.time()  
#Params
E = 50 #GeV
z = 0.5
qhat = 1.5 #Gev^2/fm


#GRID SETTINGS
Lu = 10 #fm
Lv = 2 #fm
Nu1 = 80
Nu2 = 80
Nv1 = 60
Nv2 = 60

#Time
L_medium = 2 #fm

ht = 0.01 #time step


sis = phsys(E, z, qhat, Lu, Lv) #E, z, qF, Medium size (grid)
sis.set_dim(Nu1,Nu2,Nv1,Nv2)   #Grid dimensions
sis.init_fsol()


simul = simulate(sis, ht, L_medium)

dir = "simulations/"
file_name = "E={}_z={}_qhat={}_Lu={}_Lv={}_Nu1={}_Nu2={}_Nv1={}_Nv2={}_L={}.npy".format(E, z, qhat, Lu, Lv, Nu1, Nu2, Nv1, Nv2, L_medium)

end_time = time.time() 

np.save(dir + file_name, sis.Fsol)

print("Total time: ", end_time-start_time, " seconds")