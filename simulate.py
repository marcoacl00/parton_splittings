from parton_splittings import *
import time

start_time = time.time()  
#Params
E = 10 #GeV
z = 0.5
qhat = 1.5 #Gev^2/fm


#GRID SETTINGS
Lu = 4 #fm
Lv = 2 #fm
Nu1 = 32
Nu2 = 32
Nv1 = 40
Nv2 = 40

#Time
L_medium = 2 #fm

ht = 0.01 #time step

sis = phsys(E, z, qhat, Lu, Lv) #simulation with GPU
#sis = phsys(E, z, qhat, Lu, Lv, optimization="default", prec=np.float64) #simulation with CPU (much slower)

sis.set_dim(Nu1,Nu2,Nv1,Nv2)   #Grid dimensions
sis.init_fsol()


simul = simulate(sis, ht, L_medium)

dir = "simulations/"
file_name = "E={}_z={}_qhat={}_Lu={}_Lv={}_Nu1={}_Nu2={}_Nv1={}_Nv2={}_L={}.npy".format(E, z, qhat, Lu, Lv, Nu1, Nu2, Nv1, Nv2, L_medium)

end_time = time.time() 

np.save(dir + file_name, sis.Fsol)

print("Total time: ", end_time-start_time, " seconds")