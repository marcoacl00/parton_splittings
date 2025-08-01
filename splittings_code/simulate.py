from splittings_code import *
import time

start_time = time.time()  
#Params
E = 100 #GeV
z = 0.001
qhat = 1.5 #Gev^2/fm


#GRID SETTINGS
Lu = 2 #fm
Lv = 2 #fm
Nu1 = 50
Nu2 = 50
Nv1 = 50 
Nv2 = 50

#Time
L_medium = 2 #fm

ht = 0.01 #time step

#-----------------#
#       GPU       #
#-----------------#
sis = phsys(E, z, qhat, Lu, Lv, optimization="cpu") 

#-----------------#
#       CPU       #
#-----------------#
#sis = phsys(E, z, qhat, Lu, Lv, optimization="default", prec=np.float64)


sis.set_dim(Nu1,Nu2,Nv1,Nv2)   
sis.init_fsol()


simul = simulate(sis, ht, L_medium)

dir = "simulations/"
file_name = "E={}_z={}_qhat={}_Lu={}_Lv={}_Nu1={}_Nu2={}_Nv1={}_Nv2={}_L={}.npy".format(E, z, qhat, Lu, Lv, Nu1, Nu2, Nv1, Nv2, L_medium)

end_time = time.time() 

np.save(dir + file_name, sis.Fsol)

elapsed_time = end_time - start_time

#print time in hh:mm:ss
elapsed_time = end_time - start_time
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Simulation completed in {int(hours):02}:{int(minutes):02}:{seconds:.2f}")
print(f"Saved simulation to {dir + file_name}")