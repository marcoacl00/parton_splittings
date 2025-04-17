import sys
import os


# Add the root directory of the project to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('__file__'), '..')))

from parton_splittings import *

sis = phsys(100, 0.3, 1.5, 1.5) #E, z, qF, Medium size (grid)
sis.set_dim(32,32,32,32) #Grid dimensions
sis.init_fsol()
ht = 0.04 #time step

simul = simulate(sis, ht, 2, 5)