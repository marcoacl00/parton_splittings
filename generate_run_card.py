import os
import numpy as np
from itertools import product

output_name = "run_card.txt"

N_range = np.array([32])
E_range = np.array([10.0, 100.0])
z_range = np.arange(0.1, 0.800, 0.2).round(4)
qF_range = np.array([1.5])
L_range = np.array([2.0])
theta_range = np.array([3])

with open(output_name, 'w') as file:
        for N, E, z, qF, L, theta in product(N_range,E_range, z_range, qF_range, 
                                      L_range, theta_range):
            file.write(f"{N} \t{E} \t {z}\t {qF} \t {L} \t {theta} \n")

