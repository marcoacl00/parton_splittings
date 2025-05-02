from parton_splittings_gpu import *
import cupy as cp

def compute_fourier_torch_chunks(sis, f_sol, p,
                                 chunksize_U1 = 8, 
                                 chunksize_U2 = 8):
    

    du1 = sis.du1
    du2 = sis.du2
    dv1 = sis.dv1
    dv2 = sis.dv2
    dVol = du1 * du2 * dv1 * dv2

    total_sum = 0

    V1 = sis.V1[None, None, :, None]
    
    
    chunk_cont = 1
    for start1 in range(0, len(sis.U1), chunksize_U1):
        for start2 in range(0, len(sis.U2), chunksize_U2):
        

            end1 = min(start1 + chunksize_U1, len(sis.U1))
            end2 = min(start2 + chunksize_U2, len(sis.U2))

            U1_chunk = sis.U1[start1:end1, None, None, None]
            
            U2_chunk = sis.U2[None, start2:end2, None, None]

            arg = sis.beta(sis.t) * (U1_chunk**2 + U2_chunk**2) - p * (U1_chunk - V1)

            chunk_sum = cp.sum(f_sol[1, start1:end1, start2:end2, :, :]*cp.exp(1j * arg))

            # Accumulate the result
            
            total_sum += chunk_sum

            chunk_cont +=1

    return cp.real(total_sum) * dVol