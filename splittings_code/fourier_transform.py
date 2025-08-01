import cupy as cp
from cupyx.scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

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

def compute_fourier_chunks_interp(sis, f_sol_interp, p,
                                   chunksize_U1=4,
                                   chunksize_U2=4,
                                   Nu1=512,
                                   Nu2=256,
                                   Nv1=512,
                                   Nv2=128):
    
    # Define uniform grids on CPU (NumPy, for interpolator)
    U1 = cp.linspace(-sis.Lu / 2, sis.Lu / 2, Nu1)
    U2 = cp.linspace(-sis.Lu / 2, sis.Lu / 2, Nu2)
    V1 = cp.linspace(-sis.Lv / 2, sis.Lv / 2, Nv1)
    V2 = cp.linspace(-sis.Lv / 2, sis.Lv / 2, Nv2)

    # Grid steps and volume element
    du1 = sis.Lu / (Nu1 - 1)
    du2 = sis.Lu / (Nu2 - 1)
    dv1 = sis.Lv / (Nv1 - 1)
    dv2 = sis.Lv / (Nv2 - 1)
    dVol = du1 * du2 * dv1 * dv2

    # Transfer static vectors to GPU
    V1_cp = cp.asarray(V1).reshape(1, 1, -1, 1)
    V2_cp = cp.asarray(V2).reshape(1, 1, 1, -1)

    total_sum = cp.array(0.0 + 0.0j, dtype=cp.complex64)

    beta = sis.beta(sis.t)

    for start1 in tqdm(range(0, Nu1, chunksize_U1)):
        for start2 in range(0, Nu2, chunksize_U2):

            end1 = min(start1 + chunksize_U1, Nu1)
            end2 = min(start2 + chunksize_U2, Nu2)

            # Grid chunks (on GPU)
            U1_chunk = U1[start1:end1]
            U2_chunk = U2[start2:end2]

            # Meshgrid in 4D
            U1m, U2m, V1m, V2m = cp.meshgrid(U1_chunk, U2_chunk, V1_cp.flatten(), V2_cp.flatten(), indexing='ij')

            input_points = cp.stack([U1m, U2m, V1m, V2m], axis=-1)
            # Interpolate (f_sol_interp is CPU-side RegularGridInterpolator)
            f_interp_cp = f_sol_interp(input_points)
            
            # Phase factor (still on GPU)
            U1_t = U1_chunk.reshape(-1, 1, 1, 1)
            U2_t = U2_chunk.reshape(1, -1, 1, 1)

            f_interp_cp *= cp.exp(1j * (beta * (U1_t**2 + U2_t**2) - p * (U1_t - V1_cp)))

            chunk_sum = cp.sum(f_interp_cp)
            total_sum += chunk_sum

    return cp.real(total_sum) * dVol


def compute_fourier_chunks_hybrid(sis, p, fsol,
                                  chunk_size_U1 = 16,
                                  chunk_size_U2 = 16):
    """"Compute the partial Fourier transform of F(ux, uy, qx, qy) on the ux and uy axis"""
    du1 = sis.du1
    du2 = sis.du2
    closest_q =  cp.argmin(cp.abs(sis.Q1 + p))
    
    F_u = fsol[:, :, :, closest_q, -1].copy()  

    print("Value of the momentum p:", p)
    print("Closest value in sis.Q1:", sis.Q1[closest_q])


    dVol = du1 * du2
    total_sum = 0
    for start1 in range(0, len(sis.U1), chunk_size_U1):
        for start2 in range(0, len(sis.U2), chunk_size_U2):

            end1 = min(start1 + chunk_size_U1, len(sis.U1))
            end2 = min(start2 + chunk_size_U2, len(sis.U2))

            U1_chunk = sis.U1[start1:end1, None]
            U2_chunk = sis.U2[None, start2:end2]

            arg = sis.beta(sis.t) * (U1_chunk**2 + U2_chunk**2) - p * U1_chunk

            chunk_sum = cp.sum(F_u[1, start1:end1, start2:end2] * cp.exp(1j * arg))

            total_sum += chunk_sum

    return cp.real(total_sum) * dVol


def compute_fourier_chunks_hybrid_interp(sis, fsol, p,
                                          chunk_size_U1=128,
                                          chunk_size_U2=128,
                                          Nu1=1024,
                                          Nu2=512):
    """Compute the partial Fourier transform of F(ux, uy, qx, qy) on the ux and uy axis using interpolation"""
    # Define uniform grids on CPU (NumPy, for interpolator)
    U1 = cp.linspace(-sis.Lu / 2, sis.Lu / 2, Nu1)
    U2 = cp.linspace(-sis.Lu / 2, sis.Lu / 2, Nu2)
    # Create a RegularGridInterpolator for the fsol

    fsol_interp = RegularGridInterpolator(
        (sis.U1, sis.U2, sis.Q1, sis.Q2), 
        fsol[1, :, :, :, :], 
        method='linear', 
        bounds_error=False, 
    )

    # Define uniform grids on GPU
    # Grid steps and volume element
    du1 = sis.Lu / (Nu1 - 1)
    du2 = sis.Lu / (Nu2 - 1)
    dVol = du1 * du2

    total_sum = cp.array(0.0 + 0.0j, dtype=cp.complex64)

    beta = sis.beta(sis.t)

    for start1 in tqdm(range(0, Nu1, chunk_size_U1)):
        for start2 in range(0, Nu2, chunk_size_U2):

            end1 = min(start1 + chunk_size_U1, Nu1)
            end2 = min(start2 + chunk_size_U2, Nu2)

            # Grid chunks (on GPU)
            U1_chunk = U1[start1:end1]
            U2_chunk = U2[start2:end2]

            # Meshgrid in 4D
            U1m, U2m = cp.meshgrid(U1_chunk, U2_chunk, indexing='ij')
            num_points = U1m.size
            input_points_cpu = cp.stack([
                U1m.ravel(),
                U2m.ravel(),
                -p * cp.ones(num_points),
                cp.zeros(num_points)
            ], axis=-1)
            # Interpolate (fsol_interp is CPU-side RegularGridInterpolator)
            f_interp_cp = fsol_interp(input_points_cpu).reshape(U1m.shape)
            # Phase factor (still on GPU)
            U1_chunk = U1_chunk.reshape(-1, 1)
            U2_chunk = U2_chunk.reshape(1, -1)


            f_interp_cp *= cp.exp(1j * (beta * (U1_chunk**2 + U2_chunk**2) - p * U1_chunk))
            chunk_sum = cp.sum(f_interp_cp)
            # Accumulate the result
            total_sum += chunk_sum

    return cp.real(total_sum) * dVol
