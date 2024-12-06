import torch.cuda
from numba import cuda
import math


@cuda.jit
def sigkernel_norms(M_inc, C, norms, len_x, n_anti_diagonals, M_sol, M_sol2, d_order, L, N):
    """
    We start from a list of pairs of paths [(x^1,y^1), ..., (x^n, y^n)]
    M_inc: a 3-tensor D[i,j,k] = <x^i_j, y^i_k>.
    n_anti_diagonals = 2 * max(len_x, len_y) - 1
    M_sol: a 3-tensor storing the solutions of the PDEs.
    """

    # Each block corresponds to a pair (x_i,y_i).
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    
    K1 = 0
    K2 = 2
    K3 = 1

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for _ in range(N):
        for p in range(2, n_anti_diagonals):
            for l in range(L):
                i = thread_id * L + l + 1
                j = p - i
                
                if i < min(len_x, p) and j < len_x:
                    inc = M_inc[block_id, (i - 1) >> d_order, (j - 1) >> d_order]
                    
                    k_01 = 1.0 if i == 1 else M_sol[block_id, i - 2, K2]
                    k_10 = 1.0 if j == 1 else M_sol[block_id, i - 1, K2]
                    k_00 = 1.0 if j == 1 or i == 1 else M_sol[block_id, i - 2, K3]
                    
                    k_01_2 = 0.0 if i == 1 else M_sol2[block_id, i - 2, K2]
                    k_10_2 = 0.0 if j == 1 else M_sol2[block_id, i - 1, K2]
                    k_00_2 = 0.0 if j == 1 or i == 1 else M_sol2[block_id, i - 2, K3]
                                
                    M_sol[block_id, i - 1, K1] = (k_01 + k_10)*(1. + 0.5 * inc * C[block_id]**2 + (1./12) * inc**2 * C[block_id]**4) - k_00 * (1. - (1./12) * inc**2 * C[block_id]**4)
                    #M_sol2[block_id, i - 1, K1] = (k_01_2 + k_10_2)*(1. + 0.5 * inc * C[block_id]**2 + (1./12) * (inc * C[block_id]**2)**2) - k_00_2 * (1. - (1./12) * (inc * C[block_id]**2)**2)
                    #M_sol2[block_id, 1 - 1, K1] += (k_01 + k_10) * (0.5 * inc + (1./12) * inc**2) + k_00 * ((1./12) * inc**2)
                    M_sol2[block_id, i - 1, K1] = k_01_2 + k_10_2 - k_00_2 + inc * (C[block_id]**2 * (k_01_2 + k_10_2) + C[block_id] * (k_01 + k_10))
                    
                    if p == n_anti_diagonals - 1:
                        M_sol[block_id, 0, 0] = M_sol[block_id, i - 1, K1]
                        M_sol2[block_id, 0, 0] = M_sol2[block_id, i - 1, K1]

            K1 = K1 ^ K2 ^ K3
            K2 = K1 ^ K2 ^ K3
            K3 = K1 ^ K2 ^ K3
            K1 = K1 ^ K2 ^ K3
                
            cuda.syncthreads()
            
        if thread_id == 0:
            C[block_id] = C[block_id] - (M_sol[block_id, 0, 0] - norms[block_id]) / M_sol2[block_id, 0, 0]
            
        cuda.syncthreads()