import torch.cuda
from numba import cuda
import math

@cuda.jit(device=True)
def cache_offset(n, m, j):
    s = 0
    for i in range(j):
        s += min(n - i, m)
    return s

@cuda.jit
def sigkernel_cuda2(M_inc, len_x, len_y, n_anti_diagonals, M_sol, d_order, L):
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
    for p in range(2, n_anti_diagonals):
        for l in range(L):
            i = thread_id * L + l + 1
            j = p - i
            
            if i < min(len_x, p) and j < len_y:
                inc = M_inc[block_id, (i - 1) >> d_order, (j - 1) >> d_order]
                
                k_01 = 1.0 if i == 1 else M_sol[block_id, i - 2, K2]
                k_10 = 1.0 if j == 1 else M_sol[block_id, i - 1, K2]
                k_00 = 1.0 if j == 1 or i == 1 else M_sol[block_id, i - 2, K3]
                               
                M_sol[block_id, i - 1, K1] = (k_01 + k_10)*(1. + 0.5*inc + (1./12)*inc**2) - k_00*(1. - (1./12)*inc**2)
                
                if p == n_anti_diagonals - 1:
                    M_sol[block_id, 0, 0] = M_sol[block_id, i - 1, K1]

        K1 = K1 ^ K2 ^ K3
        K2 = K1 ^ K2 ^ K3
        K3 = K1 ^ K2 ^ K3
        K1 = K1 ^ K2 ^ K3
            
        cuda.syncthreads()
        
@cuda.jit
def sigkernel_gram_cuda2(M_inc, len_x, len_y, n_anti_diagonals, M_sol, d_order, L):
    """
    We start from a list of pairs of paths [(x^1,y^1), ..., (x^n, y^n)]
    M_inc: a 3-tensor D[i,j,k] = <x^i_j, y^i_k>.
    n_anti_diagonals = 2 * max(len_x, len_y) - 1
    M_sol: a 3-tensor storing the solutions of the PDEs.
    """

    # Each block corresponds to a pair (x_i,y_i).
    block_x = cuda.blockIdx.x
    block_y = cuda.blockIdx.y
    thread_id = cuda.threadIdx.x
    
    K1 = 0
    K2 = 2
    K3 = 1

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(2, n_anti_diagonals):
        for l in range(L):
            i = thread_id * L + l + 1
            j = p - i
            
            if i < min(len_x, p) and j < len_y:
                inc = M_inc[block_x, block_y, (i - 1) >> d_order, (j - 1) >> d_order]
                
                k_01 = 1.0 if i == 1 else M_sol[block_x, block_y, i - 2, K2]
                k_10 = 1.0 if j == 1 else M_sol[block_x, block_y, i - 1, K2]
                k_00 = 1.0 if j == 1 or i == 1 else M_sol[block_x, block_y, i - 2, K3]
                               
                M_sol[block_x, block_y, i - 1, K1] = (k_01 + k_10)*(1. + 0.5*inc + (1./12)*inc**2) - k_00*(1. - (1./12)*inc**2)
                
                if p == n_anti_diagonals - 1:
                    M_sol[block_x, block_y, 0, 0] = M_sol[block_x, block_y, i - 1, K1]

        K1 = K1 ^ K2 ^ K3
        K2 = K1 ^ K2 ^ K3
        K3 = K1 ^ K2 ^ K3
        K1 = K1 ^ K2 ^ K3
            
        cuda.syncthreads()
        
        
@cuda.jit
def sigkernel_gram_sym_cuda2(M_inc, W, Out, len_x, len_y, x_off, y_off, M_grid, N_grid, n_anti_diagonals, d_order, L):
    block_x = cuda.blockIdx.x
    block_y = cuda.blockIdx.y
    thread_id = cuda.threadIdx.x
    
    if block_y + y_off > block_x + x_off:
        return
    
    c_off = cache_offset(N_grid, M_grid, block_y)
    cx = M_grid - min(N_grid - (block_y + y_off), M_grid)
    W = W[c_off + block_x - cx,:,:]
    
    K1 = 0
    K2 = 2
    K3 = 1

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(2, n_anti_diagonals):
        for l in range(L):
            i = thread_id * L + l + 1
            j = p - i
            
            if i < min(len_x, p) and j < len_y:
                inc = M_inc[block_x, block_y, (i - 1) >> d_order, (j - 1) >> d_order]
                
                k_01 = 1.0 if i == 1 else W[i - 2, K2]
                k_10 = 1.0 if j == 1 else W[i - 1, K2]
                k_00 = 1.0 if j == 1 or i == 1 else W[i - 2, K3]
                               
                W[i - 1, K1] = (k_01 + k_10)*(1. + 0.5*inc + (1./12)*inc**2) - k_00*(1. - (1./12)*inc**2)
                
                if p == n_anti_diagonals - 1:
                    Out[block_x + x_off, block_y + y_off] = W[i - 1, K1]
                    if block_x + x_off != block_y + y_off:
                        Out[block_y + y_off, block_x + x_off] = W[i - 1, K1]

        K1 = K1 ^ K2 ^ K3
        K2 = K1 ^ K2 ^ K3
        K3 = K1 ^ K2 ^ K3
        K1 = K1 ^ K2 ^ K3
        
        cuda.syncthreads()