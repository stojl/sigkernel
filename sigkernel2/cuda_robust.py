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
def sigkernel_norms2(M_inc, C, norms, len_x, n_anti_diagonals, M_sol, M_sol2, d_order, L):
    """
    We start from a list of pairs of paths [(x^1,y^1), ..., (x^n, y^n)]
    M_inc: a 3-tensor D[i,j,k] = <x^i_j, y^i_k>.
    n_anti_diagonals = 2 * max(len_x, len_y) - 1
    M_sol: a 3-tensor storing the solutions of the PDEs.
    """

    # Each block corresponds to a pair (x_i,y_i).
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    
    a = 0.0
    b = 1.0
    c = 0.5
    
    K1 = 0
    K2 = 2
    K3 = 1

    # Bisection method
    for _ in range(9):
        c = 0.5 * (a + b)
        
        for p in range(2, n_anti_diagonals):
            for l in range(L):
                i = thread_id * L + l + 1
                j = p - i
                
                if i < min(len_x, p) and j < len_x:
                    inc = M_inc[block_id, (i - 1) >> d_order, (j - 1) >> d_order]
                    
                    k_01 = 1.0 if i == 1 else M_sol[block_id, i - 2, K2]
                    k_10 = 1.0 if j == 1 else M_sol[block_id, i - 1, K2]
                    k_00 = 1.0 if j == 1 or i == 1 else M_sol[block_id, i - 2, K3]
                                
                    M_sol[block_id, i - 1, K1] = (k_01 + k_10)*(1. + 0.5 * inc * c**2 + (1./12) * inc**2 * c**4) - k_00 * (1. - (1./12) * inc**2 * c**4)
                    
                    if p == n_anti_diagonals - 1:
                        M_sol[block_id, 0, 0] = M_sol[block_id, i - 1, K1]

            K1 = K1 ^ K2 ^ K3
            K2 = K1 ^ K2 ^ K3
            K3 = K1 ^ K2 ^ K3
            K1 = K1 ^ K2 ^ K3
                
            cuda.syncthreads()
            
        if M_sol[block_id, 0, 0] < norms[block_id]:
            a = c
        else:
            b = c
            
    if thread_id == 0:
        C[block_id] = c    
    
    # Newton-Raphson iteration
    """for _ in range(5):
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
                                
                    M_sol[block_id, i - 1, K1] = (k_01 + k_10)*(1. + 0.5 * inc * c**2 + (1./12) * inc**2 * c**4) - k_00 * (1. - (1./12) * inc**2 * c**4)
                    M_sol2[block_id, i - 1, K1] = (k_01_2 + k_10_2) * (1. + inc * c**2 + (1./12) * inc**2 * c**4) - k_00_2 * (1. - (1./12) * inc**2 * c**4)
                    M_sol2[block_id, i - 1, K1] += 0.5 * inc * c * (k_01 + k_10 + k_00 + M_sol[block_id, i - 1, K1])
                    
                    if p == n_anti_diagonals - 1:
                        M_sol[block_id, 0, 0] = M_sol[block_id, i - 1, K1]
                        M_sol2[block_id, 0, 0] = M_sol2[block_id, i - 1, K1]

            K1 = K1 ^ K2 ^ K3
            K2 = K1 ^ K2 ^ K3
            K3 = K1 ^ K2 ^ K3
            K1 = K1 ^ K2 ^ K3
                
            cuda.syncthreads()
                
        if thread_id == 0:
            C[block_id] = c - (M_sol[block_id, 0, 0] - norms[block_id]) / M_sol2[block_id, 0, 0]
            c = C[block_id]
            
        cuda.syncthreads()"""

@cuda.jit
def sigkernel_norms(M_inc, C, norms, len_x, n_anti_diagonals, M_sol, M_sol2, d_order, L, N, rel_tol, abs_tol):
    """
    We start from a list of pairs of paths [(x^1,y^1), ..., (x^n, y^n)]
    M_inc: a 3-tensor D[i,j,k] = <x^i_j, y^i_k>.
    n_anti_diagonals = 2 * max(len_x, len_y) - 1
    M_sol: a 3-tensor storing the solutions of the PDEs.
    """

    # Each block corresponds to a pair (x_i,y_i).
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    
    old_C = C[block_id]
    
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
                    #M_sol2[block_id, i - 1, K1] = k_01_2 + k_10_2 - k_00_2 + inc * (C[block_id]**2 * (k_01_2 + k_10_2) + C[block_id] * (k_01 + k_10))
                    M_sol2[block_id, i - 1, K1] = (k_01_2 + k_10_2) * (1. + inc * C[block_id]**2 + (1./12) * inc**2 * C[block_id]**4) - k_00_2 * (1. - (1./12) * inc**2 * C[block_id]**4)
                    M_sol2[block_id, i - 1, K1] += 0.5 * inc * C[block_id] * (k_01 + k_10 + k_00 + M_sol[block_id, i - 1, K1])
                    
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
        
        if abs(1 - C[block_id] / old_C) < rel_tol or abs(old_C - C[block_id]) < abs_tol:
            break
        
        old_C = C[block_id]
            
        
        
        
@cuda.jit
def robust_sigkernel(M_inc, x_norm, y_norm, len_x, n_anti_diagonals, M_sol, d_order, L):
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
            
            if i < min(len_x, p) and j < len_x:
                inc = M_inc[block_id, (i - 1) >> d_order, (j - 1) >> d_order]
                
                k_01 = 1.0 if i == 1 else M_sol[block_id, i - 2, K2]
                k_10 = 1.0 if j == 1 else M_sol[block_id, i - 1, K2]
                k_00 = 1.0 if j == 1 or i == 1 else M_sol[block_id, i - 2, K3]
                            
                M_sol[block_id, i - 1, K1] = (k_01 + k_10) * (1. + 0.5 * inc * x_norm[block_id] * y_norm[block_id] + (1./12) * inc**2 * x_norm[block_id]**2 * y_norm[block_id]**2) - k_00 * (1. - (1./12) * inc**2 * x_norm[block_id]**2 * y_norm[block_id]**2)
                
                if p == n_anti_diagonals - 1:
                    M_sol[block_id, 0, 0] = M_sol[block_id, i - 1, K1]

        K1 = K1 ^ K2 ^ K3
        K2 = K1 ^ K2 ^ K3
        K3 = K1 ^ K2 ^ K3
        K1 = K1 ^ K2 ^ K3
            
        cuda.syncthreads()
        
        
@cuda.jit
def robust_sigkernel_gram(M_inc, x_norm, y_norm, len_x, len_y, n_anti_diagonals, M_sol, d_order, L):
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
                               
                M_sol[block_x, block_y, i - 1, K1] = (k_01 + k_10) * (1. + 0.5 * inc * x_norm[block_x] * y_norm[block_y] + (1./12) * inc**2 * x_norm[block_x]**2 * y_norm[block_y]**2) - k_00 * (1. - (1./12) * inc**2 * x_norm[block_x]**2 * y_norm[block_y]**2)
                
                if p == n_anti_diagonals - 1:
                    M_sol[block_x, block_y, 0, 0] = M_sol[block_x, block_y, i - 1, K1]

        K1 = K1 ^ K2 ^ K3
        K2 = K1 ^ K2 ^ K3
        K3 = K1 ^ K2 ^ K3
        K1 = K1 ^ K2 ^ K3
            
        cuda.syncthreads()
        
        
@cuda.jit
def robust_sigkernel_gram_sym(M_inc, W, Out, norms, len_x, len_y, x_off, y_off, M_grid, N_grid, n_anti_diagonals, d_order, L):
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
                               
                W[i - 1, K1] = (k_01 + k_10) * (1. + 0.5 * inc * norms[block_x + x_off] * norms[block_y + y_off] + (1./12) * inc**2 * norms[block_x + x_off]**2 * norms[block_y + y_off]**2) - k_00 * (1. - (1./12) * inc**2 * norms[block_x + x_off]**2 * norms[block_y + y_off]**2)
                
                if p == n_anti_diagonals - 1:
                    Out[block_x + x_off, block_y + y_off] = W[i - 1, K1]
                    if block_x + x_off != block_y + y_off:
                        Out[block_y + y_off, block_x + x_off] = W[i - 1, K1]

        K1 = K1 ^ K2 ^ K3
        K2 = K1 ^ K2 ^ K3
        K3 = K1 ^ K2 ^ K3
        K1 = K1 ^ K2 ^ K3
        
        cuda.syncthreads()