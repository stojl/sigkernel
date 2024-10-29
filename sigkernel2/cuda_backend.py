import torch.cuda
from numba import cuda
import math

@cuda.jit
def sigkernel_cuda(M_inc, len_x, len_y, n_anti_diagonals, M_sol, d_order, L):
    """
    We start from a list of pairs of paths [(x^1,y^1), ..., (x^n, y^n)]
    M_inc: a 3-tensor D[i,j,k] = <x^i_j, y^i_k>.
    n_anti_diagonals = 2 * max(len_x, len_y) - 1
    M_sol: a 3-tensor storing the solutions of the PDEs.
    """

    # Each block corresponds to a pair (x_i,y_i).
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(2, n_anti_diagonals):
        for l in range(L):
            i = thread_id * L + l + 1
            j = p - i
            
            if i < min(len_x, p) and j < len_y:
                inc = M_inc[block_id, (i - 1) >> d_order, (j - 1) >> d_order]
            
                k_01 = M_sol[block_id, i - 1, j]
                k_10 = M_sol[block_id, i, j - 1]
                k_00 = M_sol[block_id, i - 1, j - 1]        
                
                M_sol[block_id, i, j] = (k_01 + k_10)*(1. + 0.5*inc + (1./12)*inc**2) - k_00*(1. - (1./12)*inc**2)     
            
        cuda.syncthreads()
        
@cuda.jit
def sigkernel_cuda_strided(M_inc, len_x, len_y, n_anti_diagonals, M_sol, d_order, L):
    """
    We start from a list of pairs of paths [(x^1,y^1), ..., (x^n, y^n)]
    M_inc: a 3-tensor D[i,j,k] = <x^i_j, y^i_k>.
    n_anti_diagonals = 2 * max(len_x, len_y) - 1
    M_sol: a 3-tensor storing the solutions of the PDEs.
    """

    # Each block corresponds to a pair (x_i,y_i).
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(2, n_anti_diagonals):
        for l in range(L):
            i = cuda.blockDim.x * l + thread_id + 1
            j = p - i
            
            if i < min(len_x, p) and j < len_y:
                inc = M_inc[block_id, (i - 1) >> d_order, (j - 1) >> d_order]
            
                k_01 = M_sol[block_id, i - 1, j]
                k_10 = M_sol[block_id, i, j - 1]
                k_00 = M_sol[block_id, i - 1, j - 1]        
                
                M_sol[block_id, i, j] = (k_01 + k_10)*(1. + 0.5*inc + (1./12)*inc**2) - k_00*(1. - (1./12)*inc**2)     
            
        cuda.syncthreads()
        
@cuda.jit
def sigkernel_cuda_alt(M_inc, len_x, len_y, n_anti_diagonals, M_sol, d_order, L):
    """
    We start from a list of pairs of paths [(x^1,y^1), ..., (x^n, y^n)]
    M_inc: a 3-tensor D[i,j,k] = <x^i_j, y^i_k>.
    n_anti_diagonals = 2 * max(len_x, len_y) - 1
    M_sol: a 3-tensor storing the solutions of the PDEs.
    """

    # Each block corresponds to a pair (x_i,y_i).
    block_id = cuda.blockIdx.x
    
    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(2, n_anti_diagonals):
        for l in range(L):
            k = cuda.threadIdx.x * L + l
            i = min(p - k - 1, len_x - 1 - k)
            j = p - i
            
            if k + 1 < p and p + k < n_anti_diagonals and k + 1 < len_x:
                inc = M_inc[block_id, (i - 1) >> d_order, (j - 1) >> d_order]
            
                k_01 = M_sol[block_id, i - 1, j]
                k_10 = M_sol[block_id, i, j - 1]
                k_00 = M_sol[block_id, i - 1, j - 1]        
                
                M_sol[block_id, i, j] = (k_01 + k_10)*(1. + 0.5*inc + (1./12)*inc**2) - k_00*(1. - (1./12)*inc**2)     
            
        cuda.syncthreads()
        
@cuda.jit
def sigkernel_cuda_alt_strided(M_inc, len_x, len_y, n_anti_diagonals, M_sol, d_order, L):
    """
    We start from a list of pairs of paths [(x^1,y^1), ..., (x^n, y^n)]
    M_inc: a 3-tensor D[i,j,k] = <x^i_j, y^i_k>.
    n_anti_diagonals = 2 * max(len_x, len_y) - 1
    M_sol: a 3-tensor storing the solutions of the PDEs.
    """

    # Each block corresponds to a pair (x_i,y_i).
    block_id = cuda.blockIdx.x
    
    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(2, n_anti_diagonals):
        for l in range(L):
            k = cuda.blockDim.x * l + cuda.threadIdx.x
            i = min(p - k - 1, len_x - 1 - k)
            j = p - i
            
            if k + 1 < p and p + k < n_anti_diagonals and k + 1 < len_x:
                inc = M_inc[block_id, (i - 1) >> d_order, (j - 1) >> d_order]
            
                k_01 = M_sol[block_id, i - 1, j]
                k_10 = M_sol[block_id, i, j - 1]
                k_00 = M_sol[block_id, i - 1, j - 1]        
                
                M_sol[block_id, i, j] = (k_01 + k_10)*(1. + 0.5*inc + (1./12)*inc**2) - k_00*(1. - (1./12)*inc**2)     
            
        cuda.syncthreads()

@cuda.jit
def sigkernel_Gram_cuda(M_inc, len_x, len_y, n_anti_diagonals, M_sol, d_order):

    block_id_x = cuda.blockIdx.x
    block_id_y = cuda.blockIdx.y

    # Each thread works on a node of a diagonal.
    thread_id = cuda.threadIdx.x

    I = thread_id

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_anti_diagonals):

        # The index is actually 'p - thread_id' but need to force it in-bounds
        J = max(0, min(p - thread_id, len_y - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal
        if I + J == p and (I < len_x and J < len_y):

            inc = M_inc[block_id_x, block_id_y, (i-1) >> d_order, (j-1) >> d_order]

            k_01 = M_sol[block_id_x, block_id_y, i-1, j]
            k_10 = M_sol[block_id_x, block_id_y, i, j-1]
            k_00 = M_sol[block_id_x, block_id_y, i-1, j-1]

            M_sol[block_id_x, block_id_y, i, j] = (k_01 + k_10)*(1. + 0.5*inc + (1./12)*inc**2) - k_00*(1. - (1./12)*inc**2)

        # Wait for other threads in this block
        cuda.syncthreads()

@cuda.jit
def sigkernel_derivatives_Gram_cuda(M_inc, M_inc_diff, M_inc_diffdiff, len_x, len_y, n_anti_diagonals, M_sol, M_sol_diff, M_sol_diffdiff):

    block_id_x = cuda.blockIdx.x
    block_id_y = cuda.blockIdx.y

    # Each thread works on a node of a diagonal.
    thread_id = cuda.threadIdx.x

    I = thread_id

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_anti_diagonals):

        # The index is actually 'p - thread_id' but need to force it in-bounds
        J = max(0, min(p - thread_id, len_y - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal
        if I + J == p and (I < len_x and J < len_y):

            inc = M_inc[block_id_x, block_id_y, i-1, j-1]
            inc_diff = M_inc_diff[block_id_x, block_id_y, i-1, j-1]
            inc_diffdiff = M_inc_diffdiff[block_id_x, block_id_y, i-1, j-1]

            k_01 = M_sol[block_id_x, block_id_y, i-1, j]
            k_10 = M_sol[block_id_x, block_id_y, i, j-1]
            k_00 = M_sol[block_id_x, block_id_y, i-1, j-1]

            k_01_diff = M_sol_diff[block_id_x, block_id_y, i-1, j]
            k_10_diff = M_sol_diff[block_id_x, block_id_y, i, j-1]
            k_00_diff = M_sol_diff[block_id_x, block_id_y, i-1, j-1]

            k_01_diffdiff = M_sol_diffdiff[block_id_x, block_id_y, i-1, j]
            k_10_diffdiff = M_sol_diffdiff[block_id_x, block_id_y, i, j-1]
            k_00_diffdiff = M_sol_diffdiff[block_id_x, block_id_y, i-1, j-1]

            # M_sol[block_id_x, block_id_y, i, j] = (k_01 + k_10) * (1. + .5*inc) - k_00
            M_sol[block_id_x, block_id_y, i, j] = (k_01 + k_10)*(1. + 0.5*inc + (1./12)*inc**2) - k_00*(1. - (1./12)*inc**2)
            
            # M_sol_diff[block_id_x, block_id_y, i, j] = (k_01_diff + k_10_diff) * (1. + .5*inc) - k_00_diff + .5*inc_diff*(k_01 + k_10)
            f1 = k_00*inc_diff + k_00_diff*inc
            f2 = k_01*inc_diff + k_01_diff*inc
            f3 = k_10*inc_diff + k_10_diff*inc
            f4 = M_sol[block_id_x, block_id_y, i, j]*inc_diff + (k_01_diff + k_10_diff - k_00_diff + f1)*inc
            M_sol_diff[block_id_x, block_id_y, i, j] = k_01_diff + k_10_diff - k_00_diff + 0.25*(f1 + f2 + f3 + f4)
            
            # M_sol_diffdiff[block_id_x, block_id_y, i, j] = (k_01_diffdiff + k_10_diffdiff) * (1. + .5*inc) - k_00_diffdiff + .5*inc_diff*(k_01_diff + k_10_diff + k_01 + k_10)
            g1 = k_00*inc_diffdiff + 2.*k_00_diff*inc_diff + k_00_diffdiff*inc
            g2 = k_01*inc_diffdiff + 2.*k_01_diff*inc_diff + k_01_diffdiff*inc
            g3 = k_10*inc_diffdiff + 2.*k_10_diff*inc_diff + k_10_diffdiff*inc
            g4 = M_sol[block_id_x, block_id_y, i, j]*inc_diffdiff + 2.*M_sol_diff[block_id_x, block_id_y, i, j]*inc_diff + (k_01_diffdiff + k_10_diffdiff - k_00_diffdiff + g1)*inc
            M_sol_diffdiff[block_id_x, block_id_y, i, j] = k_01_diffdiff + k_10_diffdiff - k_00_diffdiff + 0.25*(g1 + g2 + g3 + g4)

        # Wait for other threads in this block
        cuda.syncthreads()
# ===========================================================================================================
