import torch
import torch.cuda
from numba import cuda
from .cuda_backend2 import sigkernel_cuda2, sigkernel_gram_cuda2, sigkernel_gram_sym_cuda2

def ceil_div(a, b):
    return -(-a // b)

def triangular_cache_size(n, m):
    s = 0
    for i in range(m):
        s += min(n - i, m)
        
    return s

def round_to_multiple_of_32(x):
    return ((x + 31) // 32) * 32

class SigKernel2():
    def __init__(self, static_kernel, dyadic_order):
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order
        
    def dist(self, X, Y):
        G_static = self.static_kernel.batch_kernel(X, Y)
        G_static_ = G_static[:,1:,1:] + G_static[:,:-1,:-1] - G_static[:,1:,:-1] - G_static[:,:-1,1:]
        return G_static_ / float(2**(2 * self.dyadic_order))
    
    def dist_gram(self, X, Y):
        G_static = self.static_kernel.Gram_matrix(X, Y)
        G_static_ = G_static[:, :, 1:, 1:] + G_static[:, :, :-1, :-1] - G_static[:, :, 1:, :-1] - G_static[:, :, :-1, 1:]
        return G_static_ / float(2**(2 * self.dyadic_order))
        
    def kernel(self, X, Y, max_batch=100, max_threads=1024):
        batch_size = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        
        bm = ceil_div(batch_size, max_batch)

        MM = (2**self.dyadic_order)*(M-1) + 1
        NN = (2**self.dyadic_order)*(N-1) + 1
        n_anti_diagonals = MM + NN - 1
        
        threads_per_block = min(max_threads, MM - 1, 1024)
        threads_per_block = round_to_multiple_of_32(threads_per_block)
        L = -(-(MM - 1) // threads_per_block)
        
        mb_size = min(batch_size, max_batch)
        W = torch.zeros([mb_size, MM - 1, 3], device=X.device, dtype=X.dtype)
        K = torch.zeros([batch_size], device=X.device, dtype=X.dtype)
        
        for i in range(bm):
            mb_size_i = batch_size - mb_size * (bm - 1) if i == bm - 1 else mb_size
            start = i * mb_size
            stop = i * mb_size + mb_size_i
            
            inc = self.dist(
                X[start:stop,:,:],
                Y[start:stop,:,:]
                )
            
            sigkernel_cuda2[mb_size_i, threads_per_block](
                cuda.as_cuda_array(inc.detach()),
                MM, NN, n_anti_diagonals,
                cuda.as_cuda_array(W), 
                self.dyadic_order, L
                )
            
            cuda.synchronize()
            
            K[start:stop] = W[0:mb_size_i,0,0]
            
        return K
    
    def gram(self, X, Y=None, max_batch=100, max_threads=1024):
        if Y is None:
            return self._gram_sym(X, max_batch, max_threads)
        else:
            return self._gram(X, Y, max_batch, max_threads)
    
    def _gram(self, X, Y, max_batch=10, max_threads=1024):
        batch_x = X.shape[0]
        batch_y = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]

        MM = (2**self.dyadic_order)*(M-1) + 1
        NN = (2**self.dyadic_order)*(N-1) + 1
        n_anti_diagonals = MM + NN - 1
        
        threads_per_block = min(max_threads, MM - 1, 1024)
        threads_per_block = round_to_multiple_of_32(threads_per_block)
        L = -(-(MM - 1) // threads_per_block)
        
        bm_x = ceil_div(batch_x, max_batch)
        bm_y = ceil_div(batch_y, max_batch)
        
        mb_size_x = min(batch_x, max_batch)
        mb_size_y = min(batch_y, max_batch)
        
        W = torch.zeros([mb_size_x, mb_size_y, MM - 1, 3], device=X.device, dtype=X.dtype)
        K = torch.zeros([batch_x, batch_y], device=X.device, dtype=X.dtype)
        
        for j in range(bm_y):
            mb_j = batch_y - mb_size_y * (bm_y - 1) if j == bm_y - 1 else mb_size_y
            start_j = j * mb_size_y
            stop_j = j * mb_size_y + mb_j
            for i in range(bm_x):
                mb_i = batch_x - mb_size_x * (bm_x - 1) if i == bm_x - 1 else mb_size_x
                
                start_i = i * mb_size_x
                stop_i = i * mb_size_x + mb_i
                
                inc = self.dist_gram(
                    X[start_i:stop_i,:,:],
                    Y[start_j:stop_j,:,:]
                )

                sigkernel_gram_cuda2[(mb_i, mb_j), threads_per_block](
                    cuda.as_cuda_array(inc.detach()),
                    MM, NN, n_anti_diagonals,
                    cuda.as_cuda_array(W), 
                    self.dyadic_order, L
                )
                
                cuda.synchronize()
                
                K[start_i:stop_i, start_j:stop_j] = W[0:mb_i, 0:mb_j, 0, 0]
                
        return K
    
    def _gram_sym(self, X, max_batch=10, max_threads=1024):
        batch_size = X.shape[0]
        M = X.shape[1]

        MM = (2**self.dyadic_order)*(M-1) + 1
        n_anti_diagonals = MM + MM - 1
        
        threads_per_block = min(max_threads, MM - 1, 1024)
        threads_per_block = round_to_multiple_of_32(threads_per_block)
        L = -(-(MM - 1) // threads_per_block)
        
        bm = ceil_div(batch_size, max_batch)
        mb_size = min(batch_size, max_batch)
        
        tri_cache_size = triangular_cache_size(batch_size, mb_size)
        
        W = torch.zeros([tri_cache_size, MM - 1, 3], device=X.device, dtype=X.dtype)
        K = torch.zeros([batch_size, batch_size], device=X.device, dtype=X.dtype)
        
        for j in range(bm):
            mb_j = batch_size - mb_size * (bm - 1) if j == bm - 1 else mb_size
            start_j = j * mb_size
            stop_j = j * mb_size + mb_j
            for i in range(bm):
                mb_i = batch_size - mb_size * (bm - 1) if i == bm - 1 else mb_size
                start_i = i * mb_size
                stop_i = i * mb_size + mb_i
                
                inc = self.dist_gram(
                    X[start_i:stop_i,:,:],
                    X[start_j:stop_j,:,:]
                )
                
                sigkernel_gram_sym_cuda2[(mb_i, mb_j), threads_per_block](
                    cuda.as_cuda_array(inc.detach()),
                    cuda.as_cuda_array(W),
                    cuda.as_cuda_array(K),
                    MM, MM, 
                    i * mb_size, j * mb_size,
                    mb_size, batch_size,
                    n_anti_diagonals,
                    self.dyadic_order, 
                    L
                )
                
                cuda.synchronize()
        return K
                
                