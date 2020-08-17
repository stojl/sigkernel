import torch

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

class SigKernel_VarPar(torch.autograd.Function):


    @staticmethod
    def forward(ctx, X, Y, n=0):
        """
        input
         - X a list of A paths each of shape (M,D)
         - Y a list of A paths each of shape (N,D)

        computes by solving PDEs (forward) and by variation of parameter (backward)
         -  K_XY: a vector of A pairwise kernel evaluations k(x_i,y_i)      (forward)
         -  K_dXY: A matrices ( dk_{x_{pq}}(x_i,y_i) )_{p=1,q=1}^{p=M,q=D}  (backward)

        """

        A = len(X)
        D = X[0].shape[1]
        M = X[0].shape[0]
        N = Y[0].shape[0]

        # kernel K
        K = torch.zeros((A, (2**n)*(M-1)+1, (2**n)*(N-1)+1)).type(torch.float64)
        K[:, 0, :] = 1.
        K[:, :, 0] = 1.

        # kernel reversed K_rev
        K_rev = torch.zeros((A, (2**n)*(M-1)+1, (2**n)*(N-1)+1)).type(torch.float64)
        K_rev[:, 0, :] = 1.
        K_rev[:, :, 0] = 1.

        # 1. FORWARD
        for i in range(0, (2**n)*(M-1)):
            for j in range(0,(2**n)*(N-1)):

                ii = int(i / (2 ** n))
                jj = int(j / (2 ** n))

                inc_X_i = (X[:, ii + 1, :] - X[:, ii, :])/float(2**n)  # (A,D)
                inc_Y_j = (Y[:, jj + 1, :] - Y[:, jj, :])/float(2**n)  # (A,D)

                inc_X_rev_i = (X[:, (M-1)-(ii + 1), :] - X[:, (M-1)-ii, :])/float(2**n)  # (A,D)
                inc_Y_rev_j = (Y[:, (N-1)-(jj + 1), :] - Y[:, (N-1)-jj, :])/float(2**n)  # (A,D)

                prod_inc = torch.einsum('ik,ik->i', inc_X_i, inc_Y_j)  # (A) <-> A dots prod bwn R^D and R^D
                K[:, i + 1, j + 1] = K[:, i + 1, j] + K[:, i, j + 1] + K[:, i,j]* prod_inc - K[ :, i,j]

                prod_inc_rev = torch.einsum('ik,ik->i', inc_X_rev_i, inc_Y_rev_j)
                K_rev[:, i + 1, j + 1] = K_rev[:, i + 1, j] + K_rev[:, i, j + 1] + K_rev[:, i, j] * prod_inc_rev - K_rev[:, i, j]

        # 2. GRADIENTS

        # Need to get the increments of Y on the finer grid
        inc_Y = (Y[:,1:,:]-Y[:,:-1,:])/float(2**n)  #(A,M-1,D)  increments defined by the data
        inc_Y = tile(inc_Y,1,2**n)                  #(A,(2**n)*(M-1),D)  increments on the finer grid

        # Need to reorganize the K_rev matrix
        K_rev_rev = flip(K_rev,dim=1)
        K_rev_rev = flip(K_rev_rev, dim=2)


        KK = (K[:,:-1,:-1] * K_rev_rev[:,1:,1:])                       #(A,(2**n)*(M-1),(2**n)*(N-1))

        K_grad = KK[:,:,:,None]*inc_Y[:,None,:,:]                      # (A,(2**n)*(M-1),(2**n)*(N-1),D)

        K_grad = (1./(2**n))*torch.sum(K_grad,axis=2)                  # (A,(2**n)*(M-1),D)

        K_grad =  torch.sum(K_grad.reshape(A,M-1,2**n,D),axis=2)       #(A,M,D)

        ctx.save_for_backward(K_grad)

        return torch.mean(K[:, -1, -1])

    @staticmethod
    def backward(ctx, grad_output):

        """
        During the forward pass, the gradients with respect to each increment in each dimension has been computed.
        Here we derive the gradients with respect to the points of the time series.
        """

        grad_incr, = ctx.saved_tensors

        A = grad_incr.shape[0]
        D = grad_incr.shape[2]
        grad_points = -torch.cat([grad_incr,torch.zeros((A, 1, D)).type(torch.float64)], dim=1) + torch.cat([torch.zeros((A, 1, D)).type(torch.float64), grad_incr], dim=1)

        return grad_points, None, None