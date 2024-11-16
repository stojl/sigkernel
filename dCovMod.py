import torch

class dCovMod():
    def __init__(self, X, Y):
        self.n = X.shape[0]
        self.device = X.device
        self.dtype = X.dtype
        
        X = X.view(self.n, -1)
        Y = Y.view(self.n, -1)
        
        self.A = self.center_modify(self.distance_matrix(X))
        self.B = self.center_modify(self.distance_matrix(Y))

    def test(self, m=1000, max_batch=None):        
        max_batch = m if max_batch is None else max_batch
        
        mb_size = min(m, max_batch)
        bm = -(-m // mb_size)
        
        stats = torch.zeros([m], device=self.device, dtype=self.dtype)
        
        mask = torch.eye(self.n, device=self.device, dtype=torch.bool)
        mask_inv = ~mask
        
        value = (self.A[mask_inv] * self.B[mask_inv]).sum() - 2 * (self.A[mask] * self.B[mask]).sum() / (self.n - 2)
        value /= self.n * (self.n - 3)
        
        for i in range(bm):
            mb_size_i = m - mb_size * (bm - 1) if i == bm - 1 else mb_size
            start = i * mb_size
            stop = i * mb_size + mb_size_i
            perms = torch.stack([torch.randperm(self.n, device=self.device) for _ in range(mb_size_i)])
            B_perm = torch.stack([self.B[p][:, p] for p in perms])
            
            tmp = (self.A[mask_inv] * B_perm[:,mask_inv]).sum(dim=1) - 2 * (self.A[mask] * B_perm[:,mask]).sum(dim=1) / (self.n - 2)
            tmp /= self.n * (self.n - 3)
            stats[start:stop] = tmp
            
        return (stats.abs() > value).double().mean().cpu().item()
        
    def distance_matrix(self, x):
        x_norm = (x ** 2).sum(dim=1, keepdim=True)  # (batch_size, 1)
        distances_squared = x_norm + x_norm.T - 2 * torch.mm(x, x.T)
        distances_squared = torch.clamp(distances_squared, min=0.0)
        return torch.sqrt(distances_squared)
    
    def center_modify(self, x):
        a_col_mean = x.mean(dim=0, keepdim=True)
        a_row_mean = x.mean(dim=1, keepdim=True)
        a_mean = x.mean()
        A = x - a_col_mean - a_row_mean - a_mean
        
        mask = torch.eye(self.n, device=self.device, dtype=torch.bool)
        A[mask] = a_row_mean.view(-1) - a_mean
        mask = ~mask
        A[mask] = A[mask] - x[mask] / self.n
        A = self.n * A / (self.n - 1)
        return A
        
        
        
        