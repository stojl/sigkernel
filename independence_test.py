import torch
import math
import scipy.stats.distributions as dist

def derrangement(n, device):
    indicies = torch.arange(n, device=device)
    
    while True:
        perm = torch.randperm(n, device=device)
        if torch.all(perm != indicies):
            return perm

class dCov():
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
        
        value = (self.A * self.B).mean()
        
        for i in range(bm):
            mb_size_i = m - mb_size * (bm - 1) if i == bm - 1 else mb_size
            start = i * mb_size
            stop = i * mb_size + mb_size_i
            perms = torch.stack([derrangement(self.n, device=self.device) for _ in range(mb_size_i)])
            B_perm = torch.stack([self.B[p][:, p] for p in perms])
            tmp = (self.A * B_perm).mean(dim=(1,2))
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
        return A
    
class HSIC:
    def __init__(self, X, Y):
        self.n = X.shape[0]
        H = -torch.ones((self.n, self.n), device=X.device, dtype=X.dtype) / self.n
        H[range(self.n), range(self.n)] = 1 - 1 / self.n
        
        self.X = X @ H
        self.Y = Y @ H
        
    def test(self, alpha=0.05, perms=1000, max_batch=None):
        critical_value = torch.einsum('ij,ji->', self.X, self.Y) / self.n
        return self._perm(critical_value, alpha, perms, max_batch).cpu().item()
        #return {
            #"critical_value": critical_value.cpu().item(),
            #"gamma_p_value": self._gamma_approx(critical_value.cpu().item(), alpha),
            #"perm_p_value": self._perm(critical_value, alpha, perms, max_batch).cpu().item()#,
            #"mc_p_value": self._montecarlo(critical_value, alpha, perms, max_batch).cpu().item()
        #}
    
    def _gamma_approx(self, value=None, alpha=0.05):
        mu, var = self._empirical_moments()
        loc = mu**2 / var
        scale = var / mu
        if value is None:
            return dist.gamma.ppf(1 - alpha, a = loc.cpu(), scale = scale.cpu())
        return dist.gamma.sf(value, a = loc.cpu(), scale = scale.cpu())
    
    def _empirical_moments(self):
        mu_X = torch.trace(self.X)
        mu_Y = torch.trace(self.Y)
        
        var_X = torch.einsum('ij,ji->', self.X, self.X)
        var_Y = torch.einsum('ij,ji->', self.Y, self.Y)
        
        return mu_X * mu_Y / self.n**2, 2 * var_X * var_Y / self.n**4
    
    def _perm(self, value=None, alpha=0.05, m=100, max_batch=None):
        max_batch = m if max_batch is None else max_batch
        
        mb_size = min(m, max_batch)
        bm = -(-m // mb_size)
        
        stats = torch.zeros([m], device=self.X.device, dtype=self.X.dtype)
        
        for i in range(bm):
            mb_size_i = m - mb_size * (bm - 1) if i == bm - 1 else mb_size
            start = i * mb_size
            stop = i * mb_size + mb_size_i
            perms = torch.stack([derrangement(self.n, device=self.X.device) for _ in range(mb_size_i)])
            Y_perm = torch.stack([self.Y[p][:, p] for p in perms])
            tmp = torch.einsum('ij,kji->k', self.X, Y_perm) / self.n
            stats[start:stop] = tmp
            
        if value is None:
            return stats.abs().quantile(1 - alpha, 0)
            
        return (stats.abs() > value).double().mean()
    
    def _montecarlo(self, value=None, alpha=0.05, m=100, max_batch=None):
        max_batch = m if max_batch is None else max_batch
        
        mb_size = min(m, max_batch)
        bm = -(-m // mb_size)
        
        stats = torch.zeros([m], device=self.X.device, dtype=self.X.dtype)
        eig_X = torch.linalg.eigvalsh(self.X).view(self.n, -1)
        eig_Y = torch.linalg.eigvalsh(self.Y).view(-1, self.n)
        
        for i in range(bm):
            mb_size_i = m - mb_size * (bm - 1) if i == bm - 1 else mb_size
            start = i * mb_size
            stop = i * mb_size + mb_size_i
            z = torch.randn([mb_size_i, self.n, self.n], device=self.X.device, dtype=self.X.dtype).pow(2)
            z = z * eig_X * eig_Y           
            stats[start:stop] = z.mean(dim=[1,2])
            
        if value is None:
            return stats.abs().quantile(1 - alpha, 0)
            
        return (stats.abs() > value).double().mean()
    
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
            perms = torch.stack([derrangement(self.n, device=self.device) for _ in range(mb_size_i)])
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