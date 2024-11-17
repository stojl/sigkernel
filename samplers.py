import torch
import math
from scipy.interpolate import splrep, splev

class LinearSDE():
    def __init__():
        return
    
    @staticmethod
    def sample(n, rho=1, sigma=0.3, dt=0.005, start=0, stop=1, thin=0, device=torch.device('cuda:0'), dtype=torch.float64):
        A = torch.tensor([[1, rho], [0, 1]], device=device, dtype=dtype)
        l = math.ceil((stop - start) / dt)
        x = torch.zeros((n, l + 1, 2), device=device, dtype=dtype)
        x[:, 0, :] = 0.0
        
        W = torch.randn((n, l, 2), device=device, dtype=dtype)
        W = W * math.sqrt(dt)
        
        for i in range(l):
            x[:, i + 1, :] = x[:, i, :] + x[:, i, :] @ A * dt + sigma * W[:, i, :]
        
        return x[:,range(0, l + 1, 2**thin),0].view(n, -1, 1), x[:,range(0, l + 1, 2**thin),1].view(n, -1, 1)
    
    @staticmethod
    def sample_sig(n, rho=1, sigma=0.3, dt=0.005, start=0, stop=1, thin=0, device=torch.device('cuda:0'), dtype=torch.float64):
        A = sigma * torch.tensor([[1, rho], [0, 1]], device=device, dtype=dtype)
        l = math.ceil((stop - start) / dt)
        x = torch.zeros((n, l + 1, 2), device=device, dtype=dtype)
        x[:, 0, :] = 0.0
        
        W = torch.randn((n, l, 2), device=device, dtype=dtype)
        W = W * math.sqrt(dt)

        for i in range(l):
            x[:, i + 1, :] = x[:, i, :] + x[:, i, :] * dt + W[:,i,:] @ A
        
        return x[:,range(0, l + 1, 2**thin),0].view(n, -1, 1), x[:,range(0, l + 1, 2**thin),1].view(n, -1, 1)
    
    @staticmethod 
    def add_time(x, start=0, stop=1):
        device = x.device
        dtype = x.dtype
        
        l = x.shape[1]

        t = torch.linspace(start, stop, l, device=device, dtype=dtype)
        t = t.unsqueeze(0).unsqueeze(-1)
        return torch.cat((x, t.expand(x.shape[0], x.shape[1], 1)), dim=-1)
    
    
class Fourier():
    def __init__(self):
        return
    
    @staticmethod
    def sample(n, rho=1, scale=8, T=0.5, start=0, stop=1, dt=0.005, randgrid = 30, thin=0, device=torch.device('cuda:0'), dtype=torch.float64):
        l = math.ceil((stop - start) / dt) + 1
        
        ts = torch.linspace(0, 1, l, device=device, dtype=dtype)
        
        b1 = math.sqrt(2 / T) * torch.sin(4 * torch.pi * ts / T)
        b2 = math.sqrt(2 / T) * torch.cos(6 * torch.pi * ts / T)
        c = torch.randn((n, 3), dtype=dtype, device=device)
       
        b1 = (b1 * c[:, 1].view(n, 1))
        b2 = (b2 * c[:, 2].view(n, 1))
        
        x = (b1 + b2 + c[:, 0].view(n, 1))
       
        c1 = (0.75 - 0.25) * torch.rand((n, 2), device=device, dtype=dtype) + 0.25
        
        i1 = (ts - c1[:,0].view(n, 1, 1)).view(n, -1)**2
        i2 = (ts - c1[:,1].view(n, 1, 1)).view(n, -1)**2
        
        zero_column = torch.zeros(n, 1, device=device, dtype=dtype)
        x1 = torch.cat((zero_column, (x[:,range(l-1)] * dt).cumsum(dim=1)), dim=1) * i2
        x2 = torch.cat((zero_column, (x[:,range(l-1)] * i1[:,range(l-1)] * dt).cumsum(dim=1)), dim=1)
        
        y = rho * (x2 - x1) * scale
        y = y + torch.randn(y.shape, device=device, dtype=dtype)
        x = x + torch.randn(x.shape, device=device, dtype=dtype)
        
        return x.view(n, l, 1)[:, range(0, l, 2**thin),:], y.view(n, l, 1)[:, range(0, l, 2**thin),:]
        
class sampler():
    def __init__(self, xi_dist, gam_dist, func, p=50, l=201):
        self.xi_dist = xi_dist
        self.gam_dist = gam_dist
        
        test_sample = xi_dist.sample()
        self.dtype = test_sample.dtype
        self.device = test_sample.device
        
        self.func = func
        self.p = p
        self.l = l
        
        self.ts = torch.linspace(0, 1, l, device=self.device, dtype=self.dtype)
        self.coef = torch.arange(1, p + 1, 1, dtype=self.dtype, device=self.device).view(-1, 1)
        
        self.basis = math.sqrt(2) * torch.cos(torch.pi * self.coef * self.ts)
        
    def sample(self, n=30, m=0, garbage=0):
        m = self.p if m >= self.p else m
        
        xis = self.xi_dist.sample((n, self.p))
        gams = self.gam_dist.sample((n, self.p - m))
        
        if m == 0:
            gams = gams
        else:
            gams = torch.concat((self.func(xis[:,0:m]), gams), dim=1)
        
        X = (xis.view(n, self.p, 1) * self.basis).sum(dim=1).view(n, -1, 1)
        Y = (gams.view(n, self.p, 1) * self.basis).sum(dim=1).view(n, -1, 1)
                   
        return X, Y