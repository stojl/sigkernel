import torch
import math

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
    def add_time(x, start=0, stop=1):
        device = x.device
        dtype = x.dtype
        
        l = x.shape[1]

        t = torch.linspace(start, stop, l, device=device, dtype=dtype)
        t = t.unsqueeze(0).unsqueeze(-1)
        return torch.cat((x, t.expand(x.shape[0], x.shape[1], 1)), dim=-1)