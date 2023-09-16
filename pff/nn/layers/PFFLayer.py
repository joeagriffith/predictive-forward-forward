import torch
import torch.nn as nn

class PFFLayer(nn.Module):
    def __init__(self, size_below, size, size_above=None, beta=0.1, bias=False):
        self.size = size

        if size_above is None:
            size_above = size
        self.W = nn.Linear(size_below, size, bias=bias)
        self.V = nn.Linear(size_above, size, bias=bias)
        self.L = nn.Linear(size, size, bias=bias)
        self.M = nn.Empty(size, size, bias=bias)
        self._init_M()

        self.r_sigma = 0.025
        self.g_sigma = 0.025


    def _init_M(self, K=10):
        if self.size % K != 0:
            raise ValueError("size must be divisible by k")
        S = [torch.zeros(self.size, K) for _ in range(self.size // K)]
        for i, s in enumerate(S):
            rows = [n+K*(i-1) for n in range(K)]
            for col in range(K):
                for row in rows:
                    s[row, col] = 1.0
        self.M = torch.cat(S, dim=1)
        assert(self.M.shape == (self.size, self.size))

    
    def forward(self, z_below, z, z_above=None):
        if z_above is None:
            z_above = z
        L = torch.relu(self.L.weight) * (torch.eye(self.size)) - torch.relu(self.L.weight) * self.M * (1 - torch.eye(self.size)) 

        bottom_up = torch.relu(self.W(torch.normalize(z_below, dim=1))) 
        top_down = self.V(torch.normalize(z_above, dim=1)) 
        lateral = L @ torch.normalize(z, dim=1) 
        inj_noise = torch.normal(0, self.r_sigma, size=z.shape)
        proposed_z = bottom_up + top_down + lateral + inj_noise

        return self.beta * proposed_z + (1 - self.beta) * z
        
