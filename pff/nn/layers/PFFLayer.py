import torch
import torch.nn as nn
import torch.nn.functional as F
from pff.utils.functions import my_relu


class PFFLayer(nn.Module):
    def __init__(self, size_below, size, size_above, beta=0.3, threshold=3.0, bias_r=True, bias_g=True, g_units=None):
        super(PFFLayer, self).__init__()
        self.size = size
        self.beta = beta
        self.r_sigma = 0.01
        self.g_sigma = 0.025
        self.threshold = threshold
        self.device = torch.device('cpu')

        self.W = nn.Linear(size_below, size, bias=bias_r)
        self.V = nn.Linear(size_above, size, bias=False)
        self.L = nn.Linear(size, size, bias=False)
        self.L.weight.data = self.L.weight.data.abs()
        self.M = torch.empty(size, size, requires_grad=False)
        self._init_M()

        if g_units is None:
            g_units = size_above
        self.G = nn.Linear(g_units, size, bias=bias_g)

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
    
    def step_rep(self, z_below, z, z_above):
        L =  my_relu(self.L.weight) * self.M * (1 - torch.eye(self.size).to(self.device)) - my_relu(self.L.weight) * (torch.eye(self.size).to(self.device))

        bottom_up = self.W(F.normalize(z_below, dim=1))
        top_down = self.V(F.normalize(z_above, dim=1)) 
        lateral = F.normalize(z, dim=1) @ L
        inj_noise = torch.normal(0, self.r_sigma, size=z.shape).to(self.device)
        proposed_z = my_relu(bottom_up + top_down - lateral + inj_noise)

        return self.beta * proposed_z + (1 - self.beta) * z
    
    def step_gen(self, z_above, is_top=False):
        if is_top:
            z_above = F.normalize(z_above, dim=1)
        else:
            with torch.no_grad():
                inj_noise = torch.normal(0, self.g_sigma, size=z_above.shape).to(self.device)
                z_above = F.normalize(my_relu(z_above + inj_noise), dim=1)
        return my_relu(self.G(z_above))

    def to(self, *args, **kwargs):
        self.device = args[0]
        self.M = self.M.to(self.device)
        return super().to(*args, **kwargs)
    
    def forward(self, x, is_top=False):
        if not is_top:
            return my_relu(self.W(F.normalize(x, dim=1)))
        else:
            return torch.softmax(self.W(F.normalize(x, dim=1)), dim=1)