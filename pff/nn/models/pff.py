import torch
import torch.nn as nn
import torch.nn.functional as F
from pff.nn.layers import PFFLayer
from pff.utils.functions import ReLU_full_grad


class PFF(nn.Module):
    def __init__(self, sizes, y_size, y_scale, bias_r=False, bias_g=False):
        super().__init__()
        self.sizes = sizes
        self.y_scale = y_scale

        self.layers = nn.ModuleList()
        for i in range(1, len(sizes)-1):
            self.layers.append(PFFLayer(sizes[i-1], sizes[i], sizes[i+1]), bias_r=bias_r, bias_g=bias_g)
        self.layers.append(PFFLayer(sizes[-2], sizes[-1], y_size))

        self.G = nn.Linear(sizes[1], sizes[0], bias=bias_g)
        self.relu = ReLU_full_grad()
        self.sigma_g = 0.025
        self.beta = 0.025 # lr for z_g
    
    def step_rep(self, x, y, z):
        y = y * self.y_scale
        new_z = []
        for i, layer in enumerate(self.layers):
            z_below = z[i-1] if i > 0 else x
            z_above = z[i+1] if i < len(self.layers) - 1 else y
            is_top = i == len(self.layers) - 1
            new_z.append(layer.step_rep(z_below, z[i], z_above, is_top))
        return new_z
    
    def step_gen(self, x, y, z, z_g):
        errors = []
        z_g_bar = self.relu(z_g)
        
        for i, layer in reversed(enumerate(self.layers)):
            z_above = z[i+1] if i < len(self.layers) - 1 else z_g_bar
            is_top = i == len(self.layers) - 1
            is_bottom = i == 0
            z_pred = layer.step_gen(z_above, is_top, is_bottom)
            errors.append(z_pred - z[i])
        
        with torch.no_grad():
            inj_noise = torch.normal(0, self.sigma_g, size=x.shape)
            z_above = torch.normalize(self.relu(z[0] + inj_noise), dim=1)
        z_pred = torch.clip(self.G(z_above), 0.0, 1.0)
        errors.append((z_pred - x).square().sum(dim=1))

        errors = list(reversed(errors))

        errors[-1].backward()
        z_g = z_g - self.beta * z_g.grad

        return errors, z_g
        




    def forward(self, x):
        z = []
        for i, layer in enumerate(self.layers):
            is_top = i == len(self.layers) - 1
            x = layer(x, is_top)
            z.append(x)
        return z
