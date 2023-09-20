import torch
import torch.nn as nn
import torch.nn.functional as F
from pff.nn.layers import PFFLayer
from pff.utils.functions import my_relu, goodness


class PFF(nn.Module):
    def __init__(self, sizes, y_scale=5.0, g_units=20, bias_r=True, bias_g=True):
        super(PFF, self).__init__()
        self.sizes = sizes
        self.y_scale = y_scale
        self.g_units = g_units
        self.device = torch.device('cpu')

        self.layers = nn.ModuleList()
        for i in range(1, len(sizes)-1):
            g = None if i < len(sizes) - 2 else g_units
            self.layers.append(PFFLayer(sizes[i-1], sizes[i], sizes[i+1], bias_r=bias_r, bias_g=bias_g, g_units=g))

        self.classifier = nn.Linear(sizes[-2], sizes[-1], bias=False)
        self.generator = nn.Linear(sizes[1], sizes[0], bias=bias_g)
        self.sigma_g = 0.025
        self.beta = 0.025 # lr for z_g
    

    def step_rep(self, x, y, z):
        new_z = []
        for i, layer in enumerate(self.layers):
            z_below = z[i-1] if i > 0 else x
            z_above = z[i+1] if i < len(self.layers) - 1 else y * self.y_scale
            new_z.append(layer.step_rep(z_below.detach(), z[i].detach(), z_above.detach()))

        y_hat = torch.softmax(self.classifier(F.normalize(z[-1].detach(), dim=1)), dim=1)
        return y_hat, new_z
    

    def step_gen(self, x, z, z_g):
        errors = []
        z_g_bar = my_relu(z_g)
        
        for i, layer in reversed(list(enumerate(self.layers))):
            z_above = z[i+1].detach() if i < len(self.layers) - 1 else z_g_bar
            is_top = i == len(self.layers) - 1
            z_pred = layer.step_gen(z_above, is_top)
            errors.append((z_pred - z[i].detach()).square().sum(dim=1).mean())
        
        with torch.no_grad():
            inj_noise = torch.normal(0, self.sigma_g, size=z[0].shape).to(self.device)
            z_above = F.normalize(my_relu(z[0].detach() + inj_noise), dim=1)
        x_hat = torch.clip(self.generator(z_above.detach()), 0.0, 1.0)
        errors.append((x_hat - x).square().sum(dim=1).mean())
        
        errors = list(reversed(errors))

        z_g.grad = None
        errors[-1].backward(retain_graph=True)
        with torch.no_grad():
            z_g.data += -self.beta * z_g.grad

        return x_hat, errors, z_g
    
    def to(self, device):
        self.device = device
        self.classifier.to(device)
        self.generator.to(device)
        for layer in self.layers:
            layer.to(device)
        return self

    def forward(self, x):
        z = []
        for i, layer in enumerate(self.layers):
            is_top = i == len(self.layers) - 1
            x = layer(x, is_top)
            z.append(x)
        return z

    def infer(self, x, y, label, num_steps, optimiser=None):
        z = self.forward(x)

        for _ in range(num_steps):
            y_hat, z = self.step_rep(x, y, z)
        
            if optimiser is not None:
                goodness_loss = sum([F.binary_cross_entropy_with_logits(goodness(z_i), label) for z_i in z[:-1]])
                classification_loss = -torch.sum(y*self.y_scale * torch.log(y_hat) * label)
                loss = goodness_loss + classification_loss
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        
        return y_hat, (goodness_loss, classification_loss)

    def infer_and_generate(self, x, y, label, num_steps, optimiser=None):
        z = self.forward(x)
        z_g = torch.zeros((x.shape[0], self.g_units), requires_grad=True).to(self.device)
        z_g.retain_grad()

        for i in range(num_steps):

            y_hat, z = self.step_rep(x, y, z)
            x_hat, E, z_g = self.step_gen(x, z, z_g)
        
            if optimiser is not None:
                rep_g_loss = sum([F.binary_cross_entropy_with_logits(goodness(z_i), label) for z_i in z[:-1]])
                rep_c_loss = -torch.sum(y*self.y_scale * torch.log(y_hat) * label.unsqueeze(1)) / label.sum()
                gen_loss = sum(E)
                L = rep_g_loss + rep_c_loss + gen_loss
                optimiser.zero_grad()
                L.backward()
                optimiser.step()
        
        return x_hat, y_hat, (rep_g_loss, rep_c_loss), gen_loss