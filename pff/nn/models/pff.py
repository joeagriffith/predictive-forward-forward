import torch
import torch.nn as nn
import torch.nn.functional as F


class PFF(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes

        self.R = nn.Module()
        self.R.bottom_up = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.R.bottom_up.append(nn.Linear(sizes[i], sizes[i+1]))
        self.R.top_down = nn.ModuleList()
        for i in range(1, len(sizes) - 1):
            self.R.top_down.append(nn.Linear(sizes[i+1], sizes[i]))
        self.R.lateral = nn.ModuleList()
        for i in range(1, len(sizes)):
            self.R.lateral.append(nn.Linear(sizes[i], sizes[i]))

        self.G = nn.Module()
        self.G.top_down = nn.ModuleList()
        for i in range(len(sizes)-1):
            self.G.top_down.append(nn.Linear(sizes[i+1], sizes[i]))
        self.G.top_down.append(nn.Linear(sizes[-1], sizes[-1]))
        