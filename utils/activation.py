import torch
import torch.nn as nn


class limiterActivation(nn.Module):
    def __init__(self, min_val, max_val):
        super(limiterActivation, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, min=self.min_val, max=self.max_val)


class normActivation(nn.Module):
    def __init__(self, dim=1):
        super(normActivation, self).__init__()
        self.esp = 1e-12
        self.dim = dim

    def forward(self, x):
        x = x / (torch.norm(x, dim=self.dim, keepdim=True) + self.esp)
