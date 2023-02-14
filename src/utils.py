import torch
import torch.nn as nn
import numpy as np

def to_np(x):
    return x.cpu().detach().numpy()

def to_torch(x, device):
    return torch.from_numpy(x).to(device)

class Tanh(nn.Module):
    def forward(self, inp):
        return torch.tanh(inp)

    def deriv(self, inp):
        return 1.0 - torch.tanh(inp) ** 2.0

class Linear(nn.Module):
    def forward(self, inp):
        return inp

    def deriv(self, inp):
        return torch.ones((1,)).to(inp.device)