import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# define default config for CIFAR-10
@dataclass
class ViTConfig:
    img_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    num_heads: int = 12
    num_layers: int = 12
    mlp_dim: int = 3072
    dropout: float = 0.0
    bias: bool = True
    n_embd: int = 768

# replace eventually with nn.LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.num_heads == 0
        

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()


class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
