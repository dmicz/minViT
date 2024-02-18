import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class ViTConfig:
    img_size: int = 224
    patch_size: int = 16
    num_classes: int = 10
    embed_dim: int = 768


class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self):
        pass
