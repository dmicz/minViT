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
    bias: bool = False
    n_embd: int = 768
    
# standard patch embedding, implement other module for Conv2d type embedding
class PatchEmbedding(nn.Module):
    def __init__(self, img_size = 32, patch_size = 4, in_chans = 3, embed_dim = 768):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size    # P
        self.in_chans   = in_chans      # C
        self.embed_dim  = embed_dim     # D

        self.num_patches = (img_size // patch_size) ** 2        # N = H*W/P^2
        self.flatten_dim = patch_size * patch_size * in_chans   # P^2*C
        
        self.proj = nn.Linear(self.flatten_dim, embed_dim) # (P^2*C,D)

        self.position_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.class_embed    = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, self.num_patches, -1)
        x = x.permute(0, 2, 1, 3).reshape(B, self.num_patches, -1)

        x = self.proj(x)

        cls_emb = self.class_embed.expand(B, -1, -1)
        x = torch.cat((cls_emb, x), dim = 1)

        x = x + self.position_embed
        return x

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
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.num_heads == 0

        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.num_heads
        self.n_embd = config.n_embd

    def forward(self, x):
        B, N, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, N, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, N, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, N, self.n_head, C // self.n_head).transpose(1, 2)

        # do NOT use causal attention as we are not dealing with sequential data (image patches are unordered)
        y = torch.nn.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout=self.attn_dropout if self.training else 0, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, N, C)

        y = self.resid_dropout(self.c_proj(y))
        return y
        

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        y = self.ln_1(x)
        y = self.attn(y)
        x += y
        y = self.ln_2(x)
        y = self.mlp(y)
        x += y

        return x


class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            token_encoding = nn.Embedding(config.in_channels, config.n_embd),
            position_encoding = nn.Embedding(),
            blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            ln_post = LayerNorm(config.n_embd, config.bias)
        ))

         

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
