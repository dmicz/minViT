import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# define default config for CIFAR-10
@dataclass
class ViTConfig:
    img_height: int = 32
    img_width: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    num_heads: int = 4
    num_layers: int = 8
    mlp_dim: int = 192
    dropout: float = 0.1
    bias: bool = True
    n_embd: int = 96
    pretraining: bool = False
    hybrid_embedding: bool = True
    
# patch embedding with no conv2d
class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_height = config.img_height     # H
        self.img_width  = config.img_width      # W
        self.patch_size = config.patch_size     # P
        self.in_chans   = config.in_channels    # C
        self.embed_dim  = config.n_embd         # D

        self.num_patches = (self.img_height * self.img_width) // (self.patch_size ** 2)     # N = H*W/P^2
        self.flatten_dim = self.patch_size * self.patch_size * self.in_chans   # P^2*C
        
        self.proj = nn.Linear(self.flatten_dim, self.embed_dim) # (P^2*C,D)

        self.position_embed = nn.Parameter(torch.randn(1, 1 + self.num_patches, self.embed_dim))
        self.class_embed    = nn.Parameter(torch.randn(1, 1, self.embed_dim))

    def forward(self, x):
        B = x.shape[0]

        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.reshape(1, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, self.num_patches, -1)

        x = self.proj(x)

        cls_emb = self.class_embed.expand(B, -1, -1)
        x = torch.cat((cls_emb, x), dim = 1)

        x = x + self.position_embed
        return x
    

class HybridPatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_height = config.img_height     # H
        self.img_width  = config.img_width      # W
        self.patch_size = config.patch_size     # P
        self.in_chans   = config.in_channels    # C
        self.embed_dim  = config.n_embd         # D

        self.num_patches = (self.img_height * self.img_width) // (self.patch_size ** 2)     # N = H*W/P^2
        self.flatten_dim = self.patch_size * self.patch_size * self.in_chans   # P^2*C
        
        self.conv = nn.Conv2d(self.in_chans, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

        self.position_embed = nn.Parameter(torch.randn(1, 1 + self.num_patches, self.embed_dim))
        self.class_embed    = nn.Parameter(torch.randn(1, 1, self.embed_dim))

    def forward(self, x):
        B = x.shape[0]

        x = self.conv(x)
        x = x.reshape(B, self.embed_dim, -1)
        x = x.transpose(1, 2)

        cls_emb = self.class_embed.expand(B, -1, -1)
        x = torch.cat((cls_emb, x), dim = 1)

        x = x + self.position_embed
        return x


# feed-forward MLP
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.mlp_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.mlp_dim, config.n_embd, bias=config.bias)
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

        self.embed_dim   = config.n_embd
        self.num_heads   = config.num_heads
        self.head_dim    = config.n_embd // config.num_heads

        self.query   = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.key     = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value   = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.out     = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, N, _ = x.size()

        q = self.query(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # do NOT use causal attention as we are not dealing with sequential data (image patches are unordered)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.softmax(dim=-1)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, self.embed_dim)
        out = self.attn_dropout(out)

        out = self.resid_dropout(self.out(out))
        return out
        

    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
# classification head customizable for pretraining or finetuning
class ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pretraining = config.pretraining
        self.head = nn.ModuleList()
        if self.pretraining:
            self.head.append(nn.Linear(config.n_embd, config.mlp_dim, bias=config.bias))
            self.head.append(nn.ReLU())
            self.head.append(nn.Linear(config.mlp_dim, config.num_classes, bias=config.bias))
        else:
            self.head.append(nn.Linear(config.n_embd, config.num_classes, bias=config.bias))

    def forward(self, x):
        return self.head(x)


class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            pe = HybridPatchEmbedding(config) if config.hybrid_embedding else PatchEmbedding(config),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.head = ClassificationHead(config)

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, x, targets=None):
        emb = self.transformer.pe(x)
        x = self.transformer.drop(emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        class_token = x[:, 0]
        logits = self.head(class_token)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None

        return logits, loss

