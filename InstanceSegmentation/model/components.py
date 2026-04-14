import torch
from torch import nn, Tensor
import torch.nn.functional as F


class DecoderLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, tgt, mem):
        tgt = self.norm1(tgt + self.self_attn(tgt, tgt, tgt)[0])
        tgt = self.norm2(tgt + self.cross_attn(tgt, mem, mem)[0])
        tgt = self.norm3(tgt + self.ff(tgt))
        return tgt


class SimpleTransformerDecoder(nn.Module):
    def __init__(self, dim, heads, layers):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(dim, heads) for _ in range(layers)])

    def forward(self, tgt, mem):
        for l in self.layers:
            tgt = l(tgt, mem)
        return tgt