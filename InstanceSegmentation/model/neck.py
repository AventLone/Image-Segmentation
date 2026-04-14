import torch
from torch import nn, Tensor
from typing import List, Tuple


class MaskFeaturePyramid(nn.Module):
    def __init__(self, in_ch, hidden_dim, mask_dim):
        super().__init__()
        self.in_proj = nn.Conv2d(in_ch, hidden_dim, 1)
        self.mask_proj = nn.Conv2d(hidden_dim, mask_dim, 3, padding=1)

    def forward(self, features: List[Tensor], masks: List[Tensor]) -> Tuple[Tensor, Tensor]:
        x = self.in_proj(features[-1])
        return x, self.mask_proj(x)