import torch
from torch import nn, Tensor
from typing import List, Tuple


class DinoV3Backbone(nn.Module):
    def __init__(self, backbone: nn.Module, embed_dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.proj = nn.Conv2d(embed_dim, out_channels, 1)

    def _tokens_to_map(self, tokens: Tensor, h: int, w: int):
        return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, h, w)

    def forward(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        feat = self.backbone(x)
        if feat.dim() == 3:
            feat = self._tokens_to_map(feat, h, w)
        feat = self.proj(feat)
        mask = torch.zeros((B, h, w), dtype=torch.bool, device=x.device)
        return [feat], [mask]
