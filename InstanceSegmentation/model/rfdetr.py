from dataclasses import dataclass

@dataclass
class RfDetrSegConfig:
    num_classes: int
    num_queries: int = 100
    hidden_dim: int = 256
    num_heads: int = 8
    num_decoder_layers: int = 6
    dim_feedforward: int = 1024
    mask_dim: int = 128
    backbone_embed_dim: int = 768
    backbone_patch_size: int = 16
    freeze_backbone: bool = True

# ---------------- model/rf_detr_seg.py ----------------
import torch
from torch import nn
from .backbone import DinoV3Backbone
from .neck import MaskFeaturePyramid
from .components import SimpleTransformerDecoder
from .matcher import HungarianMatcher
from .loss import SetCriterion


class RFDETRSeg(nn.Module):
    def __init__(self, backbone, cfg: RFDetrSegConfig):
        super().__init__()
        self.backbone = DinoV3Backbone(backbone, cfg.backbone_embed_dim, cfg.backbone_patch_size, cfg.hidden_dim)
        self.neck = MaskFeaturePyramid(cfg.hidden_dim, cfg.hidden_dim, cfg.mask_dim)
        self.query = nn.Embedding(cfg.num_queries, cfg.hidden_dim)
        self.decoder = SimpleTransformerDecoder(cfg.hidden_dim, cfg.num_heads, cfg.num_decoder_layers)

        self.cls = nn.Linear(cfg.hidden_dim, cfg.num_classes + 1)
        self.box = nn.Linear(cfg.hidden_dim, 4)
        self.mask = nn.Linear(cfg.hidden_dim, cfg.mask_dim)

        self.matcher = HungarianMatcher()
        self.criterion = SetCriterion(cfg.num_classes, self.matcher)

    def forward(self, x):
        feat, _ = self.backbone(x)
        feat, mask_feat = self.neck(feat, None)
        mem = feat.flatten(2).transpose(1, 2)

        q = self.query.weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
        hs = self.decoder(q, mem)

        logits = self.cls(hs)
        boxes = self.box(hs).sigmoid()
        masks = torch.einsum("bqc,bchw->bqhw", self.mask(hs), mask_feat)

        return {"pred_logits": logits, "pred_boxes": boxes, "pred_masks": masks}

    def loss(self, x, targets):
        outputs = self.forward(x)
        return self.criterion(outputs, targets)