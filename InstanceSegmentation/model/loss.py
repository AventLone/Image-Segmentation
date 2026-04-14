import torch


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.flatten(1)
    target = target.flatten(1)
    num = 2 * (pred * target).sum(1)
    den = pred.sum(1) + target.sum(1)
    return (1 - (num + eps) / (den + eps)).mean()


import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher

        weight = torch.ones(num_classes + 1)
        weight[-1] = eos_coef
        self.register_buffer("weight", weight)

    def loss_labels(self, logits, targets, indices):
        bs, Q, _ = logits.shape
        target_cls = torch.full((bs, Q), self.num_classes, dtype=torch.long, device=logits.device)

        for b, (src, tgt) in enumerate(indices):
            if len(src) > 0:
                target_cls[b, src] = targets[b]["labels"][tgt]

        return F.cross_entropy(logits.flatten(0, 1), target_cls.flatten(0, 1), weight=self.weight)

    def loss_boxes(self, boxes, targets, indices):
        l1, giou = [], []
        for b, (src, tgt) in enumerate(indices):
            if len(src) == 0:
                continue
            src_b = boxes[b, src]
            tgt_b = targets[b]["boxes"][tgt]
            l1.append(F.l1_loss(src_b, tgt_b))
            giou.append((1 - torch.diag(generalized_box_iou(src_b, tgt_b))).mean())
        if len(l1) == 0:
            zero = boxes.sum() * 0
            return zero, zero
        return torch.stack(l1).mean(), torch.stack(giou).mean()

    def loss_masks(self, masks, targets, indices):
        bce, dice = [], []
        for b, (src, tgt) in enumerate(indices):
            if len(src) == 0:
                continue
            src_m = masks[b, src].sigmoid()
            tgt_m = targets[b]["masks"][tgt].float()
            src_m = F.interpolate(src_m[:, None], size=tgt_m.shape[-2:], mode="bilinear")[:, 0]
            bce.append(F.binary_cross_entropy(src_m, tgt_m))
            dice.append(dice_loss(src_m, tgt_m))
        if len(bce) == 0:
            zero = masks.sum() * 0
            return zero, zero
        return torch.stack(bce).mean(), torch.stack(dice).mean()

    def forward(self, outputs, targets):
        indices = self.matcher(outputs["pred_logits"], outputs["pred_boxes"], targets)

        loss_ce = self.loss_labels(outputs["pred_logits"], targets, indices)
        loss_bbox, loss_giou = self.loss_boxes(outputs["pred_boxes"], targets, indices)
        loss_mask, loss_dice = self.loss_masks(outputs["pred_masks"], targets, indices)

        return {
            "loss": loss_ce + loss_bbox + loss_giou + loss_mask + loss_dice,
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
        }
