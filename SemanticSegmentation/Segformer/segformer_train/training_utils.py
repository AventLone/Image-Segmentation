import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.modeling_outputs import SemanticSegmenterOutput

from segformer_train.config import TrainConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_mean_iou(preds: torch.Tensor, labels: torch.Tensor, num_labels: int, ignore_index: int) -> float:
    per_class_ious: List[float] = []
    for class_id in range(num_labels):
        pred_mask = preds == class_id
        target_mask = labels == class_id
        valid_mask = labels != ignore_index

        pred_mask = pred_mask & valid_mask
        target_mask = target_mask & valid_mask

        intersection = torch.logical_and(pred_mask, target_mask).sum().item()
        union = torch.logical_or(pred_mask, target_mask).sum().item()
        if union > 0:
            per_class_ious.append(intersection / union)

    if not per_class_ious:
        return 0.0
    return float(sum(per_class_ious) / len(per_class_ious))


def require_loss(loss: Optional[torch.Tensor]) -> torch.Tensor:
    if loss is None:
        raise RuntimeError("SegFormer did not return a training loss. Check that labels are provided.")
    return loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_labels: int,
    ignore_index: int,
    amp_dtype: torch.dtype,
) -> Tuple[float, float]:
    model.eval()
    losses: List[float] = []
    mean_ious: List[float] = []

    val_bar = tqdm(dataloader, total=len(dataloader), desc="Validation", leave=False, dynamic_ncols=True)
    with torch.inference_mode():
        for batch in val_bar:
            pixel_values = batch.pixel_values.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            labels: torch.Tensor = batch.labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs: SemanticSegmenterOutput = model(pixel_values=pixel_values, labels=labels)
            loss = require_loss(outputs.loss)
            losses.append(loss.item())

            upsampled_logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            predictions = upsampled_logits.argmax(dim=1)
            batch_miou = compute_mean_iou(predictions, labels, num_labels=num_labels, ignore_index=ignore_index)
            mean_ious.append(batch_miou)
            val_bar.set_postfix(loss=f"{loss.item():.4f}", miou=f"{batch_miou:.4f}")

    model.train()
    avg_loss = float(sum(losses) / len(losses)) if losses else 0.0
    avg_miou = float(sum(mean_ious) / len(mean_ious)) if mean_ious else 0.0
    return avg_loss, avg_miou


def save_metadata(
    output_dir: Path,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    args: TrainConfig,
    train_size: int,
    val_size: int,
) -> None:
    metadata = {
        "id2label": {str(key): value for key, value in id2label.items()},
        "label2id": label2id,
        "dataset": str(args.dataset),
        "val_ratio": args.val_ratio,
        "train_size": train_size,
        "val_size": val_size,
        "image_size": args.image_size,
        "ignore_index": args.ignore_index,
    }
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def init_wandb(args: TrainConfig, num_labels: int) -> Optional[Run]:
    if not args.use_wandb:
        return None

    return wandb.init(
        project=args.wandb_project,
        name=datetime.now().strftime("%Y.%m.%d-%H:%M"),
        config={
            "model_name": args.model_name,
            "image_size": args.image_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "num_workers": args.num_workers,
            "amp_dtype": args.amp_dtype,
            "cuda_only": True,
            "ignore_index": args.ignore_index,
            "num_labels": num_labels,
            "dataset": str(args.dataset),
            "val_ratio": args.val_ratio,
            "output_dir": str(args.output_dir),
        },
    )
