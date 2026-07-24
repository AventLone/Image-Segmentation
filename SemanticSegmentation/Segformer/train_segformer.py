import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.optimization import get_linear_schedule_with_warmup

from segformer_train.config import TrainConfig
from segformer_train.data import (
    SegmentationDataset,
    collate_batch,
    discover_pairs,
    infer_num_labels,
    load_class_names,
)
from segformer_train.export_utils import export_onnx_model
from segformer_train.training_utils import evaluate, init_wandb, require_loss, save_metadata, set_seed
from utils.common import logging_handler

logging.basicConfig(level=logging.INFO, handlers=[logging_handler])
logger = logging.getLogger(__name__)


def resolve_model_source(model_name: str) -> str:
    model_source: str = model_name
    model_path = Path(model_name).expanduser()
    model_candidates: List[Path] = [model_path]
    if not model_path.is_absolute():
        model_candidates.append((Path(__file__).resolve().parent / model_path).resolve())

    local_model_path = next((candidate for candidate in model_candidates if candidate.exists()), None)
    if local_model_path is not None:
        model_source = str(local_model_path)
        logger.info("loading_model_source local=%s", model_source)
    else:
        looks_like_local_path = any(sep in model_name for sep in ("/", "\\")) or model_name.startswith(".")
        if looks_like_local_path:
            searched_paths = ", ".join(str(path) for path in model_candidates)
            raise FileNotFoundError(
                f"Local model path not found: {model_name}. Searched: {searched_paths}. "
                "Provide a valid local checkpoint path or a Hub model id like 'nvidia/mit-b0'."
            )
        logger.info("loading_model_source hub=%s", model_name)

    return model_source


def main() -> None:
    # args = TrainConfig(
    #     dataset=Path("/media/avent/DATA/generated_data/train/2026.07.23-14:28"),
    # )
    args = TrainConfig()
    set_seed(args.seed)

    if args.dataset is None:
        raise ValueError("Set TrainConfig.dataset to your dataset folder containing 'rgb' and 'mask'.")

    output_dir = args.output_dir

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script, but no CUDA device is available.")

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1).")

    dataset_rgb = args.dataset / "rgb"
    dataset_mask = args.dataset / "mask"
    pairs = discover_pairs(dataset_rgb, dataset_mask)

    class_names = load_class_names(args.classes)
    num_labels = len(class_names) if class_names else infer_num_labels(pairs, args.ignore_index)
    if class_names and len(class_names) != num_labels:
        raise ValueError("Class file length must match the number of labels in your masks.")

    id2label = {index: name for index, name in enumerate(class_names or [f"class_{i}" for i in range(num_labels)])}
    label2id = {name: index for index, name in id2label.items()}

    model_source = resolve_model_source(args.model_name)
    processor = SegformerImageProcessor.from_pretrained(
        model_source,
        do_resize=True,
        size={"height": args.image_size, "width": args.image_size},
        do_reduce_labels=False,
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_source,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    total_size = len(pairs)
    if args.val_ratio > 0 and total_size < 2:
        raise ValueError("Need at least 2 samples to create a train/validation split.")

    val_size = int(total_size * args.val_ratio)
    if args.val_ratio > 0 and val_size == 0:
        val_size = 1
    train_size = total_size - val_size
    if train_size <= 0:
        raise ValueError("Validation split leaves no samples for training. Reduce --val-ratio.")

    if val_size > 0:
        generator = torch.Generator().manual_seed(args.seed)
        train_pairs_subset, val_pairs_subset = random_split(pairs, [train_size, val_size], generator=generator)
        train_pairs = [train_pairs_subset[i] for i in range(len(train_pairs_subset))]
        val_pairs = [val_pairs_subset[i] for i in range(len(val_pairs_subset))]
    else:
        train_pairs = list(pairs)
        val_pairs = []

    train_dataset = SegmentationDataset(
        train_pairs,
        processor,
        image_size=args.image_size,
        use_augmentation=not args.disable_augmentation,
    )
    val_dataset = (
        SegmentationDataset(
            val_pairs,
            processor,
            image_size=args.image_size,
            use_augmentation=False,
        )
        if val_pairs
        else None
    )

    logger.info("dataset_split train=%d val=%d", train_size, val_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_batch,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_batch,
        )
        if val_dataset
        else None
    )

    device = torch.device("cuda")
    model.to(device=device, memory_format=torch.channels_last)
    train_model: nn.Module = model

    try:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, fused=True)
    except TypeError:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = args.epochs * max(len(train_loader), 1)
    warmup_steps = math.floor(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.GradScaler(enabled=amp_dtype == torch.float16)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(output_dir, id2label, label2id, args, train_size=train_size, val_size=val_size)
    run: Any | None = init_wandb(args, num_labels)

    logger.info(
        "training_start model=%s epochs=%d batch_size=%d image_size=%d amp=%s lr=%g",
        model_source,
        args.epochs,
        args.batch_size,
        args.image_size,
        args.amp_dtype,
        args.learning_rate,
    )

    best_miou = -1.0
    best_dir = output_dir / "best_model"
    final_dir = output_dir / "final_model"

    try:
        epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs", dynamic_ncols=True)
        for epoch in epoch_bar:
            train_model.train()
            running_loss = 0.0

            train_bar = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"Train {epoch}/{args.epochs}",
                leave=False,
                dynamic_ncols=True,
            )
            for step, batch in enumerate(train_bar, start=1):
                pixel_values = batch.pixel_values.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                labels = batch.labels.to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs: SemanticSegmenterOutput = train_model(pixel_values=pixel_values, labels=labels)
                    loss = require_loss(outputs.loss)

                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                loss_value = loss.item()
                running_loss += loss_value
                global_step = (epoch - 1) * len(train_loader) + step
                avg_loss = running_loss / step
                train_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

                if run is not None:
                    run.log(
                        {
                            "train/loss_step": loss_value,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

            train_loss = running_loss / max(len(train_loader), 1)
            logger.info("epoch=%d/%d train_loss=%.4f", epoch, args.epochs, train_loss)

            epoch_metrics: Dict[str, float] = {"train/loss_epoch": train_loss, "train/epoch": float(epoch)}
            if val_loader is not None:
                val_loss, val_miou = evaluate(
                    train_model,
                    val_loader,
                    device,
                    num_labels=num_labels,
                    ignore_index=args.ignore_index,
                    amp_dtype=amp_dtype,
                )
                logger.info("epoch=%d/%d val_loss=%.4f val_miou=%.4f", epoch, args.epochs, val_loss, val_miou)
                epoch_metrics["val/loss"] = val_loss
                epoch_metrics["val/mean_iou"] = val_miou

                if val_miou > best_miou:
                    best_miou = val_miou
                    best_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(best_dir)
                    processor.save_pretrained(best_dir)
                    epoch_metrics["val/best_mean_iou"] = best_miou
                    logger.info("new_best_checkpoint epoch=%d best_miou=%.4f dir=%s", epoch, best_miou, best_dir)
            elif epoch == args.epochs:
                final_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(final_dir)
                processor.save_pretrained(final_dir)
                logger.info("saved_final_checkpoint dir=%s", final_dir)

            epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}")

            if run is not None:
                run.log(epoch_metrics, step=epoch * len(train_loader))

            if args.save_every_epoch:
                epoch_dir = output_dir / f"checkpoint-epoch-{epoch}"
                epoch_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(epoch_dir)
                processor.save_pretrained(epoch_dir)
                logger.info("saved_epoch_checkpoint epoch=%d dir=%s", epoch, epoch_dir)

        if val_loader is not None and best_dir.exists():
            export_model = SegformerForSemanticSegmentation.from_pretrained(best_dir).to(device)
        else:
            export_model = model

        onnx_path = output_dir / "model.onnx"
        export_onnx_model(export_model, onnx_path, image_size=args.image_size, device=device)
        logger.info("saved_onnx=%s", onnx_path)

        if run is not None:
            run.summary["output/onnx_path"] = str(onnx_path)
            if val_loader is not None and best_miou >= 0.0:
                run.summary["val/best_mean_iou"] = best_miou
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
