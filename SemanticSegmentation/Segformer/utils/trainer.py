import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.optimization import get_linear_schedule_with_warmup

from utils.config import TrainConfig
from utils.export_utils import export_onnx_model
from utils.training_utils import evaluate, init_wandb, require_loss, save_metadata

logger = logging.getLogger(__name__)


def resolve_model_source(model_name: str) -> str:
    model_source: str = model_name
    model_path = Path(model_name).expanduser()
    model_candidates: List[Path] = [model_path]
    if not model_path.is_absolute():
        model_candidates.append((Path(__file__).resolve().parent.parent / model_path).resolve())

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


class Trainer:
    def __init__(
        self,
        network: Optional[nn.Module],
        configs: TrainConfig,
        num_labels: int,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        project_name: str = "Semantic Segmentation",
    ) -> None:
        self.configs = configs
        self.project_name = project_name
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this script, but no CUDA device is available.")

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        self.device = torch.device("cuda")
        self.amp_dtype = torch.float16 if configs.amp_dtype == "fp16" else torch.bfloat16

        self.model_source = resolve_model_source(configs.model_name)
        self.processor = SegformerImageProcessor.from_pretrained(
            self.model_source,
            do_resize=True,
            size={"height": configs.image_size, "width": configs.image_size},
            do_reduce_labels=False,
        )
        if network is not None:
            self.model = network
        else:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_source,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
            )
        self.model.to(device=self.device, memory_format=torch.channels_last)
        self.train_model: nn.Module = self.model

        try:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=configs.learning_rate,
                weight_decay=configs.weight_decay,
                fused=True,
            )
        except TypeError:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=configs.learning_rate,
                weight_decay=configs.weight_decay,
            )

        self.output_dir = configs.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_miou = -1.0
        self.best_dir = self.output_dir / "best_model"
        self.final_dir = self.output_dir / "final_model"

        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.train_size = 0
        self.val_size = 0

        self.wandb_run = init_wandb(configs, self.num_labels)

    def set_dataset(
        self,
        train_dataset: DataLoader,
        val_dataset: Optional[DataLoader] = None,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
    ) -> None:
        self.train_loader = train_dataset
        self.val_loader = val_dataset
        if train_size is None:
            train_size = len(train_dataset.dataset)
        if val_size is None:
            val_size = len(val_dataset.dataset) if val_dataset is not None else 0
        self.train_size = train_size
        self.val_size = val_size

        save_metadata(
            self.output_dir,
            self.id2label,
            self.label2id,
            self.configs,
            train_size=self.train_size,
            val_size=self.val_size,
        )

    def _create_scheduler_and_scaler(self, epochs: int) -> Tuple[torch.optim.lr_scheduler.LambdaLR, torch.GradScaler]:
        if self.train_loader is None:
            raise RuntimeError("Call set_dataset(...) before run(...).")

        total_steps = epochs * max(len(self.train_loader), 1)
        warmup_steps = math.floor(total_steps * self.configs.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        scaler = torch.GradScaler(enabled=self.amp_dtype == torch.float16)
        return scheduler, scaler

    def run_training_epoch(
        self,
        epoch: int,
        epochs: int,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        scaler: torch.GradScaler,
    ) -> float:
        if self.train_loader is None:
            raise RuntimeError("Call set_dataset(...) before run(...).")

        self.train_model.train()
        running_loss = 0.0

        train_bar = tqdm(
            self.train_loader,
            total=len(self.train_loader),
            desc=f"Train {epoch + 1}/{epochs}",
            leave=False,
            dynamic_ncols=True,
        )
        for step, batch in enumerate(train_bar, start=1):
            pixel_values = batch.pixel_values.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            labels = batch.labels.to(self.device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                outputs: SemanticSegmenterOutput = self.train_model(pixel_values=pixel_values, labels=labels)
                loss = require_loss(outputs.loss)

            self.optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            scheduler.step()

            loss_value = loss.item()
            running_loss += loss_value
            global_step = epoch * len(self.train_loader) + step
            avg_loss = running_loss / step
            train_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "train/loss_step": loss_value,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch + 1,
                    },
                    step=global_step,
                )

        train_loss = running_loss / max(len(self.train_loader), 1)
        logger.info("epoch=%d/%d train_loss=%.4f", epoch + 1, epochs, train_loss)
        return train_loss

    def validate_and_checkpoint(self, epoch: int, epochs: int, train_loss: float) -> None:
        if self.train_loader is None:
            raise RuntimeError("Call set_dataset(...) before run(...).")

        epoch_metrics: Dict[str, float] = {"train/loss_epoch": train_loss, "train/epoch": float(epoch + 1)}
        if self.val_loader is not None:
            val_loss, val_miou = evaluate(
                self.train_model,
                self.val_loader,
                self.device,
                num_labels=self.num_labels,
                ignore_index=self.configs.ignore_index,
                amp_dtype=self.amp_dtype,
            )
            logger.info("epoch=%d/%d val_loss=%.4f val_miou=%.4f", epoch + 1, epochs, val_loss, val_miou)
            epoch_metrics["val/loss"] = val_loss
            epoch_metrics["val/mean_iou"] = val_miou

            if val_miou > self.best_miou:
                self.best_miou = val_miou
                self.best_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(self.best_dir)
                self.processor.save_pretrained(self.best_dir)
                epoch_metrics["val/best_mean_iou"] = self.best_miou
                logger.info("new_best_checkpoint epoch=%d best_miou=%.4f dir=%s", epoch + 1, self.best_miou, self.best_dir)
        elif epoch + 1 == epochs:
            self.final_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(self.final_dir)
            self.processor.save_pretrained(self.final_dir)
            logger.info("saved_final_checkpoint dir=%s", self.final_dir)

        if self.wandb_run is not None:
            self.wandb_run.log(epoch_metrics, step=(epoch + 1) * len(self.train_loader))

        if self.configs.save_every_epoch:
            epoch_dir = self.output_dir / f"checkpoint-epoch-{epoch + 1}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(epoch_dir)
            self.processor.save_pretrained(epoch_dir)
            logger.info("saved_epoch_checkpoint epoch=%d dir=%s", epoch + 1, epoch_dir)

    def export_onnx(self) -> None:
        if self.val_loader is not None and self.best_dir.exists():
            export_model = SegformerForSemanticSegmentation.from_pretrained(self.best_dir).to(self.device)
        else:
            export_model = self.model

        onnx_path = self.output_dir / "model.onnx"
        export_onnx_model(export_model, onnx_path, image_size=self.configs.image_size, device=self.device)
        logger.info("saved_onnx=%s", onnx_path)

        if self.wandb_run is not None:
            self.wandb_run.summary["output/onnx_path"] = str(onnx_path)
            if self.val_loader is not None and self.best_miou >= 0.0:
                self.wandb_run.summary["val/best_mean_iou"] = self.best_miou

    def run_epoch_loop(self, epochs: int) -> None:
        scheduler, scaler = self._create_scheduler_and_scaler(epochs)
        for epoch in range(epochs):
            train_loss = self.run_training_epoch(epoch, epochs, scheduler, scaler)
            self.validate_and_checkpoint(epoch, epochs, train_loss)

    def run_training(self, epochs: Optional[int] = None) -> None:
        if self.train_loader is None:
            raise RuntimeError("Call set_dataset(...) before run_training(...).")

        actual_epochs = epochs or self.configs.epochs
        logger.info(
            "training_start project=%s model=%s epochs=%d batch_size=%d image_size=%d amp=%s lr=%g",
            self.project_name,
            self.model_source,
            actual_epochs,
            self.configs.batch_size,
            self.configs.image_size,
            self.configs.amp_dtype,
            self.configs.learning_rate,
        )

        try:
            self.run_epoch_loop(actual_epochs)
            self.export_onnx()
        finally:
            if self.wandb_run is not None:
                self.wandb_run.finish()

    def run(self, epochs: Optional[int] = None) -> None:
        self.run_training(epochs)
