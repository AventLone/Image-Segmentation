import math, logging, torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from huggingface_hub.utils import disable_progress_bars
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from transformers.utils import logging as transformers_logging
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.optimization import get_linear_schedule_with_warmup
from utils.config import TrainConfig
from utils.export_utils import export_onnx_model
from utils.training_utils import evaluate, init_wandb, require_loss, save_metadata

from utils.common import logging_handler
logging.basicConfig(level=logging.INFO, handlers=[logging_handler])
disable_progress_bars()
transformers_logging.disable_progress_bar()

# Silence ONNX optimizer chatter
logging.getLogger("onnxscript").setLevel(logging.WARNING)
logging.getLogger("onnx_ir").setLevel(logging.WARNING)



class Trainer:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    DEVICE = torch.device(type="cuda", index=0)

    def __init__(self, network: nn.Module, configs: TrainConfig, num_labels: int,
                 id2label: Dict[int, str], label2id: Dict[str, int],
                 project_name: str = "Semantic Segmentation") -> None:
        self._configs = configs
        self._project_name = project_name
        self._num_labels = num_labels
        self._id2label = id2label
        self._label2id = label2id

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this script, but no CUDA device is available.")

        self._amp_dtype = torch.float16 if configs.amp_dtype == "fp16" else torch.bfloat16
        self._processor = SegformerImageProcessor.from_pretrained(configs.pretrained_path)
        self._model = network
        self._model.to(device=Trainer.DEVICE, memory_format=torch.channels_last)
        self._train_model: nn.Module = self._model

        try:
            self._optimizer = AdamW(self._model.parameters(), lr=configs.learning_rate, weight_decay=configs.weight_decay, fused=True)
        except TypeError:
            self._optimizer = AdamW(self._model.parameters(), lr=configs.learning_rate, weight_decay=configs.weight_decay)

        self._output_dir = configs.output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._best_miou = -1.0
        self._best_dir = self._output_dir / "best_model"
        self._final_dir = self._output_dir / "final_model"

        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None
        self._train_size = 0
        self._val_size = 0

        self._wandb_run = init_wandb(configs, self._num_labels)

    def set_dataset(self, train_dataset: DataLoader, val_dataset: Optional[DataLoader] = None, 
                    train_size: Optional[int] = None, val_size: Optional[int] = None) -> None:
        self._train_loader = train_dataset
        self._val_loader = val_dataset
        if train_size is None:
            train_size = len(train_dataset.dataset)
        if val_size is None:
            val_size = len(val_dataset.dataset) if val_dataset is not None else 0
        self._train_size = train_size
        self._val_size = val_size

        save_metadata(self._output_dir, self._id2label, self._label2id, self._configs, train_size=self._train_size, val_size=self._val_size)

    def _create_scheduler_and_scaler(self, epochs: int) -> Tuple[torch.optim.lr_scheduler.LambdaLR, torch.GradScaler]:
        if self._train_loader is None:
            raise RuntimeError("Call set_dataset(...) before run(...).")

        total_steps = epochs * max(len(self._train_loader), 1)
        warmup_steps = math.floor(total_steps * self._configs.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(self._optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        # GradScaler is only needed for fp16 to prevent underflow; bf16 has a much wider exponent range.
        scaler = torch.GradScaler(enabled=self._amp_dtype == torch.float16)
        return scheduler, scaler

    def _run_training_epoch(self, epoch: int, epochs: int, scheduler: torch.optim.lr_scheduler.LambdaLR, scaler: torch.GradScaler) -> float:
        if self._train_loader is None:
            raise RuntimeError("Call set_dataset(...) before run(...).")

        self._train_model.train()
        running_loss = 0.0

        train_bar = tqdm(self._train_loader, total=len(self._train_loader), desc=f"Train {epoch + 1}/{epochs}", 
                         leave=True, dynamic_ncols=True, unit="batches")
        for step, batch in enumerate(train_bar, start=1):
            pixel_values = batch.pixel_values.to(Trainer.DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
            labels = batch.labels.to(Trainer.DEVICE, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                outputs: SemanticSegmenterOutput = self._train_model(pixel_values=pixel_values, labels=labels)
                loss = require_loss(outputs.loss)

            self._optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(self._optimizer)
                scaler.update()
            else:
                loss.backward()
                self._optimizer.step()
            scheduler.step()

            loss_value = loss.item()
            running_loss += loss_value
            global_step = epoch * len(self._train_loader) + step
            avg_loss = running_loss / step
            # train_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            train_bar.set_postfix(loss=f"{avg_loss:.3f}")

            if self._configs.use_wandb:
                self._wandb_run.log({"train/loss_step": loss_value,
                                     "train/learning_rate": scheduler.get_last_lr()[0],
                                     "train/epoch": epoch + 1}, step=global_step)

        train_loss = running_loss / max(len(self._train_loader), 1)
        # logging.info(f"Epoch {epoch + 1}/{epochs}: train_loss = {train_loss:.4f}")
        return train_loss


    def _validate_and_checkpoint(self, epoch: int, epochs: int, train_loss: float) -> None:
        if self._train_loader is None:
            raise RuntimeError("Call set_dataset(...) before run(...).")

        epoch_metrics: Dict[str, float] = {"train/loss_epoch": train_loss, "train/epoch": float(epoch + 1)}
        if self._val_loader is not None:
            val_loss, val_miou = evaluate(self._train_model, self._val_loader,Trainer.DEVICE, num_labels=self._num_labels,
                                          ignore_index=self._configs.ignore_index,amp_dtype=self._amp_dtype)
            logging.info(f"Epoch {epoch + 1}/{epochs}: val_miou = {val_miou:.4f}")

            epoch_metrics["val/loss"] = val_loss
            epoch_metrics["val/mean_iou"] = val_miou

            if val_miou > self._best_miou:
                self._best_miou = val_miou
                self._best_dir.mkdir(parents=True, exist_ok=True)
                self._model.save_pretrained(self._best_dir)
                self._processor.save_pretrained(self._best_dir)
                epoch_metrics["val/best_mean_iou"] = self._best_miou
                logging.info(f"new_best_checkpoint epoch={epoch + 1}, best_miou={self._best_miou:.4f}")
        elif epoch + 1 == epochs:
            self._final_dir.mkdir(parents=True, exist_ok=True)
            self._model.save_pretrained(self._final_dir)
            self._processor.save_pretrained(self._final_dir)
            logging.info(f"saved_final_checkpoint dir={self._final_dir}")

        if self._configs.use_wandb:
            self._wandb_run.log(epoch_metrics, step=(epoch + 1) * len(self._train_loader))


    def _export_onnx(self) -> None:
        if self._val_loader is not None and self._best_dir.exists():
            export_model = SegformerForSemanticSegmentation.from_pretrained(self._best_dir).to(Trainer.DEVICE)
        else:
            export_model = self._model

        onnx_path = self._output_dir / "model.onnx"
        export_onnx_model(export_model, onnx_path, image_size=self._configs.image_size, device=Trainer.DEVICE)
        logging.info("saved_onnx=%s", onnx_path)

        if self._configs.use_wandb:
            self._wandb_run.summary["output/onnx_path"] = str(onnx_path)
            if self._val_loader is not None and self._best_miou >= 0.0:
                self._wandb_run.summary["val/best_mean_iou"] = self._best_miou


    def _train_loop(self, epochs: int) -> None:
        scheduler, scaler = self._create_scheduler_and_scaler(epochs)
        for epoch in range(epochs):
            train_loss = self._run_training_epoch(epoch, epochs, scheduler, scaler)
            self._validate_and_checkpoint(epoch, epochs, train_loss)


    def run(self, epochs: int) -> None:
        if self._train_loader is None:
            raise RuntimeError("Call set_dataset(...) before run_training(...).")
        
        logging.info(f"Train Parameters:\n"
                     f"\t Device:          {Trainer.DEVICE.type}\n"
                     f"\t Epochs:          {epochs}\n"
                     f"\t Input size:      {self._configs.image_size}x{self._configs.image_size}\n"
                     f"\t Batch size:      {self._configs.batch_size}\n"
                     f"\t Learning rate:   {self._configs.learning_rate}\n"
                     f"\t AMP dtype:       {self._configs.amp_dtype}\n")
        try:
            self._train_loop(epochs)
            self._export_onnx()
        finally:
            if self._configs.use_wandb:
                self._wandb_run.finish()
