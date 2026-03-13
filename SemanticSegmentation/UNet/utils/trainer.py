import torch
import wandb
import logging
import pathlib
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from model.loss import *
from datetime import datetime
from typing import Iterable
from utils.common import load_config
from dataclasses import dataclass


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)


@dataclass
class TrainConfigs:
    learning_rate: float = 1.0e-2
    batch_size: int = 5

    classes_num: int = 0
    input_size: tuple[int, int, int] = (3, 512, 512)

    save_checkpoint: bool = False
    save_onnx: bool = False

    # ----- Datasets -----#
    datasets_dir_path: str = ""
    val_ratio: float = 0.2

    def __init__(self, yaml_path=None) -> None:
        if yaml_path is None:
            return
        config = load_config(yaml_path)

        self.learning_rate = config["LearningRate"]
        self.batch_size = config["BatchSize"]

        self.classes_num = config["ClassesNum"]
        self.input_size = tuple(config["InputSize"])

        self.save_checkpoint = config["SaveCheckpoint"]
        self.save_onnx = config["SaveOnnx"]

        self.datasets_dir_path = config["DatasetsDir"]
        self.val_ratio = config["ValRatio"]

    def print_info(self):
        logging.info(
            f"Train Parameters:\n"
            f"\t Device:          {DEVICE.type}\n"
            f"\t Input size:      {self.input_size}\n"
            f"\t Batch size:      {self.batch_size}\n"
            f"\t Learning rate:   {self.learning_rate}\n"
            f"\t Checkpoints:     {self.save_checkpoint}\n"
            f"\t Save ONNX:       {self.save_onnx}\n"
        )


class Trainer:
    CHECKPOINT_DIR = "./data/trained_model/checkpoints"
    ONNX_DIR = "./data/trained_model/onnx"
    PTH_DIR = "./data/trained_model/pth"

    def __init__(self, network: nn.Module, configs: TrainConfigs, project_name: str | None = None):
        self._configs = configs
        self._network: nn.Module = torch.compile(network).to(DEVICE)   # type: ignore # Modern PyTorch (JIT-free) compile path
        self._optimizer = optim.AdamW(self._network.parameters(), lr=self._configs.learning_rate, weight_decay=1e-6)

        self._train_dataset = None
        self._val_dataset = None

        self._network_name = network.__class__.__name__
        self._wandb_logger = None if project_name is None else wandb.init(project=project_name)

        # ---------- Create folders to store trained model parameters ---------- #
        pathlib.Path(Trainer.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
        pathlib.Path(Trainer.ONNX_DIR).mkdir(parents=True, exist_ok=True)
        pathlib.Path(Trainer.PTH_DIR).mkdir(parents=True, exist_ok=True)

    def set_dataset(self, train_dataset, val_dataset=None):
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset

    def save_dict(self):
        torch.save(self._network.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')

    def run(self, epochs: int):
        if self._train_dataset is None:
            raise ValueError("Train dataset is None!")

        n_train = len(self._train_dataset)
        n_val = len(self._val_dataset) if self._val_dataset else 0

        if self._wandb_logger is not None:
            self._wandb_logger.config.update(dict(model=self._network_name,
                                                  optimizer=self._optimizer.__class__.__name__,
                                                  learning_rate=self._configs.learning_rate,
                                                  batch_size=self._configs.batch_size,
                                                  epochs=epochs))
        logging.info(
            f"Starting training:\n"
            f"\t Epochs:          {epochs}\n"
            f"\t Training size:   {n_train}\n"
            f"\t Validation size: {n_val}"
        )

        criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss = LogSoftmax + NLLLoss. So it already contains `softmax`
        grad_scaler = torch.GradScaler(device=DEVICE.type)   # For mixed-precision training to keep training stable and fast.
        # scheduler = optim.lr_scheduler.OneCycleLR(self._optimizer,
        #                                           max_lr=self._configs.learning_rate,
        #                                           steps_per_epoch=len(self._train_dataset),
        #                                           epochs=epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=int(epochs / 2))

        for epoch in range(epochs):
            self._network.train()
            pbar = tqdm(self._train_dataset, desc=f"Epoch {epoch + 1}/{epochs}", unit=" batches")
            for images, masks in pbar:
                images = images.to(device=DEVICE)
                masks = masks.to(device=DEVICE)

                self._optimizer.zero_grad()
                with torch.autocast(device_type=DEVICE.type):   # AMP with explicit device type (new API)
                    logits = self._network(images)
                    probs = F.softmax(logits, dim=1)
                    one_hot = F.one_hot(masks, self._configs.classes_num).permute(0, 3, 1, 2).float()
                    loss: torch.Tensor = criterion(logits, masks) + dice_loss(probs, one_hot, multiclass=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(self._optimizer)
                grad_scaler.update()
                scheduler.step()

                pbar.set_postfix(loss=loss.item())

                # Logging (cleaner)
                if self._wandb_logger is not None:
                    self._wandb_logger.log({"train/loss": loss.item(), "train/lr": self._optimizer.param_groups[0]["lr"]})

            # -------- Validation -------- #
            train_acc = self._evaluate(self._train_dataset)
            logging.info(f"Train accuracy: mIoU: {train_acc["mIoU"]:.3f}, Pixel Acc: {train_acc["pixel acc"]:.3f}")
            if self._val_dataset is not None:
                val_acc = self._evaluate(self._val_dataset)
                # logging.info(f"Validation accuracy: {val_acc}")
                logging.info(f"Validation accuracy: mIoU: {val_acc["mIoU"]:.3f}, Pixel Acc: {val_acc["pixel acc"]:.3f}")

            if self._wandb_logger is not None:
                self._wandb_logger.log({"train/mIoU": train_acc["mIoU"], "val/mIoU": val_acc["mIoU"]})
                if self._val_dataset is not None:
                    self._wandb_logger.log({"train/pixel_acc": train_acc["pixel acc"], "val/pixel_acc": val_acc["pixel acc"]})

            if self._configs.save_checkpoint:
                torch.save(self._network.state_dict(), f"{Trainer.CHECKPOINT_DIR}/checkpoint_epoch{epoch + 1}.pth")
                logging.info(f'Checkpoint {epoch + 1} saved!')

        if self._wandb_logger is not None:
            self._wandb_logger.finish()

        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        torch.save(self._network.state_dict(), f"{Trainer.PTH_DIR}/{self._network_name}_{timestamp}.pth")

        if self._configs.save_onnx:
            C, H, W = self._configs.input_size
            sample_input = torch.randn(1, C, H, W, device=DEVICE)
            self._network.eval()
            torch.onnx.export(
                self._network, sample_input, f=f"{Trainer.ONNX_DIR}/model.onnx",  # type: ignore
                export_params=True,      # <- this embeds weights inside the ONNX file
                opset_version=18, do_constant_folding=True,
                input_names=["input"], output_names=["output"]
            )

    def _evaluate(self, dataset_loader: Iterable[tuple[torch.Tensor, torch.Tensor]]):
        N = self._configs.classes_num
        confusion_matrix = torch.zeros((N, N), dtype=torch.int64, device=DEVICE)

        self._network.eval()
        with torch.inference_mode():
            for images, masks in dataset_loader:
                images = images.to(device=DEVICE)
                masks = masks.to(device=DEVICE)

                logits: torch.Tensor = self._network(images)
                preds = logits.argmax(dim=1)

                # flatten
                preds = preds.view(-1)
                masks = masks.view(-1)

                valid = (masks >= 0) & (masks < N)
                preds = preds[valid]
                masks = masks[valid]

                indices = N * masks + preds
                cm = torch.bincount(indices, minlength=N ** 2).reshape(N, N)
                confusion_matrix += cm

        # metrics
        tp = confusion_matrix.diag()
        fp = confusion_matrix.sum(0) - tp
        fn = confusion_matrix.sum(1) - tp

        IoU = tp / (tp + fp + fn + 1e-6)
        mIoU = IoU.mean()
        pixel_acc = tp.sum() / confusion_matrix.sum()

        return {"mIoU": mIoU.item(), "pixel acc": pixel_acc.item(), "IoU": IoU.cpu()}
