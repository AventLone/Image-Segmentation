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

from dataclasses import dataclass

@dataclass
class TrainConfigs:
    classes_num: int = 0
    learning_rate: float = 1.0e-4
    batch_size: int = 5
    input_size: tuple[int, int, int] = (1, 512, 512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)
    save_checkpoint: bool = False
    save_onnx: bool = False

    def print_info(self):
        logging.info(
            f"Train Parameters:\n"
            f"\t Device:          {self.device.type}\n"
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
        self._network: nn.Module = torch.compile(network).to(self._configs.device)   # type: ignore # Modern PyTorch (JIT-free) compile path
        self._optimizer = optim.AdamW(self._network.parameters(), lr=self._configs.learning_rate, weight_decay=1e-6)
        self._criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss = LogSoftmax + NLLLoss. So it already contains `softmax`

        # For mixed-precision training to keep training stable and fast.
        self._grad_scaler = torch.GradScaler(device=self._configs.device.type, enabled=True)

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
            self._wandb_logger.config.update(dict(epochs=epochs,
                                                  batch_size=self._configs.batch_size,
                                                  learning_rate=self._configs.learning_rate))
        logging.info(
            f"Starting training:\n"
            f"\t Epochs:          {epochs}\n"
            f"\t Training size:   {n_train}\n"
            f"\t Validation size: {n_val}"
        )

        scheduler = optim.lr_scheduler.OneCycleLR(self._optimizer,
                                                  max_lr=self._configs.learning_rate,
                                                  steps_per_epoch=len(self._train_dataset),
                                                  epochs=epochs)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=int(epochs / 2))

        for epoch in range(epochs):
            self._network.train()
            pbar = tqdm(self._train_dataset, desc=f"Epoch {epoch + 1}/{epochs}", unit=" batch")
            for image, label_mask in pbar:
                image: torch.Tensor = image.to(self._configs.device)
                label_mask: torch.Tensor = label_mask.to(self._configs.device)

                self._optimizer.zero_grad()
                with torch.autocast(device_type=self._configs.device.type, enabled=True):   # AMP with explicit device type (new API)
                    logits = self._network(image)
                    probs = F.softmax(logits, dim=1)
                    one_hot = F.one_hot(label_mask, self._configs.classes_num).permute(0, 3, 1, 2).float()
                    loss: torch.Tensor = self._criterion(logits, label_mask) + dice_loss(probs, one_hot, multiclass=True)
                self._grad_scaler.scale(loss).backward()
                self._grad_scaler.step(self._optimizer)
                self._grad_scaler.update()
                scheduler.step()

                pbar.set_postfix(loss=loss.item())

                # Logging (cleaner)
                if self._wandb_logger is not None:
                    self._wandb_logger.log({"train loss": loss.item()})

            # -------- Validation -------- #
            # train_acc = self._evaluate(self._train_dataset)
            # val_acc = self._evaluate(self._val_dataset)
            # logging.info(f"Train accuracy:      {train_acc:.3f}\n"
            #              f"\t Validation accuracy: {val_acc:.3f}")

            # if self._wandb_logger is not None:
            #     self._wandb_logger.log({"train acc": train_acc, "val acc": val_acc})

            if self._configs.save_checkpoint:
                torch.save(self._network.state_dict(), f"{Trainer.CHECKPOINT_DIR}/checkpoint_epoch{epoch + 1}.pth")
                logging.info(f'Checkpoint {epoch + 1} saved!')

        if self._wandb_logger is not None:
            self._wandb_logger.finish()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        torch.save(self._network.state_dict(), f"{Trainer.PTH_DIR}/{self._network_name}_{timestamp}.pth")

        if self._configs.save_onnx:
            C, H, W = self._configs.input_size
            sample_input = torch.randn(1, C, H, W, device=self._configs.device)
            self._network.eval()
            torch.onnx.export(
                self._network, sample_input, f=f"{Trainer.ONNX_DIR}/model.onnx",  # type: ignore
                export_params=True,      # <- this embeds weights inside the ONNX file
                opset_version=18, do_constant_folding=True,
                input_names=["input"], output_names=["output"]
            )

    def _evaluate(self, dataset_loader):
        self._network.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for x, y in dataset_loader:
                x: torch.Tensor = x.to(self._configs.device)
                y: torch.Tensor = y.to(self._configs.device)
                with torch.autocast(device_type=self._configs.device.type, enabled=torch.cuda.is_available()):
                    logits: torch.Tensor = self._network(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y.argmax(dim=1)).sum().item()
                total += x.size(0)
        return correct / total
