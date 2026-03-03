import torch
import wandb
import logging
import pathlib
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm


class Trainer:
    # ---------- Hyper parameters ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)
    batch_size = 1
    learning_rate: float = 0.01
    val_ratio: float = 0.1
    save_checkpoint: bool = False
    img_scale: float = 0.5
    classes_num: int = 0
    amp: bool = False
    save_onnx = True

    checkpoint_dir = "./data/trained_model/checkpoints"
    onnx_dir = "./data/trained_model/onnx"
    pth_dir = "./data/trained_model/pth"

    @classmethod
    def set_hyper_params(cls, config: dict):
        trainer_config = config["Trainer"]
        # cls.batch_size = trainer_config["BatchSize"]
        cls.learning_rate = trainer_config["LearningRate"]
        # cls.save_checkpoint = config["Save"]
        cls.img_scale = trainer_config["Scale"]
        cls.amp = trainer_config["AMP"]
        cls.save_onnx = trainer_config["SaveONNX"]

        cls.classes_num = config["Network"]["ClassesNum"]
        cls.batch_size = config["Dataset"]["BatchSize"]

    def __init__(self, network: nn.Module, project_name: str | None = None):
        # type: ignore # Modern PyTorch (JIT-free) compile path
        self._network: nn.Module = torch.compile(network).to(Trainer.device)
        self._optimizer = optim.AdamW(self._network.parameters(), lr=Trainer.learning_rate, weight_decay=1e-8)
        self._criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss = LogSoftmax + NLLLoss. So it already contains `softmax`

        # self._scheduler = optim.lr_scheduler.OneCycleLR(self._optimizer, "max", patience=2)   # Goal: Maximize Dice score
        # for mixed-precision training to keep training stable and fast.
        self._grad_scaler = torch.GradScaler(device=Trainer.device.type, enabled=Trainer.amp)

        self._train_dataset = None
        self._val_dataset = None

        self._wandb_logger = None if project_name is None else wandb.init(project=project_name)

        pathlib.Path(Trainer.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(Trainer.onnx_dir).mkdir(parents=True, exist_ok=True)

        logging.info(
            f"Trainer Parameters:\n"
            f"\t Batch size:      {Trainer.batch_size}\n"
            f"\t Learning rate:   {Trainer.learning_rate}\n"
            f"\t Checkpoints:     {Trainer.save_checkpoint}\n"
            f"\t Device:          {Trainer.device.type}\n"
            f"\t Image scaling:   {Trainer.img_scale}\n"
            f"\t Mixed Precision: {Trainer.amp}"
        )

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
                                                  batch_size=Trainer.batch_size,
                                                  learning_rate=Trainer.learning_rate,
                                                  amp=Trainer.amp))
        logging.info(
            f"Starting training:\n"
            f"\t Epochs:          {epochs}\n"
            f"\t Training size:   {n_train}\n"
            f"\t Validation size: {n_val}"
        )

        scheduler = optim.lr_scheduler.OneCycleLR(self._optimizer,
                                                  max_lr=Trainer.learning_rate,
                                                  steps_per_epoch=len(self._train_dataset),
                                                  epochs=epochs)
        for epoch in range(epochs):
            self._network.train()
            pbar = tqdm(self._train_dataset, desc=f"Epoch {epoch + 1}/{epochs}", unit=" batch")
            for x, y in pbar:
                x: torch.Tensor = x.to(Trainer.device)
                y: torch.Tensor = y.to(Trainer.device)

                self._optimizer.zero_grad()

                # AMP with explicit device type (new API)
                with torch.autocast(device_type=Trainer.device.type, enabled=True):
                    logits = self._network(x)
                    loss: torch.Tensor = self._criterion(logits, y)
                self._grad_scaler.scale(loss).backward()
                self._grad_scaler.step(self._optimizer)
                self._grad_scaler.update()
                scheduler.step()

                pbar.set_postfix(loss=loss.item())

                # Logging (cleaner)
                if self._wandb_logger is not None:
                    self._wandb_logger.log({"train loss": loss.item()})

            # -------- Validation -------- #
            train_acc = self._evaluate(self._train_dataset)
            val_acc = self._evaluate(self._val_dataset)
            logging.info(f"Train accuracy:      {train_acc:.3f}\n"
                         f"\t Validation accuracy: {val_acc:.3f}")

            if self._wandb_logger is not None:
                self._wandb_logger.log({"train acc": train_acc, "val acc": val_acc})

            if Trainer.save_checkpoint:
                torch.save(self._network.state_dict(), f"{Trainer.checkpoint_dir}/checkpoint_epoch{epoch + 1}.pth")
                logging.info(f'Checkpoint {epoch + 1} saved!')

        if self._wandb_logger is not None:
            self._wandb_logger.finish()

        torch.save(self._network.state_dict(), f"{Trainer.pth_dir}/model.pth")
        if Trainer.save_onnx:
            sample_input = torch.randn(1, 1, 32, 32, device=Trainer.device)
            self._network.eval()
            torch.onnx.export(self._network, sample_input, f=f"{Trainer.onnx_dir}/model.onnx",  # type: ignore
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
                x: torch.Tensor = x.to(Trainer.device)
                y: torch.Tensor = y.to(Trainer.device)
                with torch.autocast(device_type=Trainer.device.type, enabled=torch.cuda.is_available()):
                    logits: torch.Tensor = self._network(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y.argmax(dim=1)).sum().item()
                total += x.size(0)
        return correct / total
