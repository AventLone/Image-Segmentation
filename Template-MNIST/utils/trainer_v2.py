import torch, wandb, logging, pathlib
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

class Trainer:
    #---------- Hyper parameters ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)
    batch_size = 1
    learning_rate: float = 1e-4
    val_ratio: float = 0.1
    save_checkpoint: bool = False
    img_scale: float = 0.5
    classes_num: int = 0
    amp: bool = False
    save_onnx: bool = True

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
        self._network: nn.Module = torch.compile(network).to(Trainer.device)    # type: ignore # Modern PyTorch (JIT-free) compile path
        self._optimizer = optim.RMSprop(self._network.parameters(), lr=Trainer.learning_rate, weight_decay=1e-8, momentum=0.9)
        self._criterion = nn.CrossEntropyLoss() # CrossEntropyLoss = LogSoftmax + NLLLoss. So it already contains `softmax`

        # self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, "max", patience=2)   # Goal: Maximize Dice score
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

        global_step = 0
        for epoch in range(epochs):
            self._network.train()
            epoch_loss = 0.0
            with tqdm(total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img") as pbar:
                for x, y in self._train_dataset:
                    x: torch.Tensor = x.to(Trainer.device)
                    y: torch.Tensor = y.to(Trainer.device)

                    # AMP with explicit device type (new API)
                    with torch.autocast(device_type=Trainer.device.type, enabled=True):
                        logits = self._network(x)
                        # loss: torch.Tensor = F.cross_entropy(y_hat, y)
                        loss: torch.Tensor = self._criterion(logits, y)
                    self._optimizer.zero_grad(set_to_none=True)
                    # self._grad_scaler.scale(loss).backward()
                    # self._grad_scaler.step(self._optimizer)
                    # self._grad_scaler.update()
                    self._grad_scaler.scale(loss).backward()
                    self._grad_scaler.unscale_(self._optimizer)
                    nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=5.0)
                    self._grad_scaler.step(self._optimizer)
                    self._grad_scaler.update()
                    # self._scheduler.step()

                    global_step += 1
                    epoch_loss += loss.item() * x.size(0)
                    # Update the process bar
                    pbar.update(1)
                    pbar.set_postfix(loss=epoch_loss / (pbar.n + 1) * Trainer.batch_size)

                   

                    # Logging (cleaner)
                    if self._wandb_logger is not None:
                        self._wandb_logger.log({"train/loss": loss.sum().item(), "step": global_step, "epoch": epoch})

                    # -------- Validation -------- #
                    # if self._val_dataset is not None and global_step % max(1, len(self._train_dataset) // 10) == 0:
                    #     val_score = self._evaluate()
                        # self._scheduler.step(val_score)
                        # self._wandb_logger.log({"lr": self._optimizer.param_groups[0]["lr"],
                        #                         "val/dice": val_score,
                        #                         # "image": wandb.Image(x[0].cpu()),
                        #                         "step": global_step,
                        #                         "epoch": epoch})

                        # logging.info(f"Validation Dice: {val_score:.4f}")

            if Trainer.save_checkpoint:
                torch.save(self._network.state_dict(), f"{Trainer.checkpoint_dir}/checkpoint_epoch{epoch + 1}.pth")
                logging.info(f'Checkpoint {epoch + 1} saved!')

        if self._wandb_logger is not None:
            self._wandb_logger.finish()

        torch.save(self._network.state_dict(), f"{Trainer.pth_dir}/model.pth")
        if Trainer.save_onnx:
            sample_input = torch.randn(1, 1, 32, 32, device=Trainer.device)
            self._network.eval()
            torch.onnx.export(self._network, sample_input, f=f"{Trainer.onnx_dir}/model.onnx", # type: ignore
                export_params=True,      # <- this embeds weights inside the ONNX file
                opset_version=18, do_constant_folding=True,
                input_names=["input"], output_names=["output"]
            )


    def _evaluate(self):
        self._network.eval()

        val_batches_num = len(self._val_dataset) if self._val_dataset is not None else 0
        dice_score = 0

        # Iterate over the validation set
        # for batch in tqdm(self._val_dataset, total=val_batches_num, desc="Validation round", unit="batch", leave=False):
        if self._val_dataset is None:
            return
        for batch in self._val_dataset:
            data: torch.Tensor = batch[0].to(device=Trainer.device)
            label: torch.Tensor = batch[1].to(device=Trainer.device)
            # image: torch.Tensor = batch["image"].to(device=Trainer.device, dtype=torch.float32)
            # true_mask: torch.Tensor = batch["mask"].to(device=Trainer.device, dtype=torch.long)
            # true_mask = F.one_hot(true_mask, Trainer.classes_num).permute(0, 3, 1, 2).float()

            with torch.no_grad():
                pred: torch.Tensor = self._network(data)   # Predict the mask
                loss: torch.Tensor = F.cross_entropy(pred, label)


                # Convert to one-hot format
                # if Trainer.classes_num == 1:
                #     mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                #     dice_score += dice_coeff(mask_pred, true_mask, reduce_batch_first=False)   # Compute the Dice score
                # else:
                #     mask_pred = F.one_hot(mask_pred.argmax(dim=1), Trainer.classes_num).permute(0, 3, 1, 2).float()
                #     # Compute the Dice score, ignoring background
                #     dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], true_mask[:, 1:, ...], reduce_batch_first=False)
        self._network.train()

        return loss.item() / val_batches_num
    
