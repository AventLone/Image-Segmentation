from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    # Set this in script before running, for example:
    dataset = Path("/media/avent/DATA/generated_data/train/2026.07.23-14:28")
    val_ratio: float = 0.2
    model_name: str = "./pretrained/segformer-b4-finetuned-ade-512-512"
    output_dir: Path = Path(f"outputs/{datetime.now().strftime('%Y.%m.%d')}")
    classes: Optional[Path] = None
    image_size: int = 512
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 6e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_workers: int = 4
    disable_augmentation: bool = False
    amp_dtype: str = "bf16"
    seed: int = 42
    ignore_index: int = 255
    save_every_epoch: bool = False
    use_wandb: bool = True
    wandb_project: str = "SegFormer"
