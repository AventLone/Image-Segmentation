from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    # Set this in script before running, for example:
    dataset = Path("/home/linde/Desktop/Datasets/SemanticSegmentation/2026.07.28-14_16")
    val_ratio: float = 0.1
    pretrained_path: str = "./outputs/2026.08.05/best_model"
    output_dir: Path = Path(f"outputs/{datetime.now().strftime('%Y.%m.%d')}")
    classes: Optional[Path] = None
    image_size: int = 512
    epochs: int = 100
    batch_size: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_workers: int = 12
    disable_augmentation: bool = False
    amp_dtype: str = "bf16"
    seed: int = 42
    ignore_index: int = 255
    use_wandb: bool = True
    wandb_project: str = "SegFormer"
