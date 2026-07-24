from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2
from transformers import SegformerImageProcessor

from utils.config import TrainConfig
# from utils.trainer import resolve_model_source


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# logger = logging.getLogger(__name__)


def load_class_names(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def discover_pairs(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Missing mask directory: {masks_dir}")

    mask_candidates: Dict[str, Path] = {}
    for mask_path in masks_dir.iterdir():
        if mask_path.suffix.lower() in IMG_EXTENSIONS:
            mask_candidates[mask_path.stem] = mask_path

    pairs: List[Tuple[Path, Path]] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        mask_path = mask_candidates.get(image_path.stem)
        if mask_path is None:
            raise FileNotFoundError(
                f"No matching mask found for image '{image_path.name}' in {masks_dir}"
            )
        pairs.append((image_path, mask_path))

    if not pairs:
        raise RuntimeError(f"No image/mask pairs found in {images_dir} and {masks_dir}")

    return pairs


def infer_num_labels(pairs: Sequence[Tuple[Path, Path]], ignore_index: int) -> int:
    max_label = -1
    for _, mask_path in pairs:
        mask = np.array(Image.open(mask_path))
        if mask.ndim != 2:
            raise ValueError(
                f"Mask '{mask_path}' must be single-channel with integer class ids per pixel."
            )
        valid_pixels = mask[mask != ignore_index]
        if valid_pixels.size == 0:
            continue
        max_label = max(max_label, int(valid_pixels.max()))

    if max_label < 0:
        raise ValueError("Could not infer labels from masks. Check mask values and ignore index.")

    return max_label + 1


class SegmentationDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        processor: SegformerImageProcessor,
        image_size: int,
        use_augmentation: bool,
    ) -> None:
        self.pairs = list(pairs)
        self.processor = processor
        self.spatial_transform: Optional[Any] = None
        self.color_transform: Optional[Any] = None
        if use_augmentation:
            self.spatial_transform = v2.Compose(
                [
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomResizedCrop(
                        size=(image_size, image_size),
                        scale=(0.8, 1.0),
                        ratio=(0.9, 1.1),
                        antialias=True,
                    ),
                ]
            )
            self.color_transform = v2.Compose(
                [
                    v2.RandomApply(
                        [
                            v2.ColorJitter(
                                brightness=0.2,
                                contrast=0.2,
                                saturation=0.2,
                                hue=0.05,
                            )
                        ],
                        p=0.5,
                    )
                ]
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image = decode_image(str(image_path), mode=ImageReadMode.RGB)
        mask = decode_image(str(mask_path), mode=ImageReadMode.GRAY).to(torch.int64)

        image_tv = tv_tensors.Image(image)
        mask_tv = tv_tensors.Mask(mask)

        if self.spatial_transform is not None:
            image_tv, mask_tv = self.spatial_transform(image_tv, mask_tv)
        if self.color_transform is not None:
            image_tv = self.color_transform(image_tv)

        image_np = image_tv.permute(1, 2, 0).cpu().numpy()
        mask_np = mask_tv.squeeze(0).cpu().numpy()

        encoded = self.processor(images=image_np, segmentation_maps=mask_np, return_tensors="pt")
        sample = {key: value.squeeze(0) for key, value in encoded.items()}
        return sample


@dataclass
class Batch:
    pixel_values: torch.Tensor
    labels: torch.Tensor


def collate_batch(samples: Sequence[Dict[str, torch.Tensor]]) -> Batch:
    pixel_values = torch.stack([sample["pixel_values"] for sample in samples])
    labels = torch.stack([sample["labels"] for sample in samples])
    return Batch(pixel_values=pixel_values, labels=labels)


def get_dataloaders(configs: TrainConfig) -> Tuple[DataLoader, Optional[DataLoader], Dict[str, object]]:
    if configs.dataset is None:
        raise ValueError("Set TrainConfig.dataset to your dataset folder containing 'rgb' and 'mask'.")
    if not 0 <= configs.val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1).")

    dataset_rgb = configs.dataset / "rgb"
    dataset_mask = configs.dataset / "mask"
    pairs = discover_pairs(dataset_rgb, dataset_mask)

    class_names = load_class_names(configs.classes)
    num_labels = len(class_names) if class_names else infer_num_labels(pairs, configs.ignore_index)
    if class_names and len(class_names) != num_labels:
        raise ValueError("Class file length must match the number of labels in your masks.")

    id2label = {index: name for index, name in enumerate(class_names or [f"class_{i}" for i in range(num_labels)])}
    label2id = {name: index for index, name in id2label.items()}

    # model_source = resolve_model_source(configs.pretrained_path)
    # processor = SegformerImageProcessor.from_pretrained(model_source, do_resize=True,
    #                                                     size={"height": configs.image_size, "width": configs.image_size}, do_reduce_labels=False)
    processor = SegformerImageProcessor.from_pretrained(configs.pretrained_path)

    total_size = len(pairs)
    if configs.val_ratio > 0 and total_size < 2:
        raise ValueError("Need at least 2 samples to create a train/validation split.")

    val_size = int(total_size * configs.val_ratio)
    if configs.val_ratio > 0 and val_size == 0:
        val_size = 1
    train_size = total_size - val_size
    if train_size <= 0:
        raise ValueError("Validation split leaves no samples for training. Reduce --val-ratio.")

    if val_size > 0:
        generator = torch.Generator().manual_seed(configs.seed)
        train_pairs_subset, val_pairs_subset = random_split(pairs, [train_size, val_size], generator=generator)
        train_pairs = [train_pairs_subset[i] for i in range(len(train_pairs_subset))]
        val_pairs = [val_pairs_subset[i] for i in range(len(val_pairs_subset))]
    else:
        train_pairs = list(pairs)
        val_pairs = []

    train_dataset = SegmentationDataset(train_pairs, processor, image_size=configs.image_size, use_augmentation=not configs.disable_augmentation)
    val_dataset = SegmentationDataset(val_pairs, processor, image_size=configs.image_size, use_augmentation=False) if val_pairs else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        num_workers=configs.num_workers,
        pin_memory=True,
        persistent_workers=configs.num_workers > 0,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(val_dataset,
                            batch_size=configs.batch_size,
                            shuffle=False,
                            num_workers=configs.num_workers,
                            pin_memory=True,
                            persistent_workers=configs.num_workers > 0,
                            collate_fn=collate_batch) if val_dataset else None
    

    metadata: Dict[str, object] = {"num_labels": num_labels, "id2label": id2label, "label2id": label2id, "train_size": train_size, "val_size": val_size}
    return train_loader, val_loader, metadata
