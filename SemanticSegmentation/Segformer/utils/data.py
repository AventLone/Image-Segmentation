from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2
from transformers import SegformerImageProcessor


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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
