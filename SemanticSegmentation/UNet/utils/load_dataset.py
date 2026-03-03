from pathlib import Path
import torch
import json
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import v2  # Use modern v2 transforms

CLASS_LABELS = {"pallet": 1, "goods": 2, "ramp": 3, "trailer": 4}


def get_label_dict(json_file: str) -> dict[tuple, int]:
    """
    Change the format of json file provided by Isaac Sim and make the key tuple, the value int (label value)
    """
    with open(json_file, 'r') as file:
        data: dict[str, dict] = json.load(file)

    label_dict = dict()

    for key, value in data.items():
        label = value["class"]
        if label in CLASS_LABELS:
            label_dict[tuple(map(int, key.strip('()').split(',')))] = CLASS_LABELS[label]

    return label_dict


def get_label_map(color_map_path: str, json_file_path: str):
    color_map = decode_image(color_map_path, ImageReadMode.UNCHANGED)
    label_dict = get_label_dict(json_file_path)
    C, H, W = color_map.shape
    output = torch.zeros(H, W, dtype=torch.int64)   # Initialize output with zeros

    # Apply mapping
    for (r, g, b, a), label in label_dict.items():
        mask = (color_map[0] == r) & (color_map[1] == g) & (color_map[2] == b) & (color_map[3] == a)
        output[mask] = label

    return output


class SemanticSegDataset(Dataset):
    def __init__(self, root_dir: str, transform=None) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir does not exist: {self.root_dir}")

        if transform is not None:
            self.transform = transform
        else:
            self.transform = v2.Identity()

        rgb_dir = self.root_dir / "rgb"
        colormap_dir = self.root_dir / "semantic_segmentation"
        json_dir = self.root_dir / "semantic_segmentation_labels"
        if not rgb_dir.exists():
            raise FileNotFoundError(f"rgb_dir does not exist!")
        if not colormap_dir.exists():
            raise FileNotFoundError(f"colormap_dir does not exist!")
        if not json_dir.exists():
            raise FileNotFoundError(f"json_dir does not exist!")

        # label = F.one_hot(torch.tensor([i]), num_classes=10).squeeze().to(dtype=torch.float32)

        IMG_FORMATS = {".png", ".jpg", ".jpeg"}
        self.image_paths: list[Path] = [path for path in rgb_dir.iterdir() if path.is_file()
                                        and path.suffix in IMG_FORMATS]
        self.image_paths.sort(key=lambda x: x.name)

        self.colormap_paths: list[Path] = [path for path in colormap_dir.iterdir() if path.is_file()
                                           and path.suffix in IMG_FORMATS]
        self.colormap_paths.sort(key=lambda x: x.name)

        self.json_label_paths: list[Path] = [
            path for path in json_dir.iterdir() if path.is_file() and path.suffix == ".json"]
        self.json_label_paths.sort(key=lambda x: x.name)

        rgbs_len, colormaps_len, jsons_len = len(self.image_paths), len(self.colormap_paths), len(self.json_label_paths)
        if not (rgbs_len == colormaps_len == jsons_len):
            raise ValueError(
                f"Length mismatch: image_paths ({rgbs_len}), colormap_paths ({colormaps_len}), jsons_len ({jsons_len})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = decode_image(self.image_paths[index], mode=ImageReadMode.RGB)
        image = self.transform(image)
        label_mask = get_label_map(self.colormap_paths[index], self.json_label_paths[index])
        # label_mask = self.transform(label_mask)
        return image, label_mask


# Modern v2 Transform Pipeline
# Note: v2 transforms expect Tensors. No need for ToTensor().
transform_train = v2.Compose([
    v2.Resize((1024, 1024), antialias=True),
    # v2.RandomRotation(degrees=15), # type: ignore
    # v2.RandomAffine(degrees=0, translate=(0.1, 0.1)), # type: ignore
    v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
    v2.ElasticTransform(alpha=30.0),  # No need for RandomApply if p=1.0, but keeping logic
    v2.ToDtype(torch.float32, scale=True),   # Scale uint8 [0, 255] to float32 [0, 1]
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Better for convergence
])

transform_val = v2.Compose([
    v2.Resize((1024, 1024), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_dataloaders(dir_path: str, batch_size: int, val_ratio: float = 0.2, num_workers: int = 8):
    """
    Consolidated helper for 2025 best practices.
    """
    # Use training transform for base dataset
    full_dataset = SemanticSegDataset(dir_path, transform=transform_train)

    if val_ratio <= 0:
        return DataLoader(full_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)

    dataset_size = len(full_dataset)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Overwrite the validation subset's transform to use transform_val
    # In v2/Subset, you access the underlying dataset via .dataset
    val_dataset.dataset.transform = transform_val  # type: ignore

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            prefetch_factor=2, persistent_workers=True)

    return train_loader, val_loader
