from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import v2  # Use modern v2 transforms

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']




class VOCSegDataset(Dataset):
    def __init__(self, root_dir: str, transform=None) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir does not exist: {self.root_dir}")
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = v2.Identity()
        
        self.read_mode = ImageReadMode.GRAY  # Force grayscale reading during the decode phase
        
        self.image_paths = []
        self.labels = []
        
        for i in range(10):
            class_dir = self.root_dir / str(i)
            if not class_dir.exists():
                continue
            
            label = F.one_hot(torch.tensor([i]), num_classes=10).squeeze().to(dtype=torch.float32)
            
            for img_file in class_dir.glob("*"):
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    self.image_paths.append(img_file)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 1. Load image directly into a Tensor (uint8, [C, H, W])
        # decode_image is the preferred modern alternative to read_image
        image = decode_image(self.image_paths[idx], mode=self.read_mode)
        
        # 2. Apply v2 transforms (operates on Tensors)
        image = self.transform(image)
            
        label = self.labels[idx]
        return image, label

# Modern v2 Transform Pipeline
# Note: v2 transforms expect Tensors. No need for ToTensor().
transform_train = v2.Compose([
    v2.Resize((32, 32), antialias=True),
    v2.RandomRotation(degrees=15), # type: ignore
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1)), # type: ignore
    v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
    v2.ElasticTransform(alpha=30.0), # No need for RandomApply if p=1.0, but keeping logic
    v2.ToDtype(torch.float32, scale=True),   # Scale uint8 [0, 255] to float32 [0, 1]
    v2.Normalize(mean=[0.5], std=[0.5]) # Better for convergence
])

transform_val = v2.Compose([
    v2.Resize((32, 32), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5], std=[0.5])
])

def get_dataloaders(dir_path: str, batch_size: int, val_ratio: float = 0.2, num_workers: int = 4):
    """
    Consolidated helper for 2025 best practices.
    """
    # Use training transform for base dataset
    full_dataset = MnistDataset(dir_path, transform=transform_train)
    
    if val_ratio <= 0:
        return DataLoader(full_dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=num_workers, pin_memory=True)

    dataset_size = len(full_dataset)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Overwrite the validation subset's transform to use transform_val
    # In v2/Subset, you access the underlying dataset via .dataset
    val_dataset.dataset.transform = transform_val # type: ignore

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

