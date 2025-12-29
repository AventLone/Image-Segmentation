# from PIL import Image
# from pathlib import Path
# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision import transforms
# import torchvision


# class MnistDadaset(Dataset):
#     def __init__(self, root_dir: str, transform: transforms.Compose) -> None:
#         """
#         Args:
#             root_dir: Root directory containing folders 0-9 with images
#             transform: Optional transforms to apply to images
#         """
#         super().__init__()

#         self.root_dir = Path(root_dir)
#         if not self.root_dir.exists():
#             raise FileNotFoundError(f"root_dir does not exist: {self.root_dir}")
#         self.transform = transform
#         self.read_mode = torchvision.io.image.ImageReadMode.UNCHANGED
        
#         # Collect all image paths and corresponding labels
#         self.image_paths = []
#         self.labels = []
        
#         # Iterate through each class folder (0-9)
#         for i in range(10):
#             class_dir = self.root_dir / str(i)
            
#             if not class_dir.exists():
#                 continue
            
#             label = torch.zeros(10, dtype=torch.float32)
#             label[i] = 1.0
#             # Get all image files in the class folder
#             for img_file in class_dir.iterdir():
#                 if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
#                     self.image_paths.append(img_file)
#                     self.labels.append(label)

    
#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         # Load image
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path)
        
#         # Convert to grayscale if needed (MNIST is grayscale)
#         if image.mode != 'L':
#             image = image.convert('L')
        
#         image = self.transform(image)   # Apply transforms
#         label = self.labels[idx]        # Get label
        
#         return image, label
    

# # Define separate transforms for training and validation/test
# transform_train = transforms.Compose([
#     # Resize and padding (if images vary in size)
#     transforms.Resize((32, 32)),  # Slightly larger than MNIST's 28x28
#     # transforms.RandomCrop(28),  # Random crop back to 28x28
    
#     # Geometric transformations
#     transforms.RandomRotation(degrees=15),  # Rotate ±15 degrees
#     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
    
#     # Color/contrast transformations (even for grayscale)
#     transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
    
#     # Elastic deformations (simulates handwriting variations)
#     transforms.RandomApply([transforms.ElasticTransform(alpha=30.0)], p=0.3),
#     transforms.ToTensor()
# ])

# # Simple transforms for validation/test (no augmentation)
# transform_val = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor()
# ])


# def get_train_dataset(dir_path: str, batch_size: int, shuffle=True, num_workers=4, pin_memory=True):
#     dataset = MnistDadaset(dir_path, transform_train)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

# def get_val_dataset(dir_path: str, batch_size: int, shuffle=True, num_workers=4, pin_memory=True):
#     dataset = MnistDadaset(dir_path, transform_val)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

# def get_dataloaders(dir_path: str, batch_size: int, val_ratio: float, shuffle=True, num_workers=3, pin_memory=True):
#     dataset = MnistDadaset(dir_path, transform_train)
#     if val_ratio != 0.0:
#         dataset_size = len(dataset)
#         val_size = int(val_ratio * dataset_size)
#         train_size = dataset_size - val_size
#         train_dataset, val_datset = random_split(dataset, [train_size, val_size])
#         return (DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory),
#                 DataLoader(val_datset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory))
#     else:
#         return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import v2  # Use modern v2 transforms

class MnistDataset(Dataset):
    def __init__(self, root_dir: str, transform=None) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir does not exist: {self.root_dir}")
        
        self.transform = transform
        # Force grayscale reading during the decode phase
        self.read_mode = ImageReadMode.GRAY
        
        self.image_paths = []
        self.labels = []
        
        for i in range(10):
            class_dir = self.root_dir / str(i)
            if not class_dir.exists():
                continue
            
            # One-hot encoding as requested in your original code
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
        if self.transform:
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

    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    # Overwrite the validation subset's transform to use transform_val
    # In v2/Subset, you access the underlying dataset via .dataset
    val_ds.dataset.transform = transform_val # type: ignore

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

    