from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class MnistDadaset(Dataset):
    def __init__(self, root_dir: str, transform: transforms.Compose) -> None:
        """
        Args:
            root_dir: Root directory containing folders 0-9 with images
            transform: Optional transforms to apply to images
        """
        super().__init__()

        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir does not exist: {self.root_dir}")
        self.transform = transform
        
        # Collect all image paths and corresponding labels
        self.image_paths = []
        self.labels = []
        
        # Iterate through each class folder (0-9)
        for i in range(10):
            class_dir = self.root_dir / str(i)
            
            if not class_dir.exists():
                continue
            
            label = torch.zeros(10, dtype=torch.float32)
            label[i] = 1.0
            # Get all image files in the class folder
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    self.image_paths.append(img_file)
                    self.labels.append(label)

    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        
        # Convert to grayscale if needed (MNIST is grayscale)
        if image.mode != 'L':
            image = image.convert('L')
        
        image = self.transform(image)   # Apply transforms
        label = self.labels[idx]        # Get label
        
        return image, label
    

# Define separate transforms for training and validation/test
transform_train = transforms.Compose([
    # Resize and padding (if images vary in size)
    transforms.Resize((32, 32)),  # Slightly larger than MNIST's 28x28
    # transforms.RandomCrop(28),  # Random crop back to 28x28
    
    # Geometric transformations
    transforms.RandomRotation(degrees=15),  # Rotate ±15 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
    
    # Color/contrast transformations (even for grayscale)
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
    
    # Elastic deformations (simulates handwriting variations)
    transforms.RandomApply([transforms.ElasticTransform(alpha=30.0)], p=0.3),
    transforms.ToTensor()
])

# Simple transforms for validation/test (no augmentation)
transform_val = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


def get_train_dataset(dir_path: str, batch_size: int, shuffle=True, num_workers=4, pin_memory=True):
    dataset = MnistDadaset(dir_path, transform_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

def get_val_dataset(dir_path: str, batch_size: int, shuffle=True, num_workers=4, pin_memory=True):
    dataset = MnistDadaset(dir_path, transform_val)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

def get_datasets(dir_path: str, batch_size: int, val_ratio: float, shuffle=True, num_workers=3, pin_memory=True):
    dataset = MnistDadaset(dir_path, transform_train)
    if val_ratio != 0.0:
        dataset_size = len(dataset)
        val_size = int(val_ratio * dataset_size)
        train_size = dataset_size - val_size
        train_dataset, val_datset = random_split(dataset, [train_size, val_size])
        return (DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory),
                DataLoader(val_datset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory))
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    