from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
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
        self.transform = transform
        
        # Collect all image paths and corresponding labels
        self.image_paths = []
        self.labels = []
        
        # Iterate through each class folder (0-9)
        for class_label in range(10):
            class_dir = self.root_dir / str(class_label)
            
            if not class_dir.exists():
                continue
                
            # Get all image files in the class folder
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    self.image_paths.append(img_file)
                    self.labels.append(class_label)

    
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
    
    # Standard transformations
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Simple transforms for validation/test (no augmentation)
transform_val = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def get_train_dataset(dir_path: str, batch_size: int, shuffle=True, num_workers=4, pin_memory=True):
    dataset = MnistDadaset(dir_path, transform_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

def get_val_dataset(dir_path: str, batch_size: int, shuffle=True, num_workers=4, pin_memory=True):
    dataset = MnistDadaset(dir_path, transform_val)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)