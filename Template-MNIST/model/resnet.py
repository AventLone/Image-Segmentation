import torch
import torch.nn as nn
import torch.nn.functional as F

    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.block(x) + self.shortcut(x))
    
class ResNet(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            ResidualBlock(1, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, 128)
        )
        self.classify_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # -> 1x1
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classify_head(x)
    

class MLP(nn.Module):
    def __init__(self, classes_num=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, classes_num)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# model (small, effective)
# -------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),                   # 28x28 -> 14x14

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),                   # 14x14 -> 7x7
            nn.AdaptiveAvgPool2d(1),           # -> 1x1
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
