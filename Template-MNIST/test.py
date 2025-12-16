import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam

# --------------------------
#  Model
# --------------------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


# --------------------------
#  Training
# --------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        root="mnist",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = MLP().to(device)

    # PyTorch 2.9.1 compile with default backend (speed boost)
    model: nn.Module = torch.compile(model) # type: ignore

    optimizer = Adam(model.parameters(), lr=1e-3)
    scaler = torch.GradScaler(device_type="cuda")

    for epoch in range(3):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type="cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f"epoch {epoch+1}  loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
