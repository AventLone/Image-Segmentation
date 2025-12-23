# train_mnist.py
import os
import random
from pathlib import Path
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# -------------------------
# config / reproducibility
# -------------------------
seed = 42
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = False   # set True for full determinism (slower)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")  # allow TF32 on Ampere+ for speed

# -------------------------
# hyperparams
# -------------------------
batch_size = 256
epochs = 3
lr = 1e-3
weight_decay = 1e-4
num_workers = 4
data_dir = "./data"
ckpt_dir = Path("checkpoints")
ckpt_dir.mkdir(exist_ok=True)
exp_name = f"mnist_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# -------------------------
# data
# -------------------------
mean, std = (0.1307,), (0.3081,)   # standard MNIST statistics

train_tfms = transforms.Compose([
    transforms.RandomRotation(10),            # light augmentation
    transforms.ToTensor(),
    # transforms.Normalize(mean, std),
])
test_tfms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean, std),
])

full_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_tfms)
train_len = int(0.9 * len(full_train))
train_set, val_set = random_split(full_train, [train_len, len(full_train) - train_len],
                                  generator=torch.Generator().manual_seed(seed))
# use test_tfms for validation to avoid augmentation at eval time
val_set.dataset.transform = test_tfms

test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_tfms)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

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
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),                   # 14x14 -> 7x7
            nn.AdaptiveAvgPool2d(1),           # -> 1x1
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = SmallCNN().to(device)

# optional: compilation for speed (PyTorch 2.x)
try:
    model: SmallCNN = torch.compile(model)
except Exception:
    # if compile fails for some platform, fall back gracefully
    pass

# -------------------------
# optimizer, loss, scaler
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# OneCycleLR works well for short training runs
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                          steps_per_epoch=len(train_loader),
                                          epochs=epochs)

scaler = torch.GradScaler(enabled=torch.cuda.is_available())

# -------------------------
# training & validation
# -------------------------
def evaluate(loader):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total

best_val_acc = 0.0
# writer = SummaryWriter(log_dir=f"runs/{exp_name}")

for epoch in range(1, epochs + 1):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="img")
    running_loss = 0.0
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
            logits = model(x)
            loss: torch.Tensor = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * x.size(0)
        pbar.set_postfix(loss=running_loss / ((pbar.n + 1) * batch_size))

    train_loss, train_acc = evaluate(train_loader)
    val_loss, val_acc = evaluate(val_loader)
    # writer.add_scalar("Loss/train", train_loss, epoch)
    # writer.add_scalar("Loss/val", val_loss, epoch)
    # writer.add_scalar("Acc/train", train_acc, epoch)
    # writer.add_scalar("Acc/val", val_acc, epoch)

    print(f"Epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    # checkpoint best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
        }, ckpt_dir / f"best.pt")

# final test
test_loss, test_acc = evaluate(test_loader)
print("Test acc:", test_acc)
# writer.add_hparams({"lr": lr, "batch_size": batch_size}, {"hparam/test_acc": test_acc})
# writer.close()
