train_acc = 0.2342
val_acc = 0.2342
epochs = 3
n_train = 4
n_val = 5

# print(f"Train accuracy:      {train_acc:.3f}\n"
#       f"Validation accuracy: {val_acc:.3f}")
# print(        f"Starting training:\n"
#             f"\t Epochs:          {epochs}\n"
#             f"\t Training size:   {n_train}\n"
#             f"\t Validation size: {n_val}")

import torch.nn.functional as F
import torch

# a = F.one_hot(torch.tensor([0]), num_classes=10)
# a = F.one_hot(torch.tensor([2]), num_classes=10)

# print(a)


label = torch.zeros(10, dtype=torch.float32)
label[2] = 1.0
label_2 = F.one_hot(torch.tensor([2]), num_classes=10).to(dtype=torch.float32).squeeze()

print(label)
print(label_2)