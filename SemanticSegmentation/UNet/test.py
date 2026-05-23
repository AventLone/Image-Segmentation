import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)
print(torch.cuda.get_device_name(0))