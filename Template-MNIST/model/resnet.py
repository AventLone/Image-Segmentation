import torch
import torch.nn as nn
import torch.nn.functional as F

    
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return F.relu(y + x)
    

class Test(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=3, stride=2),
            nn.Conv2d(in_channels, 3, kernel_size=3, stride=2),
            nn.Conv2d(in_channels, 3, kernel_size=3, stride=2),
            nn.BatchNorm2d(3),
            nn.MaxPool2d(kernel_size=3),
            Residual(3, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)
    

# from PIL import Image

# # convert to tensor (PyTorch)
# import torchvision.transforms as torch_trans
# # from torchvision import transforms, transforms.functional as F
# from torchvision.transforms.functional import to_tensor, to_pil_image


# # tensor_convertor = torch_trans.ToTensor()
# # to_pil = torch_trans.ToPILImage()

# img = Image.open("/home/avent/Pictures/gargantua-black-3840x2160-9621.jpg")

# img_tensor = to_tensor(img)
# print(img_tensor.shape)
# img_tensor = img_tensor.unsqueeze(0)
# print(img_tensor.size())

# net = Test(in_channels=3, out_channels=3)

# result: torch.Tensor = net(img_tensor)
# result = result.squeeze(0)

# print(result.shape)

# result_img: Image.Image = to_pil_image(result)
# result_img.save("./result.png")
