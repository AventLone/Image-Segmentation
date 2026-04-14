import torch
import torch.nn as nn
import torch.nn.functional as F




class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Downsample(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Upsample(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)


    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

def get_dino(device):
    dino: nn.Module = torch.hub.load("facebookresearch/dinov3", "dinov3_vits16", pretrained=True) # type: ignore
    dino = dino.to(device)
    dino.eval()
    return dino


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)
    dino = get_dino(device)

    from torchvision.io import decode_image, ImageReadMode
    from torchvision.transforms import v2  # Use modern v2 transforms
    from torchvision.transforms.functional import to_pil_image

    input = decode_image("/home/avent/Desktop/generated_data/2026-01-27-123024/rgb/0011.png")
    transform_val = v2.Compose([
        # v2.Resize((32, 32), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    result = dino(transform_val(input))
    result_img = to_pil_image(result)



