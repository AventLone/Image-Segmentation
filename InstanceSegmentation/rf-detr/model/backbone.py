import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights, RegNet


class Backbone(nn.Module):
    def __init__(self, name="resnet50", weights=ResNet50_Weights.DEFAULT) -> None:
        super().__init__()

        self.pretrain_net = getattr(models, name)(weights=weights)

        # stem
        self.conv1 = self.pretrain_net.conv1
        self.bn1 = self.pretrain_net.bn1
        self.relu = self.pretrain_net.relu
        self.maxpool = self.pretrain_net.maxpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.pretrain_net.layer1(x)
        c3 = self.pretrain_net.layer2(c2)
        c4 = self.pretrain_net.layer3(c3)
        c5 = self.pretrain_net.layer4(c4)

        return c2, c3, c4, c5


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()

        self.lateral_convs = nn.ModuleList()   # lateral conv (1x1)
        self.output_convs = nn.ModuleList()    # output conv (3x3)

        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, features):
        c2 = features[0]
        c3 = features[1]
        c4 = features[2]
        c5 = features[3]

        # lateral
        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4)
        p3 = self.lateral_convs[1](c3)
        p2 = self.lateral_convs[0](c2)

        # top-down
        p4 = p4 + nn.functional.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p3 = p3 + nn.functional.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        p2 = p2 + nn.functional.interpolate(p3, size=p2.shape[-2:], mode="nearest")

        # output conv (smooth)
        p5 = self.output_convs[3](p5)
        p4 = self.output_convs[2](p4)
        p3 = self.output_convs[1](p3)
        p2 = self.output_convs[0](p2)

        return p2, p3, p4, p5


class BackboneFPN(nn.Module):
    def __init__(self, backbone_name="resnet50"):
        super().__init__()
        self.backbone = Backbone(backbone_name)   # ResNet50 channel: C2=256, C3=512, C4=1024, C5=2048
        self.fpn = FPN(in_channels_list=[256, 512, 1024, 2048], out_channels=256)

    def forward(self, x):
        features = self.backbone(x)
        fpn_features = self.fpn(features)
        return fpn_features


if __name__ == "__main__":
    from torchvision.io import decode_image, ImageReadMode
    from torchvision.transforms.functional import to_pil_image
    import torch, logging
    import numpy as np
    from utils.common import logging_handler
    from model import UNet
    from torchvision.transforms import v2  # Use modern v2 transforms
    torch.backends.cudnn.deterministic = False   # Set True for full determinism (slower)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")   # Allow TF32 on Ampere+ for speed

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)

    net = BackboneFPN()
    net: BackboneFPN = torch.compile(net).to(DEVICE)   # type: ignore
    reprocess = v2.Compose([
        v2.Resize((512, 512), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_path = "/home/avent/Desktop/Image-Segmentation/InstanceSegmentation/rf-detr/colormap.png"

    img_tensor: torch.Tensor = decode_image(input_path, ImageReadMode.RGB)
    img_tensor = reprocess(img_tensor).unsqueeze(0).to(DEVICE).float()   # type: ignore

    with torch.inference_mode():
        feature_map: torch.Tensor = net(img_tensor)

    # print(feature_map[0].shape)

    for i in range(4):
        print(feature_map[i].shape)

    map1 = to_pil_image(feature_map[0].squeeze(0)[0, :, :])
    map2 = to_pil_image(feature_map[0].squeeze(0)[1, :, :])

    map1.show()
    map2.show()

    # map1 = to_pil_image(feature_map[0].squeeze(0)[:2, :, :])
    # # map2 = to_pil_image(feature_map[1])
    # # map3 = to_pil_image(feature_map[2])
    # # map4 = to_pil_image(feature_map[3])

    # map1.show()
