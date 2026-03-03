from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms.functional import to_pil_image
import torch, logging, cv2
import numpy as np
from utils.common import logging_handler
from model import UNet
from torchvision.transforms import v2  # Use modern v2 transforms
import torch.nn.functional as F
from typing import Optional

logging.basicConfig(level=logging.INFO, handlers=[logging_handler])
torch.backends.cudnn.deterministic = False   # Set True for full determinism (slower)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")   # Allow TF32 on Ampere+ for speed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)

net = UNet(channels_num=3, classes_num=5)
net: UNet = torch.compile(net).to(DEVICE)   # type: ignore
net.load_state_dict(torch.load("./data/trained_model/pth/UNet_2026-02-28-173652.pth", map_location=DEVICE))
net.eval()

preprocess = v2.Compose([
    v2.Resize((1024, 1024), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_path = "/home/avent/Desktop/generated_data/2026-02-02-131248/rgb/0002.png"

img_tensor: torch.Tensor = decode_image(input_path, ImageReadMode.RGB)
img_tensor: torch.Tensor = preprocess(img_tensor).unsqueeze(0).to(DEVICE)   # type: ignore

with torch.inference_mode():
    logits: torch.Tensor = net(img_tensor)

result = logits.argmax(dim=1)
result = result.to(torch.uint8)


def torch2cv2(tensor: torch.Tensor):
    """
    纯PyTorch接口完成核心转换，仅最后一步转numpy
    """
    # 1. PyTorch原生操作：处理设备、批量维度、维度重排
    tensor = tensor.detach().cpu()  # PyTorch接口：CPU+脱离计算图
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)  # PyTorch接口：去批量维度
    if len(tensor.shape) == 3 and tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)  # PyTorch接口：(C,H,W)→(H,W,C)
    
    img_np = tensor.clamp(0, 255).numpy().astype(np.uint8)
    
    # 4. OpenCV通道转换
    if img_np.shape[-1] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    return img_np

COLORMAP_DICT = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)}
def visualize(label_map: torch.Tensor, origin_img: Optional[cv2.Mat] = None):
    if len(label_map.shape) > 2:
        label_map = label_map.squeeze(0)
    H, W = label_map.shape
    colormap = torch.zeros((3, H, W), dtype=torch.uint8)
    for label, (r, g, b) in COLORMAP_DICT.items():
        mask = (label_map == label)
        colormap[0][mask] = r
        colormap[1][mask] = g
        colormap[2][mask] = b
    colormap = torch2cv2(colormap)

    if origin_img is not None:
        combined = cv2.addWeighted(origin_img, 0.66, colormap, 0.34, 0)
        return combined
    
    return colormap

origin_img = cv2.imread(input_path, 1)
colormap = visualize(result, origin_img)
cv2.imwrite("./colormap.png", colormap)