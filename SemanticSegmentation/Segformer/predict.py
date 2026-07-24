from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import requests
import numpy as np
import torch, cv2
import torch.nn.functional as F
from typing import Optional

# feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-ade-512-512")
processor = SegformerImageProcessor.from_pretrained("./outputs/2026.07.24/best_model")
model = SegformerForSemanticSegmentation.from_pretrained("./outputs/2026.07.24/best_model")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("/media/avent/DATA/generated_data/train/2026.07.23-14:28/rgb/0018.png").convert("RGB")

# inputs = processor(images=image, return_tensors="pt")
inputs = processor(
    images=image,
    return_tensors="pt",
    # size={"height": 1024, "width": 1024},  # try 768, 1024, etc.
)
outputs = model(**inputs)
logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

# print(f"output's shape is {outputs.shape}")
print(f"logits's shape is {logits.shape}")

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

COLORMAP_DICT = {0: (0, 0, 0)}


def get_or_create_color(label: int):
    if label not in COLORMAP_DICT:
        COLORMAP_DICT[label] = tuple(np.random.randint(0, 256, size=3).tolist())
    return COLORMAP_DICT[label]


def visualize(label_map: torch.Tensor, origin_img: Optional[cv2.Mat] = None):
    if len(label_map.shape) > 2:
        label_map = label_map.squeeze(0)
    H, W = label_map.shape
    colormap = torch.zeros((3, H, W), dtype=torch.uint8)
    for label in torch.unique(label_map).tolist():
        r, g, b = get_or_create_color(int(label))
        mask = (label_map == label)
        colormap[0][mask] = r
        colormap[1][mask] = g
        colormap[2][mask] = b
    colormap = torch2cv2(colormap)

    if origin_img is not None:
        combined = cv2.addWeighted(origin_img, 0.3, colormap, 0.7, 0)
        return combined
    
    return colormap

origin_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
# Upsample logits before argmax for cleaner boundaries.
upsampled_logits = F.interpolate(
    logits,
    size=(origin_img.shape[0], origin_img.shape[1]),
    mode='bilinear',
    align_corners=False)
result = upsampled_logits.argmax(dim=1)
colormap = visualize(result, origin_img)
cv2.imwrite("./colormap.png", colormap)