from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
# from PIL import Image
# import requests
# import torch
# import numpy as np

# model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
# model_path = "./segformer-b0-finetuned-ade-512-512"

# processor = SegformerImageProcessor.from_pretrained(model_path)

# model = SegformerForSemanticSegmentation.from_pretrained(model_path).cuda()

# image = Image.open("demo/demo.png")

# inputs = processor(images=image,return_tensors="pt").to("cuda")

# with torch.inference_mode():
#     outputs = model(**inputs)

# logits = outputs.logits

# print(logits.shape)


# def build_palette(num_colors=256):
#     """Create a deterministic color palette (VOC-style bit pattern)."""
#     palette = np.zeros((num_colors, 3), dtype=np.uint8)
#     for i in range(num_colors):
#         lab = i
#         for j in range(8):
#             palette[i, 0] |= (((lab >> 0) & 1) << (7 - j))
#             palette[i, 1] |= (((lab >> 1) & 1) << (7 - j))
#             palette[i, 2] |= (((lab >> 2) & 1) << (7 - j))
#             lab >>= 3
#     return palette


# # Argmax over classes to get per-pixel class IDs.
# pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)

# # Resize prediction back to original image size for visualization.
# pred_img = Image.fromarray(pred, mode="L").resize(image.size, Image.NEAREST)

# palette = build_palette(256)
# color_mask = pred_img.convert("P")
# color_mask.putpalette(palette.flatten().tolist())
# color_mask.save("seg_pred_mask.png")

# overlay = Image.blend(image.convert("RGB"), color_mask.convert("RGB"), alpha=0.55)
# overlay.save("seg_pred_overlay.png")

# print("Saved: seg_pred_mask.png")
# print("Saved: seg_pred_overlay.png")

from transformers import SegformerForSemanticSegmentation
from PIL import Image
import requests
import numpy as np
import torch, cv2
import torch.nn.functional as F
from typing import Optional

# feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-ade-512-512")
processor = SegformerImageProcessor.from_pretrained("./segformer-b4-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("./segformer-b4-finetuned-ade-512-512")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("demo/demo.png")

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
        combined = cv2.addWeighted(origin_img, 0.66, colormap, 0.34, 0)
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

# def build_palette(num_colors=256):
# 	palette = np.zeros((num_colors, 3), dtype=np.uint8)
# 	for i in range(num_colors):
# 		lab = i
# 		for j in range(8):
# 			palette[i, 0] |= (((lab >> 0) & 1) << (7 - j))
# 			palette[i, 1] |= (((lab >> 1) & 1) << (7 - j))
# 			palette[i, 2] |= (((lab >> 2) & 1) << (7 - j))
# 			lab >>= 3
# 	return palette


# pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
# pred_img = Image.fromarray(pred, mode="L").resize(image.size, Image.NEAREST)

# palette = build_palette(256)
# color_mask = pred_img.convert("P")
# color_mask.putpalette(palette.flatten().tolist())
# color_mask.save("seg_pred_mask.png")

# overlay = Image.blend(image.convert("RGB"), color_mask.convert("RGB"), alpha=0.55)
# overlay.save("seg_pred_overlay.png")

# print("Saved: seg_pred_mask.png")
# print("Saved: seg_pred_overlay.png")
