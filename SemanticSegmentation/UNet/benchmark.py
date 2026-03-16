import torch
import torchvision.models as models
from torch.utils.benchmark import Timer, Compare

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


# # --------------------------
# # 1. Prepare Model & Input
# # --------------------------
# # Load pre-trained model (replace with your custom model)
# model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# model.eval()  # MANDATORY: inference mode

# # Create dummy input (batch_size=1, 3 channels, 224x224 image)
# input_cpu = torch.randn(1, 3, 224, 224)

# # Move model/input to GPU (if available)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_gpu = model.to(device)
# input_gpu = input_cpu.to(device)


net = UNet(channels_num=3, classes_num=5)
net: UNet = torch.compile(net).to(DEVICE)   # type: ignore
net.load_state_dict(torch.load("./data/trained_model/pth/UNet_2026-03-16-125528.pth", map_location=DEVICE))
net.eval()

preprocess = v2.Compose([
    v2.Resize((512, 512), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_path = "/home/linde/Desktop/Datasets/2026-02-02-131248/rgb/0019.png"

img_tensor: torch.Tensor = decode_image(input_path, ImageReadMode.RGB)
img_tensor = preprocess(img_tensor).unsqueeze(0).to(DEVICE)   # type: ignore

# with torch.inference_mode():
#     logits: torch.Tensor = net(img_tensor)

# result = logits.argmax(dim=1)
# result = result.to(torch.uint8)

# --------------------------
# 3. Benchmark GPU Inference
# --------------------------
gpu_benchmark = Timer(
    stmt="model(input)",
    setup="with torch.no_grad(): pass",
    globals={"model": net, "input": img_tensor}
)

# Run benchmark
gpu_result = gpu_benchmark.timeit(100)
print("\n=== GPU Inference Result ===")
print(gpu_result)

# --------------------------
# 4. Extract Exact Timing Metrics
# --------------------------
print("\n=== Key Inference Metrics ===")
print(f"GPU Mean Time: {gpu_result.mean:.6f}s | Median: {gpu_result.median:.6f}s")