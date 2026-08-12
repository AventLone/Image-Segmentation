from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch
from utils.export_utils import export_onnx_model
from pathlib import Path

DEVICE = torch.device(type="cuda", index=0)
export_model = SegformerForSemanticSegmentation.from_pretrained("outputs/2026.07.28/best_model").to(DEVICE)
onnx_path = Path("outputs/2026.07.28/segformer.onnx")
export_onnx_model(export_model, onnx_path, image_size=512, device=DEVICE)