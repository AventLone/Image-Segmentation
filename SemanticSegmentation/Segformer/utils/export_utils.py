from pathlib import Path
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation


class SegformerOnnxWrapper(nn.Module):
    def __init__(self, model: SegformerForSemanticSegmentation) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


def export_onnx_model(model: SegformerForSemanticSegmentation, output_path: Path, image_size: int, device: torch.device) -> None:
    wrapper = SegformerOnnxWrapper(model).to(device)
    wrapper.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper, dummy_input, output_path, input_names=["pixel_values"], output_names=["logits"], 
                      opset_version=21, verbose=False)
