import torch
from torch import Tensor
from typing import Sequence, Tuple


def nested_tensor_from_tensor_list(images: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
    max_h = max(img.shape[-2] for img in images)
    max_w = max(img.shape[-1] for img in images)
    batch = images[0].new_zeros((len(images), 3, max_h, max_w))
    mask = torch.ones((len(images), max_h, max_w), dtype=torch.bool, device=images[0].device)
    for i, img in enumerate(images):
        _, h, w = img.shape
        batch[i, :, :h, :w] = img
        mask[i, :h, :w] = False
    return batch, mask