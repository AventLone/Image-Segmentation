from PIL import Image
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from matplotlib import colormaps
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from torch import nn

# def get_img():
#     import requests
#     url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#     image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
#     return image

# def make_transform(resize_size: int | list[int] = 768):
#     to_tensor = v2.ToImage()
#     resize = v2.Resize((resize_size, resize_size), antialias=True)
#     to_float = v2.ToDtype(torch.float32, scale=True)
#     normalize = v2.Normalize(
#         mean=(0.485, 0.456, 0.406),
#         std=(0.229, 0.224, 0.225),
#     )
#     return v2.Compose([to_tensor, resize, to_float, normalize])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)
transform_val = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


dinov3_vitl16: nn.Module = torch.hub.load("./dinov3", 'dinov3_vitl16', source='local', 
                               weights="weights/pretrained_dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth") # type: ignore

# print(dinov3_vitl16)

img_size = 1024
img = Image.open("/home/avent/Desktop/generated_data/2026-01-29-142706/rgb/0044.png").convert("RGB")
input_tensor = decode_image("/home/avent/Desktop/generated_data/2026-01-29-142706/rgb/0044.png", mode=ImageReadMode.RGB)

with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        batch_img = transform_val(input_tensor)[None]  # [None] adds a batch dimension.
        # batch_img = batch_img
        # feature_map = dinov3_vitl16(batch_img)
        outputs = dinov3_vitl16.forward_features(batch_img)

        # This name may differ slightly depending on release
        patch_tokens = outputs["x_norm_patchtokens"]
        # shape: (B, N, D)

        B, N, D = patch_tokens.shape
        h = w = int(N ** 0.5)

        feature_map = patch_tokens.transpose(1, 2).reshape(B, D, h, w)

print(feature_map.shape)

# 2. Extract and prepare a subset for visualization
# Since 1024 channels is too many to see at once, we'll take the first 64
# We squeeze the batch dim and add a 'channel' dim for make_grid: [64, 1, 64, 64]
vis_tensor = feature_map[0, :64, :, :].unsqueeze(1)

# 3. Create a grid and normalize for better contrast
# 'normalize=True' scales values to [0, 1] range automatically
grid = make_grid(vis_tensor, nrow=8, normalize=True, padding=2)

# 4. Convert to NumPy and plot
# PyTorch tensors are [C, H, W], but Matplotlib expects [H, W, C]
grid_np = grid.permute(1, 2, 0).cpu().numpy()

plt.figure(figsize=(12, 12))
plt.imshow(grid_np)
plt.title("Feature Map Activations (First 64 Channels)")
plt.axis('off')
plt.savefig('feature_map.png', dpi=360, bbox_inches='tight')
plt.show()
