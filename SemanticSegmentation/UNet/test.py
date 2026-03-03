from model import UNet
from torchvision.transforms import v2  # Use modern v2 transforms
import torch.nn.functional as F

net = UNet(channels_num=3, classes_num=5)
net.eval()
print(type(net))
print(net.__class__.__name__)
import torch
from torchvision.io import decode_image, ImageReadMode

preprocess = v2.Compose([
    v2.Resize((1024, 1024), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

from utils.load_dataset import get_dataloaders

train_dataloader, val_dataloader = get_dataloaders("/home/avent/Desktop/generated_data/2026-02-02-131248", batch_size=10, 
                                                   val_ratio=0.1)

val_data: list[tuple[torch.Tensor, torch.Tensor]] = list(val_dataloader)
label_mask = val_data[0][1]
print(label_mask.shape)


one_hot = F.one_hot(label_mask, 5).permute(0, 3, 1, 2).float()
# one_hot = F.one_hot(label_mask, 5).float()

print(one_hot.shape)


# img_tensor: torch.Tensor = decode_image("/home/avent/Desktop/generated_data/2026-02-02-131248/rgb/0002.png",
#                                         ImageReadMode.RGB)
# img_tensor = img_tensor.unsqueeze(0)
# img_tensor = preprocess(img_tensor)

# with torch.inference_mode():
#     logits: torch.Tensor = net(img_tensor)

# print(logits.shape)
# print(logits)

# from torchvision.transforms.functional import to_pil_image
# probs = torch.softmax(logits, dim=1)
# print(probs.shape)
# print(probs)

# result = logits.argmax(dim=1)
# print(result.shape)
# result = result.to(torch.uint8)
# result *= 20
# # logits.squeeze(0)
# print(result.dtype)
# pil_img = to_pil_image(result)

# pil_img.show()