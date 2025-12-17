import logging, torch
from utils.common import logging_handler
from model.resnet import MLP
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, handlers=[logging_handler])


net = MLP()

net: MLP = torch.compile(net)   # type: ignore
device = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)
net.to(device)
net.load_state_dict(torch.load("./data/trained_model/pth/model.pth", map_location=device))
net.eval()

# logits = net()
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# img = Image.open("./data/img_9.png")
# img = Image.open("./data/dataset/train/3/5125.png")
img = Image.open("./data/dataset/train/2/4911.png")

img_tensor: torch.Tensor = preprocess(img).to(device)   # type: ignore
logging.info(f"input shape is {img_tensor.shape}")

logits: torch.Tensor = F.softmax(net(img_tensor), dim=1)
logits = logits.cpu()

logging.info(f"Logits is {logits}")
logging.info(f"Outcome is {torch.argmax(logits).item()}")


