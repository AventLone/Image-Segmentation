import logging, torch
from utils.common import logging_handler
from model.resnet import MLP, SmallCNN, ResNet
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, handlers=[logging_handler])
torch.backends.cudnn.deterministic = False   # Set True for full determinism (slower)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")   # Allow TF32 on Ampere+ for speed


net = ResNet()

net: ResNet = torch.compile(net)   # type: ignore
device = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)
net.to(device)
net.load_state_dict(torch.load("./data/trained_model/pth/model.pth", map_location=device))
net.eval()

# logits = net()
preprocess = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# img = Image.open("./data/img_9.png")
img = Image.open("./data/dataset/test/4/5273.png")
# img = Image.open("./data/dataset/train/2/4911.png")

img_tensor: torch.Tensor = preprocess(img).to(device)   # type: ignore
img_tensor = img_tensor.unsqueeze(0)
logging.info(f"input shape is {img_tensor.shape}")

# logits: torch.Tensor = F.softmax(net(img_tensor), dim=1)
with torch.no_grad():
    logits: torch.Tensor = net(img_tensor)
logits = torch.softmax(logits.cpu(), dim=1)

logging.info(f"Logits is {logits}")
logging.info(f"Outcome is {logits.argmax(dim=1).item()}")


